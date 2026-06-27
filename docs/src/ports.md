# Ports: Rust vs Julia

Per-crate reports. Plots are violin distributions of per-evaluation time (lower = better),
single-thread, median annotated.

## memchr → `StringSearch` — ✅ beats the crate

Base's multi-byte `findfirst(needle, haystack)` is a scalar scan (0.12× `memchr::memmem`). We added the
SIMD pass Base never wrote: a **first+last-byte prefilter** over `Vec{64,UInt8}` (memmem's own trick) +
a bounded scalar tail.

- **Result:** parity-to-slight-beat — **1.03–1.10×** the `memmem` crate (both near the ~67 GB/s
  single-core bandwidth ceiling), and ~8–12× over Base `findfirst`.
- Byte search was already at parity (Base is memchr-backed), so only `m ≥ 2` takes the SIMD path.
- `@assert_vectorized` / `@assert_noalloc` / `@assert_typestable` green.

![memchr substring: ours vs memmem](assets/stringsearch.png)

## itoa → `IntFormat` — ✅ beats the crate

The famous "8.7× gap" was a **dead-code-elimination artifact** — the Rust shim used only
`buf.format(x).len()`, so the optimizer elided itoa's digit-writes. Measured fairly (`black_box` /
`donotdelete`), itoa formats at ~7.5 ns/int, not 1.83.

- **Result:** **1.05×** on positive-only (pure format), **1.5×** on full mixed-sign `Int64`, ~2× on
  small numbers. ~3× faster than Base `string()` (which heap-allocates).
- Levers: divide-and-conquer digit extraction, jeaiii division-free 8-digit (verified), 16-bit packed
  LUT stores, and — the decisive one — **branchless sign** (`if x<0` mispredicts ~4× at 50/50 signs).

![itoa: ours vs the crate](assets/itoa.png)

## blake3 → `Blake3` — ✅ beats every compiler (incl. Rust); loses only to bundled hand-asm

This one taught us the most, so it gets the long version. `Blake3Hash.jl` (the pure-Julia ecosystem
package) is scalar — 0.085× the crate. We ported a `Vec{16,UInt32}` AVX-512 `hash_many`, byte-exact on the
official BLAKE3 test vectors **and** against the crate across the whole SIMD path (super-group boundaries,
remainders, partial chunks).

BLAKE3 has two phases: a 16-wide SIMD **compress** (≈94% of the work) and a small tree **reduce**. The
compress is where throughput is decided.

![BLAKE3 two phases](assets/blake3_pipeline.png)

### The measurement (this is the real finding)

We built a shim that calls blake3's *own* code with a selectable backend, and benchmarked our kernel
head-to-head against it — compress-only, 16 MiB, single-thread, same hardware. The fourth bar is **our
own `blake3_asm` switch path** (a `ccall` into the vendored `.S` plus the output transpose-back) — it
lands on the crate's hand-asm bar, proving the switch reaches the asm ceiling *from Julia*:

![BLAKE3 compress: Rust vs Julia vs hand-asm vs our blake3_asm switch](assets/blake3_kernels.png)

| | GB/s | what it is |
|---|---|---|
| **Julia `SIMD.jl` → LLVM, AVX-512 16-wide** | **7.50** | our kernel — pure Julia, no assembly |
| blake3 pure-Rust (`rust_avx2` → LLVM, 8-wide) | 4.67 | the best **the Rust compiler** produces |
| blake3 hand-written assembly (`.S`, AVX-512) | 8.34 | a hand-tuned `.S` file the crate bundles |
| **BlazingPorts asm-leaf (`blake3_asm` switch)** | **8.26** | the *same* `.S`, reached via our `ccall` = 0.99× the crate |

The result that matters: **at the language level — LLVM vs LLVM — Julia wins, 1.60×.** blake3 has *no*
pure-Rust AVX-512 path (there is no `rust_avx512.rs`; AVX-512 in blake3 is **assembly only**), so without
its bundled `.S` the crate falls back to 8-wide AVX2 and loses to our 16-wide. **"The safe language Rust"
isn't beating us — a hand-written assembly file is**, and that asm out-runs what *either* language's
compiler emits by 13%.

### Why the assembly is faster — and where the gap *actually* lives

![Register file: why hand-asm wins](assets/blake3_registers.png)

We read blake3's own `.S` next to our `code_native`, then decomposed our kernel layer by layer — and the
real picture is **more favorable than the 13% suggests.** It took correcting two wrong guesses to find it
(first we blamed the tree-reduce; then the compress kernel; both wrong).

The asm holds the message **in registers** (348 of its 350 round adds are register-to-register — it does
*not* reload), exactly like ours; both use all 32 zmm. So the only difference is **stack spills**: ~0 in the
asm, ~52–57 in ours. But *where* those spills come from is the finding:

- **The compress G-mix alone — message in registers — is `0 spills` and runs at 8.6–9.0 GB/s, *faster than
  the asm's full pipeline (8.4)`.** Our *compute* already beats hand-written assembly. (Confirmed in plain
  SIMD.jl **and** in hand-written flat LLVM IR — both spill-free.)
- **Every spill is in the *chained leaf*** — the 16-block loop that transposes the input into lanes and
  carries the 8-word chaining value across iterations. Holding those 8 CV registers across a loop body that
  already needs all 32 overflows the file. The asm hand-allocates the entire 16-block sequence into 32
  registers with zero spills; LLVM won't — and **neither Julia nor hand-written portable LLVM IR moves it.**

We pushed this to exhaustion. We rebuilt the kernel as flat LLVM IR and measured six structural variants of
the chained leaf:

| variant | spills | GB/s |
|---|---|---|
| single block, compute only | **0** | 8.6–9.0 |
| 16 blocks fully unrolled | 1612 | — |
| block loop (chaining value in phi) | 52 | 7.5 |
| + memory-clobber barrier | 56 | 7.5 |
| + `llvm.loop` unroll/interleave-disable | 52 | 7.5 |
| + constant rematerialization | 54 | 7.5 |

All six chained variants land at the same ~7.5 GB/s. So the 13% is **not** the algorithm, the cache, the
language, the tree-reduce, or the compress kernel — it is one thing: the **global register allocation of the
chained block loop**, a hand-tuning the asm performs and no compiler reproduces from any frontend. The honest
one-liner: *our BLAKE3 computes faster than hand-tuned assembly; the asm's only remaining edge is a
register-allocation trick on the input plumbing.* (StrictMode's `register_report` diagnostic — F31 — came
out of this hunt.)

### Pure Julia by default — opt into the asm with one preference

The portable, correct-by-construction default is **pure SIMD.jl, no assembly**: ~87% of hand-asm and
**faster than every compiler including Rust's**. That last 13% is buyable only with hand-scheduled
assembly. We make it **opt-in** rather than refusing it: `Blake3` ships blake3's own CC0 AVX-512 kernel
(`deps/blake3/blake3_avx512_x86-64_unix.S`, the proven 8.36 GB/s `.S`, not our from-scratch attempt) and
routes the leaf compress through it **behind the `Preferences.jl` switch `blake3_asm`**.

- **Default = on where available.** At load, when the host is x86-64 Linux with AVX-512F and a `cc` is on
  `PATH`, the vendored `.S` is assembled and `dlopen`ed; otherwise the pure path is used. The switch only
  swaps the **leaf** — the tree reduce and root stay pure Julia.
- **End-to-end effect** (full `blake3()`, 16 MiB, single-thread): the asm leaf lifts throughput **1.16×**
  (6.45 → 7.50 GB/s), essentially closing the 13% kernel gap at the pipeline level.

![BLAKE3 full hash(): the blake3_asm switch](assets/blake3_asm_switch.png)

To force the portable pure path (e.g. for reproducible/portable builds), set the preference and restart:

```julia
using Preferences
set_preferences!(Base.UUID("6a76645a-1c79-4c35-96ac-450b50bde595"), "blake3_asm" => false)
```

The pure kernel remains the fallback whenever the asm is unavailable, so correctness never depends on the
toolchain. (Our earlier *hand-written* asm experiment — which had a subtle byte-exact bug, illustrating
asm's cost — lives on the `blake3-handasm` branch; the shipped switch uses blake3's vetted `.S` instead.)
Reproduce the whole comparison: `bench/probe_blake3_kernels.jl` (3-way kernel proof **+** the asm-switch
full-pipeline probe) and `bench/probe_blake3.jl` (full pipeline); `bench/plot_blake3_kernels.jl` regenerates
both plots from the saved JSON.

## hashbrown → `SwissDict` — ⚖️ a fundamental trade-off

Reading Base's `dict.jl` reframed this: **Base `Dict` is already a SwissTable** (control bytes = h2,
SoA keys/vals) — only the *probe width* differs (scalar 1-slot vs SIMD 16). We ported a full
`SwissDict{K,V} <: AbstractDict` with a `Vec{16,UInt8}` group probe (TypeContracts-verified interface).

- **Result:** lookup-**miss 2.5× faster** than Base `Dict`; lookup-**hit 1.8× slower**. The SIMD probe
  derives the matching index *from* a reduction, so the value load serializes (no memory-level
  parallelism) — Base's scalar probe knows the address early. The mature `DataStructures.SwissDict`
  (group-aligned, prefetch-tuned) shows the **identical** profile, so it's inherent, not our bug.
- **Verdict:** a *miss-optimized* dict (membership / dedup / set-ops), not a clean win.

## ryu → skip (Base already ships Ryu)

Same DCE bug as itoa, fixed. Fairly, `Base.Ryu.writeshortest` (zero-alloc, Julia ships it) is **2.05×
faster** than the crate on integer-valued floats but **0.76–0.81×** on full-mantissa — value-dependent,
~parity overall. The residual is Base.Ryu's codegen, not the algorithm. Low ROI to port.

![ryu: Base.Ryu vs the crate](assets/ryu.png)

## roaring · bumpalo · fxhash · ahash → skip

- **roaring** (compressed bitsets) vs Base `BitSet`: the "38× dense" was build domination. Op-only,
  `BitSet` wins membership at every density (45× even sparse) and dense set-algebra; roaring wins only
  sparse-large union/intersect. Value-dependent — no port.
- **bumpalo** (arena allocator): `Bumper.jl` is the Julia analogue — parity at scale (both
  bandwidth-bound) and **zero GC allocations/call even at 1.1 GB**. No port.
- **fxhash**: Base `hash(::UInt64)` is at parity (0.92×). **ahash**: Base `hash` is **2.83× faster**
  (per-call hasher build dominates the AES advantage). Skip.
