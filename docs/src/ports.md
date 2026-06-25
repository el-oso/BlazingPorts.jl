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

### The three-way measurement (this is the real finding)

We built a shim that calls blake3's *own* code with a selectable backend, and benchmarked our kernel
head-to-head against it — compress-only, 16 MiB, single-thread, same hardware:

![BLAKE3 compress: Julia vs Rust vs hand-asm](assets/blake3_kernels.png)

| | GB/s | what it is |
|---|---|---|
| **Julia `SIMD.jl` → LLVM, AVX-512 16-wide** | **7.46** | our kernel — pure Julia, no assembly |
| blake3 pure-Rust (`rust_avx2` → LLVM, 8-wide) | 4.68 | the best **the Rust compiler** produces |
| blake3 hand-written assembly (`.S`, AVX-512) | 8.36 | a hand-tuned `.S` file the crate bundles |

The result that matters: **at the language level — LLVM vs LLVM — Julia wins, 1.60×.** blake3 has *no*
pure-Rust AVX-512 path (there is no `rust_avx512.rs`; AVX-512 in blake3 is **assembly only**), so without
its bundled `.S` the crate falls back to 8-wide AVX2 and loses to our 16-wide. **"The safe language Rust"
isn't beating us — a hand-written assembly file is**, and that asm out-runs what *either* language's
compiler emits by 13%.

### Why the assembly is faster — and why no compiler closes it

![Register file: why hand-asm wins](assets/blake3_registers.png)

The compress kernel fills **all 32 AVX-512 registers** (16 hash-state + 16 message words). We measured it
directly with `code_native`: 32/32 zmm used, 53 spills — *register-saturated*. The tree-reduce is the same
G-mixing; to hide it inside the compress's spare execution-port slots you'd need free registers, and there
are none. Hand-asm packs the 32 by hand and interleaves the reduce; LLVM greedily spends all 32 on the
compress's ILP and runs the reduce as a separate, exposed phase. We exhausted the alternatives to be sure:
a faithful **recursive wide-stack reduction** (the crate's `compress_subtree` structure) is byte-exact and
**identical** speed (0.925× vs 0.930×); an **8-way fused leaf** needs 64 registers, spills, and is *slower*
(0.96×). It's not the algorithm, the cache, or the language — it's LLVM's register scheduling vs hand-asm,
**the same wall Rust hits** (which is why blake3 ships a `.S` to get around it).

### We chose to stay pure Julia — no assembly

We even hand-wrote our own AVX-512 `.S` to confirm the ceiling is reachable from Julia (it is — `ccall`
into asm hits 8.36). But our from-scratch attempt, with every line verified against the spec, still had a
subtle byte-exact bug — the perfect illustration of asm's cost. So the decision: **stay pure SIMD.jl, no
assembly.** Correct-by-construction, ~87% of hand-asm, and **faster than every compiler including Rust's**.
The last 13% is buyable only with hand-scheduled assembly and the fragility that comes with it — not a
trade we make. (The asm experiment lives on the `blake3-handasm` branch.) Reproduce the whole comparison:
`bench/probe_blake3_kernels.jl` (3-way kernel proof) and `bench/probe_blake3.jl` (full pipeline).

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
