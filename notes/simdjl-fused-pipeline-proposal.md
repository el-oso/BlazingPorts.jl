# Proposal (SIMD.jl): interleaving independent SIMD pipelines across the `@noinline` barrier

**Status:** ⛔ **RESOLVED — no SIMD.jl feature would help (see "Final resolution" at the bottom).** The
actionable output became a *StrictMode* diagnostic (`register_report`, F31), not a SIMD.jl feature. Kept as
a documented ceiling. Motivated by BlazingPorts.jl's BLAKE3 (`src/Blake3.jl`).
**Reproducer:** `bench/probe_blake3_kernels.jl` (3-way), `bench/probe_blake3.jl` (full pipeline).

## The finding that motivates this

BlazingPorts ported BLAKE3's `hash_many` to pure SIMD.jl (`Vec{16,UInt32}`, AVX-512). Measured,
single-thread, pinned 4.5 GHz:

| | GB/s (16 MiB) | vs `blake3` crate |
|---|---|---|
| our compress **kernel** (pure SIMD.jl) | 7.5 | **1.04–1.06×** |
| our full pipeline | 6.7 | 0.92× |
| `blake3` crate (hand-written AVX-512 asm) | 7.1–7.3 | — |

**The SIMD compress kernel beats the crate's hand asm** — LLVM-from-SIMD.jl matches/exceeds hand asm at
the kernel level. The full-pipeline deficit is **entirely orchestration**: the BLAKE3 tree-reduce
(transpose + parent compressions) runs at ~6.5% overhead on L1-hot data but **~14% at 16 MiB**.

We exhausted the structural explanations *empirically*:
- Not cache eviction / working-set size — a fully L1-hot **recursive wide-stack reduction** (the crate's
  `compress_subtree` structure: 32-chunk groups, 16-wide deinterleave-combine) was built, verified
  **byte-exact vs the crate at every size**, and measured **identical** throughput (0.925× vs 0.930×).
- Not buffer alignment, not the reduce algorithm, not software prefetch (all tested, all null).

The residual is that **the reduce is a phase distinct from compress, fully exposed on the vector ports.**
The crate's asm interleaves the two independent computations so the reduce hides in the compress's
port/latency bubbles. We cannot reproduce that in portable Julia because the hot kernels are necessarily
`@noinline` — inlining the 7-round, 16-wide compress spills the register file — and `@noinline` is a hard
**instruction-scheduling barrier**: LLVM will not interleave the reduce of super-group *N* with the
compress of super-group *N+1*, and the hardware OoO window (~300 µops) cannot bridge a whole super-group.

So the last ~5–8% is a **scheduling/codegen** gap, not an algorithmic one — and it is the one place
portable SIMD.jl currently cannot follow hand-tuned asm.

## What SIMD.jl could offer

### A. A software-pipelining / fused-stream construct (primary ask)
A way to express *two or more independent SIMD op-streams that the backend should interleave*, without
forcing full inlining (which spills) or accepting the `@noinline` scheduling barrier. Sketch:

```julia
# Run independent SIMD kernels whose instructions LLVM is free to interleave for port utilisation,
# emitted as one schedulable region rather than separate non-inlinable calls.
@simd_fuse begin
    a = compress_kernel(next_input)     # latency/dependency-chain heavy
    b = reduce_kernel(current_cvs)      # independent; fills a's port bubbles
end
```

Implementation directions to investigate: a macro that outlines each kernel but emits the call sites into
a single LLVM region with `alwaysinline`-at-region (not at every call), or a documented `llvmcall`
wrapper pattern that concatenates two kernel bodies into one function so the scheduler sees both.

### B. Guarantee cross-lane permute lowering (secondary)
The 16×16 `Vec{16,UInt32}` transpose in the reduce is a 4-stage `shufflevector` butterfly. Confirm/guarantee
that the relevant `shufflevector` masks lower to AVX-512 `vpermt2d`/`vpermi2d` (single cross-lane permute)
rather than `vpunpck*`/`vshufi64x2` chains, or expose a `permute2(a, b, idx)` primitive that does. (Lower
priority — measurement shows the transpose is not the dominant cost; the scheduling barrier is.)

## Why this is worth doing
BLAKE3 is a clean, self-contained witness that **pure SIMD.jl already beats hand-written asm at the kernel
level** — the remaining gap is purely the inability to schedule independent SIMD pipelines together across
Julia's inlining boundary. A `@simd_fuse`-style construct would close it here and generalise to any
"compute-then-reduce" SIMD pipeline (hashing, FFT butterflies, blocked GEMM tail updates, etc.).

## Update (2026-06-25): the limit is the register file, not scheduling

We tried the concrete fusion — an **8-way leaf** (2 batches, 8 independent `_g4` chains interleaved per
half-round). Verified byte-exact. Result: **0.959× — slower, not faster.** A pure-G microbench (minimal
state) showed 8-way is 1.48× faster than 4-way, i.e. the vector *ports* have ~48% headroom. But the real
compress carries 16 message + 16 state `Vec{16,UInt32}` = **32 zmm per batch — the entire AVX-512 register
file**. Two batches = 64 zmm ⇒ heavy spills that erase the ILP gain.

**Conclusion:** the kernel is **register-bound at 4-way**, not port-bound and not scheduling-bound. A
`@simd_fuse` construct would not help BLAKE3, because fusing compress+reduce needs more live state than 32
registers hold. The crate is under the identical constraint — which is *why* our 4-way SIMD.jl kernel
already matches/beats its hand asm. So Option A above is **moot for register-saturated kernels** like this
one; it would only help fusions whose combined working set still fits in registers. Option B (cross-lane
permute) remains a minor independent nicety. The real, immovable ceiling here is the 32×512-bit register
file — hardware, not SIMD.jl. Kept as an honest negative result.

## Test harness
`BlazingPorts.jl/bench/probe_blake3.jl` (full pipeline vs crate) plus the scratch decomposition probes
(leaf-only 1.06×, reduce-isolated 6.5%, at-scale reduce 14%, wide-stack byte-exact + equal-perf). Any
`@simd_fuse` prototype can be A/B'd directly against these numbers.

## Final resolution (2026-06-25) — measured to ground truth, no SIMD.jl feature applies

The original premise (fuse compress+reduce so LLVM schedules them together) was disproven, and the *whole*
question was then settled by benchmarking our kernel directly against blake3's own backends (the
`bp_blake3_hashmany` shim calls `blake3::platform::Platform::{AVX512,AVX2}` — its real asm and its real
Rust). Compress-only, 16 MiB, single-thread:

| | GB/s | what |
|---|---|---|
| **Julia SIMD.jl → LLVM, AVX-512 16-wide** | **7.46** | ours |
| blake3 pure-Rust `rust_avx2` → LLVM, 8-wide | 4.68 | the Rust *compiler*'s best (no `rust_avx512` exists) |
| blake3 hand-asm `.S`, AVX-512 16-wide | 8.36 | a bundled assembly file |

Conclusions, all measured:
1. **SIMD.jl already wins the language comparison — 1.60× over LLVM-compiled Rust.** There is nothing to
   propose here: Julia→LLVM beats Rust→LLVM outright (Rust has no AVX-512 path *in the language*; blake3
   ships `.S`).
2. **The 13% gap to hand-asm is LLVM register-scheduling, which SIMD.jl cannot change.** The kernel is
   register-saturated (32/32 zmm, 53 spills); the asm hand-packs registers to overlap the reduce, LLVM
   won't, and a fused kernel needs 64 zmm → spills. No `@simd_fuse`/permute/anything in SIMD.jl alters
   register allocation. Rust hits the identical wall.
3. **The one genuinely useful artifact is a diagnostic, and it landed in StrictMode, not SIMD.jl:**
   `register_report(f, types)` (StrictMode F31) reads `code_native` and reports SIMD-register count +
   spills — turning "this kernel is register-saturated ⇒ you are at the portable-compiler ceiling" into an
   automatic finding instead of a manual asm grep.

So this proposal is closed as a **documented ceiling**, not a feature request: pure SIMD.jl reaches
~85–87% of hand-written assembly on register-saturated kernels and beats every compiler; the residual is
hardware (32 registers) + LLVM scheduling, unreachable from any portable IR. The minor cross-lane-permute
nicety (Option B) is the only thing that remains a plausible SIMD.jl PR, and it is not on the critical path.
