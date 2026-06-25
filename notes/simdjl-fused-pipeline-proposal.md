# Proposal (SIMD.jl): interleaving independent SIMD pipelines across the `@noinline` barrier

**Status:** draft proposal, to pursue later. Motivated by BlazingPorts.jl's BLAKE3 (`src/Blake3.jl`).
**Reproducer:** `bench/probe_blake3.jl`, `bench/blake3_*` scratch probes.

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

## Test harness
`BlazingPorts.jl/bench/probe_blake3.jl` (full pipeline vs crate) plus the scratch decomposition probes
(leaf-only 1.06×, reduce-isolated 6.5%, at-scale reduce 14%, wide-stack byte-exact + equal-perf). Any
`@simd_fuse` prototype can be A/B'd directly against these numbers.
