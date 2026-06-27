# BlazingPorts.jl

**A probe-first campaign: reimplement battle-tested fast Rust crates in pure Julia where Julia
actually loses — and measure honestly.**

The point was never to win benchmarks. It was to find where Julia genuinely trails Rust, port the
crate where a gap is real, and document where the "gap" turns out to be a measurement artifact or a
fundamental trade-off. Every result below is single-threaded (`julia -t 1`, `taskset`,
`RAYON_NUM_THREADS=1`), median-of-many, with dead-code elimination defeated on both sides.

## Rust vs Julia — the scoreboard

| Rust crate | Julia | Single-thread result | Verdict |
|---|---|---|---|
| **memchr** (`memmem`) | `StringSearch.find_substr` (ours) | **1.03–1.10×** vs the crate | ✅ **Beats** |
| **itoa** | `IntFormat.format_int!` (ours) | **1.05×** (positive) – **1.5×** (mixed-sign) | ✅ **Beats** |
| **faer** Cholesky | `Factorizations` (ours) | **beats faer every size 256–2048** (1.17–1.53×) | ✅ **Beats** |
| **faer** QR | `Factorizations` (ours) | **beats faer every size 256–2048** (1.03–1.15×) | ✅ **Beats** |
| **hashbrown** | `SwissDict <: AbstractDict` (ours) | miss **2.5× faster**, hit 0.55× | ⚖️ **Trade-off** |
| **blake3** | `Blake3` (ours) | **compute beats pure-Rust 1.60× AND the bundled hand-asm** (8.6–9.0 vs 8.4 GB/s, 0 spills); full pipeline trails 13%, entirely the chained-loop register allocation; 7.4× over Julia ecosystem | ✅ **Beats Rust & asm on compute** (pure Julia) |
| **simd-json** | JSON.jl ≥1.6 + stage-1 SIMD POC (fork) | tape 0.66×; stage-1 kernel ~28× scalar; **~4× simd-json on long-string JSON**, ~0.85× short | ⚠ **Gap / POC** (+ fixed StrictMode F32) |
| **regex** | Base `Regex` (PCRE2, C) | regex crate **13×** (alternation) / **54×** (backtracking+anchor); 1.3–1.5× simple patterns | ⚠ **Gap** — but PCRE2(C)-vs-Rust; pure-Julia engine is a massive port |
| **simdutf8** | `isvalid(::String)` | ASCII **parity** (0.92×, already SIMD); **multibyte 0.09× (11× gap)** vs simdutf8 | ⚠ **Gap** (pure Julia) — bounded SIMD-validator port; StrictMode shuffle-kernel target |
| **ryu** | `Base.Ryu.writeshortest` | 0.76–2.05× (value-dependent) | ⏭ Skip — Base ships it |
| **roaring** | `Base.BitSet` | value-dependent (membership wins) | ⏭ Skip |
| **bumpalo** | `Bumper.jl` | parity + true zero-GC | ⏭ Skip — ecosystem has it |
| **fxhash** | `Base.hash(UInt64)` | 0.92× (parity) | ⏭ Skip |
| **ahash** | `Base.hash(UInt64)` | Julia **2.83× faster** | ⏭ Julia wins |
| **matrixmultiply** | OpenBLAS / Octavian | Julia **1.3–1.6×** | ⏭ Julia wins |
| **libm** erf/gamma | SpecialFunctions.jl | Julia **2–3.5×** | ⏭ Julia wins |
| **glam / nalgebra** | StaticArrays.jl | Julia wins | ⏭ Skip |
| **rand / rand_distr** | Base `Xoshiro` | Julia **1.5–2.6×** | ⏭ Julia wins |
| **argmin** | Optim.jl | 1.07× | ⏭ Skip |

See **[Ports: Rust vs Julia](ports.md)** for the per-crate reports + plots, and
**[Factorizations](factorizations.md)** for the faer Cholesky/QR flagship.

## The five lessons

1. **Most "gaps" were measurement artifacts, not Julia deficiencies.** itoa's famous "8.7× gap" was
   dead-code elimination (the Rust shim discarded the formatted bytes). roaring's "38×" was build
   domination. ryu's gap was allocation. bumpalo's single-point win was memory-bandwidth-bound. Each
   evaporated under a fair measurement.
2. **Where the win was real, the lever was orchestration or algorithm — not the language.** memchr fell
   to a SIMD first/last-byte prefilter; itoa to a branchless sign + divide-and-conquer; faer's QR edge
   to one gemm-orchestration choice (read the big operand in place, don't pack it). Pure SIMD.jl matched
   or beat hand-written assembly every time it mattered.
3. **Performance is value- and size-distribution-dependent.** One benchmark input is not a verdict —
   ryu swings 0.76×↔2.05× by float shape; roaring flips by set density; arena allocators flip by size.
4. **Some gaps are fundamental trade-offs.** `SwissDict`'s SIMD probe wins lookup-*miss* 2.5× but loses
   lookup-*hit* — the matching index comes from a SIMD reduction, so the value load serializes. The
   mature ecosystem `DataStructures.SwissDict` has the identical profile. Not a bug; a design coin.
5. **A handful of real Base/ecosystem gaps surfaced** — a missing `prefetch` intrinsic, a serial
   `div`-chain in `Base.Ryu`'s digit trimming — each a concrete upstream contribution.

## Method

For each crate: pick the Julia baseline (Base → stdlib → ecosystem), add a C-ABI shim over the real
crate (`ccall`, no Python), benchmark single-threaded with Chairmarks (median, DCE-defeated), and gate
at **0.96×** — at or above, document-skip; below, it's a real gap worth porting. Every port is gated by
[StrictMode.jl](https://github.com/el-oso/StrictMode.jl) guarantees (`@assert_vectorized`,
`@assert_noalloc`, `@assert_typestable`) and a correctness oracle.
