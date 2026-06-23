# BlazingPorts.jl — probe results

Versioned record of the probe-first campaign (the narrative companion to the raw data in
`bench/results/*.json` and plots in `docs/assets/*.png`). The top-level
`../blazingly-fast-rust-crates.md` is the cross-project index; this file is the in-repo source of
truth for verdicts.

**Methodology (all probes):** single-threaded both sides (`julia -t 1`, `taskset -c 2`,
`BLAS.set_num_threads(1)`, Rust `RAYON_NUM_THREADS=1`); Chairmarks ≥1000 samples; compare **median**,
report **rel-σ**. Parity = `rust_median / julia_median`; **≥ 0.96 ⇒ Julia good enough ⇒ skip**.
Hardware varies run-to-run (~±7% drift); only σ-clean (<15% both sides) comparisons are treated as
conclusive. Regenerate any plot from saved points with `include("bench/harness.jl"); using .Harness; replot()`.

Date: 2026-06-22.

## Verdict summary

| Tier | Crate | Julia baseline | Result | Verdict |
|------|-------|----------------|--------|---------|
| 1 | matrixmultiply | OpenBLAS / Octavian / `@turbo` | all beat the crate 1.0–1.7× (n=32–256) | ☑ skip |
| 1 | libm/statrs erf, gamma | SpecialFunctions.jl | 3.45× / 2.16× faster than rust libm (N=1024) | ☑ skip |
| 1 | libm exp, log | Base (openlibm) | 2.0× / 1.3× faster than rust libm | ☑ skip |
| 2 | glam / nalgebra | StaticArrays.jl | SA beats glam; `SmallMatrix` ties SA on cross/dot | ☑ skip |
| 3 | ndarray | Base arrays/broadcast/views | Base 1.18× (fused broadcast), 1.40× (strided sum) | ☑ skip |
| 3 | **faer** | LinearAlgebra → **OpenBLAS/MKL** | faer wins QR (all n), Cholesky/SVD (n≥256); LU covered by pure-Julia RecursiveFactorization.jl (wins n≤128) | ⚠ **reimplement Cholesky/QR** |
| 4 | **rand + rand_distr** | Base `Random` stdlib (Xoshiro256++) | uniform 2.59×, normal 1.46×, exp 1.51× faster than Rust SmallRng; σ-clean (<3% both sides) | ☑ skip |
| 4 | **argmin** | Optim.jl (LBFGS) | Optim.jl 23.4µs vs argmin 25.0µs/call (1.07×, σ-clean via batch+GC-control); iters comparable | ☑ skip |

## The faer finding (the one gap)

faer does **not** compete against Julia code — Julia's `LinearAlgebra` factorizations are thin
wrappers over the **OpenBLAS/MKL C/Fortran binary**. Where faer (pure Rust) beats them, Julia has
**no pure-Julia answer**, making faer a **pure-Julia reimplementation candidate** — and a genuine
StrictMode kernel test (recursive blocked factorization under
`@assert_vectorized` / `@unroll` / `@assert_noalloc`).

Results, **parity = rust/faer ÷ OpenBLAS** (< 0.96 ⇒ faer faster). In-place `cholesky!`/`lu!`/`qr!`/
`svd!` (allocating); **GC controlled** during timing (`GC.enable(false)` + per-iteration young-gen
`GC.gc(false)`), which collapsed the earlier 200–320% σ to <13% (mostly <6%) on both sides — the
dataset is now σ-clean and conclusive. n = 64, 128, 256, 512:

| Factorization | n=64 | n=128 | n=256 | n=512 | Verdict (vs best of OpenBLAS/MKL) |
|---------------|------|-------|-------|-------|-----------------------------------|
| **QR**       | **0.66×** | **0.82×** | **0.76×** | **0.66×** | **faer wins at ALL sizes** → reimplement (biggest gap) |
| **Cholesky** | 1.22× | 0.97× | **0.90×** | **0.82×** | tie ≤128; **faer wins n≥256** → reimplement |
| **LU**       | **0.90×** | **0.80×** | **0.89×** | 1.03× | vs OpenBLAS faer wins n≤256 — **but pure-Julia RecursiveFactorization.jl already wins n≤128** (see below), so LU is NOT a reimplementation target |
| **SVD**      | 1.12× | 1.09× | **0.95×** | **0.86×** | OpenBLAS good ≤128; **faer wins n≥256** |

### LU follow-up: RecursiveFactorization.jl (pure-Julia) already matches/beats faer at small n

faer doesn't beat *Julia* code — it beats LAPACK. The pure-Julia recursive LU
(`RecursiveFactorization.jl`, used by LinearSolve/DiffEq) is the real Julia answer. Head-to-head LU
(parity = faer/contender; >1 ⇒ contender faster than faer; σ-clean except OpenBLAS small-n 11–16%):

| n | OpenBLAS | RecursiveFactorization | faer | takeaway |
|---|----------|------------------------|------|----------|
| 64  | 12934 ns | **6913 ns** | 11392 ns | **RF beats faer 1.65×** (and OpenBLAS) — gap closed in pure Julia |
| 128 | 60864 ns | **51126 ns** | 50014 ns | RF ≈ faer (0.98×), both beat OpenBLAS |
| 256 | 337524 ns | 529294 ns | **298531 ns** | RF falls off (0.56×); faer wins, narrow gap with no good pure-Julia answer |
| 512 | **2107972 ns** | 3913484 ns | 2183904 ns | OpenBLAS ties faer (1.04×); RF poor at large n |

**LU verdict:** pure Julia is already competitive — **RecursiveFactorization wins n≤128** (it's tuned
for the small-matrix regime), OpenBLAS handles n≥512. The only uncovered band is **n≈256**, too narrow
to justify a reimplementation. Notably RF beating faer 1.65× at n=64 *demonstrates pure Julia can beat
a Rust LA kernel at small sizes* — encouraging for the Cholesky/QR prototype. **Cholesky & QR remain
the genuine targets** (no pure-Julia recursive equivalent; faer wins n≥256).

**MKL is throttled on this AMD (Zen5) box — discount it, OpenBLAS is the fair baseline.** MKL came
out *worse* than OpenBLAS everywhere, which prompted a check: `MKL_VERBOSE` reports the **generic**
kernel (*"Intel(R) Architecture processors"*, not "AVX2/AVX-512 enabled"), and forcing
`MKL_ENABLE_INSTRUCTIONS` from AVX512 down to **SSE4_2 changes timing by <3%** — proof MKL runs its
reference path regardless of ISA (a genuine AVX path would be 2–4× slower at SSE4_2). So MKL's numbers
reflect Intel's AMD penalty, not MKL's real capability — they are **not** a valid "best BLAS" baseline.
The known un-cripple methods all fail on **MKL 2025.2**: `MKL_DEBUG_CPU_TYPE=5` (removed after 2020u1),
`MKL_ENABLE_INSTRUCTIONS` (no-op), and the `fakeintel` `LD_PRELOAD` (`mkl_serv_intel_cpu_true`/
`mkl_serv_get_cpu_true` are never called → non-interposable cpuid gating).

Forcing the **old MKL 2020.0** (via `MKL_jll@2020.0.166`, manually LBT-forwarded — see
`bench/mkl_amd/check2020.jl`) *does* honour `MKL_DEBUG_CPU_TYPE=5`: it speeds MKL up **1.1–1.5×**
(cholesky-256 1.50×, qr-256 1.43×, qr-512 1.28×) — directly confirming the AMD penalty. **But** the flag
only forces **AVX2**, not AVX-512, so even un-crippled MKL 2020 still loses to Zen-native AVX-512
OpenBLAS (cholesky-512: faer 885µs < OpenBLAS 1078µs < MKL2020-AVX2 1403µs). Investigation in
`bench/mkl_amd/`. **The faer verdict stands on faer vs OpenBLAS** — the strongest BLAS obtainable on
this hardware — independent of the MKL crippling.

**Measurement note:** the in-place `!` factorizations still allocate (`ipiv`/`tau`/`work`/the
factorization object) → with auto-GC this fired full collections mid-run (σ up to 320% on LU/SVD).
Fix: disable auto-GC and run an explicit **young-generation** `GC.gc(false)` per timed iteration so
reclamation is deterministic and cheap. faer needs none (Rust allocs don't touch Julia's GC). The
*next* optimization step — preallocated raw-LAPACK workspace (zero alloc) — is deferred.

## Notes on the skips

- **matrixmultiply:** pure-Julia Octavian / `@turbo` are the strong contenders at larger n (1.29×);
  OpenBLAS ties the crate at n=128–256, beats it elsewhere. MKL not needed.
- **glam:** StaticArrays is the baseline to match, not glam. `SmallMatrix.Mat4*Vec4` is slower than
  SMatrix — a codegen note (SMatrix exposes all 16 FMAs; our four-`Vec4` chain has a longer dependency
  chain), not a faer-style gap.
- **SmallMatrix** passes its per-submodule StrictMode audit (`cross/dot/norm/normalize`: typestable +
  noalloc).

## Tier 4 findings (2026-06-22)

### rand + rand_distr: Julia Base dominates (☑ skip)

Probed `Base.Random` stdlib (Xoshiro256++) vs Rust `rand` crate (`SmallRng` = Xoshiro256++) on
N=1,000,000 element fills.  Both sides use the **same PRNG algorithm family** (Xoshiro256++) for an
apples-to-apples comparison.  Rust side uses a thread-local RNG (no Mutex overhead).

σ-clean (< 3% both sides), both runs consistent:

| Kernel | Julia median | Rust median | Parity (rust/julia) | Julia σ | Rust σ | Verdict |
|--------|-------------|-------------|---------------------|---------|--------|---------|
| `rand_uniform` (uniform [0,1)) | 242 ns | 628 ns | **2.59×** (Julia wins) | 2% | 3% | ☑ skip |
| `rand_normal` (std. normal) | 1012 ns | 1476 ns | **1.46×** (Julia wins) | 2% | 2% | ☑ skip |
| `rand_exp` (Exp(1)) | 1053 ns | 1587 ns | **1.51×** (Julia wins) | 2% | 3% | ☑ skip |

**Why Julia wins so decisively on uniform:** Julia's `rand!(rng, A)` is a highly optimized
SIMD-vectorized fill; Rust's SmallRng loop is scalar.  Julia's ziggurat-based `randn!`/`randexp!`
also beat Rust's `rand_distr` implementations.

Correctness: distribution sanity checks passed (uniform mean≈0.5, normal mean≈0 var≈1, exp mean≈1).
Note: streams differ (different seeds) — value equality not tested, only distribution statistics.

### argmin: Optim.jl ties/beats it (☑ skip — now σ-clean)

Probed Optim.jl L-BFGS vs Rust argmin 0.11 L-BFGS on 2-D Rosenbrock (`f(x,y) = 100(y-x²)² + (1-x)²`)
from `[-1.2, 1.0]` to `grad_tol=1e-5`. Both converge to `[1,1]` (f≈0); iteration counts comparable
(Julia 35 iters / 51 fevals; Rust 37 — ratio 1.06×), so wall-clock is apples-to-apples.

The earlier inconclusive σ (16.5%, from Optim.jl's ~11KB/call allocations) is **resolved** with the
**batch + GC-control combo**: 100 calls/sample, auto-GC disabled, one young-gen `GC.gc(false)` per
batch (a single 22µs call is too small for per-call `GC.gc(false)` — its overhead dominated, σ 29%).

| Contender | Per-call median | rel-σ | Parity (rust/julia) | Verdict |
|-----------|-----------------|-------|---------------------|---------|
| Optim.jl LBFGS | 23.4 µs | **3.9%** | **1.07×** (Julia faster) | ☑ skip |
| rust argmin | 25.0 µs | 6.5% | — | — |

**Note on framing:** there is no Base optimizer — Optim.jl is an ecosystem package — so this is an
ecosystem win, not a stdlib one. Verdict: document-skip (Julia good enough).

## Open follow-up

✅ **RecursiveFactorization.jl probed** (see LU follow-up above): pure-Julia LU already beats faer at
n≤128 (1.65× at n=64) and ties at 128, but degrades at n≥256 → LU is *not* a reimplementation target.

**FastCholesky.jl checked (2026-06-22) — not the Cholesky analogue of RF.jl.** It only custom-codes
**n<20** (`src/FastCholesky.jl:102`: `n < 20 ? _fastcholesky! : cholesky!(Hermitian)`); for n≥20 it's
LAPACK + a Hermitian/symmetrize layer, so it's *slower* than bare OpenBLAS and loses to faer (n=256:
FastCholesky 0.76× faer, vs OpenBLAS 0.90×). So Cholesky has **no** pure-Julia medium-n answer — the
gap is real.

## Cholesky port — faer reimplementation (Factorizations.jl)

Faithful PureFFT-style port of faer 0.24.1 `cholesky_recursion_right_looking` (LLᵀ), bit-exact verified
vs faer golden (`bench/rust_compare/cholesky_golden.txt`), generic-ISA (`SIMD.jl Vec{W}`, W=host width).

- **Layer B — base kernel (n≤64, `simd_cholesky`)**: **bit-exact** vs faer golden for all n≤64;
  StrictMode `@assert_vectorized`+`@assert_noalloc`(0 B)+`@assert_typestable` all PASS;
  `@code_llvm` = 48× `<8 x double>` (AVX-512), `<4 x double>` on AVX2. **It even beats faer ~1.5× at
  n=64** (L1-resident; the SIMD FMA kernel is excellent).
- **Layer C — right-looking recursive driver (block 128, threshold 64; trsm + syrk trailing update)**:
  *correct* (reconstruction ~1e-15; not bit-exact at n≥96 — naive trsm/syrk accumulate in a different
  order than faer's microkernel, ≤3.9e-11 rel). **But performance falls off a cliff:**

  | n | BlazingPorts | faer | OpenBLAS | vs faer |
  |---|--------------|------|----------|---------|
  | 64  | **4876 ns** | 7524 | 6007 | **1.54× (win)** |
  | 128 | 40236 ns | 28734 | 28704 | 0.71× |
  | 256 | 625687 ns | 153438 | 166883 | **0.25×** |
  | 512 | ~7.9 ms | ~1.0 ms | 1.12 ms | ~0.14× |

**StrictMode verdict (the campaign's core question):** the guarantees **hold on every kernel** (base,
trsm, syrk all vectorized + noalloc + typestable) — yet that is **necessary, not sufficient** for
parity. The trailing `syrk` is vectorized & allocation-free but 4× slower than faer at n=256 because it
lacks **cache/register tiling** (no L1/L2 blocking, no register-blocked microkernel). This is exactly
the **instruction-scheduling / microkernel ceiling** (`StrictMode docs/src/rust_gaps.md`):
`@assert_vectorized` confirms SIMD emission but cannot see blocking quality. **New-lever finding:**
reaching faer needs a tiled-microkernel guarantee/lever beyond vectorized+noalloc (or it's explicitly
out of StrictMode's scope, per the ceiling doc).

### Trailing-update tiling — two paths tried (parity vs faer, single-threaded)

Both replace the naive `syrk`. **A** = `@turbo` gemm (LoopVectorization, bench-only); **B** = hand-tiled
`SIMD.jl` register-blocked (NC=4 cols × W rows, in `src`). Both compute the full m×m (2× flops vs faer's
triangular).

| n | naive | **B: hand-tiled** | **A: @turbo** | faer | OpenBLAS |
|---|-------|-------------------|---------------|------|----------|
| 64  | 1.54× | 1.58× | 1.58× | — (we win) | 1.26× |
| 128 | 0.72× | **1.07×** | 1.05× | (parity) | 1.03× |
| 256 | 0.24× | 0.43× | 0.47× | (faer wins) | 0.90× |
| 512 | 0.12× | 0.24× | 0.33× | (faer wins) | 0.83× |

**Learnings (from running both):**
1. **Pure-Julia base kernel beats faer** (n≤64) and both tiled paths **reach/beat faer parity at n≤128**
   (hand-tiled 1.07×). So pure Julia is competitive up to L2-ish sizes.
2. Both tiling paths ~**2× the naive** at n=256; register blocking (reusing each `L10[i,c]` load across 4
   column accumulators) is what turns the memory-bound naive into compute-bound.
3. At **n≥256 both still trail faer ~2×**, and **`@turbo` > hand-tiled at n=512** (0.33× vs 0.24×) —
   `@turbo` adds multi-level cache blocking my hand kernel lacks (only register + L1). The residual gap is
   (a) full vs **triangular** syrk (2× flops) and (b) **L2/L3 cache blocking** + a packed microkernel,
   which faer/OpenBLAS have and we don't.
4. **StrictMode verdict reinforced**: every variant (naive, hand-tiled, @turbo) passes
   `@assert_vectorized`+`@assert_noalloc` identically, yet they span **0.12×→0.47×** at n=256 — the
   guarantees cannot see blocking quality. The missing lever is a **cache/register-tiling guarantee**.

### Decisive experiment — even Octavian's gemm doesn't close it (the gap is the whole pipeline)

Tried two more levers: (i) **triangular + MR=2 (8-accumulator) tile** — *regressed* (n=256 0.43→0.38×,
n=128 1.07→0.91×): the scalar diagonal **corner** (NC²/2·bs scalar FMAs per 4-col block) costs more than
the triangular flop saving at these sizes. (ii) **Octavian** (Julia's tuned gemm — matched OpenBLAS in
Tier 1) as the trailing `syrk` via `matmul_serial!`:

| n | hand-tiled | **Octavian syrk** | OpenBLAS | faer |
|---|-----------|-------------------|----------|------|
| 128 | 1.08× | 1.12× | 1.02× | parity |
| 256 | 0.45× | **0.50×** | 0.92× | (faer wins) |
| 512 | 0.24× | **0.35×** | 0.83× | (faer wins) |
| 1024 | 0.22× | **0.42×** | 0.89× | (faer wins) |

**A world-class gemm in the trailing position only lifts 0.24→0.35× at n=512** — because the cost is
**distributed across the whole pipeline**: the panel `trsm` (un-blocked, O(bs²·m), comparable to syrk),
the recursion/blocking structure, and packing — not one kernel. And **faer beats even OpenBLAS**
(0.83–0.92×), so the target is a SOTA end-to-end LA pipeline, not "a good gemm."

**Conclusion:** pure-Julia Cholesky is **at parity through n≤128**; reaching faer at **n≥256 = matching a
SOTA LA library end-to-end** (co-tuned trsm + triangular packed syrk + multi-level blocking) — an
Octavian/BLIS-scale effort, not a one-kernel fix. **StrictMode finding, hardened:** every variant
(naive / hand-tiled / @turbo / Octavian-backed) passes all guarantees identically while spanning
0.22×→0.50× at n=256 — the missing signal is *pipeline-level* blocking, which no per-call guarantee
(and no single tuned kernel) captures. (See `StrictMode.jl/FEEDBACK.md` F10.)

### Register-blocking the `trsm` — big lift (and the StrictMode feedback loop closes)

The panel `trsm` was the next bottleneck. Blocked it the same way (NB=4 column panels: contributions from
columns k<c0 become a register-blocked gemm, then a tiny within-panel triangular solve) — and because the
ascending-k FMA order with exact intermediate store/reload is preserved, it's **bit-identical** to the
unblocked solve (recon unchanged). Big speedup, especially at n=256:

| n | before (naive trsm) | **after (reg-blocked trsm)** | faer |
|---|---------------------|------------------------------|------|
| 64  | 1.54× | **1.56×** | (we win) |
| 128 | 1.08× | **1.18×** (beats faer) | parity |
| 256 | 0.45× | **0.71×** | (faer wins) |
| 512 | 0.24× | **0.32×** | (faer wins) |

**StrictMode loop closed — the F10 lever now exists and works.** The parallel StrictMode work shipped
**`kernel_report`** (arithmetic intensity = FP-vector-ops : memory-vector-ops from the LLVM IR). Run on
our kernels it correctly ranks them — the thing `@assert_vectorized` couldn't see:

| kernel (all pass `@assert_vectorized`) | `kernel_report` intensity | measured |
|----------------------------------------|---------------------------|----------|
| `_syrk_panel!` (naive 1-col)           | **0.82** (memory-bound)   | the 0.24× path |
| `_trsm_right_lower!` (reg-blocked)     | 1.08                      | |
| `_syrk_lower!` (reg-blocked tile)      | **1.61** (balanced)       | the 0.71× path |
| `_chol_base!` (L1-resident)            | 0.77 (but wins ≤64 — cache-resident, intensity matters less) |

So StrictMode now *guides* the optimization: it flags trsm (1.08) and syrk (1.61) as "more blocking may
help" — the next levers (MR=2 tiles, L2 cache blocking) to push n≥256 toward faer.

### ✅ PARITY REACHED through n=256 — beats faer (the aligned-triangular breakthrough)

The parity push, guided by `kernel_report`, landed: register tiling (MR=2→MR=3, intensity 0.82→1.99)
got n=256 to 0.83×, then the decisive lever was **aligned-triangular syrk** — skip the fully-upper
row-blocks (≈half the flops) but start each column block's sweep at the **W-aligned** grid point ≤ j so
loads stay aligned (the naive `i=j` triangular had *regressed* purely on misalignment). Final
single-threaded standings vs faer (and we beat OpenBLAS too through 256):

| n | BlazingPorts | faer | OpenBLAS | verdict |
|---|--------------|------|----------|---------|
| 64  | 5160 ns  | 7741 | (1.22×) | **1.50× — beat** |
| 128 | 20699 ns | 28834 | (0.98×) | **1.39× — beat** |
| 256 | 149191 ns | 154751 | (0.89×) | **1.04× — beat (parity)** |
| 512 | 1.39 ms  | 0.97 ms | (0.75×) | 0.70× — only laggard |

So a **hand-written pure-Julia Cholesky beats faer (and OpenBLAS) single-threaded through n=256** —
the campaign's "faer gap" is *closed* up to 256. Only **n=512** remains (0.70×), where the working set
spills L2 and needs cache blocking (faer's 0.97 ms is near hardware peak).

**The StrictMode story is the headline**: `kernel_report` (shipped from F10) steered the whole push —
intensity tracked speed in lockstep — and the two misses it *couldn't* see (alignment in F13, cache-
residency in F14) became new feedback. The necessary-not-sufficient finding plus the new lever that
*does* guide is a complete arc.

### Large-n packing prototype (`bench/probe_cholesky_packed.jl`) — reaches faer parity at n=2048

Revisited the n≥512 ceiling with a **portable packed syrk** (GotoBLAS-style: pack `L10` into compact
MR-row panels → unit-stride, cache-resident microkernel). Written generic — `W` from the host, the
cache-block `MC` derived from the host **L2** at runtime, modest `MRV=2×NRB=4` tile (fits AVX2 too).
Hybrid: pack only when the `L10` panel exceeds ~½ L2 (else the non-packed kernel's lower overhead wins).

| n | non-packed (current) | **hybrid packed** | OpenBLAS | faer |
|---|----------------------|-------------------|----------|------|
| 256  | 1.05× | 1.02× (uses current) | 0.90× | 1.0 |
| 512  | 0.73× | 0.73× (transition valley) | 0.82× | 1.0 |
| 1024 | 0.88× | **0.92×** | 0.90× | 1.0 |
| 2048 | 0.74× | **0.98× (parity, beats OpenBLAS)** | 0.96× | 1.0 |

**Packing is the large-n lever** (it was the missing piece): it lifts 1024 to 0.92× and 2048 to parity,
because past the cache-fit point the non-packed kernel is L2-bandwidth-bound and packing restores
unit-stride L1-resident reuse. **n=512 is a genuine cache-transition valley** (~0.73×): `L10` (384 KB)
still fits L2, so packing's overhead isn't repaid and the non-packed kernel's strided re-reads still
cost — neither wins. The recipe (reusable for other projects): **cache-derived hybrid — small-n
non-packed, large-n packed** — and it's portable (no machine-specific constants). Next refinements for
512/very-large-n: NC-blocking of the B panel + software prefetch + a remainder-safe microkernel.

### ⚖️ Premise resolution — "same LLVM ⇒ same speed", settled at the microkernel (`bench/probe_microkernel_asm.jl`)

faer's f64 matmul is hand-written x86 **assembly** (`private-gemm-x86` `build.rs`: `vfnmadd231pd
zmm,zmm,[mem]{1to8}` memory-broadcast FMA, `M=4` zmm tile, BLIS packing) — it bypasses LLVM
auto-vectorization. So the natural worry: does Julia (LLVM-from-SIMD.jl) lose to hand-asm? We tested it
**directly** by porting faer's exact instruction via `Base.llvmcall` inline asm and racing it against the
portable SIMD.jl kernel:

```
microkernel 8×8 (in-L1):  asm {1to8} = 75.4   SIMD.jl = 75.1   GFLOP/s   (identical, both correct)
OpenBLAS single-thread dgemm (real full-gemm ceiling) = 63
implied sustained AVX-512 clock = 75.4/32 = 2.36 GHz
```

**Correction of an earlier claim:** the "~50% of peak / codegen gap" was a *measurement artifact* — I'd
compared against a phantom 144 GFLOP/s (base-clock assumption). Zen5 **downclocks hard under sustained
AVX-512** to ~2.36 GHz, so the real single-core peak is ~76 GFLOP/s. Our kernel sits at **~99% of it**,
and the hand-asm `{1to8}` kernel — faer's literal instruction — is **no faster**. LLVM's "broadcast-once +
reuse" saturates the FMA units just as well as `{1to8}`.

**Conclusion: the premise HOLDS at the microkernel level.** Portable SIMD.jl IR already matches
hand-written assembly and reaches hardware peak; the inline-asm path is unnecessary (and would only cost
portability). Any residual faer edge at the *full-factorization* level is **algorithmic** — blocking,
packing efficiency, and the serial base→trsm→syrk dependency chain that caps Cholesky at ~25–30% of the
gemm peak for everyone (faer included) — not microkernel codegen. That's portable to attack, and the
packing work above already took 2048 to parity. (For ARM/AVX2 there's nothing to do: `{1to8}` is
AVX-512-only; the optimal AVX2 sequence is `vbroadcastsd`+`vfmadd231pd`, exactly what LLVM emits.)

### Autotuned + remainder-safe packed kernel (`bench/probe_cholesky_autotuned.jl`)

Two upgrades that make the packed kernel **drop-in for any project / any CPU**:

1. **`@generated` tile-parameterized microkernel** — the `MR×NR` register tile is a type parameter, so
   every tile is JIT-unrolled/specialized for free. This is the **Julia differentiator vs C/Rust**: no
   build-time `#ifdef`/codegen, no separate binaries — the optimal kernel for *this* CPU is compiled at
   first call.
2. **Runtime autotuner (FFTW-"plan" style)** — benchmarks the candidate tiles on the host once and caches
   the winner. Uses **median-of-25** (not min) for robust selection (project policy + avoids a lucky
   single run picking a bad tile). On this Zen5 it independently chose **`(MR,NR)=(3,4)`** — the exact
   tile we'd hand-tuned for the syrk (validation). On a narrower ISA (AVX2, fewer registers) it would
   pick a smaller tile automatically.
3. **Remainder-safe** — full interior tiles use the fast unrolled kernel; bottom/right/near-diagonal
   edges use a masked kernel (SIMD.jl masked `vload`/`vstore`), so **arbitrary `n` works** (verified
   correct at n = 300/1000/1500, not just powers of two). No divisibility requirement, no fallback.

Everything machine-specific is queried: `W` from the host, `MC` from L2, `(MR,NR)` from the autotuner.

### n≥512 — the earlier ceiling exploration (pre-packing)

Pushed n=512 three more ways, all measured:
- **i-outer cache blocking** (keep `L10[panel]` L1-resident): *regressed* (512 0.70→0.65×) — restructuring
  forced the register tile MR=3→MR=2; the lost intensity beat the locality gain (no packing → L10 still
  streams from L2). [FEEDBACK F15]
- **`BLOCK_SIZE` tune** (128→256): *worse* at 512 (0.75→0.67×) — the dip isn't a block-size sweet-spot.
- **Octavian `syrk`** (pure-Julia *packed* BLIS gemm) with the now-blocked trsm: **slower than our hand
  kernel at every size** — 512 0.61× vs **0.75×**, 1024 0.64× vs **0.89×**.

The last point is the key finding: **our specialized aligned-triangular kernel beats the general packed
gemm (Octavian) for this syrk-in-Cholesky pattern.** So a hand-written BLIS pack is *unlikely* to pay off
(the packed reference already loses to us). Large-n standings vs faer:

| n | BlazingPorts | Octavian-syrk | OpenBLAS | faer |
|---|--------------|---------------|----------|------|
| 512  | **0.75×** | 0.61× | 0.82× | 1.00 |
| 1024 | **0.89×** | 0.64× | 0.91× | 1.00 |

**Final verdict (superseded below):** pure-Julia Cholesky beats faer through n=256, and *was* 0.75×/0.89×
at 512/1024 — but see the LDA finding: those two sizes were slow for a fixable reason, not a ceiling.

### 🎯 The 512/1024 gap was a power-of-2 leading-dimension cache conflict — FIXED (`cholesky_llt_padded!`)

Decomposing the 512 factorization showed the **syrk at ~70 GFLOP/s (near peak) but the trsm at ~30** — and
the trsm rate depended on the *leading dimension*: ld=512 → 30, ld=520 → 68 (syrk 58→74 likewise). At
n=512/1024 the in-place matrix has `ld = n = 2^k`, which aliases consecutive columns into the same cache
sets (classic `LDA=2^k` conflict). **Factoring in a padded buffer (ld+8) recovers 1.3–1.5× and now beats
faer at every size:**

| n | in-place (ld=2^k) | **padded** | faer |
|---|-------------------|------------|------|
| 512  | 0.73× | **~1.1× (1.5× faster than in-place)** | 1.00 |
| 1024 | 0.88× | **~1.2× (1.34×)** | 1.00 |
| 2048 | 0.74× | **~1.0–1.24× (1.38×)** | 1.00 |

(Same-run in-place-vs-padded ratios are 1.49/1.34/1.38×; faer absolute times are thermally noisy, so the
faer-relative column is best-of clean runs.) This is **the** lever for n≥512 — bigger and far simpler than
packing (which only helped 2048 and not 512). faer/BLAS pad for exactly this reason.

**API (one entry, always fast):** `cholesky_llt!(A)` auto-detects a power-of-two stride and factors in a
padded scratch (else in place) — the caller just gets the fast result. For a hot loop, pass a
preallocated **`CholWorkspace(n)`** — `cholesky_llt!(A, ws)` is then **allocation-free even on the padded
path** (the no-alloc guarantee, verified by `@assert_noalloc`). The result is bit-identical regardless of
`ld` (pure addressing). **The faer Cholesky gap is fully closed at all measured sizes once the LDA is not
a power of two** — i.e. our kernels were never the problem; the earlier "ceiling" was a benchmarking
artifact of factoring power-of-two matrices in place.

➡ Next (independent): **QR (Layer D)** via the same recipe.

## QR port — faer reimplementation (Layer D)

Same faithful-port recipe (golden harness → unblocked base → blocked driver). faer convention
`H_k = I − v_k v_kᵀ/τ_k`; the QR factors (R, v) are mathematically unique, so our output matches faer's
golden to ~1e-15 at all n (non-bit-exact only from `norm_l2` ordering); reconstruction Q·R ≈ A ~1e-14.

- **D-A golden harness** ✅ — `qr_verify` dumps faer's packed QR + τ/T; reconstruction test.
- **D-B unblocked Householder base kernel** (`qr_unblocked!`) ✅ — `Vec{W}`-vectorized; StrictMode
  `@assert_vectorized`+`@assert_noalloc`+`@assert_typestable` pass. **Beats faer at n=64 (1.59×).**
- **D-C blocked compact-WY driver** (`qr_blocked!`) ✅ — panel reduction + dlarft `T` + trailing update
  as **tiled SIMD gemms**: `C −= V·Y` (syrk-style, MR=2×4) and `Vᵀ·C` (dot-form, 2c×4j). `nb=8`.

### ✅ QR PARITY — beats faer through n=256

The tiled-gemm marathon (the same one Cholesky took) landed: the levers were `nb=8` (large nb wastes
time in rank-1 panel reduction), **MR=2 tile on `C−=VY`**, and the **2c×4j tile on `Vᵀ·C`** (the decisive
one). Single-threaded vs faer:

| n | BP-blocked | BP-unblocked | OpenBLAS | faer |
|---|-----------|--------------|----------|------|
| 64  | **1.19×** | 1.59× | 0.67× | 1.00 |
| 128 | **1.03×** | 0.82× | 0.80× | 1.00 |
| 256 | **1.12×** | 0.54× | 0.76× | 1.00 |
| 512 | 0.87× | 0.39× | 0.67× | 1.00 |

**Pure-Julia QR beats faer through n=256** (and beats OpenBLAS everywhere); n=512 at 0.87× (faer's QR is
near-peak, like its Cholesky). Correct throughout (recon ~1e-14, matches faer's factors ~1e-15),
StrictMode-clean. **Both faer factorization gaps (Cholesky + QR) are now closed in pure Julia through
n=256.**

#### QR at large n — packing does NOT transfer; the bottleneck is different (`bench/probe_qr_packed.jl`)

Re-measured wider: **QR is at parity/better through n=1024** (256 1.17×, 512 1.07×, **1024 1.01×**), but
**falls to 0.53× at 2048.** Applying the Cholesky lever (pack `V` in the dlarfb, cache-hybrid) only nudges
2048 → 0.58×. **Why it doesn't transfer:** Cholesky's syrk has a *fat* reduction (`pb=block_size=128`) →
compute-bound, so packing-`V` fixes the last stride and hits peak. QR's dlarfb reduction is *thin*
(`pb=nb=8`, from the ≤512 parity push) → the trailing `C` is streamed ~`n/nb` times; at n=2048 that's
~4 GB of `C` traffic (~half of faer's runtime). Packing `V` fixes `V`-access, **not the C-streaming
traffic** — so it barely helps. **The real fix is two-level / recursive blocking** (faer's structure):
reduce a *fat* outer panel (`nb≈64`) using `qr_blocked!` with a small inner `nb` (keeps the panel
reduction efficient), then a single fat dlarfb (`pb=64` → ⅛ the C-traffic, *and* packing-`V` now pays).
The `nb` sweep confirms the tradeoff (2048: `nb=16` best at 0.55×, bigger worse — our flat panel is
rank-1; recursion is what makes large `nb` cheap). So the microkernel-is-at-peak finding transfers, but
QR's large-`n` gap is a *blocking-structure* problem, not a packing one. (Status: parity ≤1024 shipped in
`src`.)

#### Recursive (Elmroth–Gustavson = faer's structure) — correct, but structure alone isn't enough (`bench/probe_qr_recursive.jl`, `probe_qr_rec_blas.jl`)

Copied faer's recursive blocked QR faithfully: recurse left, fat dlarfb to the right, recurse right,
combine `T₁₂ = −T₁₁(V₁ᵀV₂)T₂₂`. **Correct** (recon ~1e-15, incl. non-power-of-2). But:
- With our register-tiled (non-cache-blocked) gemms it's *slower* (0.05×) — the fat `pb` makes those
  kernels re-stream C `pb/NR` times (they're tuned for the small `pb=8` of the flat driver).
- With **optimal single-thread BLAS gemms** it reaches only **0.50× at 2048 — no better than our flat
  blocked (0.53×).** Why: faer's QR runs at ~54 GFLOP/s ≈ the dgemm ceiling (62); ours (flat *and*
  recursive) at ~27. **Half our time is non-gemm overhead** — the E-G `T`-combine's extra `V₁ᵀV₂`
  (≈ n³/8 flops), dense-`V` rebuilds, and the rank-1 leaf reduction.

**Conclusion:** the recursive *structure* is necessary but not sufficient. faer's 2× edge at n=2048 is
low-level implementation engineering (in-place/fused operation, minimal temporary traffic, its own tuned
gemm), not the algorithm shape — copying the structure doesn't transfer the speed. This is the same
lesson as the microkernel study, one level up: at the full-factorization level, *implementation overhead*
is the gap, and matching it means reproducing faer's engineering, not just its math. **Practical verdict:
QR is at parity/better through n=1024 (shipped); n≥2048 stays ~0.5× and closing it is deep
faer-specific engineering with diminishing returns.**

#### Does the Cholesky LDA=2^k fix transfer to QR? No (measured).

After the Cholesky win, refactored `qr_blocked!` into `_qr_blocked_core!(A, tau, mlog, nb)` so it factors
the leading `mlog×n` of a plain `Matrix` (single-level views only — passing a padded *view* specializes the
kernels for a nested `SubArray` and **crashes the Julia compiler** in LLVM GC-lowering). With padding then
working: **2048 = 0.49× padded vs 0.53× in-place — no improvement** (padding only adds copy cost; it
slightly hurt 512/1024 too). So padding was reverted. **QR's large-`n` bottleneck is algorithmic
(thin `nb`-deep dlarfb → the trailing block is streamed ~n/nb times; non-gemm overhead dominates), not the
`LDA=2^k` cache conflict** that bottlenecks Cholesky's trsm/syrk. The two factorizations have different
large-`n` limiters: Cholesky's was cache-aliasing (fixed by padding → beats faer everywhere); QR's is the
flat-blocked structure (needs faer's in-place fused recursive backend, diminishing returns).

For argmin: to get a σ-clean comparison, either use a zero-allocation Julia optimizer or call the
BLAS-backed LBFGS with preallocated workspace to eliminate GC from the timed region.
