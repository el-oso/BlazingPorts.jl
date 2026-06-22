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

➡ **Next: prototype a pure-Julia recursive blocked Cholesky / QR** in a `Factorizations` submodule,
gated by `@assert_vectorized`/`@unroll`/`@assert_noalloc` — these have no pure-Julia recursive
equivalent and faer wins them at n≥256. RF beating faer at small-n LU shows pure Julia *can* win at
these sizes; the open question is whether StrictMode's levers reach faer for Cholesky/QR or re-expose
the instruction-scheduling ceiling (`StrictMode docs/src/rust_gaps.md`).

For argmin: to get a σ-clean comparison, either use a zero-allocation Julia optimizer or call the
BLAS-backed LBFGS with preallocated workspace to eliminate GC from the timed region.
