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
| 3 | **faer** | LinearAlgebra → **OpenBLAS/MKL** | Cholesky & QR: faer wins n≥256 | ⚠ **reimplement** |

## The faer finding (the one gap)

faer does **not** compete against Julia code — Julia's `LinearAlgebra` factorizations are thin
wrappers over the **OpenBLAS/MKL C/Fortran binary**. Where faer (pure Rust) beats them, Julia has
**no pure-Julia answer**, making faer a **pure-Julia reimplementation candidate** — and a genuine
StrictMode kernel test (recursive blocked factorization under
`@assert_vectorized` / `@unroll` / `@assert_noalloc`).

σ-aware results (in-place `cholesky!`/`lu!`/`qr!`/`svd!`; n = 64, 128, 256, 512), parity = rust/julia:

| Factorization | n=64 | n=128 | n=256 | n=512 | Trusted verdict |
|---------------|------|-------|-------|-------|-----------------|
| **Cholesky** (σ-clean) | 1.02× | 1.00× | **0.90×** | **0.83×** | faer wins n≥256 (MKL loses harder, 0.64–0.85×) → **reimplement** |
| **QR** | 0.71×* | 0.82×* | **0.76×** | **0.68×** | faer wins n≥256 (MKL 0.56–0.59×) → **reimplement** |
| **LU** | 0.88×* | 0.81×* | 0.89×* | **1.05×** | inconclusive; only clean point (n=512) has **OpenBLAS winning** |
| **SVD** | 1.22× | 1.08× | 0.99× | 0.88×* | ~parity, no firm gap (LAPACK dgesdd competitive) |

`*` = OpenBLAS rel-σ > 15% (untrusted — see note).

**Measurement note:** Julia-side σ explodes on `lu!`/`svd!`/`qr!` (workspace/ipiv allocation →
GC pauses, σ up to 320%), while faer stays tight (Rust allocs don't hit Julia's GC). This asymmetry
makes small-n LU/SVD medians unreliable here; firm verdicts need LAPACK called with preallocated
workspace (no GC in the timed region).

## Notes on the skips

- **matrixmultiply:** pure-Julia Octavian / `@turbo` are the strong contenders at larger n (1.29×);
  OpenBLAS ties the crate at n=128–256, beats it elsewhere. MKL not needed.
- **glam:** StaticArrays is the baseline to match, not glam. `SmallMatrix.Mat4*Vec4` is slower than
  SMatrix — a codegen note (SMatrix exposes all 16 FMAs; our four-`Vec4` chain has a longer dependency
  chain), not a faer-style gap.
- **SmallMatrix** passes its per-submodule StrictMode audit (`cross/dot/norm/normalize`: typestable +
  noalloc).

## Open follow-up

Probe **RecursiveFactorization.jl** (pure-Julia recursive LU, known to beat OpenBLAS small-n) and, to
chase the real gap, prototype a pure-Julia recursive **Cholesky / QR** in a `Factorizations`
submodule — the definitive test of whether the n≥256 gap is a true pure-Julia gap or only
stdlib-vs-faer.
