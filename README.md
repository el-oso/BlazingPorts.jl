# BlazingPorts.jl

Reimplementations of fast Rust crates in Julia, **gated by [StrictMode](../StrictMode.jl)** and
benchmarked against the real crate (Rust via a vendored cdylib, `ccall`). Sibling project to
[PureFFT.jl](../PureFFT.jl) (rustfft → pure-Julia FFT), which is the proven template.

The point isn't to win benchmarks — it's to find where StrictMode's performance guarantees hold,
miss, or need a new lever. See the campaign tracker `../blazingly-fast-rust-crates.md`.

## Probe-first

For each crate we **first benchmark current Julia** (Base / stdlib / ecosystem) against the Rust
crate, **single-threaded**. If Julia is already good enough (parity ≥ 0.96), we *do not implement* —
we record the evidence in the gap log and move on. We only ship a Julia implementation here when a
probe falls below the gate, or as an explicit StrictMode-coverage kernel.

## Methodology (locked)

- **Single-threaded both sides** — instruction-level comparison, not parallelism: `julia -t 1`,
  `taskset -c N`, `BLAS.set_num_threads(1)`, Rust built with `RAYON_NUM_THREADS=1`.
- **[Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl)** for local micro-timing,
  **≥ 1000 samples**, compare **median** + check **rel-σ** (noise floor 5%).
- **Parity gate**: `rust_median / julia_median ≥ 0.96`.
- **[Plots.jl](https://github.com/JuliaPlots/Plots.jl)** comparison charts → `docs/assets/`.

## Layout

| Path | What |
|------|------|
| `src/SmallMatrix.jl`   | Vec3/Vec4/Mat4 stack math (glam/nalgebra) — StrictMode `@assert_noboxing/_noalloc` probe |
| `src/SpecialFns.jl`    | erf/gamma kernels (libm/statrs) — probe-first, likely document-skip vs SpecialFunctions.jl |
| `src/MatrixMultiply.jl`| optional gemm microkernel (matrixmultiply) — probe-first vs OpenBLAS/MKL/Octavian/`@turbo` |
| `bench/harness.jl`     | shared probe harness (Chairmarks median+σ, parity gate, Plots) |
| `bench/probe_*.jl`     | one probe per crate |
| `bench/rust_compare/`  | single Cargo workspace → one cdylib (`libblazing_compare.so`) with C-ABI shims per crate |
| `bench/strictmode_audit.jl` | per-submodule StrictMode `audit` |
| `test/*_tests.jl`      | ReTestItems `@testitem`, one per crate, **tagged** for isolated runs |

## Running

```bash
# Tests — full suite, or one crate in isolation via its tag
julia --project -e 'using Pkg; Pkg.test()'
julia --project=test -e 'using ReTestItems, BlazingPorts; runtests(BlazingPorts; tags=[:smallmatrix])'

# Build the Rust comparison cdylib (single-threaded)
bash bench/rust_compare/build.sh

# Probe a crate (pin a core; keep its SMT sibling idle)
taskset -c 2 julia -t 1 --project=bench bench/probe_smallmatrix.jl

# Per-submodule StrictMode audit
julia --project=bench bench/strictmode_audit.jl
```

### Saved data & re-plotting

Each probe writes its full (capped) sample distribution to `bench/results/<crate>.json` (tracked in
git). To restyle plots **without re-running benchmarks**:

```julia
include("bench/harness.jl"); using .Harness
replot()              # regenerate every plot from cached JSON (violin + box)
replot(kind = :box)   # box-only variant
```

Plots are **violin + box** (per-eval time, ns, lower = better) — the full distribution, not just the
median; statistics use every collected sample while only a ≤2000-point subsample is stored/plotted.

## No Python

Per the global rule, there is no Python anywhere — the Rust crates are compared via a native cdylib
(`ccall`), the Julia side is Base/stdlib + SIMD.

## License

MIT.
