# BlazingPorts.jl

Reimplementations of fast Rust crates in Julia, **gated by [StrictMode](../StrictMode.jl)** and
benchmarked against the real crate (Rust via a vendored cdylib, `ccall`). Sibling project to
[PureFFT.jl](../PureFFT.jl) (rustfft → pure-Julia FFT), which is the proven template.

The point isn't to win benchmarks — it's to find where StrictMode's performance guarantees hold,
miss, or need a new lever. See [`RESULTS.md`](RESULTS.md) for the verdict table and findings (the
in-repo source of truth), and the cross-project index `../blazingly-fast-rust-crates.md`.

## Status (start here)

Probe-first campaign, **mostly complete.** Most crates **document-skip** — Base/stdlib/ecosystem already win or
match (matrixmultiply, exp/log/erf/gamma, glam/nalgebra, ndarray, rand, argmin, ryu, roaring, fxhash/ahash,
memchr-byte). Julia ports shipped only where a *fair re-probe* found a real gap:

- **`Factorizations` (faer Cholesky + QR)** — beats faer at **all** sizes 256–2048, pure SIMD.jl, no asm. The win
  was the gemm *orchestration choice* (read the large operand in place), not codegen. Done.
- **`Blake3`** — beats blake3's **pure-Rust** path 1.60×; the bundled **hand-asm** wins by 13% (the chained-leaf's
  global register allocation, which no compiler frontend reproduces). Pure Julia is the portable default; the
  `blake3_asm` **Preferences switch** (default on-where-available) opts the leaf into blake3's own CC0 `.S` to
  buy that 13% back — full-hash **1.16×** end-to-end. Done.
- **`StringSearch` (memchr substring)** — parity-to-beat vs memmem; **`IntFormat` (itoa)** — beats the crate. Done.
- **`SwissDict` (hashbrown)** — `<:AbstractDict`; wins **miss-heavy** workloads, loses hit-heavy (a workload
  tradeoff, not a clean win). Done.
- **`Utf8` (simdutf8)** — pure-Julia SIMD UTF-8 validator (lemire, `Vec{32}` AVX2 `pshufb`). Base's `isvalid`
  falls to scalar on multibyte; ours is **11× over Base and beats `simdutf8` on both regimes** (multibyte
  1.05×, ASCII parity), byte-exact. Pushing it to parity surfaced two StrictMode findings — **F33**
  (`kernel_report` blind to shuffle/`pshufb` ops) and **F34** (latency- vs bandwidth-bound). Done.
- **`ByteOps` (base64-simd + faster-hex)** — the shuffle-SIMD transcoding library. **All four kernels beat
  their Rust crate** (kernel-only, preallocated): base64 encode **1.69×** / decode **1.12×**, hex encode
  **1.01×** / decode **1.26×**; 13–43× over Julia stdlib; byte-exact, both decoders validating. The probed
  "27× base64 gap" was a measurement artifact (output alloc + `String` + GC timed in the loop) — isolate the
  kernel and pure Julia wins. Done.

**Canonical state:** [`RESULTS.md`](RESULTS.md) (in-repo verdicts + the faer detail) and the cross-project tracker
[`../blazingly-fast-rust-crates.md`](../blazingly-fast-rust-crates.md) (every crate + gap log + next probes).
**Open threads:** `Base.Ryu` serial-divchain PR (drafted, unfiled); **SIMD UTF-8 validation → Julia Base**
(issue + PR drafted in `contrib/upstream/`, unfiled); **StrictMode F33/F34** (cold-agent specs in
`StrictMode.jl/FEEDBACK.md`); SwissDict **group-aligned probing** (to win hits too).

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
