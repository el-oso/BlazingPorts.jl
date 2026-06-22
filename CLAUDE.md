# BlazingPorts.jl — agent guidelines

Project-specific requirements. Sibling to PureFFT.jl; the campaign tracker is
`../blazingly-fast-rust-crates.md` (the canonical status + per-crate verdicts + gap log).

## The discipline (MUST follow)

1. **Probe-first.** For every crate, benchmark current Julia (Base / stdlib / ecosystem) vs the Rust
   crate *before* writing any implementation. If parity ≥ 0.96 single-threaded, **document-skip** —
   record the medians + a gap-log row, implement nothing. Only implement on a sub-0.96 probe or as an
   explicit StrictMode-coverage kernel.

2. **Single-threaded, both sides.** `julia -t 1`, `taskset -c N` (keep the SMT sibling idle),
   `BLAS.set_num_threads(1)`, Rust cdylib built/run with `RAYON_NUM_THREADS=1`. This isolates the
   *kernel*, the part StrictMode reasons about — not the scheduler.

3. **Benchmark correctly.** Chairmarks `@be`, **≥ 1000 samples**, compare **median** (never min),
   report **rel-σ** and require it tight (< 5%); pin the CPU clock for low noise. Call kernels via a
   `@noinline` concrete wrapper over preallocated inputs with a DCE sink — never a closure in the
   timed region. For ns-scale ops, **batch** to amortise `ccall`/timer overhead.

4. **Source carries no StrictMode dep** (mirrors PureFFT). Hot paths use Base `@generated` / SIMD.jl
   / `Base.llvmcall`; the StrictMode `audit` and `@assert_*` guarantees live only in `bench/` + `test/`.
   - **Never index a tuple with a runtime variable** in a hot path (boxes/allocates — 135× in PureFFT).
     Unroll with literal indices or `@generated`. Do not add Unroll.jl.

5. **Baseline policy (Hybrid).** Implementations prefer Base + stdlib + SIMD.jl; a well-established
   stdlib-adjacent package may be a dep only when Base genuinely lacks the primitive (note it in the
   gap log). Ecosystem packages (StaticArrays, SpecialFunctions, Octavian, LoopVectorization) are
   benchmark baselines; the BLAS backend is OpenBLAS, → **MKL** (`using MKL`) if OpenBLAS lags.

6. **One audit per submodule** (`audit(BlazingPorts.<Crate>; sweep=true, guarantees=...)`), not a
   whole-package sweep. One `@testitem` per crate, **tagged**, so a crate runs in isolation.

## Standing rules

- `isnothing(x)` / `!isnothing(x)` — never `=== nothing`.
- Regenerate the relevant `docs/assets/*.png` probe plot before pushing a result.
- Every probe/skip gets a dated row in `../blazingly-fast-rust-crates.md`'s gap log.
- Commit author email: `15278831+el-oso@users.noreply.github.com`.
- End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- No Python anywhere (global rule) — Rust compared via the native cdylib (`ccall`).
