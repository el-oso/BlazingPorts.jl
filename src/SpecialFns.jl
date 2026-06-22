"""
    SpecialFns

erf / gamma special functions — the `libm` / `statrs` probe.

**Status: probe-first (likely documented-skip).** Julia's `SpecialFunctions.jl` already provides
mature pure-Julia `erf`/`gamma`; the plan is to benchmark it single-threaded against statrs/libm
(`bench/probe_specialfns.jl`) and only populate this module with our own polynomial/rational
kernels if SpecialFunctions falls below the 0.96 parity gate, or to add the fully-static StrictMode
coverage (`@assert_inlined` + `@assert_noalloc` + `@assert_trim_safe`).

Base already covers `exp`/`log` (openlibm) — those are probe-only, never implemented here.
"""
module SpecialFns

# Intentionally empty until a probe demands an implementation (see module docstring / gap log).

end # module SpecialFns
