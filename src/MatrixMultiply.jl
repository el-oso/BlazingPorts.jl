"""
    MatrixMultiply

gemm microkernel — the `matrixmultiply` (bluss) probe.

**Status: probe-first (likely documented-skip for the use case).** Julia's stdlib `*` / `mul!`
(OpenBLAS, → MKL via `using MKL` if OpenBLAS lags) is expected to beat matrixmultiply outright.
The interesting comparison is the pure-Julia tier — Octavian.jl and a LoopVectorization `@turbo`
kernel — measured single-threaded in `bench/probe_matrixmultiply.jl`.

This module optionally hosts our own pure-Julia microkernel built purely as the StrictMode
`@assert_vectorized` + `@unroll`(`@generated`) + `@assert_noalloc` probe, benchmarked against the
`@turbo`/Octavian references. Implemented only when we decide that StrictMode coverage is worth it
(or a probe falls below 0.96).
"""
module MatrixMultiply

# Intentionally empty until the probe decides an implementation is warranted (see gap log).

end # module MatrixMultiply
