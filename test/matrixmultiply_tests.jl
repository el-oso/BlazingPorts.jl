@testitem "matrixmultiply" tags = [:matrixmultiply] begin
    # MatrixMultiply is probe-first (see src/MatrixMultiply.jl): stdlib BLAS (→ MKL) is expected to
    # beat matrixmultiply; the pure-Julia comparison is Octavian / @turbo, in
    # bench/probe_matrixmultiply.jl. This testitem is a placeholder until/unless we ship our own
    # StrictMode-coverage microkernel here, at which point add correctness vs LinearAlgebra.mul!
    # and a per-submodule audit (:vectorized,:noalloc).
    using BlazingPorts: MatrixMultiply
    @test MatrixMultiply isa Module
end
