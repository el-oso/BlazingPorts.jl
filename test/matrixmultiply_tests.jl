@testitem "matrixmultiply" tags = [:matrixmultiply] begin
    # MatrixMultiply is probe-first (see src/MatrixMultiply.jl): stdlib BLAS (→ MKL) is expected to
    # beat matrixmultiply; the pure-Julia comparison is Octavian / @turbo, in
    # bench/probe_matrixmultiply.jl. This testitem is a placeholder until/unless we ship our own
    # StrictMode-coverage microkernel here, at which point add correctness vs LinearAlgebra.mul!
    # and a per-submodule audit (:vectorized,:noalloc).
    using BlazingPorts: MatrixMultiply
    @test MatrixMultiply isa Module
end

@testitem "matrixmultiply_strictmode" tags = [:matrixmultiply] begin
    # Probe-first: the module is an empty shell until the probe ships a microkernel. The audit is
    # wired NOW so any future kernel lands under (:vectorized, :noalloc) automatically instead of
    # shipping unchecked (an empty module audits to zero checks — harmless).
    using BlazingPorts: MatrixMultiply
    using StrictMode, AllocCheck, JET
    fs = audit(MatrixMultiply; sweep = true, guarantees = (:vectorized, :noalloc), io = devnull)
    @test nfailures(fs) == 0
end
