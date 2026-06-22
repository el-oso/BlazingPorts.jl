# Probe: our pure-Julia Cholesky port (Factorizations.cholesky_llt!) vs faer vs OpenBLAS.
# Single-threaded, GC-controlled (GC.enable(false) + per-iter young-gen GC.gc(false)), Chairmarks
# ≥1000 samples, median+σ. The headline parity question: do we reach faer at n≥256 (where faer wins
# OpenBLAS)? Build the Rust lib first: bash bench/rust_compare/build.sh
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_cholesky_port.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using BlazingPorts.Factorizations: cholesky_llt!
using LinearAlgebra

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

@noinline faer_chol(A::Matrix{Float64}) =
    ccall((:faer_cholesky, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

function probe_size(n::Int)
    A = Matrix(let M = randn(n, n); M'M + n * I end)
    sb = copy(A); symb = Symmetric(sb, :L)
    sp = copy(A); sf = copy(A)
    fb = @noinline () -> (copyto!(sb, A); cholesky!(symb); GC.gc(false); sb[1])
    fp = @noinline () -> (copyto!(sp, A); cholesky_llt!(sp); GC.gc(false); sp[1])
    ff = @noinline () -> (copyto!(sf, A); faer_chol(sf))
    # correctness sanity (reconstruction)
    let s = copy(A); cholesky_llt!(s); L = LowerTriangular(s)
        @assert maximum(abs.(L * L' .- A)) / maximum(abs.(A)) < 1e-12 "port reconstruction off at n=$n"
    end
    fb(); fp(); ff()
    GC.enable(false); GC.gc(false)
    pb = run_probe("OpenBLAS", fb; seconds = 3.0)
    pp = run_probe("BlazingPorts", fp; seconds = 3.0)
    GC.enable(true); GC.gc()
    pf = run_probe("faer", ff; seconds = 3.0)
    probes = Probe[pb, pp, pf]
    crate = "cholesky_port_$(n)x$(n)"
    report(crate, probes; rust_label = "faer")
    save_probes(crate, probes)
    plot_probe(crate, probes)
    println()
    return probes
end

println("\n=== Cholesky port vs faer vs OpenBLAS (single-threaded) ===\n")
for n in (64, 128, 256, 512)
    probe_size(n)
end
println("Done — probe_cholesky_port.jl")
