# Probe: QR — our blocked/unblocked Householder vs faer vs OpenBLAS (LAPACK geqrf). Single-threaded,
# GC-controlled, Chairmarks ≥1000 samples, median+σ. Build Rust lib first: bash bench/rust_compare/build.sh
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_qr.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using BlazingPorts.Factorizations: qr_unblocked!, qr_blocked!
using LinearAlgebra

Harness.single_thread!()
const LIB = Harness.RUST_LIB
@noinline faer_qr(A::Matrix{Float64}) =
    ccall((:faer_qr, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

function probe_size(n::Int)
    A = randn(n, n)
    let s = copy(A); tau = zeros(n); qr_blocked!(s, tau)
        # reconstruction sanity
        R = triu(s); Q = Matrix{Float64}(I, n, n)
        for k in 1:n
            t = tau[k]; isinf(t) && continue
            v = [i == 1 ? 1.0 : s[k+i-1, k] for i in 1:(n - k + 1)]
            Q[:, k:n] .-= (Q[:, k:n] * v) * v' ./ t
        end
        @assert maximum(abs.(Q * R .- A)) / maximum(abs.(A)) < 1e-11 "qr_blocked recon off n=$n"
    end
    sb = copy(A); su = copy(A); sk = copy(A); sf = copy(A)
    tu = zeros(n); tk = zeros(n)
    fb = @noinline () -> (copyto!(sb, A); qr!(sb); GC.gc(false); sb[1])
    fu = @noinline () -> (copyto!(su, A); qr_unblocked!(su, tu); GC.gc(false); su[1])
    fk = @noinline () -> (copyto!(sk, A); qr_blocked!(sk, tk); GC.gc(false); sk[1])
    ff = @noinline () -> (copyto!(sf, A); faer_qr(sf))
    fb(); fu(); fk(); ff()
    GC.enable(false); GC.gc(false)
    pb = run_probe("OpenBLAS", fb; seconds = 3.0)
    pu = run_probe("BP-unblocked", fu; seconds = 3.0)
    pk = run_probe("BP-blocked", fk; seconds = 3.0)
    GC.enable(true); GC.gc()
    pf = run_probe("faer", ff; seconds = 3.0)
    probes = Probe[pb, pu, pk, pf]
    crate = "qr_$(n)x$(n)"
    report(crate, probes; rust_label = "faer")
    save_probes(crate, probes)
    plot_probe(crate, probes)
    println()
    return probes
end

println("\n=== QR: blocked / unblocked vs faer / OpenBLAS (single-threaded) ===\n")
for n in (64, 128, 256, 512)
    probe_size(n)
end
println("Done — probe_qr.jl")
