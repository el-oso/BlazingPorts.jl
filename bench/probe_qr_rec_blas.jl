# Structure-validation: recursive QR (Elmroth–Gustavson / faer) with the fat trailing gemms done by an
# OPTIMAL gemm (single-thread BLAS) — to confirm the recursive structure reaches faer at large n. If yes,
# the only missing pure-Julia piece is a cache-blocked gemm for the fat dlarfb (our register-tiled kernels
# re-stream C at large pb). Octavian (pure-Julia BLIS gemm) would be the drop-in.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_qr_rec_blas.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using BlazingPorts.Factorizations: qr_unblocked!, qr_blocked!
using LinearAlgebra, Printf
Harness.single_thread!()
const LIB = Harness.RUST_LIB
faer_qr(A::Matrix{Float64}) = ccall((:faer_qr, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))
const LEAF = 32

function build_T!(A, tau, T, d, nc, m)
    @inbounds for c in 1:nc
        gc = d + c - 1; tc = tau[gc]; λ = isfinite(tc) ? 1.0 / tc : 0.0; T[gc, gc] = λ
        if c > 1 && λ != 0.0
            w = zeros(c - 1)                       # w[k] = V_kᵀ V_c (unit diag)
            for k in 1:c-1
                gk = d + k - 1; s = A[gc, gk]
                for i in gc+1:m; s += A[i, gk] * A[i, gc]; end
                w[k] = s
            end
            for r in 1:c-1                          # T[1:c-1,c] = −λ T[1:c-1,1:c-1] w
                s = 0.0; for k in r:c-1; s += T[d+r-1, d+k-1] * w[k]; end
                T[d+r-1, gc] = -λ * s
            end
        end
    end
end

# dense unit-lower-trapezoid V for reflectors [d:d+nc-1], rows [d:m]
function denseV(A, d, nc, m)
    mp = m - d + 1; V = zeros(mp, nc)
    @inbounds for c in 1:nc, i in c:mp
        V[i, c] = i == c ? 1.0 : A[d+i-1, d+c-1]
    end
    V
end

function qr_rec!(A, tau, T, d, nc, m)
    if nc <= LEAF
        @views qr_unblocked!(A[d:m, d:d+nc-1], tau[d:d+nc-1]); build_T!(A, tau, T, d, nc, m); return
    end
    n1 = nc ÷ 2; d2 = d + n1; n2 = nc - n1
    qr_rec!(A, tau, T, d, n1, m)
    # fat dlarfb: C = A[d:m, d2:d+nc-1]  ;  C −= V1 (T11ᵀ (V1ᵀ C))   [BLAS]
    V1 = denseV(A, d, n1, m)
    @views begin
        C = A[d:m, d2:d+nc-1]
        Wm = V1' * C                                  # n1×n2
        Y = UpperTriangular(T[d:d+n1-1, d:d+n1-1])' * Wm
        mul!(C, V1, Y, -1.0, 1.0)                      # C −= V1 Y
    end
    qr_rec!(A, tau, T, d2, n2, m)
    # combine T12 = −T11 (V1lowᵀ V2) T22
    @views begin
        V1low = A[d2:m, d:d+n1-1]
        V2 = denseV(A, d2, n2, m)
        M = V1low' * V2                                # n1×n2
        T11 = UpperTriangular(T[d:d+n1-1, d:d+n1-1]); T22 = UpperTriangular(T[d2:d2+n2-1, d2:d2+n2-1])
        T[d:d+n1-1, d2:d2+n2-1] .= .-(T11 * (M * T22))
    end
end

function qr_rec_blas!(A, tau)
    m, n = size(A); T = zeros(n, n); qr_rec!(A, tau, T, 1, n, m); true
end

# correctness
for n in (64, 128, 257)
    A = randn(n, n); A0 = copy(A); tau = zeros(n); qr_rec_blas!(A, tau)
    R = triu(A); Q = Matrix{Float64}(I, n, n)
    for k in 1:n
        t = tau[k]; isinf(t) && continue
        v = [i == 1 ? 1.0 : A[k+i-1, k] for i in 1:(n-k+1)]
        Q[:, k:n] .-= (Q[:, k:n] * v) * v' ./ t
    end
    @printf("recon n=%-4d rel=%.1e\n", n, maximum(abs.(Q * R .- A0)) / maximum(abs.(A0)))
end

println("\nrecursive(BLAS gemm) vs faer vs flat-blocked  (ratio faer/ours):")
bestt(f; r=15) = (f(); minimum(@elapsed(f()) for _ in 1:r))
for n in (512, 1024, 2048)
    A = randn(n, n)
    sr = copy(A); tr = zeros(n); sk = copy(A); tk = zeros(n); sf = copy(A)
    fr = @noinline () -> (copyto!(sr, A); qr_rec_blas!(sr, tr); sr[1])
    fk = @noinline () -> (copyto!(sk, A); qr_blocked!(sk, tk); GC.gc(false); sk[1])
    ff = @noinline () -> (copyto!(sf, A); faer_qr(sf))
    tr2 = bestt(fr); tk2 = bestt(fk); tf = bestt(ff)
    @printf("  n=%-4d  recursive %.2f×   flat %.2f×   (faer=%.1fms)\n", n, tf / tr2, tf / tk2, tf * 1e3)
end
