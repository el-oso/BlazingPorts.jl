# Faithful copy of faer's RECURSIVE blocked QR (Elmroth–Gustavson, = faer qr_in_place_blocked structure):
#   recurse left half → apply its block reflector to the right (FAT dlarfb) → recurse right →
#   combine T:  T12 = −T11·(V1ᵀV2)·T22.
# This makes every trailing update fat (pb = ncols/2) ⇒ low C-traffic + packing pays — the lever the
# flat nb=8 driver lacked at large n. Stage 1 here = CORRECTNESS (naive gemms) to validate the recursion
# + T convention; perf kernels swapped in once recon is green.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_qr_recursive.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.Factorizations as F
using BlazingPorts.Factorizations: qr_unblocked!, qr_blocked!
using LinearAlgebra, Printf
Harness.single_thread!()
const LIB = Harness.RUST_LIB
faer_qr(A::Matrix{Float64}) = ccall((:faer_qr, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

const LEAF = 24

# build T (LAPACK dlarft, forward, λ=1/τ; Q = I − V T Vᵀ) for the nc reflectors at diagonal d.
# fills T[d:d+nc-1, d:d+nc-1]. V_k (global col d+k-1): unit at row d+k-1, essential A[i, d+k-1] for i>.
function build_T!(A, tau, T, d::Int, nc::Int, m::Int)
    @inbounds for c in 1:nc
        gc = d + c - 1
        tc = tau[gc]; λ = isfinite(tc) ? 1.0 / tc : 0.0
        T[d+c-1, d+c-1] = λ
        if c > 1 && λ != 0.0
            w = zeros(c - 1)                       # w[k] = V_kᵀ V_c  (rows gc..m, unit at diag)
            for k in 1:c-1
                gk = d + k - 1; s = 1.0 * A[gc, gk]    # row gc: V_k=A[gc,gk], V_c=1
                for i in gc+1:m; s += A[i, gk] * A[i, gc]; end
                w[k] = s
            end
            for r in 1:c-1                          # T[1:c-1,c] = −λ T[1:c-1,1:c-1] w
                s = 0.0
                for k in r:c-1; s += T[d+r-1, d+k-1] * w[k]; end
                T[d+r-1, d+c-1] = -λ * s
            end
        end
    end
end

# preallocated scratch (reused across all recursion nodes — depth-first, so no temporal overlap)
struct WS
    V::Vector{Float64}; Wm::Vector{Float64}; Y::Vector{Float64}
    V2::Vector{Float64}; M::Vector{Float64}; TM::Vector{Float64}
end
WS(n) = WS((Vector{Float64}(undef, n * n) for _ in 1:6)...)

# apply Q1ᵀ to trailing cols [jc:jc+nt-1]:  C −= V1·(T11ᵀ·(V1ᵀ C)) — fast tiled kernels, FAT pb=n1.
function dlarfb!(pA, T, d::Int, n1::Int, jc::Int, nt::Int, m::Int, ld::Int, ws::WS)
    mp = m - d + 1
    pV = pointer(ws.V); pWm = pointer(ws.Wm); pY = pointer(ws.Y)
    @inbounds for c in 1:n1, i in 1:mp                # build dense V1 (unit lower-trapezoid), ld=mp
        gi = d + i - 1; gc = d + c - 1
        unsafe_store!(pV, gi == gc ? 1.0 : (gi > gc ? unsafe_load(pA, (gc - 1) * ld + gi) : 0.0), (c - 1) * mp + i)
    end
    pC = pA + ((jc - 1) * ld + (d - 1)) * 8
    F._qr_VtC!(pWm, pV, pC, mp, nt, n1, ld)           # W = V1ᵀ C
    @inbounds for j in 1:nt, c in 1:n1                # Y = T11ᵀ W   (ld of Wm,Y = n1)
        s = 0.0
        for r in 1:c; s = muladd(T[d+r-1, d+c-1], unsafe_load(pWm, (j - 1) * n1 + r), s); end
        unsafe_store!(pY, s, (j - 1) * n1 + c)
    end
    F._qr_subVY!(pC, pV, pY, mp, nt, n1, ld)          # C −= V1 Y
end

# combine T:  T12 = −T11·(V1lowᵀ V2)·T22 into T[d:d+n1-1, d+n1:d+nc-1].
function combine_T!(pA, T, d::Int, n1::Int, nc::Int, m::Int, ld::Int, ws::WS)
    n2 = nc - n1; d2 = d + n1; K = m - d2 + 1
    pV1 = pointer(ws.V); pV2 = pointer(ws.V2); pM = pointer(ws.M); pTM = pointer(ws.TM)
    @inbounds for c in 1:n1, i in 1:K; unsafe_store!(pV1, unsafe_load(pA, (d + c - 2) * ld + d2 + i - 1), (c - 1) * K + i); end
    @inbounds for j in 1:n2, i in 1:K
        gi = d2 + i - 1; gj = d2 + j - 1
        unsafe_store!(pV2, gi == gj ? 1.0 : (gi > gj ? unsafe_load(pA, (gj - 1) * ld + gi) : 0.0), (j - 1) * K + i)
    end
    F._qr_VtC!(pM, pV1, pV2, K, n2, n1, K)            # M = V1lowᵀ V2  (ld M = n1)
    @inbounds for j in 1:n2, r in 1:n1               # TM = T11 · M  (ld TM = n1)
        s = 0.0
        for k in r:n1; s = muladd(T[d+r-1, d+k-1], unsafe_load(pM, (j - 1) * n1 + k), s); end
        unsafe_store!(pTM, s, (j - 1) * n1 + r)
    end
    @inbounds for j in 1:n2, r in 1:n1               # T12 = −TM · T22
        s = 0.0
        for k in 1:j; s = muladd(unsafe_load(pTM, (k - 1) * n1 + r), T[d2+k-1, d2+j-1], s); end
        T[d+r-1, d2+j-1] = -s
    end
end

function qr_rec!(A, pA, tau, T, d::Int, nc::Int, m::Int, ld::Int, ws::WS)
    if nc <= LEAF
        qr_unblocked!(view(A, d:m, d:d+nc-1), view(tau, d:d+nc-1))
        build_T!(A, tau, T, d, nc, m)
        return
    end
    n1 = nc ÷ 2
    qr_rec!(A, pA, tau, T, d, n1, m, ld, ws)
    dlarfb!(pA, T, d, n1, d + n1, nc - n1, m, ld, ws)
    qr_rec!(A, pA, tau, T, d + n1, nc - n1, m, ld, ws)
    combine_T!(pA, T, d, n1, nc, m, ld, ws)
end

function qr_recursive!(A::Matrix{Float64}, tau::Vector{Float64}, ws::WS = WS(size(A, 1)))
    m, n = size(A); T = zeros(n, n)
    GC.@preserve A ws qr_rec!(A, pointer(A), tau, T, 1, n, m, stride(A, 2), ws)
    return true
end

# ---- correctness ----
function recon_ok(n)
    A = randn(n, n); A0 = copy(A); tau = zeros(n)
    qr_recursive!(A, tau)
    R = triu(A); Q = Matrix{Float64}(I, n, n)
    for k in 1:n
        t = tau[k]; isinf(t) && continue
        v = [i == 1 ? 1.0 : A[k+i-1, k] for i in 1:(n-k+1)]
        Q[:, k:n] .-= (Q[:, k:n] * v) * v' ./ t
    end
    maximum(abs.(Q * R .- A0)) / maximum(abs.(A0))
end
println("recursive QR reconstruction error (expect ~1e-13):")
for n in (32, 64, 100, 128, 256, 300)
    @printf("  n=%-4d  rel=%.1e\n", n, recon_ok(n))
end

println("\nperf vs faer / current qr_blocked (ratio = faer/ours; >1 beats faer):")
bestt(f; r=20) = (f(); minimum(@elapsed(f()) for _ in 1:r))
for n in (256, 512, 1024, 2048)
    A = randn(n, n)
    sr = copy(A); tr = zeros(n); sk = copy(A); tk = zeros(n); sf = copy(A); wsr = WS(n)
    fr = @noinline () -> (copyto!(sr, A); qr_recursive!(sr, tr, wsr); GC.gc(false); sr[1])
    fk = @noinline () -> (copyto!(sk, A); qr_blocked!(sk, tk); GC.gc(false); sk[1])
    ff = @noinline () -> (copyto!(sf, A); faer_qr(sf))
    fr(); fk(); ff()
    GC.enable(false); GC.gc(false); tr2 = bestt(fr); tk2 = bestt(fk); GC.enable(true)
    tf = bestt(ff)
    @printf("  n=%-4d  recursive %.2f×   flat-blocked %.2f×   (faer=%.1fms)\n", n, tf / tr2, tf / tk2, tf * 1e3)
end
