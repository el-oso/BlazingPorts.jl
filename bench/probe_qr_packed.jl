# Apply the Cholesky algorithmic findings to QR — and the (partly NEGATIVE) result.
#
# FINDING: packing V does NOT transfer to QR the way it did to Cholesky. Measured: packed only nudges
# 2048 (0.53→0.58×) and the naive non-packed fallbacks here regress smaller n (keep qr_blocked!'s tiled
# kernels for a fair small-n comparison). WHY: Cholesky's syrk has a FAT reduction (pb=block_size=128) →
# compute-bound, packing-V fixes the last stride → peak. QR's dlarfb has a THIN reduction (pb=nb=8, from
# the ≤512 parity push) → the trailing C is streamed ~n/nb times; at n=2048,nb=8 that's ~4 GB of C traffic
# (~half of faer's runtime). Packing V fixes V-access, NOT the C-streaming traffic — so it barely helps.
#
# QR STATUS: at parity/better through n=1024 (256 1.17×, 512 1.07×, 1024 1.01×); only 2048 falls (0.53×).
# REAL FIX (next step, not packing): TWO-LEVEL / recursive blocking — reduce a FAT outer panel (nb≈64)
# using qr_blocked! with a small inner nb (efficient), then one fat dlarfb (pb=64 → ⅛ the C-traffic, and
# NOW packing V helps because the reduction is fat). faer does exactly this (recursive block panel).
# The nb sweep confirms the tradeoff: at 2048 nb=16 is best (0.55×); bigger nb worse because our flat
# panel reduction is rank-1 — recursion is what makes large nb cheap.
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_qr_packed.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.Factorizations as F
using BlazingPorts.Factorizations: qr_unblocked!, qr_blocked!
using LinearAlgebra, Printf
import CPUSummary
import SIMD: Vec, vload, vstore

Harness.single_thread!()
const LIB = Harness.RUST_LIB
faer_qr(A::Matrix{Float64}) = ccall((:faer_qr, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

const W = F.W
const MR = 2
const NR = 4
const MWR = MR * W
@inline _el(p, i) = unsafe_load(p, i)
@inline _bptr(p, e) = p + e * sizeof(Float64)
l2half() = Int(CPUSummary.cache_size(Val(2))) ÷ 2

# pack rows [0:m) of dense V (mp×pb, ld=mp) into MWR-row panels (c-contiguous, zero-padded).
@inline function _packV!(dst, pV, m::Int, pb::Int)
    np = cld(m, MWR)
    @inbounds for s in 0:np-1
        base = s * MWR * pb
        for c in 1:pb
            o = base + (c - 1) * MWR
            for r in 0:MWR-1
                lr = s * MWR + r
                unsafe_store!(dst, lr < m ? _el(pV, (c - 1) * m + lr + 1) : 0.0, o + r + 1)
            end
        end
    end
end

# packed C −= Vp·Y :  C(mp×nt, ld) , Vp packed (MWR panels) , Y(pb×nt, ld=pb). remainder-safe (masked).
function _subVY_packed!(pC, pVp, pY, mp::Int, nt::Int, pb::Int, ldC::Int)
    lanes = Vec{W,Int}(ntuple(identity, W))
    npan = cld(mp, MWR)
    @inbounds for jb in 0:NR:nt-1
        nc = min(NR, nt - jb)
        for s in 0:npan-1
            r0 = s * MWR
            aoff = s * MWR * pb
            mrows = min(MWR, mp - r0)
            if mrows == MWR && nc == NR
                b0 = _bptr(pC, jb * ldC + r0);        a0 = vload(Vec{W,Float64}, b0); a1 = vload(Vec{W,Float64}, b0 + W * 8)
                b1 = _bptr(pC, (jb + 1) * ldC + r0);  c0 = vload(Vec{W,Float64}, b1); c1 = vload(Vec{W,Float64}, b1 + W * 8)
                b2 = _bptr(pC, (jb + 2) * ldC + r0);  d0 = vload(Vec{W,Float64}, b2); d1 = vload(Vec{W,Float64}, b2 + W * 8)
                b3 = _bptr(pC, (jb + 3) * ldC + r0);  e0 = vload(Vec{W,Float64}, b3); e1 = vload(Vec{W,Float64}, b3 + W * 8)
                for c in 1:pb
                    u0 = vload(Vec{W,Float64}, _bptr(pVp, aoff + (c - 1) * MWR))
                    u1 = vload(Vec{W,Float64}, _bptr(pVp, aoff + (c - 1) * MWR + W))
                    g0 = Vec{W,Float64}(-_el(pY, (jb) * pb + c));     a0 = muladd(g0, u0, a0); a1 = muladd(g0, u1, a1)
                    g1 = Vec{W,Float64}(-_el(pY, (jb + 1) * pb + c)); c0 = muladd(g1, u0, c0); c1 = muladd(g1, u1, c1)
                    g2 = Vec{W,Float64}(-_el(pY, (jb + 2) * pb + c)); d0 = muladd(g2, u0, d0); d1 = muladd(g2, u1, d1)
                    g3 = Vec{W,Float64}(-_el(pY, (jb + 3) * pb + c)); e0 = muladd(g3, u0, e0); e1 = muladd(g3, u1, e1)
                end
                vstore(a0, b0); vstore(a1, b0 + W * 8); vstore(c0, b1); vstore(c1, b1 + W * 8)
                vstore(d0, b2); vstore(d1, b2 + W * 8); vstore(e0, b3); vstore(e1, b3 + W * 8)
            else
                nv = cld(mrows, W)
                for dj in 0:nc-1
                    col = jb + dj
                    for vv in 0:nv-1
                        rr = r0 + vv * W
                        valid = min(W, mrows - vv * W)
                        mask = lanes <= valid
                        cp = _bptr(pC, col * ldC + rr)
                        acc = vload(Vec{W,Float64}, cp, mask)
                        for c in 1:pb
                            av = vload(Vec{W,Float64}, _bptr(pVp, aoff + (c - 1) * MWR + vv * W))
                            acc = muladd(Vec{W,Float64}(-_el(pY, col * pb + c)), av, acc)
                        end
                        vstore(acc, cp, mask)
                    end
                end
            end
        end
    end
end

# packed QR: panel reduce (qr_unblocked! on view) + dlarft T + W=VᵀC + cache-hybrid C−=VY
function qr_packed!(A::Matrix{Float64}, tau::Vector{Float64}; nb::Int = 8)
    m, n = size(A); ld = stride(A, 2); k = min(m, n)
    Vp = Vector{Float64}(undef, (cld(m, MWR) * MWR) * nb)
    pc = 1
    @inbounds while pc <= k
        pb = min(nb, k - pc + 1)
        qr_unblocked!(view(A, pc:m, pc:pc+pb-1), view(tau, pc:pc+pb-1))
        jt0 = pc + pb
        if jt0 <= n
            mp = m - pc + 1; nt = n - jt0 + 1
            V = Matrix{Float64}(undef, mp, pb)
            for c in 1:pb, i in 1:mp
                V[i, c] = i == c ? 1.0 : (i > c ? A[pc+i-1, pc+c-1] : 0.0)
            end
            T = zeros(pb, pb)
            for c in 1:pb
                tc = tau[pc+c-1]; λ = isfinite(tc) ? 1.0 / tc : 0.0; T[c, c] = λ
                if c > 1 && λ != 0.0
                    w = zeros(c - 1)
                    for kk in 1:c-1, i in 1:mp; w[kk] = muladd(V[i, kk], V[i, c], w[kk]); end
                    for r in 1:c-1
                        s = 0.0; for kk in r:c-1; s = muladd(T[r, kk], w[kk], s); end; T[r, c] = -λ * s
                    end
                end
            end
            Wm = Matrix{Float64}(undef, pb, nt); Y = Matrix{Float64}(undef, pb, nt)
            C = view(A, pc:m, jt0:n)
            for j in 1:nt, c in 1:pb              # W = Vᵀ C (dense V columns are contiguous; fine)
                s = 0.0; @simd for i in 1:mp; s = muladd(V[i, c], C[i, j], s); end; Wm[c, j] = s
            end
            for j in 1:nt, c in 1:pb              # Y = Tᵀ W
                s = 0.0; for r in 1:c; s = muladd(T[r, c], Wm[r, j], s); end; Y[c, j] = s
            end
            pC = pointer(A, (jt0 - 1) * ld + pc)
            if mp * nt * sizeof(Float64) > l2half()    # cache-hybrid: pack only when trailing spills ½L2
                GC.@preserve A V Vp Y begin
                    _packV!(pointer(Vp), pointer(V), mp, pb)
                    _subVY_packed!(pC, pointer(Vp), pointer(Y), mp, nt, pb, ld)
                end
            else
                for j in 1:nt, c in 1:pb
                    y = Y[c, j]; y == 0.0 && continue
                    @simd for i in 1:mp; C[i, j] = muladd(-V[i, c], y, C[i, j]); end
                end
            end
        end
        pc += pb
    end
    return true
end

function probe(n)
    A = randn(n, n)
    let s = copy(A); t = zeros(n); qr_packed!(s, t)
        R = triu(s); Q = Matrix{Float64}(I, n, n)
        for kk in 1:n
            tt = t[kk]; isinf(tt) && continue
            v = [i == 1 ? 1.0 : s[kk+i-1, kk] for i in 1:(n-kk+1)]
            Q[:, kk:n] .-= (Q[:, kk:n] * v) * v' ./ tt
        end
        @assert maximum(abs.(Q * R .- A)) / maximum(abs.(A)) < 1e-10 "recon off n=$n"
    end
    sk = copy(A); tk = zeros(n); sp = copy(A); tp = zeros(n); sf = copy(A)
    fk = @noinline () -> (copyto!(sk, A); qr_blocked!(sk, tk); GC.gc(false); sk[1])
    fp = @noinline () -> (copyto!(sp, A); qr_packed!(sp, tp); GC.gc(false); sp[1])
    ff = @noinline () -> (copyto!(sf, A); faer_qr(sf))
    fk(); fp(); ff()
    bestt(f; r=20) = (GC.enable(false); GC.gc(false); v = minimum(@elapsed(f()) for _ in 1:r); GC.enable(true); v)
    tk2 = bestt(fk); tp2 = bestt(fp); tf = (ff(); minimum(@elapsed(ff()) for _ in 1:20))
    @printf("  n=%-4d  current %.2f×   packed %.2f×   (faer=%.1fms)\n", n, tf / tk2, tf / tp2, tf * 1e3)
end

println("\n=== QR: packed dlarfb (cache-hybrid) vs current vs faer (ratio=faer/ours) ===")
for n in (256, 512, 1024, 2048); probe(n); end
println("Done — probe_qr_packed.jl")
