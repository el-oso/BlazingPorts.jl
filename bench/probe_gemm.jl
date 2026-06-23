# Pure-Julia BLIS-packed gemm (C = αAB + C). Built as Phase 1 of a faer-grade QR backend.
#
# RESULT: it reaches/【beats】 OpenBLAS at SQUARE (2048³: ours 65.7 vs OpenBLAS 61.7 GFLOP/s — a genuine
# pure-Julia peak gemm). BUT it is the WRONG tool for the QR dlarfb: those gemms are SKINNY (W=VᵀC has
# M=pb≈64–128; C−=VY has K=pb), where the A/B packing overhead is ~the gemm cost, and the many tiny inner
# within-panel updates pay the full packed-loop machinery. Integrating it into the QR dlarfb gave 0.58× at
# 2048 — WORSE than the shipped non-packed register-tiled two-level (0.73×). So packing helps square gemm
# but not the skinny QR shapes; the shipped two-level register kernels are better there. Kept as a
# standalone artifact (a pure-Julia gemm that beats OpenBLAS at square). Run with --project=bench.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_gemm.jl

import BlazingPorts.Factorizations as F
import SIMD: Vec, vload, vstore
using LinearAlgebra, Printf
BLAS.set_num_threads(1)

const W = F.W                 # 8 (AVX-512)
const MR = 2                  # zmm rows  → MRW = 16 rows per microkernel tile
const NR = 6                  # cols per tile  (MR*NR=12 zmm accumulators)
const MRW = MR * W
@inline _vp(p, e) = p + e * sizeof(Float64)

# cache blocks (this host; src will query VectorizationBase.cache_size)
const L1 = 49152
const L2 = 1048576
const KC = let kc = (L1 ÷ 2) ÷ (NR * 8); max(W, (kc ÷ W) * W) end          # B̃ NR-panel (KC×NR) in L1
const MC = let mc = (L2 ÷ 2) ÷ (KC * 8); max(MRW, (mc ÷ MRW) * MRW) end    # Ã (MC×KC) in L2
const NC = 4096                                                            # B̃ (KC×NC) in L3

# pack A's (mc×kc) block at (i0,k0) of A(ld) into Ãp: MR-panels, within panel col-major (k outer, MRW rows)
function packA!(pAp, pA, i0, k0, mc, kc, ld)
    np = cld(mc, MRW)
    @inbounds for p in 0:np-1
        base = p * MRW * kc
        for k in 0:kc-1
            o = base + k * MRW
            for r in 0:MRW-1
                gr = i0 + p * MRW + r
                unsafe_store!(pAp, (p * MRW + r < mc) ? unsafe_load(pA, (k0 + k) * ld + gr + 1) : 0.0, o + r + 1)
            end
        end
    end
end
# pack B's (kc×nc) block at (k0,j0) of B(ld) into B̃p: NR-panels, within panel row-major (k outer, NR cols)
function packB!(pBp, pB, k0, j0, kc, nc, ld)
    np = cld(nc, NR)
    @inbounds for p in 0:np-1
        base = p * kc * NR
        for k in 0:kc-1
            o = base + k * NR
            for c in 0:NR-1
                gc = j0 + p * NR + c
                unsafe_store!(pBp, (p * NR + c < nc) ? unsafe_load(pB, gc * ld + (k0 + k) + 1) : 0.0, o + c + 1)
            end
        end
    end
end

# microkernel: C[ir.., jr..] += Ãpanel(MRW×kc) · B̃panel(kc×NR), α applied. Writes only valid mr×nr.
@inline function uk!(pC, ldc, pAp, aoff, pBp, boff, kc, ir, jr, mr, nr, α)
    a0 = Vec{W,Float64}(0.0); a1 = Vec{W,Float64}(0.0); a2 = Vec{W,Float64}(0.0)
    a3 = Vec{W,Float64}(0.0); a4 = Vec{W,Float64}(0.0); a5 = Vec{W,Float64}(0.0)
    b0 = Vec{W,Float64}(0.0); b1 = Vec{W,Float64}(0.0); b2 = Vec{W,Float64}(0.0)
    b3 = Vec{W,Float64}(0.0); b4 = Vec{W,Float64}(0.0); b5 = Vec{W,Float64}(0.0)
    @inbounds for k in 0:kc-1
        av0 = vload(Vec{W,Float64}, _vp(pAp, aoff + k * MRW))
        av1 = vload(Vec{W,Float64}, _vp(pAp, aoff + k * MRW + W))
        bk = boff + k * NR
        s0 = Vec{W,Float64}(unsafe_load(pBp, bk + 1)); a0 = muladd(av0, s0, a0); b0 = muladd(av1, s0, b0)
        s1 = Vec{W,Float64}(unsafe_load(pBp, bk + 2)); a1 = muladd(av0, s1, a1); b1 = muladd(av1, s1, b1)
        s2 = Vec{W,Float64}(unsafe_load(pBp, bk + 3)); a2 = muladd(av0, s2, a2); b2 = muladd(av1, s2, b2)
        s3 = Vec{W,Float64}(unsafe_load(pBp, bk + 4)); a3 = muladd(av0, s3, a3); b3 = muladd(av1, s3, b3)
        s4 = Vec{W,Float64}(unsafe_load(pBp, bk + 5)); a4 = muladd(av0, s4, a4); b4 = muladd(av1, s4, b4)
        s5 = Vec{W,Float64}(unsafe_load(pBp, bk + 6)); a5 = muladd(av0, s5, a5); b5 = muladd(av1, s5, b5)
    end
    accs = (a0, a1, a2, a3, a4, a5); accs2 = (b0, b1, b2, b3, b4, b5)
    av = Vec{W,Float64}(α)
    @inbounds for c in 1:nr                              # store α·acc into C (valid rows only)
        col = (jr + c - 1) * ldc + ir
        if mr >= W                                       # full first vector
            p0 = _vp(pC, col); vstore(muladd(av, accs[c], vload(Vec{W,Float64}, p0)), p0)
            if mr >= 2W
                p1 = _vp(pC, col + W); vstore(muladd(av, accs2[c], vload(Vec{W,Float64}, p1)), p1)
            else
                for r in W:mr-1
                    e = col + r; unsafe_store!(pC, unsafe_load(pC, e + 1) + α * accs2[c][r-W+1], e + 1)
                end
            end
        else
            for r in 0:mr-1
                e = col + r; unsafe_store!(pC, unsafe_load(pC, e + 1) + α * accs[c][r+1], e + 1)
            end
        end
    end
end

# C(m×n, ldc) += α · A(m×k, lda) · B(k×n, ldb)   (packed, NN)
function pgemm!(pC, ldc, pA, lda, pB, ldb, m, n, k, α, Ap, Bp)
    pAp = pointer(Ap); pBp = pointer(Bp)
    jc = 0
    @inbounds while jc < n
        nc = min(NC, n - jc)
        kk = 0
        while kk < k
            kc = min(KC, k - kk)
            packB!(pBp, pB, kk, jc, kc, nc, ldb)
            ic = 0
            while ic < m
                mc = min(MC, m - ic)
                packA!(pAp, pA, ic, kk, mc, kc, lda)
                jr = 0
                while jr < nc
                    nr = min(NR, nc - jr)
                    bpan = (jr ÷ NR) * kc * NR
                    ir = 0
                    while ir < mc
                        mr = min(MRW, mc - ir)
                        apan = (ir ÷ MRW) * MRW * kc
                        uk!(pC, ldc, pAp, apan, pBp, bpan, kc, ic + ir, jc + jr, mr, nr, α)
                        ir += MRW
                    end
                    jr += NR
                end
                ic += mc
            end
            kk += kc
        end
        jc += nc
    end
end

# ---- test ----
println("BLIS packed gemm  (MR=$MR×W=$MRW rows, NR=$NR; KC=$KC MC=$MC; peak ~76 in-cache, dgemm ~62)")
function check(m, n, k)
    A = randn(m, k); B = randn(k, n); C = randn(m, n); C0 = copy(C); α = 1.0
    Ap = Vector{Float64}(undef, (cld(MC, MRW) * MRW) * KC); Bp = Vector{Float64}(undef, (cld(NC, NR) * NR) * KC)
    GC.@preserve A B C Ap Bp pgemm!(pointer(C), m, pointer(A), m, pointer(B), k, m, n, k, α, Ap, Bp)
    err = maximum(abs.(C .- (C0 .+ α .* A * B))) / max(1, maximum(abs.(A * B)))
    err
end
@printf("correctness 200×200×200 err=%.1e ;  37×53×91 err=%.1e\n", check(200, 200, 200), check(37, 53, 91))

bt(f; r=10) = (f(); minimum(@elapsed(f()) for _ in 1:r))
function gfl(m, n, k)
    A = randn(m, k); B = randn(k, n); C = zeros(m, n)
    Ap = Vector{Float64}(undef, (cld(MC, MRW) * MRW) * KC); Bp = Vector{Float64}(undef, (cld(NC, NR) * NR) * KC)
    pA = pointer(A); pB = pointer(B); pC = pointer(C)
    f = @noinline () -> GC.@preserve A B C Ap Bp pgemm!(pC, m, pA, m, pB, k, m, n, k, 1.0, Ap, Bp)
    t = bt(f); 2.0 * m * n * k / t / 1e9
end
println("GFLOP/s (ours vs the shapes that matter):")
@printf("  square 2048³        : %.1f\n", gfl(2048, 2048, 2048))
@printf("  W=VᵀC  (64×1984×2000): %.1f\n", gfl(64, 1984, 2000))   # M=pb small, K=mp large
@printf("  C-=VY  (2000×1984×64): %.1f\n", gfl(2000, 1984, 64))   # K=pb small
let n = 2048
    A = randn(n, n); B = randn(n, n); C = zeros(n, n); fb = @noinline () -> mul!(C, A, B)
    @printf("  OpenBLAS square 2048: %.1f\n", 2.0 * n^3 / bt(fb) / 1e9)
end
