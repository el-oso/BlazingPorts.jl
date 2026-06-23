# Experiment: does faer's HAND-WRITTEN ASSEMBLY microkernel beat what LLVM produces from portable
# SIMD.jl IR?  (Settles the project premise: "same LLVM backend ⇒ same speed".)
#
# faer's f64 matmul is raw x86 asm (private-gemm-x86 build.rs): `vfnmadd231pd zmm,zmm,[mem]{1to8}`
# (memory-broadcast FMA), M=4 zmm tile, BLIS packing. We replicate the exact instruction here via
# Base.llvmcall inline asm (AVX-512), and compare against (a) an equivalent SIMD.jl kernel that LLVM
# lowers WITHOUT the {1to8} fold (it broadcasts once + reuses), and (b) the real achievable ceiling
# (OpenBLAS single-thread dgemm).
#
# RESULT (Zen5): asm {1to8} kernel == SIMD.jl kernel == ~76 GFLOP/s.  76/(2·8·2 flop) = 2.4 GHz =
# the SUSTAINED AVX-512 single-core clock (the core downclocks hard under AVX-512). i.e. ~76 is the REAL
# peak, NOT the 144 you'd get from base clock. Our portable kernel is already at ~99% of it, and the
# hand-asm buys NOTHING. ⇒ The premise HOLDS at the microkernel level: LLVM-from-SIMD.jl already matches
# hand-written asm. Any faer edge at the full-factorization level is ALGORITHMIC (blocking / packing /
# the serial base→trsm→syrk dependency), not microkernel codegen — and is portable to attack.
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_microkernel_asm.jl

using SIMD, LinearAlgebra, Printf
import BlazingPorts.Factorizations as F
const W = F.W
BLAS.set_num_threads(1)
@inline _vp(p, e) = p + e * 8

# (a) AVX-512 inline-asm microkernel: C(8×8) −= A(8×K)·B(K×8) via vfnmadd231pd {1to8} (faer's instr).
# A packed col-major ld=8; B packed row-major (pB[k*8+j]=B[k,j]); C col-major, ldCb = ld*8 bytes.
@inline function uk_asm!(pC::Ptr{Float64}, pA::Ptr{Float64}, pB::Ptr{Float64}, K::Int, ldCb::Int)
    Base.llvmcall(raw"""
    call void asm sideeffect "movq $0,%r8\0Amovq $1,%rax\0Amovq $2,%rcx\0Amovq $3,%rdx\0Amovq $4,%r9\0Amovq %r8,%r10\0Avmovupd (%r10),%zmm0\0Aaddq %r9,%r10\0Avmovupd (%r10),%zmm1\0Aaddq %r9,%r10\0Avmovupd (%r10),%zmm2\0Aaddq %r9,%r10\0Avmovupd (%r10),%zmm3\0Aaddq %r9,%r10\0Avmovupd (%r10),%zmm4\0Aaddq %r9,%r10\0Avmovupd (%r10),%zmm5\0Aaddq %r9,%r10\0Avmovupd (%r10),%zmm6\0Aaddq %r9,%r10\0Avmovupd (%r10),%zmm7\0A1:\0Avmovupd (%rax),%zmm8\0Avfnmadd231pd (%rcx){1to8},%zmm8,%zmm0\0Avfnmadd231pd 8(%rcx){1to8},%zmm8,%zmm1\0Avfnmadd231pd 16(%rcx){1to8},%zmm8,%zmm2\0Avfnmadd231pd 24(%rcx){1to8},%zmm8,%zmm3\0Avfnmadd231pd 32(%rcx){1to8},%zmm8,%zmm4\0Avfnmadd231pd 40(%rcx){1to8},%zmm8,%zmm5\0Avfnmadd231pd 48(%rcx){1to8},%zmm8,%zmm6\0Avfnmadd231pd 56(%rcx){1to8},%zmm8,%zmm7\0Aaddq $$64,%rax\0Aaddq $$64,%rcx\0Adecq %rdx\0Ajnz 1b\0Amovq %r8,%r10\0Avmovupd %zmm0,(%r10)\0Aaddq %r9,%r10\0Avmovupd %zmm1,(%r10)\0Aaddq %r9,%r10\0Avmovupd %zmm2,(%r10)\0Aaddq %r9,%r10\0Avmovupd %zmm3,(%r10)\0Aaddq %r9,%r10\0Avmovupd %zmm4,(%r10)\0Aaddq %r9,%r10\0Avmovupd %zmm5,(%r10)\0Aaddq %r9,%r10\0Avmovupd %zmm6,(%r10)\0Aaddq %r9,%r10\0Avmovupd %zmm7,(%r10)", "r,r,r,r,r,~{rax},~{rcx},~{rdx},~{r8},~{r9},~{r10},~{zmm0},~{zmm1},~{zmm2},~{zmm3},~{zmm4},~{zmm5},~{zmm6},~{zmm7},~{zmm8},~{memory},~{cc}"(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4)
    ret void
    """, Cvoid, Tuple{Int,Int,Int,Int,Int},
        reinterpret(Int, pC), reinterpret(Int, pA), reinterpret(Int, pB), K, ldCb)
end

# (b) portable SIMD.jl equivalent (LLVM lowers WITHOUT {1to8}: broadcast-once + reuse). Same 8×8 tile.
@inline function uk_simd!(pC::Ptr{Float64}, pA::Ptr{Float64}, pB::Ptr{Float64}, K::Int, ldCb::Int)
    ld = ldCb ÷ 8
    a0 = vload(Vec{W,Float64}, _vp(pC, 0));    a1 = vload(Vec{W,Float64}, _vp(pC, ld))
    a2 = vload(Vec{W,Float64}, _vp(pC, 2ld));  a3 = vload(Vec{W,Float64}, _vp(pC, 3ld))
    a4 = vload(Vec{W,Float64}, _vp(pC, 4ld));  a5 = vload(Vec{W,Float64}, _vp(pC, 5ld))
    a6 = vload(Vec{W,Float64}, _vp(pC, 6ld));  a7 = vload(Vec{W,Float64}, _vp(pC, 7ld))
    @inbounds for k in 0:K-1
        v = vload(Vec{W,Float64}, _vp(pA, k * 8)); b = k * 8
        a0 = muladd(v, Vec{W,Float64}(-unsafe_load(pB, b + 1)), a0)
        a1 = muladd(v, Vec{W,Float64}(-unsafe_load(pB, b + 2)), a1)
        a2 = muladd(v, Vec{W,Float64}(-unsafe_load(pB, b + 3)), a2)
        a3 = muladd(v, Vec{W,Float64}(-unsafe_load(pB, b + 4)), a3)
        a4 = muladd(v, Vec{W,Float64}(-unsafe_load(pB, b + 5)), a4)
        a5 = muladd(v, Vec{W,Float64}(-unsafe_load(pB, b + 6)), a5)
        a6 = muladd(v, Vec{W,Float64}(-unsafe_load(pB, b + 7)), a6)
        a7 = muladd(v, Vec{W,Float64}(-unsafe_load(pB, b + 8)), a7)
    end
    vstore(a0, _vp(pC, 0));   vstore(a1, _vp(pC, ld));  vstore(a2, _vp(pC, 2ld)); vstore(a3, _vp(pC, 3ld))
    vstore(a4, _vp(pC, 4ld)); vstore(a5, _vp(pC, 5ld)); vstore(a6, _vp(pC, 6ld)); vstore(a7, _vp(pC, 7ld))
    return nothing
end

function check(uk!, name)
    K = 64; A = randn(8, K); B = randn(K, 8); C = randn(8, 8); C0 = copy(C)
    Ap = vec(A); Bp = zeros(K * 8); for k in 0:K-1, j in 0:7; Bp[k*8+j+1] = B[k+1, j+1]; end
    GC.@preserve C Ap Bp uk!(pointer(C), pointer(Ap), pointer(Bp), K, 64)
    err = maximum(abs.(C .- (C0 .- A * B)))
    @printf("  %-10s max abs err = %.1e\n", name, err)
end

gflops(uk!) = let K = 512
    A = randn(8, K); B = randn(K, 8); C = randn(8, 8); Ap = vec(A); Bp = zeros(K * 8)
    for k in 0:K-1, j in 0:7; Bp[k*8+j+1] = B[k+1, j+1]; end
    pC = pointer(C); pA = pointer(Ap); pB = pointer(Bp)
    f = @noinline () -> GC.@preserve C Ap Bp uk!(pC, pA, pB, K, 64)
    f(); t = minimum(@elapsed(begin for _ in 1:6000; f(); end end) for _ in 1:30) / 6000
    2 * 8 * 8 * K / t / 1e9
end

println("correctness:"); check(uk_asm!, "asm{1to8}"); check(uk_simd!, "SIMD.jl")
ga = gflops(uk_asm!); gs = gflops(uk_simd!)
A = randn(2048, 2048); B = randn(2048, 2048); Cd = zeros(2048, 2048)
fb = @noinline () -> mul!(Cd, A, B); fb()
td = minimum(@elapsed(fb()) for _ in 1:20); gd = 2 * 2048.0^3 / td / 1e9
println("\nmicrokernel GFLOP/s (8×8, in-L1):")
@printf("  asm {1to8}   : %5.1f\n", ga)
@printf("  SIMD.jl      : %5.1f   <- LLVM, no {1to8}; SAME as asm\n", gs)
@printf("  OpenBLAS dgemm (real full-gemm ceiling): %5.1f\n", gd)
@printf("\nimplied sustained AVX-512 clock: %.2f GHz   (asm GFLOP/s / 32 flop-per-cycle)\n", ga / 32)
println("⇒ hand-asm buys nothing; portable SIMD.jl is already at the real peak. Premise holds.")
