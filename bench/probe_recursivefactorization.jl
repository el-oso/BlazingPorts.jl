# Probe: LU — does the pure-Julia RecursiveFactorization.jl already match faer? (faer LU follow-up)
#
# Contenders (single-threaded, square Float64, GC-controlled — all allocate ipiv/LU wrapper):
#   1. OpenBLAS                — LinearAlgebra.lu!(A)            (LAPACK dgetrf, the stdlib default)
#   2. RecursiveFactorization  — RF.lu!(A, ipiv, Val(true), Val(false))  (pure-Julia recursive blocked)
#   3. faer                    — faer_lu via ccall              (Rust, the reference)
# Sizes: n = 64, 128, 256, 512.
#
# Context: our faer probe found faer beats OpenBLAS LU at n≤256 (OpenBLAS wins n=512). faer doesn't
# compete with Julia code — LinearAlgebra is OpenBLAS/LAPACK (C/Fortran). RecursiveFactorization.jl is
# the EXISTING pure-Julia recursive LU (used by LinearSolve/DiffEq because it beats getrf small-n). If
# RF.jl ≈ faer, the LU gap is already closed in pure Julia → only Cholesky/QR (no pure-Julia recursive
# equivalent) remain genuine reimplementation targets.
#
# Run:  taskset -c 2 julia -t 1 --project=bench bench/probe_recursivefactorization.jl
# Build Rust lib first:  bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using LinearAlgebra
import RecursiveFactorization as RF

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

@noinline function faer_lu!(A::Matrix{Float64})
    ccall((:faer_lu, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))
end

function probe_size(n::Int)
    A = randn(n, n)
    Cref = lu(copy(A))                       # reference factorization for correctness
    # OpenBLAS lu! (allocates ipiv + LU wrapper)
    sb = copy(A)
    fb = @noinline let A = A, s = sb
        () -> (copyto!(s, A); lu!(s); GC.gc(false); s[1])
    end
    # RecursiveFactorization lu! — preallocated ipiv, pivot=Val(true), threading=Val(false)
    sr = copy(A); ipiv = Vector{Int}(undef, n)
    fr = @noinline let A = A, s = sr, ipiv = ipiv
        () -> (copyto!(s, A); RF.lu!(s, ipiv, Val(true), Val(false)); GC.gc(false); s[1])
    end
    # faer (rust) — no Julia allocation
    sf = copy(A)
    ff = @noinline let A = A, s = sf
        () -> (copyto!(s, A); faer_lu!(s))
    end

    # correctness: reconstruct P*L*U ≈ A for the two Julia contenders
    let s = copy(A); F = lu!(s); @assert F.L * F.U ≈ A[F.p, :] rtol = 1e-9 "OpenBLAS lu wrong n=$n"; end
    let s = copy(A); F = RF.lu!(s, Vector{Int}(undef, n), Val(true), Val(false));
        @assert F.L * F.U ≈ A[F.p, :] rtol = 1e-9 "RecursiveFactorization lu wrong n=$n"; end
    @assert !isnan(faer_lu!(copy(A))) "faer lu NaN n=$n"

    # Julia contenders under controlled GC; faer (no Julia alloc) under normal GC
    fb(); fr()
    GC.enable(false); GC.gc(false)
    p_blas = run_probe("OpenBLAS", fb; seconds = 3.0)
    p_rf   = run_probe("RecursiveFactorization", fr; seconds = 3.0)
    GC.enable(true); GC.gc()
    p_faer = run_probe("faer", ff; seconds = 3.0)

    probes = Probe[p_blas, p_rf, p_faer]
    crate = "lu_rf_$(n)x$(n)"
    report(crate, probes; rust_label = "faer")
    save_probes(crate, probes)
    plot_probe(crate, probes)
    println()
    return probes
end

println("\n=== RecursiveFactorization vs OpenBLAS vs faer — LU ===\n")
for n in (64, 128, 256, 512)
    probe_size(n)
end
println("Done — probe_recursivefactorization.jl")
