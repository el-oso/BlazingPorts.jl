# Probe: matrixmultiply gemm ladder (Tier 1 — the headline probe).
# Contenders (all single-threaded, square Float64, C = A * B in-place):
#   1. OpenBLAS  — LinearAlgebra.mul!(C, A, B)
#   2. Octavian  — Octavian.matmul!(C, A, B)
#   3. @turbo    — pure-Julia 3-loop gemm with LoopVectorization.@turbo
#   4. rust      — matrixmultiply crate via mm_dgemm ccall
# Sizes: n = 32, 64, 128, 256 — each gets its own report().
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_matrixmultiply.jl
# Build Rust lib first: bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using LinearAlgebra
using Octavian
using LoopVectorization

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

# ── pure-Julia @turbo gemm (3-loop, column-major) ─────────────────────────────
function turbo_gemm!(C::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64})
    m, k = size(A)
    kB, n = size(B)
    @assert k == kB
    fill!(C, 0.0)
    @turbo for j in 1:n, p in 1:k, i in 1:m
        C[i, j] += A[i, p] * B[p, j]
    end
    return C
end

# ── Rust ccall wrapper ────────────────────────────────────────────────────────
function rust_dgemm!(C::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64})
    m, k = size(A)
    kB, n = size(B)
    @assert k == kB
    ccall((:mm_dgemm, LIB), Cvoid,
        (Csize_t, Csize_t, Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        m, k, n, pointer(A), pointer(B), pointer(C))
    return C
end

# ── run one size ──────────────────────────────────────────────────────────────
function probe_size(n::Int)
    A  = rand(n, n)
    B  = rand(n, n)
    Cref = A * B      # reference for correctness
    C_blas  = zeros(n, n)
    C_oct   = zeros(n, n)
    C_turbo = zeros(n, n)
    C_rust  = zeros(n, n)

    # warm-up + correctness
    mul!(C_blas, A, B)
    @assert C_blas ≈ Cref rtol=1e-10 "OpenBLAS disagrees at n=$n"
    Octavian.matmul!(C_oct, A, B)
    @assert C_oct  ≈ Cref rtol=1e-10 "Octavian disagrees at n=$n"
    turbo_gemm!(C_turbo, A, B)
    @assert C_turbo ≈ Cref rtol=1e-10 "@turbo disagrees at n=$n"
    rust_dgemm!(C_rust, A, B)
    @assert C_rust  ≈ Cref rtol=1e-10 "rust disagrees at n=$n"

    # wrap in @noinline closures over preallocated arrays — no alloc in timed region
    f_blas  = let A=A, B=B, C=C_blas;  @noinline () -> (mul!(C, A, B);           C[1]) end
    f_oct   = let A=A, B=B, C=C_oct;   @noinline () -> (Octavian.matmul!(C, A, B); C[1]) end
    f_turbo = let A=A, B=B, C=C_turbo; @noinline () -> (turbo_gemm!(C, A, B);    C[1]) end
    f_rust  = let A=A, B=B, C=C_rust;  @noinline () -> (rust_dgemm!(C, A, B);   C[1]) end

    probes = Probe[]
    for (label, f) in (("OpenBLAS", f_blas), ("Octavian", f_oct),
                       ("@turbo", f_turbo),  ("rust", f_rust))
        push!(probes, run_probe(label, f; seconds = 3.0))
    end
    report("matmul_$(n)x$(n)", probes; rust_label = "rust")
    save_probes("matmul_$(n)x$(n)", probes)
    plot_probe("matmul_$(n)x$(n)", probes)
    return probes
end

println("\n=== matrixmultiply gemm ladder ===")
for n in (32, 64, 128, 256)
    println()
    probe_size(n)
end

println("\nDone — probe_matrixmultiply.jl")
