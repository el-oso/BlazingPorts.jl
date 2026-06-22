@testitem "factorizations" tags = [:factorizations] begin
    # Sanity-checks the golden harness (Layer A verification):
    #  1. Each golden A is symmetric and positive-definite.
    #  2. The golden L reconstructs A: L*L' ≈ A (rtol 1e-9).
    #     This validates that faer's output is correct AND our hex parser is right.
    # The actual Julia Cholesky implementation (Layer B/C) is not yet present; these tests will be
    # extended in Layer B to also compare our L against the golden L.

    using LinearAlgebra: isposdef, issymmetric
    include(joinpath(@__DIR__, "golden.jl"))

    golden = load_cholesky_golden()
    @test !isempty(golden)

    expected_sizes = Set([1, 2, 3, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512])
    @test Set(keys(golden)) == expected_sizes

    for n in sort(collect(keys(golden)))
        A = golden[n].A
        L = golden[n].L

        # A must be square n×n
        @test size(A) == (n, n)
        @test size(L) == (n, n)

        # A must be symmetric (we wrote both upper and lower triangles from a_data)
        @test issymmetric(A)

        # A must be positive-definite
        @test isposdef(A)

        # L must be lower-triangular (upper triangle should be exactly 0.0)
        upper_ok = all(L[row, col] == 0.0 for col in 1:n for row in 1:(col-1))
        @test upper_ok

        # Golden L must reconstruct A: L * L' ≈ A
        LLt = L * L'
        @test isapprox(LLt, A; rtol=1e-9)
    end

    # ── Layer B: our base kernel must be BIT-EXACT vs faer's golden L for n ≤ 64 ──
    # (n ≤ recursion_threshold=64 → faer's active path is the simd_cholesky base case we ported.)
    using BlazingPorts.Factorizations: cholesky_llt!
    for n in sort(collect(keys(golden)))
        n <= 64 || continue
        A = copy(golden[n].A)
        Lg = golden[n].L
        @test cholesky_llt!(A) === true
        # compare lower triangles bit-for-bit (upper triangle of A is untouched / ignored)
        bitexact = all(
            reinterpret(UInt64, A[row, col]) === reinterpret(UInt64, Lg[row, col])
            for col in 1:n for row in col:n
        )
        @test bitexact
    end
end

@testitem "factorizations_strictmode" tags = [:factorizations] begin
    # StrictMode guarantees — the campaign's headline experiment. The full driver `cholesky_llt!` is
    # type-stable and allocation-free; the SIMD lives in the three pointer kernels (the wrapper itself
    # isn't where `<W x double>` is emitted), so `@assert_vectorized` is applied to each kernel.
    import BlazingPorts.Factorizations as F
    using BlazingPorts.Factorizations: cholesky_llt!
    using StrictMode, AllocCheck, JET
    using LinearAlgebra

    A = Matrix(let M = randn(96, 96); M'M + 96I end)
    cholesky_llt!(copy(A))  # warm
    @assert_typestable cholesky_llt!(A)
    @assert_noalloc cholesky_llt!(A)
    @test (@allocated cholesky_llt!(A)) == 0

    # Each hot kernel must vectorize to host-width <W x double> (base case, panel solve, rank-k update).
    GC.@preserve A begin
        p = pointer(A)
        ld = 96
        @assert_vectorized F._chol_base!(p, 64, ld)
        @assert_vectorized F._trsm_right_lower!(p, p, 32, 32, ld)
        @assert_vectorized F._syrk_lower!(p, p, 32, 32, ld)
    end
    @test true  # reaching here means all guarantees held (StrictViolation throws otherwise)
end
