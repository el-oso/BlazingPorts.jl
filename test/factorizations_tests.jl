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

@testitem "cholesky_padded" tags = [:factorizations] begin
    # cholesky_llt! auto-pads when the stride is a power of two (the fast path). The result must be
    # BIT-IDENTICAL to factoring the same data in place at a non-pow2 stride (leading dimension is pure
    # addressing — it never changes the FMA order or the bits). n=256 is pow2 (auto-pads); 257 is not.
    import BlazingPorts.Factorizations as F
    using BlazingPorts.Factorizations: cholesky_llt!
    using LinearAlgebra
    for n in (128, 256, 257)
        M = randn(n, n); A = M'M + n * I
        Aauto = Matrix(A)
        @test cholesky_llt!(Aauto) === true                 # pow2 → padded path; else in-place
        Pad = Matrix{Float64}(undef, n + 8, n); ref = view(Pad, 1:n, 1:n); copyto!(ref, A)
        @test F._chol_inplace!(ref) === true                # in-place at non-pow2 stride (n+8)
        bit = all(reinterpret(UInt64, Aauto[r, c]) === reinterpret(UInt64, ref[r, c]) for c in 1:n for r in c:n)
        @test bit
        @test isapprox(LowerTriangular(Aauto) * LowerTriangular(Aauto)', A; rtol = 1e-10)
    end
end

@testitem "qr_golden" tags = [:qr] begin
    # Layer D-A: validates the QR golden harness WITHOUT any Julia QR implementation.
    # Reconstruction: apply Householder reflectors to R and verify Q*R ≈ A.
    #
    # faer's convention: H_k = I − v_k v_kᵀ / τ_k  (divides by τ_k, NOT multiplies).
    # τ_k lives at T[k%block_size + 1, k] (1-based; diagonal of each bs×bs T-block).
    # τ_k = Inf ⟹ H_k = I (trivial reflector, skip).
    # v_k: leading 1 is implicit (not stored); essential part = QR[k+1:n, k].
    # Q = H_1 H_2 ⋯ H_n;  we build Q by applying each H_k to the identity from the left.

    using LinearAlgebra: triu, I
    include(joinpath(@__DIR__, "golden.jl"))

    golden = load_qr_golden()
    @test !isempty(golden)

    expected_sizes = Set([1, 2, 3, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512])
    @test Set(keys(golden)) == expected_sizes

    for n in sort(collect(keys(golden)))
        A  = golden[n].A
        QR = golden[n].QR
        T  = golden[n].T
        bs = golden[n].block_size

        @test size(A)  == (n, n)
        @test size(QR) == (n, n)
        @test size(T)  == (bs, n)

        # Extract R: upper triangle of QR (including diagonal).
        R = triu(QR)

        # Reconstruct Q = H_1 H_2 ⋯ H_n by accumulating into an n×n identity.
        # Each H_k: Q ← Q * H_k  (applying from the right to the accumulated rows of Q)
        # Equivalently, build Q column by column: start with I, apply H_k from the left.
        Q = Matrix{Float64}(I, n, n)
        for k in 1:n
            # τ_k is on the diagonal of the k-th T-block: row = (k-1) % bs, col = k (1-based).
            τ = T[(k - 1) % bs + 1, k]
            isinf(τ) && continue   # identity reflector — skip

            # Build v_k: [1; essential part from QR below diagonal].
            v = Vector{Float64}(undef, n - k + 1)
            v[1] = 1.0
            for i in 2:(n - k + 1)
                v[i] = QR[k + i - 1, k]
            end

            # Apply H_k from the right to Q: Q ← Q * H_k = Q − (Q v) vᵀ / τ
            # (Q v) is an n-vector; outer product with vᵀ gives n×(n-k+1) update.
            Qv = Q[:, k:n] * v          # n-vector: Q[:,k:n]*v
            Q[:, k:n] .-= (Qv * v') ./ τ   # n×(n-k+1) rank-1 update
        end

        # Q * R must reproduce A (rtol 1e-9).
        recon = Q * R
        ok = isapprox(recon, A; rtol=1e-9)
        @test ok
        if !ok
            err = maximum(abs, recon .- A) / maximum(abs, A)
            @warn "QR reconstruction failed n=$n relerr=$err"
        end
    end
end

@testitem "qr_unblocked" tags = [:qr] begin
    # Layer D-B: our unpivoted Householder QR (qr_unblocked!) must reconstruct A and match faer's QR
    # factors numerically (R/v are mathematically unique, so our unblocked output ≈ faer's, even where
    # faer used its blocked path; not bit-exact due to norm_l2 ordering).
    using BlazingPorts.Factorizations: qr_unblocked!
    using LinearAlgebra: triu, I
    include(joinpath(@__DIR__, "golden.jl"))
    golden = load_qr_golden()

    reconQR(QR, tau, n) = begin
        R = triu(QR); Q = Matrix{Float64}(I, n, n)
        for k in 1:n
            t = tau[k]; isinf(t) && continue
            v = [i == 1 ? 1.0 : QR[k+i-1, k] for i in 1:(n - k + 1)]
            Q[:, k:n] .-= (Q[:, k:n] * v) * v' ./ t
        end
        Q * R
    end

    for n in sort(collect(keys(golden)))
        A0 = golden[n].A
        A = copy(A0); tau = zeros(n)
        @test qr_unblocked!(A, tau) === true
        @test isapprox(reconQR(A, tau, n), A0; rtol = 1e-9)        # Q*R ≈ A
        @test isapprox(A, golden[n].QR; rtol = 1e-10, atol = 1e-12) # matches faer's packed factor

        # blocked driver: same packed factor + reconstruction (compact-WY, gemm trailing update)
        Ab = copy(A0); taub = zeros(n)
        @test BlazingPorts.Factorizations.qr_blocked!(Ab, taub; nb = 16) === true
        @test isapprox(reconQR(Ab, taub, n), A0; rtol = 1e-9)
        @test isapprox(Ab, golden[n].QR; rtol = 1e-10, atol = 1e-12)
    end
end

@testitem "qr_2level" tags = [:qr] begin
    # The n≥1536 two-level fat-panel fast path (with QRWorkspace + LDA padding). Exercises non-pow2
    # (1600, in place) and pow2 (2048, padded scratch). Validate by reconstruction Q·R ≈ A.
    using BlazingPorts.Factorizations: qr_blocked!, QRWorkspace
    using LinearAlgebra
    reconQR(A, tau, n) = begin
        R = triu(A); Q = Matrix{Float64}(I, n, n)
        for k in 1:n
            t = tau[k]; isinf(t) && continue
            v = [i == 1 ? 1.0 : A[k+i-1, k] for i in 1:(n - k + 1)]
            Q[:, k:n] .-= (Q[:, k:n] * v) * v' ./ t
        end
        Q * R
    end
    for n in (1600, 2048)
        A0 = randn(n, n); A = copy(A0); tau = zeros(n)
        @test qr_blocked!(A, tau) === true                      # auto-selects two-level (n≥1536)
        @test isapprox(reconQR(A, tau, n), A0; rtol = 1e-9)
        # workspace method gives the same factorization
        Aw = copy(A0); tw = zeros(n)
        @test qr_blocked!(Aw, tw, QRWorkspace(n)) === true
        @test isapprox(reconQR(Aw, tw, n), A0; rtol = 1e-9)
    end
end

@testitem "qr_strictmode" tags = [:qr] begin
    # The QR base kernel must be vectorized, allocation-free, type-stable.
    using BlazingPorts.Factorizations: qr_unblocked!
    using StrictMode, AllocCheck, JET
    using LinearAlgebra
    A = randn(96, 96); tau = zeros(96)
    qr_unblocked!(copy(A), tau)  # warm
    @assert_typestable qr_unblocked!(A, tau)
    @assert_noalloc qr_unblocked!(A, tau)
    @assert_vectorized qr_unblocked!(A, tau)
    @test (@allocated qr_unblocked!(A, tau)) == 0
end

@testitem "factorizations_strictmode" tags = [:factorizations] begin
    # StrictMode guarantees — the campaign's headline experiment. The full driver `cholesky_llt!` is
    # type-stable and allocation-free; the SIMD lives in the three pointer kernels (the wrapper itself
    # isn't where `<W x double>` is emitted), so `@assert_vectorized` is applied to each kernel.
    import BlazingPorts.Factorizations as F
    using BlazingPorts.Factorizations: cholesky_llt!
    using StrictMode, AllocCheck, JET
    using LinearAlgebra

    import BlazingPorts.Factorizations as F
    using BlazingPorts.Factorizations: CholWorkspace
    # With a preallocated workspace the padded fast path (pow2 stride) is allocation-free — the headline
    # guarantee on the entry users call in a hot loop. (@allocated factors in place, so use fresh copies.)
    A = Matrix(let M = randn(256, 256); M'M + 256I end)   # n=256 ⇒ pow2 stride ⇒ padded path
    ws = CholWorkspace(256)
    cholesky_llt!(copy(A), ws); F._chol_inplace!(copy(A))  # warm both entries (compile before @allocated)
    @assert_typestable cholesky_llt!(A, ws)               # AllocCheck/JET are static — A state irrelevant
    @assert_noalloc cholesky_llt!(A, ws)
    B = copy(A); C = copy(A)                              # fresh SPD (each measured call factors in place)
    @test (@allocated cholesky_llt!(B, ws)) == 0
    @test (@allocated F._chol_inplace!(C)) == 0          # in-place core likewise alloc-free

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
