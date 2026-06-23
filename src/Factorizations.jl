"""
    Factorizations

Faithful pure-Julia port of faer's Cholesky factorization, developed in layers:

  - **Layer B** (next): unblocked SIMD base kernel — a left-looking panel-Cholesky that operates on
    a W×W tile (W = `VectorizationBase.pick_vector_width(Float64)`, e.g. 4 on AVX2, 8 on AVX-512).
    Gated by `@assert_vectorized` / `@assert_noalloc` from StrictMode.jl (test/bench only; this
    source carries no StrictMode dep).

  - **Layer C** (after B): blocked **right-looking recursive** driver — faer's active path is
    `cholesky_recursion_right_looking` (the left-looking block code in `cholesky_block_left_looking` is
    dead behind `if true || …`). Splits into `block_size = min(n.next_power_of_two()/2, 128)` blocks,
    recurses on the diagonal block down to `recursion_threshold = 64` (→ this base kernel), triangular-
    solves the off-diagonal panel, then trailing `A11 -= A10·A10ᵀ`. Target: parity with faer at n≥256.

The golden harness lives in `bench/rust_compare/cholesky_golden.txt` (regenerate:
`cd bench/rust_compare/rust && cargo build --release --bin cholesky_verify &&
 ./target/release/cholesky_verify > ../cholesky_golden.txt`).
Tests compare our L against the golden L bit-for-bit (or within 1 ULP for rounding-order differences).

See also: `RESULTS.md` §"The faer finding", `CLAUDE.md`.
"""
module Factorizations

using SIMD: Vec, vload, vstore
import VectorizationBase

export cholesky_llt!, CholWorkspace

# Host SIMD width: 4 on AVX2, 8 on AVX-512 — one source, ISA-generic. (Bit-exactness is independent
# of W: each lane is an independent FMA accumulation, so changing W never changes the result bits.)
const W = Int(VectorizationBase.pick_vector_width(Float64))

# faer's auto block params (LdltParams::auto): right-looking recursion threshold + block size.
const RECURSION_THRESHOLD = 64
const BLOCK_SIZE = 128

# All kernels operate on column-major blocks addressed by a base pointer to the (1,1) element and a
# leading dimension `ld` (the parent matrix's row count), so sub-blocks are plain pointer offsets.
@inline _lidx(i, k, ld) = (k - 1) * ld + i                 # 1-based linear index of block[i,k]
@inline _vptr(p, i, k, ld) = p + (((k - 1) * ld + (i - 1)) * sizeof(Float64))  # Ptr to block[i,k]

# ── base case: faithful port of faer `simd_cholesky` (LLᵀ, left-looking, ascending-k FMA, scale by
#    reciprocal of sqrt(diag)). Bit-exact vs faer for the n ≤ recursion_threshold base case. ──
function _chol_base!(p::Ptr{Float64}, n::Int, ld::Int)
    @inbounds for j in 1:n
        i = j
        while i + W - 1 <= n
            base = _vptr(p, i, j, ld)
            acc = vload(Vec{W,Float64}, base)
            for k in 1:j-1
                njk = -unsafe_load(p, _lidx(j, k, ld))
                acc = muladd(Vec{W,Float64}(njk), vload(Vec{W,Float64}, _vptr(p, i, k, ld)), acc)
            end
            vstore(acc, base)
            i += W
        end
        while i <= n
            s = unsafe_load(p, _lidx(i, j, ld))
            for k in 1:j-1
                s = muladd(-unsafe_load(p, _lidx(j, k, ld)), unsafe_load(p, _lidx(i, k, ld)), s)
            end
            unsafe_store!(p, s, _lidx(i, j, ld))
            i += 1
        end
        d = unsafe_load(p, _lidx(j, j, ld))
        (d > 0.0) || return false
        inv = 1.0 / sqrt(d)
        vinv = Vec{W,Float64}(inv)
        i = j
        while i + W - 1 <= n
            base = _vptr(p, i, j, ld)
            vstore(vload(Vec{W,Float64}, base) * vinv, base)
            i += W
        end
        while i <= n
            unsafe_store!(p, unsafe_load(p, _lidx(i, j, ld)) * inv, _lidx(i, j, ld))
            i += 1
        end
    end
    return true
end

# ── panel solve: L10 (m×bs) from L10·L00ᵀ = A10, L00 lower (bs×bs). In place on the A10 block. ──
# Column c: L10[:,c] = (A10[:,c] − Σ_{k<c} L00[c,k]·L10[:,k]) · (1/L00[c,c]). Register-blocked in NB-column
# panels: contributions from already-solved columns (k < c0) are applied as a register-blocked gemm (each
# A10[i,k] load reused across NB output columns), then a tiny within-panel triangular solve handles the
# k∈[c0,c) coupling. The split keeps ascending-k FMA order with exact intermediate store/reload, so the
# result is **bit-identical** to the unblocked solve. Vectorized over rows.
const NB_TRSM = 4

@inline function _trsm_gemm_col!(p00, p10, cc::Int, c0::Int, m::Int, ld::Int)
    i = 1                                  # A10[:,cc] -= Σ_{k<c0} L00[cc,k]·A10[:,k]
    @inbounds while i + W - 1 <= m
        o = _vptr(p10, i, cc, ld); a = vload(Vec{W,Float64}, o)
        for k in 1:c0-1
            a = muladd(Vec{W,Float64}(-unsafe_load(p00, _lidx(cc, k, ld))),
                vload(Vec{W,Float64}, _vptr(p10, i, k, ld)), a)
        end
        vstore(a, o); i += W
    end
    @inbounds while i <= m
        s = unsafe_load(p10, _lidx(i, cc, ld))
        for k in 1:c0-1
            s = muladd(-unsafe_load(p00, _lidx(cc, k, ld)), unsafe_load(p10, _lidx(i, k, ld)), s)
        end
        unsafe_store!(p10, s, _lidx(i, cc, ld)); i += 1
    end
end

function _trsm_right_lower!(p00::Ptr{Float64}, p10::Ptr{Float64}, bs::Int, m::Int, ld::Int)
    c0 = 1
    @inbounds while c0 <= bs
        nb = min(NB_TRSM, bs - c0 + 1)
        # (1) gemm update of the panel by columns k < c0
        if c0 > 1
            if nb == NB_TRSM
                i = 1
                while i + W - 1 <= m
                    o0 = _vptr(p10, i, c0, ld);     a0 = vload(Vec{W,Float64}, o0)
                    o1 = _vptr(p10, i, c0 + 1, ld); a1 = vload(Vec{W,Float64}, o1)
                    o2 = _vptr(p10, i, c0 + 2, ld); a2 = vload(Vec{W,Float64}, o2)
                    o3 = _vptr(p10, i, c0 + 3, ld); a3 = vload(Vec{W,Float64}, o3)
                    for k in 1:c0-1
                        vk = vload(Vec{W,Float64}, _vptr(p10, i, k, ld))   # reused across 4 output cols
                        a0 = muladd(Vec{W,Float64}(-unsafe_load(p00, _lidx(c0, k, ld))), vk, a0)
                        a1 = muladd(Vec{W,Float64}(-unsafe_load(p00, _lidx(c0 + 1, k, ld))), vk, a1)
                        a2 = muladd(Vec{W,Float64}(-unsafe_load(p00, _lidx(c0 + 2, k, ld))), vk, a2)
                        a3 = muladd(Vec{W,Float64}(-unsafe_load(p00, _lidx(c0 + 3, k, ld))), vk, a3)
                    end
                    vstore(a0, o0); vstore(a1, o1); vstore(a2, o2); vstore(a3, o3)
                    i += W
                end
                while i <= m
                    for dj in 0:NB_TRSM-1
                        cc = c0 + dj; s = unsafe_load(p10, _lidx(i, cc, ld))
                        for k in 1:c0-1
                            s = muladd(-unsafe_load(p00, _lidx(cc, k, ld)), unsafe_load(p10, _lidx(i, k, ld)), s)
                        end
                        unsafe_store!(p10, s, _lidx(i, cc, ld))
                    end
                    i += 1
                end
            else
                for dj in 0:nb-1
                    _trsm_gemm_col!(p00, p10, c0 + dj, c0, m, ld)
                end
            end
        end
        # (2) within-panel triangular solve (k ∈ [c0, c)) + scale
        for dj in 0:nb-1
            c = c0 + dj
            invc = 1.0 / unsafe_load(p00, _lidx(c, c, ld))
            vinv = Vec{W,Float64}(invc)
            i = 1
            while i + W - 1 <= m
                o = _vptr(p10, i, c, ld); a = vload(Vec{W,Float64}, o)
                for k in c0:c-1
                    a = muladd(Vec{W,Float64}(-unsafe_load(p00, _lidx(c, k, ld))),
                        vload(Vec{W,Float64}, _vptr(p10, i, k, ld)), a)
                end
                vstore(a * vinv, o); i += W
            end
            while i <= m
                s = unsafe_load(p10, _lidx(i, c, ld))
                for k in c0:c-1
                    s = muladd(-unsafe_load(p00, _lidx(c, k, ld)), unsafe_load(p10, _lidx(i, k, ld)), s)
                end
                unsafe_store!(p10, s * invc, _lidx(i, c, ld)); i += 1
            end
        end
        c0 += NB_TRSM
    end
    return nothing
end

# ── trailing symmetric rank-bs update: A11 (m×m) −= L10·L10ᵀ. Register-blocked NC columns × W rows:
#    each loaded L10[i,c] vector is reused across NC column accumulators (turns the memory-bound naive
#    version into a compute-bound microkernel). Computes the full m×m (upper is overwritten but never
#    read — the factorization only touches the lower triangle); per-lower-element result is unchanged. ──
const NC = 4

@inline function _syrk_panel!(p11, p10, j::Int, m::Int, bs::Int, ld::Int)
    # one column j of A11 (lower, W-aligned start): A11[i,j] -= Σ_c L10[j,c]·L10[i,c]
    i = ((j - 1) ÷ W) * W + 1
    @inbounds while i + W - 1 <= m
        b = _vptr(p11, i, j, ld); a = vload(Vec{W,Float64}, b)
        for c in 1:bs
            a = muladd(Vec{W,Float64}(-unsafe_load(p10, _lidx(j, c, ld))),
                vload(Vec{W,Float64}, _vptr(p10, i, c, ld)), a)
        end
        vstore(a, b); i += W
    end
    @inbounds while i <= m
        s = unsafe_load(p11, _lidx(i, j, ld))
        for c in 1:bs
            s = muladd(-unsafe_load(p10, _lidx(j, c, ld)), unsafe_load(p10, _lidx(i, c, ld)), s)
        end
        unsafe_store!(p11, s, _lidx(i, j, ld)); i += 1
    end
end

function _syrk_lower!(p11::Ptr{Float64}, p10::Ptr{Float64}, m::Int, bs::Int, ld::Int)
    j = 1
    @inbounds while j + NC - 1 <= m
        # ALIGNED-TRIANGULAR: skip the fully-upper row-blocks above this column block, but start at the
        # W-aligned grid point ≤ j so the vector tiles stay aligned (the naive `i=j` triangular regressed
        # on misalignment). Recovers ~half the flops; the <W upper sliver in [istart, j) is computed into
        # never-read memory (harmless).
        i = ((j - 1) ÷ W) * W + 1
        while i + 3W - 1 <= m   # MR=3 × NC=4 = 12 accumulators (reuse 3 row-vector loads across 4 cols)
            r1 = i + W; r2 = i + 2W
            e00 = _vptr(p11, i, j, ld);      A00 = vload(Vec{W,Float64}, e00)
            e10 = _vptr(p11, r1, j, ld);     C00 = vload(Vec{W,Float64}, e10)
            e20 = _vptr(p11, r2, j, ld);     D00 = vload(Vec{W,Float64}, e20)
            e01 = _vptr(p11, i, j + 1, ld);  A01 = vload(Vec{W,Float64}, e01)
            e11 = _vptr(p11, r1, j + 1, ld); C01 = vload(Vec{W,Float64}, e11)
            e21 = _vptr(p11, r2, j + 1, ld); D01 = vload(Vec{W,Float64}, e21)
            e02 = _vptr(p11, i, j + 2, ld);  A02 = vload(Vec{W,Float64}, e02)
            e12 = _vptr(p11, r1, j + 2, ld); C02 = vload(Vec{W,Float64}, e12)
            e22 = _vptr(p11, r2, j + 2, ld); D02 = vload(Vec{W,Float64}, e22)
            e03 = _vptr(p11, i, j + 3, ld);  A03 = vload(Vec{W,Float64}, e03)
            e13 = _vptr(p11, r1, j + 3, ld); C03 = vload(Vec{W,Float64}, e13)
            e23 = _vptr(p11, r2, j + 3, ld); D03 = vload(Vec{W,Float64}, e23)
            for c in 1:bs
                v0 = vload(Vec{W,Float64}, _vptr(p10, i, c, ld))
                v1 = vload(Vec{W,Float64}, _vptr(p10, r1, c, ld))
                v2 = vload(Vec{W,Float64}, _vptr(p10, r2, c, ld))
                g0 = Vec{W,Float64}(-unsafe_load(p10, _lidx(j, c, ld)));     A00 = muladd(g0, v0, A00); C00 = muladd(g0, v1, C00); D00 = muladd(g0, v2, D00)
                g1 = Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 1, c, ld))); A01 = muladd(g1, v0, A01); C01 = muladd(g1, v1, C01); D01 = muladd(g1, v2, D01)
                g2 = Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 2, c, ld))); A02 = muladd(g2, v0, A02); C02 = muladd(g2, v1, C02); D02 = muladd(g2, v2, D02)
                g3 = Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 3, c, ld))); A03 = muladd(g3, v0, A03); C03 = muladd(g3, v1, C03); D03 = muladd(g3, v2, D03)
            end
            vstore(A00, e00); vstore(A01, e01); vstore(A02, e02); vstore(A03, e03)
            vstore(C00, e10); vstore(C01, e11); vstore(C02, e12); vstore(C03, e13)
            vstore(D00, e20); vstore(D01, e21); vstore(D02, e22); vstore(D03, e23)
            i += 3W
        end
        while i + 2W - 1 <= m   # MR=2 × NC=4 = 8 accumulators (reuse 2 row-vector loads across 4 cols)
            r1 = i + W
            d00 = _vptr(p11, i, j, ld);     A00 = vload(Vec{W,Float64}, d00)
            d10 = _vptr(p11, r1, j, ld);    B00 = vload(Vec{W,Float64}, d10)
            d01 = _vptr(p11, i, j + 1, ld); A01 = vload(Vec{W,Float64}, d01)
            d11 = _vptr(p11, r1, j + 1, ld); B01 = vload(Vec{W,Float64}, d11)
            d02 = _vptr(p11, i, j + 2, ld); A02 = vload(Vec{W,Float64}, d02)
            d12 = _vptr(p11, r1, j + 2, ld); B02 = vload(Vec{W,Float64}, d12)
            d03 = _vptr(p11, i, j + 3, ld); A03 = vload(Vec{W,Float64}, d03)
            d13 = _vptr(p11, r1, j + 3, ld); B03 = vload(Vec{W,Float64}, d13)
            for c in 1:bs
                v0 = vload(Vec{W,Float64}, _vptr(p10, i, c, ld))
                v1 = vload(Vec{W,Float64}, _vptr(p10, r1, c, ld))
                g0 = Vec{W,Float64}(-unsafe_load(p10, _lidx(j, c, ld)));     A00 = muladd(g0, v0, A00); B00 = muladd(g0, v1, B00)
                g1 = Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 1, c, ld))); A01 = muladd(g1, v0, A01); B01 = muladd(g1, v1, B01)
                g2 = Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 2, c, ld))); A02 = muladd(g2, v0, A02); B02 = muladd(g2, v1, B02)
                g3 = Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 3, c, ld))); A03 = muladd(g3, v0, A03); B03 = muladd(g3, v1, B03)
            end
            vstore(A00, d00); vstore(A01, d01); vstore(A02, d02); vstore(A03, d03)
            vstore(B00, d10); vstore(B01, d11); vstore(B02, d12); vstore(B03, d13)
            i += 2W
        end
        while i + W - 1 <= m
            b0 = _vptr(p11, i, j, ld);     a0 = vload(Vec{W,Float64}, b0)
            b1 = _vptr(p11, i, j + 1, ld); a1 = vload(Vec{W,Float64}, b1)
            b2 = _vptr(p11, i, j + 2, ld); a2 = vload(Vec{W,Float64}, b2)
            b3 = _vptr(p11, i, j + 3, ld); a3 = vload(Vec{W,Float64}, b3)
            for c in 1:bs
                lic = vload(Vec{W,Float64}, _vptr(p10, i, c, ld))   # reused across the 4 columns
                a0 = muladd(Vec{W,Float64}(-unsafe_load(p10, _lidx(j, c, ld))), lic, a0)
                a1 = muladd(Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 1, c, ld))), lic, a1)
                a2 = muladd(Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 2, c, ld))), lic, a2)
                a3 = muladd(Vec{W,Float64}(-unsafe_load(p10, _lidx(j + 3, c, ld))), lic, a3)
            end
            vstore(a0, b0); vstore(a1, b1); vstore(a2, b2); vstore(a3, b3)
            i += W
        end
        while i <= m  # row tail for the 4-column block
            for dj in 0:NC-1
                s = unsafe_load(p11, _lidx(i, j + dj, ld))
                for c in 1:bs
                    s = muladd(-unsafe_load(p10, _lidx(j + dj, c, ld)), unsafe_load(p10, _lidx(i, c, ld)), s)
                end
                unsafe_store!(p11, s, _lidx(i, j + dj, ld))
            end
            i += 1
        end
        j += NC
    end
    while j <= m   # remaining (<NC) columns
        _syrk_panel!(p11, p10, j, m, bs, ld)
        j += 1
    end
    return nothing
end

# ── right-looking recursive driver: faer `cholesky_recursion_right_looking`. ──
function _chol_rl!(p::Ptr{Float64}, n::Int, ld::Int, block_size::Int, threshold::Int)
    n <= threshold && return _chol_base!(p, n, ld)
    bs_outer = min(nextpow(2, n) ÷ 2, block_size)
    j = 0                                          # 0-based offset into this block
    while j < n
        bs = min(bs_outer, n - j)
        p00 = _vptr(p, j + 1, j + 1, ld)
        _chol_rl!(p00, bs, ld, block_size, threshold) || return false
        m = n - j - bs
        if m > 0
            p10 = _vptr(p, j + bs + 1, j + 1, ld)
            p11 = _vptr(p, j + bs + 1, j + bs + 1, ld)
            _trsm_right_lower!(p00, p10, bs, m, ld)
            _syrk_lower!(p11, p10, m, bs, ld)
        end
        j += bs
    end
    return true
end

"""
    cholesky_llt!(A::AbstractMatrix{Float64}) -> Bool

In-place real LLᵀ Cholesky — faithful port of faer 0.24.1 `cholesky_recursion_right_looking`.
Overwrites the **lower triangle** of `A` with `L`; the upper triangle is untouched. Returns `false`
(A left partially factored) on a non-positive pivot.

Right-looking recursive: factor the `block_size` diagonal block (recursing to the `simd_cholesky` base
kernel at n ≤ `RECURSION_THRESHOLD`), triangular-solve the panel below, then `A11 −= L10·L10ᵀ`. The
base kernel is bit-exact vs faer (n ≤ 64); at the blocked level the trsm/syrk accumulate in ascending
order with fused multiply-adds (faer's microkernel ordering differs, so ≥128 may differ by a few ULP
in the accumulation — reconstruction stays ~1e-13). Vectorized over rows with host-width `Vec{W}`.
"""
# in-place core (no-alloc): factor A using its own leading dimension.
@inline function _chol_inplace!(A::AbstractMatrix{Float64})
    n = size(A, 1)
    GC.@preserve A begin
        return _chol_rl!(pointer(A), n, stride(A, 2), BLOCK_SIZE, RECURSION_THRESHOLD)
    end
end

"""
    CholWorkspace(n)

Reusable scratch for [`cholesky_llt!`](@ref) sized for an `n×n` matrix. Preallocate once and pass it in
to make the factorization **allocation-free even on the padded fast path** (the path taken when the input
stride is a power of two). Size it for your largest `n`.
"""
struct CholWorkspace
    P::Vector{Float64}     # padded scratch, holds an (n+8)×n column-major block
end
CholWorkspace(n::Integer) = CholWorkspace(Vector{Float64}(undef, (n + 8) * n))

@inline _needs_pad(A) = size(A, 1) >= 128 && ispow2(stride(A, 2))

"""
    cholesky_llt!(A [, ws::CholWorkspace]) -> true

In-place Cholesky (LLᵀ); the lower triangle of `A` is overwritten with `L`. Returns `false` on a
non-positive pivot. **One entry, always the fast path:** a power-of-two leading dimension aliases columns
into the same cache sets (the classic `LDA=2^k` conflict, ~1.3–1.5× slower trsm/syrk at n≥512), so when
`A`'s stride is a power of two the factorization runs in a padded scratch (`ld+8`) and is copied back —
bit-identical, since `ld` is pure addressing (faer/BLAS pad for the same reason). Otherwise it factors in
place. Pass a preallocated [`CholWorkspace`](@ref) to make even the padded path allocation-free (the
no-alloc guarantee); the no-argument form allocates the scratch on demand.
"""
function cholesky_llt!(A::AbstractMatrix{Float64}, ws::CholWorkspace)
    n = size(A, 1)
    n == 0 && return true
    Base.require_one_based_indexing(A)
    _needs_pad(A) || return _chol_inplace!(A)
    ldp = n + 8
    ld = stride(A, 2)
    GC.@preserve A ws begin
        pA = pointer(A); pP = pointer(ws.P)
        @inbounds for j in 0:n-1                          # A (ld) → padded scratch (ldp), column by column
            unsafe_copyto!(pP + j * ldp * 8, pA + j * ld * 8, n)
        end
        ok = _chol_rl!(pP, n, ldp, BLOCK_SIZE, RECURSION_THRESHOLD)
        @inbounds for j in 0:n-1                          # result back
            unsafe_copyto!(pA + j * ld * 8, pP + j * ldp * 8, n)
        end
        return ok
    end
end

function cholesky_llt!(A::AbstractMatrix{Float64})
    n = size(A, 1)
    n == 0 && return true
    Base.require_one_based_indexing(A)
    _needs_pad(A) || return _chol_inplace!(A)
    return cholesky_llt!(A, CholWorkspace(n))            # convenience: allocate scratch on demand
end


# ── QR Factorization — Layer D (planned) ────────────────────────────────────────────────────────
#
# Faithful pure-Julia port of faer 0.24.1 Householder QR (no pivoting).
# Convention: H_k = I − v_k v_kᵀ / τ_k  (divides by τ, matching faer's `make_householder_in_place`).
# τ_k = Inf encodes a trivial (identity) reflector; v_k has implicit leading 1 at index k.
#
# Layer D-B (done): qr_unblocked!  — faithful port of faer `qr_in_place_unblocked` + `make_householder_in_place`.
# Layer D-C (done): qr_blocked!    — compact-WY blocked driver (panel reduction + gemm trailing update).
# Golden: bench/rust_compare/qr_golden.txt (regen: cargo build --release --bin qr_verify > ../qr_golden.txt).

export qr_unblocked!, qr_blocked!

"""
    qr_unblocked!(A, tau) -> true

In-place unpivoted Householder QR — faithful port of faer 0.24.1 `qr_in_place_unblocked` +
`make_householder_in_place` (real f64). On output the upper triangle of `A` (incl. diagonal) is `R`, the
essential Householder vectors `v_k` (with implicit `v_k[k]=1`) are stored below the diagonal of column k,
and `tau[k]` holds the coefficient (faer convention `H_k = I − v_k v_kᵀ / tau_k`; `tau=Inf` ⇒ identity).
Per-element arithmetic mirrors faer (ascending-i FMA dot/update; `hypot` norm; `sign(head)`; scale by
`1/(head+signed_norm)`); the only non-bit-exact piece vs the golden is `‖tail‖₂` (faer's `norm_l2` may
rescale) — reconstruction stays ~1e-13. Vectorized over rows with `Vec{W}`.
"""
function qr_unblocked!(A::AbstractMatrix{Float64}, tau::AbstractVector{Float64})
    m, n = size(A)
    ld = stride(A, 2)
    GC.@preserve A begin
        p = pointer(A)
        @inbounds for col in 1:min(m, n)
            row = col
            # ── reflector from x = A[row:m, col] ──
            tn = 0.0
            i = row + 1
            while i + W - 1 <= m                      # ‖tail‖² (vectorized)
                x = vload(Vec{W,Float64}, _vptr(p, i, col, ld))
                tn += sum(x * x)
                i += W
            end
            while i <= m
                x = unsafe_load(p, _lidx(i, col, ld)); tn = muladd(x, x, tn); i += 1
            end
            tail_norm = sqrt(tn)
            head = unsafe_load(p, _lidx(row, col, ld))
            head_norm = abs(head)
            if tail_norm < floatmin(Float64)
                tau[col] = Inf                          # trivial reflector (identity)
                continue
            end
            nrm = hypot(head_norm, tail_norm)
            signed_norm = (head >= 0.0 ? nrm : -nrm)    # sign(head)·norm (faer: +norm if head==0)
            hwb = head + signed_norm
            hwb_inv = 1.0 / hwb
            vinv = Vec{W,Float64}(hwb_inv)
            i = row + 1                                  # v_essential = tail · (1/hwb)
            while i + W - 1 <= m
                b = _vptr(p, i, col, ld); vstore(vload(Vec{W,Float64}, b) * vinv, b); i += W
            end
            while i <= m
                unsafe_store!(p, unsafe_load(p, _lidx(i, col, ld)) * hwb_inv, _lidx(i, col, ld)); i += 1
            end
            unsafe_store!(p, -signed_norm, _lidx(row, col, ld))   # R diagonal = β
            t = 0.5 * (1.0 + (tail_norm * abs(hwb_inv))^2)
            tau[col] = t
            tinv = 1.0 / t
            # ── apply H_col to trailing columns j: x_j -= (vᵀx_j / tau)·v ──
            for j in col+1:n
                acc = Vec{W,Float64}(0.0)
                dscal = unsafe_load(p, _lidx(row, j, ld))        # v_row=1 contribution
                i = row + 1
                while i + W - 1 <= m
                    acc = muladd(vload(Vec{W,Float64}, _vptr(p, i, col, ld)),
                        vload(Vec{W,Float64}, _vptr(p, i, j, ld)), acc)
                    i += W
                end
                dot = dscal + sum(acc)
                while i <= m
                    dot = muladd(unsafe_load(p, _lidx(i, col, ld)), unsafe_load(p, _lidx(i, j, ld)), dot); i += 1
                end
                k = -dot * tinv
                unsafe_store!(p, unsafe_load(p, _lidx(row, j, ld)) + k, _lidx(row, j, ld))
                vk = Vec{W,Float64}(k)
                i = row + 1
                while i + W - 1 <= m
                    bj = _vptr(p, i, j, ld)
                    vstore(muladd(vk, vload(Vec{W,Float64}, _vptr(p, i, col, ld)), vload(Vec{W,Float64}, bj)), bj)
                    i += W
                end
                while i <= m
                    unsafe_store!(p, muladd(k, unsafe_load(p, _lidx(i, col, ld)), unsafe_load(p, _lidx(i, j, ld))), _lidx(i, j, ld)); i += 1
                end
            end
        end
    end
    return true
end

# dlarfb gemm 1: Wm[c,j] = Σ_i V[i,c]·C[i,j]  (V: mp×pb ld=mp; C: mp×nt ld=ldC; Wm: pb×nt ld=pb).
# Dot-form: Vec-accumulate over contiguous rows i, reuse each V[:,c] chunk across NJ output columns,
# horizontal-sum at the end.
# 1-column dot helper (V[:,c]ᵀ C[:,j] for j in a 4-block or singly) — used for c-remainder.
@inline function _qr_VtC_col!(pWm, pV, pC, c::Int, mp::Int, nt::Int, pb::Int, ldC::Int)
    j = 1
    @inbounds while j <= nt
        s = Vec{W,Float64}(0.0); t = 0.0; i = 1
        while i + W - 1 <= mp
            s = muladd(vload(Vec{W,Float64}, _vptr(pV, i, c, mp)),
                vload(Vec{W,Float64}, _vptr(pC, i, j, ldC)), s)
            i += W
        end
        while i <= mp
            t = muladd(unsafe_load(pV, _lidx(i, c, mp)), unsafe_load(pC, _lidx(i, j, ldC)), t); i += 1
        end
        unsafe_store!(pWm, sum(s) + t, _lidx(c, j, pb))
        j += 1
    end
end

function _qr_VtC!(pWm, pV, pC, mp::Int, nt::Int, pb::Int, ldC::Int)
    c = 1
    @inbounds while c + 1 <= pb                      # 2 c-rows × 4 j-cols = 8 dot accumulators
        j = 1
        while j + 3 <= nt
            a0 = Vec{W,Float64}(0.0); a1 = Vec{W,Float64}(0.0); a2 = Vec{W,Float64}(0.0); a3 = Vec{W,Float64}(0.0)
            b0 = Vec{W,Float64}(0.0); b1 = Vec{W,Float64}(0.0); b2 = Vec{W,Float64}(0.0); b3 = Vec{W,Float64}(0.0)
            i = 1
            while i + W - 1 <= mp
                u0 = vload(Vec{W,Float64}, _vptr(pV, i, c, mp))
                u1 = vload(Vec{W,Float64}, _vptr(pV, i, c + 1, mp))
                w0 = vload(Vec{W,Float64}, _vptr(pC, i, j, ldC));     a0 = muladd(u0, w0, a0); b0 = muladd(u1, w0, b0)
                w1 = vload(Vec{W,Float64}, _vptr(pC, i, j + 1, ldC)); a1 = muladd(u0, w1, a1); b1 = muladd(u1, w1, b1)
                w2 = vload(Vec{W,Float64}, _vptr(pC, i, j + 2, ldC)); a2 = muladd(u0, w2, a2); b2 = muladd(u1, w2, b2)
                w3 = vload(Vec{W,Float64}, _vptr(pC, i, j + 3, ldC)); a3 = muladd(u0, w3, a3); b3 = muladd(u1, w3, b3)
                i += W
            end
            r0 = sum(a0); r1 = sum(a1); r2 = sum(a2); r3 = sum(a3)
            q0 = sum(b0); q1 = sum(b1); q2 = sum(b2); q3 = sum(b3)
            while i <= mp
                x0 = unsafe_load(pV, _lidx(i, c, mp)); x1 = unsafe_load(pV, _lidx(i, c + 1, mp))
                y0 = unsafe_load(pC, _lidx(i, j, ldC));     r0 = muladd(x0, y0, r0); q0 = muladd(x1, y0, q0)
                y1 = unsafe_load(pC, _lidx(i, j + 1, ldC)); r1 = muladd(x0, y1, r1); q1 = muladd(x1, y1, q1)
                y2 = unsafe_load(pC, _lidx(i, j + 2, ldC)); r2 = muladd(x0, y2, r2); q2 = muladd(x1, y2, q2)
                y3 = unsafe_load(pC, _lidx(i, j + 3, ldC)); r3 = muladd(x0, y3, r3); q3 = muladd(x1, y3, q3)
                i += 1
            end
            unsafe_store!(pWm, r0, _lidx(c, j, pb));     unsafe_store!(pWm, q0, _lidx(c + 1, j, pb))
            unsafe_store!(pWm, r1, _lidx(c, j + 1, pb)); unsafe_store!(pWm, q1, _lidx(c + 1, j + 1, pb))
            unsafe_store!(pWm, r2, _lidx(c, j + 2, pb)); unsafe_store!(pWm, q2, _lidx(c + 1, j + 2, pb))
            unsafe_store!(pWm, r3, _lidx(c, j + 3, pb)); unsafe_store!(pWm, q3, _lidx(c + 1, j + 3, pb))
            j += 4
        end
        while j <= nt                               # j-remainder for the 2 c-rows
            a = Vec{W,Float64}(0.0); b = Vec{W,Float64}(0.0); i = 1
            while i + W - 1 <= mp
                wv = vload(Vec{W,Float64}, _vptr(pC, i, j, ldC))
                a = muladd(vload(Vec{W,Float64}, _vptr(pV, i, c, mp)), wv, a)
                b = muladd(vload(Vec{W,Float64}, _vptr(pV, i, c + 1, mp)), wv, b)
                i += W
            end
            ra = sum(a); rb = sum(b)
            while i <= mp
                yy = unsafe_load(pC, _lidx(i, j, ldC))
                ra = muladd(unsafe_load(pV, _lidx(i, c, mp)), yy, ra)
                rb = muladd(unsafe_load(pV, _lidx(i, c + 1, mp)), yy, rb); i += 1
            end
            unsafe_store!(pWm, ra, _lidx(c, j, pb)); unsafe_store!(pWm, rb, _lidx(c + 1, j, pb))
            j += 1
        end
        c += 2
    end
    @inbounds while c <= pb                          # c-remainder (single column)
        _qr_VtC_col!(pWm, pV, pC, c, mp, nt, pb, ldC)
        c += 1
    end
end

# dlarfb gemm 2: C[i,j] -= Σ_c V[i,c]·Y[c,j]  (syrk-style: vectorize rows i, broadcast Y[c,j],
# reuse each V[i,c] load across NR=4 output columns). C: mp×nt ld=ldC; V: mp×pb ld=mp; Y: pb×nt ld=pb.
function _qr_subVY!(pC, pV, pY, mp::Int, nt::Int, pb::Int, ldC::Int)
    j = 1
    @inbounds while j + 3 <= nt
        i = 1
        while i + 2W - 1 <= mp                         # MR=2 × 4 cols = 8 accumulators
            r1 = i + W
            d00 = _vptr(pC, i, j, ldC);     A0 = vload(Vec{W,Float64}, d00)
            e00 = _vptr(pC, r1, j, ldC);    B0 = vload(Vec{W,Float64}, e00)
            d01 = _vptr(pC, i, j + 1, ldC); A1 = vload(Vec{W,Float64}, d01)
            e01 = _vptr(pC, r1, j + 1, ldC); B1 = vload(Vec{W,Float64}, e01)
            d02 = _vptr(pC, i, j + 2, ldC); A2 = vload(Vec{W,Float64}, d02)
            e02 = _vptr(pC, r1, j + 2, ldC); B2 = vload(Vec{W,Float64}, e02)
            d03 = _vptr(pC, i, j + 3, ldC); A3 = vload(Vec{W,Float64}, d03)
            e03 = _vptr(pC, r1, j + 3, ldC); B3 = vload(Vec{W,Float64}, e03)
            for c in 1:pb
                u0 = vload(Vec{W,Float64}, _vptr(pV, i, c, mp))
                u1 = vload(Vec{W,Float64}, _vptr(pV, r1, c, mp))
                g0 = Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j, pb)));     A0 = muladd(g0, u0, A0); B0 = muladd(g0, u1, B0)
                g1 = Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j + 1, pb))); A1 = muladd(g1, u0, A1); B1 = muladd(g1, u1, B1)
                g2 = Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j + 2, pb))); A2 = muladd(g2, u0, A2); B2 = muladd(g2, u1, B2)
                g3 = Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j + 3, pb))); A3 = muladd(g3, u0, A3); B3 = muladd(g3, u1, B3)
            end
            vstore(A0, d00); vstore(A1, d01); vstore(A2, d02); vstore(A3, d03)
            vstore(B0, e00); vstore(B1, e01); vstore(B2, e02); vstore(B3, e03)
            i += 2W
        end
        while i + W - 1 <= mp
            b0 = _vptr(pC, i, j, ldC);     a0 = vload(Vec{W,Float64}, b0)
            b1 = _vptr(pC, i, j + 1, ldC); a1 = vload(Vec{W,Float64}, b1)
            b2 = _vptr(pC, i, j + 2, ldC); a2 = vload(Vec{W,Float64}, b2)
            b3 = _vptr(pC, i, j + 3, ldC); a3 = vload(Vec{W,Float64}, b3)
            for c in 1:pb
                vic = vload(Vec{W,Float64}, _vptr(pV, i, c, mp))
                a0 = muladd(Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j, pb))), vic, a0)
                a1 = muladd(Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j + 1, pb))), vic, a1)
                a2 = muladd(Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j + 2, pb))), vic, a2)
                a3 = muladd(Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j + 3, pb))), vic, a3)
            end
            vstore(a0, b0); vstore(a1, b1); vstore(a2, b2); vstore(a3, b3)
            i += W
        end
        while i <= mp
            for dj in 0:3
                s = unsafe_load(pC, _lidx(i, j + dj, ldC))
                for c in 1:pb
                    s = muladd(-unsafe_load(pV, _lidx(i, c, mp)), unsafe_load(pY, _lidx(c, j + dj, pb)), s)
                end
                unsafe_store!(pC, s, _lidx(i, j + dj, ldC))
            end
            i += 1
        end
        j += 4
    end
    @inbounds while j <= nt
        i = 1
        while i + W - 1 <= mp
            b = _vptr(pC, i, j, ldC); a = vload(Vec{W,Float64}, b)
            for c in 1:pb
                a = muladd(Vec{W,Float64}(-unsafe_load(pY, _lidx(c, j, pb))),
                    vload(Vec{W,Float64}, _vptr(pV, i, c, mp)), a)
            end
            vstore(a, b); i += W
        end
        while i <= mp
            s = unsafe_load(pC, _lidx(i, j, ldC))
            for c in 1:pb
                s = muladd(-unsafe_load(pV, _lidx(i, c, mp)), unsafe_load(pY, _lidx(c, j, pb)), s)
            end
            unsafe_store!(pC, s, _lidx(i, j, ldC)); i += 1
        end
        j += 1
    end
end

# Core: factor the leading `mlog × n` block of `A` (A may have extra trailing rows for padding; the
# storage stride is A's). Operates only on single-level `view(A, …)` of a plain `Matrix`, so it never
# specializes the kernels for a nested SubArray type (which crashes the compiler).
function _qr_blocked_core!(A::AbstractMatrix{Float64}, tau::AbstractVector{Float64}, mlog::Int, nb::Int)
    n = size(A, 2)
    ld = stride(A, 2)
    k = min(mlog, n)
    pc = 1
    @inbounds while pc <= k
        pb = min(nb, k - pc + 1)
        qr_unblocked!(view(A, pc:mlog, pc:pc+pb-1), view(tau, pc:pc+pb-1))   # panel reduction (within-panel)
        jt0 = pc + pb
        if jt0 <= n
            mp = mlog - pc + 1
            nt = n - jt0 + 1
            # V (mp×pb): unit-diagonal trapezoid, essential parts from below the panel diagonal.
            V = Matrix{Float64}(undef, mp, pb)
            for c in 1:pb, i in 1:mp
                V[i, c] = i == c ? 1.0 : (i > c ? A[pc + i - 1, pc + c - 1] : 0.0)
            end
            # compact-WY T (pb×pb upper-tri), λ_c = 1/tau_c; Q = I − V T Vᵀ  (LAPACK dlarft).
            T = zeros(pb, pb)
            for c in 1:pb
                tc = tau[pc + c - 1]
                λ = isfinite(tc) ? 1.0 / tc : 0.0
                T[c, c] = λ
                if c > 1 && λ != 0.0
                    w = zeros(c - 1)                       # w = V[:,1:c-1]ᵀ V[:,c]
                    for kk in 1:c-1, i in 1:mp
                        w[kk] = muladd(V[i, kk], V[i, c], w[kk])
                    end
                    for r in 1:c-1                          # T[1:c-1,c] = −λ · T[1:c-1,1:c-1] · w
                        s = 0.0
                        for kk in r:c-1
                            s = muladd(T[r, kk], w[kk], s)
                        end
                        T[r, c] = -λ * s
                    end
                end
            end
            # trailing update C −= V · (Tᵀ · (Vᵀ · C)), C = A[pc:m, jt0:n], via tiled SIMD gemms.
            Wm = Matrix{Float64}(undef, pb, nt)
            Y = Matrix{Float64}(undef, pb, nt)
            GC.@preserve A V Wm Y begin
                pC = pointer(A, (jt0 - 1) * ld + pc)       # &A[pc, jt0]
                _qr_VtC!(pointer(Wm), pointer(V), pC, mp, nt, pb, ld)   # Wm = Vᵀ C
                for j in 1:nt, c in 1:pb                    # Y = Tᵀ Wm  (Tᵀ lower-tri; small)
                    s = 0.0
                    for r in 1:c
                        s = muladd(T[r, c], Wm[r, j], s)
                    end
                    Y[c, j] = s
                end
                _qr_subVY!(pC, pointer(V), pointer(Y), mp, nt, pb, ld)  # C −= V Y
            end
        end
        pc += pb
    end
    return true
end

"""
    qr_blocked!(A, tau; nb=8) -> true

Blocked compact-WY Householder QR (the perf path; same packed output as `qr_unblocked!`). Reduces each
`nb`-column panel with `qr_unblocked!`, builds the compact-WY factor `T` (LAPACK dlarft, `λ_k = 1/tau_k`
since faer's `H_k = I − v_k v_kᵀ/tau_k`), then applies `Qᵀ` to the trailing block as **gemms**
`C −= V·(Tᵀ·(Vᵀ·C))`. `tau` follows the faer convention (`Inf` ⇒ identity reflector); reconstruction
Q·R ≈ A to ~1e-13. Factors in place.

Note: unlike `cholesky_llt!`, padding a power-of-two leading dimension does **not** help here — measured
(2048: 0.49× padded vs 0.53× in-place, padding only adds copy cost). QR's large-`n` gap is algorithmic
(the thin `nb`-deep dlarfb streams the trailing block ~n/nb times; non-gemm overhead dominates), not the
`LDA=2^k` cache conflict that bottlenecks Cholesky's trsm/syrk. The core is split out as
`_qr_blocked_core!(A, tau, mlog, nb)` (factors the leading `mlog×n`) so it only ever takes single-level
`view`s of a `Matrix` — passing a nested `SubArray` (e.g. a padded view) crashes the Julia compiler.
"""
qr_blocked!(A::AbstractMatrix{Float64}, tau::AbstractVector{Float64}; nb::Int = 8) =
    _qr_blocked_core!(A, tau, size(A, 1), nb)

end # module Factorizations
