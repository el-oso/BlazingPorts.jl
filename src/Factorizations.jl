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

export cholesky_llt!

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
# Column c of L10: L10[:,c] = (A10[:,c] − Σ_{k<c} L00[c,k]·L10[:,k]) · (1/L00[c,c]). Vectorized over rows.
function _trsm_right_lower!(p00::Ptr{Float64}, p10::Ptr{Float64}, bs::Int, m::Int, ld::Int)
    @inbounds for c in 1:bs
        invc = 1.0 / unsafe_load(p00, _lidx(c, c, ld))
        vinv = Vec{W,Float64}(invc)
        i = 1
        while i + W - 1 <= m
            base = _vptr(p10, i, c, ld)
            acc = vload(Vec{W,Float64}, base)
            for k in 1:c-1
                nck = -unsafe_load(p00, _lidx(c, k, ld))
                acc = muladd(Vec{W,Float64}(nck), vload(Vec{W,Float64}, _vptr(p10, i, k, ld)), acc)
            end
            vstore(acc * vinv, base)
            i += W
        end
        while i <= m
            s = unsafe_load(p10, _lidx(i, c, ld))
            for k in 1:c-1
                s = muladd(-unsafe_load(p00, _lidx(c, k, ld)), unsafe_load(p10, _lidx(i, k, ld)), s)
            end
            unsafe_store!(p10, s * invc, _lidx(i, c, ld))
            i += 1
        end
    end
    return nothing
end

# ── trailing symmetric rank-bs update: A11 (m×m, lower) −= L10·L10ᵀ. Vectorized over rows i. ──
function _syrk_lower!(p11::Ptr{Float64}, p10::Ptr{Float64}, m::Int, bs::Int, ld::Int)
    @inbounds for j in 1:m
        i = j
        while i + W - 1 <= m
            base = _vptr(p11, i, j, ld)
            acc = vload(Vec{W,Float64}, base)
            for c in 1:bs
                njc = -unsafe_load(p10, _lidx(j, c, ld))
                acc = muladd(Vec{W,Float64}(njc), vload(Vec{W,Float64}, _vptr(p10, i, c, ld)), acc)
            end
            vstore(acc, base)
            i += W
        end
        while i <= m
            s = unsafe_load(p11, _lidx(i, j, ld))
            for c in 1:bs
                s = muladd(-unsafe_load(p10, _lidx(j, c, ld)), unsafe_load(p10, _lidx(i, c, ld)), s)
            end
            unsafe_store!(p11, s, _lidx(i, j, ld))
            i += 1
        end
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
function cholesky_llt!(A::AbstractMatrix{Float64})
    n = size(A, 1)
    n == 0 && return true
    Base.require_one_based_indexing(A)
    ld = stride(A, 2)
    GC.@preserve A begin
        return _chol_rl!(pointer(A), n, ld, BLOCK_SIZE, RECURSION_THRESHOLD)
    end
end

end # module Factorizations
