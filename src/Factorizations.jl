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
function cholesky_llt!(A::AbstractMatrix{Float64})
    n = size(A, 1)
    n == 0 && return true
    Base.require_one_based_indexing(A)
    ld = stride(A, 2)
    GC.@preserve A begin
        return _chol_rl!(pointer(A), n, ld, BLOCK_SIZE, RECURSION_THRESHOLD)
    end
end


# ── QR Factorization — Layer D (planned) ────────────────────────────────────────────────────────
#
# Faithful pure-Julia port of faer 0.24.1 Householder QR (no pivoting).
# Convention: H_k = I − v_k v_kᵀ / τ_k  (divides by τ, matching faer's `make_householder_in_place`).
# τ_k = Inf encodes a trivial (identity) reflector; v_k has implicit leading 1 at index k.
#
# Planned layers (Layer D):
#   D-A  golden harness only (done: bench/rust_compare/qr_verify.rs + test/factorizations_tests.jl @testitem "qr_golden")
#   D-B  qr_in_place_unblocked! — column-by-column Householder reduction (base kernel, n ≤ threshold)
#   D-C  qr_in_place_blocked!   — blocked driver: panel reduction + trailing update via WY representation
#
# The golden file is bench/rust_compare/qr_golden.txt.  Regenerate:
#   cd bench/rust_compare/rust && cargo build --release --bin qr_verify
#   ./target/release/qr_verify > ../qr_golden.txt

end # module Factorizations
