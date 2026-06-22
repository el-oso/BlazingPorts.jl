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

"""
    cholesky_llt!(A::Matrix{Float64}) -> Bool

In-place real LLᵀ Cholesky — faithful port of faer 0.24.1 `simd_cholesky` (the base case of
`cholesky_recursion_right_looking`). Overwrites the **lower triangle** of `A` with `L`; the upper
triangle is left untouched. Returns `false` (leaving A partially factored) on a non-positive pivot.

Per-element arithmetic matches faer exactly for bit-exact agreement with its golden output:
left-looking, `L[i,j] = (A[i,j] − Σ_{k<j} L[j,k]·L[i,k])` accumulated by **ascending-k FMA**, then the
whole column (diagonal included) is **multiplied by `inv = 1/sqrt(diag)`** (reciprocal-multiply, as
faer's `mul_real(·, diag.recip())` — *not* a divide). Vectorized over rows `i` with `Vec{W}`.

This is the Layer B base kernel (correct for any n; the Layer C driver will call it on ≤64 blocks).
"""
function cholesky_llt!(A::AbstractMatrix{Float64})
    n = size(A, 1)
    n == 0 && return true
    Base.require_one_based_indexing(A)
    ld = stride(A, 2)                       # column stride (= n for a dense Matrix)
    lin(i, k) = (k - 1) * ld + i            # 1-based linear index of A[i,k]
    GC.@preserve A begin
        @inbounds for j in 1:n
            # ── pass 1: left-looking update of column j, rows j:n (diagonal included) ──
            i = j
            while i + W - 1 <= n
                acc = vload(Vec{W,Float64}, pointer(A, lin(i, j)))
                for k in 1:j-1
                    njk = -A[j, k]                                   # exact negate
                    aik = vload(Vec{W,Float64}, pointer(A, lin(i, k)))
                    acc = muladd(Vec{W,Float64}(njk), aik, acc)      # fused, per lane
                end
                vstore(acc, pointer(A, lin(i, j)))
                i += W
            end
            while i <= n                     # scalar tail (same FMA → same bits)
                s = A[i, j]
                for k in 1:j-1
                    s = muladd(-A[j, k], A[i, k], s)
                end
                A[i, j] = s
                i += 1
            end
            # ── diagonal + scale column j by reciprocal of sqrt(diag) ──
            d = A[j, j]
            (d > 0.0) || return false        # non-positive pivot
            inv = 1.0 / sqrt(d)
            vinv = Vec{W,Float64}(inv)
            i = j
            while i + W - 1 <= n
                v = vload(Vec{W,Float64}, pointer(A, lin(i, j)))
                vstore(v * vinv, pointer(A, lin(i, j)))
                i += W
            end
            while i <= n
                A[i, j] *= inv
                i += 1
            end
        end
    end
    return true
end

end # module Factorizations
