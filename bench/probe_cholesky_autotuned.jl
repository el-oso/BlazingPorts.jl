# Prototype: PACKED + AUTOTUNED + REMAINDER-SAFE Cholesky trailing update.
#
# Exploits Julia's dynamic-yet-compiled nature (the differentiator vs C/Rust): the microkernel is a
# `@generated` function parameterized by the register tile (MR row-vectors × NR cols), so EVERY tile size
# is JIT-specialized/unrolled for free. A one-time runtime autotuner benchmarks a set of (MR,NR)
# candidates on THIS CPU and caches the winner (FFTW-"plan" style) — so the kernel adapts to the host's
# register file / cache without any hand-set constants or recompilation.
#
# Remainder-safe: full interior tiles use the fast unrolled kernel; bottom/right/near-diagonal edges use a
# masked kernel (SIMD.jl masked vload/vstore), so ARBITRARY n works (no divisibility requirement).
# Portable: W from the host; (MR,NR) chosen by the autotuner; packing makes the reduction unit-stride.
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_cholesky_autotuned.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.Factorizations as F
using BlazingPorts.Factorizations: cholesky_llt!
using LinearAlgebra
import CPUSummary
import SIMD: Vec, vload, vstore
using Statistics: median

Harness.single_thread!()
const LIB = Harness.RUST_LIB
@noinline faer_chol(A::Matrix{Float64}) =
    ccall((:faer_cholesky, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

const W = F.W
@inline _el(p, i) = unsafe_load(p, i)
@inline _bptr(p, e) = p + e * sizeof(Float64)

# ── full MR×NR microkernel, JIT-specialized per (MR,NR) via @generated (unrolled accumulators) ──
# C[tile] −= packed_A · packed_B over the bs reduction; plain (unmasked) loads/stores — interior only.
@generated function _uk_full!(::Val{MR}, ::Val{NR}, pC, pA, pB,
        aoff::Int, boff::Int, cr0::Int, cc0::Int, ld::Int, bs::Int) where {MR,NR}
    acc = [Symbol(:acc_, mv, :_, nb) for mv in 1:MR, nb in 1:NR]
    cp = [Symbol(:cp_, mv, :_, nb) for mv in 1:MR, nb in 1:NR]
    av = [Symbol(:av_, mv) for mv in 1:MR]
    g = [Symbol(:g_, nb) for nb in 1:NR]
    init = Expr(:block)
    for nb in 1:NR, mv in 1:MR
        push!(init.args, :($(cp[mv, nb]) = F._vptr(pC, cr0 + $((mv - 1)) * $W, cc0 + $(nb - 1), ld)))
        push!(init.args, :($(acc[mv, nb]) = vload(Vec{$W,Float64}, $(cp[mv, nb]))))
    end
    body = Expr(:block)
    for mv in 1:MR
        push!(body.args, :($(av[mv]) = vload(Vec{$W,Float64}, _bptr(pA, ao + $((mv - 1)) * $W))))
    end
    for nb in 1:NR
        push!(body.args, :($(g[nb]) = Vec{$W,Float64}(-_el(pB, bo + $nb))))
    end
    for nb in 1:NR, mv in 1:MR
        push!(body.args, :($(acc[mv, nb]) = muladd($(g[nb]), $(av[mv]), $(acc[mv, nb]))))
    end
    st = Expr(:block)
    for nb in 1:NR, mv in 1:MR
        push!(st.args, :(vstore($(acc[mv, nb]), $(cp[mv, nb]))))
    end
    quote
        $init
        @inbounds for c in 1:bs
            ao = aoff + (c - 1) * $(MR * W)
            bo = boff + (c - 1) * $NR
            $body
        end
        $st
        return nothing
    end
end

# ── masked edge kernel: tile with `mrows` valid rows (≤ MWR) and `nc` valid cols (≤ NR). ──
@inline function _uk_edge!(pC, pA, pB, aoff::Int, boff::Int, cr0::Int, cc0::Int, ld::Int, bs::Int,
        MWR::Int, NRp::Int, mrows::Int, nc::Int)
    nv = cld(mrows, W)
    lanes = Vec{W,Int}(ntuple(identity, W))
    @inbounds for nb in 1:nc
        col = cc0 + nb - 1
        for vv in 0:nv-1
            r0 = cr0 + vv * W
            valid = min(W, mrows - vv * W)
            mask = lanes <= valid
            cp = F._vptr(pC, r0, col, ld)
            acc = vload(Vec{W,Float64}, cp, mask)
            for c in 1:bs
                av = vload(Vec{W,Float64}, _bptr(pA, aoff + (c - 1) * MWR + vv * W))
                acc = muladd(Vec{W,Float64}(-_el(pB, boff + (c - 1) * NRp + nb)), av, acc)
            end
            vstore(acc, cp, mask)
        end
    end
end

# pack rows [r0+1:r0+rows] of L10 (col-major at pL, ld) into RW-row panels (compact, zero-padded).
@inline function _pack!(dst::Ptr{Float64}, pL::Ptr{Float64}, r0::Int, rows::Int, bs::Int, ld::Int, RW::Int)
    np = cld(rows, RW)
    @inbounds for s in 0:np-1
        base = s * RW * bs
        for c in 1:bs
            o = base + (c - 1) * RW
            for r in 0:RW-1
                lr = s * RW + r
                unsafe_store!(dst, lr < rows ? _el(pL, (c - 1) * ld + r0 + lr + 1) : 0.0, o + r + 1)
            end
        end
    end
end

# packed, remainder-safe trailing update A11(m×m,lower) −= L10·L10ᵀ for tile (Val{MR},Val{NR}).
function _syrk_tuned!(::Val{MR}, ::Val{NR}, pC, pL, m::Int, bs::Int, ld::Int,
        Ap::Vector{Float64}, Bp::Vector{Float64}) where {MR,NR}
    MWR = MR * W
    pAp = pointer(Ap); pBp = pointer(Bp)
    GC.@preserve Ap Bp begin
        _pack!(pAp, pL, 0, m, bs, ld, MWR)        # pack A (all rows) into MWR panels
        _pack!(pBp, pL, 0, m, bs, ld, NR)         # pack B (all rows) into NR panels
        ncb = cld(m, NR)                          # column blocks
        @inbounds for sb in 0:ncb-1
            jb = sb * NR
            nc = min(NR, m - jb)
            boff = sb * NR * bs
            sa0 = jb ÷ MWR                         # first row-strip at/below the diagonal
            nsa = cld(m, MWR)
            for sa in sa0:nsa-1
                ar0 = sa * MWR                     # 0-based row start
                aoff = sa * MWR * bs
                mrows = min(MWR, m - ar0)
                if mrows == MWR && nc == NR
                    _uk_full!(Val(MR), Val(NR), pC, pAp, pBp, aoff, boff, ar0 + 1, jb + 1, ld, bs)
                else
                    _uk_edge!(pC, pAp, pBp, aoff, boff, ar0 + 1, jb + 1, ld, bs, MWR, NR, mrows, nc)
                end
            end
        end
    end
    return nothing
end

# ── right-looking driver parameterized by the tuned tile ──
function _chol_tuned!(::Val{MR}, ::Val{NR}, p, n::Int, ld::Int, bsz::Int, thr::Int, Ap, Bp) where {MR,NR}
    n <= thr && return F._chol_base!(p, n, ld)
    bs_outer = min(nextpow(2, n) ÷ 2, bsz)
    j = 0
    while j < n
        bs = min(bs_outer, n - j)
        _chol_tuned!(Val(MR), Val(NR), F._vptr(p, j + 1, j + 1, ld), bs, ld, bsz, thr, Ap, Bp) || return false
        m = n - j - bs
        if m > 0
            p10 = F._vptr(p, j + bs + 1, j + 1, ld)
            p11 = F._vptr(p, j + bs + 1, j + bs + 1, ld)
            F._trsm_right_lower!(F._vptr(p, j + 1, j + 1, ld), p10, bs, m, ld)
            # pack only pays off past ~½L2; small trailing → fast non-packed kernel.
            if m * bs * sizeof(Float64) > Int(CPUSummary.cache_size(Val(2))) ÷ 2
                _syrk_tuned!(Val(MR), Val(NR), p11, p10, m, bs, ld, Ap, Bp)
            else
                F._syrk_lower!(p11, p10, m, bs, ld)
            end
        end
        j += bs
    end
    return true
end

# ── one-time autotuner: JIT-specialize each candidate, benchmark on this CPU, cache the winner ──
const _PLAN = Ref((0, 0))
const CANDIDATES = ((2, 4), (3, 4), (4, 4), (2, 6), (3, 6), (2, 8), (6, 2), (4, 6))

function syrk_plan()
    _PLAN[][1] != 0 && return _PLAN[]
    m = 480; bs = 128                              # representative trailing update
    L = randn(m, bs); A11 = Matrix(let M = randn(m, m); M'M + m * I end)
    Ap = Vector{Float64}(undef, (cld(m, 8 * W) * 8 * W) * bs)   # oversized for any MR≤8
    Bp = Vector{Float64}(undef, (cld(m, 8) * 8) * bs)
    best = (2, 4); bt = Inf
    for (MR, NR) in CANDIDATES
        C = copy(A11)
        f = () -> GC.@preserve C L _syrk_tuned!(Val(MR), Val(NR), pointer(C), pointer(L), m, bs, m, Ap, Bp)
        f()  # warm/compile (discard first)
        t = median(@elapsed(f()) for _ in 1:25)    # median-of-25: robust selection (project policy: never min)
        if t < bt; bt = t; best = (MR, NR); end
    end
    _PLAN[] = best
    return best
end

function cholesky_auto!(A::Matrix{Float64})
    MR, NR = syrk_plan()
    bsz = F.BLOCK_SIZE
    Ap = Vector{Float64}(undef, (cld(size(A, 1), MR * W) * MR * W) * bsz)
    Bp = Vector{Float64}(undef, (cld(size(A, 1), NR) * NR) * bsz)
    GC.@preserve A begin
        # dispatch on the tuned tile (Val) so the specialized kernel is used
        return _dispatch_auto!(Val(MR), Val(NR), A, bsz, Ap, Bp)
    end
end
@inline _dispatch_auto!(::Val{MR}, ::Val{NR}, A, bsz, Ap, Bp) where {MR,NR} =
    _chol_tuned!(Val(MR), Val(NR), pointer(A), size(A, 1), stride(A, 2), bsz, F.RECURSION_THRESHOLD, Ap, Bp)

# ── probe (incl. NON-power-of-2 sizes to exercise remainder safety) ──
function probe_size(n::Int)
    A = Matrix(let M = randn(n, n); M'M + n * I end)
    let s = copy(A); cholesky_auto!(s); L = LowerTriangular(s)
        rel = maximum(abs.(L * L' .- A)) / maximum(abs.(A))
        @assert rel < 1e-10 "auto recon off n=$n: $rel"
    end
    sb = copy(A); symb = Symmetric(sb, :L); st = copy(A); sa = copy(A); sf = copy(A)
    fb = @noinline () -> (copyto!(sb, A); cholesky!(symb); GC.gc(false); sb[1])
    ft = @noinline () -> (copyto!(st, A); cholesky_llt!(st); GC.gc(false); st[1])
    fa = @noinline () -> (copyto!(sa, A); cholesky_auto!(sa); GC.gc(false); sa[1])
    ff = @noinline () -> (copyto!(sf, A); faer_chol(sf))
    fb(); ft(); fa(); ff()
    GC.enable(false); GC.gc(false)
    pb = run_probe("OpenBLAS", fb; seconds = 2.5)
    pt = run_probe("BP-current", ft; seconds = 2.5)
    pa = run_probe("BP-auto", fa; seconds = 2.5)
    GC.enable(true); GC.gc()
    pf = run_probe("faer", ff; seconds = 2.5)
    report("cholesky_auto_$(n)x$(n)", Probe[pb, pt, pa, pf]; rust_label = "faer")
    save_probes("cholesky_auto_$(n)x$(n)", Probe[pb, pt, pa, pf])
    println()
end

println("\n=== Autotuning syrk tile on this CPU... ===")
println("  chosen (MR, NR) = ", syrk_plan(), "   (W=$W; candidates=$(CANDIDATES))")
println("\n=== Cholesky autotuned+packed+remainder-safe vs current / faer / OpenBLAS ===\n")
for n in (300, 512, 1000, 1024, 1500, 2048)
    probe_size(n)
end
println("Done — probe_cholesky_autotuned.jl")
