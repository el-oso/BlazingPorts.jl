# Prototype: PACKED syrk for the Cholesky trailing update (GotoBLAS/BLIS-style), written to be PORTABLE
# (no machine-specific constants): SIMD width W and the L2-derived cache-block MC are queried from the
# host; the register tile MRV×NRB is a modest, portable choice that also fits AVX2's 16 registers.
#
# Idea (why it should beat the non-packed kernel at large n): the non-packed syrk reads L10 with the
# parent leading-dimension stride and re-reads it across column blocks → L2-bandwidth-bound. Packing L10
# into compact MR-row panels makes the microkernel's reduction unit-stride and keeps the active panel
# L1-resident across the column sweep. Pack once, reuse.
#
# Compares cholesky_packed! vs the committed src cholesky_llt! vs faer at n = 256/512/1024.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_cholesky_packed.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.Factorizations as F
using BlazingPorts.Factorizations: cholesky_llt!
using LinearAlgebra
import CPUSummary
import SIMD: Vec, vload, vstore

Harness.single_thread!()
const LIB = Harness.RUST_LIB
@noinline faer_chol(A::Matrix{Float64}) =
    ccall((:faer_cholesky, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

const W = F.W
const MRV = 2                 # A row-vectors per microkernel tile (portable; 2*W rows)
const NRB = 4                 # B columns per microkernel tile
const MWR = MRV * W           # rows per packed A-panel strip

@inline _el(p, i) = unsafe_load(p, i)
@inline _bptr(p, e) = p + e * sizeof(Float64)     # 0-based element offset → Ptr

# host-generic cache block: largest MWR-multiple with an MC×bs panel fitting ~half L2
function mc_block(bs::Int)
    l2 = Int(CPUSummary.cache_size(Val(2)))
    cap = max(MWR, (l2 ÷ 2) ÷ (bs * 8))
    (cap ÷ MWR) * MWR
end

# pack rows [r0+1 : r0+rows] of L10 (col-major at pL, leading dim ld) into RW-row panels stored compact
# (within a panel: col-major, leading dim RW; panels laid out contiguously). Zero-pads a short last panel.
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

# packed trailing update: A11 (m×m, lower) −= L10·L10ᵀ.  Requires m % MWR == 0 and m % NRB == 0.
function _syrk_packed!(pC::Ptr{Float64}, pL::Ptr{Float64}, m::Int, bs::Int, ld::Int,
        Ap::Vector{Float64}, Bp::Vector{Float64})
    pAp = pointer(Ap); pBp = pointer(Bp)
    _pack!(pBp, pL, 0, m, bs, ld, NRB)                  # pack all of L10's rows as B (once)
    MC = mc_block(bs)
    GC.@preserve Ap Bp begin
        ic = 0
        @inbounds while ic < m
            mc = min(MC, m - ic)
            _pack!(pAp, pL, ic, mc, bs, ld, MWR)        # pack A-rows of this MC-panel
            na = cld(mc, MWR)
            for sa in 0:na-1
                arow0 = ic + sa * MWR                    # 0-based row start of this A-strip
                aoff = sa * MWR * bs
                nb_last = (arow0 + MWR - 1) ÷ NRB        # last lower B-strip (incl. near-diagonal sliver)
                for sb in 0:nb_last
                    jb = sb * NRB                        # 0-based col start
                    boff = sb * NRB * bs
                    r0 = arow0 + 1; r1 = arow0 + 1 + W   # 1-based C rows for the two W-vectors
                    c0 = jb + 1
                    p00 = F._vptr(pC, r0, c0, ld);     A0 = vload(Vec{W,Float64}, p00)
                    q00 = F._vptr(pC, r1, c0, ld);     B0 = vload(Vec{W,Float64}, q00)
                    p01 = F._vptr(pC, r0, c0 + 1, ld); A1 = vload(Vec{W,Float64}, p01)
                    q01 = F._vptr(pC, r1, c0 + 1, ld); B1 = vload(Vec{W,Float64}, q01)
                    p02 = F._vptr(pC, r0, c0 + 2, ld); A2 = vload(Vec{W,Float64}, p02)
                    q02 = F._vptr(pC, r1, c0 + 2, ld); B2 = vload(Vec{W,Float64}, q02)
                    p03 = F._vptr(pC, r0, c0 + 3, ld); A3 = vload(Vec{W,Float64}, p03)
                    q03 = F._vptr(pC, r1, c0 + 3, ld); B3 = vload(Vec{W,Float64}, q03)
                    for c in 1:bs
                        av0 = vload(Vec{W,Float64}, _bptr(pAp, aoff + (c - 1) * MWR))
                        av1 = vload(Vec{W,Float64}, _bptr(pAp, aoff + (c - 1) * MWR + W))
                        bo = boff + (c - 1) * NRB
                        g0 = Vec{W,Float64}(-_el(pBp, bo + 1)); A0 = muladd(g0, av0, A0); B0 = muladd(g0, av1, B0)
                        g1 = Vec{W,Float64}(-_el(pBp, bo + 2)); A1 = muladd(g1, av0, A1); B1 = muladd(g1, av1, B1)
                        g2 = Vec{W,Float64}(-_el(pBp, bo + 3)); A2 = muladd(g2, av0, A2); B2 = muladd(g2, av1, B2)
                        g3 = Vec{W,Float64}(-_el(pBp, bo + 4)); A3 = muladd(g3, av0, A3); B3 = muladd(g3, av1, B3)
                    end
                    vstore(A0, p00); vstore(A1, p01); vstore(A2, p02); vstore(A3, p03)
                    vstore(B0, q00); vstore(B1, q01); vstore(B2, q02); vstore(B3, q03)
                end
            end
            ic += MC
        end
    end
    return nothing
end

# right-looking driver (mirrors src), packed syrk for divisible large m, src fallback otherwise.
function _chol_packed!(p::Ptr{Float64}, n::Int, ld::Int, bsz::Int, thr::Int, Ap, Bp)
    n <= thr && return F._chol_base!(p, n, ld)
    bs_outer = min(nextpow(2, n) ÷ 2, bsz)
    j = 0
    while j < n
        bs = min(bs_outer, n - j)
        F._vptr(p, j + 1, j + 1, ld)
        _chol_packed!(F._vptr(p, j + 1, j + 1, ld), bs, ld, bsz, thr, Ap, Bp) || return false
        m = n - j - bs
        if m > 0
            p10 = F._vptr(p, j + bs + 1, j + 1, ld)
            p11 = F._vptr(p, j + bs + 1, j + bs + 1, ld)
            F._trsm_right_lower!(F._vptr(p, j + 1, j + 1, ld), p10, bs, m, ld)
            # cache-derived hybrid: pack only when the L10 panel (m×bs) exceeds ~half L2 — below that the
            # non-packed kernel's lower overhead wins (it already fits cache). Generic (host L2).
            usepack = (m * bs * sizeof(Float64) > Int(CPUSummary.cache_size(Val(2))) ÷ 2) &&
                      m % MWR == 0 && m % NRB == 0 && m >= MWR
            if usepack
                _syrk_packed!(p11, p10, m, bs, ld, Ap, Bp)
            else
                F._syrk_lower!(p11, p10, m, bs, ld)
            end
        end
        j += bs
    end
    return true
end

function cholesky_packed!(A::Matrix{Float64})
    n = size(A, 1)
    bsz = F.BLOCK_SIZE
    Ap = Vector{Float64}(undef, mc_block(bsz) * bsz)            # A-panel scratch
    Bp = Vector{Float64}(undef, (cld(n, NRB) * NRB) * bsz)      # B (all rows) scratch
    GC.@preserve A _chol_packed!(pointer(A), n, stride(A, 2), bsz, F.RECURSION_THRESHOLD, Ap, Bp)
end

function probe_size(n::Int)
    A = Matrix(let M = randn(n, n); M'M + n * I end)
    let s = copy(A); cholesky_packed!(s); L = LowerTriangular(s)
        @assert maximum(abs.(L * L' .- A)) / maximum(abs.(A)) < 1e-11 "packed recon off n=$n"
    end
    sb = copy(A); symb = Symmetric(sb, :L); st = copy(A); sp = copy(A); sf = copy(A)
    fb = @noinline () -> (copyto!(sb, A); cholesky!(symb); GC.gc(false); sb[1])
    ft = @noinline () -> (copyto!(st, A); cholesky_llt!(st); GC.gc(false); st[1])
    fp = @noinline () -> (copyto!(sp, A); cholesky_packed!(sp); GC.gc(false); sp[1])
    ff = @noinline () -> (copyto!(sf, A); faer_chol(sf))
    fb(); ft(); fp(); ff()
    GC.enable(false); GC.gc(false)
    pb = run_probe("OpenBLAS", fb; seconds = 3.0)
    pt = run_probe("BP-current", ft; seconds = 3.0)
    pp = run_probe("BP-packed", fp; seconds = 3.0)
    GC.enable(true); GC.gc()
    pf = run_probe("faer", ff; seconds = 3.0)
    probes = Probe[pb, pt, pp, pf]
    crate = "cholesky_packed_$(n)x$(n)"
    report(crate, probes; rust_label = "faer")
    save_probes(crate, probes)
    plot_probe(crate, probes)
    println()
    return probes
end

println("\n=== Cholesky PACKED syrk vs current / faer / OpenBLAS (W=$W, MRV=$MRV, NRB=$NRB, MC=$(mc_block(F.BLOCK_SIZE))) ===\n")
for n in (256, 512, 1024, 2048)
    probe_size(n)
end
println("Done — probe_cholesky_packed.jl")
