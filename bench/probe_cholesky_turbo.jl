# Path A experiment: does a LoopVectorization @turbo trailing update (auto cache/register blocking)
# close the Cholesky tiling gap? Right-looking driver reusing the src bit-exact base kernel for the
# diagonal blocks, with @views + @turbo for the panel solve (trsm) and trailing rank-k update (syrk).
# LoopVectorization stays a BENCH dep (not src). Compared vs the naive src driver, faer, OpenBLAS.
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_cholesky_turbo.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.Factorizations as F
using BlazingPorts.Factorizations: cholesky_llt!
using LinearAlgebra
using LoopVectorization: @turbo

Harness.single_thread!()
const LIB = Harness.RUST_LIB
@noinline faer_chol(A::Matrix{Float64}) =
    ccall((:faer_cholesky, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

# trsm: reuse the known-correct src panel solve (pointer form) so this experiment isolates the @turbo
# syrk — the O(n³) bottleneck — with everything else verified.

# syrk: A11 −= L10·L10ᵀ. Compute the FULL m×m rank-bs product with @turbo (upper triangle overwritten
# but never read — the factorization only touches lower). One clean @turbo gemm = auto blocking.
function syrk_turbo!(A11, A10, m::Int, bs::Int)
    @turbo for j in 1:m, i in 1:m
        s = 0.0
        for c in 1:bs
            s += A10[i, c] * A10[j, c]
        end
        A11[i, j] -= s
    end
    return nothing
end

# right-looking recursive driver; `o` is the 0-based absolute offset of this block within A.
function _rl_turbo!(A::Matrix{Float64}, o::Int, n::Int, ld::Int, block::Int, thr::Int)
    if n <= thr
        return F._chol_base!(pointer(A, o * ld + o + 1), n, ld)
    end
    bs_outer = min(nextpow(2, n) ÷ 2, block)
    j = 0
    while j < n
        bs = min(bs_outer, n - j)
        _rl_turbo!(A, o + j, bs, ld, block, thr) || return false
        m = n - j - bs
        if m > 0
            r0 = o + j + bs
            c0 = o + j
            p00 = pointer(A, c0 * ld + c0 + 1)   # A[c0+1, c0+1]
            p10 = pointer(A, c0 * ld + r0 + 1)   # A[r0+1, c0+1]
            F._trsm_right_lower!(p00, p10, bs, m, ld)
            A10 = @view A[r0+1:r0+m, c0+1:c0+bs]
            A11 = @view A[r0+1:r0+m, r0+1:r0+m]
            syrk_turbo!(A11, A10, m, bs)
        end
        j += bs
    end
    return true
end

function cholesky_turbo!(A::Matrix{Float64})
    n = size(A, 1)
    GC.@preserve A _rl_turbo!(A, 0, n, n, F.BLOCK_SIZE, F.RECURSION_THRESHOLD)
end

# ── probe ──
function probe_size(n::Int)
    A = Matrix(let M = randn(n, n); M'M + n * I end)
    # correctness of the @turbo driver
    let s = copy(A); cholesky_turbo!(s); L = LowerTriangular(s)
        rel = maximum(abs.(L * L' .- A)) / maximum(abs.(A))
        @assert rel < 1e-11 "turbo recon off at n=$n: $rel"
    end
    sb = copy(A); symb = Symmetric(sb, :L); sN = copy(A); sT = copy(A); sf = copy(A)
    fb = @noinline () -> (copyto!(sb, A); cholesky!(symb); GC.gc(false); sb[1])
    fN = @noinline () -> (copyto!(sN, A); cholesky_llt!(sN); GC.gc(false); sN[1])
    fT = @noinline () -> (copyto!(sT, A); cholesky_turbo!(sT); GC.gc(false); sT[1])
    ff = @noinline () -> (copyto!(sf, A); faer_chol(sf))
    fb(); fN(); fT(); ff()
    GC.enable(false); GC.gc(false)
    pb = run_probe("OpenBLAS", fb; seconds = 3.0)
    pN = run_probe("BP-tiled", fN; seconds = 3.0)
    pT = run_probe("BP-@turbo", fT; seconds = 3.0)
    GC.enable(true); GC.gc()
    pf = run_probe("faer", ff; seconds = 3.0)
    probes = Probe[pb, pN, pT, pf]
    crate = "cholesky_turbo_$(n)x$(n)"
    report(crate, probes; rust_label = "faer")
    save_probes(crate, probes)
    plot_probe(crate, probes)
    println()
    return probes
end

println("\n=== Cholesky @turbo trailing update vs naive / faer / OpenBLAS ===\n")
for n in (64, 128, 256, 512)
    probe_size(n)
end
println("Done — probe_cholesky_turbo.jl")
