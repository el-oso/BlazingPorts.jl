# Ceiling experiment: can a tuned pure-Julia gemm (Octavian — matched OpenBLAS in the Tier-1 probe)
# for the trailing update reach faer? Right-looking driver reusing the src base kernel + src trsm, with
# Octavian.matmul_serial! for the syrk (A11 = -L10·L10ᵀ + A11, single-threaded). vs the src hand-tiled
# kernel, faer, OpenBLAS. This isolates "is the Cholesky gap closable in pure Julia?" from our hand kernel.
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_cholesky_octavian.jl

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.Factorizations as F
using BlazingPorts.Factorizations: cholesky_llt!
using LinearAlgebra
import Octavian

Harness.single_thread!()
const LIB = Harness.RUST_LIB
@noinline faer_chol(A::Matrix{Float64}) =
    ccall((:faer_cholesky, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

function _rl_oct!(A::Matrix{Float64}, o::Int, n::Int, ld::Int, block::Int, thr::Int)
    n <= thr && return F._chol_base!(pointer(A, o * ld + o + 1), n, ld)
    bs_outer = min(nextpow(2, n) ÷ 2, block)
    j = 0
    while j < n
        bs = min(bs_outer, n - j)
        _rl_oct!(A, o + j, bs, ld, block, thr) || return false
        m = n - j - bs
        if m > 0
            r0 = o + j + bs; c0 = o + j
            F._trsm_right_lower!(pointer(A, c0 * ld + c0 + 1), pointer(A, c0 * ld + r0 + 1), bs, m, ld)
            A10 = @view A[r0+1:r0+m, c0+1:c0+bs]
            A11 = @view A[r0+1:r0+m, r0+1:r0+m]
            Octavian.matmul_serial!(A11, A10, transpose(A10), -1.0, 1.0)   # A11 -= L10·L10ᵀ
        end
        j += bs
    end
    return true
end
cholesky_oct!(A::Matrix{Float64}) =
    GC.@preserve A _rl_oct!(A, 0, size(A, 1), size(A, 1), F.BLOCK_SIZE, F.RECURSION_THRESHOLD)

function probe_size(n::Int)
    A = Matrix(let M = randn(n, n); M'M + n * I end)
    let s = copy(A); cholesky_oct!(s); L = LowerTriangular(s)
        @assert maximum(abs.(L * L' .- A)) / maximum(abs.(A)) < 1e-11 "octavian recon off n=$n"
    end
    sb = copy(A); symb = Symmetric(sb, :L); st = copy(A); so = copy(A); sf = copy(A)
    fb = @noinline () -> (copyto!(sb, A); cholesky!(symb); GC.gc(false); sb[1])
    ft = @noinline () -> (copyto!(st, A); cholesky_llt!(st); GC.gc(false); st[1])
    fo = @noinline () -> (copyto!(so, A); cholesky_oct!(so); GC.gc(false); so[1])
    ff = @noinline () -> (copyto!(sf, A); faer_chol(sf))
    fb(); ft(); fo(); ff()
    GC.enable(false); GC.gc(false)
    pb = run_probe("OpenBLAS", fb; seconds = 3.0)
    pt = run_probe("BP-tiled", ft; seconds = 3.0)
    po = run_probe("BP-Octavian", fo; seconds = 3.0)
    GC.enable(true); GC.gc()
    pf = run_probe("faer", ff; seconds = 3.0)
    probes = Probe[pb, pt, po, pf]
    crate = "cholesky_octavian_$(n)x$(n)"
    report(crate, probes; rust_label = "faer")
    save_probes(crate, probes)
    plot_probe(crate, probes)
    println()
    return probes
end

println("\n=== Cholesky: Octavian trailing update vs hand-tiled / faer / OpenBLAS ===\n")
for n in (128, 256, 512, 1024)
    probe_size(n)
end
println("Done — probe_cholesky_octavian.jl")
