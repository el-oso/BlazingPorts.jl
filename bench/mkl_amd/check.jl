# MKL-on-AMD check: time single-threaded Cholesky/QR under MKL and report the median, so we can see
# whether MKL_DEBUG_CPU_TYPE / MKL_ENABLE_INSTRUCTIONS / the fakeintel LD_PRELOAD shim change the
# dispatched kernel speed on this Zen CPU. Run under different env via the shell (see run.sh).
#   taskset -c 2 julia -t 1 --project=bench bench/mkl_amd/check.jl
include(joinpath(@__DIR__, "..", "harness.jl"))
using .Harness
using MKL
using LinearAlgebra

BLAS.set_num_threads(1)
println("backend          : ", BLAS.get_config())
println("MKL_DEBUG_CPU_TYPE      = ", get(ENV, "MKL_DEBUG_CPU_TYPE", "<unset>"))
println("MKL_ENABLE_INSTRUCTIONS = ", get(ENV, "MKL_ENABLE_INSTRUCTIONS", "<unset>"))
println("LD_PRELOAD              = ", get(ENV, "LD_PRELOAD", "<unset>"))
println("-"^60)

for n in (256, 512)
    Aspd = let M = randn(n, n); Matrix(M'M + n * I); end
    s = copy(Aspd); sym = Symmetric(s, :L)
    fchol = @noinline () -> (copyto!(s, Aspd); cholesky!(sym); GC.gc(false); s[1])
    Agen = randn(n, n); sg = copy(Agen)
    fqr = @noinline () -> (copyto!(sg, Agen); qr!(sg); GC.gc(false); sg[1])
    for (name, f) in (("cholesky", fchol), ("qr", fqr))
        f()
        GC.enable(false); GC.gc(false)
        med, relσ = time_median_sigma(f; seconds = 2.0)
        GC.enable(true); GC.gc()
        println(rpad("$name n=$n", 16), "  median = ", round(Int, 1e9 * med), " ns   σ=",
            round(100relσ, digits = 1), "%")
    end
end
