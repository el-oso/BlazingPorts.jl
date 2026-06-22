# Force the OLD MKL 2020.0 (predates MKL_DEBUG_CPU_TYPE removal) via MKL_jll, forwarded into LBT
# manually (MKL.jl won't wire an MKL this old). Test whether MKL_DEBUG_CPU_TYPE=5 unlocks the AVX2
# path on this Zen5. Run under the /tmp/mkl2020 env:
#   MKL_DEBUG_CPU_TYPE=5 MKL_VERBOSE=1 taskset -c 2 julia -t 1 --project=/tmp/mkl2020 \
#       /home/el_oso/Documents/claude/BlazingPorts.jl/bench/mkl_amd/check2020.jl
using MKL_jll, LinearAlgebra
using Chairmarks: @be

const RT = joinpath(MKL_jll.artifact_dir, "lib", "libmkl_rt.so")  # full path (old MKL_jll gives bare name)
BLAS.lbt_forward(RT; clear = true)
BLAS.set_num_threads(1)
println("config: ", BLAS.get_config())
println("MKL_DEBUG_CPU_TYPE = ", get(ENV, "MKL_DEBUG_CPU_TYPE", "<unset>"))
println("-"^60)

med(f) = (f(); GC.enable(false); GC.gc(false);
    b = @be f seconds = 2; GC.enable(true); GC.gc();
    ts = sort(Float64[x.time for x in b.samples]); ts[(length(ts)+1)÷2])

for n in (256, 512)
    Aspd = Matrix(let M = randn(n, n); M'M + n * I end)
    s = copy(Aspd); sym = Symmetric(s, :L)
    fchol = @noinline () -> (copyto!(s, Aspd); cholesky!(sym); GC.gc(false); s[1])
    Agen = randn(n, n); sg = copy(Agen)
    fqr = @noinline () -> (copyto!(sg, Agen); qr!(sg); GC.gc(false); sg[1])
    # correctness
    @assert cholesky(Aspd).L * cholesky(Aspd).U ≈ Aspd rtol = 1e-8
    println("cholesky n=$n  median = ", round(Int, 1e9 * med(fchol)), " ns")
    println("qr       n=$n  median = ", round(Int, 1e9 * med(fqr)), " ns")
end
