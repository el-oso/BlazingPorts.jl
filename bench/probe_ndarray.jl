# Probe: ndarray (Tier 3) — fused broadcast & strided view reduction.
#
# Contenders: Julia Base vs ndarray (Rust cdylib).
#
# Op 1: Fused broadcast — D .= A .* B .+ c (N=1_000_000 f64)
#   Julia: preallocated D, fused dot-broadcast D .= A .* B .+ c
#   Rust:  ndarray element-wise loop into preallocated out
#
# Op 2: Strided reduction — sum every k-th element of a large vector (stride=7, N=1_000_000)
#   Julia: @view(A[1:k:end]) |> sum
#   Rust:  ndarray slice with step, then sum
#
# Expected: Base ties/beats ndarray (Julia's home turf) → document-skip.
#
# Run:  taskset -c 2 julia -t 1 --project=bench bench/probe_ndarray.jl
# Build Rust lib first:  bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

const N       = 1_000_000
const C_SCALAR = 3.14159
const STRIDE   = 7

# Preallocated arrays (no alloc in timed region)
const vec_A = rand(Float64, N)
const vec_B = rand(Float64, N)
const vec_D_jl  = zeros(Float64, N)
const vec_D_rs  = zeros(Float64, N)

# ─────────────────────────────────────────────────────────────────────────────
# Op 1: Fused broadcast D = A * B + c
# ─────────────────────────────────────────────────────────────────────────────

@noinline function jl_fused_broadcast!()
    @. vec_D_jl = vec_A * vec_B + C_SCALAR
    return vec_D_jl[1]
end

@noinline function rust_fused_broadcast!()
    ccall((:ndarray_fused_broadcast, LIB), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64}, Csize_t),
        vec_A, vec_B, C_SCALAR, vec_D_rs, N)
    return vec_D_rs[1]
end

# Sanity: outputs agree (rtol 1e-12)
jl_fused_broadcast!(); rust_fused_broadcast!()
@assert isapprox(vec_D_jl, vec_D_rs; rtol=1e-12) "fused broadcast outputs disagree!"
println("fused broadcast sanity: OK")

# ─────────────────────────────────────────────────────────────────────────────
# Op 2: Strided reduction
# ─────────────────────────────────────────────────────────────────────────────

@noinline function jl_strided_sum()
    return sum(@view(vec_A[1:STRIDE:end]))
end

@noinline function rust_strided_sum()
    return ccall((:ndarray_strided_sum, LIB), Float64,
        (Ptr{Float64}, Csize_t, Csize_t),
        vec_A, N, STRIDE)
end

# Sanity: agree to rtol 1e-12
s_jl = jl_strided_sum(); s_rs = rust_strided_sum()
@assert isapprox(s_jl, s_rs; rtol=1e-12) "strided sum disagrees: Julia=$s_jl, Rust=$s_rs"
println("strided sum sanity: OK  (Julia=$s_jl, Rust=$s_rs)\n")

# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

println("=== ndarray_fused_broadcast (N=$N) ===")
bc_probes = Probe[
    run_probe("Julia/Base", jl_fused_broadcast!; seconds=3.0),
    run_probe("rust/ndarray", rust_fused_broadcast!; seconds=3.0),
]
report("ndarray_fused_broadcast", bc_probes; rust_label="rust/ndarray")
save_probes("ndarray_fused_broadcast", bc_probes)
out = plot_probe("ndarray_fused_broadcast", bc_probes)
println("plot → $out\n")

println("=== ndarray_strided_sum (N=$N, stride=$STRIDE) ===")
sr_probes = Probe[
    run_probe("Julia/Base", jl_strided_sum; seconds=3.0),
    run_probe("rust/ndarray", rust_strided_sum; seconds=3.0),
]
report("ndarray_strided_sum", sr_probes; rust_label="rust/ndarray")
save_probes("ndarray_strided_sum", sr_probes)
out = plot_probe("ndarray_strided_sum", sr_probes)
println("plot → $out\n")

println("Done — probe_ndarray.jl")
