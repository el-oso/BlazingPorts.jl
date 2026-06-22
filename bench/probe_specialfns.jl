# Probe: special functions — erf & gamma (Tier 1 libm/statrs).
# Contenders:
#   SpecialFunctions.erf / .gamma  vs  Rust libm bp_erf_array / bp_gamma_array
# Workload: 1024 Float64 element-wise into a preallocated output vector.
# Expected: document-skip (SpecialFunctions.jl ≥ Rust libm for both).
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_specialfns.jl
# Build Rust lib first: bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using SpecialFunctions: erf, gamma

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

const N = 1024

# ── erf probe ────────────────────────────────────────────────────────────────
const erf_x   = collect(range(-3.0, 3.0; length = N))
const erf_out_jl   = zeros(Float64, N)
const erf_out_rs   = zeros(Float64, N)

@noinline function sf_erf_batch!()
    @inbounds for i in eachindex(erf_x)
        erf_out_jl[i] = erf(erf_x[i])
    end
    return erf_out_jl[1]   # DCE sink
end

@noinline function rust_erf_batch!()
    ccall((:bp_erf_array, LIB), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Csize_t),
        erf_x, erf_out_rs, N)
    return erf_out_rs[1]
end

# sanity check
sf_erf_batch!(); rust_erf_batch!()
@assert erf_out_jl ≈ erf_out_rs atol=1e-12 "erf outputs disagree!"

println("\n=== erf (N=$N) ===")
erf_probes = Probe[
    run_probe("SpecialFunctions", sf_erf_batch!),
    run_probe("rust", rust_erf_batch!),
]
report("specialfns_erf", erf_probes; rust_label = "rust")
save_probes("specialfns_erf", erf_probes)
plot_probe("specialfns_erf", erf_probes)

# ── gamma probe ───────────────────────────────────────────────────────────────
const gamma_x      = collect(range(0.5, 10.0; length = N))
const gamma_out_jl = zeros(Float64, N)
const gamma_out_rs = zeros(Float64, N)

@noinline function sf_gamma_batch!()
    @inbounds for i in eachindex(gamma_x)
        gamma_out_jl[i] = gamma(gamma_x[i])
    end
    return gamma_out_jl[1]
end

@noinline function rust_gamma_batch!()
    ccall((:bp_gamma_array, LIB), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Csize_t),
        gamma_x, gamma_out_rs, N)
    return gamma_out_rs[1]
end

sf_gamma_batch!(); rust_gamma_batch!()
@assert gamma_out_jl ≈ gamma_out_rs atol=1e-12 "gamma outputs disagree!"

println("\n=== gamma (N=$N) ===")
gamma_probes = Probe[
    run_probe("SpecialFunctions", sf_gamma_batch!),
    run_probe("rust", rust_gamma_batch!),
]
report("specialfns_gamma", gamma_probes; rust_label = "rust")
save_probes("specialfns_gamma", gamma_probes)
plot_probe("specialfns_gamma", gamma_probes)

println("\nDone — probe_specialfns.jl")
