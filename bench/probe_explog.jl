# Probe: exp & log (Tier 1 libm/Base openlibm).
# Contenders:
#   Base exp / log (openlibm)  vs  Rust libm bp_exp_array / bp_log_array
# Workload: 1024 Float64 element-wise into a preallocated output vector.
# Expected: document-skip (Base ≥ Rust libm single-threaded).
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_explog.jl
# Build Rust lib first: bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

const N = 1024

# ── exp probe ────────────────────────────────────────────────────────────────
const exp_x      = collect(range(-10.0, 10.0; length = N))
const exp_out_jl = zeros(Float64, N)
const exp_out_rs = zeros(Float64, N)

@noinline function base_exp_batch!()
    @inbounds for i in eachindex(exp_x)
        exp_out_jl[i] = exp(exp_x[i])
    end
    return exp_out_jl[1]
end

@noinline function rust_exp_batch!()
    ccall((:bp_exp_array, LIB), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Csize_t),
        exp_x, exp_out_rs, N)
    return exp_out_rs[1]
end

base_exp_batch!(); rust_exp_batch!()
@assert exp_out_jl ≈ exp_out_rs atol=1e-12 "exp outputs disagree!"

println("\n=== exp (N=$N) ===")
exp_probes = Probe[
    run_probe("Base", base_exp_batch!),
    run_probe("rust", rust_exp_batch!),
]
report("explog_exp", exp_probes; rust_label = "rust")
save_probes("explog_exp", exp_probes)
plot_probe("explog_exp", exp_probes)

# ── log probe ────────────────────────────────────────────────────────────────
const log_x      = collect(range(0.1, 100.0; length = N))
const log_out_jl = zeros(Float64, N)
const log_out_rs = zeros(Float64, N)

@noinline function base_log_batch!()
    @inbounds for i in eachindex(log_x)
        log_out_jl[i] = log(log_x[i])
    end
    return log_out_jl[1]
end

@noinline function rust_log_batch!()
    ccall((:bp_log_array, LIB), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Csize_t),
        log_x, log_out_rs, N)
    return log_out_rs[1]
end

base_log_batch!(); rust_log_batch!()
@assert log_out_jl ≈ log_out_rs atol=1e-12 "log outputs disagree!"

println("\n=== log (N=$N) ===")
log_probes = Probe[
    run_probe("Base", base_log_batch!),
    run_probe("rust", rust_log_batch!),
]
report("explog_log", log_probes; rust_label = "rust")
save_probes("explog_log", log_probes)
plot_probe("explog_log", log_probes)

println("\nDone — probe_explog.jl")
