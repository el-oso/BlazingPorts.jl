# Probe: rand + rand_distr (Tier 4 — stochastic/PRNG).
# Contenders:
#   Julia Base Random stdlib (Xoshiro256++ default RNG) vs Rust rand crate (SmallRng = Xoshiro256++)
# Workload: fill a preallocated Float64 vector of length N = 1_000_000 with:
#   1. rand_uniform  — uniform [0,1)     — Base rand!  vs  rand_uniform_fill
#   2. rand_normal   — standard normal    — Base randn! vs  rand_normal_fill
#   3. rand_exp      — Exp(1)             — Base randexp! vs rand_exp_fill
#
# PRNG algorithm note: Both sides use Xoshiro256++ (same algorithm family).
#   Julia: Base.Random.Xoshiro (default RNG since 1.7)
#   Rust:  SmallRng = Xoshiro256++ (via rand 0.9's small_rng feature)
# This is an apples-to-apples PRNG comparison.
#
# σ-discipline: no alloc in timed region (preallocate buf; reuse one RNG object;
#   rand!/randn!/randexp! are in-place). Expect Julia GC-free → tight σ.
#
# Correctness: streams differ (seeded differently) — sanity-check distribution
#   statistics only: uniform mean≈0.5, normal mean≈0 var≈1, exp mean≈1.
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_rand.jl
# Build Rust lib first: bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Random: Xoshiro, rand!, randn!, randexp!

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

const N = 1_000_000

# ── preallocated buffers (no alloc in timed region) ──────────────────────────
const buf_jl = zeros(Float64, N)
const buf_rs = zeros(Float64, N)

# ── reused RNG objects (never reseeded in timed region) ──────────────────────
const jl_rng = Xoshiro(0x1234_5678_abcd_ef01)   # same seed as Rust side

# ── Rust ccall wrappers ───────────────────────────────────────────────────────
@noinline function rust_uniform!()
    ccall((:rand_uniform_fill, LIB), Cvoid, (Ptr{Float64}, Csize_t), buf_rs, N)
    return buf_rs[1]   # DCE sink
end

@noinline function rust_normal!()
    ccall((:rand_normal_fill, LIB), Cvoid, (Ptr{Float64}, Csize_t), buf_rs, N)
    return buf_rs[1]
end

@noinline function rust_exp!()
    ccall((:rand_exp_fill, LIB), Cvoid, (Ptr{Float64}, Csize_t), buf_rs, N)
    return buf_rs[1]
end

# ── Julia Base stdlib wrappers ────────────────────────────────────────────────
@noinline function jl_uniform!()
    rand!(jl_rng, buf_jl)
    return buf_jl[1]
end

@noinline function jl_normal!()
    randn!(jl_rng, buf_jl)
    return buf_jl[1]
end

@noinline function jl_exp!()
    randexp!(jl_rng, buf_jl)
    return buf_jl[1]
end

# ── warm-up (JIT compile + prime caches) ─────────────────────────────────────
jl_uniform!(); rust_uniform!()
jl_normal!();  rust_normal!()
jl_exp!();     rust_exp!()

# ── distribution sanity checks ────────────────────────────────────────────────
function check_stats(buf, label; mean_lo, mean_hi, var_lo=nothing, var_hi=nothing)
    m = sum(buf) / length(buf)
    @assert mean_lo < m < mean_hi "$(label) mean $(m) out of range [$(mean_lo), $(mean_hi)]"
    if !isnothing(var_lo)
        v = sum(x -> (x-m)^2, buf) / (length(buf)-1)
        @assert var_lo < v < var_hi "$(label) variance $(v) out of range [$(var_lo), $(var_hi)]"
    end
    println("  sanity OK: $label mean=$(round(m,digits=4))")
end

println("\n=== Distribution sanity checks ===")
jl_uniform!()
check_stats(buf_jl, "Julia uniform"; mean_lo=0.48, mean_hi=0.52)
rust_uniform!()
check_stats(buf_rs, "Rust uniform";  mean_lo=0.48, mean_hi=0.52)

jl_normal!()
check_stats(buf_jl, "Julia normal";  mean_lo=-0.01, mean_hi=0.01, var_lo=0.98, var_hi=1.02)
rust_normal!()
check_stats(buf_rs, "Rust normal";   mean_lo=-0.01, mean_hi=0.01, var_lo=0.98, var_hi=1.02)

jl_exp!()
check_stats(buf_jl, "Julia exp";     mean_lo=0.98, mean_hi=1.02)
rust_exp!()
check_stats(buf_rs, "Rust exp";      mean_lo=0.98, mean_hi=1.02)

# ── probe 1: rand_uniform ─────────────────────────────────────────────────────
println("\n=== rand_uniform (N=$N, Xoshiro256++ both sides) ===")
uniform_probes = Probe[
    run_probe("Base (Xoshiro)", jl_uniform!; seconds=3.0),
    run_probe("rust SmallRng",  rust_uniform!; seconds=3.0),
]
report("rand_uniform", uniform_probes; rust_label="rust SmallRng")
save_probes("rand_uniform", uniform_probes)
plot_probe("rand_uniform", uniform_probes)

# ── probe 2: rand_normal ──────────────────────────────────────────────────────
println("\n=== rand_normal (N=$N, Xoshiro256++ both sides) ===")
normal_probes = Probe[
    run_probe("Base (Xoshiro)", jl_normal!; seconds=3.0),
    run_probe("rust SmallRng",  rust_normal!; seconds=3.0),
]
report("rand_normal", normal_probes; rust_label="rust SmallRng")
save_probes("rand_normal", normal_probes)
plot_probe("rand_normal", normal_probes)

# ── probe 3: rand_exp ─────────────────────────────────────────────────────────
println("\n=== rand_exp (N=$N, Xoshiro256++ both sides) ===")
exp_probes = Probe[
    run_probe("Base (Xoshiro)", jl_exp!; seconds=3.0),
    run_probe("rust SmallRng",  rust_exp!; seconds=3.0),
]
report("rand_exp", exp_probes; rust_label="rust SmallRng")
save_probes("rand_exp", exp_probes)
plot_probe("rand_exp", exp_probes)

println("\nDone — probe_rand.jl")
