# Shared probe harness for BlazingPorts. Compares a Julia implementation against the real Rust crate
# (via the vendored cdylib, ccall — see bench/rust_compare/) and against the current-Julia baseline
# (Base / stdlib / ecosystem). Methodology (locked, see blazingly-fast-rust-crates.md plan):
#
#   * SINGLE-THREADED both sides — instruction-level comparison, not parallelism. Run Julia with
#     `-t 1`, core-pin (`taskset -c N julia ...`), `BLAS.set_num_threads(1)`, and build the Rust
#     cdylib with parallelism disabled (RAYON_NUM_THREADS=1).
#   * Chairmarks.@be with ≥1000 samples; report MEDIAN and relative-σ. Warn if σ exceeds the floor.
#   * Parity gate: julia_median / rust_median ≥ 0.96.
#   * Plot with Plots.jl into docs/assets/<crate>.png.
#
# Run a probe (example): taskset -c 2 julia -t 1 --project=bench bench/probe_<crate>.jl

module Harness

using Chairmarks: @be
using Printf
using LinearAlgebra: BLAS
import Plots

const PARITY_GATE = 0.96
const MIN_SAMPLES = 1000
const SIGMA_FLOOR = 0.05  # rel-σ above this → noisy run, widen samples / pin CPU

export Probe, time_median_sigma, parity, report, single_thread!, RUST_LIB, plot_probe

"""Force single-threaded execution for a fair instruction-level comparison."""
function single_thread!()
    BLAS.set_num_threads(1)
    if Threads.nthreads() > 1
        @warn "Julia started with $(Threads.nthreads()) threads; launch with `-t 1` for clean single-thread probes."
    end
    return nothing
end

"""
    time_median_sigma(f; min_samples=MIN_SAMPLES, seconds=2.0) -> (median_seconds, rel_sigma)

Time `f()` with Chairmarks under a wall-clock budget, returning the median per-evaluation time and
relative σ over **at least** `min_samples` samples (the budget is extended if the first pass falls
short — Chairmarks auto-tunes `evals` per sample, amortising timer overhead for ns-scale kernels).
`f` should be a zero-arg `@noinline` wrapper over preallocated concrete inputs (no allocation in the
timed region) with a DCE sink.
"""
function time_median_sigma(f; min_samples::Int = MIN_SAMPLES, seconds::Float64 = 2.0)
    @assert min_samples ≥ MIN_SAMPLES "probes must take ≥ $MIN_SAMPLES samples (asked $min_samples)"
    b = @be f seconds = seconds
    if length(b.samples) < min_samples
        factor = cld(min_samples, max(1, length(b.samples))) * 1.2
        b = @be f seconds = seconds * factor
    end
    ts = sort!(Float64[s.time for s in b.samples])
    length(ts) ≥ min_samples || @warn "collected only $(length(ts)) samples (< $min_samples); increase `seconds`"
    med = ts[(length(ts) + 1) ÷ 2]
    mean = sum(ts) / length(ts)
    sd = length(ts) > 1 ? sqrt(sum(abs2, ts .- mean) / (length(ts) - 1)) : 0.0
    relσ = med > 0 ? sd / med : 0.0
    relσ > SIGMA_FLOOR && @warn @sprintf("noisy probe: rel-σ=%.1f%% > floor %.0f%% — pin CPU / add samples", 100relσ, 100SIGMA_FLOOR)
    return (med, relσ)
end

"""parity(julia_median, rust_median) → ratio (≥ PARITY_GATE means Julia is 'good enough')."""
parity(julia_med, rust_med) = rust_med / julia_med  # ≥ 0.96 ⇒ Julia within 4% of (or beats) Rust

"A single timed contender in a probe."
struct Probe
    label::String
    median::Float64   # seconds
    relσ::Float64
end

"""Print a probe table and the parity verdict against the named Rust baseline."""
function report(crate::AbstractString, probes::Vector{Probe}; rust_label::AbstractString = "rust")
    println("── probe: $crate ", "─"^max(0, 40 - length(crate)))
    @printf("  %-22s %12s  %8s\n", "contender", "median", "rel-σ")
    for p in probes
        @printf("  %-22s %10.2f ns  %6.1f%%\n", p.label, 1e9 * p.median, 100p.relσ)
    end
    ri = findfirst(p -> p.label == rust_label, probes)
    if !isnothing(ri)
        rust = probes[ri].median
        println("  vs $rust_label (parity = rust/julia ≥ $PARITY_GATE ⇒ Julia good enough):")
        for p in probes
            p.label == rust_label && continue
            r = parity(p.median, rust)
            verdict = r ≥ PARITY_GATE ? "GOOD ENOUGH" : "BELOW GATE → implement"
            @printf("    %-22s %6.2f×   %s\n", p.label, r, verdict)
        end
    end
    return nothing
end

"""Plot the probe medians as a bar chart into docs/assets/<crate>.png (Plots.jl loaded lazily)."""
function plot_probe(crate::AbstractString, probes::Vector{Probe}; dir = joinpath(@__DIR__, "..", "docs", "assets"))
    labels = [p.label for p in probes]
    meds = [1e9 * p.median for p in probes]
    plt = Plots.bar(labels, meds; legend = false, ylabel = "median (ns, lower=better)",
        title = crate, xrotation = 30)
    mkpath(dir)
    out = joinpath(dir, "$crate.png")
    Plots.savefig(plt, out)
    return out
end

# Path to the vendored Rust cdylib (built by bench/rust_compare/build.sh).
const RUST_LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")

end # module Harness
