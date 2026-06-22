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
using Statistics: quantile
using StatsPlots  # re-exports Plots; provides violin / boxplot / dotplot
import JSON

const PARITY_GATE = 0.96
const MIN_SAMPLES = 1000
const SIGMA_FLOOR = 0.05  # rel-σ above this → noisy run, widen samples / pin CPU
const MAX_STORE_SAMPLES = 2000  # cap stored/plotted points (stats use ALL collected samples)

export Probe, run_probe, time_median_sigma, parity, report, single_thread!, RUST_LIB, plot_probe
export save_probes, load_probes, replot, RESULTS_DIR

"""Force single-threaded execution for a fair instruction-level comparison."""
function single_thread!()
    BLAS.set_num_threads(1)
    if Threads.nthreads() > 1
        @warn "Julia started with $(Threads.nthreads()) threads; launch with `-t 1` for clean single-thread probes."
    end
    return nothing
end

"""
    collect_samples(f; min_samples=MIN_SAMPLES, seconds=2.0) -> Vector{Float64}

Time `f()` with Chairmarks under a wall-clock budget; return the **full** vector of per-evaluation
times (seconds), at least `min_samples` long (the budget is extended if the first pass falls short —
Chairmarks auto-tunes `evals` per sample, amortising timer overhead for ns-scale kernels). `f` should
be a zero-arg `@noinline` wrapper over preallocated concrete inputs (no allocation in the timed
region) with a DCE sink. The whole distribution is kept so probes can be drawn as violin/box plots.
"""
function collect_samples(f; min_samples::Int = MIN_SAMPLES, seconds::Float64 = 2.0)
    @assert min_samples ≥ MIN_SAMPLES "probes must take ≥ $MIN_SAMPLES samples (asked $min_samples)"
    b = @be f seconds = seconds
    if length(b.samples) < min_samples
        factor = cld(min_samples, max(1, length(b.samples))) * 1.2
        b = @be f seconds = seconds * factor
    end
    ts = Float64[s.time for s in b.samples]
    length(ts) ≥ min_samples || @warn "collected only $(length(ts)) samples (< $min_samples); increase `seconds`"
    return ts
end

"Median and relative σ of a sample vector (seconds)."
function median_relσ(ts::AbstractVector{<:Real})
    s = sort(ts)
    med = s[(length(s) + 1) ÷ 2]
    mean = sum(s) / length(s)
    sd = length(s) > 1 ? sqrt(sum(abs2, s .- mean) / (length(s) - 1)) : 0.0
    relσ = med > 0 ? sd / med : 0.0
    return (med, relσ)
end

"""
    time_median_sigma(f; kwargs...) -> (median_seconds, rel_sigma)

Convenience: median + relative σ for `f` (see [`collect_samples`](@ref)). Warns if rel-σ exceeds the
noise floor. Prefer [`run_probe`](@ref) when you want to plot the distribution.
"""
function time_median_sigma(f; kwargs...)
    ts = collect_samples(f; kwargs...)
    med, relσ = median_relσ(ts)
    relσ > SIGMA_FLOOR && @warn @sprintf("noisy probe: rel-σ=%.1f%% > floor %.0f%% — pin CPU / add samples", 100relσ, 100SIGMA_FLOOR)
    return (med, relσ)
end

"""parity(julia_median, rust_median) → ratio (≥ PARITY_GATE means Julia is 'good enough')."""
parity(julia_med, rust_med) = rust_med / julia_med  # ≥ 0.96 ⇒ Julia within 4% of (or beats) Rust

"A single timed contender in a probe — keeps the full sample distribution for violin/box plots."
struct Probe
    label::String
    median::Float64           # seconds
    relσ::Float64
    samples::Vector{Float64}  # per-evaluation times, seconds
end

"""
    run_probe(label, f; min_samples=MIN_SAMPLES, seconds=2.0) -> Probe

Time contender `f` and return a [`Probe`](@ref) carrying its median, rel-σ, and full sample vector.
Warns if rel-σ exceeds the noise floor.
"""
function run_probe(label::AbstractString, f; kwargs...)
    ts = collect_samples(f; kwargs...)
    med, relσ = median_relσ(ts)           # stats from the FULL sample set
    relσ > SIGMA_FLOOR && @warn @sprintf("noisy probe %s: rel-σ=%.1f%% > floor %.0f%% — pin CPU / add samples", label, 100relσ, 100SIGMA_FLOOR)
    return Probe(label, med, relσ, _subsample(ts, MAX_STORE_SAMPLES))
end

"Even quantile-strided subsample of `ts` (≤ `k` points) that preserves distribution shape for plots."
function _subsample(ts::AbstractVector{<:Real}, k::Int)
    length(ts) ≤ k && return Float64.(ts)
    s = sort(ts)
    idx = unique(round.(Int, range(1, length(s); length = k)))
    return Float64.(s[idx])
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

"""
    plot_probe(crate, probes; kind=:violin) -> path

Plot each contender's **full timing distribution** into docs/assets/<crate>.png. `kind` is `:violin`
(violin + overlaid box, default) or `:box` (box only). The y-axis is per-evaluation time in ns
(lower = better), clipped to the 99th percentile so the heavy right tail (GC / scheduling spikes)
doesn't squash the bulk; the median is annotated per contender.
"""
function plot_probe(crate::AbstractString, probes::Vector{Probe};
        kind::Symbol = :violin, dir = joinpath(@__DIR__, "..", "docs", "assets"))
    # Explicit numeric x-positions (in probes order) so violin/box and tick labels stay aligned —
    # StatsPlots would otherwise sort string categories alphabetically and misplace annotations.
    xs = Float64[]
    ys = Float64[]
    for (i, p) in enumerate(probes)
        append!(xs, fill(float(i), length(p.samples)))
        append!(ys, (1e9) .* p.samples)
    end
    ylim_hi = isempty(ys) ? 1.0 : quantile(ys, 0.99)
    ylab = "per-eval time (ns, lower = better)"
    # fold the median into each tick label so there's no overlap with the distribution shapes
    ticklabels = [@sprintf("%s\n%.0f ns", p.label, 1e9 * p.median) for p in probes]
    xt = (collect(1:length(probes)), ticklabels)
    if kind === :box
        plt = boxplot(xs, ys; legend = false, ylabel = ylab, title = crate,
            xticks = xt, outliers = false, ylims = (0, ylim_hi))
    else
        plt = violin(xs, ys; legend = false, ylabel = ylab, title = crate,
            xticks = xt, alpha = 0.6, ylims = (0, ylim_hi))
        boxplot!(plt, xs, ys; legend = false, fillalpha = 0.0, linewidth = 1.2,
            outliers = false, bar_width = 0.25)
    end
    mkpath(dir)
    out = joinpath(dir, "$crate.png")
    savefig(plt, out)
    return out
end

# ── persistence: save raw sample points so plots can be regenerated without re-benchmarking ───────

"Directory where probe sample distributions are cached as JSON."
const RESULTS_DIR = joinpath(@__DIR__, "results")

"""
    save_probes(crate, probes; dir=RESULTS_DIR) -> path

Persist a probe's **full sample distributions** (every per-eval time, in seconds) plus median/rel-σ
to `results/<crate>.json`, so `replot`/`plot_probe` can redraw later without re-running the benchmark.
"""
function save_probes(crate::AbstractString, probes::Vector{Probe}; dir = RESULTS_DIR)
    mkpath(dir)
    data = Dict(
        "crate" => crate,
        "saved_unix" => time(),
        "unit" => "seconds",
        "min_samples" => MIN_SAMPLES,
        "contenders" => [Dict("label" => p.label, "median" => p.median,
            "rel_sigma" => p.relσ, "samples" => p.samples) for p in probes],
    )
    out = joinpath(dir, "$crate.json")
    open(io -> JSON.print(io, data), out, "w")
    return out
end

"""
    load_probes(crate; dir=RESULTS_DIR) -> Vector{Probe}

Reconstruct probes (with full samples) from `results/<crate>.json`.
"""
function load_probes(crate::AbstractString; dir = RESULTS_DIR)
    f = joinpath(dir, "$crate.json")
    isfile(f) || error("no saved probe data at $f — run the probe first")
    d = JSON.parsefile(f)
    return Probe[Probe(c["label"], Float64(c["median"]), Float64(c["rel_sigma"]),
        Float64.(c["samples"])) for c in d["contenders"]]
end

"""
    replot(; results_dir=RESULTS_DIR, kind=:violin, kwargs...) -> Vector{String}

Regenerate **every** saved probe's plot from cached JSON — no benchmarking. Use after changing plot
style (e.g. `replot(kind=:box)`).
"""
function replot(; results_dir = RESULTS_DIR, kind::Symbol = :violin, kwargs...)
    outs = String[]
    for f in sort(readdir(results_dir; join = true))
        endswith(f, ".json") || continue
        crate = splitext(basename(f))[1]
        push!(outs, plot_probe(crate, load_probes(crate; dir = results_dir); kind = kind, kwargs...))
    end
    return outs
end

# Path to the vendored Rust cdylib (built by bench/rust_compare/build.sh).
const RUST_LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")

end # module Harness
