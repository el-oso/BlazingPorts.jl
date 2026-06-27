# Grouped VIOLIN comparison plots from the SAVED sample distributions in
# bench/results/compare_factorizations.json — NO benchmarking (regenerate any time, or after restyling).
# Populate/refresh the data first with:  …julia --project=bench bench/compare_factorizations.jl
#
#   julia --project=bench bench/plot_faer_compare.jl
#
# One plot per factorization (QR, Cholesky): x grouped by matrix size, three violins per group
# (OpenBLAS · faer · ours) of the per-sample GFLOP/s distribution (higher = better, so "ours" sits on top).
# The ours/faer median ratio is annotated over each group.
using JSON, StatsPlots, Printf, Statistics
using StatsPlots: mm          # margin unit (not exported by default)

const HERE = @__DIR__
const DATA = joinpath(HERE, "results", "compare_factorizations.json")
const ASSETS = (joinpath(HERE, "..", "docs", "assets"), joinpath(HERE, "..", "docs", "src", "assets"))

flop(kind, n) = kind == :chol ? n^3 / 3 : 4 * n^3 / 3
const ORDER = ["openblas", "faer", "ours"]
const NICE  = Dict("openblas" => "OpenBLAS", "faer" => "faer (Rust)", "ours" => "Factorizations.jl (ours)")
const COLOR = Dict("openblas" => :gray70, "faer" => :darkorange, "ours" => :seagreen)
const OFF   = Dict("openblas" => -0.26, "faer" => 0.0, "ours" => 0.26)   # in-group x offsets

function plot_algo(label, kind, rows)
    sizes = sort(parse.(Int, collect(keys(rows))))
    plt = plot(; legend = :topleft, xlabel = "matrix size n", ylabel = "GFLOP/s  (higher = better)",
        title = "$label — single-thread, per-sample distribution", titlefontsize = 11,
        framestyle = :box, dpi = 200, size = (820, 500),
        left_margin = 8mm, bottom_margin = 5mm, top_margin = 3mm, right_margin = 3mm,
        xticks = (1:length(sizes), string.(sizes)))
    ymax = 0.0
    for (gi, n) in enumerate(sizes)            # one group per size
        cell = rows[string(n)]
        for c in ORDER
            haskey(cell, c) || continue
            g = flop(kind, n) ./ Float64.(cell[c]["samples"]) ./ 1e9   # GFLOP/s per sample
            ymax = max(ymax, quantile(g, 0.98))
            violin!(plt, fill(gi + OFF[c], length(g)), g; bar_width = 0.24, alpha = 0.65,
                color = COLOR[c], linewidth = 0, label = gi == 1 ? NICE[c] : "")
        end
    end
    ylims!(plt, 0, ymax * 1.12)
    for (gi, n) in enumerate(sizes)                        # annotate ours/faer median ratio over each group
        cell = rows[string(n)]
        (haskey(cell, "faer") && haskey(cell, "ours")) || continue
        gf = flop(kind, n) / cell["faer"]["median_s"] / 1e9; go = flop(kind, n) / cell["ours"]["median_s"] / 1e9
        annotate!(plt, gi + 0.26, go * 1.05, text(@sprintf("%.2f×", go / gf), 8, :seagreen, :bottom))
    end
    outs = [joinpath(d, "$(lowercase(label))_comparison.png") for d in ASSETS]
    for o in outs; mkpath(dirname(o)); savefig(plt, o); end
    println("wrote ", join(outs, ", "))
end

isfile(DATA) || error("no data at $DATA — run bench/compare_factorizations.jl first")
d = JSON.parsefile(DATA)
haskey(d, "Cholesky") && plot_algo("Cholesky", :chol, d["Cholesky"])
haskey(d, "QR")       && plot_algo("QR",       :qr,   d["QR"])
