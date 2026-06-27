# Combined "byte-ops / shuffle-SIMD family" GAP plot — VIOLIN of the saved per-sample distributions
# (bytecount + transcode + lexical), Julia stdlib vs the Rust SIMD crate. No re-benchmark.
# Regenerate:  julia --project=bench bench/plot_byteops.jl
using JSON, StatsPlots, Printf, Statistics
using StatsPlots: mm          # margin unit (not exported by default)
const HERE = @__DIR__
const MiB = 1024 * 1024
load(c) = Dict(x["label"] => x for x in JSON.parsefile(joinpath(HERE, "results", "$c.json"))["contenders"])
bc, tc, lx = load("bytecount"), load("transcode"), load("lexical")
gbs(c, nb) = nb ./ Float64.(c["samples"]) ./ 1e9
med(c, nb) = nb / c["median"] / 1e9

# (name, julia-contender, rust-contender, source-dict, nb) — ordered by widening gap
ops = [("bytecount",     bc["Julia count(==(b))"],   bc["bytecount (SIMD)"], 16MiB),
       ("hex encode",    tc["Julia hex"],            tc["Rust SIMD hex"],    12MiB),
       ("float parse",   lx["Julia parse(Float64)"], lx["lexical-core"],     12.09MiB),
       ("base64 encode", tc["Julia base64"],         tc["Rust SIMD base64"], 12MiB)]
roles = [(:seagreen, "Julia stdlib"), (:slateblue, "Rust SIMD crate")]
step = 3

p = plot(; ylabel = "GB/s  (log, higher = better)", yscale = :log10, ylims = (0.1, 80),
    title = "Byte-ops: Julia stdlib vs Rust SIMD (single-thread) — the shuffle/lookup-SIMD gap class",
    titlefontsize = 10, legend = :topright, framestyle = :box, dpi = 200, size = (1000, 560),
    left_margin = 8mm, bottom_margin = 5mm, top_margin = 3mm, right_margin = 3mm)
for (g, (name, jc, rc, nb)) in enumerate(ops)
    for (c, cc) in enumerate((jc, rc))
        x = (g - 1) * step + c
        gv = gbs(cc, nb); lo, hi = quantile(gv, 0.02), quantile(gv, 0.98); gv = filter(v -> lo <= v <= hi, gv)
        violin!(p, fill(x, length(gv)), gv; color = roles[c][1], alpha = 0.65, linewidth = 0, bar_width = 0.7,
            label = (g == 1 ? roles[c][2] : ""))
        annotate!(p, x, med(cc, nb) * 1.25, text(@sprintf("%.2f", med(cc, nb)), 7, :center))
    end
    r = med(jc, nb) / med(rc, nb)
    annotate!(p, (g - 1) * step + 1.5, 0.12, text(r > 0.8 ? "parity" : @sprintf("%.0f× gap", 1 / r), 8, :center))
end
plot!(p; xticks = ([(g - 1) * step + 1.5 for g in 1:length(ops)], [o[1] for o in ops]),
      xlims = (0.3, length(ops) * step - 0.7))
for dir in (joinpath(HERE, "..", "docs", "assets"), joinpath(HERE, "..", "docs", "src", "assets"))
    mkpath(dir); savefig(p, joinpath(dir, "byteops.png"))
end
println("wrote docs/assets/byteops.png (+ docs/src/assets)")
