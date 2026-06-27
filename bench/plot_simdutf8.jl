# UTF-8 validation throughput — VIOLIN of the saved per-sample distributions (no re-benchmark). Log-y.
# Regenerate:  julia --project=bench bench/plot_simdutf8.jl
using JSON, StatsPlots, Printf, Statistics
using StatsPlots: mm          # margin unit (not exported by default)
const HERE = @__DIR__
const NB = 16 * 1024 * 1024
d = JSON.parsefile(joinpath(HERE, "results", "simdutf8.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
gbs(c) = NB ./ Float64.(c["samples"]) ./ 1e9
med(c) = NB / c["median"] / 1e9

corpora = ["ASCII", "mixed UTF-8"]
# (key-prefix, color, legend role)
roles = [("Julia isvalid (scalar)",   :seagreen,  "Base isvalid"),
         ("BlazingPorts.Utf8 (SIMD)", :firebrick, "BlazingPorts.Utf8 (ours)"),
         ("simdutf8 (SIMD)",          :slateblue, "simdutf8 (Rust)"),
         ("Rust std (scalar)",        :gray,      "Rust std (scalar)")]
step = length(roles) + 1

p = plot(; ylabel = "validation GB/s  (log, higher = better)", yscale = :log10, ylims = (0.3, 140),
    title = "UTF-8 validation: our SIMD validator vs Base & simdutf8  (16 MiB, single-thread)",
    titlefontsize = 10, legend = :topright, framestyle = :box, dpi = 200, size = (1000, 560),
    left_margin = 8mm, bottom_margin = 5mm, top_margin = 3mm, right_margin = 3mm)
for (g, corp) in enumerate(corpora)
    for (c, (pre, col, role)) in enumerate(roles)
        key = "$pre: $corp"; haskey(cmap, key) || continue
        x = (g - 1) * step + c
        gv = gbs(cmap[key]); lo, hi = quantile(gv, 0.02), quantile(gv, 0.98)
        gv = filter(v -> lo <= v <= hi, gv)
        violin!(p, fill(x, length(gv)), gv; color = col, alpha = 0.65, linewidth = 0, bar_width = 0.7,
            label = (g == 1 ? role : ""))
        annotate!(p, x, med(cmap[key]) * 1.2, text(@sprintf("%.1f", med(cmap[key])), 7, :center))
    end
    o = med(cmap["BlazingPorts.Utf8 (SIMD): $corp"]); s = med(cmap["simdutf8 (SIMD): $corp"])
    annotate!(p, (g - 1) * step + 2.5, 0.62, text(@sprintf("ours = %.2f× simdutf8", o / s), 8, :center))
end
plot!(p; xticks = ([(g - 1) * step + 2.5 for g in 1:length(corpora)],
                   ["ASCII\n(fast path)", "mixed UTF-8\n(multibyte)"]),
      xlims = (0.3, length(corpora) * step - 0.7))
for dir in (joinpath(HERE, "..", "docs", "assets"), joinpath(HERE, "..", "docs", "src", "assets"))
    mkpath(dir); savefig(p, joinpath(dir, "simdutf8.png"))
end
println("wrote docs/assets/simdutf8.png (+ docs/src/assets)")
