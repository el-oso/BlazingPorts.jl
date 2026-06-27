# regex match throughput — VIOLIN of the saved per-sample distributions (no re-benchmark). Log-y.
# Regenerate:  julia --project=bench bench/plot_regex.jl
using JSON, StatsPlots, Printf, Statistics
const HERE = @__DIR__
const NB = 8 * 1024 * 1024
d = JSON.parsefile(joinpath(HERE, "results", "regex.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
gbs(c) = NB ./ Float64.(c["samples"]) ./ 1e9
med(c) = NB / c["median"] / 1e9

pats   = ["literal `GADGET`", "alternation `(alpha|…|echo)`", "email `[a-z]+@[a-z]+\\.com`", "phone `[0-9]{3}-[0-9]{4}`"]
plabel = ["literal\nGADGET", "alternation\n(5 words)", "email\n[a-z]+@…", "phone\n[0-9]{3}-[0-9]{4}"]
roles = [("PCRE2 engine", :firebrick, "Base Regex / PCRE2 (JIT, C)"), ("regex crate", :seagreen, "Rust regex crate")]
step = 3

p = plot(; ylabel = "match throughput GB/s  (log, higher = better)", yscale = :log10, ylims = (0.1, 120),
    title = "regex match throughput: Rust regex crate vs Julia PCRE2  (8 MiB, single-thread)",
    titlefontsize = 10, legend = :topright, framestyle = :box, dpi = 200, size = (1000, 560))
for (g, pat) in enumerate(pats)
    for (c, (pre, col, role)) in enumerate(roles)
        key = "$pre: $pat"; haskey(cmap, key) || continue
        x = (g - 1) * step + c
        gv = gbs(cmap[key]); lo, hi = quantile(gv, 0.02), quantile(gv, 0.98); gv = filter(v -> lo <= v <= hi, gv)
        violin!(p, fill(x, length(gv)), gv; color = col, alpha = 0.65, linewidth = 0, bar_width = 0.7,
            label = (g == 1 ? role : ""))
        annotate!(p, x, med(cmap[key]) * 1.3, text(@sprintf("%.1f", med(cmap[key])), 7, :center))
    end
    r = med(cmap["regex crate: $pat"]) / med(cmap["PCRE2 engine: $pat"])
    annotate!(p, (g - 1) * step + 1.5, 0.12, text(r < 10 ? @sprintf("%.1f×", r) : @sprintf("%.0f×", r), 8, :center))
end
plot!(p; xticks = ([(g - 1) * step + 1.5 for g in 1:length(pats)], plabel), xlims = (0.3, length(pats) * step - 0.7))
for dir in (joinpath(HERE, "..", "docs", "assets"), joinpath(HERE, "..", "docs", "src", "assets"))
    mkpath(dir); savefig(p, joinpath(dir, "regex.png"))
end
println("wrote docs/assets/regex.png (+ docs/src/assets)")
