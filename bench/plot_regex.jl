# regex match-throughput grouped bars from SAVED data (no re-benchmark). Log-y (range spans ~160×).
# Regenerate:  julia --project=bench bench/plot_regex.jl
using JSON, StatsPlots, Printf, Statistics
const HERE = @__DIR__
const NB = 8 * 1024 * 1024
d = JSON.parsefile(joinpath(HERE, "results", "regex.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
pats   = ["literal `GADGET`", "alternation `(alpha|…|echo)`", "email `[a-z]+@[a-z]+\\.com`", "phone `[0-9]{3}-[0-9]{4}`"]
plabel = ["literal\nGADGET", "alternation\n(5 words)", "email\n[a-z]+@[a-z]+.com", "phone\n[0-9]{3}-[0-9]{4}"]
gbs(label) = NB / cmap[label]["median"] / 1e9
pcre = [gbs("PCRE2 engine: $p") for p in pats]
regx = [gbs("regex crate: $p") for p in pats]
ratio = regx ./ pcre

p = groupedbar(plabel, hcat(pcre, regx);
    label = ["Base Regex / PCRE2 (JIT, C)" "Rust regex crate"], color = [:firebrick :seagreen],
    yscale = :log10, ylabel = "match throughput GB/s  (log, higher = better)",
    title = "regex match throughput: Rust regex crate vs Julia PCRE2  (8 MiB, single-thread)",
    titlefontsize = 10, legend = :topright, framestyle = :box, dpi = 200, size = (960, 560),
    ylims = (0.1, 100), bar_width = 0.7)
for (i, r) in enumerate(ratio)
    annotate!(p, i, regx[i] * 1.4, text(r < 10 ? @sprintf("%.1f×", r) : @sprintf("%.0f×", r), 9, :center))
end
for dir in (joinpath(HERE,"..","docs","assets"), joinpath(HERE,"..","docs","src","assets"))
    mkpath(dir); savefig(p, joinpath(dir, "regex.png"))
end
println("wrote docs/assets/regex.png (+ docs/src/assets)")
