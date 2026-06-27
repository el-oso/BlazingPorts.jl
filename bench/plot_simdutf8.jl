# UTF-8 validation throughput grouped bars from SAVED data (no re-benchmark). Log-y (range ~200×).
# Regenerate:  julia --project=bench bench/plot_simdutf8.jl
using JSON, StatsPlots, Printf, Statistics
const HERE = @__DIR__
const NB = 16 * 1024 * 1024
d = JSON.parsefile(joinpath(HERE, "results", "simdutf8.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
gbs(label) = NB / cmap[label]["median"] / 1e9
corp = ["ASCII", "mixed UTF-8"]
jl   = [gbs("Julia isvalid (scalar): $c") for c in corp]
ours = [gbs("BlazingPorts.Utf8 (SIMD): $c") for c in corp]
std  = [gbs("Rust std (scalar): $c") for c in corp]
sj   = [gbs("simdutf8 (SIMD): $c") for c in corp]

p = groupedbar(["ASCII\n(fast path)", "mixed UTF-8\n(multibyte)"], hcat(jl, ours, sj, std);
    label = ["Base isvalid" "BlazingPorts.Utf8 (ours)" "simdutf8 (Rust)" "Rust std (scalar)"],
    color = [:seagreen :firebrick :slateblue :gray], yscale = :log10,
    ylabel = "validation GB/s  (log, higher = better)",
    title = "UTF-8 validation: our pure-Julia SIMD validator vs Base & simdutf8  (16 MiB, single-thread)",
    titlefontsize = 10, legend = :bottomleft, framestyle = :box, dpi = 200, size = (920, 560),
    ylims = (0.2, 150), bar_width = 0.7)
# annotate the ours/Base ratio above each group
for (i, r) in enumerate(ours ./ jl)
    annotate!(p, i, maximum((jl[i], ours[i], sj[i])) * 1.7, text(@sprintf("ours = %.1f× Base", r), 9, :center))
end
plot!(p; xlims = (0.4, 2.6))
for dir in (joinpath(HERE,"..","docs","assets"), joinpath(HERE,"..","docs","src","assets"))
    mkpath(dir); savefig(p, joinpath(dir, "simdutf8.png"))
end
println("wrote docs/assets/simdutf8.png (+ docs/src/assets)")
