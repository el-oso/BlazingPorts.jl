# Combined "byte-ops / shuffle-SIMD family" plot from SAVED data (bytecount + transcode + lexical).
# Regenerate:  julia --project=bench bench/plot_byteops.jl
using JSON, StatsPlots, Printf
const HERE = @__DIR__
const MiB = 1024 * 1024
load(c) = Dict(x["label"] => x for x in JSON.parsefile(joinpath(HERE, "results", "$c.json"))["contenders"])
bc, tc, lx = load("bytecount"), load("transcode"), load("lexical")
gb(d, label, nb) = nb / d[label]["median"] / 1e9
ops = [   # (name, Julia GB/s, Rust-SIMD GB/s) — ordered by widening gap
    ("bytecount",      gb(bc, "Julia count(==(b))",  16MiB),  gb(bc, "bytecount (SIMD)",  16MiB)),
    ("hex encode",     gb(tc, "Julia hex",           12MiB),  gb(tc, "Rust SIMD hex",     12MiB)),
    ("float parse",    gb(lx, "Julia parse(Float64)", 12.09MiB), gb(lx, "lexical-core",   12.09MiB)),
    ("base64 encode",  gb(tc, "Julia base64",        12MiB),  gb(tc, "Rust SIMD base64",  12MiB)),
]
labels = [o[1] for o in ops]; jl = [o[2] for o in ops]; rs = [o[3] for o in ops]
p = groupedbar(labels, hcat(jl, rs); label = ["Julia stdlib" "Rust SIMD crate"],
    color = [:seagreen :slateblue], yscale = :log10, ylabel = "GB/s  (log, higher = better)",
    title = "Byte-ops: Julia stdlib vs Rust SIMD  (single-thread) — shuffle/lookup-SIMD gap class",
    titlefontsize = 10, legend = :topright, framestyle = :box, dpi = 200, size = (960, 560),
    ylims = (0.1, 60), bar_width = 0.7)
for (i, o) in enumerate(ops)
    r = o[2] / o[3]
    annotate!(p, i, max(o[2], o[3]) * 1.6, text(r > 0.8 ? "parity" : @sprintf("%.0f× gap", 1/r), 9, :center))
end
plot!(p; xlims = (0.3, 4.7))
for dir in (joinpath(HERE,"..","docs","assets"), joinpath(HERE,"..","docs","src","assets"))
    mkpath(dir); savefig(p, joinpath(dir, "byteops.png"))
end
println("wrote docs/assets/byteops.png (+ docs/src/assets)")
