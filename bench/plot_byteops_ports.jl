# Ported byte-ops kernels vs the Rust crates — VIOLIN of the saved per-sample distributions (no
# re-benchmark). Kernel-only (both sides preallocated). Regenerate:
#   julia --project=bench bench/plot_byteops_ports.jl
using JSON, StatsPlots, Printf, Statistics
using StatsPlots: mm          # margin unit (not exported by default)
const HERE = @__DIR__
const NB = 12 * 1024 * 1024
d = JSON.parsefile(joinpath(HERE, "results", "byteops_ports.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
gbs(c) = NB ./ Float64.(c["samples"]) ./ 1e9                 # per-sample time → GB/s
med(c) = NB / c["median"] / 1e9

ops = ["base64 encode", "base64 decode", "hex encode", "hex decode"]
crate_of(op) = startswith(op, "base64") ? "base64-simd" : "faster-hex"
# (key-suffix, color, legend role) for the 3 contenders per op
roles = [("Julia stdlib (scalar, alloc)", :seagreen,  "Julia stdlib (scalar)"),
         ("BlazingPorts.ByteOps (SIMD)",  :firebrick, "BlazingPorts.ByteOps (ours)"),
         (nothing,                        :slateblue, "Rust crate (SIMD)")]   # nothing → crate-specific key

p = plot(; ylabel = "GB/s  (log, higher = better)", yscale = :log10, ylims = (0.1, 40),
    title = "Ported byte-ops kernels vs Rust crates — kernel-only (12 MiB, single-thread)",
    titlefontsize = 10, legend = :topright, framestyle = :box, dpi = 200, size = (1000, 560),
    left_margin = 8mm, bottom_margin = 5mm, top_margin = 3mm, right_margin = 3mm)
allmed = Float64[]
for (g, op) in enumerate(ops)
    keys3 = ["Julia stdlib (scalar, alloc): $op", "BlazingPorts.ByteOps (SIMD): $op",
             "$(crate_of(op)) (SIMD kernel): $op"]
    for (c, key) in enumerate(keys3)
        haskey(cmap, key) || continue
        x = (g - 1) * 4 + c
        gv = gbs(cmap[key]); lo, hi = quantile(gv, 0.02), quantile(gv, 0.98)
        gv = filter(v -> lo <= v <= hi, gv)
        col = roles[c][2]
        violin!(p, fill(x, length(gv)), gv; color = col, alpha = 0.65, linewidth = 0, bar_width = 0.7,
            label = (g == 1 ? roles[c][3] : ""))
        m = med(cmap[key]); push!(allmed, m)
        annotate!(p, x, m * 1.18, text(@sprintf("%.1f", m), 7, :center))
    end
    # ratio vs crate under the group
    o = med(cmap["BlazingPorts.ByteOps (SIMD): $op"]); r = med(cmap["$(crate_of(op)) (SIMD kernel): $op"])
    annotate!(p, (g - 1) * 4 + 2, 0.13, text(@sprintf("%.2f× crate", o / r), 8, :center))
end
plot!(p; xticks = ([(g - 1) * 4 + 2 for g in 1:length(ops)], ops), xlims = (0.3, length(ops) * 4 - 0.7))
for dir in (joinpath(HERE, "..", "docs", "assets"), joinpath(HERE, "..", "docs", "src", "assets"))
    mkpath(dir); savefig(p, joinpath(dir, "byteops_ports.png"))
end
println("wrote docs/assets/byteops_ports.png (+ docs/src/assets)")
