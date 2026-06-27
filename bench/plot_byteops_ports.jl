# Ported byte-ops kernels vs the Rust crates, from SAVED data (no re-benchmark). Log-y.
# Regenerate:  julia --project=bench bench/plot_byteops_ports.jl
using JSON, StatsPlots, Printf
const HERE = @__DIR__
const NB = 12 * 1024 * 1024
d = JSON.parsefile(joinpath(HERE, "results", "byteops_ports.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
gbs(label) = NB / cmap[label]["median"] / 1e9
ops = ["base64 encode", "hex encode"]
jl   = [gbs("Julia stdlib (scalar, alloc): $op") for op in ops]
ours = [gbs("BlazingPorts.ByteOps (SIMD): $op") for op in ops]
rust = [gbs(op == "base64 encode" ? "base64-simd (SIMD kernel): $op" : "faster-hex (SIMD kernel): $op") for op in ops]

p = groupedbar(["base64 encode", "hex encode"], hcat(jl, ours, rust);
    label = ["Julia stdlib (scalar)" "BlazingPorts.ByteOps (ours)" "Rust crate (SIMD)"],
    color = [:seagreen :firebrick :slateblue], yscale = :log10,
    ylabel = "throughput GB/s  (log, higher = better)",
    title = "Ported byte-ops kernels vs Rust crates — KERNEL-ONLY (12 MiB, single-thread)",
    titlefontsize = 10, legend = :topright, framestyle = :box, dpi = 200, size = (880, 560),
    ylims = (0.4, 60), bar_width = 0.7)
for (i, r) in enumerate(ours ./ rust)
    annotate!(p, i, maximum((jl[i], ours[i], rust[i])) * 1.5,
        text(@sprintf("ours = %.2f× crate", r), 9, :center))
end
plot!(p; xlims = (0.4, 2.6))
for dir in (joinpath(HERE, "..", "docs", "assets"), joinpath(HERE, "..", "docs", "src", "assets"))
    mkpath(dir); savefig(p, joinpath(dir, "byteops_ports.png"))
end
println("wrote docs/assets/byteops_ports.png (+ docs/src/assets)")
