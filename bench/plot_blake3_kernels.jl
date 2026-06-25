# 3-way blake3 compress-kernel VIOLIN from SAVED data (no re-benchmark). Each contender's per-sample
# GB/s distribution. Regenerate with:  julia --project=bench bench/plot_blake3_kernels.jl
using JSON, StatsPlots, Printf, Statistics
const HERE = @__DIR__
const NB = 16 * 1024 * 1024
d = JSON.parsefile(joinpath(HERE, "results", "blake3_kernels.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
keys3  = ["Rust intrinsics (LLVM, AVX2)", "Julia SIMD.jl (LLVM, AVX-512)", "blake3 hand-asm (AVX-512)"]
labels = ["Rust\n(LLVM, AVX2 8-wide)", "Julia SIMD.jl\n(LLVM, AVX-512 16-wide)", "blake3 hand-asm\n(AVX-512 16-wide)"]
cols   = [:firebrick, :seagreen, :slategray]
gbs(c) = NB ./ Float64.(c["samples"]) ./ 1e9                 # per-sample GB/s
meds   = [NB / cmap[k]["median"] / 1e9 for k in keys3]
ttl = @sprintf("BLAKE3 compress, single-thread:  Julia (LLVM) beats Rust (LLVM) %.2f×,  hand-asm beats both %.2f×",
    meds[2]/meds[1], meds[3]/meds[2])
p = plot(; legend = false, ylabel = "GB/s  (higher = better)", title = ttl, titlefontsize = 10,
    xticks = ([1,2,3], labels), xlims = (0.4, 3.6), framestyle = :box, dpi = 200,
    size = (860, 500), ylims = (0, maximum(meds) * 1.15))
for (i, k) in enumerate(keys3)
    g = gbs(cmap[k]); hi = quantile(g, 0.98); lo = quantile(g, 0.02)
    g = filter(x -> lo <= x <= hi, g)                       # clip rare timer-outlier tails
    violin!(p, fill(i, length(g)), g; color = cols[i], alpha = 0.65, linewidth = 0, bar_width = 0.7)
    annotate!(p, i, meds[i] + maximum(meds) * 0.05, text(@sprintf("%.2f", meds[i]), 11, :center))
end
for dir in (joinpath(HERE,"..","docs","assets"), joinpath(HERE,"..","docs","src","assets"))
    mkpath(dir); savefig(p, joinpath(dir, "blake3_kernels.png"))
end
println("wrote docs/assets/blake3_kernels.png (+ docs/src/assets)")
