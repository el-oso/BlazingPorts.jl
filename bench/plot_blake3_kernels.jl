# 4-way blake3 compress-kernel VIOLIN from SAVED data (no re-benchmark). Each contender's per-sample
# GB/s distribution. The 4th bar is OUR blake3_asm switch path (ccall into the vendored .S) — it lands
# on the crate's hand-asm bar, proving the switch reaches the asm ceiling from Julia.
# Regenerate with:  julia --project=bench bench/plot_blake3_kernels.jl
using JSON, StatsPlots, Printf, Statistics
using StatsPlots: mm          # margin unit (not exported by default)
const HERE = @__DIR__
const NB = 16 * 1024 * 1024
d = JSON.parsefile(joinpath(HERE, "results", "blake3_kernels.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
keys3  = ["Rust intrinsics (LLVM, AVX2)", "Julia SIMD.jl (LLVM, AVX-512)", "blake3 hand-asm (AVX-512)", "BlazingPorts asm-leaf (blake3_asm)"]
labels = ["Rust\n(LLVM, AVX2 8-wide)", "Julia SIMD.jl\n(LLVM, AVX-512 16-wide)", "blake3 hand-asm\n(crate's .S)", "BlazingPorts asm-leaf\n(blake3_asm switch)"]
cols   = [:firebrick, :seagreen, :slategray, :steelblue]
gbs(c) = NB ./ Float64.(c["samples"]) ./ 1e9                 # per-sample GB/s
meds   = [NB / cmap[k]["median"] / 1e9 for k in keys3]
ttl = @sprintf("BLAKE3 compress, single-thread:  Julia (LLVM) beats Rust (LLVM) %.2f×;  the blake3_asm switch reaches the hand-asm ceiling (%.2f / %.2f GB/s)",
    meds[2]/meds[1], meds[4], meds[3])
p = plot(; legend = false, ylabel = "GB/s  (higher = better)", title = ttl, titlefontsize = 9,
    xticks = ([1,2,3,4], labels), xlims = (0.4, 4.6), framestyle = :box, dpi = 200,
    size = (1040, 520), ylims = (0, maximum(meds) * 1.15),
    left_margin = 8mm, bottom_margin = 5mm, top_margin = 3mm, right_margin = 3mm)
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

# ── Preferences switch: full blake3() pipeline, asm-leaf vs pure (from the SAME saved JSON) ─────────
switch_keys = ["BlazingPorts.blake3() pure", "BlazingPorts.blake3() asm-leaf"]
if all(haskey(cmap, k) for k in switch_keys)
    slabels = ["pure SIMD.jl leaf\n(portable default)", "asm leaf\n(blake3_asm = on)"]
    scols   = [:seagreen, :slategray]
    smeds   = [NB / cmap[k]["median"] / 1e9 for k in switch_keys]
    sttl = @sprintf("BLAKE3 full hash(), single-thread:  the blake3_asm switch lifts end-to-end throughput %.2f×", smeds[2]/smeds[1])
    ps = plot(; legend = false, ylabel = "GB/s  (higher = better)", title = sttl, titlefontsize = 10,
        xticks = ([1,2], slabels), xlims = (0.4, 2.6), framestyle = :box, dpi = 200,
        size = (720, 500), ylims = (0, maximum(smeds) * 1.15),
        left_margin = 8mm, bottom_margin = 5mm, top_margin = 3mm, right_margin = 3mm)
    for (i, k) in enumerate(switch_keys)
        gg = gbs(cmap[k]); hi = quantile(gg, 0.98); lo = quantile(gg, 0.02)
        gg = filter(x -> lo <= x <= hi, gg)
        violin!(ps, fill(i, length(gg)), gg; color = scols[i], alpha = 0.65, linewidth = 0, bar_width = 0.6)
        annotate!(ps, i, smeds[i] + maximum(smeds) * 0.05, text(@sprintf("%.2f", smeds[i]), 11, :center))
    end
    for dir in (joinpath(HERE,"..","docs","assets"), joinpath(HERE,"..","docs","src","assets"))
        mkpath(dir); savefig(ps, joinpath(dir, "blake3_asm_switch.png"))
    end
    println("wrote docs/assets/blake3_asm_switch.png (+ docs/src/assets)")
end
