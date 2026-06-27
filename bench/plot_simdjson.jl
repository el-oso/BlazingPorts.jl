# simd-json vs Julia JSON parsers VIOLIN from SAVED data (no re-benchmark).
# Regenerate with:  julia --project=bench bench/plot_simdjson.jl
using JSON, StatsPlots, Printf, Statistics
const HERE = @__DIR__
const NB = 4 * 1024 * 1024   # doc size from probe_simdjson.jl (gen_json target); GB/s = NB / time
d = JSON.parsefile(joinpath(HERE, "results", "simdjson.json"))
cmap = Dict(c["label"] => c for c in d["contenders"])
# memcpy (~62 GB/s) is reported as a number, not plotted — it would crush the <1 GB/s parse axis.
# Grouped: structural pair (isvalidjson vs simd-json tape), eager pair (JSON.parse vs simd-json DOM),
# then JSON3.read for reference. JSON.isvalidjson is the σ-clean 0-alloc Julia number.
keys5  = ["JSON.isvalidjson (structural scan)", "simd-json to_tape (lazy)",
          "JSON.parse (eager Dict)", "simd-json to_borrowed_value (DOM)", "JSON3.read (lazy tape)"]
labels = ["JSON.isvalidjson\n(0-alloc scan)", "simd-json tape\n(copy+parse)",
          "JSON.parse\n(eager Dict)", "simd-json DOM\n(copy+parse)", "JSON3.read\n(lazy tape, ref)"]
cols   = [:seagreen, :slategray, :mediumseagreen, :gray, :darkgreen]
gbs(c) = NB ./ Float64.(c["samples"]) ./ 1e9
present = [k for k in keys5 if haskey(cmap, k)]
meds = [NB / cmap[k]["median"] / 1e9 for k in present]
gb(label) = haskey(cmap, label) ? NB / cmap[label]["median"] / 1e9 : NaN
ttl = @sprintf("JSON parse, single-thread (JSON.jl ≥ 1.6):  isvalidjson %.2f vs simd-json tape %.2f GB/s = %.2f×  (structural)",
    gb("JSON.isvalidjson (structural scan)"), gb("simd-json to_tape (lazy)"),
    gb("JSON.isvalidjson (structural scan)") / gb("simd-json to_tape (lazy)"))
p = plot(; legend = false, ylabel = "GB/s  (higher = better)", title = ttl, titlefontsize = 9,
    xticks = (collect(1:length(present)), labels[1:length(present)]), xlims = (0.4, length(present) + 0.6),
    framestyle = :box, dpi = 200, size = (1040, 520), ylims = (0, maximum(meds) * 1.15))
for (i, k) in enumerate(present)
    g = gbs(cmap[k]); hi = quantile(g, 0.98); lo = quantile(g, 0.02)
    g = filter(x -> lo <= x <= hi, g)
    violin!(p, fill(i, length(g)), g; color = cols[i], alpha = 0.65, linewidth = 0, bar_width = 0.7)
    annotate!(p, i, meds[i] + maximum(meds) * 0.05, text(@sprintf("%.2f", meds[i]), 10, :center))
end
for dir in (joinpath(HERE,"..","docs","assets"), joinpath(HERE,"..","docs","src","assets"))
    mkpath(dir); savefig(p, joinpath(dir, "simdjson.png"))
end
println("wrote docs/assets/simdjson.png (+ docs/src/assets)")
