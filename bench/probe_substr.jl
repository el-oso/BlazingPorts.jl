# Probe: SIMD substring search — our `find_substr` vs Base `findfirst` vs the memchr::memmem crate.
# Worst-case scan (needle only at the very end of a 32 MiB haystack ⇒ full read). Representative
# needle length m=8; the full m∈{2,8,32} sweep is in ../blazingly-fast-rust-crates.md's gap log.
# Plot: ours vs memmem (both ~0.5 ms, bandwidth-bound); Base (~56 ms here) stays in the table only —
# including it in the violin would clip ours/memmem off-scale.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_substr.jl
#   (build the Rust lib first: bash bench/rust_compare/build.sh)

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using BlazingPorts.StringSearch
using Printf

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

const N = 32 * 1024 * 1024
const M = 8
const NEEDLE = rand(UInt8(33):UInt8(126), M)              # printable, won't collide with the filler
const HAY = let h = fill(0x20, N); h[end-M+1:end] .= NEEDLE; h end

base_find(h, p) = (r = findfirst(p, h); isnothing(r) ? 0 : first(r))
@noinline jl_find()   = base_find(HAY, NEEDLE)
@noinline our_find()  = find_substr(HAY, NEEDLE)
@noinline rust_find() = ccall((:bp_memmem, LIB), Cssize_t,
    (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t), HAY, length(HAY), NEEDLE, length(NEEDLE))

# correctness + warm-up
@assert our_find() == N - M + 1 == rust_find() + 1   # ours 1-based, memmem 0-based
jl_find(); our_find(); rust_find()

println("\n=== stringsearch: substring search, 32 MiB worst-case, m=$M ===")
p_base = run_probe("Base findfirst", jl_find;   seconds = 3.0)
p_ours = run_probe("ours (SIMD)",    our_find;  seconds = 3.0)
p_rust = run_probe("memmem crate",   rust_find; seconds = 3.0)

report("stringsearch", Probe[p_base, p_ours, p_rust]; rust_label = "memmem crate")

# Plot + persist only the fast head-to-head (Base is 100× off-scale).
headtohead = Probe[p_ours, p_rust]
save_probes("stringsearch", headtohead)
out = plot_probe("stringsearch", headtohead)
println("\n  plot → $out")
@printf("  Base %.2f ms  |  ours %.3f ms  memmem %.3f ms  (ours/Base %.0f×, parity memmem/ours %.2f)\n",
    1e3 * p_base.median, 1e3 * p_ours.median, 1e3 * p_rust.median,
    p_base.median / p_ours.median, p_rust.median / p_ours.median)
println("\nDone — probe_substr.jl")
