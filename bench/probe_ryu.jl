# Probe: Base.Ryu.writeshortest (Julia already ships Ryu, zero-alloc) vs the `ryu` crate. Format-only,
# DCE-defeated on BOTH sides (Julia: Base.donotdelete; Rust `bp_ryu_bb`: black_box). The earlier
# `.len()`-only shim let the optimizer elide ryu's writes (same artifact as itoa) → re-measuring fairly.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_ryu.jl   (build lib first: bench/rust_compare/build.sh)

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Random, Printf

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

const N = 100_000
Random.seed!(1)
const XF = randn(N)                              # representative Float64 mix
const BUF = Vector{UInt8}(undef, 32)

@noinline function ours_fmt()
    acc = 0
    @inbounds for x in XF; acc += Base.Ryu.writeshortest(BUF, 1, x) - 1; end
    Base.donotdelete(BUF); acc
end
@noinline rust_fmt() = ccall((:bp_ryu_bb, LIB), Csize_t, (Ptr{Float64}, Csize_t), XF, N)
@noinline base_fmt() = sum(x -> ncodeunits(string(x)), XF)    # Base, allocates a String per call

# sanity: both produce parseable shortest output that round-trips (NOT a bit-exact oracle — the two
# formatters may emit different valid shortest forms).
let b = Vector{UInt8}(undef, 32)
    @assert all(parse(Float64, String(b[1:Base.Ryu.writeshortest(b,1,x)-1])) === x for x in XF[1:1000])
end
ours_fmt(); rust_fmt(); base_fmt()

println("\n=== ryu: format 100k Float64 (randn), format-only ===")
p_ours = run_probe("Base.Ryu (Julia)", ours_fmt; seconds = 4.0)
p_rust = run_probe("ryu crate",        rust_fmt; seconds = 4.0)
p_base = run_probe("Base string",      base_fmt; seconds = 4.0)
report("ryu", Probe[p_base, p_ours, p_rust]; rust_label = "ryu crate")

headtohead = Probe[p_ours, p_rust]               # Base allocates → table only
save_probes("ryu", headtohead)
println("\n  plot → ", plot_probe("ryu", headtohead))
@printf("  per-float: Base.Ryu %.2f ns   ryu crate %.2f ns   Base string %.1f ns  (parity ryu/ours %.2f)\n",
    1e9 * p_ours.median / N, 1e9 * p_rust.median / N, 1e9 * p_base.median / N,
    p_rust.median / p_ours.median)
println("\nDone — probe_ryu.jl")
