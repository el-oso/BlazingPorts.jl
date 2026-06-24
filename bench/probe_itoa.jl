# Probe: our IntFormat.format_int! vs the `itoa` crate vs Base `string`. Format-only, DCE-defeated on
# BOTH sides (Julia: `Base.donotdelete`; Rust shim `bp_itoa_bb`: `std::hint::black_box`) — the earlier
# `.len()`-only shim let the optimizer elide itoa's digit-writes (a ~5× artifact); see the gap log.
# Run: taskset -c 1 julia -t 1 --project=bench bench/probe_itoa.jl  (build the lib first: bench/rust_compare/build.sh)

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using BlazingPorts.IntFormat
using Random

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

const N = 100_000
Random.seed!(1)
const XI = rand(Int64, N)                       # full Int64 range (~50% negative, all magnitudes)
const BUF = Vector{UInt8}(undef, 24)

@noinline function ours_fmt()
    acc = 0
    @inbounds for x in XI; acc += format_int!(BUF, x); end
    Base.donotdelete(BUF); acc
end
@noinline rust_fmt() = ccall((:bp_itoa_bb, LIB), Csize_t, (Ptr{Int64}, Csize_t), XI, N)
@noinline base_fmt() = sum(x -> ncodeunits(string(x)), XI)   # Base, allocates a String per call

# correctness + warm-up
let b = Vector{UInt8}(undef, 24)
    @assert all((n = format_int!(b, x); String(b[1:n])) == string(x) for x in XI)
end
ours_fmt(); rust_fmt(); base_fmt()

println("\n=== itoa: format 100k Int64 (full range), format-only ===")
p_ours = run_probe("ours (IntFormat)", ours_fmt; seconds = 4.0)
p_rust = run_probe("itoa crate",       rust_fmt; seconds = 4.0)
p_base = run_probe("Base string",      base_fmt; seconds = 4.0)
report("itoa", Probe[p_base, p_ours, p_rust]; rust_label = "itoa crate")

headtohead = Probe[p_ours, p_rust]                 # Base allocates (~off-scale) → table only
save_probes("itoa", headtohead)
println("\n  plot → ", plot_probe("itoa", headtohead))
@printf("  per-int: ours %.2f ns   itoa %.2f ns   Base %.1f ns  (parity itoa/ours %.2f)\n",
    1e9 * p_ours.median / N, 1e9 * p_rust.median / N, 1e9 * p_base.median / N,
    p_rust.median / p_ours.median)
println("\nDone — probe_itoa.jl")
