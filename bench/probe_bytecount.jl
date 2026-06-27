# Probe: Rust `bytecount` (SIMD masked-compare + popcount reduction) vs Julia `count(==(b), v)`.
include(joinpath(@__DIR__, "harness.jl")); using .Harness; using Printf, Random
Harness.single_thread!(); const LIB = Harness.RUST_LIB
const V = rand(Xoshiro(0xB17EC0), UInt8, 16*1024*1024); const NB = length(V); const NEEDLE = 0x41
@noinline jl_count() = count(==(NEEDLE), V)
@noinline rs_count() = GC.@preserve V ccall((:bp_bytecount, LIB), Csize_t, (Ptr{UInt8}, Csize_t, UInt8), V, NB, NEEDLE)
n1 = jl_count(); n2 = Int(rs_count()); n1 == n2 || @warn "count mismatch $n1 vs $n2"
g(p) = NB / p.median / 1e9
println("\n=== byte counting, $(NB>>20) MiB, single-thread ===")
p_jl = run_probe("Julia count(==(b))", jl_count; seconds=4.0); p_rs = run_probe("bytecount (SIMD)", rs_count; seconds=4.0)
@printf("  matches=%d  Julia %.2f  bytecount %.2f GB/s  → Julia/bytecount = %.2f×\n", n1, g(p_jl), g(p_rs), g(p_jl)/g(p_rs))
save_probes("bytecount", Probe[p_jl, p_rs]); println("saved → bench/results/bytecount.json")
