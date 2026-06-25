# Probe: Base Dict vs hashbrown (SwissTable), PHASE-SEPARATED. The original bp_hashbrown_roundtrip
# conflated build+lookup (build-dominated) and compared hashbrown's with_capacity vs a no-sizehint Dict.
# Here: Julia Dict is sizehint!'d for build and pre-built for lookups; lookups timed via handle shims.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_hashbrown.jl  (build lib: bench/rust_compare/build.sh)
using Chairmarks, Printf, Random
import Chairmarks: median
const LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")
hb_build(d) = @ccall LIB.bp_hb_build(d::Ptr{UInt64}, length(d)::Csize_t)::Ptr{Cvoid}
hb_free(h)  = @ccall LIB.bp_hb_free(h::Ptr{Cvoid})::Cvoid
hb_hits(h, k) = @ccall LIB.bp_hb_get_hits(h::Ptr{Cvoid}, k::Ptr{UInt64}, length(k)::Csize_t)::UInt64
hb_miss(h, k) = @ccall LIB.bp_hb_get_miss(h::Ptr{Cvoid}, k::Ptr{UInt64}, length(k)::Csize_t)::UInt64

const N = 1_000_000
Random.seed!(1)
const KEYS   = UInt64.(shuffle(1:N))            # present (unique)
const ABSENT = shuffle(UInt64.((N+1):(2N)))     # guaranteed misses

@noinline function jl_build(ks)
    m = Dict{UInt64,UInt64}(); sizehint!(m, length(ks))   # match hashbrown with_capacity (the fix)
    @inbounds for (i, x) in enumerate(ks); m[x] = UInt64(i - 1); end   # 0-based to match Rust `i as u64`
    m
end
@noinline function jl_hits(m, ks); a = UInt64(0); @inbounds for x in ks; a += m[x]; end; a; end
@noinline function jl_miss(m, ks); c = 0; @inbounds for x in ks; haskey(m, x) && (c += 1); end; c; end

djl = jl_build(KEYS); hh = hb_build(KEYS)
@assert jl_hits(djl, KEYS) == hb_hits(hh, KEYS)
@assert jl_miss(djl, ABSENT) == hb_miss(hh, ABSENT) == 0     # both: 0 hits among absent keys

nsop(b) = median(b).time * 1e9 / N
row(name, jl, rs) = @printf("%-12s Dict %.2f ns/op   hashbrown %.2f ns/op   parity(hb/Dict) %.2f\n", name, jl, rs, rs / jl)
row("build",       nsop(@be jl_build(KEYS) seconds=3),       nsop(@be (h = hb_build(KEYS); hb_free(h)) seconds=3))
row("lookup-hit",  nsop(@be jl_hits(djl, KEYS) seconds=3),   nsop(@be hb_hits(hh, KEYS) seconds=3))
row("lookup-miss", nsop(@be jl_miss(djl, ABSENT) seconds=3), nsop(@be hb_miss(hh, ABSENT) seconds=3))
hb_free(hh)
println("load factor ≈ N/capacity (unique keys, both pre-sized); parity ≥0.96 ⇒ Dict good enough")
