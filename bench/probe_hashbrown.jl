# Probe: Base Dict vs hashbrown (SwissTable) vs BlazingPorts.SwissDict, PHASE-SEPARATED.
# Dict and SwissDict are sizehint!'d for build; lookups timed on a pre-built table.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_hashbrown.jl  (build lib: bench/rust_compare/build.sh)
using Chairmarks, Printf, Random
import Chairmarks: median
using BlazingPorts.SwissDict: SwissDict

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
    m = Dict{UInt64,UInt64}(); sizehint!(m, length(ks))   # match hashbrown with_capacity
    @inbounds for (i, x) in enumerate(ks); m[x] = UInt64(i - 1); end   # 0-based to match Rust
    m
end
@noinline function sw_build(ks)
    m = SwissDict{UInt64,UInt64}(); sizehint!(m, length(ks))
    @inbounds for (i, x) in enumerate(ks); m[x] = UInt64(i - 1); end
    m
end
@noinline function jl_hits(m, ks); a = UInt64(0); @inbounds for x in ks; a += m[x]; end; a; end
@noinline function jl_miss(m, ks); c = 0; @inbounds for x in ks; haskey(m, x) && (c += 1); end; c; end

djl = jl_build(KEYS); dsw = sw_build(KEYS); hh = hb_build(KEYS)
@assert jl_hits(djl, KEYS) == jl_hits(dsw, KEYS) == hb_hits(hh, KEYS)
@assert jl_miss(djl, ABSENT) == jl_miss(dsw, ABSENT) == hb_miss(hh, ABSENT) == 0

nsop(b) = median(b).time * 1e9 / N
function row(name, jl, sw, rs)
    @printf("%-12s  Dict %6.2f ns/op  SwissDict %6.2f ns/op  hashbrown %6.2f ns/op   SwissDict/Dict %.2f   hb/Dict %.2f\n",
            name, jl, sw, rs, sw / jl, rs / jl)
end
row("build",
    nsop(@be jl_build(KEYS) seconds=3),
    nsop(@be sw_build(KEYS) seconds=3),
    nsop(@be (h = hb_build(KEYS); hb_free(h)) seconds=3))
row("lookup-hit",
    nsop(@be jl_hits(djl, KEYS) seconds=3),
    nsop(@be jl_hits(dsw, KEYS) seconds=3),
    nsop(@be hb_hits(hh, KEYS) seconds=3))
row("lookup-miss",
    nsop(@be jl_miss(djl, ABSENT) seconds=3),
    nsop(@be jl_miss(dsw, ABSENT) seconds=3),
    nsop(@be hb_miss(hh, ABSENT) seconds=3))
hb_free(hh)
println("load factor ≈ N/capacity (unique keys, both pre-sized); parity ≥0.96 ⇒ good enough")
