# Probe: Tier-1/2 GP crates — int/float→string + hashing + hashmap, vs Julia Base. Batched over N,
# single-thread, median. Run: taskset -c 2 julia -t 1 --project=bench bench/probe_gp_tier1.jl
using Chairmarks, Printf, Random
import Chairmarks: median
const LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")
const N = 100_000
xi = rand(Int64, N); xf = randn(N); xu = UInt64.(shuffle(1:N))   # unique keys for the map

jl_itoa(xs) = sum(x -> ncodeunits(string(x)), xs)
jl_ryu(xs)  = sum(x -> ncodeunits(string(x)), xs)
jl_hash(xs) = (a = UInt64(0); for x in xs; a ⊻= hash(x); end; a)
function jl_dict(xs)
    m = Dict{UInt64,UInt64}(); for (i, x) in enumerate(xs); m[x] = UInt64(i); end
    a = UInt64(0); for x in xs; a += m[x]; end; a
end
@noinline fj_itoa(xs) = jl_itoa(xs); @noinline fj_ryu(xs) = jl_ryu(xs)
@noinline fj_hash(xs) = jl_hash(xs); @noinline fj_dict(xs) = jl_dict(xs)
rs_itoa(xs) = @ccall LIB.bp_itoa_len(xs::Ptr{Int64}, length(xs)::Csize_t)::Csize_t
rs_ryu(xs)  = @ccall LIB.bp_ryu_len(xs::Ptr{Float64}, length(xs)::Csize_t)::Csize_t
rs_fx(xs)   = @ccall LIB.bp_fxhash_sum(xs::Ptr{UInt64}, length(xs)::Csize_t)::UInt64
rs_ah(xs)   = @ccall LIB.bp_ahash_sum(xs::Ptr{UInt64}, length(xs)::Csize_t)::UInt64
rs_hb(xs)   = @ccall LIB.bp_hashbrown_roundtrip(xs::Ptr{UInt64}, length(xs)::Csize_t)::UInt64

ms(b) = median(b).time * 1e3
row(name, j, r) = @printf("%-26s julia %.3f ms   rust %.3f ms   parity(rust/julia) %.2f\n", name, j, r, r/j)
row("itoa  string(Int)",     ms(@be fj_itoa(xi) seconds=3), ms(@be rs_itoa(xi) seconds=3))
row("ryu   string(Float64)", ms(@be fj_ryu(xf)  seconds=3), ms(@be rs_ryu(xf)  seconds=3))
row("fxhash  hash(UInt64)",  ms(@be fj_hash(xu) seconds=3), ms(@be rs_fx(xu)   seconds=3))
row("ahash   hash(UInt64)",  ms(@be fj_hash(xu) seconds=3), ms(@be rs_ah(xu)   seconds=3))
row("hashbrown vs Dict",     ms(@be fj_dict(xu) seconds=3), ms(@be rs_hb(xu)   seconds=3))
