# Probe: Base BitSet vs the `roaring` crate (compressed bitsets). Set algebra (union/intersect) +
# membership, build-from-array INCLUDED on both sides (realistic "set ops from data"). Measured across
# densities (F26: value-dependent) — dense small-domain vs sparse large-domain.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_roaring.jl  (build lib: bench/rust_compare/build.sh)
using Chairmarks, Printf, Random
import Chairmarks: median
const LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")
rs_or(a, b)  = @ccall LIB.bp_roaring_or(a::Ptr{UInt32}, length(a)::Csize_t, b::Ptr{UInt32}, length(b)::Csize_t)::UInt64
rs_and(a, b) = @ccall LIB.bp_roaring_and(a::Ptr{UInt32}, length(a)::Csize_t, b::Ptr{UInt32}, length(b)::Csize_t)::UInt64
rs_con(a, q) = @ccall LIB.bp_roaring_contains(a::Ptr{UInt32}, length(a)::Csize_t, q::Ptr{UInt32}, length(q)::Csize_t)::UInt64

@noinline jl_or(a, b)  = length(union(BitSet(a), BitSet(b)))
@noinline jl_and(a, b) = length(intersect(BitSet(a), BitSet(b)))
@noinline function jl_con(a, q)
    s = BitSet(a); c = 0
    for x in q; (x in s) && (c += 1); end
    c
end

Random.seed!(1)
gen(n, hi) = rand(UInt32(1):UInt32(hi), n)
ms(b) = median(b).time * 1e3
function row(lbl, jl, rs, args...)
    @assert jl(args...) == rs(args...) "cardinality mismatch in $lbl"   # correctness
    to = ms(@be jl(args...) seconds=3); tr = ms(@be rs(args...) seconds=3)
    @printf("%-26s julia %.3f ms   roaring %.3f ms   parity(roaring/julia) %.2f\n", lbl, to, tr, tr/to)
end

# Dense: 100k elements over a small domain [1,10^6] (~10% full); BitSet bit-array ~125 KB.
da, db, dq = gen(100_000, 10^6), gen(100_000, 10^6), gen(100_000, 10^6)
# Sparse: 10k elements over a large domain [1,10^8]; BitSet bit-array ~12.5 MB (mostly empty words).
sa, sb, sq = gen(10_000, 10^8), gen(10_000, 10^8), gen(10_000, 10^8)

println("=== dense (100k over 1e6) ===")
row("dense union",     jl_or,  rs_or,  da, db)
row("dense intersect", jl_and, rs_and, da, db)
row("dense contains",  jl_con, rs_con, da, dq)
println("=== sparse (10k over 1e8) ===")
row("sparse union",     jl_or,  rs_or,  sa, sb)
row("sparse intersect", jl_and, rs_and, sa, sb)
row("sparse contains",  jl_con, rs_con, sa, sq)
