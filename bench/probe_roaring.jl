# Probe: Base BitSet vs the `roaring` crate. OPERATION-ONLY — structures are pre-built outside the
# timed region (handle-based roaring shims), so we measure set algebra, not build cost. (An earlier
# build-included version was build-dominated and unfair: per-element insert is roaring's worst build
# path, and BitSet pays a 12.5 MB alloc on sparse.) Densities per F26. Build cost reported separately.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_roaring.jl  (build lib: bench/rust_compare/build.sh)
using Chairmarks, Printf, Random
import Chairmarks: median
const LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")
rb_build(a)   = @ccall LIB.bp_roaring_build(a::Ptr{UInt32}, length(a)::Csize_t)::Ptr{Cvoid}
rb_free(h)    = @ccall LIB.bp_roaring_free(h::Ptr{Cvoid})::Cvoid
rb_or(a, b)   = @ccall LIB.bp_roaring_or_h(a::Ptr{Cvoid}, b::Ptr{Cvoid})::UInt64
rb_and(a, b)  = @ccall LIB.bp_roaring_and_h(a::Ptr{Cvoid}, b::Ptr{Cvoid})::UInt64
rb_con(a, q)  = @ccall LIB.bp_roaring_contains_h(a::Ptr{Cvoid}, q::Ptr{UInt32}, length(q)::Csize_t)::UInt64
@noinline jl_or(a, b)  = length(union(a, b))
@noinline jl_and(a, b) = length(intersect(a, b))
@noinline function jl_con(s, q); c = 0; for x in q; (x in s) && (c += 1); end; c; end

Random.seed!(1); gen(n, hi) = rand(UInt32(1):UInt32(hi), n)
ms(b) = median(b).time * 1e3

function density(name, n, hi)
    a = gen(n, hi); b = gen(n, hi); q = gen(n, hi)
    abs_ = BitSet(a); bbs = BitSet(b); ha = rb_build(a); hb = rb_build(b)
    @assert jl_or(abs_, bbs) == rb_or(ha, hb) && jl_and(abs_, bbs) == rb_and(ha, hb) && jl_con(abs_, q) == rb_con(ha, q)
    println("=== $name ($n over $hi) ===")
    @printf("  union     julia %.4f ms  roaring %.4f ms  parity %.2f\n", ms(@be jl_or(abs_, bbs) seconds=2),  ms(@be rb_or(ha, hb) seconds=2),  ms(@be rb_or(ha, hb) seconds=2)  / ms(@be jl_or(abs_, bbs) seconds=2))
    @printf("  intersect julia %.4f ms  roaring %.4f ms  parity %.2f\n", ms(@be jl_and(abs_, bbs) seconds=2), ms(@be rb_and(ha, hb) seconds=2), ms(@be rb_and(ha, hb) seconds=2) / ms(@be jl_and(abs_, bbs) seconds=2))
    @printf("  contains  julia %.4f ms  roaring %.4f ms  parity %.2f\n", ms(@be jl_con(abs_, q) seconds=2),   ms(@be rb_con(ha, q) seconds=2),  ms(@be rb_con(ha, q) seconds=2)  / ms(@be jl_con(abs_, q) seconds=2))
    @printf("  [build]   julia %.4f ms  roaring %.4f ms  (build+free, FYI)\n", ms(@be BitSet(a) seconds=2), ms(@be (h = rb_build(a); rb_free(h)) seconds=2))
    rb_free(ha); rb_free(hb)
end

density("dense",  100_000, 10^6)
density("sparse",  10_000, 10^8)
