# Probe: bump/arena allocation. Allocate N 24-byte objects into an arena, touch them, bulk-free.
# Contenders: Rust `bumpalo` vs Julia `Bumper.jl` (the direct analogue) vs a hand-rolled `Vector{T}`
# arena vs naive per-object heap allocation. Reports ns/object + per-call allocations (the zero-GC
# question). DCE defeated via the checksum sink + Base.donotdelete.
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_bumpalo.jl  (build lib: bench/rust_compare/build.sh)
using Chairmarks, Printf, Bumper
import Chairmarks: median
const LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")
const N = 1_000_000

struct T3; a::UInt64; b::UInt64; c::UInt64; end          # 24-byte immutable (value type)
mutable struct M3; a::UInt64; b::UInt64; c::UInt64; end  # mutable ⇒ heap-allocated per object

@noinline function bumper_alloc(n)
    acc = UInt64(0)
    @no_escape begin
        a = @alloc(T3, n)                                # one bump from the slab buffer (reused, zero-GC)
        @inbounds for i in 1:n; u = UInt64(i - 1);a[i] = T3(u, u * 2, u * 3); end
        @inbounds for i in 1:n; t = a[i]; acc ⊻= t.a ⊻ t.b ⊻ t.c; end
    end
    acc
end
@noinline function vec_arena(n)
    a = Vector{T3}(undef, n); acc = UInt64(0)            # allocates n*24 B every call (GC-tracked)
    @inbounds for i in 1:n; u = UInt64(i - 1);a[i] = T3(u, u * 2, u * 3); end
    @inbounds for i in 1:n; t = a[i]; acc ⊻= t.a ⊻ t.b ⊻ t.c; end
    Base.donotdelete(a); acc
end
@noinline function naive_heap(n)
    a = Vector{M3}(undef, n); acc = UInt64(0)            # N individual heap allocations (the thing arenas avoid)
    @inbounds for i in 1:n; u = UInt64(i - 1);a[i] = M3(u, u * 2, u * 3); end
    @inbounds for i in 1:n; m = a[i]; acc ⊻= m.a ⊻ m.b ⊻ m.c; end
    Base.donotdelete(a); acc
end
rs_bump(n) = @ccall LIB.bp_bumpalo_alloc3(n::Csize_t)::UInt64

# Bumper doing N INDIVIDUAL bumps (apples-to-apples with bumpalo's per-object alloc, vs the array idiom).
@noinline function bumper_indiv(n)
    acc = UInt64(0)
    @no_escape begin
        @inbounds for i in 1:n
            p = @alloc(T3, 1); u = UInt64(i - 1); p[1] = T3(u, u * 2, u * 3)
            t = p[1]; acc ⊻= t.a ⊻ t.b ⊻ t.c
        end
    end
    acc
end

# SIZE SWEEP — the single-N=1M number was memory-bandwidth-bound (a trivial tie). The allocator only
# shows at cache-resident sizes; at DRAM scale it's bandwidth → parity. CAVEAT: the bumpalo shim has a
# `black_box` barrier per object (DCE defense) that Bumper lacks, so Bumper's small-N edge is partly a
# measurement asymmetry. The un-confounded result is the ALLOC COUNT: Bumper = 0 GC alloc/call at every
# size (true zero-GC slab reuse, even at 1+ GB). naive per-object heap is the contrast (~13× + GC).
nsp(b, n) = median(b).time * 1e9 / n
for n in (1_000, 100_000, 1_000_000, 10_000_000, 50_000_000)
    @assert bumper_alloc(n) == bumper_indiv(n) == rs_bump(n)
    @printf("N=%9d (%6.1f MB)  bumpalo %.2f  Bumper-indiv %.2f  Bumper-array %.2f ns/obj   Bumper alloc=%d B\n",
        n, n * 24 / 2^20, nsp(@be(rs_bump(n), seconds=2), n), nsp(@be(bumper_indiv(n), seconds=2), n),
        nsp(@be(bumper_alloc(n), seconds=2), n), @allocated bumper_alloc(n))
end
@printf("naive heap (M3) @1M  %.2f ns/obj   %.1f MB/call (GC; what arenas avoid)\n",
    nsp(@be(naive_heap(1_000_000), seconds=2), 1_000_000), (@allocated naive_heap(1_000_000)) / 2^20)
