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

@assert bumper_alloc(N) == vec_arena(N) == naive_heap(N) == rs_bump(N)   # correctness (same checksum)
ns(b) = median(b).time * 1e9 / N
al(f) = @allocated f(N)
@printf("bumpalo (Rust)     %.2f ns/obj\n",                  ns(@be rs_bump(N)     seconds=3))
@printf("Bumper.jl          %.2f ns/obj   %d alloc/call\n",  ns(@be bumper_alloc(N) seconds=3), al(bumper_alloc))
@printf("Vector{T} arena    %.2f ns/obj   %.1f MB/call\n",   ns(@be vec_arena(N)   seconds=3), al(vec_arena)/2^20)
@printf("naive heap (M3)    %.2f ns/obj   %.1f MB/call\n",   ns(@be naive_heap(N)  seconds=3), al(naive_heap)/2^20)
