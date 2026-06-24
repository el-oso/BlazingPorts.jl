# Probe: our SIMD find_substr vs Base findfirst vs the memchr::memmem crate. Worst-case scan
# (needle only at the very end). Run: taskset -c 2 julia -t1 --project=bench bench/probe_substr.jl
using Chairmarks, Printf, Random
import Chairmarks: median
using BlazingPorts.StringSearch
const LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")
rs_memmem(h, n) = @ccall LIB.bp_memmem(h::Ptr{UInt8}, length(h)::Csize_t, n::Ptr{UInt8}, length(n)::Csize_t)::Cssize_t
base_find(h, p) = (r = findfirst(p, h); isnothing(r) ? 0 : first(r))

const N = 32 * 1024 * 1024
ms(b) = median(b).time * 1e3
Random.seed!(1)

for m in (2, 8, 32)
    needle = rand(UInt8(33):UInt8(126), m)              # printable, unlikely to collide with filler
    hay = fill(0x20, N); hay[end-m+1:end] .= needle     # match only at the very end (full scan)
    @noinline fj(h, p) = base_find(h, p)
    @noinline fo(h, p) = find_substr(h, p)
    @noinline fr(h, p) = rs_memmem(h, p)
    @assert fo(hay, needle) == N - m + 1 == fr(hay, needle) + 1   # ours 1-based, rust 0-based
    tj = ms(@be fj(hay, needle) seconds=3)
    to = ms(@be fo(hay, needle) seconds=3)
    tr = ms(@be fr(hay, needle) seconds=3)
    @printf("m=%-2d  base %.3f ms   ours %.3f ms   memmem %.3f ms   |  ours/base %.1fx  parity(memmem/ours) %.2f\n",
            m, tj, to, tr, tj/to, tr/to)
end
