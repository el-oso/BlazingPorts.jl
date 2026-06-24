# Probe: Julia byte/substring search vs the `memchr` crate (Tier 1 GP). Single-thread, worst-case
# (needle at the end → full scan). Run: taskset -c 2 julia -t 1 --project=bench bench/probe_memchr.jl
using Chairmarks, Printf
const LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")
rs_memchr(v, b) = @ccall LIB.bp_memchr(v::Ptr{UInt8}, length(v)::Csize_t, b::UInt8)::Cssize_t
rs_memmem(h, n) = @ccall LIB.bp_memmem(h::Ptr{UInt8}, length(h)::Csize_t, n::Ptr{UInt8}, length(n)::Csize_t)::Cssize_t

const N = 32 * 1024 * 1024                       # 32 MiB haystack
hay = fill(0x20, N); hay[end] = 0xAA             # needle byte only at the very end
jl_byte(v) = findfirst(==(0xAA), v)              # natural Julia byte search
@noinline f_jl_byte(v) = jl_byte(v)
@noinline f_rs_byte(v) = rs_memchr(v, 0xAA)
@assert f_jl_byte(hay) == N && f_rs_byte(hay) == N - 1   # 1-based vs 0-based

needle = collect(codeunits("needle_pattern_42"))
hs = fill(0x20, N); hs[end-length(needle)+1:end] .= needle
hstr = String(copy(hs)); nstr = String(copy(needle))
@noinline f_jl_mm(s, n) = findfirst(n, s)        # Julia String substring search
@noinline f_rs_mm(h, n) = rs_memmem(h, n)
@assert !isnothing(f_jl_mm(hstr, nstr)) && f_rs_mm(hs, needle) >= 0

import Chairmarks: median
ms(b) = median(b).time * 1e3
report(name, j, r) = @printf("%-10s julia %.3f ms   rust %.3f ms   parity(rust/julia) %.2f\n", name, j, r, r/j)
report("byte",      ms(@be f_jl_byte(hay) seconds=3),       ms(@be f_rs_byte(hay) seconds=3))
report("substring", ms(@be f_jl_mm(hstr, nstr) seconds=3),  ms(@be f_rs_mm(hs, needle) seconds=3))
