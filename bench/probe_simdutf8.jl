# Probe: Rust `simdutf8` (lemire SIMD UTF-8 validation — range checks + pshufb shuffles, NOT arithmetic
# SIMD) vs Julia `isvalid(::String)` (scalar) and Rust std `from_utf8` (scalar). Single-thread.
#   RAYON_NUM_THREADS=1 taskset -c 11 julia -O3 -t 1 --project=bench bench/probe_simdutf8.jl
# Two corpora: all-ASCII (validation fast-path) and mixed UTF-8 (real multibyte validation — where the
# SIMD range/shuffle work matters). Validation throughput = bytes / median time. The point of this probe:
# UTF-8 validation is a SHUFFLE/LOOKUP-dominated SIMD kernel with ~zero arithmetic intensity — a clean
# test of where Julia's scalar `isvalid` stands vs SIMD, and (if ported) of how StrictMode characterizes
# non-arithmetic SIMD.
include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Printf, Random
Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

function gen_ascii(target::Int)
    rng = Xoshiro(0xA5C11)
    String(rand(rng, UInt8['a':'z'; ' '; '0':'9'], target))
end
function gen_mixed(target::Int)
    rng = Xoshiro(0x52178)
    # valid UTF-8 with ~1-in-6 multibyte chars (Latin accents 2B, CJK 3B) so validation must do real work
    pool = Char['a':'z'; ' '; 'é'; 'ñ'; 'ü'; 'ß'; '一'; '二'; '三']
    io = IOBuffer()
    while io.size < target; print(io, rand(rng, pool)); end
    String(take!(io))
end

import BlazingPorts.Utf8: isvalid_utf8
@noinline jl_valid(s::String) = isvalid(s)
@noinline ours_valid(b::Vector{UInt8}) = isvalid_utf8(b)
@noinline rs_simd(b::Vector{UInt8}) =
    GC.@preserve b ccall((:bp_simdutf8_validate, LIB), Bool, (Ptr{UInt8}, Csize_t), b, length(b))
@noinline rs_std(b::Vector{UInt8}) =
    GC.@preserve b ccall((:bp_std_validate, LIB), Bool, (Ptr{UInt8}, Csize_t), b, length(b))

probes = Probe[]
for (cname, str) in (("ASCII", gen_ascii(16 * 1024 * 1024)), ("mixed UTF-8", gen_mixed(16 * 1024 * 1024)))
    bytes = Vector{UInt8}(codeunits(str)); nb = length(bytes)
    # sanity: all three agree it's valid
    (jl_valid(str) && ours_valid(bytes) && rs_simd(bytes) && rs_std(bytes)) || @warn "validation disagreement ($cname)"
    g(p) = nb / p.median / 1e9
    println("\n=== UTF-8 validation, $cname, $(round(nb/1024^2; digits=2)) MiB, single-thread ===")
    p_jl   = run_probe("Julia isvalid (scalar): $cname", () -> jl_valid(str);    seconds = 4.0)
    p_ours = run_probe("BlazingPorts.Utf8 (SIMD): $cname", () -> ours_valid(bytes); seconds = 4.0)
    p_sj   = run_probe("simdutf8 (SIMD): $cname",         () -> rs_simd(bytes);   seconds = 4.0)
    p_std  = run_probe("Rust std (scalar): $cname",       () -> rs_std(bytes);    seconds = 4.0)
    @printf("  Julia isvalid %.2f  |  OURS %.2f  |  simdutf8 %.2f  |  Rust std %.2f GB/s  → ours/Base %.1f×, ours/simdutf8 %.2f×\n",
        g(p_jl), g(p_ours), g(p_sj), g(p_std), g(p_ours) / g(p_jl), g(p_ours) / g(p_sj))
    push!(probes, p_jl, p_ours, p_std, p_sj)
end

save_probes("simdutf8", probes)
println("\nsaved → bench/results/simdutf8.json")
