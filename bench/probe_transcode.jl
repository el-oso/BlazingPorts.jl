# Probe: SIMD transcoding — base64-simd & faster-hex (pshufb lookup) vs Julia Base64/bytes2hex (scalar).
# Both sides allocate the output String (fair idiomatic comparison; isolates the SIMD-vs-scalar transform).
include(joinpath(@__DIR__, "harness.jl")); using .Harness; using Printf, Random, Base64
Harness.single_thread!(); const LIB = Harness.RUST_LIB
const DATA = rand(Xoshiro(0x64ABC), UInt8, 12*1024*1024); const NB = length(DATA)
@noinline jl_b64()  = base64encode(DATA)
@noinline jl_hex()  = bytes2hex(DATA)
@noinline rs_b64()  = GC.@preserve DATA ccall((:bp_base64_encode_alloc, LIB), Csize_t, (Ptr{UInt8}, Csize_t), DATA, NB)
@noinline rs_hex()  = GC.@preserve DATA ccall((:bp_hex_encode_alloc, LIB), Csize_t, (Ptr{UInt8}, Csize_t), DATA, NB)
length(jl_b64()) == Int(rs_b64()) || @warn "base64 length mismatch"
length(jl_hex()) == Int(rs_hex()) || @warn "hex length mismatch"
g(p) = NB / p.median / 1e9; probes = Probe[]
println("\n=== SIMD transcoding (encode), $(NB>>20) MiB input, single-thread ===")
for (nm, jf, rf) in (("base64", jl_b64, rs_b64), ("hex", jl_hex, rs_hex))
    pj = run_probe("Julia $nm", jf; seconds=4.0); pr = run_probe("Rust SIMD $nm", rf; seconds=4.0)
    @printf("  %-7s Julia %.2f  Rust SIMD %.2f GB/s  → Julia/Rust = %.2f×\n", nm, g(pj), g(pr), g(pj)/g(pr))
    push!(probes, pj, pr)
end
save_probes("transcode", probes); println("saved → bench/results/transcode.json")
