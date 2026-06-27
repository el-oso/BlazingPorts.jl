# Probe: PORTED byte-ops kernels (BlazingPorts.ByteOps) vs the Rust crates, **kernel-only** — the FAIR
# comparison (both sides write into a preallocated buffer; no output allocation in the timed region).
#   RAYON_NUM_THREADS=1 taskset -c 11 julia -O3 -t 1 --project=bench bench/probe_byteops_ports.jl
# The campaign lesson this encodes: base64 encode probed "27× slower" only because the allocating form
# timed a 21 MiB Vector + String() + GC inside the loop. Isolate the *kernel* (discipline #2) and the
# pure-Julia SIMD kernel beats the crate. Julia stdlib (Base64.base64encode / bytes2hex, allocating) is
# shown as the realistic scalar baseline.
include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Printf, Random, Base64
Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

import BlazingPorts.ByteOps: base64_encode!, hex_encode!, hex_decode!
const N = 12 * 1024 * 1024
const data = rand(Xoshiro(0xB17E0), UInt8, N)
const b64out = Vector{UInt8}(undef, cld(N, 3) * 4)
const hexout = Vector{UInt8}(undef, 2N)
const hexin = Vector{UInt8}(codeunits(bytes2hex(data)))    # 2N hex chars → N bytes
const decout = Vector{UInt8}(undef, N)

@noinline j_b64(d) = base64encode(d)                       # Julia stdlib baseline (allocating)
@noinline o_b64(d) = base64_encode!(b64out, d)             # ours, preallocated kernel
@noinline r_b64(d) = GC.@preserve d b64out ccall((:bp_base64_encode, LIB), Csize_t,
    (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t), d, length(d), b64out, length(b64out))
@noinline j_hex(d) = bytes2hex(d)
@noinline o_hex(d) = hex_encode!(hexout, d)
@noinline r_hex(d) = GC.@preserve d hexout ccall((:bp_hex_encode, LIB), Bool,
    (Ptr{UInt8}, Csize_t, Ptr{UInt8}), d, length(d), hexout)
@noinline j_hexd(_) = hex2bytes(hexin)
@noinline o_hexd(_) = hex_decode!(decout, hexin)
@noinline r_hexd(_) = GC.@preserve hexin decout ccall((:bp_hex_decode, LIB), Bool,
    (Ptr{UInt8}, Csize_t, Ptr{UInt8}), hexin, length(hexin), decout)

# correctness gates
o_b64(data); String(copy(b64out)) == base64encode(data) || error("base64 encode mismatch")
o_hex(data); String(copy(hexout)) == bytes2hex(data) || error("hex encode mismatch")
o_hexd(data); decout == data || error("hex decode mismatch")

probes = Probe[]
g(p) = N / p.median / 1e9
for (op, jl, ours, rust, crate) in (
    ("base64 encode", j_b64,  o_b64,  r_b64,  "base64-simd"),
    ("hex encode",    j_hex,  o_hex,  r_hex,  "faster-hex"),
    ("hex decode",    j_hexd, o_hexd, r_hexd, "faster-hex"))
    println("\n=== $op, $(N ÷ 1024^2) MiB, single-thread, KERNEL-ONLY (preallocated) ===")
    p_j = run_probe("Julia stdlib (scalar, alloc): $op", () -> jl(data);   seconds = 4.0)
    p_o = run_probe("BlazingPorts.ByteOps (SIMD): $op",   () -> ours(data); seconds = 4.0)
    p_r = run_probe("$crate (SIMD kernel): $op",          () -> rust(data); seconds = 4.0)
    @printf("  Julia %.2f  |  OURS %.2f  |  %s %.2f GB/s  → ours/Julia %.1f×, ours/%s %.2f×\n",
        g(p_j), g(p_o), crate, g(p_r), g(p_o) / g(p_j), crate, g(p_o) / g(p_r))
    push!(probes, p_j, p_o, p_r)
end
save_probes("byteops_ports", probes)
println("\nsaved → bench/results/byteops_ports.json")
