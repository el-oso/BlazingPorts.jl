# Probe: Blake3Hash.jl (pure-Julia BLAKE3, no SIMD) vs the blake3 Rust crate.
# BLAKE3's performance comes entirely from SIMD hash_many — processing N independent
# blocks/chunks in parallel (AVX2=8-wide, AVX-512=16-wide). Blake3Hash.jl is a scalar
# SVector-based implementation (README: "needs SIMD updates"). Measure the gap, then
# decide: document-skip (unlikely) or Phase B SIMD hash_many port.
#
# Correctness: both sides produce byte-exact output matching the official BLAKE3
# test vectors (same input encoding: byte i = i % 251).
#
# DCE defeat: both sides consume the 32-byte digest via sum()/xor fold + Base.donotdelete.
# The Rust shim writes the digest to an out-pointer (externally visible; no elision).
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_blake3.jl
#   (build Rust lib first: bash bench/rust_compare/build.sh)

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Blake3Hash
using Printf

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

# ── inputs ──────────────────────────────────────────────────────────────────────────────────────────
# Official BLAKE3 test input convention: byte i = i % 251.
const BUF_1MIB  = [UInt8(i % 251) for i in 0:(1*1024*1024 - 1)]
const BUF_16MIB = [UInt8(i % 251) for i in 0:(16*1024*1024 - 1)]

# Preallocate the output buffer (Rust writes here; Julia will sum it for DCE defeat).
const OUT32 = Vector{UInt8}(undef, 32)

# ── wrappers ─────────────────────────────────────────────────────────────────────────────────────────
@noinline function jl_hash_1m()
    ctx = Blake3Ctx()
    update!(ctx, BUF_1MIB)
    h = digest(ctx)
    Base.donotdelete(h)
    return sum(h)   # DCE sink: consume all 32 bytes
end

@noinline function jl_hash_16m()
    ctx = Blake3Ctx()
    update!(ctx, BUF_16MIB)
    h = digest(ctx)
    Base.donotdelete(h)
    return sum(h)
end

@noinline function rust_hash_1m()
    GC.@preserve BUF_1MIB OUT32 begin
        ccall((:bp_blake3, LIB), Cvoid,
            (Ptr{UInt8}, Csize_t, Ptr{UInt8}), BUF_1MIB, length(BUF_1MIB), OUT32)
        Base.donotdelete(OUT32)
        return sum(OUT32)
    end
end

@noinline function rust_hash_16m()
    GC.@preserve BUF_16MIB OUT32 begin
        ccall((:bp_blake3, LIB), Cvoid,
            (Ptr{UInt8}, Csize_t, Ptr{UInt8}), BUF_16MIB, length(BUF_16MIB), OUT32)
        Base.donotdelete(OUT32)
        return sum(OUT32)
    end
end

# ── correctness vs official BLAKE3 test vectors ──────────────────────────────────────────────────────
# Input encoding: byte i = i % 251 (BLAKE3 official test convention).
make_input(n) = [UInt8(i % 251) for i in 0:n-1]

const VECTORS = [
    (0,    "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
    (1,    "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
    (63,   "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"),
    (64,   "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"),
    (65,   "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"),
    (1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"),
    (1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"),
    (1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"),
    (2048, "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"),
]

println("\n=== blake3 correctness vs official test vectors ===")
function hex_digest_jl(data)
    ctx = Blake3Ctx(); update!(ctx, data); join(string.(digest(ctx), base=16, pad=2))
end
function hex_digest_rust(data)
    out = Vector{UInt8}(undef, 32)
    GC.@preserve data out ccall((:bp_blake3, LIB), Cvoid,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}), data, length(data), out)
    join(string.(out, base=16, pad=2))
end

correct_results = map(VECTORS) do (n, expected)
    data = make_input(n)
    jl_hex   = hex_digest_jl(data)
    rust_hex = hex_digest_rust(data)
    jl_ok   = jl_hex   == expected
    rust_ok = rust_hex == expected
    agree   = jl_hex   == rust_hex
    ok = jl_ok && rust_ok && agree
    @printf("  n=%-6d Julia=%-4s Rust=%-4s agree=%-4s  %s\n",
        n, jl_ok ? "OK" : "FAIL", rust_ok ? "OK" : "FAIL", agree ? "YES" : "NO",
        ok ? "" : "  ← MISMATCH")
    ok
end
all(correct_results) || error("Correctness check FAILED — do not trust throughput numbers")
println("  All correct ✓ (byte-exact vs official BLAKE3 test vectors, both sides agree)")

# ── throughput probe: 1 MiB ──────────────────────────────────────────────────────────────────────────
# Warm up JIT
jl_hash_1m(); rust_hash_1m()

println("\n=== blake3: 1 MiB buffer, single-thread ===")
p1_jl   = run_probe("Blake3Hash.jl",  jl_hash_1m;   seconds = 4.0)
p1_rust = run_probe("blake3 crate",   rust_hash_1m; seconds = 4.0)

# ── throughput probe: 16 MiB ─────────────────────────────────────────────────────────────────────────
jl_hash_16m(); rust_hash_16m()

println("\n=== blake3: 16 MiB buffer, single-thread ===")
p16_jl   = run_probe("Blake3Hash.jl",  jl_hash_16m;   seconds = 4.0)
p16_rust = run_probe("blake3 crate",   rust_hash_16m; seconds = 4.0)

# ── report ───────────────────────────────────────────────────────────────────────────────────────────
for (label, n_bytes, p_jl, p_rust) in [
        ("1 MiB",  1*1024*1024,  p1_jl,  p1_rust),
        ("16 MiB", 16*1024*1024, p16_jl, p16_rust)]
    gbps_jl   = n_bytes / p_jl.median   / 1e9
    gbps_rust = n_bytes / p_rust.median / 1e9
    ns_jl     = p_jl.median  * 1e9 / n_bytes
    ns_rust   = p_rust.median * 1e9 / n_bytes
    par       = Harness.parity(p_jl.median, p_rust.median)
    verdict   = par >= Harness.PARITY_GATE ? "GOOD ENOUGH (document-skip)" : "BELOW GATE → port justified"
    println("\n── $label ──")
    @printf("  Blake3Hash.jl  %6.2f GB/s  %.3f ns/byte  (median %.2f ms, rel-σ %.1f%%)\n",
        gbps_jl, ns_jl, 1e3*p_jl.median, 100*p_jl.relσ)
    @printf("  blake3 crate   %6.2f GB/s  %.3f ns/byte  (median %.2f ms, rel-σ %.1f%%)\n",
        gbps_rust, ns_rust, 1e3*p_rust.median, 100*p_rust.relσ)
    @printf("  parity (rust/julia) = %.3f   %s\n", par, verdict)
end

# ── persist + plot (head-to-head 1 MiB — the canonical single number) ──────────────────────────────
headtohead_1m = Probe[p1_jl, p1_rust]
save_probes("blake3", headtohead_1m)
out = plot_probe("blake3", headtohead_1m)
println("\n  plot → $out")
println("\nDone — probe_blake3.jl")
