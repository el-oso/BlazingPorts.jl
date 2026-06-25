# Probe: Blake3Hash.jl (pure-Julia scalar) vs blake3 Rust crate vs BlazingPorts.Blake3 (SIMD).
# BLAKE3's performance comes from SIMD hash_many — N independent chunks in parallel (AVX2=8-wide).
# Phase A showed Blake3Hash.jl at 0.084× the crate (11× gap). Phase B ports the SIMD kernel.
#
# Three contenders:
#   1. Blake3Hash.jl    — scalar SVector (ecosystem baseline, no SIMD)
#   2. blake3 crate     — Rust reference with AVX2/AVX-512 hash_many (the target)
#   3. BlazingPorts.Blake3 — our SIMD port, Vec{8,UInt32} hash_many (this session)
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_blake3.jl
#   (build Rust lib first: bash bench/rust_compare/build.sh)

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Blake3Hash
using BlazingPorts.Blake3: blake3 as our_blake3
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

# ── wrappers for BlazingPorts.Blake3 ─────────────────────────────────────────────────────────────
@noinline function our_hash_1m()
    h = our_blake3(BUF_1MIB)
    Base.donotdelete(h)
    return sum(h)
end

@noinline function our_hash_16m()
    h = our_blake3(BUF_16MIB)
    Base.donotdelete(h)
    return sum(h)
end

# Also verify our impl against official vectors
function hex_digest_ours(data)
    join(string.(our_blake3(data), base=16, pad=2))
end

println("\n=== BlazingPorts.Blake3 correctness (SIMD hash_many) ===")
ours_correct = map(VECTORS) do (n, expected)
    data = make_input(n)
    got = hex_digest_ours(data)
    ok = (got == expected)
    ok || @printf("  FAIL n=%d: got=%s exp=%s\n", n, got, expected)
    ok
end
all(ours_correct) ? println("  All correct ✓ (byte-exact vs official BLAKE3 test vectors)") :
                    error("BlazingPorts.Blake3 correctness FAILED")

# ── throughput probe: 1 MiB ──────────────────────────────────────────────────────────────────────────
jl_hash_1m(); rust_hash_1m(); our_hash_1m()  # warm up

println("\n=== blake3: 1 MiB buffer, single-thread ===")
p1_jl   = run_probe("Blake3Hash.jl",  jl_hash_1m;  seconds = 4.0)
p1_rust = run_probe("blake3 crate",   rust_hash_1m; seconds = 4.0)
p1_ours = run_probe("ours (BP.Blake3)", our_hash_1m; seconds = 4.0)

# ── throughput probe: 16 MiB ─────────────────────────────────────────────────────────────────────────
jl_hash_16m(); rust_hash_16m(); our_hash_16m()

println("\n=== blake3: 16 MiB buffer, single-thread ===")
p16_jl   = run_probe("Blake3Hash.jl",  jl_hash_16m;  seconds = 4.0)
p16_rust = run_probe("blake3 crate",   rust_hash_16m; seconds = 4.0)
p16_ours = run_probe("ours (BP.Blake3)", our_hash_16m; seconds = 4.0)

# ── report ───────────────────────────────────────────────────────────────────────────────────────────
for (label, n_bytes, p_jl, p_rust, p_ours) in [
        ("1 MiB",  1*1024*1024,  p1_jl,  p1_rust, p1_ours),
        ("16 MiB", 16*1024*1024, p16_jl, p16_rust, p16_ours)]
    gbps(p) = n_bytes / p.median / 1e9
    nsb(p)  = p.median * 1e9 / n_bytes
    par_jl  = Harness.parity(p_jl.median,  p_rust.median)
    par_ours = Harness.parity(p_ours.median, p_rust.median)
    println("\n── $label ──")
    @printf("  %-24s %6.2f GB/s  %.3f ns/byte  (median %.2f ms, rel-σ %.1f%%)\n",
        "Blake3Hash.jl",    gbps(p_jl),   nsb(p_jl),   1e3*p_jl.median,   100*p_jl.relσ)
    @printf("  %-24s %6.2f GB/s  %.3f ns/byte  (median %.2f ms, rel-σ %.1f%%)\n",
        "ours (BP.Blake3)", gbps(p_ours), nsb(p_ours), 1e3*p_ours.median, 100*p_ours.relσ)
    @printf("  %-24s %6.2f GB/s  %.3f ns/byte  (median %.2f ms, rel-σ %.1f%%)\n",
        "blake3 crate",     gbps(p_rust), nsb(p_rust), 1e3*p_rust.median, 100*p_rust.relσ)
    @printf("  parity Blake3Hash.jl / crate = %.3f\n", par_jl)
    @printf("  parity ours / crate          = %.3f   %s\n", par_ours,
        par_ours >= Harness.PARITY_GATE ? "≥ 0.96 PARITY" : "< 0.96 gap remains")
end

# ── persist + plot (three-way 1 MiB — canonical probe) ──────────────────────────────────────────────
threeway = Probe[p1_jl, p1_ours, p1_rust]
save_probes("blake3", threeway)
out = plot_probe("blake3", threeway)
println("\n  plot → $out")
println("\nDone — probe_blake3.jl")
