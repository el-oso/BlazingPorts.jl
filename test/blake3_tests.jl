@testitem "blake3_correctness" tags = [:blake3] begin
    using BlazingPorts.Blake3: blake3

    # Official BLAKE3 test vectors. Input encoding: byte i = i % 251 (official test convention).
    # Lengths cross all block (64B) and chunk (1024B) boundaries.
    make_input(n) = [UInt8(i % 251) for i in 0:n-1]

    VECTORS = [
        (0,      "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
        (1,      "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
        (63,     "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"),
        (64,     "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"),
        (65,     "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"),
        (1023,   "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"),
        (1024,   "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"),
        (1025,   "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"),
        (2048,   "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"),
        (102400, "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085"),
        # Additional odd-size vectors (verified against bp_blake3 Rust crate)
        (1500,    "e71e67e776854018461734062301034d08f1be79d8fa59ae1daf61eda2c65318"),
        (17408,   "993924ff3dcbd868be9cf3fed98d4538fe579ffccf390a5aa1ddba0f6a20bfed"),
        (1000000, "5e82c663d164c54e4fcdfcd70e3ca464662228bdbad45cce2e0c2bff999064ef"),
    ]

    # Test every official vector: byte-exact
    for (n, expected_hex) in VECTORS
        data = make_input(n)
        got = bytes2hex(blake3(data))
        @test got == expected_hex
    end

    # Multi-chunk correctness via self-agreement (Blake3Hash.jl as oracle)
    using Blake3Hash
    function jl_ref(data)
        ctx = Blake3Ctx(); update!(ctx, data); digest(ctx)
    end

    # Spot-check that our output matches Blake3Hash.jl at several chunk counts.
    for nchunks in [1, 2, 4, 7, 8, 9, 15, 16, 17, 32, 64]
        data = make_input(nchunks * 1024)
        @test blake3(data) == jl_ref(data)
    end

    # 1 MiB agreement with Blake3Hash.jl
    data_1m = make_input(1024 * 1024)
    @test blake3(data_1m) == jl_ref(data_1m)
end

@testitem "blake3_strictmode" tags = [:blake3] begin
    # The SIMD compress kernel must vectorize. The public blake3() must be type-stable.
    # We test the SIMD inner kernel directly (the per-chunk compression for N=8 chunks).
    import BlazingPorts.Blake3 as B3
    using StrictMode, AllocCheck, JET

    const N = 8
    data = [UInt8(i % 251) for i in 0:(N * B3.CHUNK_LEN - 1)]

    GC.@preserve data begin
        p = pointer(data)
        ptrs = ntuple(k -> p + (k-1)*B3.CHUNK_LEN, Val(N))

        # Warm up JIT
        B3._compress_N_chunks_full(ptrs,
            B3.KEY1,B3.KEY2,B3.KEY3,B3.KEY4,B3.KEY5,B3.KEY6,B3.KEY7,B3.KEY8, UInt64(0))

        # The SIMD hash_many kernel must vectorize (<8 x i32> in the LLVM IR).
        @assert_vectorized B3._compress_N_chunks_full(ptrs,
            B3.KEY1,B3.KEY2,B3.KEY3,B3.KEY4,B3.KEY5,B3.KEY6,B3.KEY7,B3.KEY8, UInt64(0))

        # The scalar compress must be allocation-free.
        @assert_noalloc B3.compress(
            B3.KEY1,B3.KEY2,B3.KEY3,B3.KEY4,B3.KEY5,B3.KEY6,B3.KEY7,B3.KEY8,
            UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),
            UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),
            UInt32(0),UInt32(0),UInt32(64),UInt32(0))
    end

    # Public API: type-stable (returns Vector{UInt8})
    @assert_typestable B3.blake3(data)
end
