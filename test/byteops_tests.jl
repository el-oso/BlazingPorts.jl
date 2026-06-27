@testitem "base64_encode_correctness" tags = [:byteops] begin
    using BlazingPorts.ByteOps: base64_encode, base64_encode!
    using Base64, Random

    # Byte-exact with Base64.base64encode across all lengths (block boundaries + the 1/2-byte padding tails)
    for L in 0:200
        b = rand(UInt8, L); @test base64_encode(b) == base64encode(b)
    end
    Random.seed!(0xB64E)
    for _ in 1:20000
        b = rand(UInt8, rand(0:500)); @test base64_encode(b) == base64encode(b)
    end
    for _ in 1:2000
        b = rand(UInt8, rand(500:4000)); @test base64_encode(b) == base64encode(b)
    end
    @test base64_encode("hello, world") == base64encode("hello, world")

    # preallocated kernel path agrees and is allocation-free
    big = rand(UInt8, 64 * 1024); out = Vector{UInt8}(undef, cld(length(big), 3) * 4)
    base64_encode!(out, big)
    @test String(copy(out)) == base64encode(big)
    base64_encode!(out, big)
    @test (@allocated base64_encode!(out, big)) == 0
end

@testitem "hex_encode_correctness" tags = [:byteops] begin
    using BlazingPorts.ByteOps: hex_encode, hex_encode!
    using Random
    for L in 0:200
        b = rand(UInt8, L); @test hex_encode(b) == bytes2hex(b)
    end
    Random.seed!(0x4EE)
    for _ in 1:20000
        b = rand(UInt8, rand(0:400)); @test hex_encode(b) == bytes2hex(b)
    end
    big = rand(UInt8, 64 * 1024); out = Vector{UInt8}(undef, 2 * length(big))
    hex_encode!(out, big); @test String(copy(out)) == bytes2hex(big)
    hex_encode!(out, big); @test (@allocated hex_encode!(out, big)) == 0
end

@testitem "hex_decode_correctness" tags = [:byteops] begin
    using BlazingPorts.ByteOps: hex_decode
    using Random
    Random.seed!(0xDEC0)
    for _ in 1:20000
        b = rand(UInt8, rand(0:400)); @test hex_decode(bytes2hex(b)) == b
    end
    @test hex_decode("DEADBEEF") == hex2bytes("DEADBEEF")        # uppercase
    @test_throws ArgumentError hex_decode("xy")                  # invalid char (scalar path)
    @test_throws ArgumentError hex_decode("deadbeefdeadbeeZ")    # invalid char (SIMD path)
    @test_throws ArgumentError hex_decode("abc")                 # odd length
end

@testitem "base64_strictmode" tags = [:byteops] begin
    # The base64 block kernel is a SHUFFLE/LOOKUP kernel (vpshufb reshuffle + vpshufb LUT translate +
    # vpmulhuw bit-spread). Audit it AND re-probe F33 (kernel_report blindness to shuffle ops) at <32 x i8>.
    import BlazingPorts.ByteOps as B
    using SIMD: Vec
    using StrictMode, AllocCheck, JET

    x = Vec{32,UInt8}(ntuple(i -> UInt8(i + 33), 32))
    B._enc24(x)                                                  # warm

    @assert_vectorized B._enc24(x)
    @assert_noalloc    B._enc24(x)
    @assert_typestable B._enc24(x)

    if isdefined(StrictMode, :kernel_report)
        @info "kernel_report on the base64 shuffle kernel (F33 probe)" report = sprint(show,
            StrictMode.kernel_report(B._enc24, (Vec{32,UInt8},)))
    end
end
