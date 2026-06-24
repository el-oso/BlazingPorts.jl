@testitem "intformat" tags = [:intformat] begin
    using BlazingPorts.IntFormat
    using Random
    buf = Vector{UInt8}(undef, 24)
    fmt(x) = (n = format_int!(buf, x); String(buf[1:n]))

    # Randomized vs string() oracle — full Int64 range (≈50% negative, all magnitudes).
    Random.seed!(20260624)
    ok = true
    for _ in 1:100_000
        x = rand(Int64)
        if fmt(x) != string(x)
            ok = false; @info "mismatch" x got = fmt(x); break
        end
    end
    @test ok

    # Edge cases incl. sign boundaries, powers of 10, chunk boundaries, typemin/typemax.
    for x in Int64[0, 1, -1, 9, -9, 10, -10, 99, -99, 100, -100, 999, 1000, 9999, 10000, 99999,
                   100000, 10^7, 10^8 - 1, 10^8, 10^16, 10^18, typemax(Int64), typemin(Int64)]
        @test fmt(x) == string(x)
    end
    # Unsigned + narrower integer types.
    for x in UInt64[0, 1, 99, typemax(UInt64), UInt64(10)^19]
        @test fmt(x) == string(x)
    end
    @test fmt(Int32(-12345)) == "-12345"
    @test fmt(UInt8(255)) == "255"

    # Convenience String API.
    @test format_int(-42) == "-42"
    @test format_int(typemin(Int64)) == string(typemin(Int64))
end

@testitem "intformat_strictmode" tags = [:intformat] begin
    # The formatter must be allocation-free and type-stable (concrete Int return — deliberately no
    # Union, contrast StrictMode F21). itoa is scalar/branchless, so @assert_vectorized does not apply.
    using BlazingPorts.IntFormat
    using StrictMode, AllocCheck, JET
    buf = Vector{UInt8}(undef, 24)
    format_int!(buf, 12345)  # warm

    @assert_typestable format_int!(buf, 12345)
    @assert_noalloc format_int!(buf, typemin(Int64))     # longest (negative) path
    @assert_noalloc format_int!(buf, typemax(UInt64))
    @test (@allocated format_int!(buf, rand(Int64))) == 0
end
