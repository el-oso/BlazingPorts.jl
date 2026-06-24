@testitem "stringsearch" tags = [:stringsearch] begin
    using BlazingPorts.StringSearch
    using Random

    # Base.findfirst is the oracle (returns a UnitRange or nothing).
    base(h, p) = (r = findfirst(p, h); isnothing(r) ? nothing : first(r))

    # Randomized: needles drawn from the haystack (real matches) and random (mostly misses),
    # across m below/at/above the SIMD width W=64.
    Random.seed!(20260624)
    ok = true
    for _ in 1:5000
        n = rand(1:300); m = rand(1:80); h = rand(UInt8, n)
        p = (rand(Bool) && m <= n) ? h[rand(1:max(1, n - m + 1)) .+ (0:m-1)] : rand(UInt8, m)
        if find_substr(h, p) != base(h, p)
            ok = false
            @info "mismatch" n m got=find_substr(h, p) want=base(h, p)
            break
        end
    end
    @test ok

    # Edge cases.
    @test find_substr("hello world", "world") == 7
    @test find_substr("hello world", "o") == 5              # m == 1 (Base-delegated)
    @test find_substr("hello", "xyz") === nothing
    @test find_substr("aaaa", "aa") == 1                    # overlapping repeats
    @test find_substr("abc", "abc") == 1                    # needle == haystack
    @test find_substr("abc", "abcd") === nothing            # m > n
    @test find_substr("abc", "") == 1                       # empty needle
    @test find_substr(UInt8[0x00, 0xff, 0x00], UInt8[0xff]) == 2     # non-ASCII bytes
    @test find_substr(UInt8[0x78, 0x78, 0x61, 0x62, 0x63], UInt8[0x61, 0x62, 0x63]) == 3  # vector path

    # Needle longer than the SIMD width, only at the very end (exercises the wide scan + tail).
    h = collect(codeunits("z"^200 * "NEEDLE_PATTERN_DEFINITELY_LONGER_THAN_SIXTYFOUR_BYTES_0123456789ABCDEF"))
    p = h[201:end]
    @test find_substr(h, p) == 201
end

@testitem "stringsearch_strictmode" tags = [:stringsearch] begin
    # The SIMD core must vectorize, be allocation-free, and type-stable (concrete Int return). The
    # public `find_substr` returns Union{Int,Nothing} (isbits union — type-stable in practice but not
    # isconcretetype), so the StrictMode kernel guarantees target `_find_substr`; the public entry is
    # checked allocation-free at runtime on the m ≥ 2 SIMD path.
    using BlazingPorts.StringSearch: find_substr
    import BlazingPorts.StringSearch as SS
    using StrictMode, AllocCheck, JET

    h = rand(UInt8, 4096); p = h[4000:4031]      # m = 32 match near the end
    SS._find_substr(pointer(h), length(h), pointer(p), length(p))  # warm

    GC.@preserve h p begin
        ph = pointer(h); pp = pointer(p)
        @assert_typestable SS._find_substr(ph, length(h), pp, length(p))
        @assert_noalloc SS._find_substr(ph, length(h), pp, length(p))
        @assert_vectorized SS._find_substr(ph, length(h), pp, length(p))
    end
    @test (@allocated find_substr(h, p)) == 0
end
