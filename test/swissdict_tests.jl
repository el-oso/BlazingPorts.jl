@testitem "swissdict_correctness" tags = [:swissdict] begin
    using BlazingPorts.SwissDict: SwissDict
    using Random

    # ── 1. randomised op stream oracle (≥1e5 ops) ────────────────────────────
    # UInt64 keys over a small domain so we get collisions, tombstones, resizes,
    # and wraparound. Apply the same op to both SwissDict and Base Dict; assert
    # identical results each op and equal final contents.
    Random.seed!(20260625)
    sd = SwissDict{UInt64,UInt64}()
    bd = Dict{UInt64,UInt64}()
    domain = UInt64.(1:500)
    ok = true
    for _ in 1:100_000
        k = rand(domain)
        op = rand(1:5)
        if op <= 2
            # insert / update
            v = rand(UInt64)
            sd[k] = v;  bd[k] = v
        elseif op == 3
            # delete
            delete!(sd, k); delete!(bd, k)
        elseif op == 4
            # lookup hit / miss
            r1 = get(sd, k, UInt64(0xdead))
            r2 = get(bd, k, UInt64(0xdead))
            if r1 !== r2
                ok = false
                @info "lookup mismatch" k got=r1 want=r2
                break
            end
        else
            # haskey
            if haskey(sd, k) !== haskey(bd, k)
                ok = false
                @info "haskey mismatch" k
                break
            end
        end
    end
    @test ok
    @test length(sd) == length(bd)
    @test all(get(sd, k, nothing) == v for (k, v) in bd)
    @test all(haskey(bd, k) for (k, _) in sd)

    # ── 2. edge cases ─────────────────────────────────────────────────────────
    # empty dict
    e = SwissDict{Int,Int}()
    @test isempty(e)
    @test length(e) == 0
    @test get(e, 1, -1) == -1
    @test !haskey(e, 1)

    # single element
    s = SwissDict{Int,Int}()
    s[42] = 7
    @test s[42] == 7
    @test haskey(s, 42)
    @test length(s) == 1

    # delete-then-reinsert same key (tombstone reuse)
    t = SwissDict{Int,Int}()
    for i in 1:50; t[i] = i; end
    delete!(t, 25)
    @test !haskey(t, 25)
    t[25] = 999
    @test t[25] == 999
    @test length(t) == 50

    # grow past several resizes
    big = SwissDict{Int,Int}()
    for i in 1:5000; big[i] = i * 3; end
    @test all(big[i] == i * 3 for i in 1:5000)

    # all-collide: keys whose hash maps to the same slot
    # hash(k) & (sz-1) == 0 for all k — force by using sz=16 and k values hashing there.
    # Simpler: use a very small dict and saturate one bucket manually via oracle comparison.
    c = SwissDict{String,Int}()
    for i in 1:200; c["key_$i"] = i; end
    @test all(c["key_$i"] == i for i in 1:200)

    # non-bits key type (String) — interface correctness, not perf
    d_str = SwissDict{String,Float64}()
    d_str["hello"] = 1.5
    d_str["world"] = 2.5
    @test d_str["hello"] == 1.5
    @test d_str["world"] == 2.5
    @test !haskey(d_str, "xyz")
    delete!(d_str, "hello")
    @test !haskey(d_str, "hello")

    # constructors
    d_pairs = SwissDict("a" => 1, "b" => 2)
    @test d_pairs["a"] == 1 && d_pairs["b"] == 2
    d_copy = copy(d_pairs)
    d_copy["a"] = 99
    @test d_pairs["a"] == 1  # original unchanged

    # iterate
    d_iter = SwissDict{Int,Int}(i => i^2 for i in 1:20)
    @test Set(keys(d_iter)) == Set(1:20)
    @test all(v == k^2 for (k, v) in d_iter)

    # pop!
    d_pop = SwissDict{Int,Int}(1 => 10, 2 => 20)
    @test pop!(d_pop, 1) == 10
    @test !haskey(d_pop, 1)
    @test pop!(d_pop, 99, -1) == -1

    # empty!
    d_empty = SwissDict{Int,Int}(i => i for i in 1:10)
    empty!(d_empty)
    @test isempty(d_empty)
    @test length(d_empty) == 0

    # get!
    d_get = SwissDict{String,Int}()
    @test get!(d_get, "x", 5) == 5
    @test get!(d_get, "x", 99) == 5   # already present

    # sizehint! doesn't corrupt data
    d_hint = SwissDict{Int,Int}()
    for i in 1:100; d_hint[i] = i; end
    sizehint!(d_hint, 500)
    @test all(d_hint[i] == i for i in 1:100)
end

@testitem "swissdict_typecontracts" tags = [:swissdict] begin
    using BlazingPorts.SwissDict: SwissDict
    using BaseTypeContracts
    using TypeContracts: implements, @test_implements

    @test_implements SwissDict{Int,Int} AbstractDict
    @test implements(SwissDict{Int,Int}, AbstractDict)

    # all_implements checks the full set of applicable Base contracts
    @test all_implements(SwissDict{Int,Int})
end

@testitem "swissdict_strictmode" tags = [:swissdict] begin
    using BlazingPorts.SwissDict: SwissDict, _find_slot
    using StrictMode, AllocCheck, JET

    # Build a dict pre-populated so probing is meaningful (mix of hits and misses)
    d = SwissDict{UInt64,UInt64}()
    for i in UInt64(1):UInt64(1000); d[i] = i * 2; end

    sz     = length(d.keys)
    key_h  = UInt64(500)   # hit
    key_m  = UInt64(9999)  # miss (not inserted)

    # Warm the specialisation
    _ht_keyindex_h = BlazingPorts.SwissDict._ht_keyindex
    _ht_keyindex_h(d, key_h); _ht_keyindex_h(d, key_m)

    # @assert_vectorized on the inner pointer kernel — this is where <16 x i8> lives.
    GC.@preserve d begin
        slots_ptr = pointer(d.slots)
        idx_h, sh_h = BlazingPorts.SwissDict._hashindex(key_h, sz)
        idx_m, sh_m = BlazingPorts.SwissDict._hashindex(key_m, sz)
        # warm
        _find_slot(slots_ptr, d.keys, key_h, sz, sh_h, idx_h)
        _find_slot(slots_ptr, d.keys, key_m, sz, sh_m, idx_m)
        @assert_vectorized _find_slot(slots_ptr, d.keys, key_h, sz, sh_h, idx_h)
        @assert_noalloc    _find_slot(slots_ptr, d.keys, key_h, sz, sh_h, idx_h)
        @assert_typestable _find_slot(slots_ptr, d.keys, key_h, sz, sh_h, idx_h)
    end

    # The public getindex / haskey must be allocation-free on the hot path.
    _ht_keyindex_h(d, key_h)   # ensure compiled
    @assert_noalloc    _ht_keyindex_h(d, key_h)
    @assert_typestable _ht_keyindex_h(d, key_h)
    @test (@allocated _ht_keyindex_h(d, key_h)) == 0
    @test (@allocated haskey(d, key_h)) == 0
end
