"""
    StringSearch

SIMD substring search — the Julia analogue of Rust's `memchr::memmem`. Base's multi-byte
`findfirst(needle, haystack)` is a scalar first-byte scan + verify (no SIMD prefilter), which the
2026-06-24 probe measured at 0.12× the `memmem` crate. This adds exactly the missing pass: a
**first+last-byte SIMD prefilter** (memmem's own trick) that slides a wide `Vec{W,UInt8}` compare
over the haystack and only falls to a scalar `memcmp` on candidate hits.

The single-byte case is already at parity in Base (memchr-backed `findfirst`), so `m ≤ 1` is
delegated to Base; only needles of length `m ≥ 2` take the SIMD path. The bounded `< W+m` scalar
tail is a deliberate, documented escape (SIMD.jl has no count-based partial load).

No StrictMode dependency here (package rule) — hot path is Base + SIMD.jl; the `@assert_vectorized`
/ `@assert_noalloc` guarantees are applied in `test/` and `bench/`.
"""
module StringSearch

using SIMD: Vec, vload, bitmask

export find_substr

const _W = 64   # UInt8 SIMD width (Vec{64,UInt8} → <64 x i8>, full AVX-512); see bench/probe_substr.jl

# Verify the full needle at 0-based start `pos`, given the first and last bytes already matched.
@inline function _verify(ph::Ptr{UInt8}, pp::Ptr{UInt8}, pos::Int, m::Int)
    @inbounds for t in 2:(m - 1)        # p[t] (1-based) aligns with h at unsafe_load index pos+t
        unsafe_load(pp, t) == unsafe_load(ph, pos + t) || return false
    end
    return true
end

# Core: 0-based scan over a raw byte range. Assumes `2 ≤ m ≤ n`. Returns the 1-based start of the
# first match, or 0 for no match. Caller holds the buffers alive (`GC.@preserve`).
function _find_substr(ph::Ptr{UInt8}, n::Int, pp::Ptr{UInt8}, m::Int)
    f = unsafe_load(pp, 1)
    l = unsafe_load(pp, m)
    vf = Vec{_W,UInt8}(f)
    vl = Vec{_W,UInt8}(l)
    simd_hi = n - m - _W + 1            # 0-based inclusive max `i` with both W-chunks fully in bounds
    last0 = n - m                       # 0-based inclusive max start
    i = 0
    @inbounds while i <= simd_hi
        c1 = vload(Vec{_W,UInt8}, ph + i)             # h[i .. i+W-1]
        c2 = vload(Vec{_W,UInt8}, ph + (i + m - 1))   # h[i+m-1 .. i+m-1+W-1]
        bm = bitmask((c1 == vf) & (c2 == vl))         # lanes where first AND last byte align
        while bm != zero(bm)
            pos = i + trailing_zeros(bm)              # 0-based candidate start
            _verify(ph, pp, pos, m) && return pos + 1
            bm &= bm - one(bm)                        # clear lowest set bit
        end
        i += _W
    end
    @inbounds while i <= last0                         # bounded scalar tail (< W+m positions)
        (unsafe_load(ph, i + 1) == f && unsafe_load(ph, i + m) == l && _verify(ph, pp, i, m)) &&
            return i + 1
        i += 1
    end
    return 0
end

"""
    find_substr(haystack, needle) -> Union{Int,Nothing}

1-based index of the first occurrence of `needle` in `haystack`, or `nothing`. Accepts `String`s or
`AbstractVector{UInt8}`. Empty needle returns `1`; single-byte needle delegates to Base.
"""
function find_substr(h::AbstractVector{UInt8}, p::AbstractVector{UInt8})
    n = length(h); m = length(p)
    m == 0 && return 1
    m == 1 && return findfirst(==(@inbounds p[1]), h)
    m > n && return nothing
    r = GC.@preserve h p _find_substr(pointer(h), n, pointer(p), m)
    return r == 0 ? nothing : r
end

function find_substr(h::Union{String,SubString{String}}, p::Union{String,SubString{String}})
    n = ncodeunits(h); m = ncodeunits(p)
    m == 0 && return 1
    m == 1 && return findfirst(==(@inbounds codeunit(p, 1)), codeunits(h))
    m > n && return nothing
    r = GC.@preserve h p _find_substr(pointer(h), n, pointer(p), m)
    return r == 0 ? nothing : r
end

end # module StringSearch
