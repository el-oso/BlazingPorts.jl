"""
    IntFormat

Branchless integer → decimal formatting, the Julia analogue of Rust's `itoa`. Base `string(::Int)`
heap-allocates per call. This writes into a **caller-held buffer** (zero per-call alloc) and combines two
levers: **divide-and-conquer** (split the u64 into independent 8-digit chunks via `÷10^8`/`÷10^16`, then
4- and 2-digit halves — the divisions form a balanced tree, not a serial chain) and **packed 16-bit
stores** from a 100-entry `UInt16` LUT (one store per digit-pair).

LITTLE-ENDIAN: the LUT packs the tens digit in the low byte so a 16-bit store lands tens-then-ones in
ascending memory. Correct on x86-64 / AArch64-LE (the only targets here).

No StrictMode dependency (package rule) — `@assert_noalloc`/`@assert_typestable` are applied in
`test/`+`bench/`.
"""
module IntFormat

export format_int!, format_int

# Packed 2-digit LUT: D2[i+1] = the two ASCII digits of i (0≤i≤99), tens in the low byte.
const D2 = let b = Vector{UInt16}(undef, 100)
    for i in 0:99
        tens = UInt16(0x30 + (i ÷ 10) % UInt8)
        ones = UInt16(0x30 + (i % 10) % UInt8)
        b[i + 1] = tens | (ones << 8)
    end
    b
end

@inline _w2!(p::Ptr{UInt8}, off::Int, v::UInt64) =          # 2 digits (v<100) at 0-based byte offset off
    unsafe_store!(Ptr{UInt16}(p + off), @inbounds D2[v + 1])
@inline function _w4!(p, off, v::UInt64)                    # 4 digits (v<10^4)
    hi = v ÷ 0x64; lo = v - hi * 0x64
    _w2!(p, off, hi); _w2!(p, off + 2, lo)
end
# Division-free 8-digit write (v < 10^8). Fixed-point: t = v·⌈2^57/10^6⌉; each pair is t>>57, then
# multiply the fraction by 100 and repeat. Provably exact (error < 1 after 3 ×100 steps); no divisions.
const _C57 = 144115188076               # ⌈2^57 / 10^6⌉
const _M57 = (UInt64(1) << 57) - 0x1
@inline function _w8!(p, off, v::UInt64)
    t = v * _C57
    _w2!(p, off, t >> 57);     t = (t & _M57) * 0x64
    _w2!(p, off + 2, t >> 57); t = (t & _M57) * 0x64
    _w2!(p, off + 4, t >> 57); t = (t & _M57) * 0x64
    _w2!(p, off + 6, t >> 57)
end

@inline _w1!(p, off, v::UInt64) = unsafe_store!(p + off, 0x30 + v % UInt8)   # one digit (v<10)

# Leading (variable-width, no leading zeros) group v < 10^8 at byte offset `off`. Balanced magnitude
# branch (≤3 comparisons, direct writes, no digit-count scan or loop) — itoa-class small-number path.
@inline function _wlead!(p, off, v::UInt64)
    if v < 0x2710                       # < 10^4 (1–4 digits)
        if v < 0x64                     # 1–2
            v < 0xa && (_w1!(p, off, v); return 1)
            _w2!(p, off, v); return 2
        else                            # 3–4
            hi = v ÷ 0x64; lo = v - hi * 0x64
            hi < 0xa && (_w1!(p, off, hi); _w2!(p, off + 1, lo); return 3)
            _w2!(p, off, hi); _w2!(p, off + 2, lo); return 4
        end
    else                                # 10^4 … 10^8−1 (5–8 digits)
        hi = v ÷ 0x2710; lo = v - hi * 0x2710      # lo: exactly 4 digits
        if hi < 0x64                    # 5–6
            hi < 0xa && (_w1!(p, off, hi); _w4!(p, off + 1, lo); return 5)
            _w2!(p, off, hi); _w4!(p, off + 2, lo); return 6
        else                            # 7–8
            h = hi ÷ 0x64; l = hi - h * 0x64
            h < 0xa && (_w1!(p, off, h); _w2!(p, off + 1, l); _w4!(p, off + 3, lo); return 7)
            _w2!(p, off, h); _w2!(p, off + 2, l); _w4!(p, off + 4, lo); return 8
        end
    end
end

const _E8  = UInt64(100_000_000)
const _E16 = UInt64(10)^16

@inline function _format_u64!(p::Ptr{UInt8}, off::Int, u::UInt64)
    if u < _E8
        return _wlead!(p, off, u)
    elseif u < _E16
        hi = u ÷ _E8; lo = u - hi * _E8
        d = _wlead!(p, off, hi); _w8!(p, off + d, lo)
        return d + 8
    else
        top = u ÷ _E16; rest = u - top * _E16
        mid = rest ÷ _E8; lo = rest - mid * _E8
        d = _wlead!(p, off, top); _w8!(p, off + d, mid); _w8!(p, off + d + 8, lo)
        return d + 16
    end
end

@inline function _format_signed!(buf::Vector{UInt8}, x::Int64)
    GC.@preserve buf begin
        p = pointer(buf)
        # Branchless sign: a data-dependent `if x<0` mispredicts at ~50/50 signs (≈4× penalty). Instead
        # compute |x| with cmov, always write '-' at p[0], and start digits at offset `nb` (0 ⇒ the
        # first digit overwrites the '-' for positives; 1 ⇒ keep it for negatives). typemin-safe.
        neg = x < 0
        u = ifelse(neg, ~reinterpret(UInt64, x) + 0x1, reinterpret(UInt64, x))
        unsafe_store!(p, 0x2d)
        nb = ifelse(neg, 1, 0)
        return nb + _format_u64!(p, nb, u)
    end
end

"""
    format_int!(buf::Vector{UInt8}, x::Integer) -> Int

Write the decimal representation of `x` into `buf` (caller guarantees `length(buf) ≥ 24`), returning the
number of bytes written. Allocation-free; the buffer is reused across calls.
"""
@inline format_int!(buf::Vector{UInt8}, x::Union{Int8,Int16,Int32,Int64}) = _format_signed!(buf, Int64(x))
@inline function format_int!(buf::Vector{UInt8}, x::Union{UInt8,UInt16,UInt32,UInt64})
    GC.@preserve buf return _format_u64!(pointer(buf), 0, UInt64(x))
end

"""
    format_int(x::Integer) -> String

Convenience: decimal string of `x` (allocates one `String`). For hot loops use [`format_int!`](@ref).
"""
function format_int(x::Integer)
    buf = Vector{UInt8}(undef, 24)
    n = format_int!(buf, x)
    return String(buf[1:n])
end

end # module IntFormat
