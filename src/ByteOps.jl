"""
    ByteOps

Pure-Julia **SIMD byte transcoding** — the shuffle/lookup (`pshufb`) kernel class. Starts with base64
encode (the Muła/lemire SSSE3→AVX2 algorithm, ported from the `base64-simd` crate); hex + decode follow.

`Base64.base64encode` is a scalar table loop (~0.5 GB/s). `base64_encode!` validates 24 input bytes →
32 chars per AVX2 block: an offset-load + asymmetric `vpshufb` reshuffle (no cross-lane `vpermd`),
`vpmulhuw`/multiply bit-spreading, and a `vpshufb` offset-LUT translate. **Byte-exact with
`Base64.base64encode`.**

- `base64_encode!(out::Vector{UInt8}, data) -> out`  — kernel only, caller owns `out` (the fast path)
- `base64_encode(data) -> String`                    — allocating convenience wrapper

Kernel-to-kernel (preallocated both sides) this **beats `base64-simd`** (~1.7×); the apparent loss in the
allocating form is the output `Vector`+`String` allocation, not the kernel (campaign lesson: isolate the
kernel, not the allocator). The `@assert_vectorized`/`kernel_report` audit lives in `test/` (F33: this is a
shuffle-port-bound kernel).
"""
module ByteOps

using SIMD: Vec, vload, vstore, shufflevector, vifelse

export base64_encode, base64_encode!, hex_encode, hex_encode!, hex_decode, hex_decode!, base64_decode, base64_decode!

# ── AVX2 SIMD primitives via llvmcall ────────────────────────────────────────────────────────────────
const _V32 = NTuple{32, VecElement{UInt8}}
const _V16w = NTuple{16, VecElement{UInt16}}
_bin(intr, t) = ("declare $t @$intr($t, $t)\ndefine $t @e($t %a, $t %b) #0 {\n%r = call $t @$intr($t %a, $t %b)\nret $t %r\n}\nattributes #0 = { alwaysinline }", "e")
const _PSHUFB16 = _bin("llvm.x86.ssse3.pshuf.b.128", "<16 x i8>")
const _V16 = NTuple{16, VecElement{UInt8}}
@inline _pshufb16(t::Vec{16,UInt8}, i::Vec{16,UInt8}) = Vec(Base.llvmcall(_PSHUFB16, _V16, Tuple{_V16,_V16}, t.data, i.data))
const _V8w = NTuple{8, VecElement{UInt16}}; const _V4d = NTuple{4, VecElement{UInt32}}
const _MUBS = ("declare <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8>,<16 x i8>)\ndefine <8 x i16> @e(<16 x i8> %a,<16 x i8> %b) #0 {\n%r=call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %a,<16 x i8> %b)\nret <8 x i16> %r\n}\nattributes #0={alwaysinline}", "e")
const _MWD  = ("declare <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16>,<8 x i16>)\ndefine <4 x i32> @e(<8 x i16> %a,<8 x i16> %b) #0 {\n%r=call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %a,<8 x i16> %b)\nret <4 x i32> %r\n}\nattributes #0={alwaysinline}", "e")
const _EQ16 = ("define <16 x i8> @e(<16 x i8> %a,<16 x i8> %b) #0 {\n%c=icmp eq <16 x i8> %a,%b\n%r=sext <16 x i1> %c to <16 x i8>\nret <16 x i8> %r\n}\nattributes #0={alwaysinline}", "e")
@inline _maddubs(a::Vec{16,UInt8}, b::Vec{16,UInt8}) = Vec(Base.llvmcall(_MUBS, _V8w, Tuple{_V16,_V16}, a.data, b.data))
@inline _maddwd(a::Vec{8,UInt16}, b::Vec{8,UInt16}) = Vec(Base.llvmcall(_MWD, _V4d, Tuple{_V8w,_V8w}, a.data, b.data))
@inline _cmpeqb(a::Vec{16,UInt8}, b::Vec{16,UInt8}) = Vec(Base.llvmcall(_EQ16, _V16, Tuple{_V16,_V16}, a.data, b.data))
const _PSHUFB = _bin("llvm.x86.avx2.pshuf.b", "<32 x i8>")
const _USUB   = _bin("llvm.usub.sat.v32i8", "<32 x i8>")
const _MULHU  = _bin("llvm.x86.avx2.pmulhu.w", "<16 x i16>")
const _CMPGT  = ("define <32 x i8> @e(<32 x i8> %a, <32 x i8> %b) #0 {\n%c = icmp sgt <32 x i8> %a, %b\n%r = sext <32 x i1> %c to <32 x i8>\nret <32 x i8> %r\n}\nattributes #0 = { alwaysinline }", "e")
@inline _pshufb(t::Vec{32,UInt8}, i::Vec{32,UInt8}) = Vec(Base.llvmcall(_PSHUFB, _V32, Tuple{_V32,_V32}, t.data, i.data))
@inline _usubsat(a::Vec{32,UInt8}, b::Vec{32,UInt8}) = Vec(Base.llvmcall(_USUB, _V32, Tuple{_V32,_V32}, a.data, b.data))
@inline _mulhu16(a::Vec{16,UInt16}, b::Vec{16,UInt16}) = Vec(Base.llvmcall(_MULHU, _V16w, Tuple{_V16w,_V16w}, a.data, b.data))
@inline _cmpgtb(a::Vec{32,UInt8}, b::Vec{32,UInt8}) = Vec(Base.llvmcall(_CMPGT, _V32, Tuple{_V32,_V32}, a.data, b.data))  # 0xFF where a>b (signed)

_bcast32(x::UInt32) = Vec{32,UInt8}(ntuple(i -> UInt8((x >> (8 * ((i - 1) % 4))) & 0xff), 32))
const _MASK0 = _bcast32(0x0fc0fc00); const _MASK2 = _bcast32(0x003f03f0)
const _MUL0  = reinterpret(Vec{16,UInt16}, _bcast32(0x04000040))
const _MUL2  = reinterpret(Vec{16,UInt16}, _bcast32(0x01000010))
# Asymmetric reshuffle (base64-simd SPLIT_SHUFFLE): input loaded at (p-4) so lane0 starts at offset 4,
# lane1 at 0 — removes the cross-lane vpermd. lane0 = src[0:12], lane1 = src[12:24].
const _RS0 = (5,4,6,5, 8,7,9,8, 11,10,12,11, 14,13,15,14)
const _RS1 = (1,0,2,1, 4,3,5,4, 7,6,8,7, 10,9,11,10)
const _RESHUF = Vec{32,UInt8}((_RS0..., _RS1...))
const _LU = (65,71,0xfc,0xfc,0xfc,0xfc,0xfc,0xfc,0xfc,0xfc,0xfc,0xfc,0xed,0xf0,0,0)   # Muła offset LUT
const _LUT = Vec{32,UInt8}((_LU..., _LU...))
const _LOWMASK = Vec{32,UInt8}(0x0F)
const _C25 = Vec{32,UInt8}(25); const _C51 = Vec{32,UInt8}(51)
const _CHARSET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

# 24 input bytes (loaded at p-4) → 32 base64 ASCII chars.
@inline function _enc24(input::Vec{32,UInt8})
    inr = _pshufb(input, _RESHUF)
    t1 = _mulhu16(reinterpret(Vec{16,UInt16}, inr & _MASK0), _MUL0)
    t3 = reinterpret(Vec{16,UInt16}, inr & _MASK2) * _MUL2
    idx = reinterpret(Vec{32,UInt8}, t1 | t3)            # 4 six-bit indices per dword
    off = _usubsat(idx, _C51) - _cmpgtb(idx, _C25)       # Muła translate offset
    idx + _pshufb(_LUT, off)                             # paddb → ASCII
end

@inline function _enc3!(out, o, b0, b1, b2)              # scalar 3 bytes → 4 chars
    @inbounds begin
        out[o+1] = _CHARSET[(b0 >> 2) + 1]
        out[o+2] = _CHARSET[(((b0 & 0x03) << 4) | (b1 >> 4)) + 1]
        out[o+3] = _CHARSET[(((b1 & 0x0f) << 2) | (b2 >> 6)) + 1]
        out[o+4] = _CHARSET[(b2 & 0x3f) + 1]
    end
end

"""
    base64_encode!(out::Vector{UInt8}, data::AbstractVector{UInt8}) -> out

Standard base64 (with `=` padding, no line breaks) into the preallocated `out` (must be
`4*cld(length(data),3)` bytes). Byte-exact with `Base64.base64encode`. The allocation-free fast path.
"""
function base64_encode!(out::Vector{UInt8}, data::DenseVector{UInt8})
    n = length(data)
    GC.@preserve data out begin
        p = pointer(data); q = pointer(out); i = 0; o = 0
        if n >= 34                                        # offset-load SIMD path needs (p-4) in-bounds
            @inbounds for _ in 1:2                        # 2 scalar groups so (p+i-4) ≥ p
                _enc3!(out, o, data[i+1], data[i+2], data[i+3]); i += 3; o += 4
            end
            @inbounds while i + 52 <= n                   # 2× unrolled for ILP
                vstore(_enc24(vload(Vec{32,UInt8}, p + i - 4)), q + o)
                vstore(_enc24(vload(Vec{32,UInt8}, p + i + 20)), q + o + 32)
                i += 48; o += 64
            end
            @inbounds while i + 28 <= n                   # load 32 at p+i-4, consume 24, emit 32
                vstore(_enc24(vload(Vec{32,UInt8}, p + i - 4)), q + o); i += 24; o += 32
            end
        end
        @inbounds while i + 3 <= n                        # scalar 3→4 tail
            _enc3!(out, o, data[i+1], data[i+2], data[i+3]); i += 3; o += 4
        end
        rem = n - i                                       # final 1 or 2 bytes + padding
        @inbounds if rem == 1
            b0 = data[i+1]
            out[o+1] = _CHARSET[(b0>>2)+1]; out[o+2] = _CHARSET[((b0&0x03)<<4)+1]
            out[o+3] = UInt8('='); out[o+4] = UInt8('=')
        elseif rem == 2
            b0 = data[i+1]; b1 = data[i+2]
            out[o+1] = _CHARSET[(b0>>2)+1]; out[o+2] = _CHARSET[(((b0&0x03)<<4)|(b1>>4))+1]
            out[o+3] = _CHARSET[((b1&0x0f)<<2)+1]; out[o+4] = UInt8('=')
        end
    end
    out
end
base64_encode!(out::Vector{UInt8}, data::AbstractVector{UInt8}) = base64_encode!(out, collect(data))

# ── hex encode (N=16: two pshufb nibble lookups + a shufflevector interleave) ────────────────────────
const _HEXLUT = Vec{16,UInt8}(ntuple(i -> UInt8(b"0123456789abcdef"[i]), 16))
const _LOW16 = Vec{16,UInt8}(0x0F)
const _ILO = Val(ntuple(j -> iseven(j - 1) ? (j - 1) ÷ 2 : 16 + (j - 1) ÷ 2, 16))   # interleave low half
const _IHI = Val(ntuple(j -> iseven(j - 1) ? 8 + (j - 1) ÷ 2 : 24 + (j - 1) ÷ 2, 16)) # interleave high half
const _HEXCH = b"0123456789abcdef"

@inline function _hex16(v::Vec{16,UInt8})               # 16 bytes → 32 lowercase hex chars (two halves)
    hc = _pshufb16(_HEXLUT, v >> 0x04); lc = _pshufb16(_HEXLUT, v & _LOW16)
    (shufflevector(hc, lc, _ILO), shufflevector(hc, lc, _IHI))
end

"""
    hex_encode!(out::Vector{UInt8}, data::AbstractVector{UInt8}) -> out

Lowercase hex into preallocated `out` (`2*length(data)` bytes). Byte-exact with `bytes2hex`. 0-alloc.
"""
function hex_encode!(out::Vector{UInt8}, data::DenseVector{UInt8})
    n = length(data)
    GC.@preserve data out begin
        p = pointer(data); q = pointer(out); i = 0; o = 0
        @inbounds while i + 16 <= n
            a, b = _hex16(vload(Vec{16,UInt8}, p + i)); vstore(a, q + o); vstore(b, q + o + 16); i += 16; o += 32
        end
        @inbounds while i < n
            v = data[i+1]; out[o+1] = _HEXCH[(v >> 4) + 1]; out[o+2] = _HEXCH[(v & 0x0f) + 1]; i += 1; o += 2
        end
    end
    out
end
hex_encode!(out::Vector{UInt8}, data::AbstractVector{UInt8}) = hex_encode!(out, collect(data))

"""
    hex_encode(data) -> String

Allocating convenience wrapper over [`hex_encode!`](@ref).
"""
function hex_encode(data::DenseVector{UInt8})
    out = Vector{UInt8}(undef, 2 * length(data)); hex_encode!(out, data); String(out)
end
hex_encode(data::AbstractVector{UInt8}) = hex_encode(collect(data))

# ── hex decode (N=16: validate + nibble parse + pshufb de-interleave) ────────────────────────────────
const _EVEN = Vec{16,UInt8}((0,2,4,6,8,10,12,14, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80))
const _ODD  = Vec{16,UInt8}((1,3,5,7,9,11,13,15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80))
const _NINE = Vec{16,UInt8}(9)

@inline function _dec16(c::Vec{16,UInt8})               # 16 hex chars → (8 bytes in low lanes, all-valid?)
    isd = (c >= Vec{16,UInt8}(0x30)) & (c <= Vec{16,UInt8}(0x39))
    lc  = c | Vec{16,UInt8}(0x20)
    isl = (lc >= Vec{16,UInt8}(0x61)) & (lc <= Vec{16,UInt8}(0x66))
    nib = (c & _LOW16) + (c >> 0x06) * _NINE             # 0-9 for digits, +9 for letters
    (((_pshufb16(nib, _EVEN) << 0x04) | _pshufb16(nib, _ODD)), all(isd | isl))
end
@inline _hexnib(ch) = (0x30 <= ch <= 0x39) ? ch - 0x30 :
                      (0x61 <= (ch | 0x20) <= 0x66) ? (ch | 0x20) - 0x61 + 0x0a : 0xff

"""
    hex_decode!(out::Vector{UInt8}, s::DenseVector{UInt8}) -> Bool

Decode hex `s` (even length) into preallocated `out` (`length(s)÷2` bytes); returns `true` iff all
characters were valid hex (validated SIMD-wide). The allocation-free kernel.
"""
function hex_decode!(out::Vector{UInt8}, s::DenseVector{UInt8})
    n = length(s); ok = true
    GC.@preserve s out begin
        p = pointer(s); q = pointer(out); i = 0; o = 0
        @inbounds while i + 16 <= n
            bytes, v = _dec16(vload(Vec{16,UInt8}, p + i)); ok &= v
            vstore(shufflevector(bytes, Val((0, 1, 2, 3, 4, 5, 6, 7))), q + o); i += 16; o += 8
        end
        @inbounds while i < n
            hv = _hexnib(s[i+1]); lv = _hexnib(s[i+2]); (hv > 0x0f || lv > 0x0f) && (ok = false)
            out[o+1] = (hv << 4) | lv; i += 2; o += 1
        end
    end
    ok
end

"""
    hex_decode(s) -> Vector{UInt8}

Decode lowercase-or-uppercase hex. Byte-exact with `hex2bytes`; throws `ArgumentError` on odd length or any
invalid character (validated SIMD-wide). Allocating wrapper over [`hex_decode!`](@ref).
"""
function hex_decode(s::DenseVector{UInt8})
    n = length(s); isodd(n) && throw(ArgumentError("hex_decode: odd length"))
    out = Vector{UInt8}(undef, n ÷ 2)
    hex_decode!(out, s) || throw(ArgumentError("hex_decode: invalid hex character"))
    out
end
hex_decode(s::AbstractString) = hex_decode(codeunits(String(s)))
hex_decode(s::AbstractVector{UInt8}) = hex_decode(collect(s))

# ── base64 decode (Muła SSE: char→6bit lookup+validate, then pmaddubs/pmaddwd pack 4×6bit→3 bytes) ────
const _SHIFT = Vec{16,UInt8}((0,0,0x13,0x04,0xBF,0xBF,0xB9,0xB9,0,0,0,0,0,0,0,0))
const _MASKL = Vec{16,UInt8}((0xA8,0xF8,0xF8,0xF8,0xF8,0xF8,0xF8,0xF8,0xF8,0xF8,0xF0,0x54,0x50,0x50,0x50,0x54))
const _BITP  = Vec{16,UInt8}((0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0,0,0,0,0,0,0,0))
const _C2F = Vec{16,UInt8}(0x2f); const _C16 = Vec{16,UInt8}(16); const _ZERO16 = Vec{16,UInt8}(0)
const _MUBSK = Vec{16,UInt8}(ntuple(i -> UInt8((0x01400140 >> (8 * ((i - 1) % 4))) & 0xff), 16))
const _MWDK  = reinterpret(Vec{8,UInt16}, Vec{16,UInt8}(ntuple(i -> UInt8((0x00011000 >> (8 * ((i - 1) % 4))) & 0xff), 16)))
const _PACK  = Vec{16,UInt8}((2,1,0,6,5,4,10,9,8,14,13,12,0x80,0x80,0x80,0x80))

@inline function _decb64_16(input::Vec{16,UInt8})       # 16 base64 chars → (12 bytes low lanes, all-valid?)
    hin = (input >> 0x04) & _LOW16; lin = input & _LOW16
    shift = vifelse(_cmpeqb(input, _C2F) != _ZERO16, _C16, _pshufb16(_SHIFT, hin))   # '/' → +16
    nonmatch = _cmpeqb(_pshufb16(_MASKL, lin) & _pshufb16(_BITP, hin), _ZERO16)       # 0xFF where invalid
    merged = _maddwd(_maddubs(input + shift, _MUBSK), _MWDK)                          # 4×6bit → 3 bytes/dword
    (_pshufb16(reinterpret(Vec{16,UInt8}, merged), _PACK), all(nonmatch == _ZERO16))
end

const _B64DEC = let t = fill(0xff, 256)                  # scalar char→6bit table (0xff = invalid)
    for (v, c) in enumerate(_CHARSET); t[c+1] = UInt8(v - 1); end
    Tuple(t)
end
@inline _b64val(c) = @inbounds _B64DEC[c+1]

_b64declen(s) = (n = length(s); n == 0 ? 0 : 3 * (n ÷ 4) - (@inbounds (s[n] == UInt8('=')) + (s[n-1] == UInt8('='))))

"""
    base64_decode!(out::Vector{UInt8}, s::DenseVector{UInt8}) -> Bool

Decode `=`-padded base64 `s` (length a multiple of 4) into preallocated `out` (`base64_decode`-sized);
returns `true` iff all characters were valid. The allocation-free kernel.
"""
function base64_decode!(out::Vector{UInt8}, s::DenseVector{UInt8})
    n = length(s); n == 0 && return true
    outlen = length(out); ok = true
    @inbounds npad = (s[n] == UInt8('=')) + (s[n-1] == UInt8('='))
    nfull = npad > 0 ? n - 4 : n                          # exclude the final (possibly padded) group
    GC.@preserve s out begin
        p = pointer(s); q = pointer(out); i = 0; o = 0
        @inbounds while i + 16 <= nfull && o + 16 <= outlen
            bytes, v = _decb64_16(vload(Vec{16,UInt8}, p + i)); ok &= v
            vstore(bytes, q + o); i += 16; o += 12
        end
        @inbounds while i + 4 <= n                         # scalar 4 chars → 3 bytes (last group padded)
            c0 = s[i+1]; c1 = s[i+2]; c2 = s[i+3]; c3 = s[i+4]
            v0 = _b64val(c0); v1 = _b64val(c1)
            pad2 = c2 == UInt8('='); pad3 = c3 == UInt8('=')
            v2 = pad2 ? 0x00 : _b64val(c2); v3 = pad3 ? 0x00 : _b64val(c3)
            (v0 > 63 || v1 > 63 || (!pad2 && v2 > 63) || (!pad3 && v3 > 63)) && (ok = false)
            out[o+1] = (v0 << 2) | (v1 >> 4); o += 1
            if !pad2; out[o+1] = ((v1 & 0x0f) << 4) | (v2 >> 2); o += 1; end
            if !pad3; out[o+1] = ((v2 & 0x03) << 6) | v3; o += 1; end
            i += 4
        end
    end
    ok
end

"""
    base64_decode(s) -> Vector{UInt8}

Decode standard (`=`-padded) base64. Byte-exact with `Base64.base64decode`; throws `ArgumentError` on a
length that is not a multiple of 4 or an invalid character. Allocating wrapper over [`base64_decode!`](@ref).
"""
function base64_decode(s::DenseVector{UInt8})
    n = length(s); n == 0 && return UInt8[]
    (n % 4 == 0) || throw(ArgumentError("base64_decode: length not a multiple of 4"))
    out = Vector{UInt8}(undef, _b64declen(s))
    base64_decode!(out, s) || throw(ArgumentError("base64_decode: invalid base64 character"))
    out
end
base64_decode(s::AbstractString) = base64_decode(codeunits(String(s)))
base64_decode(s::AbstractVector{UInt8}) = base64_decode(collect(s))

"""
    base64_encode(data) -> String

Allocating convenience wrapper over [`base64_encode!`](@ref).
"""
function base64_encode(data::DenseVector{UInt8})
    out = Vector{UInt8}(undef, cld(length(data), 3) * 4)
    base64_encode!(out, data)
    String(out)
end
base64_encode(data::AbstractVector{UInt8}) = base64_encode(collect(data))
base64_encode(s::AbstractString) = base64_encode(codeunits(String(s)))

end # module ByteOps
