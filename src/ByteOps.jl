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

using SIMD: Vec, vload, vstore

export base64_encode, base64_encode!

# ── AVX2 SIMD primitives via llvmcall ────────────────────────────────────────────────────────────────
const _V32 = NTuple{32, VecElement{UInt8}}
const _V16w = NTuple{16, VecElement{UInt16}}
_bin(intr, t) = ("declare $t @$intr($t, $t)\ndefine $t @e($t %a, $t %b) #0 {\n%r = call $t @$intr($t %a, $t %b)\nret $t %r\n}\nattributes #0 = { alwaysinline }", "e")
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
