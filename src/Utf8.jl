"""
    Utf8

Pure-Julia **SIMD UTF-8 validation** (lemire / simdjson algorithm, ported from the `simdutf8` crate).

Base's `isvalid(::String)` SIMD-checks the all-ASCII fast path (~parity with SIMD) but falls to a scalar
codepath the moment multibyte content appears (~11× slower than `simdutf8` on accented/CJK/emoji text).
`isvalid_utf8` keeps the ASCII fast path **and** validates multibyte SIMD-wide: classify each byte against
its predecessors with three `pshufb` nibble-lookup tables + range checks (`Vec{16,UInt8}` → `<16 x i8>`,
SSSE3 `pshufb`), accumulating an error vector. Byte-exact with `Base.isvalid`.

- `isvalid_utf8(data::AbstractVector{UInt8}) -> Bool`
- `isvalid_utf8(s::AbstractString) -> Bool`

This is a **shuffle/lookup-dominated** kernel (≈0 arithmetic intensity) — the `@assert_vectorized`/
`kernel_report` audit lives in `test/` (source carries no StrictMode dep, per the BlazingPorts discipline).
N=16 (SSSE3); an AVX2 (N=32) widening would roughly double it toward `simdutf8`.
"""
module Utf8

using SIMD: Vec, vload, shufflevector, vifelse

export isvalid_utf8

# ── SIMD primitives via llvmcall (SSSE3 pshufb / usub.sat / pmovmskb) ────────────────────────────────
const _V16 = NTuple{16, VecElement{UInt8}}
_binir(intr) = ("declare <16 x i8> @$intr(<16 x i8>, <16 x i8>)\ndefine <16 x i8> @e(<16 x i8> %a, <16 x i8> %b) #0 {\n%r = call <16 x i8> @$intr(<16 x i8> %a, <16 x i8> %b)\nret <16 x i8> %r\n}\nattributes #0 = { alwaysinline }", "e")
const _PSHUFB  = _binir("llvm.x86.ssse3.pshuf.b.128")
const _USUBSAT = _binir("llvm.usub.sat.v16i8")
const _MOVEMASK = ("declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>)\ndefine i32 @e(<16 x i8> %a) #0 {\n%r = call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %a)\nret i32 %r\n}\nattributes #0 = { alwaysinline }", "e")
# `pshufb(table, idx)`: 16-byte table lookup, lane j = table[idx[j] & 0x0F] (idx high bit ⇒ 0).
@inline _pshufb(t::Vec{16,UInt8}, i::Vec{16,UInt8}) = Vec(Base.llvmcall(_PSHUFB, _V16, Tuple{_V16,_V16}, t.data, i.data))
@inline _usubsat(a::Vec{16,UInt8}, b::Vec{16,UInt8}) = Vec(Base.llvmcall(_USUBSAT, _V16, Tuple{_V16,_V16}, a.data, b.data))
@inline _movemask(v::Vec{16,UInt8}) = Base.llvmcall(_MOVEMASK, Int32, Tuple{_V16}, v.data)

# ── lookup tables (exact, from simdutf8 `algorithm.rs`) ──────────────────────────────────────────────
# Error-class bit flags packed into the three nibble tables; any surviving bit after the AND = error.
const _B1H = Vec{16,UInt8}((0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02, 0x80,0x80,0x80,0x80, 0x21,0x01,0x15,0x49))
const _B1L = Vec{16,UInt8}((0xE7,0xA3,0x83,0x83,0x8B,0xCB,0xCB,0xCB,0xCB,0xCB,0xCB,0xCB,0xCB,0xDB,0xCB,0xCB))
const _B2H = Vec{16,UInt8}((0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01, 0xE6,0xAE,0xBA,0xBA, 0x01,0x01,0x01,0x01))
const _INCOMPLETE = Vec{16,UInt8}((0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0xef,0xdf,0xbf))
const _LOWMASK = Vec{16,UInt8}(0x0F)
const _HI80    = Vec{16,UInt8}(0x80)
const _ZERO    = Vec{16,UInt8}(0x00)
const _P1 = ntuple(i->15+(i-1), 16)   # input.prev<1>(prev): shift right 1, carry prev's tail
const _P2 = ntuple(i->14+(i-1), 16)
const _P3 = ntuple(i->13+(i-1), 16)

# ── the validation kernel (per 16-byte block; carries `prev`/`incomplete`/`error`) ──────────────────
@inline function _check_special(input::Vec{16,UInt8}, prev1::Vec{16,UInt8})
    (_pshufb(_B1H, prev1 >> 0x04)) & (_pshufb(_B1L, prev1 & _LOWMASK)) & (_pshufb(_B2H, input >> 0x04))
end
@inline function _check_multibyte(input::Vec{16,UInt8}, prev::Vec{16,UInt8}, sc::Vec{16,UInt8})
    prev2 = shufflevector(prev, input, Val(_P2))
    prev3 = shufflevector(prev, input, Val(_P3))
    must23 = _usubsat(prev2, Vec{16,UInt8}(0xDF)) | _usubsat(prev3, Vec{16,UInt8}(0xEF))  # 3-/4-byte lead?
    pos = reinterpret(Vec{16,Int8}, must23) > Vec{16,Int8}(0)                              # signed_gt(0) mask
    vifelse(pos, _HI80, _ZERO) ⊻ sc                                                        # required-cont ⊻ specials
end
@inline function _check_block(error, prev, input)
    prev1 = shufflevector(prev, input, Val(_P1))
    sc = _check_special(input, prev1)
    (error | _check_multibyte(input, prev, sc), input, _usubsat(input, _INCOMPLETE))  # error, new prev, incomplete
end

@inline _ascii(v::Vec{16,UInt8}) = _movemask(v) == Int32(0)

# Pointer core (caller GC-preserves the backing storage). Allocation-free.
function _isvalid_utf8(p::Ptr{UInt8}, n::Int)
    prev = _ZERO; incomplete = _ZERO; error = _ZERO
    i = 0
    @inbounds while i + 64 <= n                       # 64-byte chunk: one ASCII check amortized
        v0 = vload(Vec{16,UInt8}, p + i);      v1 = vload(Vec{16,UInt8}, p + i + 16)
        v2 = vload(Vec{16,UInt8}, p + i + 32); v3 = vload(Vec{16,UInt8}, p + i + 48)
        if _ascii(v0 | v1 | v2 | v3)
            error |= incomplete; prev = v3; incomplete = _ZERO
        else
            error, prev, incomplete = _check_block(error, prev, v0)
            error, prev, incomplete = _check_block(error, prev, v1)
            error, prev, incomplete = _check_block(error, prev, v2)
            error, prev, incomplete = _check_block(error, prev, v3)
        end
        i += 64
    end
    @inbounds while i + 16 <= n                       # 16-byte remainder
        input = vload(Vec{16,UInt8}, p + i)
        if _ascii(input)
            error |= incomplete; prev = input; incomplete = _ZERO
        else
            error, prev, incomplete = _check_block(error, prev, input)
        end
        i += 16
    end
    if i < n                                          # tail (< 16 bytes): copy into a zero-padded buffer
        buf = zeros(UInt8, 16)
        @inbounds for k in 1:(n - i); buf[k] = unsafe_load(p, i + k); end
        GC.@preserve buf begin
            input = vload(Vec{16,UInt8}, pointer(buf))
            if _ascii(input)
                error |= incomplete; incomplete = _ZERO
            else
                error, prev, incomplete = _check_block(error, prev, input)
            end
        end
    end
    error |= incomplete
    return reduce(|, error) == 0x00                   # any_bit_set: error flags live in arbitrary bits
end

"""
    isvalid_utf8(data::AbstractVector{UInt8}) -> Bool
    isvalid_utf8(s::AbstractString) -> Bool

Return `true` iff `data` is well-formed UTF-8. Byte-exact with `Base.isvalid`; SIMD-validates multibyte
sequences (not just the ASCII fast path).
"""
isvalid_utf8(data::DenseVector{UInt8}) = GC.@preserve data _isvalid_utf8(pointer(data), length(data))
isvalid_utf8(cu::Base.CodeUnits{UInt8}) = GC.@preserve cu _isvalid_utf8(pointer(cu), length(cu))
isvalid_utf8(s::AbstractString) = isvalid_utf8(codeunits(String(s)))
isvalid_utf8(data::AbstractVector{UInt8}) = isvalid_utf8(collect(data))   # fallback for non-dense

end # module Utf8
