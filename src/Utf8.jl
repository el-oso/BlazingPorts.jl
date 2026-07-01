"""
    Utf8

Pure-Julia **SIMD UTF-8 validation** (lemire / simdjson algorithm, ported from the `simdutf8` crate).

Base's `isvalid(::String)` SIMD-checks the all-ASCII fast path (~parity with SIMD) but falls to a scalar
codepath the moment multibyte content appears (~11× slower than `simdutf8` on accented/CJK/emoji text).
`isvalid_utf8` keeps the ASCII fast path **and** validates multibyte SIMD-wide: classify each byte against
its predecessors with three `pshufb` nibble-lookup tables + range checks (`Vec{32,UInt8}` → `<32 x i8>`,
AVX2 `vpshufb`), accumulating an error vector. Byte-exact with `Base.isvalid`.

- `isvalid_utf8(data::AbstractVector{UInt8}) -> Bool`
- `isvalid_utf8(s::AbstractString) -> Bool`

This is a **shuffle/lookup-dominated** kernel (≈0 arithmetic intensity); the `@assert_vectorized` /
`kernel_report` audit lives in `test/` (source carries no StrictMode dep). N=32 (AVX2).
"""
module Utf8

using SIMD: Vec, vload, shufflevector, vifelse

export isvalid_utf8

# ── SIMD primitives via llvmcall (AVX2 vpshufb / usub.sat.v32 / pmovmskb) ────────────────────────────
const _V32 = NTuple{32, VecElement{UInt8}}
_bin(intr) = ("declare <32 x i8> @$intr(<32 x i8>, <32 x i8>)\ndefine <32 x i8> @e(<32 x i8> %a, <32 x i8> %b) #0 {\n%r = call <32 x i8> @$intr(<32 x i8> %a, <32 x i8> %b)\nret <32 x i8> %r\n}\nattributes #0 = { alwaysinline }", "e")
const _PSHUFB  = _bin("llvm.x86.avx2.pshuf.b")     # per-128-bit-lane 16-byte lookup
const _USUBSAT = _bin("llvm.usub.sat.v32i8")
const _MOVEMASK = ("declare i32 @llvm.x86.avx2.pmovmskb(<32 x i8>)\ndefine i32 @e(<32 x i8> %a) #0 {\n%r = call i32 @llvm.x86.avx2.pmovmskb(<32 x i8> %a)\nret i32 %r\n}\nattributes #0 = { alwaysinline }", "e")
@inline _pshufb(t::Vec{32,UInt8}, i::Vec{32,UInt8}) = Vec(Base.llvmcall(_PSHUFB, _V32, Tuple{_V32,_V32}, t.data, i.data))
@inline _usubsat(a::Vec{32,UInt8}, b::Vec{32,UInt8}) = Vec(Base.llvmcall(_USUBSAT, _V32, Tuple{_V32,_V32}, a.data, b.data))
@inline _movemask(v::Vec{32,UInt8}) = Base.llvmcall(_MOVEMASK, Int32, Tuple{_V32}, v.data)
const _PREFETCH = ("declare void @llvm.prefetch.p0(ptr, i32, i32, i32)\ndefine void @e(i64 %p) #0 {\n%a = inttoptr i64 %p to ptr\ncall void @llvm.prefetch.p0(ptr %a, i32 0, i32 3, i32 1)\nret void\n}\nattributes #0 = { alwaysinline }", "e")
@inline _prefetch(p::Ptr{UInt8}) = Base.llvmcall(_PREFETCH, Cvoid, Tuple{UInt}, reinterpret(UInt, p))

# ── lookup tables (exact, from simdutf8 `algorithm.rs`) — 16-byte tables duplicated across both lanes ─
_dup(t::NTuple{16,UInt8}) = Vec{32,UInt8}((t..., t...))
const _B1H = _dup((0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02, 0x80,0x80,0x80,0x80, 0x21,0x01,0x15,0x49))
const _B1L = _dup((0xE7,0xA3,0x83,0x83,0x8B,0xCB,0xCB,0xCB,0xCB,0xCB,0xCB,0xCB,0xCB,0xDB,0xCB,0xCB))
const _B2H = _dup((0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01, 0xE6,0xAE,0xBA,0xBA, 0x01,0x01,0x01,0x01))
# is_incomplete threshold: 0xff except the last 3 lanes (4-/3-/any-byte-lead thresholds)
const _INCOMPLETE = Vec{32,UInt8}((ntuple(_->0xff, 29)..., 0xef, 0xdf, 0xbf))
const _LOWMASK = Vec{32,UInt8}(0x0F)
const _HI80    = Vec{32,UInt8}(0x80)
const _ZERO    = Vec{32,UInt8}(0x00)
const _P1 = ntuple(i->31+(i-1), 32)   # input.prev<k>(prev): shift right k, carry prev's tail (cross-lane)
const _P2 = ntuple(i->30+(i-1), 32)
const _P3 = ntuple(i->29+(i-1), 32)

# ── the validation kernel (per 32-byte block; carries `prev`/`incomplete`/`error`) ──────────────────
@inline function _check_special(input::Vec{32,UInt8}, prev1::Vec{32,UInt8})
    (_pshufb(_B1H, prev1 >> 0x04)) & (_pshufb(_B1L, prev1 & _LOWMASK)) & (_pshufb(_B2H, input >> 0x04))
end
@inline function _check_multibyte(input::Vec{32,UInt8}, prev::Vec{32,UInt8}, sc::Vec{32,UInt8})
    prev2 = shufflevector(prev, input, Val(_P2))
    prev3 = shufflevector(prev, input, Val(_P3))
    must23 = _usubsat(prev2, Vec{32,UInt8}(0xDF)) | _usubsat(prev3, Vec{32,UInt8}(0xEF))   # 3-/4-byte lead?
    pos = reinterpret(Vec{32,Int8}, must23) > Vec{32,Int8}(0)                               # signed_gt(0) mask
    vifelse(pos, _HI80, _ZERO) ⊻ sc                                                         # required-cont ⊻ specials
end
@inline function _check_block(error, prev, input)
    prev1 = shufflevector(prev, input, Val(_P1))
    sc = _check_special(input, prev1)
    (error | _check_multibyte(input, prev, sc), input, _usubsat(input, _INCOMPLETE))   # error, new prev, incomplete
end

@inline _ascii(v::Vec{32,UInt8}) = _movemask(v) == Int32(0)

# Pointer core (caller GC-preserves the backing storage). Allocation-free.
function _isvalid_utf8(p::Ptr{UInt8}, n::Int)
    prev = _ZERO; incomplete = _ZERO; error = _ZERO
    i = 0
    @inbounds while i + 128 <= n                      # 128-byte chunk (4 blocks): amortize the ASCII check
        i + 256 <= n && _prefetch(p + i + 256)        # prefetch one chunk ahead (hide memory latency)
        v0 = vload(Vec{32,UInt8}, p + i);      v1 = vload(Vec{32,UInt8}, p + i + 32)
        v2 = vload(Vec{32,UInt8}, p + i + 64); v3 = vload(Vec{32,UInt8}, p + i + 96)
        if _ascii(v0 | v1 | v2 | v3)
            error |= incomplete; prev = v3; incomplete = _ZERO
        else
            error, prev, incomplete = _check_block(error, prev, v0)
            error, prev, incomplete = _check_block(error, prev, v1)
            error, prev, incomplete = _check_block(error, prev, v2)
            error, prev, incomplete = _check_block(error, prev, v3)
        end
        i += 128
    end
    @inbounds while i + 32 <= n                       # 32-byte remainder
        input = vload(Vec{32,UInt8}, p + i)
        if _ascii(input)
            error |= incomplete; prev = input; incomplete = _ZERO
        else
            error, prev, incomplete = _check_block(error, prev, input)
        end
        i += 32
    end
    if i < n                                          # tail (< 32 bytes): zero-padded, stack-only
        # NTuple-built padded block instead of a heap buffer: `zeros(UInt8, 32)` here allocated on
        # every non-multiple-of-32 input (caught by the StrictMode bench audit, 2026-07-02). The
        # `let` gives the closure never-reassigned captures (`i` is loop-mutated → Core.Box trap).
        input = let pt = p + i, r = n - i
            Vec{32, UInt8}(ntuple(k -> k <= r ? unsafe_load(pt, k) : 0x00, Val(32)))
        end
        if _ascii(input)
            error |= incomplete; incomplete = _ZERO
        else
            error, prev, incomplete = _check_block(error, prev, input)
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
