"""
    Blake3

Pure-Julia BLAKE3 with a SIMD `hash_many` kernel (N=8 wide, AVX2/AVX-512 via SIMD.jl).

BLAKE3's throughput is dominated by compressing 1024-byte chunks. `hash_many` processes N
independent chunks in parallel by transposing state: each lane of `Vec{N,UInt32}` holds one
chunk's word, then G mixing runs across all N lanes simultaneously with a single SIMD op.

- `blake3(data) -> Vector{UInt8}`  — 32-byte digest (public API)
- Internal: `compress` (scalar), `_compress_N_chunks_full` (SIMD N-wide chunk compression)

Source: scalar algorithm from the BLAKE3 spec (CC0/Apache). SIMD adaptation is original.
No StrictMode dep (package rule). `@assert_vectorized`/`@assert_noalloc` live in test/.
"""
module Blake3

using SIMD: Vec, shufflevector

export blake3

# ── constants ──────────────────────────────────────────────────────────────────────────────────────
const IV1 = UInt32(0x6A09E667); const IV2 = UInt32(0xBB67AE85)
const IV3 = UInt32(0x3C6EF372); const IV4 = UInt32(0xA54FF53A)
const IV5 = UInt32(0x510E527F); const IV6 = UInt32(0x9B05688C)
const IV7 = UInt32(0x1F83D9AB); const IV8 = UInt32(0x5BE0CD19)

const FLAG_CHUNK_START = UInt32(1)
const FLAG_CHUNK_END   = UInt32(2)
const FLAG_PARENT      = UInt32(4)
const FLAG_ROOT        = UInt32(8)

const BLOCK_LEN = 64
const CHUNK_LEN = 1024
# SIMD width: 8 × UInt32 = 256-bit AVX2 register. N=8 targets AVX2 (the primary gap kernel).
const N = 16   # SIMD lanes: Vec{16,UInt32} → AVX-512 (zmm) on this Zen5; the crate auto-selects N=16 too

# Unkeyed hashing uses IV as the key
const KEY1 = IV1; const KEY2 = IV2; const KEY3 = IV3; const KEY4 = IV4
const KEY5 = IV5; const KEY6 = IV6; const KEY7 = IV7; const KEY8 = IV8

# ── G mixing function (scalar) ────────────────────────────────────────────────────────────────────
@inline function _g(a::UInt32, b::UInt32, c::UInt32, d::UInt32, mx::UInt32, my::UInt32)
    a = a + b + mx
    d = bitrotate(d ⊻ a, -16)
    c = c + d
    b = bitrotate(b ⊻ c, -12)
    a = a + b + my
    d = bitrotate(d ⊻ a, -8)
    c = c + d
    b = bitrotate(b ⊻ c, -7)
    return a, b, c, d
end

# ── Software-pipelined 4-way G (scalar, for parent compress) ─────────────────────────────────────
# Same logic as _g4 but for UInt32 instead of Vec{N,UInt32}.
# Interleaves 8 steps of 4 independent G calls for 4-way ILP in the scalar compress path.
@inline function _g4s(
        a1::UInt32,b1::UInt32,c1::UInt32,d1::UInt32,mx1::UInt32,my1::UInt32,
        a2::UInt32,b2::UInt32,c2::UInt32,d2::UInt32,mx2::UInt32,my2::UInt32,
        a3::UInt32,b3::UInt32,c3::UInt32,d3::UInt32,mx3::UInt32,my3::UInt32,
        a4::UInt32,b4::UInt32,c4::UInt32,d4::UInt32,mx4::UInt32,my4::UInt32)
    a1=a1+b1+mx1; a2=a2+b2+mx2; a3=a3+b3+mx3; a4=a4+b4+mx4
    d1=bitrotate(d1⊻a1,-16); d2=bitrotate(d2⊻a2,-16)
    d3=bitrotate(d3⊻a3,-16); d4=bitrotate(d4⊻a4,-16)
    c1=c1+d1; c2=c2+d2; c3=c3+d3; c4=c4+d4
    b1=bitrotate(b1⊻c1,-12); b2=bitrotate(b2⊻c2,-12)
    b3=bitrotate(b3⊻c3,-12); b4=bitrotate(b4⊻c4,-12)
    a1=a1+b1+my1; a2=a2+b2+my2; a3=a3+b3+my3; a4=a4+b4+my4
    d1=bitrotate(d1⊻a1,-8); d2=bitrotate(d2⊻a2,-8)
    d3=bitrotate(d3⊻a3,-8); d4=bitrotate(d4⊻a4,-8)
    c1=c1+d1; c2=c2+d2; c3=c3+d3; c4=c4+d4
    b1=bitrotate(b1⊻c1,-7); b2=bitrotate(b2⊻c2,-7)
    b3=bitrotate(b3⊻c3,-7); b4=bitrotate(b4⊻c4,-7)
    return a1,b1,c1,d1, a2,b2,c2,d2, a3,b3,c3,d3, a4,b4,c4,d4
end

# ── G mixing function (SIMD, single) ─────────────────────────────────────────────────────────────
# Same math, but each argument is Vec{N,UInt32}; lane k processes chunk k independently.
# rotate-right via (x >> r) | (x << (32-r)) — no bitrotate on Vec.
@inline function _g_simd(
        a::Vec{N,UInt32}, b::Vec{N,UInt32}, c::Vec{N,UInt32}, d::Vec{N,UInt32},
        mx::Vec{N,UInt32}, my::Vec{N,UInt32}) where N
    a = a + b + mx
    d = (d ⊻ a); d = (d >> 16) | (d << 16)
    c = c + d
    b = (b ⊻ c); b = (b >> 12) | (b << 20)
    a = a + b + my
    d = (d ⊻ a); d = (d >> 8)  | (d << 24)
    c = c + d
    b = (b ⊻ c); b = (b >> 7)  | (b << 25)
    return a, b, c, d
end

# ── Software-pipelined 4-way G (columns or diagonals) ────────────────────────────────────────────
# Interleaves the 8 steps of 4 independent G calls to expose 4-way ILP to the backend.
# The arithmetic is identical to 4 sequential _g_simd calls — only the statement order differs,
# so byte-exactness is guaranteed.
# ponytail: explicit 24-arg function avoids tuple boxing; LLVM sees all 4 chains at each step.
@inline function _g4(
        a1::V, b1::V, c1::V, d1::V, mx1::V, my1::V,
        a2::V, b2::V, c2::V, d2::V, mx2::V, my2::V,
        a3::V, b3::V, c3::V, d3::V, mx3::V, my3::V,
        a4::V, b4::V, c4::V, d4::V, mx4::V, my4::V) where {V <: Vec}
    # step 1: a = a + b + mx  (all 4, independent → 4-wide ILP)
    a1=a1+b1+mx1; a2=a2+b2+mx2; a3=a3+b3+mx3; a4=a4+b4+mx4
    # step 2: d = ror(d^a, 16)
    d1=(d1⊻a1); d1=(d1>>16)|(d1<<16)
    d2=(d2⊻a2); d2=(d2>>16)|(d2<<16)
    d3=(d3⊻a3); d3=(d3>>16)|(d3<<16)
    d4=(d4⊻a4); d4=(d4>>16)|(d4<<16)
    # step 3: c = c + d
    c1=c1+d1; c2=c2+d2; c3=c3+d3; c4=c4+d4
    # step 4: b = ror(b^c, 12)
    b1=(b1⊻c1); b1=(b1>>12)|(b1<<20)
    b2=(b2⊻c2); b2=(b2>>12)|(b2<<20)
    b3=(b3⊻c3); b3=(b3>>12)|(b3<<20)
    b4=(b4⊻c4); b4=(b4>>12)|(b4<<20)
    # step 5: a = a + b + my
    a1=a1+b1+my1; a2=a2+b2+my2; a3=a3+b3+my3; a4=a4+b4+my4
    # step 6: d = ror(d^a, 8)
    d1=(d1⊻a1); d1=(d1>>8)|(d1<<24)
    d2=(d2⊻a2); d2=(d2>>8)|(d2<<24)
    d3=(d3⊻a3); d3=(d3>>8)|(d3<<24)
    d4=(d4⊻a4); d4=(d4>>8)|(d4<<24)
    # step 7: c = c + d
    c1=c1+d1; c2=c2+d2; c3=c3+d3; c4=c4+d4
    # step 8: b = ror(b^c, 7)
    b1=(b1⊻c1); b1=(b1>>7)|(b1<<25)
    b2=(b2⊻c2); b2=(b2>>7)|(b2<<25)
    b3=(b3⊻c3); b3=(b3>>7)|(b3<<25)
    b4=(b4⊻c4); b4=(b4>>7)|(b4<<25)
    return a1,b1,c1,d1, a2,b2,c2,d2, a3,b3,c3,d3, a4,b4,c4,d4
end

# ── Scalar compress: 16-word output (spec §2.3) ───────────────────────────────────────────────────
# 7 rounds × 8 G-calls each; all message indices are literals (no runtime tuple indexing).
# Message schedule for rounds 1–7 (0-indexed word positions, expanded from permute^k):
#   Round 1: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#   Round 2: [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
#   Round 3: [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
#   Round 4: [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
#   Round 5: [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
#   Round 6: [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
#   Round 7: [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
@inline function compress(
        cv1::UInt32, cv2::UInt32, cv3::UInt32, cv4::UInt32,
        cv5::UInt32, cv6::UInt32, cv7::UInt32, cv8::UInt32,
        m1::UInt32,  m2::UInt32,  m3::UInt32,  m4::UInt32,
        m5::UInt32,  m6::UInt32,  m7::UInt32,  m8::UInt32,
        m9::UInt32,  m10::UInt32, m11::UInt32, m12::UInt32,
        m13::UInt32, m14::UInt32, m15::UInt32, m16::UInt32,
        counter_lo::UInt32, counter_hi::UInt32,
        block_len::UInt32, flags::UInt32)

    v1  = cv1;  v2  = cv2;  v3  = cv3;  v4  = cv4
    v5  = cv5;  v6  = cv6;  v7  = cv7;  v8  = cv8
    v9  = IV1;  v10 = IV2;  v11 = IV3;  v12 = IV4
    v13 = counter_lo;  v14 = counter_hi;  v15 = block_len;  v16 = flags

    # 7 rounds, each with 4 column + 4 diagonal G calls.
    # Each half-round uses _g4s: 4 independent G calls interleaved step-by-step
    # so the CPU back-end sees 4-way ILP at each arithmetic step.
    # Round 1 — schedule [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    v1,v5,v9,v13, v2,v6,v10,v14, v3,v7,v11,v15, v4,v8,v12,v16 =
        _g4s(v1,v5,v9,v13, m1,m2, v2,v6,v10,v14, m3,m4,
             v3,v7,v11,v15, m5,m6, v4,v8,v12,v16, m7,m8)
    v1,v6,v11,v16, v2,v7,v12,v13, v3,v8,v9,v14, v4,v5,v10,v15 =
        _g4s(v1,v6,v11,v16, m9,m10, v2,v7,v12,v13, m11,m12,
             v3,v8,v9,v14, m13,m14, v4,v5,v10,v15, m15,m16)
    # Round 2 — schedule [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
    v1,v5,v9,v13, v2,v6,v10,v14, v3,v7,v11,v15, v4,v8,v12,v16 =
        _g4s(v1,v5,v9,v13, m3,m7, v2,v6,v10,v14, m4,m11,
             v3,v7,v11,v15, m8,m1, v4,v8,v12,v16, m5,m14)
    v1,v6,v11,v16, v2,v7,v12,v13, v3,v8,v9,v14, v4,v5,v10,v15 =
        _g4s(v1,v6,v11,v16, m2,m12, v2,v7,v12,v13, m13,m6,
             v3,v8,v9,v14, m10,m15, v4,v5,v10,v15, m16,m9)
    # Round 3 — schedule [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
    v1,v5,v9,v13, v2,v6,v10,v14, v3,v7,v11,v15, v4,v8,v12,v16 =
        _g4s(v1,v5,v9,v13, m4,m5, v2,v6,v10,v14, m11,m13,
             v3,v7,v11,v15, m14,m3, v4,v8,v12,v16, m8,m15)
    v1,v6,v11,v16, v2,v7,v12,v13, v3,v8,v9,v14, v4,v5,v10,v15 =
        _g4s(v1,v6,v11,v16, m7,m6, v2,v7,v12,v13, m10,m1,
             v3,v8,v9,v14, m12,m16, v4,v5,v10,v15, m9,m2)
    # Round 4 — schedule [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
    v1,v5,v9,v13, v2,v6,v10,v14, v3,v7,v11,v15, v4,v8,v12,v16 =
        _g4s(v1,v5,v9,v13, m11,m8, v2,v6,v10,v14, m13,m10,
             v3,v7,v11,v15, m15,m4, v4,v8,v12,v16, m14,m16)
    v1,v6,v11,v16, v2,v7,v12,v13, v3,v8,v9,v14, v4,v5,v10,v15 =
        _g4s(v1,v6,v11,v16, m5,m1, v2,v7,v12,v13, m12,m3,
             v3,v8,v9,v14, m6,m9, v4,v5,v10,v15, m2,m7)
    # Round 5 — schedule [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
    v1,v5,v9,v13, v2,v6,v10,v14, v3,v7,v11,v15, v4,v8,v12,v16 =
        _g4s(v1,v5,v9,v13, m13,m14, v2,v6,v10,v14, m10,m12,
             v3,v7,v11,v15, m16,m11, v4,v8,v12,v16, m15,m9)
    v1,v6,v11,v16, v2,v7,v12,v13, v3,v8,v9,v14, v4,v5,v10,v15 =
        _g4s(v1,v6,v11,v16, m8,m3, v2,v7,v12,v13, m6,m4,
             v3,v8,v9,v14, m1,m2, v4,v5,v10,v15, m7,m5)
    # Round 6 — schedule [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
    v1,v5,v9,v13, v2,v6,v10,v14, v3,v7,v11,v15, v4,v8,v12,v16 =
        _g4s(v1,v5,v9,v13, m10,m15, v2,v6,v10,v14, m12,m6,
             v3,v7,v11,v15, m9,m13, v4,v8,v12,v16, m16,m2)
    v1,v6,v11,v16, v2,v7,v12,v13, v3,v8,v9,v14, v4,v5,v10,v15 =
        _g4s(v1,v6,v11,v16, m14,m4, v2,v7,v12,v13, m1,m11,
             v3,v8,v9,v14, m3,m7, v4,v5,v10,v15, m5,m8)
    # Round 7 — schedule [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
    v1,v5,v9,v13, v2,v6,v10,v14, v3,v7,v11,v15, v4,v8,v12,v16 =
        _g4s(v1,v5,v9,v13, m12,m16, v2,v6,v10,v14, m6,m1,
             v3,v7,v11,v15, m2,m10, v4,v8,v12,v16, m9,m7)
    v1,v6,v11,v16, v2,v7,v12,v13, v3,v8,v9,v14, v4,v5,v10,v15 =
        _g4s(v1,v6,v11,v16, m15,m11, v2,v7,v12,v13, m3,m13,
             v3,v8,v9,v14, m4,m5, v4,v5,v10,v15, m8,m14)

    # Finalize: XOR top half (v1..v8) with bottom half (v9..v16)
    # Lower 8 words of output = first 8 state words ⊻ second 8 state words
    # Upper 8 words of output = second 8 state words ⊻ input CV
    return (
        v1 ⊻ v9,   v2 ⊻ v10,  v3 ⊻ v11,  v4 ⊻ v12,
        v5 ⊻ v13,  v6 ⊻ v14,  v7 ⊻ v15,  v8 ⊻ v16,
        v9 ⊻ cv1,  v10 ⊻ cv2, v11 ⊻ cv3, v12 ⊻ cv4,
        v13 ⊻ cv5, v14 ⊻ cv6, v15 ⊻ cv7, v16 ⊻ cv8,
    )
end

# ── Block loading ─────────────────────────────────────────────────────────────────────────────────
# Full 64-byte block: 16 LE UInt32 loads. On x86-64 LE this is 16 unaligned 32-bit loads.
@inline function _load_block(p::Ptr{UInt8})
    q = Ptr{UInt32}(p)
    return (unsafe_load(q,1),  unsafe_load(q,2),  unsafe_load(q,3),  unsafe_load(q,4),
            unsafe_load(q,5),  unsafe_load(q,6),  unsafe_load(q,7),  unsafe_load(q,8),
            unsafe_load(q,9),  unsafe_load(q,10), unsafe_load(q,11), unsafe_load(q,12),
            unsafe_load(q,13), unsafe_load(q,14), unsafe_load(q,15), unsafe_load(q,16))
end

# Partial block: copy into a zero-padded 64-byte buffer first, then load.
@inline function _load_partial_block(p::Ptr{UInt8}, nbytes::Int)
    stage = (UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),
             UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0))
    # ponytail: allocate a small on-stack buffer via a Vector (small alloc, only in tail path)
    buf = zeros(UInt8, 64)
    unsafe_copyto!(pointer(buf), p, nbytes)
    q = Ptr{UInt32}(pointer(buf))
    return (unsafe_load(q,1),  unsafe_load(q,2),  unsafe_load(q,3),  unsafe_load(q,4),
            unsafe_load(q,5),  unsafe_load(q,6),  unsafe_load(q,7),  unsafe_load(q,8),
            unsafe_load(q,9),  unsafe_load(q,10), unsafe_load(q,11), unsafe_load(q,12),
            unsafe_load(q,13), unsafe_load(q,14), unsafe_load(q,15), unsafe_load(q,16))
end

# ── 16×16 UInt32 matrix transpose via 4-stage butterfly ──────────────────────────────────────────
# Input:  r1..r16, each Vec{16,UInt32}, where r[c] = word c for chunk r.
# Output: w1..w16, each Vec{16,UInt32}, where w[r] = word w for chunk r.
# Uses 4 butterfly stages of shufflevector (all masks are compile-time Val tuples → no runtime
# variable indexing). Generates vpunpckld/vpunpckqd/vperm2i128/vshufi64x2 etc. on AVX-512.
# ponytail: 4-stage butterfly; upgrade to vpermt2d if LLVM stops folding these to it.
@inline function _transpose16x16(
        r1::V,r2::V,r3::V,r4::V,r5::V,r6::V,r7::V,r8::V,
        r9::V,r10::V,r11::V,r12::V,r13::V,r14::V,r15::V,r16::V) where {V<:Vec{16,UInt32}}
    # Stage 1: zip at 32-bit (1-element) granularity
    # lo(a,b): [a0,b0,a1,b1,...,a7,b7]; hi(a,b): [a8,b8,a9,b9,...,a15,b15]
    s1_lo_mask = Val((0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23))
    s1_hi_mask = Val((8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31))
    a01 = shufflevector(r1,  r2,  s1_lo_mask);  b01 = shufflevector(r1,  r2,  s1_hi_mask)
    a23 = shufflevector(r3,  r4,  s1_lo_mask);  b23 = shufflevector(r3,  r4,  s1_hi_mask)
    a45 = shufflevector(r5,  r6,  s1_lo_mask);  b45 = shufflevector(r5,  r6,  s1_hi_mask)
    a67 = shufflevector(r7,  r8,  s1_lo_mask);  b67 = shufflevector(r7,  r8,  s1_hi_mask)
    a89 = shufflevector(r9,  r10, s1_lo_mask);  b89 = shufflevector(r9,  r10, s1_hi_mask)
    aAB = shufflevector(r11, r12, s1_lo_mask);  bAB = shufflevector(r11, r12, s1_hi_mask)
    aCD = shufflevector(r13, r14, s1_lo_mask);  bCD = shufflevector(r13, r14, s1_hi_mask)
    aEF = shufflevector(r15, r16, s1_lo_mask);  bEF = shufflevector(r15, r16, s1_hi_mask)
    # Stage 2: zip at 64-bit (2-element) granularity
    # lo(a,b): [a0,a1,b0,b1,a2,a3,b2,b3,...]; hi(a,b): [a8,a9,b8,b9,...]
    s2_lo_mask = Val((0,1,16,17,2,3,18,19,4,5,20,21,6,7,22,23))
    s2_hi_mask = Val((8,9,24,25,10,11,26,27,12,13,28,29,14,15,30,31))
    c0  = shufflevector(a01, a23, s2_lo_mask);  c1  = shufflevector(a01, a23, s2_hi_mask)
    c2  = shufflevector(b01, b23, s2_lo_mask);  c3  = shufflevector(b01, b23, s2_hi_mask)
    c4  = shufflevector(a45, a67, s2_lo_mask);  c5  = shufflevector(a45, a67, s2_hi_mask)
    c6  = shufflevector(b45, b67, s2_lo_mask);  c7  = shufflevector(b45, b67, s2_hi_mask)
    c8  = shufflevector(a89, aAB, s2_lo_mask);  c9  = shufflevector(a89, aAB, s2_hi_mask)
    c10 = shufflevector(b89, bAB, s2_lo_mask);  c11 = shufflevector(b89, bAB, s2_hi_mask)
    c12 = shufflevector(aCD, aEF, s2_lo_mask);  c13 = shufflevector(aCD, aEF, s2_hi_mask)
    c14 = shufflevector(bCD, bEF, s2_lo_mask);  c15 = shufflevector(bCD, bEF, s2_hi_mask)
    # Stage 3: zip at 128-bit (4-element) granularity
    # lo(a,b): [a0..a3,b0..b3,a4..a7,b4..b7]; hi(a,b): [a8..a11,b8..b11,a12..a15,b12..b15]
    s3_lo_mask = Val((0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23))
    s3_hi_mask = Val((8,9,10,11,24,25,26,27,12,13,14,15,28,29,30,31))
    d0  = shufflevector(c0,  c4,  s3_lo_mask);  d1  = shufflevector(c0,  c4,  s3_hi_mask)
    d2  = shufflevector(c1,  c5,  s3_lo_mask);  d3  = shufflevector(c1,  c5,  s3_hi_mask)
    d4  = shufflevector(c2,  c6,  s3_lo_mask);  d5  = shufflevector(c2,  c6,  s3_hi_mask)
    d6  = shufflevector(c3,  c7,  s3_lo_mask);  d7  = shufflevector(c3,  c7,  s3_hi_mask)
    d8  = shufflevector(c8,  c12, s3_lo_mask);  d9  = shufflevector(c8,  c12, s3_hi_mask)
    d10 = shufflevector(c9,  c13, s3_lo_mask);  d11 = shufflevector(c9,  c13, s3_hi_mask)
    d12 = shufflevector(c10, c14, s3_lo_mask);  d13 = shufflevector(c10, c14, s3_hi_mask)
    d14 = shufflevector(c11, c15, s3_lo_mask);  d15 = shufflevector(c11, c15, s3_hi_mask)
    # Stage 4: zip at 256-bit (8-element) granularity (lane crossing)
    # lo(a,b): [a0..a7,b0..b7]; hi(a,b): [a8..a15,b8..b15]
    s4_lo_mask = Val((0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23))
    s4_hi_mask = Val((8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31))
    w1  = shufflevector(d0,  d8,  s4_lo_mask);  w2  = shufflevector(d0,  d8,  s4_hi_mask)
    w3  = shufflevector(d1,  d9,  s4_lo_mask);  w4  = shufflevector(d1,  d9,  s4_hi_mask)
    w5  = shufflevector(d2,  d10, s4_lo_mask);  w6  = shufflevector(d2,  d10, s4_hi_mask)
    w7  = shufflevector(d3,  d11, s4_lo_mask);  w8  = shufflevector(d3,  d11, s4_hi_mask)
    w9  = shufflevector(d4,  d12, s4_lo_mask);  w10 = shufflevector(d4,  d12, s4_hi_mask)
    w11 = shufflevector(d5,  d13, s4_lo_mask);  w12 = shufflevector(d5,  d13, s4_hi_mask)
    w13 = shufflevector(d6,  d14, s4_lo_mask);  w14 = shufflevector(d6,  d14, s4_hi_mask)
    w15 = shufflevector(d7,  d15, s4_lo_mask);  w16 = shufflevector(d7,  d15, s4_hi_mask)
    return w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16
end

# Load one 64-byte block from each of 16 chunks and return 16 word-major Vec{16,UInt32}.
# Two variants:
#   NTuple path: ptrs[k] + off  (legacy, used for N=8 test compatibility)
#   BasePtr16 path: base + (k-1)*CHUNK_LEN + off  (hot path, avoids 16-ptr register pressure)
# Both end in _transpose16x16; only the load addresses differ.
struct _BasePtr16 base::Ptr{UInt8} end  # thin wrapper to distinguish from NTuple dispatch

@inline function _load_and_transpose16(ptrs::NTuple{16,Ptr{UInt8}}, off::Int)
    @inline _vload(p) = unsafe_load(Ptr{Vec{16,UInt32}}(p))
    r1  = _vload(ptrs[1]  + off);  r2  = _vload(ptrs[2]  + off)
    r3  = _vload(ptrs[3]  + off);  r4  = _vload(ptrs[4]  + off)
    r5  = _vload(ptrs[5]  + off);  r6  = _vload(ptrs[6]  + off)
    r7  = _vload(ptrs[7]  + off);  r8  = _vload(ptrs[8]  + off)
    r9  = _vload(ptrs[9]  + off);  r10 = _vload(ptrs[10] + off)
    r11 = _vload(ptrs[11] + off);  r12 = _vload(ptrs[12] + off)
    r13 = _vload(ptrs[13] + off);  r14 = _vload(ptrs[14] + off)
    r15 = _vload(ptrs[15] + off);  r16 = _vload(ptrs[16] + off)
    return _transpose16x16(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16)
end
@inline function _load_and_transpose16(src::_BasePtr16, off::Int)
    # Compute load addresses from base + stride, one at a time — avoids 16-ptr NTuple.
    base = src.base
    @inline _vload(k) = unsafe_load(Ptr{Vec{16,UInt32}}(base + (k-1)*CHUNK_LEN + off))
    r1  = _vload(1);  r2  = _vload(2);  r3  = _vload(3);  r4  = _vload(4)
    r5  = _vload(5);  r6  = _vload(6);  r7  = _vload(7);  r8  = _vload(8)
    r9  = _vload(9);  r10 = _vload(10); r11 = _vload(11); r12 = _vload(12)
    r13 = _vload(13); r14 = _vload(14); r15 = _vload(15); r16 = _vload(16)
    return _transpose16x16(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16)
end

# Dispatch: for N=16 NTuple path and BasePtr16 path use SIMD transpose;
# for other N fall back to strided loads.
@inline function _load_block_transposed(ptrs::NTuple{16,Ptr{UInt8}}, off::Int, ::Val{16})
    _load_and_transpose16(ptrs, off)
end
@inline function _load_block_transposed(src::_BasePtr16, off::Int, ::Val{16})
    _load_and_transpose16(src, off)
end
@inline function _load_block_transposed(ptrs::NTuple{N,Ptr{UInt8}}, off::Int, ::Val{N}) where N
    @inline load_word(word_idx::Int) =
        Vec{N,UInt32}(ntuple(k -> unsafe_load(Ptr{UInt32}(ptrs[k] + off + (word_idx-1)*4)), Val(N)))
    (load_word(1),load_word(2),load_word(3),load_word(4),
     load_word(5),load_word(6),load_word(7),load_word(8),
     load_word(9),load_word(10),load_word(11),load_word(12),
     load_word(13),load_word(14),load_word(15),load_word(16))
end

# ── SIMD compress: N full chunks in parallel ───────────────────────────────────────────────────────
# ptrs[k] is the start of chunk k (each is CHUNK_LEN = 1024 bytes).
# All 16 blocks per chunk are full (1024 = 16×64). Returns 8 Vec{N,UInt32} chaining values.
# This is the hot path that must vectorize (tagged for @assert_vectorized in test/).
# Helper: returns Val{N} for the chunk count implied by the source type.
@inline _cnf_val(::NTuple{N,Ptr{UInt8}}) where N = Val(N)
@inline _cnf_val(::_BasePtr16) = Val(16)

# Entry point: dispatch to body via Val{N} so N is a compile-time constant.
# Accepts NTuple{N,Ptr{UInt8}} (all N, including N=8 test) or _BasePtr16 (N=16 hot path).
function _compress_N_chunks_full(
        ptrs,
        key1::UInt32, key2::UInt32, key3::UInt32, key4::UInt32,
        key5::UInt32, key6::UInt32, key7::UInt32, key8::UInt32,
        chunk_counter::UInt64)
    _compress_N_chunks_body(ptrs, _cnf_val(ptrs), key1,key2,key3,key4,key5,key6,key7,key8, chunk_counter)
end

function _compress_N_chunks_body(
        ptrs, ::Val{N},
        key1::UInt32, key2::UInt32, key3::UInt32, key4::UInt32,
        key5::UInt32, key6::UInt32, key7::UInt32, key8::UInt32,
        chunk_counter::UInt64) where N

    # Each lane k processes chunk (chunk_counter + k - 1) → different counter per lane.
    # BLAKE3 spec: the counter is the chunk counter (64-bit), split into lo/hi UInt32.
    # All N chunks start at chunk_counter, chunk_counter+1, ..., chunk_counter+N-1.
    counter_lo_vec = Vec{N,UInt32}(ntuple(k -> UInt32((chunk_counter + UInt64(k-1)) & 0xffff_ffff), Val(N)))
    counter_hi_vec = Vec{N,UInt32}(ntuple(k -> UInt32((chunk_counter + UInt64(k-1)) >> 32), Val(N)))

    # Initialize CV for all N chunks (same key — IV for unkeyed hashing)
    cv1 = Vec{N,UInt32}(key1); cv2 = Vec{N,UInt32}(key2)
    cv3 = Vec{N,UInt32}(key3); cv4 = Vec{N,UInt32}(key4)
    cv5 = Vec{N,UInt32}(key5); cv6 = Vec{N,UInt32}(key6)
    cv7 = Vec{N,UInt32}(key7); cv8 = Vec{N,UInt32}(key8)

    # Process 16 full 64-byte blocks per chunk
    for blk in 0:15
        is_first = (blk == 0)
        is_last  = (blk == 15)
        flags_scalar = UInt32(0)
        is_first && (flags_scalar |= FLAG_CHUNK_START)
        is_last  && (flags_scalar |= FLAG_CHUNK_END)

        byte_off = blk * BLOCK_LEN

        # Load 16 message words (word-major Vec{N,UInt32}) for this block.
        # For N=16: 16 contiguous 64-byte loads + 4-stage butterfly transpose.
        # Fallback for other N: N strided scalar loads per word.
        vm1,vm2,vm3,vm4,vm5,vm6,vm7,vm8,vm9,vm10,vm11,vm12,vm13,vm14,vm15,vm16 =
            _load_block_transposed(ptrs, byte_off, Val(N))

        # Build initial state vectors — counter differs per lane!
        va1  = cv1;  va2  = cv2;  va3  = cv3;  va4  = cv4
        va5  = cv5;  va6  = cv6;  va7  = cv7;  va8  = cv8
        va9  = Vec{N,UInt32}(IV1);  va10 = Vec{N,UInt32}(IV2)
        va11 = Vec{N,UInt32}(IV3);  va12 = Vec{N,UInt32}(IV4)
        va13 = counter_lo_vec   # lane k = counter_lo for chunk (chunk_counter + k - 1)
        va14 = counter_hi_vec   # lane k = counter_hi for chunk (chunk_counter + k - 1)
        va15 = Vec{N,UInt32}(UInt32(BLOCK_LEN))
        va16 = Vec{N,UInt32}(flags_scalar)

        # 7 rounds of SIMD G-mixes — same message schedule as scalar compress.
        # Each half-round uses _g4: 4 independent G calls interleaved step-by-step
        # so LLVM/the CPU back-end sees 4-way ILP at each arithmetic step rather
        # than 4 sequential full dependency chains. Arithmetic is identical → byte-exact.
        # Round 1 — schedule [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
            _g4(va1,va5,va9,va13, vm1,vm2, va2,va6,va10,va14, vm3,vm4,
                va3,va7,va11,va15, vm5,vm6, va4,va8,va12,va16, vm7,vm8)
        va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
            _g4(va1,va6,va11,va16, vm9,vm10, va2,va7,va12,va13, vm11,vm12,
                va3,va8,va9,va14, vm13,vm14, va4,va5,va10,va15, vm15,vm16)
        # Round 2 — schedule [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
        va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
            _g4(va1,va5,va9,va13, vm3,vm7, va2,va6,va10,va14, vm4,vm11,
                va3,va7,va11,va15, vm8,vm1, va4,va8,va12,va16, vm5,vm14)
        va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
            _g4(va1,va6,va11,va16, vm2,vm12, va2,va7,va12,va13, vm13,vm6,
                va3,va8,va9,va14, vm10,vm15, va4,va5,va10,va15, vm16,vm9)
        # Round 3 — schedule [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
        va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
            _g4(va1,va5,va9,va13, vm4,vm5, va2,va6,va10,va14, vm11,vm13,
                va3,va7,va11,va15, vm14,vm3, va4,va8,va12,va16, vm8,vm15)
        va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
            _g4(va1,va6,va11,va16, vm7,vm6, va2,va7,va12,va13, vm10,vm1,
                va3,va8,va9,va14, vm12,vm16, va4,va5,va10,va15, vm9,vm2)
        # Round 4 — schedule [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
        va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
            _g4(va1,va5,va9,va13, vm11,vm8, va2,va6,va10,va14, vm13,vm10,
                va3,va7,va11,va15, vm15,vm4, va4,va8,va12,va16, vm14,vm16)
        va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
            _g4(va1,va6,va11,va16, vm5,vm1, va2,va7,va12,va13, vm12,vm3,
                va3,va8,va9,va14, vm6,vm9, va4,va5,va10,va15, vm2,vm7)
        # Round 5 — schedule [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
        va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
            _g4(va1,va5,va9,va13, vm13,vm14, va2,va6,va10,va14, vm10,vm12,
                va3,va7,va11,va15, vm16,vm11, va4,va8,va12,va16, vm15,vm9)
        va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
            _g4(va1,va6,va11,va16, vm8,vm3, va2,va7,va12,va13, vm6,vm4,
                va3,va8,va9,va14, vm1,vm2, va4,va5,va10,va15, vm7,vm5)
        # Round 6 — schedule [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
        va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
            _g4(va1,va5,va9,va13, vm10,vm15, va2,va6,va10,va14, vm12,vm6,
                va3,va7,va11,va15, vm9,vm13, va4,va8,va12,va16, vm16,vm2)
        va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
            _g4(va1,va6,va11,va16, vm14,vm4, va2,va7,va12,va13, vm1,vm11,
                va3,va8,va9,va14, vm3,vm7, va4,va5,va10,va15, vm5,vm8)
        # Round 7 — schedule [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
        va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
            _g4(va1,va5,va9,va13, vm12,vm16, va2,va6,va10,va14, vm6,vm1,
                va3,va7,va11,va15, vm2,vm10, va4,va8,va12,va16, vm9,vm7)
        va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
            _g4(va1,va6,va11,va16, vm15,vm11, va2,va7,va12,va13, vm3,vm13,
                va3,va8,va9,va14, vm4,vm5, va4,va5,va10,va15, vm8,vm14)

        # Update running CV: XOR top half (va1..va8) with bottom half (va9..va16)
        cv1 = va1 ⊻ va9;  cv2 = va2 ⊻ va10
        cv3 = va3 ⊻ va11; cv4 = va4 ⊻ va12
        cv5 = va5 ⊻ va13; cv6 = va6 ⊻ va14
        cv7 = va7 ⊻ va15; cv8 = va8 ⊻ va16
    end

    return cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8
end

# ── SIMD parent compress: N pairs of CVs → N parent CVs ───────────────────────────────────────────
# Each lane k takes (left_cv[k], right_cv[k]) as a 64-byte "block" for parent compression.
# key=IV, counter=0, flags=PARENT. Structurally identical to _compress_N_chunks_full but
# without the per-block loop and with message loaded directly from Vec arguments.
# ponytail: shares _g4 from the chunk path; only state init differs.
function _compress_N_parents(
        lv1::VN,lv2::VN,lv3::VN,lv4::VN,lv5::VN,lv6::VN,lv7::VN,lv8::VN,
        rv1::VN,rv2::VN,rv3::VN,rv4::VN,rv5::VN,rv6::VN,rv7::VN,rv8::VN) where {VN<:Vec}
    N_ = length(VN)
    # Message: [left_cv (8 words) || right_cv (8 words)] = 16 words
    vm1=lv1; vm2=lv2; vm3=lv3; vm4=lv4; vm5=lv5; vm6=lv6; vm7=lv7; vm8=lv8
    vm9=rv1; vm10=rv2; vm11=rv3; vm12=rv4; vm13=rv5; vm14=rv6; vm15=rv7; vm16=rv8
    # State init: CV=IV (unkeyed), counter=0 (all lanes), flags=PARENT
    va1 =VN(KEY1); va2 =VN(KEY2); va3 =VN(KEY3); va4 =VN(KEY4)
    va5 =VN(KEY5); va6 =VN(KEY6); va7 =VN(KEY7); va8 =VN(KEY8)
    va9 =VN(IV1);  va10=VN(IV2);  va11=VN(IV3);  va12=VN(IV4)
    va13=VN(UInt32(0)); va14=VN(UInt32(0))
    va15=VN(UInt32(BLOCK_LEN)); va16=VN(FLAG_PARENT)
    # 7 rounds — same G schedule as _compress_N_chunks_full
    # Round 1
    va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
        _g4(va1,va5,va9,va13, vm1,vm2, va2,va6,va10,va14, vm3,vm4,
            va3,va7,va11,va15, vm5,vm6, va4,va8,va12,va16, vm7,vm8)
    va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
        _g4(va1,va6,va11,va16, vm9,vm10, va2,va7,va12,va13, vm11,vm12,
            va3,va8,va9,va14, vm13,vm14, va4,va5,va10,va15, vm15,vm16)
    # Round 2
    va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
        _g4(va1,va5,va9,va13, vm3,vm7, va2,va6,va10,va14, vm4,vm11,
            va3,va7,va11,va15, vm8,vm1, va4,va8,va12,va16, vm5,vm14)
    va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
        _g4(va1,va6,va11,va16, vm2,vm12, va2,va7,va12,va13, vm13,vm6,
            va3,va8,va9,va14, vm10,vm15, va4,va5,va10,va15, vm16,vm9)
    # Round 3
    va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
        _g4(va1,va5,va9,va13, vm4,vm5, va2,va6,va10,va14, vm11,vm13,
            va3,va7,va11,va15, vm14,vm3, va4,va8,va12,va16, vm8,vm15)
    va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
        _g4(va1,va6,va11,va16, vm7,vm6, va2,va7,va12,va13, vm10,vm1,
            va3,va8,va9,va14, vm12,vm16, va4,va5,va10,va15, vm9,vm2)
    # Round 4
    va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
        _g4(va1,va5,va9,va13, vm11,vm8, va2,va6,va10,va14, vm13,vm10,
            va3,va7,va11,va15, vm15,vm4, va4,va8,va12,va16, vm14,vm16)
    va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
        _g4(va1,va6,va11,va16, vm5,vm1, va2,va7,va12,va13, vm12,vm3,
            va3,va8,va9,va14, vm6,vm9, va4,va5,va10,va15, vm2,vm7)
    # Round 5
    va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
        _g4(va1,va5,va9,va13, vm13,vm14, va2,va6,va10,va14, vm10,vm12,
            va3,va7,va11,va15, vm16,vm11, va4,va8,va12,va16, vm15,vm9)
    va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
        _g4(va1,va6,va11,va16, vm8,vm3, va2,va7,va12,va13, vm6,vm4,
            va3,va8,va9,va14, vm1,vm2, va4,va5,va10,va15, vm7,vm5)
    # Round 6
    va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
        _g4(va1,va5,va9,va13, vm10,vm15, va2,va6,va10,va14, vm12,vm6,
            va3,va7,va11,va15, vm9,vm13, va4,va8,va12,va16, vm16,vm2)
    va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
        _g4(va1,va6,va11,va16, vm14,vm4, va2,va7,va12,va13, vm1,vm11,
            va3,va8,va9,va14, vm3,vm7, va4,va5,va10,va15, vm5,vm8)
    # Round 7
    va1,va5,va9,va13, va2,va6,va10,va14, va3,va7,va11,va15, va4,va8,va12,va16 =
        _g4(va1,va5,va9,va13, vm12,vm16, va2,va6,va10,va14, vm6,vm1,
            va3,va7,va11,va15, vm2,vm10, va4,va8,va12,va16, vm9,vm7)
    va1,va6,va11,va16, va2,va7,va12,va13, va3,va8,va9,va14, va4,va5,va10,va15 =
        _g4(va1,va6,va11,va16, vm15,vm11, va2,va7,va12,va13, vm3,vm13,
            va3,va8,va9,va14, vm4,vm5, va4,va5,va10,va15, vm8,vm14)
    # Finalize: XOR top and bottom halves
    return (va1⊻va9, va2⊻va10, va3⊻va11, va4⊻va12,
            va5⊻va13, va6⊻va14, va7⊻va15, va8⊻va16)
end

# ── Reduce 16 chunk CVs to one subtree root using SIMD parent compress ────────────────────────────
# Given 16 consecutive chunk CVs (word-major, in Vec{16,UInt32}), compute their subtree root.
# Uses 3 rounds of SIMD parent compress (N=8 → N=4 → N=2) + 1 scalar parent compress.
# This replaces 15 sequential scalar compress calls with 3 SIMD calls + 1 scalar.
# Correctness: equivalent to running _cv_stack_push! for chunks chunk_counter..chunk_counter+15
# where chunk_counter is a multiple of 16, then reading back the resulting pushed CV.
@inline function _reduce_16cvs_to_1(
        vcv1::Vec{16,UInt32},vcv2::Vec{16,UInt32},vcv3::Vec{16,UInt32},vcv4::Vec{16,UInt32},
        vcv5::Vec{16,UInt32},vcv6::Vec{16,UInt32},vcv7::Vec{16,UInt32},vcv8::Vec{16,UInt32})
    # Deinterleave masks: even lanes (left of each pair), odd lanes (right of each pair)
    el = Val((0,2,4,6,8,10,12,14)); er = Val((1,3,5,7,9,11,13,15))  # → Vec{8}
    # Level 1: 8 independent parent compresses (N=8)
    l1,l2,l3,l4,l5,l6,l7,l8 = shufflevector(vcv1,vcv1,el), shufflevector(vcv2,vcv2,el),
        shufflevector(vcv3,vcv3,el), shufflevector(vcv4,vcv4,el),
        shufflevector(vcv5,vcv5,el), shufflevector(vcv6,vcv6,el),
        shufflevector(vcv7,vcv7,el), shufflevector(vcv8,vcv8,el)
    r1,r2,r3,r4,r5,r6,r7,r8 = shufflevector(vcv1,vcv1,er), shufflevector(vcv2,vcv2,er),
        shufflevector(vcv3,vcv3,er), shufflevector(vcv4,vcv4,er),
        shufflevector(vcv5,vcv5,er), shufflevector(vcv6,vcv6,er),
        shufflevector(vcv7,vcv7,er), shufflevector(vcv8,vcv8,er)
    p1_1,p1_2,p1_3,p1_4,p1_5,p1_6,p1_7,p1_8 = _compress_N_parents(l1,l2,l3,l4,l5,l6,l7,l8,
                                                                      r1,r2,r3,r4,r5,r6,r7,r8)
    # Level 2: 4 independent parent compresses (N=4)
    el4 = Val((0,2,4,6)); er4 = Val((1,3,5,7))  # → Vec{4}
    l2_1,l2_2,l2_3,l2_4,l2_5,l2_6,l2_7,l2_8 =
        shufflevector(p1_1,p1_1,el4), shufflevector(p1_2,p1_2,el4),
        shufflevector(p1_3,p1_3,el4), shufflevector(p1_4,p1_4,el4),
        shufflevector(p1_5,p1_5,el4), shufflevector(p1_6,p1_6,el4),
        shufflevector(p1_7,p1_7,el4), shufflevector(p1_8,p1_8,el4)
    r2_1,r2_2,r2_3,r2_4,r2_5,r2_6,r2_7,r2_8 =
        shufflevector(p1_1,p1_1,er4), shufflevector(p1_2,p1_2,er4),
        shufflevector(p1_3,p1_3,er4), shufflevector(p1_4,p1_4,er4),
        shufflevector(p1_5,p1_5,er4), shufflevector(p1_6,p1_6,er4),
        shufflevector(p1_7,p1_7,er4), shufflevector(p1_8,p1_8,er4)
    p2_1,p2_2,p2_3,p2_4,p2_5,p2_6,p2_7,p2_8 = _compress_N_parents(l2_1,l2_2,l2_3,l2_4,l2_5,l2_6,l2_7,l2_8,
                                                                      r2_1,r2_2,r2_3,r2_4,r2_5,r2_6,r2_7,r2_8)
    # Level 3: 2 independent parent compresses (N=2)
    el2 = Val((0,2)); er2 = Val((1,3))  # → Vec{2}
    l3_1,l3_2,l3_3,l3_4,l3_5,l3_6,l3_7,l3_8 =
        shufflevector(p2_1,p2_1,el2), shufflevector(p2_2,p2_2,el2),
        shufflevector(p2_3,p2_3,el2), shufflevector(p2_4,p2_4,el2),
        shufflevector(p2_5,p2_5,el2), shufflevector(p2_6,p2_6,el2),
        shufflevector(p2_7,p2_7,el2), shufflevector(p2_8,p2_8,el2)
    r3_1,r3_2,r3_3,r3_4,r3_5,r3_6,r3_7,r3_8 =
        shufflevector(p2_1,p2_1,er2), shufflevector(p2_2,p2_2,er2),
        shufflevector(p2_3,p2_3,er2), shufflevector(p2_4,p2_4,er2),
        shufflevector(p2_5,p2_5,er2), shufflevector(p2_6,p2_6,er2),
        shufflevector(p2_7,p2_7,er2), shufflevector(p2_8,p2_8,er2)
    p3_1,p3_2,p3_3,p3_4,p3_5,p3_6,p3_7,p3_8 = _compress_N_parents(l3_1,l3_2,l3_3,l3_4,l3_5,l3_6,l3_7,l3_8,
                                                                      r3_1,r3_2,r3_3,r3_4,r3_5,r3_6,r3_7,r3_8)
    # Level 4: 1 scalar parent compress (2 remaining CVs → 1 root)
    lft1=p3_1[1]; lft2=p3_2[1]; lft3=p3_3[1]; lft4=p3_4[1]
    lft5=p3_5[1]; lft6=p3_6[1]; lft7=p3_7[1]; lft8=p3_8[1]
    rgt1=p3_1[2]; rgt2=p3_2[2]; rgt3=p3_3[2]; rgt4=p3_4[2]
    rgt5=p3_5[2]; rgt6=p3_6[2]; rgt7=p3_7[2]; rgt8=p3_8[2]
    root = compress(KEY1,KEY2,KEY3,KEY4,KEY5,KEY6,KEY7,KEY8,
                    lft1,lft2,lft3,lft4,lft5,lft6,lft7,lft8,
                    rgt1,rgt2,rgt3,rgt4,rgt5,rgt6,rgt7,rgt8,
                    UInt32(0),UInt32(0),UInt32(BLOCK_LEN),FLAG_PARENT)
    return root[1],root[2],root[3],root[4],root[5],root[6],root[7],root[8]
end

# ── CV stack (tree assembly, spec §5.1.2) ─────────────────────────────────────────────────────────
# cv_stack: flat Vector{UInt32}, 8 words per entry, indexed by 1-based depth.
# We merge when trailing zeros of total_chunks allow (binary tree structure).
# Returns the new stack_len after push + any merges.
@inline function _cv_stack_push!(cv_stack::Vector{UInt32}, stack_len::Int,
        cv1::UInt32, cv2::UInt32, cv3::UInt32, cv4::UInt32,
        cv5::UInt32, cv6::UInt32, cv7::UInt32, cv8::UInt32,
        chunk_counter::UInt64)
    total_chunks = chunk_counter + 1
    lv1 = cv1; lv2 = cv2; lv3 = cv3; lv4 = cv4
    lv5 = cv5; lv6 = cv6; lv7 = cv7; lv8 = cv8
    sl = stack_len

    # Merge while trailing zeros allow (spec §5.1.2)
    @inbounds while total_chunks & 1 == 0
        d = sl
        s1 = cv_stack[8*(d-1)+1]; s2 = cv_stack[8*(d-1)+2]
        s3 = cv_stack[8*(d-1)+3]; s4 = cv_stack[8*(d-1)+4]
        s5 = cv_stack[8*(d-1)+5]; s6 = cv_stack[8*(d-1)+6]
        s7 = cv_stack[8*(d-1)+7]; s8 = cv_stack[8*(d-1)+8]
        sl -= 1

        # parent_cv(left=s, right=lv) using key=IV, counter=0, flags=PARENT
        pout = compress(
            KEY1,KEY2,KEY3,KEY4,KEY5,KEY6,KEY7,KEY8,
            s1,s2,s3,s4,s5,s6,s7,s8,
            lv1,lv2,lv3,lv4,lv5,lv6,lv7,lv8,
            UInt32(0), UInt32(0), UInt32(BLOCK_LEN), FLAG_PARENT)
        lv1 = pout[1]; lv2 = pout[2]; lv3 = pout[3]; lv4 = pout[4]
        lv5 = pout[5]; lv6 = pout[6]; lv7 = pout[7]; lv8 = pout[8]
        total_chunks >>= 1
    end

    # Push merged result
    sl += 1
    @inbounds begin
        cv_stack[8*(sl-1)+1] = lv1; cv_stack[8*(sl-1)+2] = lv2
        cv_stack[8*(sl-1)+3] = lv3; cv_stack[8*(sl-1)+4] = lv4
        cv_stack[8*(sl-1)+5] = lv5; cv_stack[8*(sl-1)+6] = lv6
        cv_stack[8*(sl-1)+7] = lv7; cv_stack[8*(sl-1)+8] = lv8
    end
    return sl
end

# ── Compress a full chunk (scalar) → 8-word CV ────────────────────────────────────────────────────
function _compress_chunk_full_scalar(p::Ptr{UInt8}, chunk_counter::UInt64)
    cv1 = KEY1; cv2 = KEY2; cv3 = KEY3; cv4 = KEY4
    cv5 = KEY5; cv6 = KEY6; cv7 = KEY7; cv8 = KEY8
    counter_lo = UInt32(chunk_counter & 0xffff_ffff)
    counter_hi = UInt32(chunk_counter >> 32)

    for blk in 0:15
        flags = UInt32(0)
        blk == 0  && (flags |= FLAG_CHUNK_START)
        blk == 15 && (flags |= FLAG_CHUNK_END)
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16 =
            _load_block(p + blk * BLOCK_LEN)
        out = compress(cv1,cv2,cv3,cv4,cv5,cv6,cv7,cv8,
            m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,
            counter_lo, counter_hi, UInt32(BLOCK_LEN), flags)
        cv1 = out[1]; cv2 = out[2]; cv3 = out[3]; cv4 = out[4]
        cv5 = out[5]; cv6 = out[6]; cv7 = out[7]; cv8 = out[8]
    end
    return cv1,cv2,cv3,cv4,cv5,cv6,cv7,cv8
end

# ── Compress the final (partial) chunk, return CV + block + flags for root finalization ───────────
function _compress_last_chunk(p::Ptr{UInt8}, nbytes::Int, chunk_counter::UInt64)
    cv1 = KEY1; cv2 = KEY2; cv3 = KEY3; cv4 = KEY4
    cv5 = KEY5; cv6 = KEY6; cv7 = KEY7; cv8 = KEY8
    counter_lo = UInt32(chunk_counter & 0xffff_ffff)
    counter_hi = UInt32(chunk_counter >> 32)
    block_idx = 0
    remaining = nbytes

    # Process all blocks except the last into the running CV
    while remaining > BLOCK_LEN
        flags = block_idx == 0 ? FLAG_CHUNK_START : UInt32(0)
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16 =
            _load_block(p + block_idx * BLOCK_LEN)
        out = compress(cv1,cv2,cv3,cv4,cv5,cv6,cv7,cv8,
            m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,
            counter_lo, counter_hi, UInt32(BLOCK_LEN), flags)
        cv1 = out[1]; cv2 = out[2]; cv3 = out[3]; cv4 = out[4]
        cv5 = out[5]; cv6 = out[6]; cv7 = out[7]; cv8 = out[8]
        remaining -= BLOCK_LEN
        block_idx += 1
    end

    # Return last block uncompressed (caller will add ROOT flag and finalize)
    last_flags = (block_idx == 0 ? FLAG_CHUNK_START : UInt32(0)) | FLAG_CHUNK_END
    m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16 =
        _load_partial_block(p + block_idx * BLOCK_LEN, remaining)

    return (cv1,cv2,cv3,cv4,cv5,cv6,cv7,cv8,
            m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,
            UInt32(remaining), last_flags, counter_lo, counter_hi)
end

# ── Public API ────────────────────────────────────────────────────────────────────────────────────
"""
    blake3(data::AbstractVector{UInt8}) -> Vector{UInt8}

Compute the BLAKE3 hash of `data`, returning a 32-byte digest.

Uses `_compress_N_chunks_full` (Vec{8,UInt32} SIMD, targeting AVX2) for batches of N=8 full
1024-byte chunks, falling back to scalar for individual chunks and the tail.
"""
function blake3(data::AbstractVector{UInt8})
    out = Vector{UInt8}(undef, 32)
    GC.@preserve data out _blake3_raw(pointer(data), length(data), pointer(out))
    return out
end

# Build N chunk pointers from a base pointer, spaced CHUNK_LEN apart.
# A named function (not a lambda) avoids boxing `p` when p is mutable in the caller.
@inline _make_ptrs(p::Ptr{UInt8}, ::Val{N}) where {N} =
    ntuple(k -> p + (k-1)*CHUNK_LEN, Val(N))

function _blake3_raw(p::Ptr{UInt8}, n::Int, out::Ptr{UInt8})
    cv_stack = zeros(UInt32, 54 * 8)   # CV stack: max 54 entries (log2(2^54) chunks)
    stack_len = 0
    chunk_counter = UInt64(0)
    batch_counter = UInt64(0)  # counts 16-chunk super-batches for SIMD tree reduction

    # ── SIMD batch: N=16 full chunks at a time ────────────────────────────────────────────────────
    # Use strict > to ensure at least 1 byte remains for the mandatory final chunk.
    # BLAKE3 spec: there is always a "current chunk" with at least 1 byte; 0-byte input
    # is handled separately as a single degenerate chunk. The loops below guarantee nbytes>0
    # at the point we call _compress_last_chunk.
    # Key optimization: instead of pushing 16 individual CVs (causing 15 cascading scalar
    # parent compresses via _cv_stack_push!), reduce the 16 CVs to 1 subtree root using
    # 3 SIMD parent compress calls (N=8 → N=4 → N=2) + 1 scalar. Then push the subtree
    # root with batch_counter. The _cv_stack_push! binary-counter logic works correctly
    # because the batch roots form a valid BLAKE3 subtree at each level. Scalar chunks that
    # follow use chunk_counter directly; the final fold merges batch roots with scalar CVs
    # via the same _cv_stack_push! mechanism (proved correct for all chunk counts).
    while n > N * CHUNK_LEN
        # _BasePtr16 wraps the base pointer — avoids NTuple{16,Ptr} register pressure.
        # _compress_N_chunks_full dispatches to the N=16 SIMD path.
        vcv1,vcv2,vcv3,vcv4,vcv5,vcv6,vcv7,vcv8 = _compress_N_chunks_full(
            _BasePtr16(p), KEY1,KEY2,KEY3,KEY4,KEY5,KEY6,KEY7,KEY8, chunk_counter)

        # Reduce 16 chunk CVs → 1 subtree root (3 SIMD + 1 scalar parent compress)
        # then push as a single virtual "batch chunk" using batch_counter.
        sb1,sb2,sb3,sb4,sb5,sb6,sb7,sb8 = _reduce_16cvs_to_1(
            vcv1,vcv2,vcv3,vcv4,vcv5,vcv6,vcv7,vcv8)
        stack_len = _cv_stack_push!(cv_stack, stack_len,
            sb1,sb2,sb3,sb4,sb5,sb6,sb7,sb8, batch_counter)

        p += N * CHUNK_LEN
        n -= N * CHUNK_LEN
        chunk_counter += N
        batch_counter += 1
    end

    # ── Scalar: remaining full chunks (strict > ensures at least 1 byte for the final chunk) ────
    while n > CHUNK_LEN
        cv1,cv2,cv3,cv4,cv5,cv6,cv7,cv8 = _compress_chunk_full_scalar(p, chunk_counter)
        stack_len = _cv_stack_push!(cv_stack, stack_len,
            cv1,cv2,cv3,cv4,cv5,cv6,cv7,cv8, chunk_counter)
        p += CHUNK_LEN
        n -= CHUNK_LEN
        chunk_counter += 1
    end

    # ── Final chunk: compress and prepare the root Output ─────────────────────────────────────────
    fin_cv1,fin_cv2,fin_cv3,fin_cv4,fin_cv5,fin_cv6,fin_cv7,fin_cv8,
    fin_m1,fin_m2,fin_m3,fin_m4,fin_m5,fin_m6,fin_m7,fin_m8,
    fin_m9,fin_m10,fin_m11,fin_m12,fin_m13,fin_m14,fin_m15,fin_m16,
    fin_blen, fin_flags, fin_ctr_lo, fin_ctr_hi =
        _compress_last_chunk(p, n, chunk_counter)

    # ── Fold the CV stack up the right edge of the tree (spec §5.1.2) ────────────────────────────
    # The root Output is the last parent node. While the stack has entries, merge from the top.
    # We must delay writing ROOT until the very last node.
    while stack_len > 0
        # The current "right child" needs to be compressed into a CV first (non-root), then
        # merged as a parent with the stack top.
        rchild = compress(
            fin_cv1,fin_cv2,fin_cv3,fin_cv4,fin_cv5,fin_cv6,fin_cv7,fin_cv8,
            fin_m1,fin_m2,fin_m3,fin_m4,fin_m5,fin_m6,fin_m7,fin_m8,
            fin_m9,fin_m10,fin_m11,fin_m12,fin_m13,fin_m14,fin_m15,fin_m16,
            fin_ctr_lo, fin_ctr_hi, fin_blen, fin_flags)
        rchild_cv1 = rchild[1]; rchild_cv2 = rchild[2]
        rchild_cv3 = rchild[3]; rchild_cv4 = rchild[4]
        rchild_cv5 = rchild[5]; rchild_cv6 = rchild[6]
        rchild_cv7 = rchild[7]; rchild_cv8 = rchild[8]

        d = stack_len
        s1 = cv_stack[8*(d-1)+1]; s2 = cv_stack[8*(d-1)+2]
        s3 = cv_stack[8*(d-1)+3]; s4 = cv_stack[8*(d-1)+4]
        s5 = cv_stack[8*(d-1)+5]; s6 = cv_stack[8*(d-1)+6]
        s7 = cv_stack[8*(d-1)+7]; s8 = cv_stack[8*(d-1)+8]
        stack_len -= 1

        # The new "Output" is a parent: CV = key, block = [left_cv | right_cv]
        fin_cv1 = KEY1; fin_cv2 = KEY2; fin_cv3 = KEY3; fin_cv4 = KEY4
        fin_cv5 = KEY5; fin_cv6 = KEY6; fin_cv7 = KEY7; fin_cv8 = KEY8
        fin_m1  = s1;          fin_m2  = s2
        fin_m3  = s3;          fin_m4  = s4
        fin_m5  = s5;          fin_m6  = s6
        fin_m7  = s7;          fin_m8  = s8
        fin_m9  = rchild_cv1;  fin_m10 = rchild_cv2
        fin_m11 = rchild_cv3;  fin_m12 = rchild_cv4
        fin_m13 = rchild_cv5;  fin_m14 = rchild_cv6
        fin_m15 = rchild_cv7;  fin_m16 = rchild_cv8
        fin_blen    = UInt32(BLOCK_LEN)
        fin_flags   = FLAG_PARENT
        fin_ctr_lo  = UInt32(0)
        fin_ctr_hi  = UInt32(0)
    end

    # Root output: recompress with ROOT flag, write 32 bytes LE
    root_out = compress(
        fin_cv1,fin_cv2,fin_cv3,fin_cv4,fin_cv5,fin_cv6,fin_cv7,fin_cv8,
        fin_m1,fin_m2,fin_m3,fin_m4,fin_m5,fin_m6,fin_m7,fin_m8,
        fin_m9,fin_m10,fin_m11,fin_m12,fin_m13,fin_m14,fin_m15,fin_m16,
        fin_ctr_lo, fin_ctr_hi, fin_blen, fin_flags | FLAG_ROOT)

    # Write 32 bytes (8 LE UInt32 words)
    q = Ptr{UInt32}(out)
    unsafe_store!(q, root_out[1], 1); unsafe_store!(q, root_out[2], 2)
    unsafe_store!(q, root_out[3], 3); unsafe_store!(q, root_out[4], 4)
    unsafe_store!(q, root_out[5], 5); unsafe_store!(q, root_out[6], 6)
    unsafe_store!(q, root_out[7], 7); unsafe_store!(q, root_out[8], 8)
    return nothing
end

end # module Blake3
