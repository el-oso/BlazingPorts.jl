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

using SIMD: Vec

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

# ── G mixing function (SIMD) ──────────────────────────────────────────────────────────────────────
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

    # Round 1 — schedule [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    v1,v5,v9,v13   = _g(v1,v5,v9,v13,   m1,  m2)
    v2,v6,v10,v14  = _g(v2,v6,v10,v14,  m3,  m4)
    v3,v7,v11,v15  = _g(v3,v7,v11,v15,  m5,  m6)
    v4,v8,v12,v16  = _g(v4,v8,v12,v16,  m7,  m8)
    v1,v6,v11,v16  = _g(v1,v6,v11,v16,  m9,  m10)
    v2,v7,v12,v13  = _g(v2,v7,v12,v13,  m11, m12)
    v3,v8,v9,v14   = _g(v3,v8,v9,v14,   m13, m14)
    v4,v5,v10,v15  = _g(v4,v5,v10,v15,  m15, m16)
    # Round 2 — schedule [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
    v1,v5,v9,v13   = _g(v1,v5,v9,v13,   m3,  m7)
    v2,v6,v10,v14  = _g(v2,v6,v10,v14,  m4,  m11)
    v3,v7,v11,v15  = _g(v3,v7,v11,v15,  m8,  m1)
    v4,v8,v12,v16  = _g(v4,v8,v12,v16,  m5,  m14)
    v1,v6,v11,v16  = _g(v1,v6,v11,v16,  m2,  m12)
    v2,v7,v12,v13  = _g(v2,v7,v12,v13,  m13, m6)
    v3,v8,v9,v14   = _g(v3,v8,v9,v14,   m10, m15)
    v4,v5,v10,v15  = _g(v4,v5,v10,v15,  m16, m9)
    # Round 3 — schedule [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
    v1,v5,v9,v13   = _g(v1,v5,v9,v13,   m4,  m5)
    v2,v6,v10,v14  = _g(v2,v6,v10,v14,  m11, m13)
    v3,v7,v11,v15  = _g(v3,v7,v11,v15,  m14, m3)
    v4,v8,v12,v16  = _g(v4,v8,v12,v16,  m8,  m15)
    v1,v6,v11,v16  = _g(v1,v6,v11,v16,  m7,  m6)
    v2,v7,v12,v13  = _g(v2,v7,v12,v13,  m10, m1)
    v3,v8,v9,v14   = _g(v3,v8,v9,v14,   m12, m16)
    v4,v5,v10,v15  = _g(v4,v5,v10,v15,  m9,  m2)
    # Round 4 — schedule [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
    v1,v5,v9,v13   = _g(v1,v5,v9,v13,   m11, m8)
    v2,v6,v10,v14  = _g(v2,v6,v10,v14,  m13, m10)
    v3,v7,v11,v15  = _g(v3,v7,v11,v15,  m15, m4)
    v4,v8,v12,v16  = _g(v4,v8,v12,v16,  m14, m16)
    v1,v6,v11,v16  = _g(v1,v6,v11,v16,  m5,  m1)
    v2,v7,v12,v13  = _g(v2,v7,v12,v13,  m12, m3)
    v3,v8,v9,v14   = _g(v3,v8,v9,v14,   m6,  m9)
    v4,v5,v10,v15  = _g(v4,v5,v10,v15,  m2,  m7)
    # Round 5 — schedule [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
    v1,v5,v9,v13   = _g(v1,v5,v9,v13,   m13, m14)
    v2,v6,v10,v14  = _g(v2,v6,v10,v14,  m10, m12)
    v3,v7,v11,v15  = _g(v3,v7,v11,v15,  m16, m11)
    v4,v8,v12,v16  = _g(v4,v8,v12,v16,  m15, m9)
    v1,v6,v11,v16  = _g(v1,v6,v11,v16,  m8,  m3)
    v2,v7,v12,v13  = _g(v2,v7,v12,v13,  m6,  m4)
    v3,v8,v9,v14   = _g(v3,v8,v9,v14,   m1,  m2)
    v4,v5,v10,v15  = _g(v4,v5,v10,v15,  m7,  m5)
    # Round 6 — schedule [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
    v1,v5,v9,v13   = _g(v1,v5,v9,v13,   m10, m15)
    v2,v6,v10,v14  = _g(v2,v6,v10,v14,  m12, m6)
    v3,v7,v11,v15  = _g(v3,v7,v11,v15,  m9,  m13)
    v4,v8,v12,v16  = _g(v4,v8,v12,v16,  m16, m2)
    v1,v6,v11,v16  = _g(v1,v6,v11,v16,  m14, m4)
    v2,v7,v12,v13  = _g(v2,v7,v12,v13,  m1,  m11)
    v3,v8,v9,v14   = _g(v3,v8,v9,v14,   m3,  m7)
    v4,v5,v10,v15  = _g(v4,v5,v10,v15,  m5,  m8)
    # Round 7 — schedule [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
    v1,v5,v9,v13   = _g(v1,v5,v9,v13,   m12, m16)
    v2,v6,v10,v14  = _g(v2,v6,v10,v14,  m6,  m1)
    v3,v7,v11,v15  = _g(v3,v7,v11,v15,  m2,  m10)
    v4,v8,v12,v16  = _g(v4,v8,v12,v16,  m9,  m7)
    v1,v6,v11,v16  = _g(v1,v6,v11,v16,  m15, m11)
    v2,v7,v12,v13  = _g(v2,v7,v12,v13,  m3,  m13)
    v3,v8,v9,v14   = _g(v3,v8,v9,v14,   m4,  m5)
    v4,v5,v10,v15  = _g(v4,v5,v10,v15,  m8,  m14)

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

# ── SIMD compress: N full chunks in parallel ───────────────────────────────────────────────────────
# ptrs[k] is the start of chunk k (each is CHUNK_LEN = 1024 bytes).
# All 16 blocks per chunk are full (1024 = 16×64). Returns 8 Vec{N,UInt32} chaining values.
# This is the hot path that must vectorize (tagged for @assert_vectorized in test/).
function _compress_N_chunks_full(
        ptrs::NTuple{N,Ptr{UInt8}},
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

        # Load 16 message words from each of the N inputs at the same block offset.
        # Each word is a Vec{N,UInt32}: lane k = word from chunk k.
        # ponytail: load_word is @inline with literal word_idx; ntuple unrolled at compile time.
        @inline function load_word(word_idx::Int)
            off = byte_off + (word_idx - 1) * 4
            Vec{N,UInt32}(ntuple(k -> unsafe_load(Ptr{UInt32}(ptrs[k] + off)), Val(N)))
        end
        vm1  = load_word(1);  vm2  = load_word(2);  vm3  = load_word(3);  vm4  = load_word(4)
        vm5  = load_word(5);  vm6  = load_word(6);  vm7  = load_word(7);  vm8  = load_word(8)
        vm9  = load_word(9);  vm10 = load_word(10); vm11 = load_word(11); vm12 = load_word(12)
        vm13 = load_word(13); vm14 = load_word(14); vm15 = load_word(15); vm16 = load_word(16)

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
        # Round 1 — schedule [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        va1,va5,va9,va13   = _g_simd(va1,va5,va9,va13,   vm1,  vm2)
        va2,va6,va10,va14  = _g_simd(va2,va6,va10,va14,  vm3,  vm4)
        va3,va7,va11,va15  = _g_simd(va3,va7,va11,va15,  vm5,  vm6)
        va4,va8,va12,va16  = _g_simd(va4,va8,va12,va16,  vm7,  vm8)
        va1,va6,va11,va16  = _g_simd(va1,va6,va11,va16,  vm9,  vm10)
        va2,va7,va12,va13  = _g_simd(va2,va7,va12,va13,  vm11, vm12)
        va3,va8,va9,va14   = _g_simd(va3,va8,va9,va14,   vm13, vm14)
        va4,va5,va10,va15  = _g_simd(va4,va5,va10,va15,  vm15, vm16)
        # Round 2 — schedule [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
        va1,va5,va9,va13   = _g_simd(va1,va5,va9,va13,   vm3,  vm7)
        va2,va6,va10,va14  = _g_simd(va2,va6,va10,va14,  vm4,  vm11)
        va3,va7,va11,va15  = _g_simd(va3,va7,va11,va15,  vm8,  vm1)
        va4,va8,va12,va16  = _g_simd(va4,va8,va12,va16,  vm5,  vm14)
        va1,va6,va11,va16  = _g_simd(va1,va6,va11,va16,  vm2,  vm12)
        va2,va7,va12,va13  = _g_simd(va2,va7,va12,va13,  vm13, vm6)
        va3,va8,va9,va14   = _g_simd(va3,va8,va9,va14,   vm10, vm15)
        va4,va5,va10,va15  = _g_simd(va4,va5,va10,va15,  vm16, vm9)
        # Round 3 — schedule [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
        va1,va5,va9,va13   = _g_simd(va1,va5,va9,va13,   vm4,  vm5)
        va2,va6,va10,va14  = _g_simd(va2,va6,va10,va14,  vm11, vm13)
        va3,va7,va11,va15  = _g_simd(va3,va7,va11,va15,  vm14, vm3)
        va4,va8,va12,va16  = _g_simd(va4,va8,va12,va16,  vm8,  vm15)
        va1,va6,va11,va16  = _g_simd(va1,va6,va11,va16,  vm7,  vm6)
        va2,va7,va12,va13  = _g_simd(va2,va7,va12,va13,  vm10, vm1)
        va3,va8,va9,va14   = _g_simd(va3,va8,va9,va14,   vm12, vm16)
        va4,va5,va10,va15  = _g_simd(va4,va5,va10,va15,  vm9,  vm2)
        # Round 4 — schedule [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
        va1,va5,va9,va13   = _g_simd(va1,va5,va9,va13,   vm11, vm8)
        va2,va6,va10,va14  = _g_simd(va2,va6,va10,va14,  vm13, vm10)
        va3,va7,va11,va15  = _g_simd(va3,va7,va11,va15,  vm15, vm4)
        va4,va8,va12,va16  = _g_simd(va4,va8,va12,va16,  vm14, vm16)
        va1,va6,va11,va16  = _g_simd(va1,va6,va11,va16,  vm5,  vm1)
        va2,va7,va12,va13  = _g_simd(va2,va7,va12,va13,  vm12, vm3)
        va3,va8,va9,va14   = _g_simd(va3,va8,va9,va14,   vm6,  vm9)
        va4,va5,va10,va15  = _g_simd(va4,va5,va10,va15,  vm2,  vm7)
        # Round 5 — schedule [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
        va1,va5,va9,va13   = _g_simd(va1,va5,va9,va13,   vm13, vm14)
        va2,va6,va10,va14  = _g_simd(va2,va6,va10,va14,  vm10, vm12)
        va3,va7,va11,va15  = _g_simd(va3,va7,va11,va15,  vm16, vm11)
        va4,va8,va12,va16  = _g_simd(va4,va8,va12,va16,  vm15, vm9)
        va1,va6,va11,va16  = _g_simd(va1,va6,va11,va16,  vm8,  vm3)
        va2,va7,va12,va13  = _g_simd(va2,va7,va12,va13,  vm6,  vm4)
        va3,va8,va9,va14   = _g_simd(va3,va8,va9,va14,   vm1,  vm2)
        va4,va5,va10,va15  = _g_simd(va4,va5,va10,va15,  vm7,  vm5)
        # Round 6 — schedule [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
        va1,va5,va9,va13   = _g_simd(va1,va5,va9,va13,   vm10, vm15)
        va2,va6,va10,va14  = _g_simd(va2,va6,va10,va14,  vm12, vm6)
        va3,va7,va11,va15  = _g_simd(va3,va7,va11,va15,  vm9,  vm13)
        va4,va8,va12,va16  = _g_simd(va4,va8,va12,va16,  vm16, vm2)
        va1,va6,va11,va16  = _g_simd(va1,va6,va11,va16,  vm14, vm4)
        va2,va7,va12,va13  = _g_simd(va2,va7,va12,va13,  vm1,  vm11)
        va3,va8,va9,va14   = _g_simd(va3,va8,va9,va14,   vm3,  vm7)
        va4,va5,va10,va15  = _g_simd(va4,va5,va10,va15,  vm5,  vm8)
        # Round 7 — schedule [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
        va1,va5,va9,va13   = _g_simd(va1,va5,va9,va13,   vm12, vm16)
        va2,va6,va10,va14  = _g_simd(va2,va6,va10,va14,  vm6,  vm1)
        va3,va7,va11,va15  = _g_simd(va3,va7,va11,va15,  vm2,  vm10)
        va4,va8,va12,va16  = _g_simd(va4,va8,va12,va16,  vm9,  vm7)
        va1,va6,va11,va16  = _g_simd(va1,va6,va11,va16,  vm15, vm11)
        va2,va7,va12,va13  = _g_simd(va2,va7,va12,va13,  vm3,  vm13)
        va3,va8,va9,va14   = _g_simd(va3,va8,va9,va14,   vm4,  vm5)
        va4,va5,va10,va15  = _g_simd(va4,va5,va10,va15,  vm8,  vm14)

        # Update running CV: XOR top half (va1..va8) with bottom half (va9..va16)
        cv1 = va1 ⊻ va9;  cv2 = va2 ⊻ va10
        cv3 = va3 ⊻ va11; cv4 = va4 ⊻ va12
        cv5 = va5 ⊻ va13; cv6 = va6 ⊻ va14
        cv7 = va7 ⊻ va15; cv8 = va8 ⊻ va16
    end

    return cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8
end

# ── CV stack (tree assembly, spec §5.1.2) ─────────────────────────────────────────────────────────
# cv_stack: flat Vector{UInt32}, 8 words per entry, indexed by 1-based depth.
# We merge when trailing zeros of total_chunks allow (binary tree structure).
# Returns the new stack_len after push + any merges.
function _cv_stack_push!(cv_stack::Vector{UInt32}, stack_len::Int,
        cv1::UInt32, cv2::UInt32, cv3::UInt32, cv4::UInt32,
        cv5::UInt32, cv6::UInt32, cv7::UInt32, cv8::UInt32,
        chunk_counter::UInt64)
    total_chunks = chunk_counter + 1
    lv1 = cv1; lv2 = cv2; lv3 = cv3; lv4 = cv4
    lv5 = cv5; lv6 = cv6; lv7 = cv7; lv8 = cv8
    sl = stack_len

    # Merge while trailing zeros allow (spec §5.1.2)
    while total_chunks & 1 == 0
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
    cv_stack[8*(sl-1)+1] = lv1; cv_stack[8*(sl-1)+2] = lv2
    cv_stack[8*(sl-1)+3] = lv3; cv_stack[8*(sl-1)+4] = lv4
    cv_stack[8*(sl-1)+5] = lv5; cv_stack[8*(sl-1)+6] = lv6
    cv_stack[8*(sl-1)+7] = lv7; cv_stack[8*(sl-1)+8] = lv8
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

    # ── SIMD batch: N=8 full chunks at a time ─────────────────────────────────────────────────────
    # Use strict > to ensure at least 1 byte remains for the mandatory final chunk.
    # BLAKE3 spec: there is always a "current chunk" with at least 1 byte; 0-byte input
    # is handled separately as a single degenerate chunk. The loops below guarantee nbytes>0
    # at the point we call _compress_last_chunk.
    while n > N * CHUNK_LEN
        # Build ptrs without capturing mutable `p` in a closure — use a helper to avoid boxing.
        ptrs = _make_ptrs(p, Val(N))

        vcv1,vcv2,vcv3,vcv4,vcv5,vcv6,vcv7,vcv8 = _compress_N_chunks_full(
            ptrs, KEY1,KEY2,KEY3,KEY4,KEY5,KEY6,KEY7,KEY8, chunk_counter)

        # Extract per-chunk scalar CVs (lane k = chunk k) and push into the tree. One pass over the N
        # lanes — cold relative to the 7-round compress above, so the runtime Vec lane-index is fine.
        @inbounds for k in 1:N
            stack_len = _cv_stack_push!(cv_stack, stack_len,
                vcv1[k],vcv2[k],vcv3[k],vcv4[k],vcv5[k],vcv6[k],vcv7[k],vcv8[k], chunk_counter + UInt64(k-1))
        end

        p += N * CHUNK_LEN
        n -= N * CHUNK_LEN
        chunk_counter += N
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
