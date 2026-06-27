# Reproducible 3-way blake3 COMPRESS-kernel proof: pure Julia (SIMD.jl→LLVM) vs blake3's pure-Rust path
# (rust_avx2 intrinsics→LLVM, 8-wide) vs blake3's hand-written AVX-512 assembly. Compress-only (no tree
# reduce), 16 MiB, single-thread. Settles "where is blake3's edge": it's bundled asm, not the language.
#   RAYON_NUM_THREADS=1 taskset -c 2 julia -O3 -t 1 --project=bench bench/probe_blake3_kernels.jl
# Saves the distributions to bench/results/blake3_kernels.json (replot with bench/plot_blake3_kernels.jl).
include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.Blake3 as B3
using SIMD: Vec
using Printf
Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB
const MIB = 1024 * 1024; const NB = 16 * MIB; const NCH = NB ÷ 1024
const data = [UInt8(i % 251) for i in 0:(NB - 1)]; const OUT = Vector{UInt8}(undef, 32)

# ours: LLVM AVX-512 16-wide compress (no tree reduce) over all 16-chunk batches.
# Calls _compress_N_chunks_body directly to force the PURE kernel even when the asm switch is on
# (otherwise _compress_N_chunks_full would route through blake3's asm and this line would not be "ours").
@noinline function ours()
    acc = Vec{16,UInt32}(0); nbat = (NB - 1) ÷ (16 * 1024)
    GC.@preserve data begin
        p = pointer(data)
        for k in 0:nbat-1
            v1,v2,v3,v4,v5,v6,v7,v8 = B3._compress_N_chunks_body(B3._BasePtr16(p + k*16*1024), Val(16),
                B3.KEY1,B3.KEY2,B3.KEY3,B3.KEY4,B3.KEY5,B3.KEY6,B3.KEY7,B3.KEY8, UInt64(k)*16)
            acc ⊻= v1 ⊻ v8
        end
    end
    Base.donotdelete(acc); sum(acc)
end

# Full end-to-end blake3() pipeline (leaf compress + tree reduce + root). The Preferences `blake3_asm`
# switch only swaps the LEAF; reduce/root stay pure — so this measures how much the switch moves the
# WHOLE hash, not just the kernel. We toggle B3._ASM_FN[] to run the asm-on vs forced-pure path in one
# process (same thermal/clock state), restoring it after.
@noinline function full_hash()
    GC.@preserve data OUT B3._blake3_raw(pointer(data), NB, pointer(OUT))
    Base.donotdelete(OUT); @inbounds OUT[1]
end

# our SWITCH path at compress-kernel scope: _compress_N_chunks_full routes through the vendored .S
# (blake3_hash_many_avx512) when blake3_asm is active. Same kernel as the crate's hand-asm bar, but
# reached via OUR ccall + the output transpose-back — i.e. exactly what the switch delivers per batch.
@noinline function ours_asm()
    acc = Vec{16,UInt32}(0); nbat = (NB - 1) ÷ (16 * 1024)
    GC.@preserve data begin
        p = pointer(data)
        for k in 0:nbat-1
            v1,v2,v3,v4,v5,v6,v7,v8 = B3._compress_N_chunks_full(B3._BasePtr16(p + k*16*1024),
                B3.KEY1,B3.KEY2,B3.KEY3,B3.KEY4,B3.KEY5,B3.KEY6,B3.KEY7,B3.KEY8, UInt64(k)*16)
            acc ⊻= v1 ⊻ v8
        end
    end
    Base.donotdelete(acc); sum(acc)
end
# blake3 via its own platform hash_many: which=0 AVX-512 asm, which=1 AVX2 pure-Rust intrinsics
@noinline function bl3(which::UInt32)
    GC.@preserve data OUT ccall((:bp_blake3_hashmany, LIB), Cvoid,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, UInt32), data, NCH, OUT, which)
    Base.donotdelete(OUT); sum(OUT)
end

ours(); bl3(UInt32(0)); bl3(UInt32(1))   # warm
println("\n=== blake3 compress kernel, 16 MiB, single-thread ===")
p_jl  = run_probe("Julia SIMD.jl (LLVM, AVX-512)", ours; seconds = 4.0)
p_asm = run_probe("blake3 hand-asm (AVX-512)",     () -> bl3(UInt32(0)); seconds = 4.0)
p_rs  = run_probe("Rust intrinsics (LLVM, AVX2)",  () -> bl3(UInt32(1)); seconds = 4.0)
g(p) = NB / p.median / 1e9
@printf("  Julia SIMD.jl (LLVM, AVX-512 16-wide): %.2f GB/s\n", g(p_jl))
@printf("  Rust intrinsics (LLVM, AVX2  8-wide):  %.2f GB/s   → Julia/Rust = %.2f×\n", g(p_rs), g(p_jl)/g(p_rs))
@printf("  blake3 hand-asm (AVX-512 16-wide):     %.2f GB/s   → asm/Julia  = %.2f×\n", g(p_asm), g(p_asm)/g(p_jl))

probes = Probe[p_jl, p_rs, p_asm]

# ── Preferences switch: add OUR asm-leaf (kernel scope) as the 4th bar, then the full-pipeline pair ──
if B3._asm_active()
    saved = B3._ASM_FN[]
    try
        B3._ASM_FN[] = saved                              # asm leaf on
        ours_asm()
        p_our_asm = run_probe("BlazingPorts asm-leaf (blake3_asm)", ours_asm; seconds = 4.0)
        push!(probes, p_our_asm)
        @printf("  BlazingPorts asm-leaf (switch, our ccall):  %.2f GB/s   → vs crate asm = %.2f×\n",
            g(p_our_asm), g(p_our_asm)/g(p_asm))
        full_hash()
        p_full_asm = run_probe("BlazingPorts.blake3() asm-leaf", full_hash; seconds = 8.0)
        B3._ASM_FN[] = Ptr{Cvoid}(0)                      # force pure leaf
        full_hash()
        p_full_pure = run_probe("BlazingPorts.blake3() pure",    full_hash; seconds = 8.0)
        push!(probes, p_full_asm, p_full_pure)
        println("\n=== full blake3() pipeline (switch), 16 MiB, single-thread ===")
        @printf("  asm-leaf:  %.2f GB/s\n", g(p_full_asm))
        @printf("  pure leaf: %.2f GB/s   → asm-switch speedup = %.2f×\n", g(p_full_pure), g(p_full_asm)/g(p_full_pure))
    finally
        B3._ASM_FN[] = saved
    end
else
    @warn "blake3 asm kernel not active (pref off / no AVX-512 / no cc) — skipping the switch probe"
end

save_probes("blake3_kernels", probes)
println("\nsaved → bench/results/blake3_kernels.json   (plot: bench/plot_blake3_kernels.jl)")
