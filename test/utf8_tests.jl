@testitem "utf8_correctness" tags = [:utf8] begin
    using BlazingPorts.Utf8: isvalid_utf8
    using Random

    # Byte-exact with Base.isvalid across crafted edge cases, block boundaries, and random inputs.
    ref(b) = isvalid(String(copy(b)))
    crafted = [
        UInt8[], UInt8[0x41], UInt8[0x80], UInt8[0xC0, 0x80], UInt8[0xC2, 0xA9],            # empty/ascii/lone-cont/overlong/©
        UInt8[0xE0, 0x80, 0x80], UInt8[0xE0, 0xA0, 0x80], UInt8[0xED, 0xA0, 0x80],          # overlong3 / valid / surrogate
        UInt8[0xF0, 0x80, 0x80, 0x80], UInt8[0xF0, 0x90, 0x80, 0x80], UInt8[0xF4, 0x8F, 0xBF, 0xBF],  # overlong4/valid/U+10FFFF
        UInt8[0xF4, 0x90, 0x80, 0x80], UInt8[0xF5, 0x80, 0x80, 0x80], UInt8[0xE2, 0x82], UInt8[0xC2], # too-large/too-large/truncated×2
    ]
    for c in crafted
        @test isvalid_utf8(c) == ref(c)
    end

    # multibyte sequences straddling 16/32-byte block boundaries
    for L in 0:40, seq in (UInt8[0xE2, 0x82, 0xAC], UInt8[0xED, 0xA0, 0x80], UInt8[0xF0, 0x90, 0x80, 0x80])
        @test isvalid_utf8(vcat(fill(0x61, L), seq)) == ref(vcat(fill(0x61, L), seq))
        b = vcat(fill(0x61, L), seq, fill(0x61, 3))
        @test isvalid_utf8(b) == ref(b)
    end

    # random valid UTF-8 (1–4 byte chars) and random raw bytes (mostly invalid)
    Random.seed!(0xC0DE)
    pool = Char['a':'z'; ' '; 'é'; '€'; '一'; '𝄞']
    for _ in 1:20000
        b = Vector{UInt8}(codeunits(String(rand(pool, rand(0:50)))))
        @test isvalid_utf8(b) == ref(b)
    end
    for _ in 1:40000
        b = rand(UInt8, rand(0:80)); @test isvalid_utf8(b) == ref(b)
    end
    for _ in 1:2000
        b = rand(UInt8, rand(64:512)); @test isvalid_utf8(b) == ref(b)
    end

    # String dispatch agrees with Base
    @test isvalid_utf8("café — 日本語 𝄞") == isvalid("café — 日本語 𝄞")
end

@testitem "utf8_strictmode" tags = [:utf8] begin
    # The validation kernel is a SHUFFLE/LOOKUP kernel (three `pshufb` nibble lookups + bitwise ops,
    # ~ZERO arithmetic). This audits it AND probes how StrictMode characterizes non-arithmetic SIMD.
    import BlazingPorts.Utf8 as U
    using SIMD: Vec
    using StrictMode, AllocCheck, JET

    input = Vec{32,UInt8}(ntuple(i -> UInt8(0xE0 + (i % 16)), 32))
    prev1 = Vec{32,UInt8}(ntuple(i -> UInt8(0x80 + (i % 16)), 32))
    U._check_special(input, prev1)                              # warm

    # The kernel must lower to vector ops (<32 x i8>) and not allocate.
    @assert_vectorized U._check_special(input, prev1)
    @assert_noalloc    U._check_special(input, prev1)
    @assert_typestable U._check_special(input, prev1)

    # FEEDBACK PROBE (F33 candidate): kernel_report's intensity is FP/int-ARITHMETIC-centric. This kernel
    # is all pshufb shuffles + bitwise — ~0 arithmetic. Print the report so a human/agent can see whether
    # it mischaracterizes a perfectly-vectorized shuffle kernel (the conjectured "data-movement" blind spot).
    if isdefined(StrictMode, :kernel_report)
        @info "kernel_report on the UTF-8 shuffle kernel (F33 probe)" report = sprint(show,
            StrictMode.kernel_report(U._check_special, (Vec{32,UInt8}, Vec{32,UInt8})))
    end
end
