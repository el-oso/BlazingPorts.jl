# Per-submodule StrictMode audit (mirrors PureFFT's bench/strictmode_audit.jl, but one audit PER
# crate submodule rather than a whole-package sweep — so each crate's guarantees are reported and
# gated independently).
#
# Run (checks must be enabled — a compile-time Preference, already set in bench/Project.toml):
#   julia --project=bench bench/strictmode_audit.jl

using BlazingPorts, StrictMode
using AllocCheck, JET   # StrictMode's analysis backend is a weak-dep extension — load it for the sweep
using SIMD: Vec         # kernel warm-up inputs (Blake3/Utf8/ByteOps take SIMD.Vec)

StrictMode.checks_enabled() || error("StrictMode checks disabled — set [preferences.StrictMode] checks_enabled=true")
StrictMode.backend_available() || error("StrictMode analysis backend not loaded — need `using AllocCheck, JET`")

# Warm each submodule's hot surface so the usage-driven sweep sees real compiled methods.
# Mirrors what each crate's *_strictmode testitem exercises.
function warm()
    SM = BlazingPorts.SmallMatrix
    a = SM.Vec3(1.0, 2.0, 3.0); b = SM.Vec3(4.0, 5.0, 6.0)
    SM.dot(a, b); SM.cross(a, b); SM.norm(a); SM.normalize(a); a + b; 2.0 * a
    v = SM.Vec4(1.0, 2.0, 3.0, 4.0)
    m = SM.Mat4(v, v, v, v); m * v; m * m

    # Factorizations: warm the Cholesky base kernel
    F = BlazingPorts.Factorizations
    A = Matrix{Float64}(undef, 16, 16)
    for j in 1:16, i in 1:16; A[i, j] = (i == j ? 16.0 : 0.25); end
    F.cholesky_llt!(A)

    # StringSearch: the SIMD memmem core
    SS = BlazingPorts.StringSearch
    h = rand(UInt8, 4096); pat = h[4000:4031]
    GC.@preserve h pat begin
        SS._find_substr(pointer(h), length(h), pointer(pat), length(pat))
    end
    SS.find_substr(h, pat)

    # IntFormat: branchless itoa
    IF = BlazingPorts.IntFormat
    buf = Vector{UInt8}(undef, 24)
    IF.format_int!(buf, 12345); IF.format_int!(buf, typemin(Int64))

    # SwissDict: group-probe kernels (build compiles the insert path too — exempted below)
    SD = BlazingPorts.SwissDict
    d = SD.SwissDict{UInt64, UInt64}()
    for i in UInt64(1):UInt64(1000); d[i] = i * 2; end
    SD._ht_keyindex(d, UInt64(500)); SD._ht_keyindex(d, UInt64(9999))
    GC.@preserve d begin
        sz = length(d.keys)
        idx, sh = SD._hashindex(UInt64(500), sz)
        SD._find_slot(pointer(d.slots), d.keys, UInt64(500), sz, sh, idx)
    end

    # Blake3: the N=8 SIMD compress + scalar compress
    B3 = BlazingPorts.Blake3
    data = [UInt8(i % 251) for i in 0:(8 * B3.CHUNK_LEN - 1)]
    GC.@preserve data begin
        p = pointer(data)
        ptrs = ntuple(k -> p + (k - 1) * B3.CHUNK_LEN, Val(8))
        B3._compress_N_chunks_full(ptrs, B3.KEY1, B3.KEY2, B3.KEY3, B3.KEY4, B3.KEY5, B3.KEY6, B3.KEY7, B3.KEY8, UInt64(0))
    end

    # Utf8: the pshufb validation kernel + public validator
    U = BlazingPorts.Utf8
    input = Vec{32, UInt8}(ntuple(i -> UInt8(0xE0 + (i % 16)), 32))
    prev1 = Vec{32, UInt8}(ntuple(i -> UInt8(0x80 + (i % 16)), 32))
    U._check_special(input, prev1)
    U.isvalid_utf8("café — 日本語 𝄞")

    # ByteOps: the base64 shuffle kernel
    B = BlazingPorts.ByteOps
    x = Vec{32, UInt8}(ntuple(i -> UInt8(i + 33), 32))
    B._enc24(x)

    return nothing
end
warm()

# (submodule, guarantees, exempt) per the triage table. Empty submodules (probe-first, not yet
# implemented) audit to zero checks — harmless. `exempt` names functions that allocate/dispatch
# BY DESIGN (public constructors, growing inserts, String-returning wrappers) — every entry is a
# deliberate, visible opt-out, not an accident.
const TARGETS = [
    (BlazingPorts.SmallMatrix, (:typestable, :noalloc), ()),
    (BlazingPorts.SpecialFns, (:inlined, :noalloc, :trimsafe), ()),
    (BlazingPorts.MatrixMultiply, (:vectorized, :noalloc), ()),
    # :vectorized module-wide is the F11 trap (index helpers/dispatchers have nothing to
    # vectorize); the per-kernel @assert_vectorized lives in factorizations_tests.jl. The 1-arg
    # cholesky_llt! allocates its workspace by design; the 2-arg hot form is asserted per-kernel.
    (BlazingPorts.Factorizations, (:typestable, :noalloc), (:cholesky_llt!,)),
    (BlazingPorts.StringSearch, (:typestable, :noalloc), ()),
    (BlazingPorts.IntFormat, (:typestable, :noalloc), ()),
    (BlazingPorts.SwissDict, (:typestable, :noalloc), ()),
    (BlazingPorts.Blake3, (:typestable, :noalloc), (:__init__,)),   # one-time module init allocates by design
    (BlazingPorts.Utf8, (:typestable, :noalloc), ()),
    (BlazingPorts.ByteOps, (:typestable, :noalloc), ()),
]

total_fail = 0
for (M, guarantees, exempt) in TARGETS
    println("\n══ audit $M  guarantees=$(guarantees) ", "═"^20)
    fs = audit(M; sweep = true, guarantees = guarantees, exempt = exempt, format = :text)
    nf = nfailures(fs)
    global total_fail += nf
    println("  → $(length(fs)) (method, guarantee) checks, $nf failure(s).")
end

println()
total_fail == 0 || error("StrictMode found $total_fail failure(s) across submodules.")
println("All implemented submodules satisfy their declared StrictMode guarantees. ✓")
