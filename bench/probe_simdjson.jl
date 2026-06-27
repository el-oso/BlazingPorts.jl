# Probe: simd-json (Rust) vs Julia JSON parsers (JSON.jl eager, JSON3.jl lazy-tape), single-thread.
#   RAYON_NUM_THREADS=1 taskset -c 4 julia -O3 -t 1 --project=bench bench/probe_simdjson.jl
# Probe-first (CLAUDE.md rule 1): measure before porting; ≥0.96 parity ⇒ document-skip.
#
# Apples-to-apples is the whole game here:
#   * EAGER:  JSON.parse → Dict/Vector (full materialization)  vs  simd-json to_borrowed_value + DOM walk.
#   * LAZY:   JSON3.read → tape (lazy values)                  vs  simd-json to_tape.
# simd-json unescapes IN PLACE, so each parse copies the immutable source into a reusable scratch first;
# `memcpy only` times that copy so we can also report simd-json parse-only = total − copy. The Rust walk
# returns a checksum and Julia results hit a DCE sink so nothing is elided. Deterministic doc (seeded) so
# anyone reproduces the same bytes on a fresh checkout — no network, no external corpus.
include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Printf, Random
import JSON, JSON3
Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

# ── deterministic, representative JSON: array of records with a realistic type mix + some escapes ──────
function gen_json(target_bytes::Int)
    rng = Xoshiro(0x5113D90)
    cities = ["Zürich", "São Paulo", "København", "東京", "New \"York\"", "C:\\tmp\\x"]  # unicode + escapes
    words  = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet"]
    rstr(n) = String(rand(rng, 'a':'z', n))
    io = IOBuffer(); print(io, "[")
    i = 0
    while io.size < target_bytes
        i > 0 && print(io, ",")
        rec = (
            id = i,
            name = rstr(rand(rng, 4:12)),
            email = string(rstr(6), "@", rstr(5), ".com"),
            score = round(rand(rng) * 100; digits = 6),
            active = rand(rng, Bool),
            balance = (rand(rng) - 0.5) * 1e6,
            tags = [words[rand(rng, 1:length(words))] for _ in 1:rand(rng, 0:5)],
            address = (street = string(rand(rng, 1:9999), " ", rstr(8), " St"),
                       city = cities[rand(rng, 1:length(cities))],
                       zip = rand(rng, 10000:99999)),
        )
        JSON.print(io, rec)   # NamedTuple ⇒ deterministic field order
        i += 1
    end
    print(io, "]")
    return take!(io), i
end

const BYTES, NREC = gen_json(4 * 1024 * 1024)
const STR = String(copy(BYTES))
const NB = length(BYTES)
const SCRATCH = Vector{UInt8}(undef, NB + 64)
@printf("doc: %.2f MiB, %d records\n", NB / 1024^2, NREC)

# ── timed wrappers: @noinline over preallocated inputs, DCE sink ───────────────────────────────────────
@noinline function sj(which::UInt32)
    GC.@preserve BYTES SCRATCH ccall((:bp_simdjson_parse, LIB), UInt64,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, UInt32), BYTES, NB, SCRATCH, length(SCRATCH), which)
end
@noinline sj_cpy() = GC.@preserve BYTES SCRATCH ccall((:bp_simdjson_memcpy, LIB), UInt8,
    (Ptr{UInt8}, Csize_t, Ptr{UInt8}), BYTES, NB, SCRATCH)
# JSON.jl ≥ 1.6 (the rewrite): `isvalidjson` is a full structural scan with ZERO allocation — the fair,
# GC-clean analog to simd-json's tape pass. `parse` materializes Dict{String,Any} (Any-boxed, eager).
@noinline jvalid() = (x = JSON.isvalidjson(BYTES); Base.donotdelete(x); x)
@noinline jparse() = (x = JSON.parse(STR);         Base.donotdelete(x); x)
@noinline j3read() = (x = JSON3.read(BYTES);       Base.donotdelete(x); x)

# warm + correctness sanity (checksum must not be the u64::MAX parse-error sentinel)
let c0 = sj(UInt32(0)), c1 = sj(UInt32(1))
    (c0 == typemax(UInt64) || c1 == typemax(UInt64)) && error("simd-json parse error on the generated doc")
    jvalid() || error("JSON.isvalidjson says the generated doc is invalid")
    jparse(); j3read()
    @printf("checksums: DOM=%d tape=%d (sanity, non-error)\n", c0, c1)
end

println("\n=== simd-json vs Julia JSON (JSON.jl ≥ 1.6), $(round(NB/1024^2; digits=2)) MiB, single-thread ===")
p_valid  = run_probe("JSON.isvalidjson (structural scan)",  jvalid;               seconds = 6.0)
p_json   = run_probe("JSON.parse (eager Dict)",             jparse;               seconds = 6.0)
p_json3  = run_probe("JSON3.read (lazy tape)",              j3read;               seconds = 6.0)
p_sjdom  = run_probe("simd-json to_borrowed_value (DOM)",   () -> sj(UInt32(0));  seconds = 6.0)
p_sjtape = run_probe("simd-json to_tape (lazy)",            () -> sj(UInt32(1));  seconds = 6.0)
p_cpy    = run_probe("memcpy only (copy overhead)",         sj_cpy;               seconds = 4.0)

g(med) = NB / med / 1e9
gp(p)  = g(p.median)
dom_only  = NB / max(p_sjdom.median  - p_cpy.median, eps()) / 1e9   # simd-json parse-only (minus copy)
tape_only = NB / max(p_sjtape.median - p_cpy.median, eps()) / 1e9
@printf("\n  STRUCTURAL  JSON.isvalidjson (0-alloc scan): %.2f GB/s\n", gp(p_valid))
@printf("              simd-json tape (copy+parse):     %.2f GB/s  (parse-only %.2f)  → Julia/simd = %.2f×\n",
    gp(p_sjtape), tape_only, gp(p_valid)/gp(p_sjtape))
@printf("  EAGER       JSON.parse → Dict:               %.2f GB/s\n", gp(p_json))
@printf("              simd-json DOM (copy+parse):      %.2f GB/s  (parse-only %.2f)  → Julia/simd = %.2f×\n",
    gp(p_sjdom), dom_only, gp(p_json)/gp(p_sjdom))
@printf("  (ref) JSON3.read tape %.2f GB/s ;  memcpy %.2f GB/s (%.0f%% of simd-json DOM time)\n",
    gp(p_json3), gp(p_cpy), 100 * p_cpy.median / p_sjdom.median)

save_probes("simdjson", Probe[p_valid, p_json, p_json3, p_sjdom, p_sjtape, p_cpy])
println("\nsaved → bench/results/simdjson.json")
