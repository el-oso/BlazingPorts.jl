# Probe: Rust `regex` crate (lazy-DFA + Teddy/memchr SIMD literal prefilter) vs Julia Base `Regex`
# (PCRE2 JIT, a C lib), single-thread. Compile-once, then time MATCH throughput (count non-overlapping
# matches over a fixed haystack). Faer-flavored: the Julia baseline wraps C, so a gap is "stdlib-wraps-C",
# not Julia-vs-Rust — still worth quantifying.
#   RAYON_NUM_THREADS=1 taskset -c 11 julia -O3 -t 1 --project=bench bench/probe_regex.jl
# Fairness: `eachmatch`+`count` allocates a RegexMatch per match (Julia API overhead), while the regex
# crate's `find_iter().count()` is allocation-free. So the PRIMARY Julia baseline is `pcre_count` — a
# direct PCRE2 `exec` loop reusing one match_data (allocation-free, isolates the engine). `eachmatch` is
# kept as the idiomatic-Julia number to show the API allocation cost.
include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Printf, Random
import Base.PCRE
Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

# ── deterministic text corpus (~8 MiB) with sprinkled matchable tokens (prime moduli ⇒ all appear) ────
function gen_corpus(target_bytes::Int)
    rng = Xoshiro(0x2E9E70)
    alt = ("alpha", "bravo", "charlie", "delta", "echo")
    rword(n) = String(rand(rng, 'a':'z', n))
    io = IOBuffer(); i = 0
    while io.size < target_bytes
        i += 1
        if     i % 503 == 0; print(io, "GADGET")
        elseif i % 401 == 0; print(io, rand(rng, 100:999), "-", rand(rng, 1000:9999))                 # phone
        elseif i % 307 == 0; print(io, rword(rand(rng, 3:8)), "@", rword(rand(rng, 3:8)), ".com")     # email
        elseif i % 211 == 0; print(io, alt[rand(rng, 1:5)])                                            # alternation
        else                 print(io, rword(rand(rng, 4:9)))
        end
        print(io, ' ')
    end
    return String(take!(io))
end

const STR = gen_corpus(8 * 1024 * 1024)
const BYTES = Vector{UInt8}(codeunits(STR))
const NB = length(BYTES)
@printf("corpus: %.2f MiB\n", NB / 1024^2)

const PATTERNS = [
    ("literal `GADGET`",             raw"GADGET"),
    ("alternation `(alpha|…|echo)`", raw"(alpha|bravo|charlie|delta|echo)"),
    ("email `[a-z]+@[a-z]+\\.com`",  raw"[a-z]{3,8}@[a-z]{3,8}\.com"),
    ("phone `[0-9]{3}-[0-9]{4}`",    raw"[0-9]{3}-[0-9]{4}"),
]

# allocation-free PCRE2 engine count (reuses one match_data; no RegexMatch objects)
@noinline function pcre_count(re::Regex, s::String)
    Base.compile(re)
    md = PCRE.create_match_data(re.regex)
    n = 0; off = 0; len = sizeof(s)
    @inbounds while off <= len
        PCRE.exec(re.regex, s, off, re.match_options, md) || break
        ov = PCRE.ovec_ptr(md)
        ms = Int(unsafe_load(ov, 1)); me = Int(unsafe_load(ov, 2))
        off = me > ms ? me : me + 1
        n += 1
    end
    PCRE.free_match_data(md)
    n
end
@noinline function each_count(re::Regex, s::String)   # idiomatic Julia (allocates a RegexMatch per match)
    n = 0; for _ in eachmatch(re, s); n += 1; end; n
end
@noinline rs_count(h::Ptr{Cvoid}) =
    GC.@preserve BYTES ccall((:bp_regex_count, LIB), Csize_t, (Ptr{Cvoid}, Ptr{UInt8}, Csize_t), h, BYTES, NB)

probes = Probe[]
g(p) = NB / p.median / 1e9
println("\n=== regex match throughput, $(round(NB/1024^2; digits=2)) MiB corpus, single-thread ===")
for (label, pat) in PATTERNS
    jr = Regex(pat)
    h = GC.@preserve pat ccall((:bp_regex_build, LIB), Ptr{Cvoid}, (Ptr{UInt8}, Csize_t), pat, sizeof(pat))
    h == C_NULL && error("regex crate failed to compile: $pat")
    n_pcre = pcre_count(jr, STR); n_each = each_count(jr, STR); n_rs = Int(rs_count(h))
    (n_pcre == n_each == n_rs) || @warn "count mismatch ($label): pcre=$n_pcre each=$n_each regex=$n_rs"
    p_pcre = run_probe("PCRE2 engine: $label", () -> pcre_count(jr, STR); seconds = 4.0)
    p_each = run_probe("PCRE2 eachmatch: $label", () -> each_count(jr, STR); seconds = 4.0)
    p_rs   = run_probe("regex crate: $label",     () -> rs_count(h);          seconds = 4.0)
    ccall((:bp_regex_free, LIB), Cvoid, (Ptr{Cvoid},), h)
    @printf("  %-30s matches=%-5d  PCRE2 %.2f (eachmatch %.2f)  regex %.2f GB/s  → PCRE2/regex = %.2f×\n",
        label, n_pcre, g(p_pcre), g(p_each), g(p_rs), g(p_pcre) / g(p_rs))
    push!(probes, p_pcre, p_each, p_rs)
end

save_probes("regex", probes)
println("\nsaved → bench/results/regex.json")
