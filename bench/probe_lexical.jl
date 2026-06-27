# Probe: Rust `lexical-core` (fast float parsing) vs Julia `parse(Float64, _)`. Sum all floats in a corpus.
include(joinpath(@__DIR__, "harness.jl")); using .Harness; using Printf, Random
Harness.single_thread!(); const LIB = Harness.RUST_LIB
function gen(n)  # n space-separated f64 tokens
    rng = Xoshiro(0xF10A7); io = IOBuffer()
    for i in 1:n; i>1 && print(io,' '); print(io, round((rand(rng)-0.5)*1e6; digits=rand(rng,1:8))); end
    String(take!(io))
end
const STR = gen(1_000_000); const BYTES = Vector{UInt8}(codeunits(STR)); const NB = length(BYTES)
const TOKS = split(STR)   # pre-tokenized (parsing kernel isolated from splitting)
@noinline function jl_sum()
    s = 0.0; @inbounds for t in TOKS; s += parse(Float64, t); end; s
end
@noinline rs_sum() = GC.@preserve BYTES ccall((:bp_lexical_sum_f64, LIB), Float64, (Ptr{UInt8}, Csize_t), BYTES, NB)
isapprox(jl_sum(), rs_sum(); rtol=1e-9) || @warn "sum mismatch $(jl_sum()) vs $(rs_sum())"
g(p) = NB / p.median / 1e9
@printf("corpus: %.2f MiB, %d floats\n", NB/1024^2, length(TOKS))
println("=== float parsing, single-thread ===")
p_jl = run_probe("Julia parse(Float64)", jl_sum; seconds=4.0); p_rs = run_probe("lexical-core", rs_sum; seconds=4.0)
@printf("  Julia %.2f  lexical %.2f GB/s  → Julia/lexical = %.2f×  (%.0f vs %.0f Mfloat/s)\n",
    g(p_jl), g(p_rs), g(p_jl)/g(p_rs), length(TOKS)/p_jl.median/1e6, length(TOKS)/p_rs.median/1e6)
save_probes("lexical", Probe[p_jl, p_rs]); println("saved → bench/results/lexical.json")
