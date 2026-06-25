# Reproducible head-to-head: ours (Factorizations.jl) vs OpenBLAS (LinearAlgebra) vs faer (Rust crate).
# The honesty harness — run it any time to check the campaign's "we beat faer" claims for yourself.
#
#   RAYON_NUM_THREADS=1 taskset -c 2 julia -O3 -t 1 --project=bench bench/compare_factorizations.jl
#   …optional explicit sizes:                                  …compare_factorizations.jl 256,512,1024,2048
#
# For low-noise numbers pin the CPU clock first (this host's base is 2 GHz; boost drifts):
#   sudo ../PureFFT.jl/bench/cpufreq_lock.sh pin 4500     # …and `restore` when done
#
# Methodology (matches CLAUDE.md): single-thread BOTH sides (BLAS.set_num_threads(1); faer built/run with
# RAYON_NUM_THREADS=1). Chairmarks `@be setup body evals=1`: the setup (copyto! refresh into scratch) runs
# per sample and is NOT timed; the factorization (body) is; evals=1 so the in-place mutation never factors
# an already-triangular matrix. Compare the MEDIAN; rel-σ printed so you can see the spread (faer is noisier
# than OpenBLAS run-to-run, so re-run a couple of times — a single sub-1.0 faer column at one size is noise,
# not a regression). GFLOP/s = (n³/3 Cholesky, 4n³/3 QR) / median. Ours uses its allocation-free fast path
# (preallocated CholWorkspace / QRWorkspace) so it's compared at its best, like faer's own in-place call.
#
# faer is auto-skipped (with a printed note) if the cdylib can't be built — so this still runs Julia-vs-
# OpenBLAS on a box without cargo. Raw samples are saved to bench/results/compare_factorizations.json.

using LinearAlgebra, Chairmarks, Printf, Random, JSON
using Statistics: median, quantile
BLAS.set_num_threads(1)
Random.seed!(0)

const HERE = @__DIR__
const ROOT = dirname(HERE)
# Standalone include of just the factorization source — no `using BlazingPorts` (keeps the load light and
# independent of the rest of the package). It's the same code that ships in the module.
include(joinpath(ROOT, "src", "Factorizations.jl"))
using .Factorizations: cholesky_llt!, qr_blocked!, CholWorkspace, QRWorkspace

# ── faer cdylib (optional): build it if cargo is here and the .so is stale/missing, else skip faer ──
const RUSTDIR = joinpath(HERE, "rust_compare", "rust")
const LIB = joinpath(RUSTDIR, "target", "release", "libblazing_compare.so")
function ensure_faer()
    isfile(LIB) && return true
    isnothing(Sys.which("cargo")) && return false
    @info "building faer shim: cargo build --release (in $RUSTDIR)"
    try
        run(setenv(Cmd(`cargo build --release`); dir = RUSTDIR))
        return isfile(LIB)
    catch e
        @warn "cargo build failed — skipping faer" exception = e
        return false
    end
end
const HAVE_FAER = ensure_faer()
@noinline faer_chol(A) = ccall((:faer_cholesky, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))
@noinline faer_qr_(A)  = ccall((:faer_qr,       LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), size(A, 1))

spd(n) = (M = randn(n, n); M * M' + n * I)                 # symmetric positive-definite
const MAX_STORE_SAMPLES = 2000                              # cap stored points (stats use ALL samples)
# Keep the full per-sample distribution (seconds) so the violin comparison plots regenerate from saved data
# without re-running. Robust spread = half-IQR / median (≈ a σ for clean data, but immune to the rare
# GC-pause outliers that blow up std/mean on the allocating paths — OpenBLAS qr! and the small-n flat
# driver). The median is the comparison statistic; relsigma just reports how tight it is.
function _subsample(s, k)
    length(s) ≤ k && return Float64.(s)
    z = sort(s); Float64.(z[unique(round.(Int, range(1, length(z); length = k)))])
end
function stats(b)
    s = Float64[x.time for x in b.samples]
    (median = median(s), relsigma = (quantile(s, 0.75) - quantile(s, 0.25)) / 2 / median(s),
     n = length(s), samples = _subsample(s, MAX_STORE_SAMPLES))
end

# returns NamedTuple of contender => (median, relsigma, n); contenders present depend on HAVE_FAER
function bench_chol(n)
    A = spd(n); cw = CholWorkspace(n); s = similar(A)
    r = Dict{String,Any}()
    r["openblas"] = stats(@be (copyto!(s, A); Symmetric(s, :L)) cholesky!(_)         evals=1 seconds=3)
    r["ours"]     = stats(@be (copyto!(s, A); s)                cholesky_llt!(_, cw)  evals=1 seconds=3)
    HAVE_FAER && (r["faer"] = stats(@be (copyto!(s, A); s)      faer_chol(_)          evals=1 seconds=3))
    r
end
function bench_qr(n)
    A = randn(n, n); tau = zeros(n); s = similar(A)
    qw = n >= 512 ? QRWorkspace(n) : nothing                # two-level path (n≥512) is alloc-free with a workspace
    r = Dict{String,Any}()
    r["openblas"] = stats(@be (copyto!(s, A); s) qr!(_) evals=1 seconds=3)
    r["ours"] = isnothing(qw) ? stats(@be (copyto!(s, A); s) qr_blocked!(_, tau)     evals=1 seconds=3) :
                                stats(@be (copyto!(s, A); s) qr_blocked!(_, tau, qw) evals=1 seconds=3)
    HAVE_FAER && (r["faer"] = stats(@be (copyto!(s, A); s) faer_qr_(_) evals=1 seconds=3))
    r
end

sizes = isempty(ARGS) ? [256, 512, 1024, 2048] : parse.(Int, split(ARGS[1], ','))
gflop(kind, n, t) = (kind == :chol ? n^3 / 3 : 4 * n^3 / 3) / t / 1e9

cpu_mhz() = try
    "$(parse(Int, strip(read("/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq", String))) ÷ 1000) MHz"
catch
    "unknown (pin with cpufreq_lock.sh for low noise)"
end
println("\nBlazingPorts factorizations — single-thread, median of Chairmarks @be (evals=1).")
println("CPU cpu2: ", cpu_mhz())
HAVE_FAER || println("⚠ faer skipped (no cargo / build failed) — showing Julia vs OpenBLAS only.")
get(ENV, "RAYON_NUM_THREADS", "") == "1" || (HAVE_FAER && println("⚠ RAYON_NUM_THREADS≠1 — faer may be multithreaded (unfair). Re-run with RAYON_NUM_THREADS=1."))

out = Dict{String,Any}()
for (label, kind, f) in (("Cholesky", :chol, bench_chol), ("QR", :qr, bench_qr))
    @printf("\n%-9s  n       OpenBLAS%s         ours        | %s\n", label, HAVE_FAER ? "        faer" : "",
            HAVE_FAER ? "ours/faer  ours/OB" : "ours/OB")
    rows = Dict{String,Any}()
    for n in sizes
        r = f(n)
        g(name) = gflop(kind, n, r[name].median)
        ob = g("openblas"); ou = g("ours")
        if HAVE_FAER
            fa = g("faer")
            @printf("  n=%-5d %7.1f(±%.0f%%) %7.1f(±%.0f%%) %7.1f(±%.0f%%) | %.3fx    %.3fx\n",
                n, ob, 100r["openblas"].relsigma, fa, 100r["faer"].relsigma, ou, 100r["ours"].relsigma, ou / fa, ou / ob)
        else
            @printf("  n=%-5d %7.1f(±%.0f%%)                 %7.1f(±%.0f%%) | %.3fx\n",
                n, ob, 100r["openblas"].relsigma, ou, 100r["ours"].relsigma, ou / ob)
        end
        rows[string(n)] = Dict(k => Dict("median_s" => v.median, "relsigma" => v.relsigma,
            "n" => v.n, "samples" => v.samples) for (k, v) in r)
    end
    out[label] = rows
end

resfile = joinpath(HERE, "results", "compare_factorizations.json")
open(resfile, "w") do io; JSON.print(io, out, 2); end
println("\nsaved raw medians → ", resfile)
println("(ratios > 1.0 ⇒ ours faster. faer noisier than OpenBLAS — re-run 2–3× before trusting a close call.)")
