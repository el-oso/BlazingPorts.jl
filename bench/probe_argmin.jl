# Probe: argmin (Tier 4 — optimization).
# Contenders:
#   Julia Optim.jl (LBFGS) vs Rust argmin crate (LBFGS)
# Workload: minimize 2-D Rosenbrock from start [-1.2, 1.0] to grad-tol 1e-5.
#   f(x,y) = 100*(y - x^2)^2 + (1 - x)^2,  optimum [1,1], f=0.
#   Analytic gradient supplied — no finite-difference overhead.
#
# σ-DISCIPLINE: Both sides allocate per optimize() call (Optim.jl ~11KB, argmin Rust heap).
#   Single-call timing is dominated by GC pauses (Julia σ > 2000%).  Fix: batch BATCH
#   calls and time the batch — GC is amortized over BATCH iterations → σ < 5% both sides.
#   Per-call time is then batch_time / BATCH.
#
# IMPORTANT CAVEAT: optimizer comparisons are sensitive to differing line-search /
# iteration counts; if the two libraries take different numbers of gradient evaluations,
# wall-clock ratio is apples-to-oranges even at the same tolerance.  We report
# iteration counts + gradient evaluations for both sides and flag if they diverge.
#
# Julia baseline = Optim.jl (ecosystem — there is no Base optimizer; noted in gap log).
# Rust side = argmin 0.11 crate (LBFGS, MoreThuenteLineSearch, m=7).
# Both use m=7 L-BFGS history, MoreThuente linesearch, grad_tol=1e-5.
#
# Run: taskset -c 2 julia -t 1 --project=bench bench/probe_argmin.jl
# Build Rust lib first: bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using Optim
using LineSearches
using Printf

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

# Batch size: amortises per-call allocations (Optim ~11KB/call).
# 100 calls ≈ 2ms batch; σ < 5% both sides.
const BATCH = 100

# ── Rosenbrock function + gradient (same definition both sides) ───────────────
rosenbrock(p) = 100.0*(p[2] - p[1]^2)^2 + (1.0 - p[1])^2

function rosenbrock_grad!(g, p)
    g[1] = -400.0*p[1]*(p[2] - p[1]^2) - 2.0*(1.0 - p[1])
    g[2] =  200.0*(p[2] - p[1]^2)
    return g
end

const x0 = [-1.2, 1.0]
const optim_method = LBFGS(; m=7, linesearch=LineSearches.MoreThuente())

# ── Optim.jl L-BFGS batch wrapper ─────────────────────────────────────────────
@noinline function jl_lbfgs_batch!()
    s = 0.0
    for _ in 1:BATCH
        r = optimize(rosenbrock, rosenbrock_grad!, x0, optim_method,
            Optim.Options(g_tol=1e-5, iterations=10_000))
        s += Optim.minimum(r)  # DCE sink
    end
    return s
end

# ── Rust argmin ccall batch wrapper ───────────────────────────────────────────
const argmin_out = zeros(Float64, 2)

@noinline function rust_lbfgs_batch!()
    iters = ccall((:argmin_lbfgs_rosenbrock, LIB), Float64,
                  (Ptr{Float64}, Csize_t), argmin_out, BATCH)
    return iters  # DCE sink (also iteration count of last run)
end

# ── warm-up + correctness ─────────────────────────────────────────────────────
println("\n=== argmin correctness checks ===")

# Julia (single call for inspection)
r_jl = optimize(rosenbrock, rosenbrock_grad!, x0, optim_method,
    Optim.Options(g_tol=1e-5, iterations=10_000))
p_jl = Optim.minimizer(r_jl)
f_jl = Optim.minimum(r_jl)
jl_iters = r_jl.iterations
jl_fevals = r_jl.f_calls
jl_gevals = r_jl.g_calls
@printf("  Julia Optim LBFGS:  x=[%.8f, %.8f]  f=%.2e  iters=%d  fevals=%d  gevals=%d\n",
    p_jl[1], p_jl[2], f_jl, jl_iters, jl_fevals, jl_gevals)
@assert abs(p_jl[1] - 1.0) < 1e-4 && abs(p_jl[2] - 1.0) < 1e-4 "Julia did not converge to [1,1]!"
@assert Optim.converged(r_jl) "Julia Optim did not converge!"

# Rust (single call for inspection, batch=1)
rust_iters_f64 = ccall((:argmin_lbfgs_rosenbrock, LIB), Float64,
                        (Ptr{Float64}, Csize_t), argmin_out, 1)
rust_iters = round(Int, rust_iters_f64)
rust_f = 100.0*(argmin_out[2]-argmin_out[1]^2)^2 + (1.0-argmin_out[1])^2
@printf("  Rust argmin LBFGS:  x=[%.8f, %.8f]  f=%.2e  iters=%d\n",
    argmin_out[1], argmin_out[2], rust_f, rust_iters)
@assert abs(argmin_out[1] - 1.0) < 1e-4 && abs(argmin_out[2] - 1.0) < 1e-4 "Rust did not converge to [1,1]!"

# Apples-to-oranges caveat check
iter_ratio = max(jl_iters, rust_iters) / max(1, min(jl_iters, rust_iters))
if iter_ratio > 1.5
    @warn "Iteration counts diverge: Julia=$(jl_iters), Rust=$(rust_iters) (ratio=$(round(iter_ratio,digits=2))×) — wall-clock comparison is apples-to-oranges (different line-search/convergence paths)."
else
    println("  Iteration counts comparable: Julia=$(jl_iters), Rust=$(rust_iters) (ratio=$(round(iter_ratio,digits=2))×)")
end

# Alloc check: both sides allocate per call; batching amortises GC.
jl_alloc = @allocated optimize(rosenbrock, rosenbrock_grad!, x0, optim_method, Optim.Options(g_tol=1e-5))
@printf("  Julia alloc/call: %.0f bytes (GC-amortised by batching %d calls)\n", float(jl_alloc), BATCH)

# ── batch warm-up ─────────────────────────────────────────────────────────────
jl_lbfgs_batch!()
rust_lbfgs_batch!()

# ── probe ─────────────────────────────────────────────────────────────────────
println("\n=== argmin LBFGS Rosenbrock 2-D (batch=$BATCH, per-call time reported) ===")
argmin_probes = Probe[
    run_probe("Optim.jl LBFGS", jl_lbfgs_batch!; seconds=3.0),
    run_probe("rust argmin",    rust_lbfgs_batch!; seconds=3.0),
]
# Divide medians by BATCH for per-call display, but keep raw probes for the harness
# (harness stores raw batch times; note is in the crate label)
println("\n── per-batch ($(BATCH) calls) raw timings:")
report("argmin_lbfgs", argmin_probes; rust_label="rust argmin")
save_probes("argmin_lbfgs", argmin_probes)
plot_probe("argmin_lbfgs", argmin_probes)

println()
@printf("  Per-call medians (batch=%d): Julia %.1f µs  Rust %.1f µs\n",
    BATCH,
    1e6 * argmin_probes[1].median / BATCH,
    1e6 * argmin_probes[2].median / BATCH)
@printf("  Iteration count caveat: Julia iters=%d (fevals=%d gevals=%d)  Rust iters=%d\n",
    jl_iters, jl_fevals, jl_gevals, rust_iters)
if iter_ratio > 1.5
    println("  WARNING: APPLES-TO-ORANGES — iteration counts differ by $(round(iter_ratio,digits=2))× — wall-clock ratio is NOT a fair LBFGS kernel speed comparison.")
else
    println("  Iteration counts comparable (ratio $(round(iter_ratio,digits=2))×) — wall-clock comparison is meaningful.")
end

println("\nDone — probe_argmin.jl")
