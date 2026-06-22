# Probe: faer (Tier 3) — LU / Cholesky / QR / SVD factorizations.
#
# Contenders: OpenBLAS (LinearAlgebra stdlib) vs faer (Rust cdylib, single-threaded).
# If OpenBLAS lags on any factorization, MKL (using MKL) is also probed.
#
# Sizes: n = 64, 128, 256.
# faer forced single-threaded via Par::Seq inside each Rust shim + RAYON_NUM_THREADS=1.
#
# Timing design: preallocated scratch buffer per side; both pay the same copyto! cost.
#
# Run:  taskset -c 2 julia -t 1 --project=bench bench/probe_faer.jl
# Build Rust lib first:  bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
using LinearAlgebra

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

# ─────────────────────────────────────────────────────────────────────────────
# Rust ccall wrappers
# ─────────────────────────────────────────────────────────────────────────────

@noinline function rust_cholesky!(A::Matrix{Float64})
    n = size(A, 1)
    ccall((:faer_cholesky, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), n)
end
@noinline function rust_lu!(A::Matrix{Float64})
    n = size(A, 1)
    ccall((:faer_lu, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), n)
end
@noinline function rust_qr!(A::Matrix{Float64})
    n = size(A, 1)
    ccall((:faer_qr, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), n)
end
@noinline function rust_svd!(A::Matrix{Float64})
    n = size(A, 1)
    ccall((:faer_svd, LIB), Float64, (Ptr{Float64}, Csize_t), pointer(A), n)
end

# ─────────────────────────────────────────────────────────────────────────────
# Matrix factories
# ─────────────────────────────────────────────────────────────────────────────
spd_matrix(n) = let A = randn(n, n); Matrix(A'A + n * I); end

# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

function sanity_cholesky(A::Matrix{Float64})
    F = cholesky(copy(A)); @assert F.L * F.L' ≈ A rtol=1e-9 "Cholesky fail"
    r = rust_cholesky!(copy(A)); @assert !isnan(r) "faer Cholesky NaN"
end
function sanity_lu(A::Matrix{Float64})
    F = lu(copy(A)); @assert F.L * F.U ≈ A[F.p, :] rtol=1e-9 "LU fail"
    r = rust_lu!(copy(A)); @assert !isnan(r) "faer LU NaN"
end
function sanity_qr(A::Matrix{Float64})
    F = qr(copy(A)); @assert Matrix(F.Q) * F.R ≈ A rtol=1e-9 "QR fail"
    r = rust_qr!(copy(A)); @assert !isnan(r) "faer QR NaN"
end
function sanity_svd(A::Matrix{Float64})
    F = svd(copy(A)); @assert F.U * Diagonal(F.S) * F.Vt ≈ A rtol=1e-9 "SVD fail"
    r = rust_svd!(copy(A)); @assert !isnan(r) "faer SVD NaN"
end

# ─────────────────────────────────────────────────────────────────────────────
# Build a timed closure over preallocated scratch (no allocation in hot loop)
# ─────────────────────────────────────────────────────────────────────────────
function make_jl_closure(name, ref_mat)
    s = copy(ref_mat)
    # All four use the IN-PLACE LAPACK wrappers (cholesky!/lu!/qr!/svd!) over the refreshed scratch
    # `s`, matching faer factoring a freshly-copied Mat each call — apples-to-apples, minimal alloc.
    if name == "cholesky"
        return @noinline let r=ref_mat, s=s; () -> (copyto!(s, r); cholesky!(Symmetric(s, :L)); s[1]); end
    elseif name == "lu"
        return @noinline let r=ref_mat, s=s; () -> (copyto!(s, r); lu!(s); s[1]); end
    elseif name == "qr"
        return @noinline let r=ref_mat, s=s; () -> (copyto!(s, r); qr!(s); s[1]); end
    elseif name == "svd"
        return @noinline let r=ref_mat, s=s; () -> (copyto!(s, r); svd!(s); s[1]); end
    end
end
function make_rust_closure(name, ref_mat)
    s = copy(ref_mat)
    fn = getfield(Main, Symbol("rust_$(name)!"))
    return @noinline let r=ref_mat, s=s, fn=fn; () -> (copyto!(s, r); fn(s)); end
end

# ─────────────────────────────────────────────────────────────────────────────
# Probe one factorization at one size
# ─────────────────────────────────────────────────────────────────────────────
function probe_one(name, n; ref_mat, seconds=3.0, with_mkl=false)
    f_jl  = make_jl_closure(name, ref_mat)
    f_rust = make_rust_closure(name, ref_mat)

    labels = ["OpenBLAS", "rust/faer"]
    fns    = [f_jl, f_rust]

    if with_mkl
        # Re-use same closures — BLAS backend already swapped by `using MKL` at top level
        # We just re-label the Julia contender
        labels = ["MKL", "rust/faer"]
    end

    probes = Probe[run_probe(l, f; seconds=seconds) for (l, f) in zip(labels, fns)]
    crate_key = "faer_$(name)_$(n)x$(n)$(with_mkl ? "_mkl" : "")"
    report(crate_key, probes; rust_label="rust/faer")
    save_probes(crate_key, probes)
    plot_probe(crate_key, probes)
    println()
    return probes
end

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: OpenBLAS
# ─────────────────────────────────────────────────────────────────────────────
println("\n=== faer factorization probes (OpenBLAS) ===\n")

results_ob = Dict{Tuple{String,Int}, Probe}()

for n in (64, 128, 256, 512)
    A_spd = spd_matrix(n)
    A_gen = randn(n, n)

    for (name, A_ref) in (("cholesky", A_spd), ("lu", A_gen), ("qr", A_gen), ("svd", A_gen))
        println("── $(name) n=$n ──")
        if name == "cholesky"; sanity_cholesky(A_spd)
        elseif name == "lu";   sanity_lu(A_gen)
        elseif name == "qr";   sanity_qr(A_gen)
        elseif name == "svd";  sanity_svd(A_gen)
        end
        ps = probe_one(name, n; ref_mat=A_ref)
        # Record OpenBLAS result for parity check
        results_ob[(name, n)] = ps[1]  # OpenBLAS probe
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: MKL — if any OpenBLAS result is below gate vs faer, re-probe with MKL
# ─────────────────────────────────────────────────────────────────────────────

# Find which (name, n) combos are below gate
below_gate = [k for (k, p) in results_ob
    if begin
        # Find corresponding rust probe
        rust_key = "faer_$(k[1])_$(k[2])x$(k[2])"
        rust_json = joinpath(Harness.RESULTS_DIR, "$rust_key.json")
        if isfile(rust_json)
            ps2 = Harness.load_probes(rust_key)
            ri = findfirst(p -> p.label == "rust/faer", ps2)
            !isnothing(ri) && Harness.parity(p.median, ps2[ri].median) < Harness.PARITY_GATE
        else
            false
        end
    end]

if !isempty(below_gate)
    println("\n=== Retrying with MKL on below-gate cases: $below_gate ===\n")
    using MKL  # swap BLAS backend to MKL
    BLAS.set_num_threads(1)
    println("BLAS backend after 'using MKL': $(BLAS.get_config())")

    for n in (64, 128, 256, 512)
        A_spd = spd_matrix(n)
        A_gen = randn(n, n)
        for (name, A_ref) in (("cholesky", A_spd), ("lu", A_gen), ("qr", A_gen), ("svd", A_gen))
            (name, n) in below_gate || continue
            println("── $(name) n=$n (MKL) ──")
            probe_one(name, n; ref_mat=A_ref, with_mkl=true)
        end
    end
else
    println("No below-gate cases; MKL probe skipped.")
end

println("\nDone — probe_faer.jl")
