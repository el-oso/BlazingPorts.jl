# Probe: SmallMatrix (Tier 2, glam/nalgebra) — BlazingPorts vs StaticArrays.jl (current-Julia
# baseline) vs glam (Rust, via the cdylib). Single-threaded; Chairmarks ≥1000 samples; median + σ.
#
# Run:  taskset -c 2 julia -t 1 --project=bench bench/probe_smallmatrix.jl
# Build the Rust lib first:  bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.SmallMatrix as SM
using BlazingPorts.SmallMatrix: Vec3
using StaticArrays: SVector
using LinearAlgebra: cross as la_cross

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

# Representative workload: Vec3 cross product (the most arithmetic-dense small op).
const A = (1.0, 2.0, 3.0)
const B = (4.0, 5.0, 6.0)

# BlazingPorts contender
const bp_a = Vec3(A...); const bp_b = Vec3(B...)
@noinline bp_cross() = SM.cross(bp_a, bp_b)

# StaticArrays baseline
const sa_a = SVector(A); const sa_b = SVector(B)
@noinline sa_cross() = la_cross(sa_a, sa_b)

# glam (Rust) baseline — write into a reusable out buffer
const out3 = zeros(Float64, 3)
@noinline function rust_cross()
    ccall((:glam_vec3_cross, LIB), Cvoid,
        (Float64, Float64, Float64, Float64, Float64, Float64, Ptr{Float64}),
        A[1], A[2], A[3], B[1], B[2], B[3], out3)
    return out3[1]
end

# sanity: all three agree
@assert [bp_cross().x, bp_cross().y, bp_cross().z] ≈ collect(sa_cross())
rust_cross(); @assert out3 ≈ collect(sa_cross())

probes = Probe[]
for (label, f) in (("BlazingPorts", bp_cross), ("StaticArrays", sa_cross), ("rust", rust_cross))
    med, relσ = time_median_sigma(f)
    push!(probes, Probe(label, med, relσ))
end

report("smallmatrix_cross", probes; rust_label = "rust")
out = plot_probe("smallmatrix_cross", probes)
println("\nplot → $out")
