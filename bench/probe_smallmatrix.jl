# Probe: SmallMatrix (Tier 2, glam/nalgebra) — batched small-matrix ops over N pairs so we
# measure the kernel, not ccall overhead.
#
# Contenders per op:
#   1. BlazingPorts.SmallMatrix  — hand-rolled NTuple structs, literal-index ops
#   2. StaticArrays (SVector/SMatrix)  — ecosystem baseline, LLVM-friendly
#   3. glam (Rust cdylib, batched shims)  — the target crate
#
# Ops benchmarked:
#   smallmatrix_cross    — Vec3 cross over N pairs
#   smallmatrix_dot      — Vec3 dot over N pairs (scalar sink = sum)
#   smallmatrix_mat4vec4 — Mat4 * Vec4 over N pairs
#
# Run:  taskset -c 2 julia -t 1 --project=bench bench/probe_smallmatrix.jl
# Build the Rust lib first:  bash bench/rust_compare/build.sh

include(joinpath(@__DIR__, "harness.jl"))
using .Harness
import BlazingPorts.SmallMatrix as SM
using BlazingPorts.SmallMatrix: Vec3, Vec4, Mat4
using StaticArrays: SVector, SMatrix, @SMatrix, @SVector
using LinearAlgebra: cross as la_cross, dot as la_dot

Harness.single_thread!()
isfile(Harness.RUST_LIB) || error("build the Rust lib first: bash bench/rust_compare/build.sh")
const LIB = Harness.RUST_LIB

const N = 10_000  # batch size — enough to amortise ccall overhead for ns-scale ops

# ── preallocate data ──────────────────────────────────────────────────────────

# Arrays in xyz-interleaved layout [x0,y0,z0, x1,y1,z1, ...] (matches glam batched shim)
const a_xyz = rand(Float64, 3 * N)
const b_xyz = rand(Float64, 3 * N)

# Build matching arrays of Julia types (no alloc in timed region)
const bp_as  = [Vec3(a_xyz[3i-2], a_xyz[3i-1], a_xyz[3i]) for i in 1:N]
const bp_bs  = [Vec3(b_xyz[3i-2], b_xyz[3i-1], b_xyz[3i]) for i in 1:N]
const sa_as  = [SVector(a_xyz[3i-2], a_xyz[3i-1], a_xyz[3i]) for i in 1:N]
const sa_bs  = [SVector(b_xyz[3i-2], b_xyz[3i-1], b_xyz[3i]) for i in 1:N]

# Out buffers for Rust ccall
const cross_out = zeros(Float64, 3 * N)
const dot_out   = zeros(Float64, N)

# Mat4 × Vec4 setup
const mat_data = rand(Float64, 16 * N)   # N matrices, 16 f64 each (column-major)
const vec4_data = rand(Float64, 4 * N)  # N vectors, 4 f64 each
const mv_out   = zeros(Float64, 4 * N)

const bp_mats = [Mat4(
    Vec4(mat_data[16i-15], mat_data[16i-14], mat_data[16i-13], mat_data[16i-12]),
    Vec4(mat_data[16i-11], mat_data[16i-10], mat_data[16i-9],  mat_data[16i-8]),
    Vec4(mat_data[16i-7],  mat_data[16i-6],  mat_data[16i-5],  mat_data[16i-4]),
    Vec4(mat_data[16i-3],  mat_data[16i-2],  mat_data[16i-1],  mat_data[16i]),
) for i in 1:N]

const bp_vecs4 = [Vec4(vec4_data[4i-3], vec4_data[4i-2], vec4_data[4i-1], vec4_data[4i])
                  for i in 1:N]

const sa_mats = [SMatrix{4,4}(mat_data[16i-15:16i]...) for i in 1:N]
const sa_vecs4 = [SVector(vec4_data[4i-3], vec4_data[4i-2], vec4_data[4i-1], vec4_data[4i])
                  for i in 1:N]

# DCE sinks (accumulate into a scalar so compiler can't elide the loop)
const _sink = Ref(0.0)

# ─────────────────────────────────────────────────────────────────────────────
# Op 1: Vec3 cross product
# ─────────────────────────────────────────────────────────────────────────────

@noinline function bp_cross_batch!()
    s = 0.0
    @inbounds for i in eachindex(bp_as)
        r = SM.cross(bp_as[i], bp_bs[i])
        s += r.x
    end
    _sink[] = s
    return s
end

@noinline function sa_cross_batch!()
    s = 0.0
    @inbounds for i in eachindex(sa_as)
        r = la_cross(sa_as[i], sa_bs[i])
        s += r[1]
    end
    _sink[] = s
    return s
end

@noinline function rust_cross_batch!()
    ccall((:glam_vec3_cross_array, LIB), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t),
        a_xyz, b_xyz, cross_out, N)
    return cross_out[1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Op 2: Vec3 dot product
# ─────────────────────────────────────────────────────────────────────────────

@noinline function bp_dot_batch!()
    s = 0.0
    @inbounds for i in eachindex(bp_as)
        s += SM.dot(bp_as[i], bp_bs[i])
    end
    _sink[] = s
    return s
end

@noinline function sa_dot_batch!()
    s = 0.0
    @inbounds for i in eachindex(sa_as)
        s += la_dot(sa_as[i], sa_bs[i])
    end
    _sink[] = s
    return s
end

@noinline function rust_dot_batch!()
    ccall((:glam_vec3_dot_array, LIB), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t),
        a_xyz, b_xyz, dot_out, N)
    return dot_out[1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Op 3: Mat4 * Vec4
# ─────────────────────────────────────────────────────────────────────────────

@noinline function bp_mat4vec4_batch!()
    s = 0.0
    @inbounds for i in eachindex(bp_mats)
        r = bp_mats[i] * bp_vecs4[i]
        s += r.x
    end
    _sink[] = s
    return s
end

@noinline function sa_mat4vec4_batch!()
    s = 0.0
    @inbounds for i in eachindex(sa_mats)
        r = sa_mats[i] * sa_vecs4[i]
        s += r[1]
    end
    _sink[] = s
    return s
end

@noinline function rust_mat4vec4_batch!()
    ccall((:glam_mat4_mul_vec4_array, LIB), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t),
        mat_data, vec4_data, mv_out, N)
    return mv_out[1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks — all three contenders agree (rtol 1e-12)
# ─────────────────────────────────────────────────────────────────────────────

# cross
bp_cross_batch!(); sa_cross_batch!(); rust_cross_batch!()
for i in 1:N
    bp_r = SM.cross(bp_as[i], bp_bs[i])
    sa_r = la_cross(sa_as[i], sa_bs[i])
    ru_r = cross_out[3i-2:3i]
    @assert isapprox([bp_r.x, bp_r.y, bp_r.z], collect(sa_r); rtol=1e-12) "cross BP vs SA mismatch at $i"
    @assert isapprox([bp_r.x, bp_r.y, bp_r.z], ru_r; rtol=1e-12) "cross BP vs Rust mismatch at $i"
end
println("cross sanity: OK")

# dot
bp_dot_batch!(); sa_dot_batch!(); rust_dot_batch!()
for i in 1:N
    d_bp = SM.dot(bp_as[i], bp_bs[i])
    d_sa = la_dot(sa_as[i], sa_bs[i])
    d_ru = dot_out[i]
    @assert isapprox(d_bp, d_sa; rtol=1e-12) "dot BP vs SA mismatch at $i"
    @assert isapprox(d_bp, d_ru; rtol=1e-12) "dot BP vs Rust mismatch at $i"
end
println("dot sanity: OK")

# mat4*vec4
bp_mat4vec4_batch!(); sa_mat4vec4_batch!(); rust_mat4vec4_batch!()
for i in 1:N
    bp_r = bp_mats[i] * bp_vecs4[i]
    sa_r = sa_mats[i] * sa_vecs4[i]
    ru_r = mv_out[4i-3:4i]
    @assert isapprox([bp_r.x, bp_r.y, bp_r.z, bp_r.w], collect(sa_r); rtol=1e-12) "M4V4 BP vs SA mismatch at $i"
    @assert isapprox([bp_r.x, bp_r.y, bp_r.z, bp_r.w], ru_r; rtol=1e-12) "M4V4 BP vs Rust mismatch at $i"
end
println("mat4*vec4 sanity: OK\n")

# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

println("=== smallmatrix_cross (N=$N) ===")
cross_probes = Probe[
    run_probe("BlazingPorts", bp_cross_batch!; seconds=3.0),
    run_probe("StaticArrays", sa_cross_batch!; seconds=3.0),
    run_probe("rust/glam",    rust_cross_batch!; seconds=3.0),
]
report("smallmatrix_cross", cross_probes; rust_label="rust/glam")
save_probes("smallmatrix_cross", cross_probes)
out = plot_probe("smallmatrix_cross", cross_probes)
println("plot → $out\n")

println("=== smallmatrix_dot (N=$N) ===")
dot_probes = Probe[
    run_probe("BlazingPorts", bp_dot_batch!; seconds=3.0),
    run_probe("StaticArrays", sa_dot_batch!; seconds=3.0),
    run_probe("rust/glam",    rust_dot_batch!; seconds=3.0),
]
report("smallmatrix_dot", dot_probes; rust_label="rust/glam")
save_probes("smallmatrix_dot", dot_probes)
out = plot_probe("smallmatrix_dot", dot_probes)
println("plot → $out\n")

println("=== smallmatrix_mat4vec4 (N=$N) ===")
mv_probes = Probe[
    run_probe("BlazingPorts", bp_mat4vec4_batch!; seconds=3.0),
    run_probe("StaticArrays", sa_mat4vec4_batch!; seconds=3.0),
    run_probe("rust/glam",    rust_mat4vec4_batch!; seconds=3.0),
]
report("smallmatrix_mat4vec4", mv_probes; rust_label="rust/glam")
save_probes("smallmatrix_mat4vec4", mv_probes)
out = plot_probe("smallmatrix_mat4vec4", mv_probes)
println("plot → $out\n")

println("Done — probe_smallmatrix.jl")
