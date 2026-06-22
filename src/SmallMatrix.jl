"""
    SmallMatrix

Fixed-size stack-allocated vector/matrix math — the Julia analogue of Rust's `glam`
(`Vec3`/`Vec4`/`Mat4`) and `nalgebra` static dimensions. Baseline to beat in benchmarks is
StaticArrays.jl; here we implement directly on immutable `NTuple`-backed structs with
straight-line (literal-index) code so every op is type-stable, non-boxing and stack-allocated —
the StrictMode `@assert_noboxing` / `@assert_noalloc` / `@assert_typestable` probe.

NEVER index a tuple field with a runtime variable in a hot path (boxes/allocates — see PureFFT
CLAUDE.md): unroll with literal indices or `@generated`.
"""
module SmallMatrix

export Vec3, Vec4, Mat4, dot, cross, norm, normalize

struct Vec3{T<:Real}
    x::T
    y::T
    z::T
end
Vec3(x, y, z) = Vec3(promote(x, y, z)...)

struct Vec4{T<:Real}
    x::T
    y::T
    z::T
    w::T
end
Vec4(x, y, z, w) = Vec4(promote(x, y, z, w)...)

# Column-major 4x4, stored as four Vec4 columns (matches glam::Mat4 layout).
struct Mat4{T<:Real}
    c1::Vec4{T}
    c2::Vec4{T}
    c3::Vec4{T}
    c4::Vec4{T}
end

Base.:+(a::Vec3, b::Vec3) = Vec3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:-(a::Vec3, b::Vec3) = Vec3(a.x - b.x, a.y - b.y, a.z - b.z)
Base.:*(s::Real, a::Vec3) = Vec3(s * a.x, s * a.y, s * a.z)
Base.:*(a::Vec3, s::Real) = s * a

Base.:+(a::Vec4, b::Vec4) = Vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)
Base.:-(a::Vec4, b::Vec4) = Vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w)
Base.:*(s::Real, a::Vec4) = Vec4(s * a.x, s * a.y, s * a.z, s * a.w)
Base.:*(a::Vec4, s::Real) = s * a

dot(a::Vec3, b::Vec3) = muladd(a.x, b.x, muladd(a.y, b.y, a.z * b.z))
dot(a::Vec4, b::Vec4) = muladd(a.x, b.x, muladd(a.y, b.y, muladd(a.z, b.z, a.w * b.w)))

cross(a::Vec3, b::Vec3) = Vec3(
    muladd(a.y, b.z, -a.z * b.y),
    muladd(a.z, b.x, -a.x * b.z),
    muladd(a.x, b.y, -a.y * b.x),
)

norm(a::Vec3) = sqrt(dot(a, a))
norm(a::Vec4) = sqrt(dot(a, a))
normalize(a::Vec3) = (inv(norm(a))) * a
normalize(a::Vec4) = (inv(norm(a))) * a

# Mat4 * Vec4 (column-major: linear combination of columns).
Base.:*(m::Mat4, v::Vec4) = (v.x * m.c1) + (v.y * m.c2) + (v.z * m.c3) + (v.w * m.c4)

# Mat4 * Mat4 (each result column is m * the corresponding column of n).
Base.:*(m::Mat4, n::Mat4) = Mat4(m * n.c1, m * n.c2, m * n.c3, m * n.c4)

end # module SmallMatrix
