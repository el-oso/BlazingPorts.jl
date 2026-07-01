@testitem "smallmatrix" tags = [:smallmatrix] begin
    using BlazingPorts.SmallMatrix

    a = Vec3(1.0, 2.0, 3.0)
    b = Vec3(4.0, 5.0, 6.0)

    @test dot(a, b) ≈ 32.0
    @test cross(a, b) === Vec3(-3.0, 6.0, -3.0)
    @test cross(b, a) === Vec3(3.0, -6.0, 3.0)  # antisymmetry: cross(b,a) == -cross(a,b)
    @test (a + b) === Vec3(5.0, 7.0, 9.0)
    @test (2.0 * a) === Vec3(2.0, 4.0, 6.0)
    @test norm(a) ≈ sqrt(14.0)
    @test norm(normalize(a)) ≈ 1.0

    # Vec4 + Mat4
    v = Vec4(1.0, 0.0, 0.0, 0.0)
    I4 = Mat4(Vec4(1.0, 0, 0, 0), Vec4(0, 1.0, 0, 0), Vec4(0, 0, 1.0, 0), Vec4(0, 0, 0, 1.0))
    @test (I4 * v) === v
    @test (I4 * I4 * v) === v

    # promotion / type stability of constructors
    @test Vec3(1, 2.0, 3) isa Vec3{Float64}
end

@testitem "smallmatrix_strictmode" tags = [:smallmatrix] begin
    # Per-kernel guarantees on the hot stack math + the per-submodule audit (CLAUDE.md rule 6).
    using BlazingPorts.SmallMatrix
    using StrictMode, AllocCheck, JET

    a = Vec3(1.0, 2.0, 3.0); b = Vec3(4.0, 5.0, 6.0)
    v = Vec4(1.0, 2.0, 3.0, 4.0)
    m = Mat4(v, v, v, v)
    dot(a, b); cross(a, b); norm(a); normalize(a); m * v; m * m   # warm

    @assert_noalloc dot(a, b)
    @assert_noboxing dot(a, b)
    @assert_noalloc cross(a, b)
    @assert_noboxing cross(a, b)
    @assert_noalloc m * v
    @assert_noboxing m * v

    fs = audit(
        BlazingPorts.SmallMatrix; sweep = true,
        guarantees = (:noboxing, :noalloc, :typestable), io = devnull
    )
    @test nfailures(fs) == 0
end
