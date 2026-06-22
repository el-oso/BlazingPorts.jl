"""
    BlazingPorts

Reimplementations of fast Rust crates in Julia, gated by StrictMode and benchmarked against the
real crate (Rust via a vendored cdylib, `ccall`) — see `blazingly-fast-rust-crates.md`.

The package is **probe-first**: for each crate we first benchmark current Julia (Base / stdlib /
ecosystem) against the Rust crate single-threaded. We only ship a Julia implementation here when
that probe falls below the 0.96 parity gate, or as an explicit StrictMode-coverage kernel.

Following the PureFFT.jl precedent, this package's source carries **no StrictMode dependency** —
hot paths use Base `@generated`/SIMD, and the StrictMode `audit` / `@assert_*` guarantees are
applied externally in `bench/` and `test/` (StrictMode is a test/bench-only dep).

Submodules (one per crate family):
- [`MatrixMultiply`](@ref) — optional pure-Julia gemm microkernel (matrixmultiply probe).
- [`SpecialFns`](@ref)     — optional erf/gamma kernels (libm/statrs probe).
- [`SmallMatrix`](@ref)    — Vec3/Vec4/Mat4 stack math (glam/nalgebra probe).
"""
module BlazingPorts

include("SmallMatrix.jl")
include("SpecialFns.jl")
include("MatrixMultiply.jl")

using .SmallMatrix
using .SpecialFns
using .MatrixMultiply

end # module BlazingPorts
