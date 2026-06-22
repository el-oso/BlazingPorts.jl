# golden.jl — reusable parser for bench/rust_compare/cholesky_golden.txt
#
# Call: load_cholesky_golden() → Dict{Int, @NamedTuple{A::Matrix{Float64}, L::Matrix{Float64}}}
#
# The golden file has one record per line:
#   A <n> <hex-bits col-major n*n entries>
#   L <n> <hex-bits col-major n*n entries>
#
# Hex bits are read back with reinterpret(Float64, parse(UInt64, h; base=16)).
# Upper triangle of L is stored as 0.0 bits (0000000000000000) in the file.

"""
    load_cholesky_golden(path) → Dict{Int, NamedTuple{(:A,:L), Tuple{Matrix{Float64},Matrix{Float64}}}}

Parse the faer Cholesky golden file at `path` (default: the one in bench/rust_compare/).
Returns a dict keyed by matrix size n; each value has fields `.A` and `.L` (column-major Float64).
"""
function load_cholesky_golden(
    path::AbstractString = joinpath(
        @__DIR__, "..", "bench", "rust_compare", "cholesky_golden.txt"
    )
)
    result = Dict{Int, @NamedTuple{A::Matrix{Float64}, L::Matrix{Float64}}}()
    pending = Dict{Int, Dict{String, Matrix{Float64}}}()

    for line in eachline(path)
        isempty(strip(line)) && continue
        parts = split(line)
        tag   = parts[1]           # "A" or "L"
        n     = parse(Int, parts[2])
        nhex  = n * n
        @assert length(parts) == nhex + 2 "line mismatch: expected $(nhex+2) tokens, got $(length(parts))"

        vals = [reinterpret(Float64, parse(UInt64, h; base=16)) for h in parts[3:end]]
        # column-major reshape: Matrix{Float64}(undef, n, n) filled col by col
        mat = Matrix{Float64}(undef, n, n)
        for col in 1:n, row in 1:n
            mat[row, col] = vals[(col - 1) * n + row]
        end

        d = get!(pending, n, Dict{String, Matrix{Float64}}())
        d[tag] = mat

        if haskey(d, "A") && haskey(d, "L")
            result[n] = (A = d["A"], L = d["L"])
            delete!(pending, n)
        end
    end

    isempty(pending) || @warn "Incomplete records for sizes: $(keys(pending))"
    return result
end
