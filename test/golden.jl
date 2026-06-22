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

"""
    load_qr_golden(path) → Dict{Int, NamedTuple{(:A,:QR,:T,:block_size), ...}}

Parse the faer QR golden file at `path` (default: bench/rust_compare/qr_golden.txt).
Returns a dict keyed by matrix size n; each value has fields:
  - `.A`          :: Matrix{Float64}  — original input (column-major, n×n)
  - `.QR`         :: Matrix{Float64}  — in-place packed factor (R upper + Householder v below, n×n)
  - `.T`          :: Matrix{Float64}  — Q_coeff Householder block factor (block_size×n)
  - `.block_size` :: Int              — faer's recommended_block_size for this n

Tau/reflector layout (see qr_verify.rs comments for the full picture):
  - Convention: H_k = I − v_k v_kᵀ / τ_k   (divides by τ, not multiplies)
  - τ_k = Inf means H_k = I (identity reflector, skip).
  - v_k: implicit leading 1 at index k (1-based); entries v_k[k+1..n] live in QR[k+1:n, k].
  - For block_size=1 (n ≤ 16 in practice): T is 1×n, T[1,k+1] = τ_k (1-based column k+1).
  - For block_size>1: T is block-upper-triangular; diagonal T[k%bs+1, k+1] = τ_k per column.
    (Each bs-column block stores a bs×bs upper-triangular block Householder factor on its diagonal.)
"""
function load_qr_golden(
    path::AbstractString = joinpath(
        @__DIR__, "..", "bench", "rust_compare", "qr_golden.txt"
    )
)
    result = Dict{Int, @NamedTuple{A::Matrix{Float64}, QR::Matrix{Float64}, T::Matrix{Float64}, block_size::Int}}()
    pending = Dict{Int, Dict{String, Any}}()

    for line in eachline(path)
        isempty(strip(line)) && continue
        parts = split(line)
        tag = parts[1]   # "A", "QR", or "T"
        n   = parse(Int, parts[2])

        d = get!(pending, n, Dict{String, Any}())

        if tag == "A" || tag == "QR"
            nhex = n * n
            @assert length(parts) == nhex + 2 "line mismatch for $tag $n: expected $(nhex+2) tokens, got $(length(parts))"
            vals = [reinterpret(Float64, parse(UInt64, h; base=16)) for h in parts[3:end]]
            mat = Matrix{Float64}(undef, n, n)
            for col in 1:n, row in 1:n
                mat[row, col] = vals[(col - 1) * n + row]
            end
            d[tag] = mat

        elseif tag == "T"
            bs   = parse(Int, parts[3])   # block_size
            nhex = bs * n                  # block_size × n entries
            @assert length(parts) == nhex + 3 "line mismatch for T $n: expected $(nhex+3) tokens, got $(length(parts))"
            vals = [reinterpret(Float64, parse(UInt64, h; base=16)) for h in parts[4:end]]
            # Q_coeff is stored column-major: block_size rows × n cols
            mat = Matrix{Float64}(undef, bs, n)
            for col in 1:n, row in 1:bs
                mat[row, col] = vals[(col - 1) * bs + row]
            end
            d["T"]          = mat
            d["block_size"] = bs
        end

        if haskey(d, "A") && haskey(d, "QR") && haskey(d, "T") && haskey(d, "block_size")
            result[n] = (A=d["A"], QR=d["QR"], T=d["T"], block_size=d["block_size"])
            delete!(pending, n)
        end
    end

    isempty(pending) || @warn "Incomplete QR records for sizes: $(collect(keys(pending)))"
    return result
end
