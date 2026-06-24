# Follow-up: was the itoa/ryu gap purely Julia's String alloc, or a real algorithmic gap?
# Re-probe with ZERO-ALLOC Julia formatting into a reused buffer. ryu: Base.Ryu.writeshortest (stdlib).
# itoa: table-based (2 digits/step, itoa's own trick). Run: taskset -c 2 julia -t1 --project=bench <this>
using Chairmarks, Printf
import Chairmarks: median
const LIB = joinpath(@__DIR__, "rust_compare", "rust", "target", "release", "libblazing_compare.so")
const N = 100_000
xi = rand(Int64, N); xf = randn(N)

const D2 = let b = Vector{UInt8}(undef, 200)          # "00","01",..,"99" lookup
    for i in 0:99; b[2i+1] = UInt8('0') + i ÷ 10; b[2i+2] = UInt8('0') + i % 10; end; b
end
@inbounds function itoa_len!(buf, x::Int64)           # write into end of buf, return byte length
    neg = x < 0
    u = neg ? (~reinterpret(UInt64, x) + 0x1) : reinterpret(UInt64, x)   # |x|, typemin-safe
    pos = length(buf)
    while u >= 0x64; q = u ÷ 0x64; r = u - q*0x64; buf[pos-1]=D2[2r+1]; buf[pos]=D2[2r+2]; pos-=2; u=q; end
    if u >= 0xa; buf[pos-1]=D2[2u+1]; buf[pos]=D2[2u+2]; pos-=2 else buf[pos]=UInt8('0')+u%UInt8; pos-=1 end
    neg && (buf[pos]=UInt8('-'); pos-=1)
    length(buf) - pos
end
buf = Vector{UInt8}(undef, 32)
jl_itoa_za(xs, buf) = (t=0; for x in xs; t += itoa_len!(buf, x); end; t)
jl_ryu_za(xs, buf)  = (t=0; for x in xs; t += Base.Ryu.writeshortest(buf, 1, x) - 1; end; t)
@noinline fj_itoa(xs, buf) = jl_itoa_za(xs, buf); @noinline fj_ryu(xs, buf) = jl_ryu_za(xs, buf)
rs_itoa(xs) = @ccall LIB.bp_itoa_len(xs::Ptr{Int64}, length(xs)::Csize_t)::Csize_t
rs_ryu(xs)  = @ccall LIB.bp_ryu_len(xs::Ptr{Float64}, length(xs)::Csize_t)::Csize_t
# correctness: length must match string()
@assert all(itoa_len!(buf, x) == ncodeunits(string(x)) for x in (0,1,-1,9,10,-10,99,100,-12345,typemax(Int64),typemin(Int64)))

ms(b) = median(b).time * 1e3
al(b) = median(b).allocs
row(name, jb, rb) = @printf("%-22s julia %.3f ms (%d alloc)   rust %.3f ms   parity %.2f\n",
                            name, ms(jb), al(jb), ms(rb), ms(rb)/ms(jb))
row("itoa zero-alloc", (@be fj_itoa(xi, buf) seconds=3), (@be rs_itoa(xi) seconds=3))
row("ryu  zero-alloc", (@be fj_ryu(xf, buf)  seconds=3), (@be rs_ryu(xf)  seconds=3))
