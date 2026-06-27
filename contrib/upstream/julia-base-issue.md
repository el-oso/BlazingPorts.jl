# `isvalid(String, s)` / UTF-8 validation is ~11× slower than SIMD once any non-ASCII byte appears

## Summary

`isvalid(String, bytes)` (via `Base.byte_string_classify`) has a fast SIMD-ish ASCII path, but the moment
a string contains **any** multibyte content it falls to the scalar DFA validator
(`_byte_string_classify_nonascii` → `_isvalid_utf8_dfa`), which runs ~11× slower than a SIMD UTF-8
validator. For accented / CJK / emoji-heavy text — i.e. most non-English text — validation is
byte-at-a-time.

A pure-Julia SIMD validator (the lemire/simdjson algorithm, `Base.llvmcall` + `VecElement`, no external
deps) closes this entirely — in fact it **beats** Rust's `simdutf8` crate. So this is an achievable Base
improvement, not a "Julia can't do SIMD" limitation.

## Reproduction

```julia
ascii = "a"^(1<<24)
mixed = repeat("aé一€", 1<<22)        # ~16 MiB, 1-in-4 multibyte
nb(s) = sizeof(s)

using Chairmarks  # or @time a loop
gbps(s) = nb(s) / (@be isvalid($s)).time / 1e9   # median

@show gbps(ascii)   # ~70 GB/s  (ASCII fast path)
@show gbps(mixed)   # ~1.7 GB/s (scalar DFA — the cliff)
```

On a Zen5 core (single thread, pinned clock): **ASCII ≈ 71 GB/s, mixed ≈ 1.66 GB/s** — a ~40× internal
cliff, and ~11× behind a SIMD validator (`simdutf8` ≈ 17.5 GB/s on the same mixed input).

## Root cause

`base/strings/string.jl`, `_byte_string_classify_nonascii`: the non-ASCII branch is a scalar DFA stepped
one byte at a time (`_isvalid_utf8_dfa`). The ASCII sub-chunks inside it are skipped fast, but every
multibyte region is scalar.

## Proposed fix

Replace the multibyte scalar path with a SIMD UTF-8 classifier (lemire algorithm: three `pshufb`
nibble-lookup tables + range checks over 32-byte vectors, carrying the previous block for the
continuation/range checks). It is pure Julia via `Base.llvmcall` over `NTuple{N,VecElement{UInt8}}` —
no new dependency — with the existing scalar DFA kept as the fallback for non-x86 targets and short tails.

A complete, byte-exact reference implementation (with a draft PR adapting it to Base primitives) is here:
**`BlazingPorts.Utf8.isvalid_utf8`** (https://github.com/el-oso/BlazingPorts.jl, `src/Utf8.jl`).

### Reference benchmarks (16 MiB, single-thread, vs Rust `simdutf8` 0.1 via C-ABI)

| corpus | `Base.isvalid` | reference SIMD validator | Rust `simdutf8` |
|---|---|---|---|
| ASCII | 71 GB/s | 72 GB/s (parity) | 76 GB/s |
| mixed UTF-8 | **1.66 GB/s** | **18.4 GB/s (11×)** | 17.5 GB/s |

The reference validator is **byte-exact with `isvalid`** across ~93k random + crafted cases (overlong,
surrogate, truncated, too-large, and sequences straddling 16/32-byte block boundaries).

## Scope notes / open questions for maintainers

- The SIMD path is x86 AVX2; ARM/other and AVX2-less x86 keep the current scalar DFA (runtime feature
  gating is the main design decision).
- `byte_string_classify` returns `1` (ASCII) vs `2` (valid non-ASCII); the SIMD path tracks the
  ASCII/non-ASCII distinction too, so the 0/1/2 contract is preserved.

Happy to open a PR (draft ready).
