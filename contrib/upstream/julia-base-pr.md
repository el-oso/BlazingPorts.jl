# DRAFT PR: SIMD UTF-8 validation for the multibyte path of `byte_string_classify`

> Status: **draft for review before submitting to JuliaLang/julia.** Pairs with the issue in
> `julia-base-issue.md`. Not yet submitted.

## What

Speeds up `isvalid(String, _)` on non-ASCII input by ~11× by validating multibyte UTF-8 SIMD-wide
(lemire/simdjson algorithm) instead of stepping a scalar DFA. The existing ASCII fast path and the scalar
DFA (now the fallback) are unchanged. No new dependency — `Base.llvmcall` over `NTuple{32,VecElement{UInt8}}`.

## Where

`base/strings/string.jl`: `_byte_string_classify_nonascii` gains a SIMD classifier for the multibyte
region. When the SIMD path is unavailable (non-x86, no AVX2, or the tail) it calls the current scalar DFA.

## Why it's safe

- **Byte-exact** with the current `isvalid` across ~93k random + crafted cases (overlong-2/3/4, surrogate,
  truncated lead, lone continuation, too-large, and sequences straddling 16/32-byte block boundaries).
- Preserves the `0/1/2` classify contract (invalid / valid-ASCII / valid-non-ASCII) — the validator's
  ASCII fast path already distinguishes ASCII from non-ASCII.
- SIMD path is purely additive; remove it and behaviour is identical via the scalar fallback.

## Benchmarks (16 MiB, single thread, Zen5)

| corpus | before (`Base.isvalid`) | after (SIMD) | Rust `simdutf8` |
|---|---|---|---|
| ASCII | 71 GB/s | 72 GB/s | 76 GB/s |
| mixed UTF-8 | 1.66 GB/s | **18.4 GB/s** | 17.5 GB/s |

The after-path also slightly **beats** the Rust `simdutf8` crate on multibyte (1.05×).

## Algorithm (one paragraph)

Process 32-byte blocks. Classify each byte against its predecessor bytes (`prev1/prev2/prev3`, obtained by
shifting the previous block in) with three `pshufb` 16-byte nibble-lookup tables (`byte_1_high` on
`prev1>>4`, `byte_1_low` on `prev1&0xF`, `byte_2_high` on `input>>4`) AND'd together — this catches
overlong/surrogate/too-large/wrong-length errors as surviving bits. A second check (`saturating_sub` +
`signed_gt(0)`) enforces that 3rd/4th continuation bytes are present. Accumulate all error bits across the
buffer; valid ⇔ no bit ever set. A trailing `is_incomplete` vector catches sequences cut off at the end.
ASCII blocks (movemask == 0) skip all of it. Tables/flags are exact from simdjson.

## Reference implementation

Working, tested, documented: **`BlazingPorts.Utf8`** (`src/Utf8.jl`,
https://github.com/el-oso/BlazingPorts.jl). It uses `SIMD.jl` `Vec`; the Base port replaces those with
`Base.llvmcall` + `VecElement` (Base already has `VecElement`; the `pshufb`/`usub.sat`/`pmovmskb`/
`prefetch` intrinsics are `Base.llvmcall` one-liners — see `src/Utf8.jl` for the exact IR strings).

## Open design questions (for maintainers, before this is non-draft)

1. **Runtime feature gating.** The SIMD path is x86 AVX2. How should Base select it — `Sys.CPU_NAME` /
   a `llvmcall` cpuid probe / the sysimg target? (The scalar fallback must remain for generic builds.)
2. **SSSE3/NEON variants.** N=16 SSSE3 and an ARM NEON port are straightforward follow-ups if wanted; this
   PR proposes AVX2-only + scalar fallback to keep the diff small.
3. **`AbstractVector{UInt8}` vs `String`.** The classifier needs a pointer + length; non-dense
   `AbstractVector` inputs fall back to scalar (as today).

## Test plan

- Extend `test/strings/` UTF-8 validity tests with the byte-exact corpus (random valid 1–4-byte strings;
  random raw bytes; crafted overlong/surrogate/truncated/too-large; block-boundary straddles) asserting
  `isvalid_new == isvalid_old`.
- Bench note in the PR body (numbers above), reproducible via the issue's snippet.

---

## How to submit this later

1. Fork `JuliaLang/julia`; branch `simd-utf8-validate`.
2. Port `BlazingPorts.Utf8` into `base/strings/string.jl` (swap `SIMD.Vec` → `Base.llvmcall`+`VecElement`;
   reuse the exact intrinsic IR strings from `src/Utf8.jl`). Gate behind the feature check (Q1) with the
   scalar DFA as fallback inside `_byte_string_classify_nonascii`.
3. Open the issue (`julia-base-issue.md`) first; then the PR referencing it, with the benchmark table and
   the byte-exact test.
