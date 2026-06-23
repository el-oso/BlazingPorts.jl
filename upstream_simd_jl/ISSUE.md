# SIMD.jl — gaps found writing BLAS-grade factorization microkernels

While implementing pure-Julia QR/Cholesky microkernels with SIMD.jl that **match and beat** a hand-tuned
Rust LA library (faer), SIMD.jl was excellent overall — pure SIMD.jl + LLVM reached and exceeded
hand-written x86 assembly. These are the gaps/ergonomics that cost real time, roughly in priority order.
Verified against SIMD.jl v3.x. Happy to PR any of these (have minimal repros + a plan for each).

---

### 1. No count-based partial load/store for loop remainders  *(highest value)*
Masked load/store exist but take a `Vec{N,Bool}` mask (`vloadx(ptr, mask)`, `vload(Vec{N,T}, ptr, mask)`,
`vstorec`). The overwhelmingly common need — the `n < N` **remainder of a SIMD loop** — has no
count-based form, so users construct the Bool mask by hand or, worse, hand-roll:

```julia
# what we ended up writing in every kernel for the tail:
tail = Vec(ntuple(j -> j <= n ? unsafe_load(ptr, j) : zero(T), Val(N)))   # verbose + boxing hazard
```

**Proposed:** `vloadx(Vec{N,T}, ptr, n::Integer)` and `vstorex(v, ptr, n::Integer)` (+ array/index
variants) that build the mask from `n` internally and delegate to the existing masked ops.

### 2. No `prefetch`
`grep -r prefetch src/` is empty. For streaming kernels we had to drop to `Base.llvmcall` with
`@llvm.prefetch` — and the single-string `llvmcall` form *fails* (needs the module form with a `declare`).
**Proposed:** `SIMD.prefetch(ptr; rw=:read, locality=3, cache=:data)` → `@llvm.prefetch.p0`.

### 3. Reduction order is undocumented; no ordered reduction
`sum(::Vec)` / `reduce(+, ::Vec)` lower to `@llvm.vector.reduce.fadd`, whose float order is unspecified.
For bit-reproducible numerics (e.g. matching a reference implementation's reduction, or regression tests)
this is a trap — we had to hand-write a pairwise fold to get a defined order.
**Proposed:** document the order the current `sum` produces, and add `reduce(+, v; order=:ordered)`
(strict, reproducible) alongside `:tree` (current/fast).

### 4. The tuple-reassignment boxing trap *(docs)*
A multi-accumulator kernel written with `ntuple`-of-`Vec` reassigned per iteration **heap-allocates** and
runs ~100× slower (we measured a microkernel at **0.1 GFLOP/s**); only explicit named accumulators
(`@generated` / `Base.Cartesian.@nexprs`) stay in registers. This bit two separate projects (and is the
same class as a documented 135× FFT regression).
**Proposed:** a manual section + a tested 0-alloc example showing the safe register-tile pattern; optionally
a `@simd_tile` helper.

### 5. Broadcast-load idiom undocumented *(minor)*
`Vec{N,T}(unsafe_load(ptr))` + `muladd` already folds to a memory-broadcast FMA (`{1to8}` on AVX-512;
verified to match hand asm) — but it isn't documented, so users don't discover it.
**Proposed:** a doc note (and optional `broadcast_load(Vec{N,T}, ptr)` helper for intent).

---

*Provenance:* these surfaced building a QR factorization where pure SIMD.jl beat faer's hand-written
assembly gemm (73 vs 70 GFLOP/s, single-thread Zen5) once the kernel orchestration matched — i.e. the
language/codegen was never the gap, which is a nice datapoint for SIMD.jl. #1 and #2 are clean, PR-sized
additions; #3 is a small API+docs win for numerics users; #4 is a footgun worth at least a docs section.
