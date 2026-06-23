# SIMD.jl improvement prompts — one per gap (from the BlazingPorts QR/Cholesky campaign, 2026-06-23)

Each prompt is **self-contained for a fresh agent** (assume no prior context). Target repo:
`https://github.com/eschnett/SIMD.jl`. API verified against the installed v3.x
(`~/.julia/packages/SIMD/UiGbs`). Common workflow for every prompt:

1. Fork + clone SIMD.jl; `julia --project -e 'using Pkg; Pkg.test()'` to get a green baseline.
2. Create the branch named in the prompt; implement minimally, mirroring existing code style.
3. Add tests under `test/`; re-run `Pkg.test()` until green.
4. Update `README`/docs/`NEWS` if the repo has them. Open a PR with the given title + a short rationale
   (link this issue/repro). Keep the diff small and focused.

---

## Prompt 1 — count-based partial load/store (highest value)

> Add count-based partial memory ops to SIMD.jl. It already has **mask-based** partial load/store that
> take a `Vec{N,Bool}` (`vloadx(ptr, mask)`, `vload(Vec{N,T}, ptr, mask, …)`, and masked store
> `vstorec` — see `src/arrayops.jl`). The most common SIMD-kernel need — handling the `n < N` **remainder
> of a loop** ("load the first `n` lanes, zero-fill the rest") — forces the user to construct the Bool
> mask by hand. Add integer-count convenience methods:
> - `vloadx(::Type{Vec{N,T}}, ptr::Ptr{T}, n::Integer)` — load lanes `1:n`, zero elsewhere.
> - `vstorex(v::Vec{N,T}, ptr::Ptr{T}, n::Integer)` — store lanes `1:n` only.
> - the `FastContiguousArray{T,1}` + index variants, mirroring the existing masked ones.
>
> Implement by building the mask internally from `n` (e.g. a lane-index `Vec{N,Int}` `<= n`, the idiom
> SIMD.jl already uses), then delegate to the existing masked `vload`/`vstore`. Do **not** write a scalar
> loop. Add to `src/arrayops.jl`; export the new names from `src/SIMD.jl`.
>
> Tests (`test/`): for `N=8, T=Float64`, every `n in 0:8`: load from a length-8 array and assert lanes
> `1:n` equal the data and `n+1:N` are zero; store into a zeroed buffer and assert only `1:n` were written.
> Wrap in a `@noinline` function and `@test @allocated(f(...)) == 0`. Confirm via `@code_llvm` it lowers to
> a masked move (no scalar loop, no boxing).
>
> Rationale: today kernel authors hand-roll `Vec(ntuple(j -> j<=n ? unsafe_load(p,j) : zero(T), Val(N)))`
> for every remainder — verbose and a boxing hazard. Branch `partial-loadstore-by-count`; PR title
> "Add count-based partial vloadx/vstorex for loop remainders".

---

## Prompt 2 — `prefetch` intrinsic

> Add a software-prefetch intrinsic to SIMD.jl — there is none today (`grep -r prefetch src/` is empty).
> Add `SIMD.prefetch(ptr::Ptr{T}; rw::Symbol=:read, locality::Integer=3, cache::Symbol=:data) where {T}`
> mapping to LLVM `@llvm.prefetch.p0(ptr, i32 rw, i32 locality, i32 cachetype)` — `rw`: `:read`=0,
> `:write`=1; `locality` 0..3; `cache`: `:instruction`=0, `:data`=1. Put it in `src/LLVM_intrinsics.jl`
> (the `Intrinsics` module) and export `prefetch` from `src/SIMD.jl`.
>
> Use the **module form** of `llvmcall` (a `(ir, entry)` tuple) — the single-string form fails for
> intrinsics that need a `declare`. Working reference (parameterize the three `i32`s by rw/locality/cache,
> ideally as IR constants per-method via `@generated` or `Val`):
> ```julia
> Base.llvmcall(("""declare void @llvm.prefetch.p0(ptr, i32, i32, i32)
>   define void @pf(i64 %p) #0 { %q = inttoptr i64 %p to ptr
>   call void @llvm.prefetch.p0(ptr %q, i32 0, i32 3, i32 1) ret void }
>   attributes #0 = { alwaysinline }""", "pf"), Cvoid, Tuple{Int}, reinterpret(Int, ptr))
> ```
>
> Tests: `prefetch(pointer(zeros(64)))` runs without error for `rw in (:read,:write)` × `locality in 0:3`;
> `@code_llvm` shows the `@llvm.prefetch` call. (Perf isn't unit-testable; correctness + codegen suffice.)
> Branch `prefetch-intrinsic`; PR title "Add SIMD.prefetch (@llvm.prefetch)".

---

## Prompt 3 — ordered / documented float reductions

> Make `Vec` float reductions reproducible. `Base.sum(::Vec)` / `reduce(+, ::Vec)` lower to
> `Intrinsics.reduce_add` (`src/simdvec.jl`, ~L479; LLVM `@llvm.vector.reduce.fadd`), whose order for
> floats is unspecified (fast/reassoc → tree). For bit-reproducible numerics this is a trap.
> (1) **Document** in the `sum`/`reduce` docstrings for `Vec` exactly which reduction the current
> implementation produces. (2) Add an **ordered** reduction — e.g. `reduce(+, v; order=:ordered)`
> (strict left-to-right, bit-reproducible) vs `:tree` (current/fast) — via `@llvm.vector.reduce.fadd`
> with a start value and **without** the `reassoc` flag for the ordered case.
>
> Tests: build a `Vec{8,Float64}` whose summation order changes the rounding; assert
> `reduce(+, v; order=:ordered)` is bit-identical (`reinterpret(UInt64, …)`) to a hand-written sequential
> `acc=v[1]; for i in 2:8; acc+=v[i]; end`; assert `:tree` matches the current `sum`.
> Rationale: matching a reference's reduction (porting/regression tests) needs a defined order; users
> currently hand-roll a fold. Branch `ordered-reductions`; PR title "Document + add ordered float reductions for Vec".

---

## Prompt 4 — docs + tested example: multi-accumulator kernels (the boxing footgun)

> Add documentation and a regression-tested example showing how to write fast multi-accumulator
> (register-tile) microkernels in SIMD.jl, and warning about the tuple-reassignment boxing trap.
> **Repro of the trap:** an accumulator written as
> `acc = ntuple(c -> ntuple(r -> muladd(v[r], b[c], acc[c][r]), Val(R)), Val(C))` **reassigned each loop
> iteration heap-allocates** (boxes the nested tuples) and runs ~100× slower; the reliable path is
> explicit named accumulators via `@generated` straight-line code or `Base.Cartesian.@nexprs`.
>
> Deliver: a manual section "Writing fast multi-accumulator (register-tile) kernels" containing (a) the
> failing pattern, (b) why it boxes, (c) a **working** small gemm-like microkernel (C×R `Vec` accumulators
> via `@nexprs`/`@generated`) that is allocation-free. Add the working kernel to `test/` with
> `@test @allocated(kernel(...)) == 0`. Optionally add a `@simd_tile` helper, but docs+example is the
> minimum.
> Rationale: this footgun cost two separate real projects (a QR microkernel at 0.1 GFLOP/s; a 135×
> regression in an FFT) — it's the top SIMD.jl usability hazard. Branch `docs-register-tile`; PR title
> "Docs + example: fast multi-accumulator kernels (avoid tuple boxing)".

---

## Prompt 5 — broadcast-load idiom (minor, mostly docs)

> Document (and optionally wrap) the memory-broadcast idiom in SIMD.jl. `Vec{N,T}(unsafe_load(ptr))`
> followed by `muladd` already folds to a single memory-broadcast FMA (`vfmadd231pd …{1to8}` on AVX-512 —
> verified to match hand-written assembly), so no new lowering is needed. (1) Add a manual note (load
> section): "to broadcast a scalar from memory to all lanes use `Vec{N,T}(unsafe_load(ptr))`; combined
> with `muladd` it lowers to a single broadcast-FMA." (2) Optionally add
> `broadcast_load(::Type{Vec{N,T}}, ptr) = Vec{N,T}(unsafe_load(ptr))` (+ array/index variant) for intent.
> Tests: `@code_native` on `muladd(Vec{8,Float64}(unsafe_load(p)), x, acc)` shows a `{1to8}` operand on an
> AVX-512 host (guard the test to AVX-512). Branch `broadcast-load-docs`; PR title "Document broadcast-load idiom (+ optional helper)".
