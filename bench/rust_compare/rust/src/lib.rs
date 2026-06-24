// C-ABI shims over the real Rust crates so Julia can `ccall` them for in-process parity probes.
// Single-threaded only (no crate here spawns threads). Memory layout matches Julia: f64 column-major
// matrices passed as raw pointers with explicit row/col strides; Julia ComplexF64 == interleaved f64.

use std::os::raw::c_double;

// ── matrixmultiply (Tier 1): C := A * B, all f64, column-major (rs=1, cs=nrows). ────────────────
// m×k times k×n → m×n. alpha=1, beta=0 (overwrite C).
#[unsafe(no_mangle)]
pub extern "C" fn mm_dgemm(
    m: usize, k: usize, n: usize,
    a: *const c_double, b: *const c_double, c: *mut c_double,
) {
    // column-major strides: row stride 1, col stride = number of rows.
    unsafe {
        matrixmultiply::dgemm(
            m, k, n,
            1.0,
            a, 1, m as isize,   // A is m×k
            b, 1, k as isize,   // B is k×n
            0.0,
            c, 1, m as isize,   // C is m×n
        );
    }
}

// ── libm (Tier 1): erf / gamma / exp / log reference kernels. ────────────────────────────────────
#[unsafe(no_mangle)]
pub extern "C" fn bp_erf(x: c_double) -> c_double { libm::erf(x) }

#[unsafe(no_mangle)]
pub extern "C" fn bp_gamma(x: c_double) -> c_double { libm::tgamma(x) }

#[unsafe(no_mangle)]
pub extern "C" fn bp_exp(x: c_double) -> c_double { libm::exp(x) }

#[unsafe(no_mangle)]
pub extern "C" fn bp_log(x: c_double) -> c_double { libm::log(x) }

/// Vectorised erf over a slice (so the probe times a kernel, not ccall overhead).
#[unsafe(no_mangle)]
pub extern "C" fn bp_erf_array(x: *const c_double, out: *mut c_double, n: usize) {
    let xs = unsafe { std::slice::from_raw_parts(x, n) };
    let os = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n { os[i] = libm::erf(xs[i]); }
}

/// Vectorised gamma (tgamma) over a slice.
#[unsafe(no_mangle)]
pub extern "C" fn bp_gamma_array(x: *const c_double, out: *mut c_double, n: usize) {
    let xs = unsafe { std::slice::from_raw_parts(x, n) };
    let os = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n { os[i] = libm::tgamma(xs[i]); }
}

/// Vectorised exp over a slice.
#[unsafe(no_mangle)]
pub extern "C" fn bp_exp_array(x: *const c_double, out: *mut c_double, n: usize) {
    let xs = unsafe { std::slice::from_raw_parts(x, n) };
    let os = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n { os[i] = libm::exp(xs[i]); }
}

/// Vectorised log over a slice.
#[unsafe(no_mangle)]
pub extern "C" fn bp_log_array(x: *const c_double, out: *mut c_double, n: usize) {
    let xs = unsafe { std::slice::from_raw_parts(x, n) };
    let os = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n { os[i] = libm::log(xs[i]); }
}

// ── glam (Tier 2): Vec3/Vec4/Mat4 (f64 via DVec3/DVec4/DMat4). ────────────────────────────────────
use glam::{DVec3, DVec4, DMat4};

#[unsafe(no_mangle)]
pub extern "C" fn glam_vec3_dot(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    DVec3::new(ax, ay, az).dot(DVec3::new(bx, by, bz))
}

#[unsafe(no_mangle)]
pub extern "C" fn glam_vec3_cross(
    ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64, out: *mut c_double,
) {
    let r = DVec3::new(ax, ay, az).cross(DVec3::new(bx, by, bz));
    let o = unsafe { std::slice::from_raw_parts_mut(out, 3) };
    o[0] = r.x; o[1] = r.y; o[2] = r.z;
}

/// Mat4 (column-major, 16 f64) * Vec4 → out (4 f64).
#[unsafe(no_mangle)]
pub extern "C" fn glam_mat4_mul_vec4(m: *const c_double, v: *const c_double, out: *mut c_double) {
    let mc = unsafe { std::slice::from_raw_parts(m, 16) };
    let vc = unsafe { std::slice::from_raw_parts(v, 4) };
    let mat = DMat4::from_cols_array(&[
        mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7],
        mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15],
    ]);
    let r = mat * DVec4::new(vc[0], vc[1], vc[2], vc[3]);
    let o = unsafe { std::slice::from_raw_parts_mut(out, 4) };
    o[0] = r.x; o[1] = r.y; o[2] = r.z; o[3] = r.w;
}

// ── glam batched (Tier 2 probe): operate over N pairs, write into preallocated out. ─────────────
// Memory layout: a_xyz[3*N] = [x0,y0,z0, x1,y1,z1, ...] (xyz interleaved, same as Julia SoA packed).

/// Batched Vec3 cross product: a_xyz[3N], b_xyz[3N] → out_xyz[3N].
#[unsafe(no_mangle)]
pub extern "C" fn glam_vec3_cross_array(
    a: *const c_double, b: *const c_double, out: *mut c_double, n: usize,
) {
    let av = unsafe { std::slice::from_raw_parts(a, 3 * n) };
    let bv = unsafe { std::slice::from_raw_parts(b, 3 * n) };
    let ov = unsafe { std::slice::from_raw_parts_mut(out, 3 * n) };
    for i in 0..n {
        let ax = av[3*i]; let ay = av[3*i+1]; let az = av[3*i+2];
        let bx = bv[3*i]; let by = bv[3*i+1]; let bz = bv[3*i+2];
        let r = DVec3::new(ax, ay, az).cross(DVec3::new(bx, by, bz));
        ov[3*i] = r.x; ov[3*i+1] = r.y; ov[3*i+2] = r.z;
    }
}

/// Batched Vec3 dot product: a_xyz[3N], b_xyz[3N] → out[N] scalar sums.
#[unsafe(no_mangle)]
pub extern "C" fn glam_vec3_dot_array(
    a: *const c_double, b: *const c_double, out: *mut c_double, n: usize,
) {
    let av = unsafe { std::slice::from_raw_parts(a, 3 * n) };
    let bv = unsafe { std::slice::from_raw_parts(b, 3 * n) };
    let ov = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n {
        let ax = av[3*i]; let ay = av[3*i+1]; let az = av[3*i+2];
        let bx = bv[3*i]; let by = bv[3*i+1]; let bz = bv[3*i+2];
        ov[i] = DVec3::new(ax, ay, az).dot(DVec3::new(bx, by, bz));
    }
}

/// Batched Mat4 * Vec4: mats[16N] (column-major each), vecs[4N] → out[4N].
#[unsafe(no_mangle)]
pub extern "C" fn glam_mat4_mul_vec4_array(
    mats: *const c_double, vecs: *const c_double, out: *mut c_double, n: usize,
) {
    let mv = unsafe { std::slice::from_raw_parts(mats, 16 * n) };
    let vv = unsafe { std::slice::from_raw_parts(vecs, 4 * n) };
    let ov = unsafe { std::slice::from_raw_parts_mut(out, 4 * n) };
    for i in 0..n {
        let mc = &mv[16*i..16*i+16];
        let mat = DMat4::from_cols_array(&[
            mc[0],mc[1],mc[2],mc[3], mc[4],mc[5],mc[6],mc[7],
            mc[8],mc[9],mc[10],mc[11], mc[12],mc[13],mc[14],mc[15],
        ]);
        let vc = &vv[4*i..4*i+4];
        let r = mat * DVec4::new(vc[0], vc[1], vc[2], vc[3]);
        ov[4*i] = r.x; ov[4*i+1] = r.y; ov[4*i+2] = r.z; ov[4*i+3] = r.w;
    }
}

// ── faer (Tier 3): LU / Cholesky / QR / SVD factorizations — SINGLE-THREADED via Par::Seq. ──────
// Julia passes column-major f64 slices; we copy into a faer Mat, factorize, and return the
// L1-norm of the factored matrix as a scalar sink (so the compiler cannot dead-code-eliminate it).
// The copy into faer Mat is inside the timed region on the Rust side — same as on the Julia side
// where cholesky/lu/qr re-pack the input too.

use faer::{Mat, Side, Par};

/// Force sequential parallelism globally.  Called once at the start of every Rust shim.
#[inline(always)]
fn seq() { faer::set_global_parallelism(Par::Seq); }

/// Copy a column-major f64 slice into a faer Mat<f64>.
#[inline(always)]
fn slice_to_mat(data: *const c_double, n: usize) -> Mat<f64> {
    unsafe {
        let s = std::slice::from_raw_parts(data, n * n);
        Mat::from_fn(n, n, |r, c| s[r + c * n])
    }
}

/// Cholesky (LLT) factorization of a symmetric positive-definite n×n matrix.
/// Returns the [0,0] element of L as a scalar sink.
#[unsafe(no_mangle)]
pub extern "C" fn faer_cholesky(data: *const c_double, n: usize) -> c_double {
    seq();
    let a = slice_to_mat(data, n);
    match a.llt(Side::Lower) {
        Ok(llt) => *llt.L().get(0, 0),
        Err(_)  => f64::NAN,
    }
}

/// LU (partial pivot) factorization of a general n×n matrix.
/// Returns the [0,0] element of U as a scalar sink.
#[unsafe(no_mangle)]
pub extern "C" fn faer_lu(data: *const c_double, n: usize) -> c_double {
    seq();
    let a = slice_to_mat(data, n);
    let lu = a.partial_piv_lu();
    *lu.U().get(0, 0)
}

/// QR factorization (no column pivoting) of an n×n matrix.
/// Returns the [0,0] element of R as a scalar sink.
#[unsafe(no_mangle)]
pub extern "C" fn faer_qr(data: *const c_double, n: usize) -> c_double {
    seq();
    let a = slice_to_mat(data, n);
    let qr = a.qr();
    *qr.R().get(0, 0)
}

/// SVD of an n×n matrix (thin).
/// Returns the largest singular value (index 0) as a scalar sink.
#[unsafe(no_mangle)]
pub extern "C" fn faer_svd(data: *const c_double, n: usize) -> c_double {
    seq();
    let a = slice_to_mat(data, n);
    match a.thin_svd() {
        Ok(svd) => *svd.S().column_vector().get(0),
        Err(_)  => f64::NAN,
    }
}

// ── ndarray (Tier 3): fused broadcast & strided reduction. ───────────────────────────────────────
use ndarray::{ArrayView1, s};

/// Fused broadcast: D = A * B + c, over N f64 elements (flat arrays).
/// Returns D[0] as a scalar sink.  Preallocated D passed in.
#[unsafe(no_mangle)]
pub extern "C" fn ndarray_fused_broadcast(
    a: *const c_double, b: *const c_double,
    c_scalar: c_double,
    out: *mut c_double,
    n: usize,
) {
    let av = unsafe { std::slice::from_raw_parts(a, n) };
    let bv = unsafe { std::slice::from_raw_parts(b, n) };
    let ov = unsafe { std::slice::from_raw_parts_mut(out, n) };
    // Element-wise: out[i] = a[i] * b[i] + c
    for i in 0..n {
        ov[i] = av[i] * bv[i] + c_scalar;
    }
}

/// Strided sum: sum every `stride`-th element of `data[n]`. Returns the sum.
#[unsafe(no_mangle)]
pub extern "C" fn ndarray_strided_sum(
    data: *const c_double,
    n: usize,
    stride: usize,
) -> c_double {
    let v = unsafe { std::slice::from_raw_parts(data, n) };
    // Build ndarray view and slice with step
    let arr = ArrayView1::from(v);
    let sliced = arr.slice(s![..;stride as isize]);
    sliced.sum()
}

// ── rand + rand_distr (Tier 4): PRNG fill benchmarks. ────────────────────────────────────────────
// Uses SmallRng (Xoshiro256++ under the hood) — the same algorithm family as Julia's default
// Xoshiro RNG — for an apples-to-apples PRNG comparison.  The RNG state is held in a static
// Mutex so it persists across calls (we never reseed in the timed region, matching the Julia side
// which reuses a single Xoshiro object).
//
// Safety: all three functions operate on a globally-shared RNG behind a Mutex; single-threaded
// probes grab the lock once per call (negligible overhead for N=1_000_000 fills).

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;
use rand_distr::{StandardNormal, Exp1, Distribution};

// Thread-local SmallRng (Xoshiro256++) — never reseeded in the timed region.
// Thread-local avoids Mutex overhead; single-threaded probes use one thread so this is correct.
std::thread_local! {
    static RAND_RNG: std::cell::RefCell<SmallRng> =
        std::cell::RefCell::new(SmallRng::seed_from_u64(0x1234_5678_abcd_ef01));
}

/// Fill `out[0..n]` with uniform [0,1) f64 samples from SmallRng (Xoshiro256++).
#[unsafe(no_mangle)]
pub extern "C" fn rand_uniform_fill(out: *mut c_double, n: usize) {
    let slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
    RAND_RNG.with(|cell| {
        let mut rng = cell.borrow_mut();
        for x in slice.iter_mut() {
            *x = rng.random::<f64>();
        }
    });
}

/// Fill `out[0..n]` with standard-normal f64 samples from SmallRng (ziggurat via rand_distr).
#[unsafe(no_mangle)]
pub extern "C" fn rand_normal_fill(out: *mut c_double, n: usize) {
    let slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
    RAND_RNG.with(|cell| {
        let mut rng = cell.borrow_mut();
        for x in slice.iter_mut() {
            *x = StandardNormal.sample(&mut *rng);
        }
    });
}

/// Fill `out[0..n]` with Exp(1) f64 samples from SmallRng (ziggurat via rand_distr).
#[unsafe(no_mangle)]
pub extern "C" fn rand_exp_fill(out: *mut c_double, n: usize) {
    let slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
    RAND_RNG.with(|cell| {
        let mut rng = cell.borrow_mut();
        for x in slice.iter_mut() {
            *x = Exp1.sample(&mut *rng);
        }
    });
}

// ── argmin (Tier 4): L-BFGS minimisation of 2-D Rosenbrock. ─────────────────────────────────────
// Returns the number of iterations taken (so the Rust side is observable / not DCE'd).
// The start point [-1.2, 1.0] and tol_grad = 1e-5 match the Julia probe exactly.

use argmin::core::{CostFunction, Gradient, Executor, State};
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;

struct Rosenbrock2D;

impl CostFunction for Rosenbrock2D {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Vec<f64>) -> Result<f64, argmin::core::Error> {
        let x = p[0]; let y = p[1];
        Ok(100.0 * (y - x * x).powi(2) + (1.0 - x).powi(2))
    }
}

impl Gradient for Rosenbrock2D {
    type Param  = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, argmin::core::Error> {
        let x = p[0]; let y = p[1];
        Ok(vec![
            -400.0 * x * (y - x * x) - 2.0 * (1.0 - x),
            200.0 * (y - x * x),
        ])
    }
}

/// Run L-BFGS on 2-D Rosenbrock from [-1.2, 1.0], `batch` times.
/// Returns iteration count of the last run cast to c_double.
/// Writes the final [x, y] of the last run into `out[0..2]`.
/// Use batch > 1 to amortise fixed overhead and match Julia-side batching.
#[unsafe(no_mangle)]
pub extern "C" fn argmin_lbfgs_rosenbrock(out: *mut c_double, batch: usize) -> c_double {
    let mut last_iters: u64 = 0;
    let mut last_p = vec![0.0f64, 0.0f64];
    for _ in 0..batch {
        let linesearch = MoreThuenteLineSearch::new();
        let solver: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 7)
            .with_tolerance_grad(1e-5).unwrap();
        let res = Executor::new(Rosenbrock2D, solver)
            .configure(|state| state.param(vec![-1.2_f64, 1.0]).max_iters(10_000))
            .run()
            .unwrap();
        last_iters = res.state().get_iter();
        if let Some(p) = res.state().get_best_param() {
            last_p[0] = p[0]; last_p[1] = p[1];
        }
    }
    let o = unsafe { std::slice::from_raw_parts_mut(out, 2) };
    o[0] = last_p[0]; o[1] = last_p[1];
    last_iters as c_double
}

// ── memchr (Tier 1 GP): SIMD byte / substring search. Return index or -1. ──────────────────────────
#[unsafe(no_mangle)]
pub extern "C" fn bp_memchr(haystack: *const u8, len: usize, needle: u8) -> isize {
    let h = unsafe { std::slice::from_raw_parts(haystack, len) };
    match memchr::memchr(needle, h) { Some(i) => i as isize, None => -1 }
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_memmem(haystack: *const u8, hlen: usize, needle: *const u8, nlen: usize) -> isize {
    let h = unsafe { std::slice::from_raw_parts(haystack, hlen) };
    let n = unsafe { std::slice::from_raw_parts(needle, nlen) };
    match memchr::memmem::find(h, n) { Some(i) => i as isize, None => -1 }
}

// ── Tier 1/2 GP probes: int/float formatting + hashing + hashmap. Batched over N to amortize ccall. ──
use std::hash::{Hash, Hasher, BuildHasher};
#[unsafe(no_mangle)]
pub extern "C" fn bp_itoa_len(data: *const i64, n: usize) -> usize {
    let xs = unsafe { std::slice::from_raw_parts(data, n) };
    let mut buf = itoa::Buffer::new();
    // XOR-fold every output byte so the optimizer cannot DCE the digit-writing down to `len()`.
    let mut acc = 0u8;
    for &x in xs { for &b in buf.format(x).as_bytes() { acc ^= b; } }
    acc as usize
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_ryu_len(data: *const f64, n: usize) -> usize {
    let xs = unsafe { std::slice::from_raw_parts(data, n) };
    let mut buf = ryu::Buffer::new();
    let mut acc = 0u8;
    for &x in xs { for &b in buf.format(x).as_bytes() { acc ^= b; } }
    acc as usize
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_fxhash_sum(data: *const u64, n: usize) -> u64 {
    let xs = unsafe { std::slice::from_raw_parts(data, n) };
    xs.iter().fold(0u64, |a, &x| { let mut h = rustc_hash::FxHasher::default(); x.hash(&mut h); a ^ h.finish() })
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_ahash_sum(data: *const u64, n: usize) -> u64 {
    let xs = unsafe { std::slice::from_raw_parts(data, n) };
    let s = ahash::RandomState::with_seeds(1, 2, 3, 4);
    xs.iter().fold(0u64, |a, &x| { let mut h = s.build_hasher(); x.hash(&mut h); a ^ h.finish() })
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_hashbrown_roundtrip(data: *const u64, n: usize) -> u64 {
    let xs = unsafe { std::slice::from_raw_parts(data, n) };
    let mut m: hashbrown::HashMap<u64, u64> = hashbrown::HashMap::with_capacity(n);
    for (i, &x) in xs.iter().enumerate() { m.insert(x, i as u64); }
    xs.iter().fold(0u64, |a, &x| a.wrapping_add(*m.get(&x).unwrap()))
}

// Format-only itoa: black_box forces the digit-writes (canonical DCE defeat), no readback/copy.
#[unsafe(no_mangle)]
pub extern "C" fn bp_itoa_bb(data: *const i64, n: usize) -> usize {
    let xs = unsafe { std::slice::from_raw_parts(data, n) };
    let mut buf = itoa::Buffer::new();
    let mut acc = 0usize;
    for &x in xs { acc += std::hint::black_box(buf.format(x)).len(); }
    acc
}

// Format-only ryu: black_box forces the digit-writes (canonical DCE defeat), no readback.
#[unsafe(no_mangle)]
pub extern "C" fn bp_ryu_bb(data: *const f64, n: usize) -> usize {
    let xs = unsafe { std::slice::from_raw_parts(data, n) };
    let mut buf = ryu::Buffer::new();
    let mut acc = 0usize;
    for &x in xs { acc += std::hint::black_box(buf.format(x)).len(); }
    acc
}

// ── roaring (Tier 2 GP): compressed bitsets. Build from u32 arrays; return a cardinality/count sink. ──
use roaring::RoaringBitmap;
#[inline]
fn rb_build(p: *const u32, n: usize) -> RoaringBitmap {
    let s = unsafe { std::slice::from_raw_parts(p, n) };
    let mut r = RoaringBitmap::new();
    for &x in s { r.insert(x); }
    r
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_roaring_or(ap: *const u32, an: usize, bp: *const u32, bn: usize) -> u64 {
    (rb_build(ap, an) | rb_build(bp, bn)).len()
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_roaring_and(ap: *const u32, an: usize, bp: *const u32, bn: usize) -> u64 {
    (rb_build(ap, an) & rb_build(bp, bn)).len()
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_roaring_contains(ap: *const u32, an: usize, qp: *const u32, qn: usize) -> u64 {
    let r = rb_build(ap, an);
    let q = unsafe { std::slice::from_raw_parts(qp, qn) };
    q.iter().filter(|&&x| r.contains(x)).count() as u64
}

// Handle-based roaring: build ONCE (Box), then time the operation only — separates structure-build
// from set-algebra (the build-dominated all-in-one shims above were unfair: per-element insert is
// roaring's worst build path, and BitSet pays a 12.5MB alloc on sparse).
#[unsafe(no_mangle)]
pub extern "C" fn bp_roaring_build(p: *const u32, n: usize) -> *mut RoaringBitmap {
    Box::into_raw(Box::new(rb_build(p, n)))
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_roaring_free(h: *mut RoaringBitmap) { unsafe { drop(Box::from_raw(h)); } }
#[unsafe(no_mangle)]
pub extern "C" fn bp_roaring_or_h(a: *const RoaringBitmap, b: *const RoaringBitmap) -> u64 {
    unsafe { ((&*a) | (&*b)).len() }            // allocates the result bitmap (as BitSet union does)
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_roaring_and_h(a: *const RoaringBitmap, b: *const RoaringBitmap) -> u64 {
    unsafe { ((&*a) & (&*b)).len() }
}
#[unsafe(no_mangle)]
pub extern "C" fn bp_roaring_contains_h(a: *const RoaringBitmap, qp: *const u32, qn: usize) -> u64 {
    let r = unsafe { &*a };
    let q = unsafe { std::slice::from_raw_parts(qp, qn) };
    q.iter().filter(|&&x| r.contains(x)).count() as u64
}
