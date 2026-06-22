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
