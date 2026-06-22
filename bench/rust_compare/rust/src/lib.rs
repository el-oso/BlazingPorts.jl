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
