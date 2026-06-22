// Golden-value generator for the faithful Julia QR port (Layer D).
//
// For each square n×n matrix, this program:
//  1. Builds a deterministic general (non-symmetric) f64 matrix A via StdRng::seed_from_u64(n as u64),
//     entries in [-1, 1] (column-major).
//  2. Calls faer's low-level `qr_in_place` (unpivoted Householder QR) directly.
//     After the call, the matrix A is modified IN PLACE:
//       - Upper triangle (including diagonal): R
//       - Strictly lower triangle: Householder essential vectors (v_k[k+1:n-1]), column-major.
//         The implicit leading 1 (v_k[k] = 1) is NOT stored in the matrix.
//     The Q_coeff matrix (block_size × n) is the block Householder factor T, NOT individual taus.
//     However, for the small matrices in the unblocked path (n*n < 32*32 = 1024), block_size=1
//     and Q_coeff is a 1×n row where Q_coeff[0,k] = tau_k (the individual Householder scalar).
//     For larger n, Q_coeff is a block_size×n upper-triangular block factor T — each block of
//     `block_size` columns holds a block_size×block_size upper-triangular T matrix on the diagonal.
//     See faer::linalg::householder docs: H_0 H_1 ... H_{b-1} = I - V T^{-1} V^H.
//
// IMPORTANT tau convention: faer uses H = I - v vᵀ / τ  (divides by τ, not multiplies).
//   - τ is REAL and positive.
//   - v_k = [1; A[k+1:, k]] after factoring (the leading 1 is implicit).
//   - To reconstruct Q: Q = H_0 H_1 ... H_{n-1}, apply via I - V T^{-1} Vᵀ per block.
//   - For the simple column-by-column reconstruction: H_k x = x - (v_kᵀ x / τ_k) v_k.
//   - τ = infinity() means the reflector is the identity (trivial column, no reflection applied).
//
// Prints three lines per size in the same format as cholesky_verify.rs:
//   A   <n> <hex bits col-major n*n>      (input, before factoring)
//   QR  <n> <hex bits col-major n*n>      (in-place packed: R upper + v below, no implicit 1s)
//   T   <n> <block_size> <hex bits col-major block_size*n>  (Q_coeff Householder factor)
//
// For sizes where block_size=1, T is a 1×n row = individual tau values (τ_0, ..., τ_{n-1}).
// For sizes with block_size>1, T is a block_size×n upper-block-triangular matrix.
//
// Run:
//   cd bench/rust_compare/rust
//   cargo build --release --bin qr_verify
//   ./target/release/qr_verify > ../qr_golden.txt
//
// Output: 3 lines per size × 13 sizes = 39 lines.

use faer::Mat;
use faer::linalg::qr::no_pivoting::factor::{
    qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

fn main() {
    let sizes: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512];

    for &n in sizes {
        // Build deterministic general matrix with entries in [-1, 1], column-major.
        // seed = n as u64 for reproducibility (matches Cholesky convention).
        let mut rng = StdRng::seed_from_u64(n as u64);
        let mut a_data = vec![0.0f64; n * n];
        for v in a_data.iter_mut() {
            *v = rng.random::<f64>() * 2.0 - 1.0;
        }

        // Build faer Mat from a_data (column-major).
        let mut qr_mat = Mat::from_fn(n, n, |r, c| a_data[r + c * n]);

        // Compute block_size and allocate Q_coeff (block_size × n).
        let block_size = recommended_block_size::<f64>(n, n);
        let size = n; // square matrix: min(m, n) = n
        let mut q_coeff = Mat::<f64>::zeros(block_size, size);

        // Allocate scratch memory.
        let scratch_req =
            qr_in_place_scratch::<f64>(n, n, block_size, Par::Seq, Default::default());
        let mut mem = MemBuffer::new(scratch_req);
        let stack = MemStack::new(&mut mem);

        // Factor in place: modifies qr_mat and fills q_coeff.
        qr_in_place(
            qr_mat.as_mut(),
            q_coeff.as_mut(),
            Par::Seq,
            stack,
            Default::default(),
        );

        // ── Print A line (input, before factoring) ──
        print!("A {}", n);
        for col in 0..n {
            for row in 0..n {
                print!(" {:016x}", a_data[row + col * n].to_bits());
            }
        }
        println!();

        // ── Print QR line: in-place packed matrix (R upper, v below, column-major) ──
        print!("QR {}", n);
        for col in 0..n {
            for row in 0..n {
                let val: f64 = *qr_mat.get(row, col);
                print!(" {:016x}", val.to_bits());
            }
        }
        println!();

        // ── Print T line: Q_coeff Householder block factor (block_size × n, column-major) ──
        // Layout: Q_coeff is block_size×n. For small n (unblocked path), block_size=1 and
        // Q_coeff[0, k] = tau_k for column k. For larger n, each block_size-wide column slab
        // stores a block_size×block_size upper-triangular T (the block Householder factor).
        // Diagonal of each T block = individual taus for that block.
        print!("T {} {}", n, block_size);
        for col in 0..size {
            for row in 0..block_size {
                let val: f64 = *q_coeff.get(row, col);
                print!(" {:016x}", val.to_bits());
            }
        }
        println!();
    }
}
