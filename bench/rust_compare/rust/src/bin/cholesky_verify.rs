// Golden-value generator for the faithful Julia Cholesky port.
//
// For each matrix size n, this program:
//  1. Builds a deterministic SPD f64 matrix A via a seeded StdRng:
//       fill n×n matrix M with uniform [-1,1] entries, then A = M^T * M + n*I.
//  2. Factors A with faer's LLT (lower Cholesky).
//  3. Prints A and the lower-triangular factor L as raw f64 bits (column-major), one line each:
//       A <n> <hex-bits of A[0,0]> <A[1,0]> ... (n*n entries, column-major)
//       L <n> <hex-bits of L[0,0]> <L[1,0]> ... (n*n entries, column-major; upper triangle = 0.0)
//
// The Julia side reads A's exact bits (NOT recomputing it) so input is bit-identical across languages,
// then factors that exact A and compares its L bits to the golden L here.
//
// Run:
//   cd bench/rust_compare/rust
//   cargo build --release --bin cholesky_verify
//   ./target/release/cholesky_verify > ../cholesky_golden.txt
//
// Output: 26 lines (A+L for 13 sizes).

use faer::{Mat, Side};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

fn main() {
    let sizes: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512];

    for &n in sizes {
        // Build deterministic SPD matrix: A = M^T * M + n*I
        // seed = n as u64 for reproducibility
        let mut rng = StdRng::seed_from_u64(n as u64);

        // Fill n×n matrix M with uniform [-1, 1] entries (column-major storage)
        let mut m_data = vec![0.0f64; n * n];
        for v in m_data.iter_mut() {
            *v = rng.random::<f64>() * 2.0 - 1.0;
        }

        // Compute A = M^T * M + n*I  (column-major)
        // A[i,j] = sum_k M[k,i] * M[k,j]   (M is stored column-major: M[row,col] = m_data[row + col*n])
        let mut a_data = vec![0.0f64; n * n];
        for col in 0..n {
            for row in 0..n {
                let mut s = 0.0f64;
                for k in 0..n {
                    // M[k, col] = m_data[k + col*n], M[k, row] = m_data[k + row*n]
                    s += m_data[k + col * n] * m_data[k + row * n];
                }
                a_data[row + col * n] = s;
            }
            // Add n*I diagonal
            a_data[col + col * n] += n as f64;
        }

        // Build faer Mat from a_data (column-major)
        let a_mat = Mat::from_fn(n, n, |r, c| a_data[r + c * n]);

        // Cholesky factorization
        let llt = a_mat.llt(Side::Lower)
            .expect("Matrix must be SPD; construction guarantees this");
        let l_mat = llt.L();

        // Print A line: column-major, hex bits
        print!("A {}", n);
        for col in 0..n {
            for row in 0..n {
                print!(" {:016x}", a_data[row + col * n].to_bits());
            }
        }
        println!();

        // Print L line: column-major, upper triangle forced to 0.0 bits
        print!("L {}", n);
        for col in 0..n {
            for row in 0..n {
                let val: f64 = if row < col {
                    0.0  // upper triangle
                } else {
                    *l_mat.get(row, col)
                };
                print!(" {:016x}", val.to_bits());
            }
        }
        println!();
    }
}
