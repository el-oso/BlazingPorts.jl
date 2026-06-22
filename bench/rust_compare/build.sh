#!/usr/bin/env bash
# Build the single-threaded Rust comparison cdylib (libblazing_compare.so).
# Single-threaded probes: RAYON_NUM_THREADS=1 at run time; no crate here spawns threads.
set -euo pipefail
cd "$(dirname "$0")/rust"
RAYON_NUM_THREADS=1 cargo build --release
echo "built: $(pwd)/target/release/libblazing_compare.so"
