/* "Uncripple" Intel MKL on AMD CPUs (legitimate, for a fair single-thread comparison).
 *
 * MKL dispatches to slow generic kernels unless its CPU-vendor check returns "GenuineIntel".
 * MKL_DEBUG_CPU_TYPE=5 used to force the AVX2 path but Intel removed it after MKL 2020u1.
 * The robust method (MKL 2025): LD_PRELOAD this so MKL's vendor checks return true → AVX2/AVX512
 * kernels are selected on Zen. (mkl_serv_intel_cpu_true was the classic symbol; MKL 2025 also
 * gained mkl_serv_get_cpu_true — override both.)
 *
 *   gcc -shared -fPIC -O2 -o fakeintel.so fakeintel.c
 *   LD_PRELOAD=$PWD/fakeintel.so MKL_ENABLE_INSTRUCTIONS=AVX512 julia ...
 *
 * The fprintf markers below confirm the shim is actually loaded and that MKL calls the symbol
 * (interposition working). Remove them for a clean run.
 */
#include <stdio.h>

__attribute__((constructor)) static void _fi_load(void) {
    fprintf(stderr, "[fakeintel] preloaded\n");
}

static int _fi_a = 0, _fi_b = 0;
int mkl_serv_intel_cpu_true(void) {
    if (!_fi_a) { _fi_a = 1; fprintf(stderr, "[fakeintel] mkl_serv_intel_cpu_true() -> 1 (called)\n"); }
    return 1;
}
int mkl_serv_get_cpu_true(void) {
    if (!_fi_b) { _fi_b = 1; fprintf(stderr, "[fakeintel] mkl_serv_get_cpu_true() -> 1 (called)\n"); }
    return 1;
}
