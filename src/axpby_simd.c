#include "axpby.h"
#include <immintrin.h>

void axpby_simd(int n, float alpha_val, const float *x, float beta_val, float *y) {
    int i = 0;
    __m256 v_alpha = _mm256_set1_ps(alpha_val);
    __m256 v_beta = _mm256_set1_ps(beta_val);

    // Fully aligned vectorized loop using 32-byte AVX2 instructions
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]); // Requires 32-byte alignment
        __m256 vy = _mm256_load_ps(&y[i]);
        
        __m256 result = _mm256_fmadd_ps(v_alpha, vx, _mm256_mul_ps(v_beta, vy));
        
        // Non-temporal streaming store bypasses cache hierarchy for raw memory throughput
        _mm256_stream_ps(&y[i], result);
    }

    // Scalar cleanup loop for remaining elements
    for (; i < n; i++) {
        y[i] = alpha_val * x[i] + beta_val * y[i];
    }
}
