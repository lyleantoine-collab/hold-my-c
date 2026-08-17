#include "axpby.h"
#include <immintrin.h>

// AVX2 vectorized implementation of AXPBY: y = alpha * x + beta * y
// Processes 8 floats simultaneously using 256-bit registers and FMA.
void axpby_simd(int n, float alpha_val, const float *x, float beta_val, float *y) {
    int i = 0;
    __m256 v_alpha = _mm256_set1_ps(alpha_val);
    __m256 v_beta = _mm256_set1_ps(beta_val);

    // Vectorized main loop (8 floats per iteration)
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_loadu_ps(&y[i]);
        
        // Fused Multiply-Add: (alpha * x) + (beta * y) in a single hardware instruction
        __m256 result = _mm256_fmadd_ps(v_alpha, vx, _mm256_mul_ps(v_beta, vy));
        
        _mm256_storeu_ps(&y[i], result);
    }

    // Scalar cleanup loop for remaining elements
    for (; i < n; i++) {
        y[i] = alpha_val * x[i] + beta_val * y[i];
    }
}
