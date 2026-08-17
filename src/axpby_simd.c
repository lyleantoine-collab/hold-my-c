#include "axpby.h"
#include <immintrin.h>
#include <omp.h>

void axpby_simd(int n, float alpha_val, const float *x, float beta_val, float *y) {
    __m256 v_alpha = _mm256_set1_ps(alpha_val);
    __m256 v_beta = _mm256_set1_ps(beta_val);

    // Parallelize the SIMD chunk loop across available CPU cores
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n - 8; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]);
        __m256 vy = _mm256_load_ps(&y[i]);
        
        __m256 result = _mm256_fmadd_ps(v_alpha, vx, _mm256_mul_ps(v_beta, vy));
        
        // Standard aligned store keeps data warm in cache for iterative loops
        _mm256_store_ps(&y[i], result);
    }

    // Handle remaining tail elements sequentially
    int remainder_start = (n / 8) * 8;
    for (int i = remainder_start; i < n; i++) {
        y[i] = alpha_val * x[i] + beta_val * y[i];
    }
}
