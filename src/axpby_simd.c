#include "axpby.h"
#include <immintrin.h>
#include <omp.h>

void axpby_simd(int n, float alpha_val, const float *x, float beta_val, float *y) {
    __m256 v_alpha = _mm256_set1_ps(alpha_val);
    __m256 v_beta = _mm256_set1_ps(beta_val);

    // Unroll by 4 vectors (32 floats per iteration) with software prefetching
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n - 32; i += 32) {
        // Prefetch data 64 bytes (2 cache lines) ahead
        _mm_prefetch((const char*)&x[i + 32], _MM_HINT_T0);
        _mm_prefetch((const char*)&y[i + 32], _MM_HINT_T0);

        // Chunk 1
        __m256 vx1 = _mm256_load_ps(&x[i]);
        __m256 vy1 = _mm256_load_ps(&y[i]);
        __m256 r1 = _mm256_fmadd_ps(v_alpha, vx1, _mm256_mul_ps(v_beta, vy1));
        _mm256_store_ps(&y[i], r1);

        // Chunk 2
        __m256 vx2 = _mm256_load_ps(&x[i + 8]);
        __m256 vy2 = _mm256_load_ps(&y[i + 8]);
        __m256 r2 = _mm256_fmadd_ps(v_alpha, vx2, _mm256_mul_ps(v_beta, vy2));
        _mm256_store_ps(&y[i + 8], r2);

        // Chunk 3
        __m256 vx3 = _mm256_load_ps(&x[i + 16]);
        __m256 vy3 = _mm256_load_ps(&y[i + 16]);
        __m256 r3 = _mm256_fmadd_ps(v_alpha, vx3, _mm256_mul_ps(v_beta, vy3));
        _mm256_store_ps(&y[i + 16], r3);

        // Chunk 4
        __m256 vx4 = _mm256_load_ps(&x[i + 24]);
        __m256 vy4 = _mm256_load_ps(&y[i + 24]);
        __m256 r4 = _mm256_fmadd_ps(v_alpha, vx4, _mm256_mul_ps(v_beta, vy4));
        _mm256_store_ps(&y[i + 24], r4);
    }

    // Handle remaining elements with standard 8-wide SIMD
    int tail_start = (n / 32) * 32;
    int i = tail_start;
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]);
        __m256 vy = _mm256_load_ps(&y[i]);
        __m256 result = _mm256_fmadd_ps(v_alpha, vx, _mm256_mul_ps(v_beta, vy));
        _mm256_store_ps(&y[i], result);
    }

    // Scalar cleanup for final tail
    for (; i < n; i++) {
        y[i] = alpha_val * x[i] + beta_val * y[i];
    }
}
