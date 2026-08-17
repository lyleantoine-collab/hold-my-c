#include "axpby.h"
#include <immintrin.h>
#include <omp.h>

void axpby_simd(int n, float alpha_val, const float *x, float beta_val, float *y) {
    __m256 v_alpha = _mm256_set1_ps(alpha_val);
    __m256 v_beta = _mm256_set1_ps(beta_val);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n - 32; i += 32) {
        _mm_prefetch((const char*)&x[i + 32], _MM_HINT_T0);
        _mm_prefetch((const char*)&y[i + 32], _MM_HINT_T0);

        __m256 vx1 = _mm256_load_ps(&x[i]);
        __m256 vy1 = _mm256_load_ps(&y[i]);
        _mm256_store_ps(&y[i], _mm256_fmadd_ps(v_alpha, vx1, _mm256_mul_ps(v_beta, vy1)));

        __m256 vx2 = _mm256_load_ps(&x[i + 8]);
        __m256 vy2 = _mm256_load_ps(&y[i + 8]);
        _mm256_store_ps(&y[i + 8], _mm256_fmadd_ps(v_alpha, vx2, _mm256_mul_ps(v_beta, vy2)));

        __m256 vx3 = _mm256_load_ps(&x[i + 16]);
        __m256 vy3 = _mm256_load_ps(&y[i + 16]);
        _mm256_store_ps(&y[i + 16], _mm256_fmadd_ps(v_alpha, vx3, _mm256_mul_ps(v_beta, vy3)));

        __m256 vx4 = _mm256_load_ps(&x[i + 24]);
        __m256 vy4 = _mm256_load_ps(&y[i + 24]);
        _mm256_store_ps(&y[i + 24], _mm256_fmadd_ps(v_alpha, vx4, _mm256_mul_ps(v_beta, vy4)));
    }

    int tail_start = (n / 32) * 32;
    int i = tail_start;
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]);
        __m256 vy = _mm256_load_ps(&y[i]);
        _mm256_store_ps(&y[i], _mm256_fmadd_ps(v_alpha, vx, _mm256_mul_ps(v_beta, vy)));
    }

    for (; i < n; i++) {
        y[i] = alpha_val * x[i] + beta_val * y[i];
    }
}
