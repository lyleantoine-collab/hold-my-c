#include "sscal.h"
#include <immintrin.h>
#include <omp.h>

void sscal_simd(int n, float alpha, float *x) {
    __m256 v_alpha = _mm256_set1_ps(alpha);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n - 32; i += 32) {
        _mm_prefetch((const char*)&x[i + 32], _MM_HINT_T0);

        __m256 vx1 = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&x[i], _mm256_mul_ps(v_alpha, vx1));

        __m256 vx2 = _mm256_loadu_ps(&x[i + 8]);
        _mm256_storeu_ps(&x[i + 8], _mm256_mul_ps(v_alpha, vx2));

        __m256 vx3 = _mm256_loadu_ps(&x[i + 16]);
        _mm256_storeu_ps(&x[i + 16], _mm256_mul_ps(v_alpha, vx3));

        __m256 vx4 = _mm256_loadu_ps(&x[i + 24]);
        _mm256_storeu_ps(&x[i + 24], _mm256_mul_ps(v_alpha, vx4));
    }

    int tail_start = (n / 32) * 32;
    int i = tail_start;
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&x[i], _mm256_mul_ps(v_alpha, vx));
    }

    for (; i < n; i++) {
        x[i] *= alpha;
    }
}
