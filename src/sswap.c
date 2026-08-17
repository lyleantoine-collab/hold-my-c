#include "sswap.h"
#include <immintrin.h>
#include <omp.h>

void sswap_simd(int n, float *x, float *y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n - 32; i += 32) {
        _mm_prefetch((const char*)&x[i + 32], _MM_HINT_T0);
        _mm_prefetch((const char*)&y[i + 32], _MM_HINT_T0);

        __m256 vx1 = _mm256_loadu_ps(&x[i]);
        __m256 vy1 = _mm256_loadu_ps(&y[i]);
        _mm256_storeu_ps(&x[i], vy1);
        _mm256_storeu_ps(&y[i], vx1);

        __m256 vx2 = _mm256_loadu_ps(&x[i + 8]);
        __m256 vy2 = _mm256_loadu_ps(&y[i + 8]);
        _mm256_storeu_ps(&x[i + 8], vy2);
        _mm256_storeu_ps(&y[i + 8], vx2);

        __m256 vx3 = _mm256_loadu_ps(&x[i + 16]);
        __m256 vy3 = _mm256_loadu_ps(&y[i + 16]);
        _mm256_storeu_ps(&x[i + 16], vy3);
        _mm256_storeu_ps(&y[i + 16], vx3);

        __m256 vx4 = _mm256_loadu_ps(&x[i + 24]);
        __m256 vy4 = _mm256_loadu_ps(&y[i + 24]);
        _mm256_storeu_ps(&x[i + 24], vy4);
        _mm256_storeu_ps(&y[i + 24], vx4);
    }

    int tail_start = (n / 32) * 32;
    int i = tail_start;
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_loadu_ps(&y[i]);
        _mm256_storeu_ps(&x[i], vy);
        _mm256_storeu_ps(&y[i], vx);
    }

    for (; i < n; i++) {
        float temp = x[i];
        x[i] = y[i];
        y[i] = temp;
    }
}
