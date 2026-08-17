#include "sger.h"
#include <immintrin.h>
#include <omp.h>

void sger_simd(int m, int n, float alpha, const float *x, const float *y, float *A) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        float xi_alpha = alpha * x[i];
        __m256 v_xi_alpha = _mm256_set1_ps(xi_alpha);
        int j = 0;

        float *row_a = &A[i * n];

        for (; j <= n - 32; j += 32) {
            _mm_prefetch((const char*)&row_a[j + 32], _MM_HINT_T0);
            _mm_prefetch((const char*)&y[j + 32], _MM_HINT_T0);

            __m256 vy1 = _mm256_loadu_ps(&y[j]);
            __m256 va1 = _mm256_loadu_ps(&row_a[j]);
            _mm256_storeu_ps(&row_a[j], _mm256_fmadd_ps(v_xi_alpha, vy1, va1));

            __m256 vy2 = _mm256_loadu_ps(&y[j + 8]);
            __m256 va2 = _mm256_loadu_ps(&row_a[j + 8]);
            _mm256_storeu_ps(&row_a[j + 8], _mm256_fmadd_ps(v_xi_alpha, vy2, va2));

            __m256 vy3 = _mm256_loadu_ps(&y[j + 16]);
            __m256 va3 = _mm256_loadu_ps(&row_a[j + 16]);
            _mm256_storeu_ps(&row_a[j + 16], _mm256_fmadd_ps(v_xi_alpha, vy3, va3));

            __m256 vy4 = _mm256_loadu_ps(&y[j + 24]);
            __m256 va4 = _mm256_loadu_ps(&row_a[j + 24]);
            _mm256_storeu_ps(&row_a[j + 24], _mm256_fmadd_ps(v_xi_alpha, vy4, va4));
        }

        int tail_start = (n / 32) * 32;
        int j_tail = tail_start;
        for (; j_tail <= n - 8; j_tail += 8) {
            __m256 vy = _mm256_loadu_ps(&y[j_tail]);
            __m256 va = _mm256_loadu_ps(&row_a[j_tail]);
            _mm256_storeu_ps(&row_a[j_tail], _mm256_fmadd_ps(v_xi_alpha, vy, va));
        }

        for (; j_tail < n; j_tail++) {
            row_a[j_tail] += xi_alpha * y[j_tail];
        }
    }
}
