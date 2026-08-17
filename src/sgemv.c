#include "sgemv.h"
#include <immintrin.h>
#include <omp.h>

void sgemv_simd(int m, int n, float alpha, const float *A, const float *x, float beta, float *y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        y[i] *= beta;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        __m256 v_sum = _mm256_setzero_ps();
        int j = 0;
        
        const float *row_a = &A[i * n];

        for (; j <= n - 32; j += 32) {
            _mm_prefetch((const char*)&row_a[j + 32], _MM_HINT_T0);
            _mm_prefetch((const char*)&x[j + 32], _MM_HINT_T0);

            __m256 va1 = _mm256_loadu_ps(&row_a[j]);
            __m256 vx1 = _mm256_loadu_ps(&x[j]);
            v_sum = _mm256_fmadd_ps(va1, vx1, v_sum);

            __m256 va2 = _mm256_loadu_ps(&row_a[j + 8]);
            __m256 vx2 = _mm256_loadu_ps(&x[j + 8]);
            v_sum = _mm256_fmadd_ps(va2, vx2, v_sum);

            __m256 va3 = _mm256_loadu_ps(&row_a[j + 16]);
            __m256 vx3 = _mm256_loadu_ps(&x[j + 16]);
            v_sum = _mm256_fmadd_ps(va3, vx3, v_sum);

            __m256 va4 = _mm256_loadu_ps(&row_a[j + 24]);
            __m256 vx4 = _mm256_loadu_ps(&x[j + 24]);
            v_sum = _mm256_fmadd_ps(va4, vx4, v_sum);
        }

        float buffer[8] __attribute__((aligned(32)));
        _mm256_storeu_ps(buffer, v_sum);
        float row_dot = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                        buffer[4] + buffer[5] + buffer[6] + buffer[7];

        int tail_start = (n / 32) * 32;
        int j_tail = tail_start;
        for (; j_tail <= n - 8; j_tail += 8) {
            __m256 va = _mm256_loadu_ps(&row_a[j_tail]);
            __m256 vx = _mm256_loadu_ps(&x[j_tail]);
            __m256 v_res = _mm256_mul_ps(va, vx);
            _mm256_storeu_ps(buffer, v_res);
            row_dot += buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                       buffer[4] + buffer[5] + buffer[6] + buffer[7];
        }

        for (; j_tail < n; j_tail++) {
            row_dot += row_a[j_tail] * x[j_tail];
        }

        y[i] += alpha * row_dot;
    }
}
