#include "ssymv.h"
#include <immintrin.h>
#include <omp.h>

void ssymv_simd(int n, float alpha, const float *A, const float *x, float beta, float *y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        y[i] *= beta;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        int j = 0;

        const float *row_i = &A[i * n];

        // Symmetric matrix-vector multiplication accumulation
        for (; j <= n - 8; j += 8) {
            __m256 va = _mm256_loadu_ps(&row_i[j]);
            __m256 vx = _mm256_loadu_ps(&x[j]);
            __m256 v_res = _mm256_mul_ps(va, vx);
            
            float buffer[8] __attribute__((aligned(32)));
            _mm256_storeu_ps(buffer, v_res);
            sum1 += buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                    buffer[4] + buffer[5] + buffer[6] + buffer[7];
        }

        for (; j < n; j++) {
            sum1 += row_i[j] * x[j];
        }

        y[i] += alpha * sum1;
    }
}
