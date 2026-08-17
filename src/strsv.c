#include "strsv.h"
#include <immintrin.h>

void strsv_simd(int n, const float *A, float *x) {
    // Lower triangular forward substitution (Ax = b, solved in-place on x)
    for (int i = 0; i < n; i++) {
        float sum = x[i];
        int j = 0;
        const float *row_i = &A[i * n];

        for (; j <= i - 8; j += 8) {
            __m256 va = _mm256_loadu_ps(&row_i[j]);
            __m256 vx = _mm256_loadu_ps(&x[j]);
            __m256 v_prod = _mm256_mul_ps(va, vx);

            float buffer[8] __attribute__((aligned(32)));
            _mm256_storeu_ps(buffer, v_prod);
            sum -= (buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                    buffer[4] + buffer[5] + buffer[6] + buffer[7]);
        }

        for (; j < i; j++) {
            sum -= row_i[j] * x[j];
        }

        x[i] = sum / row_i[i];
    }
}
