#include "strmm.h"
#include <immintrin.h>
#include <omp.h>

void strmm_simd(int m, int n, float alpha, const float *A, float *B) {
    // Temporary buffer to hold computed block rows to avoid in-place read/write corruption
    #pragma omp parallel
    {
        float *row_buf = (float *)aligned_alloc(32, n * sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                row_buf[j] = 0.0f;
            }

            for (int k = 0; k <= i; k++) {
                float aik = A[i * m + k];
                __m256 v_aik = _mm256_set1_ps(aik * alpha);
                int j = 0;

                const float *b_row = &B[k * n];
                float *buf_row = row_buf;

                for (; j <= n - 8; j += 8) {
                    __m256 vb = _mm256_loadu_ps(&b_row[j]);
                    __m256 vbuf = _mm256_loadu_ps(&buf_row[j]);
                    vbuf = _mm256_fmadd_ps(v_aik, vb, vbuf);
                    _mm256_storeu_ps(&buf_row[j], vbuf);
                }

                for (; j < n; j++) {
                    buf_row[j] += (aik * alpha) * b_row[j];
                }
            }

            // Copy computed row back to B matrix
            for (int j = 0; j < n; j++) {
                B[i * n + j] = row_buf[j];
            }
        }
        free(row_buf);
    }
}
