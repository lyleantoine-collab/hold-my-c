#include "ssyr2k.h"
#include <immintrin.h>
#include <omp.h>

#define BLOCK_SIZE 64

void ssyr2k_simd(int n, int k, float alpha, const float *A, const float *B, float beta, float *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            C[i * n + j] *= beta;
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
        for (int bj = 0; bj <= bi; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < k; bk += BLOCK_SIZE) {

                int imax = (bi + BLOCK_SIZE < n) ? bi + BLOCK_SIZE : n;
                int jmax = (bj + BLOCK_SIZE < n) ? bj + BLOCK_SIZE : n;
                int kmax = (bk + BLOCK_SIZE < k) ? bk + BLOCK_SIZE : k;

                for (int i = bi; i < imax; i++) {
                    int j_limit = (i < jmax) ? i + 1 : jmax;
                    for (int j = bj; j < j_limit; j++) {
                        
                        float sum = 0.0f;
                        int p = bk;

                        for (; p <= kmax - 8; p += 8) {
                            __m256 va_i = _mm256_loadu_ps(&A[i * k + p]);
                            __m256 vb_j = _mm256_loadu_ps(&B[j * k + p]);
                            __m256 vb_i = _mm256_loadu_ps(&B[i * k + p]);
                            __m256 va_j = _mm256_loadu_ps(&A[j * k + p]);

                            __m256 v_prod1 = _mm256_mul_ps(va_i, vb_j);
                            __m256 v_prod2 = _mm256_mul_ps(vb_i, va_j);
                            __m256 v_total = _mm256_add_ps(v_prod1, v_prod2);

                            float buffer[8] __attribute__((aligned(32)));
                            _mm256_storeu_ps(buffer, v_total);
                            sum += buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                                   buffer[4] + buffer[5] + buffer[6] + buffer[7];
                        }

                        for (; p < kmax; p++) {
                            sum += (A[i * k + p] * B[j * k + p]) + (B[i * k + p] * A[j * k + p]);
                        }

                        C[i * n + j] += alpha * sum;
                    }
                }
            }
        }
    }
}
