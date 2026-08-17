#include "sgemm.h"
#include <immintrin.h>
#include <omp.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64

void sgemm_simd(int m, int n, int k, float alpha, const float *A, const float *B, float beta, float *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m * n; i++) {
        C[i] *= beta;
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < m; bi += BLOCK_M) {
        for (int bj = 0; bj < n; bj += BLOCK_N) {
            for (int bk = 0; bk < k; bk += BLOCK_K) {

                int imax = (bi + BLOCK_M < m) ? bi + BLOCK_M : m;
                int jmax = (bj + BLOCK_N < n) ? bj + BLOCK_N : n;
                int kmax = (bk + BLOCK_K < k) ? bk + BLOCK_K : k;

                for (int i = bi; i < imax; i++) {
                    for (int p = bk; p < kmax; p++) {
                        __m256 v_alpha = _mm256_set1_ps(alpha * A[i * k + p]);
                        int j = bj;

                        for (; j <= jmax - 8; j += 8) {
                            __m256 v_b = _mm256_loadu_ps(&B[p * n + j]);
                            __m256 v_c = _mm256_loadu_ps(&C[i * n + j]);
                            v_c = _mm256_fmadd_ps(v_alpha, v_b, v_c);
                            _mm256_storeu_ps(&C[i * n + j], v_c);
                        }

                        for (; j < jmax; j++) {
                            C[i * n + j] += alpha * A[i * k + p] * B[p * n + j];
                        }
                    }
                }
            }
        }
    }
}
