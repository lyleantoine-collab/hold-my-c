#include "sgemm.h"
#include <immintrin.h>
#include <omp.h>

#define BLOCK_SIZE 64

void sgemm_simd(int m, int n, int k, float alpha, const float *A, const float *B, float beta, float *C) {
    // Handle beta scaling across matrix C
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m * n; i++) {
        C[i] *= beta;
    }

    // Blocked (tiled) GEMM for L1/L2 cache locality
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < m; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < k; bk += BLOCK_SIZE) {
                
                int imax = (bi + BLOCK_SIZE < m) ? bi + BLOCK_SIZE : m;
                int jmax = (bj + BLOCK_SIZE < n) ? bj + BLOCK_SIZE : n;
                int kmax = (bk + BLOCK_SIZE < k) ? bk + BLOCK_SIZE : k;

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

                        // Scalar tail cleanup for columns within the block
                        for (; j < jmax; j++) {
                            C[i * n + j] += alpha * A[i * k + p] * B[p * n + j];
                        }
                    }
                }
            }
        }
    }
}
