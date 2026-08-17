#include "ssymm.h"
#include <immintrin.h>
#include <omp.h>

#define BLOCK_SIZE 64

void ssymm_simd(int m, int n, float alpha, const float *A, const float *B, float beta, float *C) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] *= beta;
        }
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < m; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < m; bk += BLOCK_SIZE) {

                int imax = (bi + BLOCK_SIZE < m) ? bi + BLOCK_SIZE : m;
                int jmax = (bj + BLOCK_SIZE < n) ? bj + BLOCK_SIZE : n;
                int kmax = (bk + BLOCK_SIZE < m) ? bk + BLOCK_SIZE : m;

                for (int i = bi; i < imax; i++) {
                    for (int p = bk; p < kmax; p++) {
                        float a_val = alpha * A[i * m + p];
                        __m256 v_alpha = _mm256_set1_ps(a_val);
                        int j = bj;

                        for (; j <= jmax - 8; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[p * n + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * n + j]);
                            vc = _mm256_fmadd_ps(v_alpha, vb, vc);
                            _mm256_storeu_ps(&C[i * n + j], vc);
                        }

                        for (; j < jmax; j++) {
                            C[i * n + j] += a_val * B[p * n + j];
                        }
                    }
                }
            }
        }
    }
}
