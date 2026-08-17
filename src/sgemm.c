#include "sgemm.h"
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

#define BLOCK_SIZE 64

void sgemm_simd(int m, int n, int k, float alpha, const float *A, const float *B, float beta, float *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m * n; i++) {
        C[i] *= beta;
    }

    // Allocate thread-local packed B buffer or handle globally for benchmark simplicity
    // Here we pack B into a temporary row-major block format for sequential access
    float *B_packed = (float *)aligned_alloc(32, k * n * sizeof(float));
    if (B_packed) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int p = 0; p < k; p++) {
            for (int j = 0; j < n; j++) {
                B_packed[p * n + j] = B[p * n + j];
            }
        }
    }

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

                        const float *b_ptr = B_packed ? B_packed : B;

                        for (; j <= jmax - 8; j += 8) {
                            __m256 v_b = _mm256_load_ps(&b_ptr[p * n + j]);
                            __m256 v_c = _mm256_load_ps(&C[i * n + j]);
                            v_c = _mm256_fmadd_ps(v_alpha, v_b, v_c);
                            _mm256_store_ps(&C[i * n + j], v_c);
                        }

                        for (; j < jmax; j++) {
                            C[i * n + j] += alpha * A[i * k + p] * b_ptr[p * n + j];
                        }
                    }
                }
            }
        }
    }

    if (B_packed) {
        free(B_packed);
    }
}
