#include "strsm.h"
#include <immintrin.h>
#include <omp.h>

void strsm_simd(int m, int n, float alpha, const float *A, float *B) {
    // Lower triangular solve: AX = alpha * B via forward substitution
    for (int i = 0; i < m; i++) {
        // Scale row i of B by alpha
        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m256 vb = _mm256_loadu_ps(&B[i * n + j]);
            __m256 v_alpha = _mm256_set1_ps(alpha);
            _mm256_storeu_ps(&B[i * n + j], _mm256_mul_ps(vb, v_alpha));
        }
        for (; j < n; j++) {
            B[i * n + j] *= alpha;
        }

        // Subtract contributions from preceding rows k < i
        for (int k = 0; k < i; k++) {
            float aik = A[i * m + k];
            __m256 v_aik = _mm256_set1_ps(aik);
            int j = 0;

            const float *bk_row = &B[k * n];
            float *bi_row = &B[i * n];

            for (; j <= n - 8; j += 8) {
                __m256 vbk = _mm256_loadu_ps(&bk_row[j]);
                __m256 vbi = _mm256_loadu_ps(&bi_row[j]);
                vbi = _mm256_fnmadd_ps(v_aik, vbk, vbi); // vbi = -v_aik * vbk + vbi
                _mm256_storeu_ps(&bi_row[j], vbi);
            }
            for (; j < n; j++) {
                bi_row[j] -= aik * bk_row[j];
            }
        }

        // Divide by diagonal element A[i, i]
        float aii = A[i * m + i];
        float inv_aii = 1.0f / aii;
        __m256 v_inv = _mm256_set1_ps(inv_aii);
        int j = 0;
        float *bi_row = &B[i * n];

        for (; j <= n - 8; j += 8) {
            __m256 vbi = _mm256_loadu_ps(&bi_row[j]);
            _mm256_storeu_ps(&bi_row[j], _mm256_mul_ps(vbi, v_inv));
        }
        for (; j < n; j++) {
            bi_row[j] *= inv_aii;
        }
    }
}

