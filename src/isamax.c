#include "isamax.h"
#include <immintrin.h>
#include <omp.h>
#include <float.h>

int isamax_simd(int n, const float *x) {
    if (n <= 0) return -1;

    float global_max = -1.0f;
    int global_max_idx = 0;

    // Use OpenMP reduction-like tracking (manual thread-local max search)
    #pragma omp parallel
    {
        float local_max = -1.0f;
        int local_max_idx = n; // Sentinel

        #pragma omp for schedule(static)
        for (int i = 0; i <= n - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            
            // Absolute value via bitwise mask
            const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 v_abs = _mm256_and_ps(vx, sign_mask);

            float buffer[8] __attribute__((aligned(32)));
            _mm256_storeu_ps(buffer, v_abs);

            for (int j = 0; j < 8; j++) {
                if (buffer[j] > local_max) {
                    local_max = buffer[j];
                    local_max_idx = i + j;
                } else if (buffer[j] == local_max && (i + j) < local_max_idx) {
                    local_max_idx = i + j; // Tie-breaker: lowest index
                }
            }
        }

        #pragma omp critical
        {
            if (local_max > global_max) {
                global_max = local_max;
                global_max_idx = local_max_idx;
            } else if (local_max == global_max && local_max_idx < global_max_idx) {
                global_max_idx = local_max_idx;
            }
        }
    }

    // Tail cleanup for remaining elements
    int remainder_start = (n / 8) * 8;
    for (int i = remainder_start; i < n; i++) {
        float val = (x[i] < 0.0f) ? -x[i] : x[i];
        if (val > global_max) {
            global_max = val;
            global_max_idx = i;
        }
    }

    return global_max_idx;
}
