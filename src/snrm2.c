#include "snrm2.h"
#include <immintrin.h>
#include <omp.h>
#include <math.h>

float snrm2_simd(int n, const float *x) {
    float global_sum_sq = 0.0f;

    #pragma omp parallel reduction(+:global_sum_sq)
    {
        __m256 v_sum_sq = _mm256_setzero_ps();

        #pragma omp for schedule(static)
        for (int i = 0; i <= n - 32; i += 32) {
            _mm_prefetch((const char*)&x[i + 32], _MM_HINT_T0);

            __m256 vx1 = _mm256_loadu_ps(&x[i]);
            v_sum_sq = _mm256_fmadd_ps(vx1, vx1, v_sum_sq);

            __m256 vx2 = _mm256_loadu_ps(&x[i + 8]);
            v_sum_sq = _mm256_fmadd_ps(vx2, vx2, v_sum_sq);

            __m256 vx3 = _mm256_loadu_ps(&x[i + 16]);
            v_sum_sq = _mm256_fmadd_ps(vx3, vx3, v_sum_sq);

            __m256 vx4 = _mm256_loadu_ps(&x[i + 24]);
            v_sum_sq = _mm256_fmadd_ps(vx4, vx4, v_sum_sq);
        }

        float buffer[8] __attribute__((aligned(32)));
        _mm256_storeu_ps(buffer, v_sum_sq);
        float local_sum = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                          buffer[4] + buffer[5] + buffer[6] + buffer[7];
        global_sum_sq += local_sum;
    }

    int remainder_start = (n / 32) * 32;
    float tail_sum_sq = 0.0f;
    for (int i = remainder_start; i < n; i++) {
        tail_sum_sq += x[i] * x[i];
    }
    global_sum_sq += tail_sum_sq;

    return sqrtf(global_sum_sq);
}
