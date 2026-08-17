#include "sasum.h"
#include <immintrin.h>
#include <omp.h>

float sasum_simd(int n, const float *x) {
    float global_sum = 0.0f;
    
    // Mask to clear the sign bit for absolute value: clears bit 31
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    #pragma omp parallel reduction(+:global_sum)
    {
        __m256 v_sum = _mm256_setzero_ps();

        #pragma omp for schedule(static)
        for (int i = 0; i <= n - 32; i += 32) {
            _mm_prefetch((const char*)&x[i + 32], _MM_HINT_T0);

            __m256 vx1 = _mm256_loadu_ps(&x[i]);
            v_sum = _mm256_add_ps(_mm256_and_ps(vx1, sign_mask), v_sum);

            __m256 vx2 = _mm256_loadu_ps(&x[i + 8]);
            v_sum = _mm256_add_ps(_mm256_and_ps(vx2, sign_mask), v_sum);

            __m256 vx3 = _mm256_loadu_ps(&x[i + 16]);
            v_sum = _mm256_add_ps(_mm256_and_ps(vx3, sign_mask), v_sum);

            __m256 vx4 = _mm256_loadu_ps(&x[i + 24]);
            v_sum = _mm256_add_ps(_mm256_and_ps(vx4, sign_mask), v_sum);
        }

        float buffer[8] __attribute__((aligned(32)));
        _mm256_storeu_ps(buffer, v_sum);
        float local_sum = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                          buffer[4] + buffer[5] + buffer[6] + buffer[7];
        global_sum += local_sum;
    }

    int remainder_start = (n / 32) * 32;
    float tail_sum = 0.0f;
    for (int i = remainder_start; i < n; i++) {
        tail_sum += (x[i] < 0.0f) ? -x[i] : x[i];
    }
    global_sum += tail_sum;

    return global_sum;
}
