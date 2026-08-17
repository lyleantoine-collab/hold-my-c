#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include "../src/axpby.h"

void axpby_simd(int n, float alpha, const float *x, float beta, float *y);

#define N 10000000
#define ITERATIONS 1000

int main() {
    float alpha = 2.0f;
    float beta = 0.5f;

    size_t bytes = N * sizeof(float);
    float *x = (float *)aligned_alloc(32, bytes);
    float *y = (float *)aligned_alloc(32, bytes);

    if (!x || !y) {
        fprintf(stderr, "Aligned allocation failed\n");
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Use wall-clock time for accurate multi-threaded measurement
    double start = omp_get_wtime();
    for (int it = 0; it < ITERATIONS; it++) {
        axpby_simd(N, alpha, x, beta, y);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Executed %d iterations of OpenMP Aligned SIMD AXPBY on %d elements in %.4f seconds.\n", ITERATIONS, N, elapsed);

    free(x);
    free(y);
    return 0;
}
