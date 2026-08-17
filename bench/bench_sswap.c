#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/sswap.h"

#define N 10000000
#define ITERATIONS 1000

int main() {
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

    double start = omp_get_wtime();
    for (int it = 0; it < ITERATIONS; it++) {
        sswap_simd(N, x, y);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Executed %d iterations of OpenMP Aligned SIMD SSWAP on %d elements in %.4f seconds. Sample x: %.2f, y: %.2f\n", 
           ITERATIONS, N, elapsed, x[0], y[0]);

    free(x);
    free(y);
    return 0;
}
