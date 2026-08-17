#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/isamax.h"

#define N 10000000
#define ITERATIONS 1000

int main() {
    size_t bytes = N * sizeof(float);
    float *x = (float *)aligned_alloc(32, bytes);

    if (!x) {
        fprintf(stderr, "Aligned allocation failed\n");
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
    }
    // Inject a distinct maximum value
    x[543210] = 99.0f;

    double start = omp_get_wtime();
    int result = 0;
    for (int it = 0; it < ITERATIONS; it++) {
        result = isamax_simd(N, x);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Executed %d iterations of OpenMP Aligned SIMD ISAMAX on %d elements in %.4f seconds. Max Index: %d\n", 
           ITERATIONS, N, elapsed, result);

    free(x);
    return 0;
}
