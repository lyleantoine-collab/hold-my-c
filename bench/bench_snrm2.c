#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/snrm2.h"

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

    double start = omp_get_wtime();
    float result = 0.0f;
    for (int it = 0; it < ITERATIONS; it++) {
        result = snrm2_simd(N, x);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Executed %d iterations of OpenMP Aligned SIMD SNRM2 on %d elements in %.4f seconds. Result: %.2f\n", 
           ITERATIONS, N, elapsed, result);

    free(x);
    return 0;
}
