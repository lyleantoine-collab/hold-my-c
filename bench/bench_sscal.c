#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/sscal.h"

#define N 10000000
#define ITERATIONS 1000

int main() {
    float alpha = 2.0f;
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
    for (int it = 0; it < ITERATIONS; it++) {
        sscal_simd(N, alpha, x);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Executed %d iterations of OpenMP Aligned SIMD SSCAL on %d elements in %.4f seconds. Sample: %.2f\n", 
           ITERATIONS, N, elapsed, x[0]);

    free(x);
    return 0;
}
