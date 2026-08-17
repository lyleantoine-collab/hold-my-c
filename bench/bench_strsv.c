#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/strsv.h"

#define N 4096
#define ITERATIONS 500

int main() {
    size_t size_a = N * N * sizeof(float);
    size_t size_x = N * sizeof(float);

    float *A = (float *)aligned_alloc(32, size_a);
    float *x = (float *)aligned_alloc(32, size_x);

    if (!A || !x) {
        fprintf(stderr, "Aligned allocation failed\n");
        return 1;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0f : ((i > j) ? 0.05f : 0.0f);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
    }

    double start = omp_get_wtime();
    for (int it = 0; it < ITERATIONS; it++) {
        strsv_simd(N, A, x);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Executed %d iterations of OpenMP Aligned SIMD STRSV (%dx%d) in %.4f seconds. Sample x[0]: %.2f\n", 
           ITERATIONS, N, N, elapsed, x[0]);

    free(A);
    free(x);
    return 0;
}
