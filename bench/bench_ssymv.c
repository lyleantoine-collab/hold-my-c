#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/ssymv.h"

#define N 2048
#define ITERATIONS 100

int main() {
    size_t size_a = N * N * sizeof(float);
    size_t size_x = N * sizeof(float);
    size_t size_y = N * sizeof(float);

    float *A = (float *)aligned_alloc(32, size_a);
    float *x = (float *)aligned_alloc(32, size_x);
    float *y = (float *)aligned_alloc(32, size_y);

    if (!A || !x || !y) {
        fprintf(stderr, "Aligned allocation failed\n");
        return 1;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0f : 1.0f;
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 0.0f;
    }

    double start = omp_get_wtime();
    for (int it = 0; it < ITERATIONS; it++) {
        ssymv_simd(N, 1.0f, A, x, 0.0f, y);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Executed %d iterations of OpenMP Aligned SIMD SSYMV (%dx%d) in %.4f seconds. Sample y: %.2f\n", 
           ITERATIONS, N, N, elapsed, y[0]);

    free(A);
    free(x);
    free(y);
    return 0;
}
