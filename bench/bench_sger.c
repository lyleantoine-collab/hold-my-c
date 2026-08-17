#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/sger.h"

#define M 2048
#define N 2048
#define ITERATIONS 100

int main() {
    size_t size_a = M * N * sizeof(float);
    size_t size_x = M * sizeof(float);
    size_t size_y = N * sizeof(float);

    float *A = (float *)aligned_alloc(32, size_a);
    float *x = (float *)aligned_alloc(32, size_x);
    float *y = (float *)aligned_alloc(32, size_y);

    if (!A || !x || !y) {
        fprintf(stderr, "Aligned allocation failed\n");
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M * N; i++) A[i] = 0.0f;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) x[i] = 1.0f;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) y[i] = 1.0f;

    double start = omp_get_wtime();
    for (int it = 0; it < ITERATIONS; it++) {
        sger_simd(M, N, 1.0f, x, y, A);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Executed %d iterations of OpenMP Aligned SIMD SGER (%dx%d) in %.4f seconds. Sample A[0]: %.2f\n", 
           ITERATIONS, M, N, elapsed, A[0]);

    free(A);
    free(x);
    free(y);
    return 0;
}
