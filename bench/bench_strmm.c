#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/strmm.h"

#define M 512
#define N 512
#define ITERATIONS 50

int main() {
    size_t size_a = M * M * sizeof(float);
    size_t size_b = M * N * sizeof(float);

    float *A = (float *)aligned_alloc(32, size_a);
    float *B = (float *)aligned_alloc(32, size_b);

    if (!A || !B) {
        fprintf(stderr, "Aligned allocation failed for matrices\n");
        return 1;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            A[i * M + j] = (i >= j) ? 1.0f : 0.0f;
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M * N; i++) {
        B[i] = 1.0f;
    }

    double start = omp_get_wtime();
    for (int it = 0; it < ITERATIONS; it++) {
        strmm_simd(M, N, 1.0f, A, B);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    double gflops = (1.0 * (double)M * (double)M * (double)N * (double)ITERATIONS) / (elapsed * 1e9);

    printf("Executed %d iterations of STRMM (%dx%dx%d) in %.4f seconds. Performance: %.2f GFLOPS\n", 
           ITERATIONS, M, M, N, elapsed, gflops);

    free(A);
    free(B);
    return 0;
}
