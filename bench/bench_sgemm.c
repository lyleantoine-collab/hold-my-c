#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../src/sgemm.h"

#define M 512
#define N 512
#define K 512
#define ITERATIONS 50

int main() {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *A = (float *)aligned_alloc(32, size_a);
    float *B = (float *)aligned_alloc(32, size_b);
    float *C = (float *)aligned_alloc(32, size_c);

    if (!A || !B || !C) {
        fprintf(stderr, "Aligned allocation failed for matrices\n");
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    double start = omp_get_wtime();
    for (int it = 0; it < ITERATIONS; it++) {
        sgemm_simd(M, N, K, 1.0f, A, B, 0.0f, C);
    }
    double end = omp_get_wtime();

    double elapsed = end - start;
    double gflops = (2.0 * (double)M * (double)N * (double)K * (double)ITERATIONS) / (elapsed * 1e9);

    printf("Executed %d iterations of SGEMM (%dx%dx%d) in %.4f seconds. Performance: %.2f GFLOPS\n", 
           ITERATIONS, M, N, K, elapsed, gflops);

    free(A);
    free(B);
    free(C);
    return 0;
}
