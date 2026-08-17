#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/axpby.h"

// Declare the SIMD variant
void axpby_simd(int n, float alpha, const float *x, float beta, float *y);

#define N 10000000
#define ITERATIONS 1000

int main() {
    float alpha = 2.0f;
    float beta = 0.5f;

    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));

    if (!x || !y) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    clock_t start = clock();
    for (int it = 0; it < ITERATIONS; it++) {
        axpby_simd(N, alpha, x, beta, y);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Executed %d iterations of AVX2 SIMD AXPBY on %d elements in %.4f seconds.\n", ITERATIONS, N, elapsed);

    free(x);
    free(y);
    return 0;
}
