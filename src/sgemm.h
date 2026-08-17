#ifndef SGEMM_H
#define SGEMM_H

void sgemm_simd(int m, int n, int k, float alpha, const float *A, const float *B, float beta, float *C);

#endif
