#ifndef SGEMV_H
#define SGEMV_H

void sgemv_simd(int m, int n, float alpha, const float *A, const float *x, float beta, float *y);

#endif
