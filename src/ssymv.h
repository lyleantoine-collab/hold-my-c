#ifndef SSYMV_H
#define SSYMV_H

void ssymv_simd(int n, float alpha, const float *A, const float *x, float beta, float *y);

#endif
