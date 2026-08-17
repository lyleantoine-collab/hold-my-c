#ifndef SSYMM_H
#define SSYMM_H

void ssymm_simd(int m, int n, float alpha, const float *A, const float *B, float beta, float *C);

#endif
