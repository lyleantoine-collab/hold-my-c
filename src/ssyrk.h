#ifndef SSYRK_H
#define SSYRK_H

void ssyrk_simd(int n, int k, float alpha, const float *A, float beta, float *C);

#endif
