#include "axpby.h"

// Pure C implementation of AXPBY: y = alpha * x + beta * y
// Zero-overhead loop bypassing all framework bloat.
void axpby_scalar(int n, float alpha, const float *x, float beta, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}
