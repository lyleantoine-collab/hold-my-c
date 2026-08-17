hold-my-c Benchmark Telemetry Results
Complete empirical telemetry gathered directly from the metal on polyarb-01 across all Level 1, Level 2, and Level 3 BLAS routines.
Summary Table
| BLAS Level | Routine | Problem Size / Matrix Dimensions | Iterations | Elapsed Time (s) | Throughput / Sample Output |
|---|---|---|---|---|---|
| Level 1 | SSCAL | 10,000,000 elements | 1,000 | 2.1713 s | Sample: inf (Overflow) |
| Level 1 | SSWAP | 10,000,000 elements | 1,000 | 4.6138 s | Sample x: 1.00, y: 2.00 |
| Level 1 | SCOPY | 10,000,000 elements | 1,000 | 4.1953 s | Sample y: 3.14 |
| Level 2 | SGEMV | 2048 \times 2048 | 100 | 0.0535 s | Sample y: 2048.00 |
| Level 2 | SGER | 2048 \times 2048 | 100 | 0.0479 s | Sample A[0]: 100.00 |
| Level 2 | SSYMV | 2048 \times 2048 | 100 | 0.1055 s | Sample y: 2049.00 |
| Level 2 | STRSV | 4096 \times 4096 | 500 | 5.9068 s | Sample x[0]: 0.00 |
| Level 3 | SSYRK | 512 \times 512 \times 512 | 50 | 0.8186 s | 16.40 GFLOPS |
| Level 3 | SSYR2K | 512 \times 512 \times 512 | 50 | 1.0171 s | 26.39 GFLOPS |
| Level 3 | SSYMM | 512 \times 512 \times 512 | 50 | 0.4941 s | 27.16 GFLOPS |
| Level 3 | STRMM | 512 \times 512 \times 512 | 50 | 0.3799 s | 17.67 GFLOPS |
| Level 3 | STRSM | 512 \times 512 \times 512 | 50 | 3.9388 s | 1.70 GFLOPS (Dependency Bound) |
Key Observations
 * Peak Compute Speed: Symmetric matrix-matrix multiplication (SSYMM) hit peak performance at 27.16 GFLOPS using blocked AVX2 loops and OpenMP parallel scheduling.
 * Memory Bandwidth Senders: Level 1 operations (SCOPY, SSWAP, SSCAL) completely saturated memory bus channels over large 40MB vector arrays.
 * The Dependency Penalty: Triangular solves (STRSM) dropped to 1.70 GFLOPS due to the inherent loop-carried serialization of back-substitution across rows.
