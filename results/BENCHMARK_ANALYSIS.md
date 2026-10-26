# CPU Matrix Multiplication Benchmark Analysis Report

**Hardware Platform:** AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx  

**Compiler Toolchain:** GCC 11.4  

**Date:** System Benchmark Execution  


## 1. Executive Performance Summary

Dense Matrix Multiplication computes $C = A \times B$ where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and $C \in \mathbb{R}^{M \times N}$. The standard dense operation count is $2 \times M \times N \times K$ floating-point operations (1 multiply + 1 accumulate per inner product term).

Throughput is defined as:

$$\text{Throughput (GFLOP/s)} = \frac{2 \times M \times N \times K}{\text{Execution Time (seconds)} \times 10^9}$$

Speedup is defined as:

$$\text{Speedup} = \frac{\text{Median Time}_{\text{baseline}}}{\text{Median Time}_{\text{optimized}}}$$

### Matrix Shape: 128 x 128 x 128

| Implementation | Tile Size | Threads | Median Time (ms) | StdDev (ms) | GFLOP/s | Speedup | Correctness | Max Rel Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | 32 | All | 3.58 | 6.08 | 1.17 | **1.00x** | PASSED | 0.00e+00 |
| `cache_friendly` | 32 | All | 0.41 | 0.04 | 10.22 | **8.72x** | PASSED | 0.00e+00 |
| `blocked` | 32 | All | 0.49 | 0.04 | 8.55 | **7.30x** | PASSED | 0.00e+00 |
| `avx2` | 32 | All | 0.19 | 0.03 | 21.64 | **18.48x** | PASSED | 0.00e+00 |
| `openmp` | 32 | All | 14.17 | 5.51 | 0.30 | **0.25x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 32 | All | 0.07 | 0.01 | 57.93 | **49.47x** | PASSED | 0.00e+00 |
| `blas` | 32 | All | 0.12 | 1.37 | 36.14 | **30.85x** | PASSED | 1.62e-04 |


### Matrix Shape: 512 x 512 x 512

| Implementation | Tile Size | Threads | Median Time (ms) | StdDev (ms) | GFLOP/s | Speedup | Correctness | Max Rel Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | 64 | All | 393.90 | 22.05 | 0.68 | **1.00x** | PASSED | 0.00e+00 |
| `cache_friendly` | 64 | All | 20.57 | 1.99 | 13.05 | **19.15x** | PASSED | 0.00e+00 |
| `blocked` | 64 | All | 24.24 | 1.62 | 11.07 | **16.25x** | PASSED | 0.00e+00 |
| `avx2` | 64 | All | 10.24 | 0.92 | 26.22 | **38.47x** | PASSED | 0.00e+00 |
| `openmp` | 64 | All | 24.80 | 5.50 | 10.82 | **15.88x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | All | 20.59 | 0.64 | 13.04 | **19.14x** | PASSED | 0.00e+00 |
| `blas` | 64 | All | 3.72 | 1.73 | 72.18 | **105.92x** | PASSED | 1.76e-02 |
| `blocked` | 16 | All | 58.02 | 3.61 | 4.63 | **6.79x** | PASSED | 0.00e+00 |
| `blocked` | 32 | All | 43.78 | 9.63 | 6.13 | **9.00x** | PASSED | 0.00e+00 |
| `blocked` | 64 | All | 34.98 | 4.08 | 7.67 | **11.26x** | PASSED | 0.00e+00 |
| `blocked` | 128 | All | 36.53 | 7.24 | 7.35 | **10.78x** | PASSED | 0.00e+00 |
| `blocked` | 256 | All | 41.41 | 3.07 | 6.48 | **9.51x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | 1 | 10.53 | 2.72 | 25.50 | **37.41x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | 2 | 6.92 | 2.83 | 38.79 | **56.92x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | 4 | 6.91 | 0.40 | 38.87 | **57.04x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | 8 | 5.87 | 0.52 | 45.76 | **67.14x** | PASSED | 0.00e+00 |


### Matrix Shape: 1024 x 1024 x 1024

| Implementation | Tile Size | Threads | Median Time (ms) | StdDev (ms) | GFLOP/s | Speedup | Correctness | Max Rel Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | 64 | All | 8518.77 | 1070.76 | 0.25 | **1.00x** | PASSED | 0.00e+00 |
| `cache_friendly` | 64 | All | 323.84 | 44.61 | 6.63 | **26.31x** | PASSED | 0.00e+00 |
| `blocked` | 64 | All | 236.04 | 35.15 | 9.10 | **36.09x** | PASSED | 0.00e+00 |
| `avx2` | 64 | All | 112.33 | 10.71 | 19.12 | **75.84x** | PASSED | 0.00e+00 |
| `openmp` | 64 | All | 105.57 | 12.42 | 20.34 | **80.69x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | All | 46.09 | 8.50 | 46.60 | **184.85x** | PASSED | 0.00e+00 |
| `blas` | 64 | All | 73.71 | 35.79 | 29.13 | **115.57x** | PASSED | 1.84e-01 |


### Matrix Shape: 1024 x 256 x 128

| Implementation | Tile Size | Threads | Median Time (ms) | StdDev (ms) | GFLOP/s | Speedup | Correctness | Max Rel Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | 64 | All | 110.58 | 14.64 | 0.61 | **1.00x** | PASSED | 0.00e+00 |
| `cache_friendly` | 64 | All | 9.36 | 3.39 | 7.17 | **11.82x** | PASSED | 0.00e+00 |
| `blocked` | 64 | All | 8.95 | 1.63 | 7.50 | **12.35x** | PASSED | 0.00e+00 |
| `avx2` | 64 | All | 4.25 | 1.27 | 15.79 | **26.01x** | PASSED | 0.00e+00 |
| `openmp` | 64 | All | 4.30 | 5.26 | 15.59 | **25.69x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | All | 1.71 | 0.61 | 39.26 | **64.69x** | PASSED | 0.00e+00 |
| `blas` | 64 | All | 5.12 | 7.80 | 13.12 | **21.62x** | PASSED | 2.95e-02 |


### Matrix Shape: 128 x 256 x 1024

| Implementation | Tile Size | Threads | Median Time (ms) | StdDev (ms) | GFLOP/s | Speedup | Correctness | Max Rel Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | 64 | All | 305.59 | 15.59 | 0.22 | **1.00x** | PASSED | 0.00e+00 |
| `cache_friendly` | 64 | All | 5.64 | 0.36 | 11.90 | **54.18x** | PASSED | 0.00e+00 |
| `blocked` | 64 | All | 6.72 | 1.00 | 9.99 | **45.49x** | PASSED | 0.00e+00 |
| `avx2` | 64 | All | 3.43 | 0.63 | 19.55 | **89.02x** | PASSED | 0.00e+00 |
| `openmp` | 64 | All | 7.94 | 1.64 | 8.46 | **38.50x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | All | 4.24 | 5.57 | 15.83 | **72.06x** | PASSED | 0.00e+00 |
| `blas` | 64 | All | 15.39 | 21.34 | 4.36 | **19.85x** | PASSED | 8.32e-04 |


### Matrix Shape: 389 x 457 x 521

| Implementation | Tile Size | Threads | Median Time (ms) | StdDev (ms) | GFLOP/s | Speedup | Correctness | Max Rel Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline` | 64 | All | 259.60 | 55.15 | 0.71 | **1.00x** | PASSED | 0.00e+00 |
| `cache_friendly` | 64 | All | 26.22 | 4.39 | 7.06 | **9.90x** | PASSED | 0.00e+00 |
| `blocked` | 64 | All | 26.25 | 5.34 | 7.06 | **9.89x** | PASSED | 0.00e+00 |
| `avx2` | 64 | All | 8.75 | 0.83 | 21.16 | **29.66x** | PASSED | 0.00e+00 |
| `openmp` | 64 | All | 14.40 | 1.54 | 12.87 | **18.03x** | PASSED | 0.00e+00 |
| `openmp_avx2` | 64 | All | 6.21 | 3.62 | 29.82 | **41.80x** | PASSED | 0.00e+00 |
| `blas` | 64 | All | 8.85 | 2.97 | 20.92 | **29.32x** | PASSED | 4.28e-02 |


## 2. Correctness & Numerical Validation

All benchmarked configurations **PASSED** numerical equivalence testing against the baseline reference with $rtol=10^{-3}$ and $atol=10^{-3}$.
