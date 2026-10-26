#include "matrix/matmul.hpp"
#include "matrix/cpu_features.hpp"
#include <algorithm>

#if defined(_MSC_VER)
    #include <immintrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
#endif

namespace matrix {

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,fma")))
#endif
static void matmul_avx2_kernel(const Matrix& A, const Matrix& B, Matrix& C, size_t tile_size) {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    C.set_zero();

    const float* a_data = A.data();
    const float* b_data = B.data();
    float* c_data = C.data();

    const size_t tile_m = tile_size;
    const size_t tile_k = tile_size;
    const size_t tile_n = tile_size;

    for (size_t ii = 0; ii < M; ii += tile_m) {
        const size_t i_max = std::min(ii + tile_m, M);

        for (size_t kk = 0; kk < K; kk += tile_k) {
            const size_t k_max = std::min(kk + tile_k, K);

            for (size_t jj = 0; jj < N; jj += tile_n) {
                const size_t j_max = std::min(jj + tile_n, N);

                // Unroll rows by 4 where possible
                size_t i = ii;
                for (; i + 4 <= i_max; i += 4) {
                    const float* a0 = a_data + (i + 0) * K;
                    const float* a1 = a_data + (i + 1) * K;
                    const float* a2 = a_data + (i + 2) * K;
                    const float* a3 = a_data + (i + 3) * K;

                    float* c0 = c_data + (i + 0) * N;
                    float* c1 = c_data + (i + 1) * N;
                    float* c2 = c_data + (i + 2) * N;
                    float* c3 = c_data + (i + 3) * N;

                    // Inner vector loop in steps of 8 floats (256-bit YMM register)
                    size_t j = jj;
                    for (; j + 8 <= j_max; j += 8) {
                        __m256 c0_vec = _mm256_loadu_ps(c0 + j);
                        __m256 c1_vec = _mm256_loadu_ps(c1 + j);
                        __m256 c2_vec = _mm256_loadu_ps(c2 + j);
                        __m256 c3_vec = _mm256_loadu_ps(c3 + j);

                        for (size_t k = kk; k < k_max; ++k) {
                            __m256 b_vec = _mm256_loadu_ps(b_data + k * N + j);

                            __m256 a0_val = _mm256_set1_ps(a0[k]);
                            __m256 a1_val = _mm256_set1_ps(a1[k]);
                            __m256 a2_val = _mm256_set1_ps(a2[k]);
                            __m256 a3_val = _mm256_set1_ps(a3[k]);

                            c0_vec = _mm256_fmadd_ps(a0_val, b_vec, c0_vec);
                            c1_vec = _mm256_fmadd_ps(a1_val, b_vec, c1_vec);
                            c2_vec = _mm256_fmadd_ps(a2_val, b_vec, c2_vec);
                            c3_vec = _mm256_fmadd_ps(a3_val, b_vec, c3_vec);
                        }

                        _mm256_storeu_ps(c0 + j, c0_vec);
                        _mm256_storeu_ps(c1 + j, c1_vec);
                        _mm256_storeu_ps(c2 + j, c2_vec);
                        _mm256_storeu_ps(c3 + j, c3_vec);
                    }

                    // Remainder columns (< 8 elements)
                    for (; j < j_max; ++j) {
                        for (size_t k = kk; k < k_max; ++k) {
                            const float b_kj = b_data[k * N + j];
                            c0[j] += a0[k] * b_kj;
                            c1[j] += a1[k] * b_kj;
                            c2[j] += a2[k] * b_kj;
                            c3[j] += a3[k] * b_kj;
                        }
                    }
                }

                // Remainder rows (< 4 rows)
                for (; i < i_max; ++i) {
                    const float* a_row = a_data + i * K;
                    float* c_row = c_data + i * N;

                    size_t j = jj;
                    for (; j + 8 <= j_max; j += 8) {
                        __m256 c_vec = _mm256_loadu_ps(c_row + j);
                        for (size_t k = kk; k < k_max; ++k) {
                            __m256 b_vec = _mm256_loadu_ps(b_data + k * N + j);
                            __m256 a_val = _mm256_set1_ps(a_row[k]);
                            c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);
                        }
                        _mm256_storeu_ps(c_row + j, c_vec);
                    }

                    for (; j < j_max; ++j) {
                        for (size_t k = kk; k < k_max; ++k) {
                            c_row[j] += a_row[k] * b_data[k * N + j];
                        }
                    }
                }
            }
        }
    }
#else
    // Fallback if not on x86/x64
    matmul_blocked(A, B, C, tile_size, tile_size, tile_size);
#endif
}

/**
 * @brief Explicit AVX2 + FMA SIMD vectorized matrix multiplication.
 * 
 * Uses runtime CPUID detection to guarantee safety. If the host CPU lacks AVX2
 * or FMA support, gracefully falls back to matmul_blocked without crashing.
 */
void matmul_avx2(const Matrix& A, const Matrix& B, Matrix& C, size_t tile_size) {
    validate_matmul_dimensions(A, B, C);

    const auto& cpu = CpuFeatures::get();
    if (cpu.avx2 && cpu.fma) {
        matmul_avx2_kernel(A, B, C, tile_size);
    } else {
        std::cerr << "[Warning] AVX2 or FMA not supported on this host CPU. Falling back to blocked GEMM.\n";
        matmul_blocked(A, B, C, tile_size, tile_size, tile_size);
    }
}

} // namespace matrix
