#include "matrix/matmul.hpp"

namespace matrix {

/**
 * @brief Cache-friendly dense matrix multiplication using i-k-j loop reordering.
 * 
 * In this reordered algorithm:
 * - A(i, k) is read once in the middle loop and kept in a CPU register as a scalar.
 * - In the innermost loop over j:
 *     C(i, j) += A(i, k) * B(k, j);
 *   Both B(k, j) and C(i, j) are accessed with unit stride (step of 1 element = 4 bytes).
 * 
 * Architectural benefits:
 * 1. Unit-stride accesses allow the CPU L1 data cache and hardware stream prefetchers
 *    to pre-load subsequent cache lines into L1d ahead of time.
 * 2. Spatial locality is nearly 100% efficient (all 16 floats in a 64-byte cache line
 *    are utilized before moving to the next line).
 * 3. Modern compilers (GCC, Clang, MSVC) can easily auto-vectorize the inner loop
 *    into SIMD instructions (SSE/AVX) because operations on C and B are contiguous.
 */
void matmul_cache_friendly(const Matrix& A, const Matrix& B, Matrix& C) {
    validate_matmul_dimensions(A, B, C);

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    C.set_zero();

    const float* a_data = A.data();
    const float* b_data = B.data();
    float* c_data = C.data();

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = a_data + (i * K);
        float* c_row = c_data + (i * N);

        for (size_t k = 0; k < K; ++k) {
            const float a_ik = a_row[k];
            const float* b_row = b_data + (k * N);

            // Inner loop traverses B and C contiguously
            for (size_t j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

} // namespace matrix
