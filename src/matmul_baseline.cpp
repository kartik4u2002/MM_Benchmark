#include "matrix/matmul.hpp"

namespace matrix {

/**
 * @brief Baseline dense matrix multiplication (i-j-k order).
 * 
 * Computes C = A * B using the standard textbook triple loop.
 * A is M x K, B is K x N, C is M x N.
 * 
 * Performance characteristic:
 * The innermost loop iterates over k, where B(k, j) is accessed with stride N (row stride).
 * For large N, each access jumps to a different cache line, causing high L1/L2 cache miss
 * rates (cache thrashing) and poor CPU execution port utilization.
 */
void matmul_baseline(const Matrix& A, const Matrix& B, Matrix& C) {
    validate_matmul_dimensions(A, B, C);

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

} // namespace matrix
