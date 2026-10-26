#include "matrix/matmul.hpp"
#include <algorithm>

namespace matrix {

/**
 * @brief Cache-blocked (tiled) dense matrix multiplication.
 * 
 * Partitions the M x K x N iteration space into smaller sub-blocks (tiles) of size
 * tile_m x tile_k x tile_n.
 * 
 * Architectural benefits:
 * When matrices exceed the size of the CPU L1 and L2 caches (e.g. 1024x1024 float is 4 MB),
 * a standard pass sweeps data out of the cache before it can be reused.
 * By operating on small submatrices (e.g. 64x64 elements = 16 KB), both tile A and tile B
 * remain resident in high-speed L1/L2 data cache during the entire computation of the block,
 * dramatically reducing DRAM bandwidth consumption.
 * 
 * Boundary handling:
 * Matrix dimensions that are not divisible by tile sizes are safely clamped using std::min,
 * preventing buffer overruns and eliminating padding requirements.
 */
void matmul_blocked(const Matrix& A, const Matrix& B, Matrix& C,
                    size_t tile_m, size_t tile_k, size_t tile_n) {
    validate_matmul_dimensions(A, B, C);

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    // Default tile sizes if 0 passed
    if (tile_m == 0) tile_m = 64;
    if (tile_k == 0) tile_k = 64;
    if (tile_n == 0) tile_n = 64;

    C.set_zero();

    const float* a_data = A.data();
    const float* b_data = B.data();
    float* c_data = C.data();

    // Outer loops iterate over blocks
    for (size_t ii = 0; ii < M; ii += tile_m) {
        const size_t i_max = std::min(ii + tile_m, M);

        for (size_t kk = 0; kk < K; kk += tile_k) {
            const size_t k_max = std::min(kk + tile_k, K);

            for (size_t jj = 0; jj < N; jj += tile_n) {
                const size_t j_max = std::min(jj + tile_n, N);

                // Inner loops compute the submatrix multiplication (i-k-j order for locality)
                for (size_t i = ii; i < i_max; ++i) {
                    const float* a_row = a_data + (i * K);
                    float* c_row = c_data + (i * N);

                    for (size_t k = kk; k < k_max; ++k) {
                        const float a_ik = a_row[k];
                        const float* b_row = b_data + (k * N);

                        for (size_t j = jj; j < j_max; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

} // namespace matrix
