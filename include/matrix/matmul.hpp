#pragma once

#include "matrix/matrix.hpp"
#include <string>
#include <vector>
#include <stdexcept>

namespace matrix {

enum class MatmulType {
    Baseline,        // Standard naive i-j-k loop
    CacheFriendly,   // Loop reordered i-k-j for unit-stride access
    Blocked,         // Cache tiled (configurable tile size)
    AVX2,            // SIMD AVX2 + FMA vectorized micro-kernel
    OpenMP,          // Multithreaded tiled matrix multiplication
    OpenMP_AVX2,     // Multithreaded tiled + AVX2 vectorized
    BLAS             // OpenBLAS / vendor BLAS if linked
};

struct MatmulOptions {
    size_t tile_m = 64;
    size_t tile_k = 64;
    size_t tile_n = 64;
    int num_threads = 0; // 0 = use OpenMP default (all available logical cores)
};

/**
 * @brief Validates input and output dimensions for C = A * B.
 * 
 * A: M x K, B: K x N, C: M x N.
 * Throws std::invalid_argument if dimensions are incompatible.
 */
inline void validate_matmul_dimensions(const Matrix& A, const Matrix& B, const Matrix& C) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication: A is " +
                                    std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                                    ", B is " + std::to_string(B.rows()) + "x" + std::to_string(B.cols()));
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument("Output matrix C dimensions (" +
                                    std::to_string(C.rows()) + "x" + std::to_string(C.cols()) +
                                    ") do not match expected (" +
                                    std::to_string(A.rows()) + "x" + std::to_string(B.cols()) + ")");
    }
}

// Function declarations for each implementation
void matmul_baseline(const Matrix& A, const Matrix& B, Matrix& C);
void matmul_cache_friendly(const Matrix& A, const Matrix& B, Matrix& C);
void matmul_blocked(const Matrix& A, const Matrix& B, Matrix& C,
                    size_t tile_m = 64, size_t tile_k = 64, size_t tile_n = 64);
void matmul_avx2(const Matrix& A, const Matrix& B, Matrix& C, size_t tile_size = 64);
void matmul_openmp(const Matrix& A, const Matrix& B, Matrix& C, int num_threads = 0, size_t tile_size = 64);
void matmul_openmp_avx2(const Matrix& A, const Matrix& B, Matrix& C, int num_threads = 0, size_t tile_size = 64);
void matmul_blas(const Matrix& A, const Matrix& B, Matrix& C);
bool is_blas_available() noexcept;

// Unified dispatch API
void matmul(const Matrix& A, const Matrix& B, Matrix& C,
            MatmulType type = MatmulType::CacheFriendly,
            const MatmulOptions& options = {});

// Utility functions
std::string matmul_type_to_string(MatmulType type);
MatmulType string_to_matmul_type(const std::string& name);
std::vector<MatmulType> get_available_matmul_types();

} // namespace matrix
