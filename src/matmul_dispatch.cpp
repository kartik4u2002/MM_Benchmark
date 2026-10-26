#include "matrix/matmul.hpp"
#include <algorithm>
#include <cctype>

namespace matrix {

std::string matmul_type_to_string(MatmulType type) {
    switch (type) {
        case MatmulType::Baseline: return "baseline";
        case MatmulType::CacheFriendly: return "cache_friendly";
        case MatmulType::Blocked: return "blocked";
        case MatmulType::AVX2: return "avx2";
        case MatmulType::OpenMP: return "openmp";
        case MatmulType::OpenMP_AVX2: return "openmp_avx2";
        case MatmulType::BLAS: return "blas";
        default: return "unknown";
    }
}

MatmulType string_to_matmul_type(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    if (lower == "baseline" || lower == "naive" || lower == "ijk") return MatmulType::Baseline;
    if (lower == "cache_friendly" || lower == "loop_reordered" || lower == "ikj") return MatmulType::CacheFriendly;
    if (lower == "blocked" || lower == "tiled") return MatmulType::Blocked;
    if (lower == "avx2" || lower == "simd") return MatmulType::AVX2;
    if (lower == "openmp" || lower == "omp") return MatmulType::OpenMP;
    if (lower == "openmp_avx2" || lower == "omp_avx2") return MatmulType::OpenMP_AVX2;
    if (lower == "blas" || lower == "openblas") return MatmulType::BLAS;

    throw std::invalid_argument("Unknown matmul implementation name: " + name);
}

std::vector<MatmulType> get_available_matmul_types() {
    std::vector<MatmulType> types = {
        MatmulType::Baseline,
        MatmulType::CacheFriendly,
        MatmulType::Blocked,
        MatmulType::AVX2,
        MatmulType::OpenMP,
        MatmulType::OpenMP_AVX2
    };
    if (is_blas_available()) {
        types.push_back(MatmulType::BLAS);
    }
    return types;
}

void matmul(const Matrix& A, const Matrix& B, Matrix& C,
            MatmulType type, const MatmulOptions& options) {
    switch (type) {
        case MatmulType::Baseline:
            matmul_baseline(A, B, C);
            break;
        case MatmulType::CacheFriendly:
            matmul_cache_friendly(A, B, C);
            break;
        case MatmulType::Blocked:
            matmul_blocked(A, B, C, options.tile_m, options.tile_k, options.tile_n);
            break;
        case MatmulType::AVX2:
            matmul_avx2(A, B, C, options.tile_m);
            break;
        case MatmulType::OpenMP:
            matmul_openmp(A, B, C, options.num_threads, options.tile_m);
            break;
        case MatmulType::OpenMP_AVX2:
            matmul_openmp_avx2(A, B, C, options.num_threads, options.tile_m);
            break;
        case MatmulType::BLAS:
            matmul_blas(A, B, C);
            break;
        default:
            throw std::invalid_argument("Unsupported matmul type");
    }
}

} // namespace matrix
