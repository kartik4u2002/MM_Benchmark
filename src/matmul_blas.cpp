#include "matrix/matmul.hpp"
#include <iostream>

#if defined(HAVE_BLAS) || defined(USE_BLAS)
    #if defined(__APPLE__)
        #include <Accelerate/Accelerate.h>
    #elif defined(OPENBLAS)
        #include <cblas.h>
    #else
        #if __has_include(<cblas.h>)
            #include <cblas.h>
        #elif __has_include(<openblas/cblas.h>)
            #include <openblas/cblas.h>
        #endif
    #endif
#endif

namespace matrix {

bool is_blas_available() noexcept {
#if defined(HAVE_BLAS) || defined(USE_BLAS)
    return true;
#else
    return false;
#endif
}

/**
 * @brief BLAS-backed matrix multiplication (using cblas_sgemm).
 * 
 * Invokes an optimized vendor BLAS library (e.g. OpenBLAS, MKL, or Apple Accelerate)
 * when configured and linked at build time.
 */
void matmul_blas(const Matrix& A, const Matrix& B, Matrix& C) {
    validate_matmul_dimensions(A, B, C);

#if defined(HAVE_BLAS) || defined(USE_BLAS)
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    // C = 1.0 * A * B + 0.0 * C
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        M, N, K,
        1.0f,
        A.data(), K,
        B.data(), N,
        0.0f,
        C.data(), N
    );
#else
    std::cerr << "[Warning] BLAS support was not compiled in this build. Falling back to AVX2/Blocked GEMM.\n";
    matmul_avx2(A, B, C);
#endif
}

} // namespace matrix
