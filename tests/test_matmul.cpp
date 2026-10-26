#include "matrix/matrix.hpp"
#include "matrix/matmul.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <functional>
#include <cmath>

namespace {

int total_tests = 0;
int passed_tests = 0;
int failed_tests = 0;

void run_test(const std::string& name, const std::function<bool()>& test_fn) {
    ++total_tests;
    std::cout << "[ RUN      ] " << name << " ... " << std::flush;
    try {
        if (test_fn()) {
            ++passed_tests;
            std::cout << "PASSED\n";
        } else {
            ++failed_tests;
            std::cout << "FAILED\n";
        }
    } catch (const std::exception& e) {
        ++failed_tests;
        std::cout << "FAILED (exception: " << e.what() << ")\n";
    } catch (...) {
        ++failed_tests;
        std::cout << "FAILED (unknown exception)\n";
    }
}

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "\n    Assertion failed: " #cond " at line " << __LINE__ << " "; \
        return false; \
    } \
} while (0)

} // namespace

int main() {
    std::cout << "========================================================\n";
    std::cout << "        CPU Matrix Multiplication Test Suite            \n";
    std::cout << "========================================================\n";

    auto impls = matrix::get_available_matmul_types();

    // 1. Matrix Construction and Accessor Tests
    run_test("Matrix::BasicAllocationAndAccess", []() {
        matrix::Matrix m(3, 4, 1.5f);
        TEST_ASSERT(m.rows() == 3);
        TEST_ASSERT(m.cols() == 4);
        TEST_ASSERT(m.size() == 12);
        TEST_ASSERT(!m.empty());
        TEST_ASSERT(std::fabs(m(0, 0) - 1.5f) < 1e-6f);
        TEST_ASSERT(std::fabs(m(2, 3) - 1.5f) < 1e-6f);
        m(1, 2) = 42.0f;
        TEST_ASSERT(std::fabs(m(1, 2) - 42.0f) < 1e-6f);
        TEST_ASSERT(std::fabs(m.at(1, 2) - 42.0f) < 1e-6f);
        return true;
    });

    // 2. Matrix Copy and Move Semantics
    run_test("Matrix::CopyAndMoveSemantics", []() {
        matrix::Matrix a(4, 4, 2.0f);
        matrix::Matrix b = a; // Copy constructor
        TEST_ASSERT(b.rows() == 4 && b.cols() == 4);
        TEST_ASSERT(b(1, 1) == 2.0f);
        b(1, 1) = 99.0f;
        TEST_ASSERT(a(1, 1) == 2.0f); // Deep copy verification

        matrix::Matrix c = std::move(b); // Move constructor
        TEST_ASSERT(c.rows() == 4 && c.cols() == 4);
        TEST_ASSERT(c(1, 1) == 99.0f);
        TEST_ASSERT(b.rows() == 0 && b.data() == nullptr); // Moved-from is empty
        return true;
    });

    // 3. Matrix Bounds Checking
    run_test("Matrix::BoundsCheckingThrows", []() {
        matrix::Matrix m(2, 2, 0.0f);
        bool caught = false;
        try {
            m.at(2, 0); // Out of bounds row
        } catch (const std::out_of_range&) {
            caught = true;
        }
        TEST_ASSERT(caught);

        caught = false;
        try {
            m.at(0, 5); // Out of bounds col
        } catch (const std::out_of_range&) {
            caught = true;
        }
        TEST_ASSERT(caught);
        return true;
    });

    // 4. Incompatible Dimensions Validation
    run_test("Matmul::DimensionValidationThrows", []() {
        matrix::Matrix A(3, 4);
        matrix::Matrix B(5, 6); // A.cols() = 4 != B.rows() = 5
        matrix::Matrix C(3, 6);
        bool caught = false;
        try {
            matrix::matmul_baseline(A, B, C);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        TEST_ASSERT(caught);

        // Wrong output matrix dimensions
        matrix::Matrix B_good(4, 6);
        matrix::Matrix C_wrong(2, 6); // Should be 3x6
        caught = false;
        try {
            matrix::matmul_baseline(A, B_good, C_wrong);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        TEST_ASSERT(caught);
        return true;
    });

    // 5. Identity Matrix Multiplication
    for (auto impl : impls) {
        std::string name = "Matmul::IdentityMatrix[" + matrix::matmul_type_to_string(impl) + "]";
        run_test(name, [impl]() {
            const size_t N = 32;
            matrix::Matrix A = matrix::Matrix::random(N, N, -5.0f, 5.0f, 101);
            matrix::Matrix I = matrix::Matrix::identity(N);
            matrix::Matrix C(N, N);

            matrix::MatmulOptions opts;
            opts.tile_m = 16;
            opts.tile_k = 16;
            opts.tile_n = 16;
            matrix::matmul(A, I, C, impl, opts);

            float max_abs_diff = 0.0f;
            TEST_ASSERT(A.is_approx_equal(C, 1e-5f, 1e-5f, &max_abs_diff));
            return true;
        });
    }

    // 6. Zero Matrix Multiplication
    for (auto impl : impls) {
        std::string name = "Matmul::ZeroMatrix[" + matrix::matmul_type_to_string(impl) + "]";
        run_test(name, [impl]() {
            const size_t M = 24, K = 32, N = 16;
            matrix::Matrix A = matrix::Matrix::random(M, K, -2.0f, 2.0f, 202);
            matrix::Matrix Zero = matrix::Matrix::zeros(K, N);
            matrix::Matrix C(M, N);

            matrix::MatmulOptions opts;
            matrix::matmul(A, Zero, C, impl, opts);

            for (size_t i = 0; i < C.size(); ++i) {
                TEST_ASSERT(std::fabs(C.data()[i]) < 1e-7f);
            }
            return true;
        });
    }

    // 7. Small Matrices (1x1, 2x2, 3x3, 4x4, 7x7)
    for (auto impl : impls) {
        std::string name = "Matmul::SmallDimensions[" + matrix::matmul_type_to_string(impl) + "]";
        run_test(name, [impl]() {
            std::vector<size_t> sizes = {1, 2, 3, 4, 7};
            for (size_t s : sizes) {
                matrix::Matrix A = matrix::Matrix::random(s, s, -1.0f, 1.0f, static_cast<uint32_t>(s * 7));
                matrix::Matrix B = matrix::Matrix::random(s, s, -1.0f, 1.0f, static_cast<uint32_t>(s * 13));
                matrix::Matrix C_ref(s, s);
                matrix::Matrix C_test(s, s);

                matrix::matmul_baseline(A, B, C_ref);
                matrix::matmul(A, B, C_test, impl);

                TEST_ASSERT(C_ref.is_approx_equal(C_test, 1e-4f, 1e-4f));
            }
            return true;
        });
    }

    // 8. Non-Square / Rectangular Matrices (M != K != N)
    for (auto impl : impls) {
        std::string name = "Matmul::RectangularMatrices[" + matrix::matmul_type_to_string(impl) + "]";
        run_test(name, [impl]() {
            struct Case { size_t M, K, N; };
            std::vector<Case> cases = {
                {1, 128, 1},    // Vector dot product
                {1, 64, 32},    // Row vector x Matrix
                {32, 64, 1},    // Matrix x Column vector
                {37, 73, 59},   // Prime rectangular dimensions
                {128, 32, 64},  // M > K < N
                {48, 128, 36}   // M < K > N
            };

            for (const auto& c : cases) {
                matrix::Matrix A = matrix::Matrix::random(c.M, c.K, -2.0f, 2.0f, 303);
                matrix::Matrix B = matrix::Matrix::random(c.K, c.N, -2.0f, 2.0f, 404);
                matrix::Matrix C_ref(c.M, c.N);
                matrix::Matrix C_test(c.M, c.N);

                matrix::matmul_baseline(A, B, C_ref);
                matrix::matmul(A, B, C_test, impl);

                float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
                bool ok = C_ref.is_approx_equal(C_test, 1e-3f, 1e-3f, &max_abs_diff, &max_rel_diff);
                TEST_ASSERT(ok);
            }
            return true;
        });
    }

    // 9. Non-Divisible by Tile Size (boundary edge cases)
    for (auto impl : impls) {
        std::string name = "Matmul::NonDivisibleTileSize[" + matrix::matmul_type_to_string(impl) + "]";
        run_test(name, [impl]() {
            // Tile size = 64, dimensions 65, 67, 71 (just past tile boundary)
            const size_t M = 65, K = 67, N = 71;
            matrix::Matrix A = matrix::Matrix::random(M, K, -1.0f, 1.0f, 505);
            matrix::Matrix B = matrix::Matrix::random(K, N, -1.0f, 1.0f, 606);
            matrix::Matrix C_ref(M, N);
            matrix::Matrix C_test(M, N);

            matrix::MatmulOptions opts;
            opts.tile_m = 64;
            opts.tile_k = 64;
            opts.tile_n = 64;

            matrix::matmul_baseline(A, B, C_ref);
            matrix::matmul(A, B, C_test, impl, opts);

            float max_abs_diff = 0.0f;
            TEST_ASSERT(C_ref.is_approx_equal(C_test, 1e-3f, 1e-3f, &max_abs_diff));
            return true;
        });
    }

    // 10. Medium Square Matrices (256x256)
    for (auto impl : impls) {
        std::string name = "Matmul::MediumSquare256x256[" + matrix::matmul_type_to_string(impl) + "]";
        run_test(name, [impl]() {
            const size_t N = 256;
            matrix::Matrix A = matrix::Matrix::random(N, N, -1.0f, 1.0f, 707);
            matrix::Matrix B = matrix::Matrix::random(N, N, -1.0f, 1.0f, 808);
            matrix::Matrix C_ref(N, N);
            matrix::Matrix C_test(N, N);

            matrix::matmul_baseline(A, B, C_ref);
            matrix::matmul(A, B, C_test, impl);

            float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
            bool ok = C_ref.is_approx_equal(C_test, 1e-3f, 1e-3f, &max_abs_diff, &max_rel_diff);
            TEST_ASSERT(ok);
            return true;
        });
    }

    std::cout << "========================================================\n";
    std::cout << "Test Summary: " << passed_tests << " / " << total_tests << " passed ("
              << failed_tests << " failed)\n";
    std::cout << "========================================================\n";

    return (failed_tests == 0) ? 0 : 1;
}
