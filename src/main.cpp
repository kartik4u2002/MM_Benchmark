#include "matrix/matrix.hpp"
#include "matrix/matmul.hpp"
#include "matrix/cpu_features.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::cout << "========================================================\n";
    std::cout << "      CPU Matrix Multiplication Accelerator (Demo)     \n";
    std::cout << "========================================================\n\n";

    // 1. Detect and print host CPU features
    const auto& cpu = matrix::CpuFeatures::get();
    cpu.print(std::cout);
    std::cout << "BLAS Support: " << (matrix::is_blas_available() ? "Available (linked)" : "Not compiled") << "\n\n";

    // 2. Demonstration matrix dimensions
    const size_t M = (argc > 1) ? std::stoull(argv[1]) : 512;
    const size_t K = (argc > 2) ? std::stoull(argv[2]) : 512;
    const size_t N = (argc > 3) ? std::stoull(argv[3]) : 512;

    std::cout << "Generating test matrices: A(" << M << "x" << K << "), B(" << K << "x" << N << ")...\n";
    matrix::Matrix A = matrix::Matrix::random(M, K, -1.0f, 1.0f, 12345);
    matrix::Matrix B = matrix::Matrix::random(K, N, -1.0f, 1.0f, 67890);
    matrix::Matrix C_ref(M, N);

    const double total_flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);

    // Baseline reference
    std::cout << "\nRunning Baseline (i-j-k) for reference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    matrix::matmul_baseline(A, B, C_ref);
    auto t1 = std::chrono::high_resolution_clock::now();
    double baseline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double baseline_gflops = (total_flops / (baseline_ms * 1e-3)) / 1e9;
    std::cout << "  Baseline time: " << std::fixed << std::setprecision(2) << baseline_ms << " ms ("
              << baseline_gflops << " GFLOP/s)\n\n";

    // Test each implementation
    auto implementations = matrix::get_available_matmul_types();
    std::cout << std::left << std::setw(16) << "Implementation"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GFLOP/s"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "Correctness"
              << "Max Rel Diff\n";
    std::cout << std::string(70, '-') << "\n";

    for (auto type : implementations) {
        matrix::Matrix C_test(M, N);
        matrix::MatmulOptions opts;
        opts.tile_m = 64;
        opts.tile_k = 64;
        opts.tile_n = 64;

        // Warm up
        matrix::matmul(A, B, C_test, type, opts);

        // Timed run
        t0 = std::chrono::high_resolution_clock::now();
        matrix::matmul(A, B, C_test, type, opts);
        t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gflops = (total_flops / (ms * 1e-3)) / 1e9;
        double speedup = baseline_ms / ms;

        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        bool pass = C_ref.is_approx_equal(C_test, 1e-3f, 1e-3f, &max_abs_diff, &max_rel_diff);

        std::cout << std::left << std::setw(16) << matrix::matmul_type_to_string(type)
                  << std::setw(12) << std::fixed << std::setprecision(2) << ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << gflops
                  << std::setw(11) << (std::to_string(static_cast<int>(speedup)) + "." + std::to_string(static_cast<int>(speedup * 10) % 10) + "x ")
                  << std::setw(14) << (pass ? "PASSED" : "FAILED")
                  << std::scientific << std::setprecision(2) << max_rel_diff << "\n";
    }

    std::cout << std::string(70, '-') << "\n";
    std::cout << "Done! For full benchmarks, run `matmul_benchmark`.\n";
    return 0;
}
