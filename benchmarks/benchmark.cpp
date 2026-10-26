#include "matrix/matrix.hpp"
#include "matrix/matmul.hpp"
#include "matrix/cpu_features.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace {

struct BenchmarkConfig {
    size_t M = 512;
    size_t K = 512;
    size_t N = 512;
    size_t warmup_iters = 3;
    size_t measured_iters = 10;
    uint32_t seed = 42;
    std::string impl_name = "all";
    size_t tile_size = 64;
    int num_threads = 0;
    std::string output_csv = "";
    std::string output_json = "";
    bool verbose = false;
};

struct BenchmarkRecord {
    std::string impl_name;
    size_t M, K, N;
    size_t tile_size;
    int num_threads;
    size_t warmup_iters;
    size_t measured_iters;
    double median_ms;
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    double gflops;
    double speedup;
    bool passed_correctness;
    float max_abs_diff;
    float max_rel_diff;
    std::string cpu_model;
    std::string compiler_info;
};

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --M <int>             Number of rows in matrix A and C (default: 512)\n"
              << "  --K <int>             Number of cols in A / rows in B (default: 512)\n"
              << "  --N <int>             Number of cols in B and C (default: 512)\n"
              << "  --warmup <int>        Number of warmup iterations (default: 3)\n"
              << "  --iterations <int>    Number of measured iterations (default: 10)\n"
              << "  --seed <int>          Random seed for matrix data (default: 42)\n"
              << "  --impl <string>       Implementation ('all', 'baseline', 'cache_friendly',\n"
              << "                        'blocked', 'avx2', 'openmp', 'openmp_avx2', 'blas')\n"
              << "  --tile-size <int>     Tile size for blocked/avx2/openmp (default: 64)\n"
              << "  --threads <int>       Thread count for OpenMP (0 = default/all cores)\n"
              << "  --output-csv <path>   Export results to CSV file\n"
              << "  --output-json <path>  Export results to JSON file\n"
              << "  --verbose             Enable detailed per-iteration output\n"
              << "  --help                Show this help message\n";
}

BenchmarkConfig parse_args(int argc, char* argv[]) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--M" && i + 1 < argc) config.M = std::stoull(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) config.K = std::stoull(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) config.N = std::stoull(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) config.warmup_iters = std::stoull(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) config.measured_iters = std::stoull(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) config.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--impl" && i + 1 < argc) config.impl_name = argv[++i];
        else if (arg == "--tile-size" && i + 1 < argc) config.tile_size = std::stoull(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) config.num_threads = std::stoi(argv[++i]);
        else if (arg == "--output-csv" && i + 1 < argc) config.output_csv = argv[++i];
        else if (arg == "--output-json" && i + 1 < argc) config.output_json = argv[++i];
        else if (arg == "--verbose") config.verbose = true;
        else if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help(argv[0]);
            std::exit(1);
        }
    }
    return config;
}

std::string get_compiler_info() {
#if defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "Unknown Compiler";
#endif
}

} // namespace

int main(int argc, char* argv[]) {
    BenchmarkConfig cfg = parse_args(argc, argv);

    const auto& cpu = matrix::CpuFeatures::get();
    std::string compiler = get_compiler_info();
    std::string cpu_model = cpu.brand_string.empty() ? "x86_64 CPU" : cpu.brand_string;

    std::cout << "=======================================================================\n";
    std::cout << "             CPU Matrix Multiplication Benchmark Suite                 \n";
    std::cout << "=======================================================================\n";
    std::cout << "CPU:           " << cpu_model << "\n";
    std::cout << "Compiler:      " << compiler << "\n";
    std::cout << "Dimensions:    M=" << cfg.M << ", K=" << cfg.K << ", N=" << cfg.N << "\n";
    std::cout << "Tile Size:     " << cfg.tile_size << "\n";
    std::cout << "Threads:       " << (cfg.num_threads > 0 ? std::to_string(cfg.num_threads) : "Auto/All") << "\n";
    std::cout << "Iterations:    " << cfg.warmup_iters << " warmup, " << cfg.measured_iters << " measured\n";
    std::cout << "Operations:    " << std::fixed << std::setprecision(3)
              << (2.0 * cfg.M * cfg.K * cfg.N / 1e9) << " GFLOP per multiplication\n";
    std::cout << "=======================================================================\n\n";

    // Allocate identical inputs
    std::cout << "Generating input matrices with seed=" << cfg.seed << "...\n";
    matrix::Matrix A = matrix::Matrix::random(cfg.M, cfg.K, -1.0f, 1.0f, cfg.seed);
    matrix::Matrix B = matrix::Matrix::random(cfg.K, cfg.N, -1.0f, 1.0f, cfg.seed + 1);

    // Baseline reference for correctness
    matrix::Matrix C_ref(cfg.M, cfg.N);
    std::cout << "Computing reference solution using baseline i-j-k...\n";
    matrix::matmul_baseline(A, B, C_ref);

    // Determine implementations to benchmark
    std::vector<matrix::MatmulType> impls_to_run;
    if (cfg.impl_name == "all") {
        impls_to_run = matrix::get_available_matmul_types();
    } else {
        impls_to_run.push_back(matrix::string_to_matmul_type(cfg.impl_name));
    }

    const double total_flops = 2.0 * static_cast<double>(cfg.M) * static_cast<double>(cfg.K) * static_cast<double>(cfg.N);
    double baseline_median_ms = 1.0; // placeholder for speedup computation

    std::vector<BenchmarkRecord> records;

    std::cout << "\nStarting benchmark runs...\n";
    std::cout << std::left << std::setw(15) << "Implementation"
              << std::setw(12) << "Median (ms)"
              << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "StdDev (ms)"
              << std::setw(12) << "GFLOP/s"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Status"
              << "Max Rel Diff\n";
    std::cout << std::string(85, '-') << "\n";

    for (auto impl_type : impls_to_run) {
        std::string name = matrix::matmul_type_to_string(impl_type);
        matrix::Matrix C_test(cfg.M, cfg.N);

        matrix::MatmulOptions opts;
        opts.tile_m = cfg.tile_size;
        opts.tile_k = cfg.tile_size;
        opts.tile_n = cfg.tile_size;
        opts.num_threads = cfg.num_threads;

        // 1. Warm-up runs
        for (size_t w = 0; w < cfg.warmup_iters; ++w) {
            matrix::matmul(A, B, C_test, impl_type, opts);
        }

        // 2. Timed runs
        std::vector<double> timings_ms;
        timings_ms.reserve(cfg.measured_iters);

        for (size_t it = 0; it < cfg.measured_iters; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            matrix::matmul(A, B, C_test, impl_type, opts);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            timings_ms.push_back(elapsed_ms);
            if (cfg.verbose) {
                std::cout << "  [" << name << "] iter " << it + 1 << ": " << elapsed_ms << " ms\n";
            }
        }

        // 3. Statistics
        std::vector<double> sorted = timings_ms;
        std::sort(sorted.begin(), sorted.end());
        double median_ms = (sorted.size() % 2 == 1)
            ? sorted[sorted.size() / 2]
            : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);

        double sum = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
        double mean_ms = sum / timings_ms.size();

        double sq_sum = 0.0;
        for (double t : timings_ms) {
            sq_sum += (t - mean_ms) * (t - mean_ms);
        }
        double stddev_ms = std::sqrt(sq_sum / timings_ms.size());
        double min_ms = sorted.front();
        double max_ms = sorted.back();

        double gflops = (total_flops / (median_ms * 1e-3)) / 1e9;

        if (impl_type == matrix::MatmulType::Baseline) {
            baseline_median_ms = median_ms;
        }
        double speedup = baseline_median_ms / median_ms;

        // 4. Correctness check against reference
        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        // Float tolerance: 1e-3 accommodates operation reordering and FMA differences
        bool passed = C_ref.is_approx_equal(C_test, 1e-3f, 1e-3f, &max_abs_diff, &max_rel_diff);

        BenchmarkRecord rec;
        rec.impl_name = name;
        rec.M = cfg.M;
        rec.K = cfg.K;
        rec.N = cfg.N;
        rec.tile_size = cfg.tile_size;
        rec.num_threads = cfg.num_threads;
        rec.warmup_iters = cfg.warmup_iters;
        rec.measured_iters = cfg.measured_iters;
        rec.median_ms = median_ms;
        rec.mean_ms = mean_ms;
        rec.stddev_ms = stddev_ms;
        rec.min_ms = min_ms;
        rec.max_ms = max_ms;
        rec.gflops = gflops;
        rec.speedup = speedup;
        rec.passed_correctness = passed;
        rec.max_abs_diff = max_abs_diff;
        rec.max_rel_diff = max_rel_diff;
        rec.cpu_model = cpu_model;
        rec.compiler_info = compiler;

        records.push_back(rec);

        std::cout << std::left << std::setw(15) << name
                  << std::setw(12) << std::fixed << std::setprecision(2) << median_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << mean_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << stddev_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << gflops
                  << std::setw(10) << (std::to_string(static_cast<int>(speedup)) + "." + std::to_string(static_cast<int>(speedup * 10) % 10) + "x ")
                  << std::setw(12) << (passed ? "PASSED" : "FAILED")
                  << std::scientific << std::setprecision(2) << max_rel_diff << "\n";
    }

    std::cout << std::string(85, '-') << "\n\n";

    // Export to CSV if requested
    if (!cfg.output_csv.empty()) {
        std::ofstream csv(cfg.output_csv);
        if (csv.is_open()) {
            csv << "implementation,M,K,N,tile_size,num_threads,warmup_iters,measured_iters,"
                << "median_ms,mean_ms,stddev_ms,min_ms,max_ms,gflops,speedup,"
                << "correctness,max_abs_diff,max_rel_diff,cpu_model,compiler\n";
            for (const auto& r : records) {
                csv << r.impl_name << ","
                    << r.M << "," << r.K << "," << r.N << ","
                    << r.tile_size << "," << r.num_threads << ","
                    << r.warmup_iters << "," << r.measured_iters << ","
                    << std::fixed << std::setprecision(4) << r.median_ms << ","
                    << r.mean_ms << "," << r.stddev_ms << ","
                    << r.min_ms << "," << r.max_ms << ","
                    << std::setprecision(3) << r.gflops << ","
                    << std::setprecision(2) << r.speedup << ","
                    << (r.passed_correctness ? "PASSED" : "FAILED") << ","
                    << std::scientific << std::setprecision(4) << r.max_abs_diff << ","
                    << r.max_rel_diff << ",\""
                    << r.cpu_model << "\",\""
                    << r.compiler_info << "\"\n";
            }
            std::cout << "Results successfully exported to CSV: " << cfg.output_csv << "\n";
        } else {
            std::cerr << "Error: Could not open " << cfg.output_csv << " for writing.\n";
        }
    }

    // Export to JSON if requested
    if (!cfg.output_json.empty()) {
        std::ofstream jf(cfg.output_json);
        if (jf.is_open()) {
            jf << "{\n";
            jf << "  \"cpu_model\": \"" << cpu_model << "\",\n";
            jf << "  \"compiler\": \"" << compiler << "\",\n";
            jf << "  \"matrix_dimensions\": {\"M\": " << cfg.M << ", \"K\": " << cfg.K << ", \"N\": " << cfg.N << "},\n";
            jf << "  \"tile_size\": " << cfg.tile_size << ",\n";
            jf << "  \"num_threads\": " << cfg.num_threads << ",\n";
            jf << "  \"benchmarks\": [\n";
            for (size_t i = 0; i < records.size(); ++i) {
                const auto& r = records[i];
                jf << "    {\n";
                jf << "      \"implementation\": \"" << r.impl_name << "\",\n";
                jf << "      \"median_ms\": " << std::fixed << std::setprecision(4) << r.median_ms << ",\n";
                jf << "      \"mean_ms\": " << r.mean_ms << ",\n";
                jf << "      \"stddev_ms\": " << r.stddev_ms << ",\n";
                jf << "      \"min_ms\": " << r.min_ms << ",\n";
                jf << "      \"max_ms\": " << r.max_ms << ",\n";
                jf << "      \"gflops\": " << std::setprecision(3) << r.gflops << ",\n";
                jf << "      \"speedup\": " << std::setprecision(2) << r.speedup << ",\n";
                jf << "      \"correctness\": \"" << (r.passed_correctness ? "PASSED" : "FAILED") << "\",\n";
                jf << "      \"max_abs_diff\": " << std::scientific << std::setprecision(4) << r.max_abs_diff << ",\n";
                jf << "      \"max_rel_diff\": " << r.max_rel_diff << "\n";
                jf << "    }" << (i + 1 < records.size() ? "," : "") << "\n";
            }
            jf << "  ]\n";
            jf << "}\n";
            std::cout << "Results successfully exported to JSON: " << cfg.output_json << "\n";
        } else {
            std::cerr << "Error: Could not open " << cfg.output_json << " for writing.\n";
        }
    }

    return 0;
}
