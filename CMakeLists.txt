cmake_minimum_required(VERSION 3.16)
project(CPUMatrixAccelerator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set runtime output directory to bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler optimization options
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
    if(ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
    # Enable AVX2 and FMA instructions for x86_64 architectures
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

# Sanitizers option (for debugging and correctness verification)
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(ENABLE_SANITIZERS AND NOT MSVC)
    message(STATUS "Enabling AddressSanitizer and UndefinedBehaviorSanitizer")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: multithreading enabled")
else()
    message(WARNING "OpenMP not found: building single-threaded only")
endif()

# BLAS support (optional)
option(ENABLE_BLAS "Enable BLAS-backed GEMM benchmark" ON)
if(ENABLE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    else()
        message(STATUS "BLAS not found: BLAS fallback will be disabled")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core library sources
set(LIB_SOURCES
    src/matrix.cpp
    src/matmul_baseline.cpp
    src/matmul_optimized.cpp
    src/matmul_blocked.cpp
    src/matmul_avx2.cpp
    src/matmul_openmp.cpp
    src/matmul_blas.cpp
    src/matmul_dispatch.cpp
)

add_library(matrix_accel STATIC ${LIB_SOURCES})
target_include_directories(matrix_accel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_accel PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_BLAS AND BLAS_FOUND)
    target_compile_definitions(matrix_accel PUBLIC HAVE_BLAS=1 USE_BLAS=1)
    target_link_libraries(matrix_accel PUBLIC ${BLAS_LIBRARIES})
endif()

# Executables
# 1. Main interactive CLI / demo
add_executable(cpu_matrix_demo src/main.cpp)
target_link_libraries(cpu_matrix_demo PRIVATE matrix_accel)

# 2. Correctness test suite
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE matrix_accel)

# 3. Production benchmark runner
add_executable(matmul_benchmark benchmarks/benchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE matrix_accel)

# Enable CTest integration
enable_testing()
add_test(NAME MatmulCorrectnessTests COMMAND test_matmul)
