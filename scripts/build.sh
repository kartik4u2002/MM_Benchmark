#!/usr/bin/env bash
set -e

# Directory resolution
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="${1:-Release}"

echo "=========================================================="
echo "  Building CPU Matrix Multiplication Accelerator [${BUILD_TYPE}]"
echo "=========================================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Build Dir:    ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Determine number of parallel compilation jobs
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Configure CMake
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DENABLE_BLAS=ON \
      -DENABLE_AVX2=ON \
      "${PROJECT_ROOT}"

# Build all targets
cmake --build . --config "${BUILD_TYPE}" --parallel "${NUM_JOBS}"

echo ""
echo "Build complete! Executables available in: ${BUILD_DIR}/bin"
ls -lh "${BUILD_DIR}/bin" || true
