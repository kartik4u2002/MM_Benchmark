#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TEST_BIN="${BUILD_DIR}/bin/test_matmul"

echo "=========================================================="
echo "    Running CPU Matrix Multiplication Test Suite          "
echo "=========================================================="

if [ ! -f "${TEST_BIN}" ]; then
    echo "Test binary not found at ${TEST_BIN}. Running build first..."
    "${PROJECT_ROOT}/scripts/build.sh"
fi

"${TEST_BIN}"
