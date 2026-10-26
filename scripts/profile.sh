#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_BIN="${PROJECT_ROOT}/build/bin/matmul_benchmark"
M=${1:-512}
K=${2:-512}
N=${3:-512}

echo "=========================================================="
echo "    Linux perf Performance Profiling Workflow             "
echo "=========================================================="

if [ ! -f "${BENCHMARK_BIN}" ]; then
    echo "Error: Benchmark binary not found at ${BENCHMARK_BIN}. Run build.sh first."
    exit 1
fi

# Check perf availability
if ! command -v perf &> /dev/null; then
    echo "[Notice] 'perf' tool is not installed or not in PATH."
    echo "To install linux-tools on Ubuntu/Debian:"
    echo "  sudo apt-get update && sudo apt-get install -y linux-tools-common linux-tools-generic linux-tools-\$(uname -r)"
    echo ""
    echo "Note for WSL2 / Virtual Environments:"
    echo "WSL2 runs on a virtualized hypervisor where physical hardware performance"
    echo "counters (PMU) are typically virtualized or restricted unless WSL2 PMU support"
    echo "is explicitly enabled in .wslconfig with:"
    echo "  [wsl2]"
    echo "  hardwarePerformanceCounters=true"
    echo ""
    echo "Recommended bare-metal Linux profiling workflow commands:"
    echo "1. Measure Cycles, Instructions, and IPC:"
    echo "   perf stat -e cycles,instructions,cache-misses,cache-references,branches,branch-misses \\"
    echo "     ${BENCHMARK_BIN} --M ${M} --K ${K} --N ${N} --impl baseline --iterations 5"
    echo ""
    echo "2. Compare Cache Misses (Baseline vs Cache-Friendly):"
    echo "   perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \\"
    echo "     ${BENCHMARK_BIN} --M ${M} --K ${K} --N ${N} --impl cache_friendly --iterations 5"
    echo ""
    echo "3. Sample Hotspots with perf record and report:"
    echo "   perf record -g -F 999 ${BENCHMARK_BIN} --M ${M} --K ${K} --N ${N} --impl blocked --iterations 20"
    echo "   perf report --stdio"
    exit 0
fi

# Check permissions
PERF_PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "3")
echo "Kernel perf_event_paranoid level: ${PERF_PARANOID}"
if [ "${PERF_PARANOID}" -gt 1 ]; then
    echo "[Warning] perf_event_paranoid is > 1. Hardware PMU counter collection may require sudo."
    echo "To allow unprivileged profiling, run: sudo sysctl -w kernel.perf_event_paranoid=1"
fi

echo ""
echo "--- Profiling Baseline (i-j-k) ---"
perf stat -e cycles,instructions,cache-misses,cache-references \
    "${BENCHMARK_BIN}" --M "${M}" --K "${K}" --N "${N}" --impl baseline --iterations 5 || true

echo ""
echo "--- Profiling Cache-Friendly (i-k-j) ---"
perf stat -e cycles,instructions,cache-misses,cache-references \
    "${BENCHMARK_BIN}" --M "${M}" --K "${K}" --N "${N}" --impl cache_friendly --iterations 5 || true

echo ""
echo "--- Profiling AVX2 / SIMD ---"
perf stat -e cycles,instructions,cache-misses,cache-references \
    "${BENCHMARK_BIN}" --M "${M}" --K "${K}" --N "${N}" --impl avx2 --iterations 5 || true
