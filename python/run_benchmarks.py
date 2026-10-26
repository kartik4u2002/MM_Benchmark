#!/usr/bin/env python3
"""
Automated benchmark runner for CPU Matrix Multiplication Accelerator.
Executes the compiled C++ benchmark binary across various matrix shapes,
tile sizes, and thread configurations, aggregating results into CSV and JSON.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

def find_benchmark_binary(build_dir: Path) -> Path:
    candidates = [
        build_dir / "bin" / "matmul_benchmark",
        build_dir / "bin" / "matmul_benchmark.exe",
        build_dir / "matmul_benchmark",
        build_dir / "matmul_benchmark.exe",
        build_dir / "benchmarks" / "matmul_benchmark",
        build_dir / "benchmarks" / "matmul_benchmark.exe",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(f"Could not find matmul_benchmark executable in {build_dir}")

def run_suite(binary_path: Path, config_file: Path, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = results_dir / "raw"
    temp_dir.mkdir(exist_ok=True)

    with open(config_file, "r") as f:
        config = json.load(f)

    all_records = []
    csv_header = ""

    print(f"=== Running Matrix Multiplication Benchmark Suite ===")
    print(f"Binary: {binary_path}")
    print(f"Config: {config_file}\n")

    # 1. Main configuration tests (small, medium, large, rectangular, edge)
    configs = config.get("configurations", [])
    for idx, cfg in enumerate(configs, 1):
        name = cfg["name"]
        M, K, N = cfg["M"], cfg["K"], cfg["N"]
        warmup = cfg.get("warmup", 3)
        iters = cfg.get("iterations", 10)
        tile = cfg.get("tile_size", 64)
        out_csv = temp_dir / f"{name}.csv"
        out_json = temp_dir / f"{name}.json"

        cmd = [
            str(binary_path),
            "--M", str(M),
            "--K", str(K),
            "--N", str(N),
            "--warmup", str(warmup),
            "--iterations", str(iters),
            "--tile-size", str(tile),
            "--output-csv", str(out_csv),
            "--output-json", str(out_json),
        ]

        print(f"[{idx}/{len(configs)}] Running shape {name}: M={M}, K={K}, N={N}, tile={tile}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"  [ERROR] Benchmark failed for {name}:")
            print(res.stderr)
            continue

        if out_csv.exists():
            with open(out_csv, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    if not csv_header:
                        csv_header = lines[0]
                    all_records.extend(lines[1:])

    # 2. Tile size sweep on 512x512 matrix
    tile_sweep = config.get("tile_size_sweep", [16, 32, 64, 128, 256])
    print(f"\n=== Running Tile Size Sweep on 512x512 (tile sizes: {tile_sweep}) ===")
    for tile in tile_sweep:
        out_csv = temp_dir / f"tile_sweep_{tile}.csv"
        cmd = [
            str(binary_path),
            "--M", "512",
            "--K", "512",
            "--N", "512",
            "--warmup", "2",
            "--iterations", "5",
            "--impl", "blocked",
            "--tile-size", str(tile),
            "--output-csv", str(out_csv),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and out_csv.exists():
            with open(out_csv, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
                if len(lines) > 1:
                    all_records.append(lines[1])

    # 3. Thread scaling sweep on 512x512 matrix
    thread_sweep = config.get("thread_sweep", [1, 2, 4, 8])
    print(f"\n=== Running Thread Scaling Sweep on 512x512 (threads: {thread_sweep}) ===")
    for th in thread_sweep:
        out_csv = temp_dir / f"thread_sweep_{th}.csv"
        cmd = [
            str(binary_path),
            "--M", "512",
            "--K", "512",
            "--N", "512",
            "--warmup", "2",
            "--iterations", "5",
            "--impl", "openmp_avx2",
            "--threads", str(th),
            "--output-csv", str(out_csv),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and out_csv.exists():
            with open(out_csv, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
                if len(lines) > 1:
                    all_records.append(lines[1])

    # Save aggregated CSV
    combined_csv = results_dir / "benchmark_results.csv"
    with open(combined_csv, "w") as f:
        f.write(csv_header + "\n")
        for rec in all_records:
            f.write(rec + "\n")

    print(f"\n[DONE] Combined benchmark results written to: {combined_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full matrix multiplication benchmark suite")
    parser.add_argument("--build-dir", type=Path, default=Path("./build"), help="Path to build directory")
    parser.add_argument("--config", type=Path, default=Path("./benchmarks/benchmark_config.json"), help="Config JSON")
    parser.add_argument("--results-dir", type=Path, default=Path("./results"), help="Results output directory")
    args = parser.parse_args()

    binary = find_benchmark_binary(args.build_dir)
    run_suite(binary, args.config, args.results_dir)
