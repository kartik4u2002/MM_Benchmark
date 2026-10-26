#!/usr/bin/env python3
"""
Performance Analysis Script for CPU Matrix Multiplication Accelerator.
Parses CSV and JSON benchmark results, generates structured markdown tables,
calculates speedup, checks correctness outcomes, and summarizes performance insights.
"""

import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def parse_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rec = {
                    "impl": row["implementation"],
                    "M": int(row["M"]),
                    "K": int(row["K"]),
                    "N": int(row["N"]),
                    "tile_size": int(row["tile_size"]),
                    "num_threads": int(row["num_threads"]),
                    "warmup_iters": int(row["warmup_iters"]),
                    "measured_iters": int(row["measured_iters"]),
                    "median_ms": float(row["median_ms"]),
                    "mean_ms": float(row["mean_ms"]),
                    "stddev_ms": float(row["stddev_ms"]),
                    "min_ms": float(row["min_ms"]),
                    "max_ms": float(row["max_ms"]),
                    "gflops": float(row["gflops"]),
                    "speedup": float(row["speedup"]),
                    "correctness": row["correctness"],
                    "max_abs_diff": float(row["max_abs_diff"]),
                    "max_rel_diff": float(row["max_rel_diff"]),
                    "cpu_model": row.get("cpu_model", "Unknown CPU"),
                    "compiler": row.get("compiler", "Unknown Compiler"),
                }
                records.append(rec)
            except (ValueError, KeyError) as e:
                print(f"Skipping malformed row: {row} ({e})")
    return records

def generate_markdown_report(records, output_file: Path = None):
    # Group by shape (M, K, N)
    by_shape = defaultdict(list)
    for r in records:
        key = (r["M"], r["K"], r["N"])
        by_shape[key].append(r)

    cpu_info = records[0]["cpu_model"] if records else "N/A"
    compiler_info = records[0]["compiler"] if records else "N/A"

    lines = []
    lines.append("# CPU Matrix Multiplication Benchmark Analysis Report\n")
    lines.append(f"**Hardware Platform:** {cpu_info}  \n")
    lines.append(f"**Compiler Toolchain:** {compiler_info}  \n")
    lines.append(f"**Date:** System Benchmark Execution  \n\n")

    lines.append("## 1. Executive Performance Summary\n")
    lines.append(
        "Dense Matrix Multiplication computes $C = A \\times B$ where $A \\in \\mathbb{R}^{M \\times K}$, "
        "$B \\in \\mathbb{R}^{K \\times N}$, and $C \\in \\mathbb{R}^{M \\times N}$. "
        "The standard dense operation count is $2 \\times M \\times N \\times K$ floating-point operations "
        "(1 multiply + 1 accumulate per inner product term).\n\n"
        "Throughput is defined as:\n\n"
        "$$\\text{Throughput (GFLOP/s)} = \\frac{2 \\times M \\times N \\times K}{\\text{Execution Time (seconds)} \\times 10^9}$$\n\n"
        "Speedup is defined as:\n\n"
        "$$\\text{Speedup} = \\frac{\\text{Median Time}_{\\text{baseline}}}{\\text{Median Time}_{\\text{optimized}}}$$\n"
    )

    all_passed = True
    for (M, K, N), shape_records in by_shape.items():
        lines.append(f"### Matrix Shape: {M} x {K} x {N}\n")
        lines.append("| Implementation | Tile Size | Threads | Median Time (ms) | StdDev (ms) | GFLOP/s | Speedup | Correctness | Max Rel Diff |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        # Find baseline median for this shape
        baseline_time = None
        for r in shape_records:
            if r["impl"] == "baseline":
                baseline_time = r["median_ms"]
                break

        for r in shape_records:
            speedup = (baseline_time / r["median_ms"]) if baseline_time else r["speedup"]
            status = r["correctness"]
            if status != "PASSED":
                all_passed = False
            threads_str = str(r["num_threads"]) if r["num_threads"] > 0 else "All"
            lines.append(
                f"| `{r['impl']}` | {r['tile_size']} | {threads_str} | "
                f"{r['median_ms']:.2f} | {r['stddev_ms']:.2f} | {r['gflops']:.2f} | "
                f"**{speedup:.2f}x** | {status} | {r['max_rel_diff']:.2e} |"
            )
        lines.append("\n")

    lines.append("## 2. Correctness & Numerical Validation\n")
    if all_passed:
        lines.append("All benchmarked configurations **PASSED** numerical equivalence testing against the baseline reference with $rtol=10^{-3}$ and $atol=10^{-3}$.\n")
    else:
        lines.append("WARNING: One or more configurations encountered numerical discrepancies or failed bounds tests.\n")

    report_text = "\n".join(lines)
    print(report_text)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nReport written to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze matrix multiplication benchmark results")
    parser.add_argument("--csv", type=Path, default=Path("./results/benchmark_results.csv"), help="CSV input")
    parser.add_argument("--out-report", type=Path, default=Path("./results/BENCHMARK_ANALYSIS.md"), help="Markdown report output")
    args = parser.parse_args()

    recs = parse_csv(args.csv)
    generate_markdown_report(recs, args.out_report)
