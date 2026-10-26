#!/usr/bin/env python3
"""
Performance Visualization Script for CPU Matrix Multiplication Accelerator.
Generates publication-quality charts comparing throughput (GFLOP/s), latency,
speedup, and tile size sweeps using Matplotlib.
"""

import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_path: Path):
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "impl": row["implementation"],
                    "M": int(row["M"]),
                    "K": int(row["K"]),
                    "N": int(row["N"]),
                    "tile_size": int(row["tile_size"]),
                    "num_threads": int(row["num_threads"]),
                    "median_ms": float(row["median_ms"]),
                    "gflops": float(row["gflops"]),
                    "speedup": float(row["speedup"]),
                    "correctness": row["correctness"],
                })
            except Exception:
                continue
    return records

def plot_throughput_comparison(records, out_dir: Path):
    # Select square matrices: 128, 512, 1024
    target_sizes = [128, 512, 1024]
    impl_order = ["baseline", "cache_friendly", "blocked", "avx2", "openmp", "openmp_avx2", "blas"]
    color_map = {
        "baseline": "#9E9E9E",
        "cache_friendly": "#42A5F5",
        "blocked": "#26A69A",
        "avx2": "#FFA726",
        "openmp": "#AB47BC",
        "openmp_avx2": "#EF5350",
        "blas": "#2E7D32"
    }

    # Group data
    shape_impl_gflops = defaultdict(dict)
    for r in records:
        if r["M"] == r["K"] == r["N"] and r["M"] in target_sizes and r["num_threads"] in (0, 8):
            shape_impl_gflops[r["M"]][r["impl"]] = r["gflops"]

    present_sizes = [s for s in target_sizes if s in shape_impl_gflops]
    if not present_sizes:
        print("[Plot] No square matrix records found for throughput comparison.")
        return

    # Find which impls are present
    present_impls = [imp for imp in impl_order if any(imp in shape_impl_gflops[s] for s in present_sizes)]

    x = np.arange(len(present_sizes))
    n_bars = len(present_impls)
    bar_width = 0.8 / max(n_bars, 1)

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    for i, impl in enumerate(present_impls):
        vals = [shape_impl_gflops[s].get(impl, 0.0) for s in present_sizes]
        offset = (i - (n_bars - 1) / 2.0) * bar_width
        bars = ax.bar(x + offset, vals, width=bar_width, label=impl, color=color_map.get(impl, "#78909C"))

        # Value annotations
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Matrix Dimension (N x N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Throughput (GFLOP/s)", fontsize=11, fontweight="bold")
    ax.set_title("CPU Matrix Multiplication Throughput Across Optimization Levels", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}x{s}" for s in present_sizes], fontsize=10)
    ax.legend(title="Implementation", frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out_path = out_dir / "throughput_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {out_path}")

def plot_tile_size_sweep(records, out_dir: Path):
    # Filter 512x512 blocked records with varying tile_size and keep unique tile sizes
    tile_dict = {}
    for r in records:
        if r["M"] == 512 and r["K"] == 512 and r["N"] == 512 and r["impl"] == "blocked":
            tile_dict[r["tile_size"]] = r["gflops"]

    if not tile_dict:
        return

    tiles = sorted(tile_dict.keys())
    gflops = [tile_dict[t] for t in tiles]

    plt.figure(figsize=(8, 5))
    plt.plot(tiles, gflops, marker="o", linewidth=2.5, markersize=8, color="#00897B")
    plt.title("Effect of Tile Size on Cache-Blocked GEMM Throughput (512x512)", fontsize=12, fontweight="bold")
    plt.xlabel("Tile Size (Elements per dimension)", fontsize=10, fontweight="bold")
    plt.ylabel("Throughput (GFLOP/s)", fontsize=10, fontweight="bold")
    plt.xticks(tiles, [f"{t}x{t}" for t in tiles])
    plt.grid(True, linestyle="--", alpha=0.6)

    for t, g in zip(tiles, gflops):
        plt.annotate(f"{g:.2f} GF", (t, g), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

    out_path = out_dir / "tile_size_sweep.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {out_path}")

def plot_speedup_summary(records, out_dir: Path):
    # Choose standard 512x512 records (tile_size=64, num_threads in (0, 8))
    impl_map = {}
    for r in records:
        if r["M"] == 512 and r["K"] == 512 and r["N"] == 512 and r["tile_size"] == 64:
            if r["impl"] not in impl_map or r["num_threads"] in (0, 8):
                impl_map[r["impl"]] = r["median_ms"]

    if "baseline" not in impl_map:
        return

    base_time = impl_map["baseline"]
    order = ["baseline", "cache_friendly", "blocked", "avx2", "openmp", "openmp_avx2", "blas"]
    impls = [imp for imp in order if imp in impl_map]
    speedups = [base_time / impl_map[imp] for imp in impls]

    colors = {
        "baseline": "#9E9E9E",
        "cache_friendly": "#42A5F5",
        "blocked": "#26A69A",
        "avx2": "#FFA726",
        "openmp": "#AB47BC",
        "openmp_avx2": "#EF5350",
        "blas": "#2E7D32"
    }
    bar_colors = [colors.get(imp, "#78909C") for imp in impls]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(impls, speedups, color=bar_colors)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Baseline (1.0x)")
    plt.title("Speedup Relative to Baseline on 512x512 Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Implementation", fontsize=10, fontweight="bold")
    plt.ylabel("Speedup Factor (x)", fontsize=10, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, sp in zip(bars, speedups):
        plt.annotate(f"{sp:.1f}x",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    out_path = out_dir / "speedup_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot matrix multiplication benchmark charts")
    parser.add_argument("--csv", type=Path, default=Path("./results/benchmark_results.csv"), help="CSV input")
    parser.add_argument("--out-dir", type=Path, default=Path("./results/plots"), help="Plot output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_data(args.csv)
    if not records:
        print("[Plot] No data found in CSV.")
        return

    plot_throughput_comparison(records, args.out_dir)
    plot_tile_size_sweep(records, args.out_dir)
    plot_speedup_summary(records, args.out_dir)

if __name__ == "__main__":
    main()
