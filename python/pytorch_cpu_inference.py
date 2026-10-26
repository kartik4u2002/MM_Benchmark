#!/usr/bin/env python3
"""
PyTorch CPU Inference Benchmarking Script.
Evaluates CPU inference latency and throughput for deep learning workloads
across varying batch sizes and thread configurations.

Note:
This benchmark evaluates PyTorch's native CPU runtime (using ATen / oneDNN / OpenBLAS).
It is strictly isolated from our custom C++ GEMM library benchmarks.
"""

import os
import sys
import time
import json
import csv
import argparse
import platform
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[ERROR] PyTorch is required to run this benchmark. Please install torch.")
    sys.exit(1)

class DeepInferenceMLP(nn.Module):
    """
    Multi-Layer Perceptron representative of dense AI inference workloads
    (e.g., recommendation models, tabular embedding projections, Transformer feed-forward sublayers).
    All layers rely heavily on dense GEMM operations.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 2048, num_layers: int = 4, num_classes: int = 100):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.GELU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_cpu_info():
    uname = platform.uname()
    return f"{uname.processor or uname.machine} ({platform.system()} {platform.release()})"

def benchmark_batch_size(model: nn.Module, batch_size: int, input_dim: int,
                         warmup: int, iters: int, num_threads: int):
    # Set PyTorch CPU thread count
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    actual_threads = torch.get_num_threads()

    # Generate dummy input tensor (Float32)
    dummy_input = torch.randn(batch_size, input_dim, dtype=torch.float32)

    # 1. Warm-up iterations (outside timing)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # 2. Measured iterations
    latencies_ms = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    # Statistics
    sorted_lats = sorted(latencies_ms)
    median_ms = (sorted_lats[len(sorted_lats) // 2]
                 if len(sorted_lats) % 2 == 1
                 else 0.5 * (sorted_lats[len(sorted_lats) // 2 - 1] + sorted_lats[len(sorted_lats) // 2]))
    mean_ms = float(np.mean(latencies_ms))
    stddev_ms = float(np.std(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    throughput_samples_per_sec = (batch_size / (median_ms * 1e-3)) if median_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "input_dim": input_dim,
        "num_threads": actual_threads,
        "warmup_iters": warmup,
        "measured_iters": iters,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "stddev_ms": stddev_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p95_ms": p95_ms,
        "throughput_samples_sec": throughput_samples_per_sec
    }

def main():
    parser = argparse.ArgumentParser(description="PyTorch CPU AI Inference Benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 128], help="Batch sizes")
    parser.add_argument("--input-dim", type=int, default=768, help="Model input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden layer dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of dense hidden layers")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=30, help="Measured iterations")
    parser.add_argument("--threads", type=int, nargs="+", default=[0], help="Threads to evaluate (0 = default)")
    parser.add_argument("--out-dir", type=Path, default=Path("./results"), help="Results directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================================")
    print("                 PyTorch CPU AI Inference Benchmark                    ")
    print("=======================================================================")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version:  {platform.python_version()}")
    print(f"CPU System:      {get_cpu_info()}")
    print(f"Default Threads: {torch.get_num_threads()}")
    print(f"Model:           MLP (Input={args.input_dim}, Hidden={args.hidden_dim}, Layers={args.layers})")
    print("=======================================================================\n")

    # Instantiate model outside timed region
    model = DeepInferenceMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.layers)
    model.eval() # Strictly set evaluation mode

    all_results = []

    print(f"{'Batch':<8} {'Threads':<8} {'Median (ms)':<14} {'Mean (ms)':<12} {'p95 (ms)':<12} {'Throughput (samp/s)':<20}")
    print("-" * 75)

    for th in args.threads:
        for b in args.batch_sizes:
            res = benchmark_batch_size(
                model=model,
                batch_size=b,
                input_dim=args.input_dim,
                warmup=args.warmup,
                iters=args.iterations,
                num_threads=th
            )
            all_results.append(res)
            print(f"{res['batch_size']:<8} {res['num_threads']:<8} "
                  f"{res['median_ms']:<14.3f} {res['mean_ms']:<12.3f} {res['p95_ms']:<12.3f} "
                  f"{res['throughput_samples_sec']:<20.1f}")

    print("-" * 75)

    # Export to JSON
    json_path = args.out_dir / "pytorch_inference_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "cpu_info": get_cpu_info(),
            "model_config": {
                "input_dim": args.input_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.layers
            },
            "results": all_results
        }, f, indent=2)
    print(f"\n[DONE] PyTorch inference results saved to: {json_path}")

    # Export to CSV
    csv_path = args.out_dir / "pytorch_inference_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[DONE] PyTorch inference CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
