"""Performance benchmark for the FreshSense Phase 2 inference stack.

Runs inference on synthetic frames (no webcam required) and reports:

- Preprocessing + model inference latency (ms): avg / p50 / p95 / max
- Inference throughput (FPS)
- Python memory before/after (tracemalloc) to confirm no RAM growth
- GPU/VRAM allocation when a CUDA device is present

Usage:
    python -m scripts.benchmark_inference [--frames N] [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predictor import Predictor


def main() -> int:
    parser = argparse.ArgumentParser(description="FreshSense Phase 2 inference benchmark")
    parser.add_argument("--frames", type=int, default=30, help="Number of measured frames")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup frames")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FreshSense Phase 2 - Inference Benchmark")
    print("=" * 60)

    predictor = Predictor(args.checkpoint)
    print(f"Device        : {predictor.device}")
    print(f"Model version : {predictor.model_version}")
    print(f"Classes       : {predictor.num_classes}")

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup (lazy init, allocator warm-up) - excluded from metrics.
    for _ in range(args.warmup):
        predictor.predict(frame)

    latencies_ms: list = []
    batch = max(1, args.frames // 5)

    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()

    start = time.perf_counter()
    for i in range(args.frames):
        t0 = time.perf_counter()
        predictor.predict(frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if (i + 1) % batch == 0:
            # Report memory at checkpoints to confirm stability.
            cur, _ = tracemalloc.get_traced_memory()
            print(f"  checkpoint {i + 1:>4}: python-mem={cur / 1e6:.1f} MB")
    elapsed = time.perf_counter() - start
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(0.95 * len(latencies_ms)) - 1]
    avg = statistics.mean(latencies_ms)
    fps = args.frames / elapsed if elapsed > 0 else 0.0

    print()
    print("Latency (preprocess + inference):")
    print(f"  mean : {avg:7.1f} ms")
    print(f"  p50  : {p50:7.1f} ms")
    print(f"  p95  : {p95:7.1f} ms")
    print(f"  max  : {max(latencies_ms):7.1f} ms")
    print(f"Inference throughput: {fps:6.1f} FPS")
    print(f"Python memory: start={mem_start / 1e6:.1f} MB, peak={mem_peak / 1e6:.1f} MB")

    # GPU / VRAM reporting when available.
    if hasattr(predictor.device, "type") and predictor.device.type == "cuda":
        import torch

        print(
            f"CUDA VRAM allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB, "
            f"cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB"
        )
    else:
        print("Device is not CUDA; no VRAM measurement performed.")

    print("=" * 60)

    # Memory leak heuristic: peak should stay near the working set, not grow
    # linearly with frame count. We report it; a leak check is also covered by
    # the test suite (tracker/FPS bounded windows).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
