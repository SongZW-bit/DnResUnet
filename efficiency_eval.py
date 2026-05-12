import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from independent_resampling_eval import (
    denoise_signal_savgol,
    denoise_signal_wavelet,
    denoise_signal_wiener,
    load_checkpoint_model,
)


DEFAULT_MODELS = {
    "DnResUnet": "Models_v3_realistic/dnresunet_v2_realistic_checkpoint.pt",
    "BasicCNN": "Models_Benchmark_v2/BasicCNN_v2_realistic_checkpoint.pt",
    "DnCNN": "Models_Benchmark_v2/DnCNN_v2_realistic_checkpoint.pt",
    "UNet1D": "Models_Benchmark_v2/UNet1D_v2_realistic_checkpoint.pt",
    "TCN": "Models_Benchmark_v2/TCN_v2_realistic_checkpoint.pt",
}

CLASSICAL_METHODS = {
    "S-G Filter": denoise_signal_savgol,
    "Wiener": denoise_signal_wiener,
    "Wavelet": denoise_signal_wavelet,
}


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def checkpoint_size_mb(path):
    return Path(path).stat().st_size / (1024.0 * 1024.0)


def time_model(model, signal_length, batch_size, warmup, repeats, device):
    model.eval()
    x = torch.randn(batch_size, 1, signal_length, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            raw = model(x)
            _ = x - raw if getattr(model, "predict_noise", False) else raw
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            raw = model(x)
            _ = x - raw if getattr(model, "predict_noise", False) else raw
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    latency_ms_per_profile = elapsed * 1000.0 / (repeats * batch_size)
    throughput_profiles_per_s = repeats * batch_size / elapsed
    return latency_ms_per_profile, throughput_profiles_per_s


def time_classical(method, signal_length, warmup, repeats):
    signal = np.random.randn(signal_length).astype(np.float32)
    for _ in range(warmup):
        method(signal)
    start = time.perf_counter()
    for _ in range(repeats):
        method(signal)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / repeats
    throughput = repeats / elapsed
    return latency_ms, throughput


def write_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference efficiency of denoising deep models.")
    parser.add_argument("--output-dir", default="efficiency_results")
    parser.add_argument("--signal-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, checkpoint in DEFAULT_MODELS.items():
        path = Path(checkpoint)
        model = load_checkpoint_model(path, device)
        params = count_parameters(model)
        latency_b1, throughput_b1 = time_model(
            model,
            signal_length=args.signal_length,
            batch_size=1,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        latency_batch, throughput_batch = time_model(
            model,
            signal_length=args.signal_length,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        rows.append(
            {
                "model": name,
                "parameters": params,
                "checkpoint_mb": checkpoint_size_mb(path),
                "device": str(device),
                "threads": torch.get_num_threads(),
                "signal_length": args.signal_length,
                "batch_size": args.batch_size,
                "latency_ms_profile_b1": latency_b1,
                "throughput_profiles_s_b1": throughput_b1,
                "latency_ms_profile_batched": latency_batch,
                "throughput_profiles_s_batched": throughput_batch,
            }
        )

    write_csv(rows, output_dir / "deep_model_efficiency.csv")
    classical_rows = []
    for name, method in CLASSICAL_METHODS.items():
        latency, throughput = time_classical(
            method,
            signal_length=args.signal_length,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        classical_rows.append(
            {
                "method": name,
                "parameters": 0,
                "device": "cpu",
                "threads": torch.get_num_threads(),
                "signal_length": args.signal_length,
                "latency_ms_profile": latency,
                "throughput_profiles_s": throughput,
            }
        )
    write_csv(classical_rows, output_dir / "classical_efficiency.csv")
    print(f"Wrote {output_dir / 'deep_model_efficiency.csv'}")
    print(f"Wrote {output_dir / 'classical_efficiency.csv'}")
    for row in rows:
        print(
            f"{row['model']}: params={row['parameters']}, "
            f"B1={row['latency_ms_profile_b1']:.4f} ms/profile, "
            f"B{args.batch_size}={row['latency_ms_profile_batched']:.4f} ms/profile"
        )
    for row in classical_rows:
        print(f"{row['method']}: {row['latency_ms_profile']:.4f} ms/profile")


if __name__ == "__main__":
    main()
