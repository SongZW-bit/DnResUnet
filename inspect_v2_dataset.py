import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    from scipy.signal import spectrogram
except ImportError as exc:
    raise SystemExit(
        "inspect_v2_dataset.py requires matplotlib and scipy."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect representative samples from gravity_dataset_100k_V2_REALISTIC.pt")
    parser.add_argument("--dataset", default="gravity_dataset_100k_V2_REALISTIC.pt")
    parser.add_argument("--output-dir", default="dataset_inspection_v2")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260312)
    return parser.parse_args()


def select_representative_indices(noise_levels, scenario_labels, num_samples, seed):
    rng = np.random.default_rng(seed)
    unique_noise_levels = sorted({float(v) for v in noise_levels.tolist()})
    unique_scenarios = sorted(set(scenario_labels))
    selected = []

    for noise_level in unique_noise_levels:
        mask = np.where(np.isclose(noise_levels, noise_level))[0]
        if len(mask) > 0:
            selected.append(int(mask[len(mask) // 2]))
            if len(selected) >= num_samples:
                return selected[:num_samples]

    for scenario in unique_scenarios:
        mask = np.where(np.array(scenario_labels) == scenario)[0]
        if len(mask) > 0:
            selected.append(int(mask[len(mask) // 2]))
            if len(selected) >= num_samples:
                return selected[:num_samples]

    all_indices = np.arange(len(noise_levels))
    rng.shuffle(all_indices)
    for idx in all_indices:
        selected.append(int(idx))
        if len(selected) >= num_samples:
            break
    return list(dict.fromkeys(selected))[:num_samples]


def plot_sample(sample_id, noisy, clean, sample_max, metadata, output_dir):
    noisy_phys = noisy * sample_max
    clean_phys = clean * sample_max
    residual_phys = noisy_phys - clean_phys
    depth_axis = np.arange(len(noisy))

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    axes[0].plot(depth_axis, noisy, color="#c94f4f", alpha=0.75, label="Noisy (normalized)")
    axes[0].plot(depth_axis, clean, color="#1f1f1f", linewidth=1.8, label="Clean (normalized)")
    axes[0].set_title(f"Sample {sample_id} - normalized time-domain")
    axes[0].set_xlabel("Resampled index")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(depth_axis, noisy_phys, color="#d95f02", alpha=0.8, label="Noisy (mGal)")
    axes[1].plot(depth_axis, clean_phys, color="#1b9e77", linewidth=1.8, label="Clean (mGal)")
    axes[1].plot(depth_axis, residual_phys, color="#7570b3", linewidth=1.2, label="Noise component (mGal)")
    axes[1].set_title("Physical-domain time-domain")
    axes[1].set_xlabel("Resampled index")
    axes[1].set_ylabel("mGal")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    n_points = len(noisy)
    freq_axis = fftfreq(n_points, d=1.0)[: n_points // 2]
    noisy_amp = 2.0 / n_points * np.abs(fft(noisy_phys)[: n_points // 2])
    clean_amp = 2.0 / n_points * np.abs(fft(clean_phys)[: n_points // 2])
    residual_amp = 2.0 / n_points * np.abs(fft(residual_phys)[: n_points // 2])
    axes[2].plot(freq_axis, noisy_amp, color="#d95f02", label="Noisy")
    axes[2].plot(freq_axis, clean_amp, color="#1b9e77", label="Clean")
    axes[2].plot(freq_axis, residual_amp, color="#7570b3", label="Noise")
    axes[2].set_yscale("log")
    axes[2].set_title("Frequency-domain amplitude spectrum")
    axes[2].set_xlabel("Spatial frequency (1/index)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"sample_{sample_id:05d}_overview.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, signal, title in zip(
        axes,
        [noisy_phys, clean_phys, residual_phys],
        ["Noisy spectrogram", "Clean spectrogram", "Noise spectrogram"],
    ):
        freq, depth, power = spectrogram(signal, fs=1.0, window="hann", nperseg=64, noverlap=60, nfft=256)
        ax.pcolormesh(depth, freq, 10.0 * np.log10(power + 1e-12), shading="gouraud", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Window index")
        ax.set_ylabel("Frequency")
    fig.suptitle(
        f"Sample {sample_id} | noise={metadata['noise_level']:.3f} mGal | "
        f"spacing={metadata['spacing']} m | scenario={metadata['scenario']}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"sample_{sample_id:05d}_spectrogram.png", dpi=220)
    plt.close(fig)


def plot_dataset_summary(data, output_dir):
    noise_levels = data["noise_levels"].numpy()
    sample_intervals = data["sample_intervals"].numpy()
    clean_peaks = data["clean_peaks_mgal"].numpy()
    challenging_mask = data["challenging_mask"].numpy().astype(bool)
    raw_lengths = data["raw_lengths"].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(noise_levels, bins=np.unique(noise_levels).size, color="#4c78a8", edgecolor="white")
    axes[0, 0].set_title("Noise level distribution")
    axes[0, 0].set_xlabel("Noise std (mGal)")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(sample_intervals, bins=np.unique(sample_intervals).size, color="#f58518", edgecolor="white")
    axes[0, 1].set_title("Sampling interval distribution")
    axes[0, 1].set_xlabel("Spacing (m)")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(clean_peaks, bins=60, color="#54a24b", edgecolor="white")
    axes[1, 0].axvline(0.01, color="red", linestyle="--", linewidth=1.5, label="challenging threshold")
    axes[1, 0].set_title("Clean peak amplitude distribution")
    axes[1, 0].set_xlabel("Peak amplitude (mGal)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()

    challenging_counts = [int((~challenging_mask).sum()), int(challenging_mask.sum())]
    axes[1, 1].bar(["regular", "challenging"], challenging_counts, color=["#72b7b2", "#e45756"])
    axes[1, 1].set_title("Challenging subset count")
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_dir / "dataset_summary.png", dpi=220)
    plt.close(fig)

    summary = {
        "total_samples": int(data["X_noisy"].shape[0]),
        "signal_shape": list(data["X_noisy"].shape),
        "noise_levels_mgal": sorted({float(v) for v in noise_levels.tolist()}),
        "sample_intervals_m": sorted({float(v) for v in sample_intervals.tolist()}),
        "raw_lengths": sorted({int(v) for v in raw_lengths.tolist()}),
        "challenging_fraction": float(challenging_mask.mean()),
        "clean_peak_mgal_mean": float(clean_peaks.mean()),
        "clean_peak_mgal_std": float(clean_peaks.std()),
        "scenario_counts": {
            name: int(data["scenario_labels"].count(name))
            for name in sorted(set(data["scenario_labels"]))
        },
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    plot_dataset_summary(data, output_dir)

    indices = select_representative_indices(
        data["noise_levels"],
        data["scenario_labels"],
        args.num_samples,
        args.seed,
    )

    for sample_id in indices:
        noisy = data["X_noisy"][sample_id].numpy().squeeze()
        clean = data["Y_clean"][sample_id].numpy().squeeze()
        sample_max = float(data["sample_maxs"][sample_id].item())
        metadata = {
            "noise_level": float(data["noise_levels"][sample_id].item()),
            "spacing": float(data["sample_intervals"][sample_id].item()),
            "raw_length": int(data["raw_lengths"][sample_id].item()),
            "challenging": bool(data["challenging_mask"][sample_id].item()),
            "scenario": data["scenario_labels"][sample_id],
        }
        plot_sample(sample_id, noisy, clean, sample_max, metadata, output_dir)

    print(f"Saved dataset inspection outputs to: {output_dir}")


if __name__ == "__main__":
    main()
