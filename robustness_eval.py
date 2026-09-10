from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import chirp, savgol_filter, wiener
from skimage.restoration import denoise_wavelet


METHOD_ORDER = [
    "No denoise",
    "Savitzky-Golay",
    "Wiener",
    "Wavelet",
    "BasicCNN",
    "DnCNN",
    "UNet1D",
    "TCN",
    "DnResUnet",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_pair(noisy: np.ndarray, clean: np.ndarray):
    scale = float(np.max(np.abs(noisy)))
    if scale < 1e-8:
        scale = 1.0
    return noisy / scale, clean / scale, scale


def unseen_noise(length: int, target_std: float, rng: np.random.Generator) -> np.ndarray:
    heavy = rng.standard_t(df=3.0, size=length)
    heavy = heavy / (np.std(heavy) + 1e-12)
    frequencies = np.fft.rfftfreq(length)
    spectrum = rng.normal(size=frequencies.size) + 1j * rng.normal(size=frequencies.size)
    weights = np.zeros_like(frequencies)
    weights[1:] = 1.0 / np.sqrt(frequencies[1:])
    pink = np.fft.irfft(spectrum * weights, n=length)
    pink = pink / (np.std(pink) + 1e-12)
    coordinate = np.linspace(0.0, 1.0, length)
    swept = chirp(coordinate, f0=2.0, f1=35.0, t1=1.0, method="quadratic")
    swept *= 0.25 + 0.75 * np.sin(np.pi * coordinate) ** 2
    swept = swept / (np.std(swept) + 1e-12)
    burst = np.zeros(length)
    center = int(rng.integers(length // 5, 4 * length // 5))
    width = int(rng.integers(max(4, length // 80), max(6, length // 25)))
    left, right = max(0, center - width), min(length, center + width)
    burst[left:right] = rng.laplace(size=right - left)
    if np.std(burst) > 0:
        burst = burst / np.std(burst)
    mixture = 0.50 * heavy + 0.30 * pink + 0.30 * swept + 0.20 * burst
    mixture -= np.mean(mixture)
    return (mixture / (np.std(mixture) + 1e-12) * target_std).astype(np.float32)


def sparse_gapped_profile(clean: np.ndarray, rng: np.random.Generator):
    length = clean.size
    count = int(rng.integers(45, 76))
    indices = np.unique(np.r_[0, length - 1, np.sort(rng.choice(np.arange(1, length - 1), count - 2, replace=False))])
    gap_center = int(rng.integers(length // 4, 3 * length // 4))
    gap_half_width = int(rng.integers(length // 30, length // 14))
    keep = (indices < gap_center - gap_half_width) | (indices > gap_center + gap_half_width)
    indices = np.unique(np.r_[0, indices[keep], length - 1])
    restored = np.interp(np.arange(length), indices, clean[indices])
    spacing_multiplier = float((length - 1) / max(len(indices) - 1, 1))
    return restored.astype(np.float32), len(indices), spacing_multiplier


def classical_predictions(signal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Savitzky-Golay": savgol_filter(signal, 31, 5, mode="mirror"),
        "Wiener": np.asarray(wiener(signal, mysize=21), dtype=float),
        "Wavelet": np.asarray(
            denoise_wavelet(
                signal,
                wavelet="sym8",
                mode="soft",
                wavelet_levels=5,
                method="VisuShrink",
                rescale_sigma=True,
            ),
            dtype=float,
        ),
    }


def predict_model(model, device, signal: np.ndarray, predict_clean_signal) -> np.ndarray:
    tensor = torch.from_numpy(signal.astype(np.float32)).view(1, 1, -1).to(device)
    with torch.no_grad():
        return predict_clean_signal(model, tensor).detach().cpu().numpy().reshape(-1)


def gfc(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(estimate))
    return float(np.dot(reference, estimate) / denominator) if denominator > 1e-12 else float("nan")


def low_frequency_component(signal: np.ndarray, fraction: float = 0.08) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    cutoff = max(2, int(np.ceil(fraction * len(spectrum))))
    filtered = np.zeros_like(spectrum)
    filtered[:cutoff] = spectrum[:cutoff]
    return np.fft.irfft(filtered, n=signal.size)


def spectral_low_fraction(signal: np.ndarray, fraction: float = 0.08) -> float:
    spectrum = np.abs(np.fft.rfft(signal - np.mean(signal))) ** 2
    cutoff = max(2, int(np.ceil(fraction * len(spectrum))))
    return float(np.sum(spectrum[:cutoff]) / (np.sum(spectrum) + 1e-12))


def metrics(clean: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    error = estimate - clean
    low_clean = low_frequency_component(clean)
    low_estimate = low_frequency_component(estimate)
    clean_peak = float(np.max(np.abs(clean)))
    estimate_peak = float(np.max(np.abs(estimate)))
    return {
        "mse_norm": float(np.mean(error**2)),
        "psnr_norm_db": float(10.0 * np.log10(1.0 / max(np.mean(error**2), 1e-12))),
        "gfc": gfc(clean, estimate),
        "low_frequency_nrmse": float(
            np.sqrt(np.mean((low_estimate - low_clean) ** 2)) / (np.sqrt(np.mean(low_clean**2)) + 1e-12)
        ),
        "peak_amplitude_ratio": estimate_peak / (clean_peak + 1e-12),
        "peak_location_error_samples": float(abs(int(np.argmax(np.abs(estimate))) - int(np.argmax(np.abs(clean))))),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.nanmedian(array)),
        "q025": float(np.nanquantile(array, 0.025)),
        "q975": float(np.nanquantile(array, 0.975)),
        "n": int(np.sum(np.isfinite(array))),
    }


def aggregate(raw_rows: list[dict], subset_name: str, selector) -> list[dict]:
    selected = [row for row in raw_rows if selector(row)]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        groups[(row["condition"], row["method"])].append(row)
    metric_names = [
        "mse_norm",
        "psnr_norm_db",
        "gfc",
        "low_frequency_nrmse",
        "peak_amplitude_ratio",
        "peak_location_error_samples",
    ]
    rows: list[dict] = []
    for (condition, method), group in sorted(groups.items()):
        result = {"subset": subset_name, "condition": condition, "method": method}
        for metric_name in metric_names:
            result.update(
                {
                    f"{metric_name}_{key}": value
                    for key, value in summarize([float(item[metric_name]) for item in group]).items()
                }
            )
        rows.append(result)
    return rows


def plot_summary(summary_rows: list[dict], raw_rows: list[dict], output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    display_methods = ["No denoise", "Savitzky-Golay", "Wiener", "Wavelet", "TCN", "DnResUnet"]
    conditions = ["ID resampling", "Unseen noise", "Sparse acquisition", "Combined OOD"]
    colors = {
        "No denoise": "#9B9B9B",
        "Savitzky-Golay": "#4C78A8",
        "Wiener": "#F58518",
        "Wavelet": "#54A24B",
        "TCN": "#8E6C8A",
        "DnResUnet": "#B22222",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.08))
    fig.subplots_adjust(left=0.075, right=0.992, top=0.89, bottom=0.31, wspace=0.37)
    for condition_index, condition in enumerate(conditions):
        rows = {
            row["method"]: row
            for row in summary_rows
            if row["subset"] == "all" and row["condition"] == condition and row["method"] in display_methods
        }
        offset = (condition_index - 1.5) * 0.16
        for method_index, method in enumerate(display_methods):
            if method in rows:
                if float(rows[method]["mse_norm_mean"]) <= 0:
                    raise ValueError("Panel-a log-scale MSE values must be strictly positive.")
                axes[0].scatter(
                    method_index + offset,
                    rows[method]["mse_norm_mean"],
                    s=18,
                    color=colors[method],
                    marker=["o", "s", "^", "D"][condition_index],
                    linewidth=0,
                    zorder=2,
                )
    axes[0].set_yscale("log")
    axes[0].set_xticks(
        range(len(display_methods)),
        display_methods,
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    axes[0].set_ylabel("Normalized MSE")
    axes[0].set_title("Distribution-shift performance", pad=7, fontsize=8.5)
    axes[0].grid(axis="y", which="major", color="#D9D9D9", lw=0.55, alpha=0.75)
    axes[0].set_axisbelow(True)
    condition_markers = ["o", "s", "^", "D"]
    condition_handles = [
        axes[0].scatter([], [], marker=marker, s=18, color="#3A3A3A")
        for marker in condition_markers
    ]
    fig.legend(
        condition_handles,
        conditions,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        fontsize=6.4,
        handletextpad=0.35,
        borderaxespad=0.0,
        columnspacing=1.2,
    )

    weak_rows = [
        row
        for row in raw_rows
        if int(row["weak_long"]) == 1
        and row["condition"] == "ID resampling"
        and row["method"] in display_methods
    ]
    counts = {method: sum(row["method"] == method for row in weak_rows) for method in display_methods}
    if len(set(counts.values())) != 1 or min(counts.values()) == 0:
        raise ValueError(f"Weak-subset method counts are incomplete or unequal: {counts}")

    def interval_panel(ax: plt.Axes, metric: str) -> None:
        for method_index, method in enumerate(display_methods):
            values = np.asarray(
                [float(row[metric]) for row in weak_rows if row["method"] == method], dtype=float
            )
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"The log-scale metric {metric} must be finite and positive for {method}.")
            q025, q25, median, q75, q975 = np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975])
            color = colors[method]
            ax.vlines(method_index, q025, q975, color=color, lw=0.9, alpha=0.55, zorder=1)
            ax.vlines(method_index, q25, q75, color=color, lw=4.2, alpha=0.95, zorder=2)
            ax.scatter(
                method_index,
                median,
                s=31,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        ax.set_xticks(
            np.arange(len(display_methods)),
            display_methods,
            rotation=28,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", color="#D9D9D9", lw=0.55, alpha=0.75)
        ax.set_axisbelow(True)

    interval_panel(axes[1], "low_frequency_nrmse")
    axes[1].set_ylabel("Low-frequency NRMSE\n(lower is better)")
    axes[1].set_title("Weak long-wavelength subset", pad=7, fontsize=8.5)
    axes[1].set_ylim(0.035, 7.5)

    interval_panel(axes[2], "peak_amplitude_ratio")
    axes[2].axhline(1.0, color="#222222", lw=0.9, linestyle="--", zorder=4)
    axes[2].set_ylabel("Recovered peak / clean peak\n(closer to 1 is better)")
    axes[2].set_title("Weak-anomaly amplitude retention", pad=7, fontsize=8.5)
    axes[2].set_ylim(0.48, 23.0)
    for label, ax in zip("abc", axes):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", fontweight="bold", fontsize=8)
    prefix = output_dir / "robustness_summary"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-noise", type=int, default=120)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.05, 0.2])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--weak-peak-threshold-mgal", type=float, default=0.2)
    parser.add_argument("--long-energy-threshold", type=float, default=0.95)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code_dir))

    from DnResUnet_code import IndependentGravityGenerator, predict_clean_signal
    from independent_resampling_eval import load_checkpoint_model

    device = torch.device("cpu")
    checkpoint_names = {
        "DnResUnet": "main_model/dnresunet_v2_realistic_checkpoint.pt",
        "BasicCNN": "baselines/BasicCNN_v2_realistic_checkpoint.pt",
        "DnCNN": "baselines/DnCNN_v2_realistic_checkpoint.pt",
        "UNet1D": "baselines/UNet1D_v2_realistic_checkpoint.pt",
        "TCN": "baselines/TCN_v2_realistic_checkpoint.pt",
    }
    models = {
        name: load_checkpoint_model(args.checkpoint_dir / relative, device)
        for name, relative in checkpoint_names.items()
    }
    generator = IndependentGravityGenerator(seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    raw_rows: list[dict] = []
    sample_metadata: list[dict] = []
    for noise_std in args.noise_levels:
        for sample_index in range(args.samples_per_noise):
            noisy_norm_id, clean_norm_id, scale_id, metadata = generator.generate_sample(noise_std, True)
            clean_phys = clean_norm_id * scale_id
            id_noisy_phys = noisy_norm_id * scale_id
            sparse_clean, sparse_count, spacing_multiplier = sparse_gapped_profile(clean_phys, rng)
            cases = {
                "ID resampling": id_noisy_phys,
                "Unseen noise": clean_phys + unseen_noise(clean_phys.size, noise_std, rng),
                "Sparse acquisition": sparse_clean + rng.normal(0.0, noise_std, clean_phys.size),
                "Combined OOD": sparse_clean + unseen_noise(clean_phys.size, noise_std, rng),
            }
            clean_peak = float(np.max(np.abs(clean_phys)))
            long_fraction = spectral_low_fraction(clean_phys)
            weak_long = clean_peak <= args.weak_peak_threshold_mgal and long_fraction >= args.long_energy_threshold
            sample_metadata.append(
                {
                    "noise_std_mgal": noise_std,
                    "sample_index": sample_index,
                    "scenario": metadata["scenario"],
                    "clean_peak_mgal": clean_peak,
                    "low_frequency_energy_fraction": long_fraction,
                    "weak_long": int(weak_long),
                    "sparse_station_count": sparse_count,
                    "effective_spacing_multiplier": spacing_multiplier,
                }
            )
            for condition, noisy_phys in cases.items():
                noisy_norm, clean_norm, _ = normalize_pair(noisy_phys, clean_phys)
                predictions = {"No denoise": noisy_norm}
                predictions.update(classical_predictions(noisy_norm))
                predictions.update(
                    {
                        name: predict_model(model, device, noisy_norm, predict_clean_signal)
                        for name, model in models.items()
                    }
                )
                for method in METHOD_ORDER:
                    row = {
                        "noise_std_mgal": noise_std,
                        "sample_index": sample_index,
                        "scenario": metadata["scenario"],
                        "condition": condition,
                        "method": method,
                        "clean_peak_mgal": clean_peak,
                        "low_frequency_energy_fraction": long_fraction,
                        "weak_long": int(weak_long),
                    }
                    row.update(metrics(clean_norm, predictions[method]))
                    raw_rows.append(row)
    summary_rows = aggregate(raw_rows, "all", lambda row: True)
    summary_rows += aggregate(raw_rows, "weak_long", lambda row: int(row["weak_long"]) == 1)
    weak_count = sum(int(row["weak_long"]) for row in sample_metadata)
    if weak_count == 0:
        raise RuntimeError("No weak long-wavelength samples met the pre-specified criteria.")
    write_csv(args.output_dir / "robustness_raw_metrics.csv", raw_rows)
    write_csv(args.output_dir / "robustness_summary.csv", summary_rows)
    write_csv(args.output_dir / "robustness_sample_metadata.csv", sample_metadata)
    plot_summary(summary_rows, raw_rows, args.output_dir)
    report = {
        "samples": len(sample_metadata),
        "weak_long_samples": weak_count,
        "weak_definition": {
            "peak_threshold_mgal": args.weak_peak_threshold_mgal,
            "low_frequency_energy_fraction_threshold": args.long_energy_threshold,
        },
        "ood_conditions": {
            "Unseen noise": "Student-t, pink 1/f, quadratic chirp, and Laplace burst mixture scaled to nominal RMS.",
            "Sparse acquisition": "45-75 sampled stations with a contiguous gap, then interpolation to 512 points.",
            "Combined OOD": "Sparse/gapped acquisition plus the unseen-noise mixture.",
        },
        "summary": summary_rows,
    }
    (args.output_dir / "robustness_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"samples": len(sample_metadata), "weak_long_samples": weak_count}, indent=2))


if __name__ == "__main__":
    main()
