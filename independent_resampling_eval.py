import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from DnResUnet_code import (
    BasicCNN1D,
    DnCNN1D,
    IndependentGravityGenerator,
    ImprovedDnResUNet,
    LegacyDnResUNet,
    TCN1D,
    UNet1DBaseline,
    compute_gfc,
    compute_mse,
    compute_psnr,
    predict_clean_signal,
    set_seed,
    torch,
)

try:
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    from scipy.signal import savgol_filter, spectrogram, wiener
    from skimage.restoration import denoise_wavelet
except ImportError as exc:
    raise SystemExit(
        "independent_resampling_eval.py requires matplotlib, scipy, scikit-image, and PyWavelets."
    ) from exc


TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
PANEL_FONTSIZE = 14


def apply_plot_style():
    plt.rcParams.update(
        {
            "font.size": LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
        }
    )


def denoise_signal_wavelet(signal_noisy_np):
    return denoise_wavelet(
        signal_noisy_np,
        wavelet="sym8",
        mode="soft",
        wavelet_levels=5,
        method="VisuShrink",
        rescale_sigma=True,
    )


def denoise_signal_savgol(signal_noisy_np):
    return savgol_filter(signal_noisy_np, window_length=31, polyorder=5, mode="mirror")


def denoise_signal_wiener(signal_noisy_np):
    return wiener(signal_noisy_np, mysize=21)


def ci_stats(values, confidence):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = z_value * std / max(len(values), 1) ** 0.5
    return {"n": int(len(values)), "mean": mean, "std": std, "ci_low": mean - margin, "ci_high": mean + margin}


def tensor_from_numpy(signal_np, device):
    return torch.from_numpy(signal_np).float().view(1, 1, -1).to(device)


def write_csv(rows, output_path):
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def get_frequency_axis(depth_axis):
    if len(depth_axis) < 2:
        return np.arange(0), 1.0
    spacing = float(abs(depth_axis[1] - depth_axis[0]))
    freq_axis = fftfreq(len(depth_axis), d=spacing)[: len(depth_axis) // 2]
    return freq_axis, spacing


def plot_well_log(ax, depth_axis, noisy_phys, clean_phys, pred_phys, method_label, noise_std, color):
    ax.plot(noisy_phys, depth_axis, color="#f26b6b", alpha=0.35, linewidth=1.0, label="Noisy")
    ax.plot(clean_phys, depth_axis, color="black", linestyle="--", linewidth=1.6, label="Ground Truth")
    ax.plot(pred_phys, depth_axis, color=color, linewidth=1.8, label=method_label)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(f"{method_label}\nNoise {noise_std} Sample", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Gravity Anomaly (mGal)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Depth (m)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def plot_frequency_domain(ax, depth_axis, noisy_phys, clean_phys, pred_phys, method_label, noise_std, color):
    freq_axis, _ = get_frequency_axis(depth_axis)
    for label, signal, line_color, style, alpha in [
        ("Noisy Signal", noisy_phys, "#f26b6b", "-", 0.35),
        ("Ground Truth", clean_phys, "black", "--", 1.0),
        (method_label, pred_phys, color, "-", 1.0),
    ]:
        amplitude = 2.0 / len(signal) * np.abs(fft(signal)[: len(signal) // 2])
        ax.plot(freq_axis, amplitude, color=line_color, linestyle=style, alpha=alpha, linewidth=1.5, label=label)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(f"{method_label} Freq Domain\nNoise {noise_std} Sample", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Spatial Frequency (1/m)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Amplitude Spectrum", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def plot_signal_spectrogram(ax, depth_axis, signal_phys, title):
    _, spacing = get_frequency_axis(depth_axis)
    fs = 1.0 / spacing if spacing > 0 else 1.0
    freq, depth, power = spectrogram(signal_phys, fs=fs, window="hann", nperseg=64, noverlap=60, nfft=512)
    mesh = ax.pcolormesh(depth_axis.min() + depth, freq, 10.0 * np.log10(power + 1e-15), shading="gouraud", cmap="jet")
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Depth (m)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Frequency (1/m)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    return mesh


def add_panel_labels(fig, axes, labels):
    for ax, label in zip(axes, labels):
        ax.text(0.5, -0.16, label, transform=ax.transAxes, ha="center", va="top", fontsize=PANEL_FONTSIZE)


def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    name = checkpoint_path.name
    if name.startswith("gravity_model_v2"):
        model = LegacyDnResUNet()
        model.load_state_dict(state_dict)
    elif name.startswith("BasicCNN"):
        model = BasicCNN1D()
        if any(key.startswith("basiccnn.") for key in state_dict.keys()):
            state_dict = {
                key.replace("basiccnn.", "net.", 1): value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict)
    elif name.startswith("DnCNN"):
        model = DnCNN1D()
        if any(key.startswith("dncnn.") for key in state_dict.keys()):
            state_dict = {
                key.replace("dncnn.", "net.", 1): value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict)
    elif name.startswith("UNet1D"):
        model = UNet1DBaseline()
        model.load_state_dict(state_dict)
    elif name.startswith("TCN"):
        model = TCN1D()
        model.load_state_dict(state_dict)
    else:
        config = checkpoint.get("config", {})
        if config.get("architecture", "improved") == "legacy":
            model = LegacyDnResUNet(kernel_size=config.get("kernel_size", 7))
        elif config.get("architecture") == "unet1d":
            model = UNet1DBaseline(
                kernel_size=config.get("kernel_size", 5),
                norm=config.get("norm", "group"),
                activation=config.get("activation", "relu"),
                dropout=config.get("dropout", 0.05),
            )
        elif config.get("architecture") == "tcn":
            model = TCN1D(
                kernel_size=config.get("kernel_size", 5),
                norm=config.get("norm", "group"),
                activation=config.get("activation", "relu"),
                dropout=config.get("dropout", 0.05),
            )
        else:
            model = ImprovedDnResUNet(
                kernel_size=config.get("kernel_size", 7),
                norm=config.get("norm", "group"),
                activation=config.get("activation", "relu"),
                dropout=config.get("dropout", 0.05),
                use_residual_blocks=config.get("use_residual_blocks", True),
                predict_noise=config.get("predict_noise", True),
            )
        model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def summarize_metrics(metric_store, confidence):
    rows = []
    for noise_std, methods in metric_store.items():
        for method_name, scores in methods.items():
            mse_stats = ci_stats(scores["mse"], confidence)
            psnr_stats = ci_stats(scores["psnr"], confidence)
            gfc_stats = ci_stats(scores["gfc"], confidence)
            rows.append(
                {
                    "noise_std": noise_std,
                    "method": method_name,
                    "samples": mse_stats["n"],
                    "mse_mean": mse_stats["mean"],
                    "mse_std": mse_stats["std"],
                    "mse_ci_low": mse_stats["ci_low"],
                    "mse_ci_high": mse_stats["ci_high"],
                    "psnr_mean": psnr_stats["mean"],
                    "psnr_std": psnr_stats["std"],
                    "psnr_ci_low": psnr_stats["ci_low"],
                    "psnr_ci_high": psnr_stats["ci_high"],
                    "gfc_mean": gfc_stats["mean"],
                    "gfc_std": gfc_stats["std"],
                    "gfc_ci_low": gfc_stats["ci_low"],
                    "gfc_ci_high": gfc_stats["ci_high"],
                }
            )
    return rows


def plot_dnresunet_noise_panels(representatives, output_dir, paper_fig_dir=None):
    color = "#2445ff"
    noise_to_name = {
        0.2: "high_noise_comparison",
        0.05: "medium_noise_comparison",
        0.01: "low_noise_comparison",
    }
    for noise_std in [0.2, 0.05, 0.01]:
        rep = representatives[noise_std]
        depth_axis = rep["metadata"]["depth_axis_m"]
        scale = rep["scale"]
        noisy_phys = rep["noisy"] * scale
        clean_phys = rep["clean"] * scale
        denoised_phys = rep["predictions"]["DnResUnet"] * scale

        fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.8), constrained_layout=True)
        plot_well_log(axes[0, 0], depth_axis, noisy_phys, clean_phys, denoised_phys, "DnResUnet", noise_std, color)
        plot_frequency_domain(axes[0, 1], depth_axis, noisy_phys, clean_phys, denoised_phys, "DnResUnet", noise_std, color)
        mesh1 = plot_signal_spectrogram(axes[1, 0], depth_axis, noisy_phys, f"Spectrogram (Noisy)\nNoise {noise_std} Sample")
        mesh2 = plot_signal_spectrogram(axes[1, 1], depth_axis, denoised_phys, f"Spectrogram (DnResUnet)\nNoise {noise_std} Sample")
        cbar1 = fig.colorbar(mesh1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar2 = fig.colorbar(mesh2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar1.set_label("Power (dB)", fontsize=LABEL_FONTSIZE)
        cbar2.set_label("Power (dB)", fontsize=LABEL_FONTSIZE)
        cbar1.ax.tick_params(labelsize=TICK_FONTSIZE)
        cbar2.ax.tick_params(labelsize=TICK_FONTSIZE)
        add_panel_labels(fig, [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], ["(a)", "(b)", "(c)", "(d)"])

        file_stem = noise_to_name[noise_std]
        save_figure(fig, output_dir / f"{file_stem}.pdf")
        save_figure(fig, output_dir / f"{file_stem}.png")
        if paper_fig_dir is not None:
            save_figure(fig, paper_fig_dir / f"{file_stem}.pdf")
        plt.close(fig)


def plot_method_grid_time(representatives, methods, labels, colors, output_pdf, paper_fig_dir=None):
    noise_order = [0.01, 0.05, 0.2]
    fig_width = max(14.4, 4.1 * len(methods))
    fig, axes = plt.subplots(len(noise_order), len(methods), figsize=(fig_width, 17.8), constrained_layout=True)
    for row_idx, noise_std in enumerate(noise_order):
        rep = representatives[noise_std]
        depth_axis = rep["metadata"]["depth_axis_m"]
        scale = rep["scale"]
        noisy_phys = rep["noisy"] * scale
        clean_phys = rep["clean"] * scale
        for col_idx, (method_name, label, color) in enumerate(zip(methods, labels, colors)):
            pred_phys = rep["predictions"][method_name] * scale
            ax = axes[row_idx, col_idx]
            plot_well_log(ax, depth_axis, noisy_phys, clean_phys, pred_phys, label, noise_std, color)
            if row_idx < len(noise_order) - 1:
                ax.set_xlabel("")
            if col_idx > 0:
                ax.set_ylabel("")
    save_figure(fig, output_pdf)
    save_figure(fig, output_pdf.with_suffix(".png"))
    if paper_fig_dir is not None:
        save_figure(fig, paper_fig_dir / output_pdf.name)
    plt.close(fig)


def plot_method_grid_frequency(representatives, methods, labels, colors, output_pdf, paper_fig_dir=None):
    noise_order = [0.01, 0.05, 0.2]
    fig_width = max(14.8, 4.2 * len(methods))
    fig, axes = plt.subplots(len(noise_order), len(methods), figsize=(fig_width, 16.8), constrained_layout=True)
    for row_idx, noise_std in enumerate(noise_order):
        rep = representatives[noise_std]
        depth_axis = rep["metadata"]["depth_axis_m"]
        scale = rep["scale"]
        noisy_phys = rep["noisy"] * scale
        clean_phys = rep["clean"] * scale
        for col_idx, (method_name, label, color) in enumerate(zip(methods, labels, colors)):
            pred_phys = rep["predictions"][method_name] * scale
            ax = axes[row_idx, col_idx]
            plot_frequency_domain(ax, depth_axis, noisy_phys, clean_phys, pred_phys, label, noise_std, color)
            if row_idx < len(noise_order) - 1:
                ax.set_xlabel("")
            if col_idx > 0:
                ax.set_ylabel("")
    save_figure(fig, output_pdf)
    save_figure(fig, output_pdf.with_suffix(".png"))
    if paper_fig_dir is not None:
        save_figure(fig, paper_fig_dir / output_pdf.name)
    plt.close(fig)


def plot_method_grid_spectrogram(representatives, methods, labels, output_pdf, paper_fig_dir=None):
    noise_order = [0.01, 0.05, 0.2]
    fig_width = max(14.8, 4.1 * len(methods))
    fig, axes = plt.subplots(len(noise_order), len(methods), figsize=(fig_width, 14.2), constrained_layout=True)
    for row_idx, noise_std in enumerate(noise_order):
        rep = representatives[noise_std]
        depth_axis = rep["metadata"]["depth_axis_m"]
        scale = rep["scale"]
        for col_idx, (method_name, label) in enumerate(zip(methods, labels)):
            pred_phys = rep["predictions"][method_name] * scale
            ax = axes[row_idx, col_idx]
            plot_signal_spectrogram(ax, depth_axis, pred_phys, f"{label}\nNoise {noise_std} Sample")
            if row_idx < len(noise_order) - 1:
                ax.set_xlabel("")
            if col_idx > 0:
                ax.set_ylabel("")
    save_figure(fig, output_pdf)
    save_figure(fig, output_pdf.with_suffix(".png"))
    if paper_fig_dir is not None:
        save_figure(fig, paper_fig_dir / output_pdf.name)
    plt.close(fig)


def plot_convergence_figure(history_csv, output_pdf, paper_fig_dir=None):
    rows = []
    with history_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        return
    epochs = np.array([int(row["epoch"]) for row in rows], dtype=np.int32)
    train_loss = np.array([float(row["train_loss"]) for row in rows], dtype=np.float64)
    valid_loss = np.array([float(row["valid_loss"]) for row in rows], dtype=np.float64)
    valid_psnr = np.array([float(row["valid_psnr"]) for row in rows], dtype=np.float64)
    valid_gfc = np.array([float(row["valid_gfc"]) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), constrained_layout=True)
    axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=1.8, label="Train")
    axes[0].plot(epochs, valid_loss, color="#d62728", linewidth=1.8, label="Validation")
    axes[0].set_title("Hybrid Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=LEGEND_FONTSIZE)

    axes[1].plot(epochs, valid_psnr, color="#d62728", linewidth=1.8)
    axes[1].set_title("Validation PSNR")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].plot(epochs, valid_gfc, color="#d62728", linewidth=1.8)
    axes[2].set_title("Validation GFC")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    add_panel_labels(fig, axes, ["(a)", "(b)", "(c)"])
    save_figure(fig, output_pdf)
    save_figure(fig, output_pdf.with_suffix(".png"))
    if paper_fig_dir is not None:
        save_figure(fig, paper_fig_dir / output_pdf.name)
    plt.close(fig)


def write_field_review_template(output_dir):
    template_path = output_dir / "field_review_template.csv"
    rows = [
        {
            "sample_id": "",
            "reviewer": "",
            "method": "",
            "anomaly_continuity_score_1_5": "",
            "background_stability_score_1_5": "",
            "agreement_with_traditional_workflow_1_5": "",
            "geological_plausibility_notes": "",
            "recommended_for_interpretation_yes_no": "",
        }
    ]
    write_csv(rows, template_path)
    return template_path


def parse_args():
    parser = argparse.ArgumentParser(description="Large-sample independent resampling evaluation.")
    parser.add_argument("--output-dir", default="comparison_results_independent_large")
    parser.add_argument("--dnresunet-path", default="Models_v3_realistic/dnresunet_v2_realistic_checkpoint.pt")
    parser.add_argument("--basiccnn-path", default="Models_Benchmark_v2/BasicCNN_v2_realistic_checkpoint.pt")
    parser.add_argument("--dncnn-path", default="Models_Benchmark_v2/DnCNN_v2_realistic_checkpoint.pt")
    parser.add_argument("--unet1d-path", default="Models_Benchmark_v2/UNet1D_v2_realistic_checkpoint.pt")
    parser.add_argument("--tcn-path", default="Models_Benchmark_v2/TCN_v2_realistic_checkpoint.pt")
    parser.add_argument("--samples-per-noise", type=int, default=200)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.01, 0.05, 0.2])
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--signal-length", type=int, default=512)
    parser.add_argument("--spacing-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-legacy-generator", action="store_true")
    parser.add_argument("--paper-fig-dir", default="")
    parser.add_argument("--history-csv", default="Models_v3_realistic/dnresunet_v2_realistic_history.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    apply_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_fig_dir = Path(args.paper_fig_dir) if args.paper_fig_dir else None
    if paper_fig_dir is not None:
        paper_fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = {
        "DnResUnet": load_checkpoint_model(Path(args.dnresunet_path), device),
        "BasicCNN": load_checkpoint_model(Path(args.basiccnn_path), device),
        "DnCNN": load_checkpoint_model(Path(args.dncnn_path), device),
        "UNet1D": load_checkpoint_model(Path(args.unet1d_path), device),
        "TCN": load_checkpoint_model(Path(args.tcn_path), device),
    }
    generator = IndependentGravityGenerator(
        length=args.signal_length,
        spacing_m=args.spacing_m,
        use_v2=not args.use_legacy_generator,
        seed=args.seed,
    )
    metric_store = {}
    raw_rows = []
    representatives = {}

    for noise_std in args.noise_levels:
        metric_store[noise_std] = {
            "No Denoise": {"mse": [], "psnr": [], "gfc": []},
            "Wavelet": {"mse": [], "psnr": [], "gfc": []},
            "S-G Filter": {"mse": [], "psnr": [], "gfc": []},
            "Wiener": {"mse": [], "psnr": [], "gfc": []},
            "DnResUnet": {"mse": [], "psnr": [], "gfc": []},
            "BasicCNN": {"mse": [], "psnr": [], "gfc": []},
            "DnCNN": {"mse": [], "psnr": [], "gfc": []},
            "UNet1D": {"mse": [], "psnr": [], "gfc": []},
            "TCN": {"mse": [], "psnr": [], "gfc": []},
        }
        representative = None
        for sample_index in range(args.samples_per_noise):
            noisy_norm, clean_norm, scale, metadata = generator.generate_sample(noise_std, return_metadata=True)
            predictions = {
                "No Denoise": noisy_norm,
                "Wavelet": denoise_signal_wavelet(noisy_norm),
                "S-G Filter": denoise_signal_savgol(noisy_norm),
                "Wiener": denoise_signal_wiener(noisy_norm),
            }
            input_tensor = tensor_from_numpy(noisy_norm, device)
            with torch.no_grad():
                for name, model in models.items():
                    predictions[name] = predict_clean_signal(model, input_tensor).detach().cpu().numpy().squeeze()

            clean_tensor = tensor_from_numpy(clean_norm, device)
            for method_name, prediction in predictions.items():
                pred_tensor = tensor_from_numpy(np.asarray(prediction, dtype=np.float32), device)
                mse_value = compute_mse(clean_tensor, pred_tensor)
                psnr_value = compute_psnr(clean_tensor, pred_tensor, max_i=1.0)
                gfc_value = compute_gfc(clean_tensor, pred_tensor)
                metric_store[noise_std][method_name]["mse"].append(mse_value)
                metric_store[noise_std][method_name]["psnr"].append(psnr_value)
                metric_store[noise_std][method_name]["gfc"].append(gfc_value)
                raw_rows.append(
                    {
                        "noise_std": noise_std,
                        "sample_index": sample_index,
                        "method": method_name,
                        "mse": mse_value,
                        "psnr": psnr_value,
                        "gfc": gfc_value,
                    }
                )
            if sample_index == args.samples_per_noise - 1:
                representative = {
                    "noisy": noisy_norm,
                    "clean": clean_norm,
                    "predictions": predictions,
                    "scale": scale,
                    "metadata": metadata,
                }

        if representative is not None:
            representatives[noise_std] = representative

    summary_rows = summarize_metrics(metric_store, args.confidence)
    write_csv(raw_rows, output_dir / "independent_resampling_raw_metrics.csv")
    write_csv(summary_rows, output_dir / "independent_resampling_summary.csv")
    template_path = write_field_review_template(output_dir)

    if {0.01, 0.05, 0.2}.issubset(set(representatives.keys())):
        plot_dnresunet_noise_panels(representatives, output_dir, paper_fig_dir=paper_fig_dir)
        plot_method_grid_time(
            representatives,
            methods=["S-G Filter", "Wiener", "Wavelet"],
            labels=["Savitzky-Golay Filter", "Wiener Filter", "Wavelet Denoising"],
            colors=["#f39c12", "#8e44ad", "#2e8b57"],
            output_pdf=output_dir / "benchmark_comparison.pdf",
            paper_fig_dir=paper_fig_dir,
        )
        plot_method_grid_frequency(
            representatives,
            methods=["S-G Filter", "Wiener", "Wavelet"],
            labels=["Savitzky-Golay Filter", "Wiener Filter", "Wavelet Denoising"],
            colors=["#f39c12", "#8e44ad", "#2e8b57"],
            output_pdf=output_dir / "spectrum_comparison.pdf",
            paper_fig_dir=paper_fig_dir,
        )
        plot_method_grid_spectrogram(
            representatives,
            methods=["S-G Filter", "Wiener", "Wavelet"],
            labels=["Savitzky-Golay Filter", "Wiener Filter", "Wavelet Denoising"],
            output_pdf=output_dir / "spectrogram_comparison.pdf",
            paper_fig_dir=paper_fig_dir,
        )
        plot_method_grid_time(
            representatives,
            methods=["BasicCNN", "DnCNN", "UNet1D", "TCN"],
            labels=["BasicCNN", "DnCNN", "UNet1D", "TCN"],
            colors=["#6d4c41", "#ff00ff", "#008b8b", "#c0392b"],
            output_pdf=output_dir / "dl_baseline_time.pdf",
            paper_fig_dir=paper_fig_dir,
        )
        plot_method_grid_frequency(
            representatives,
            methods=["BasicCNN", "DnCNN", "UNet1D", "TCN"],
            labels=["BasicCNN", "DnCNN", "UNet1D", "TCN"],
            colors=["#6d4c41", "#ff00ff", "#008b8b", "#c0392b"],
            output_pdf=output_dir / "dl_baseline_freq.pdf",
            paper_fig_dir=paper_fig_dir,
        )
        plot_method_grid_spectrogram(
            representatives,
            methods=["BasicCNN", "DnCNN", "UNet1D", "TCN"],
            labels=["BasicCNN", "DnCNN", "UNet1D", "TCN"],
            output_pdf=output_dir / "dl_baseline_spectrogram.pdf",
            paper_fig_dir=paper_fig_dir,
        )

    history_csv = Path(args.history_csv)
    if history_csv.exists():
        plot_convergence_figure(history_csv, output_dir / "convergence_analysis.pdf", paper_fig_dir=paper_fig_dir)

    report = {
        "samples_per_noise": args.samples_per_noise,
        "confidence": args.confidence,
        "summary_csv": str(output_dir / "independent_resampling_summary.csv"),
        "raw_csv": str(output_dir / "independent_resampling_raw_metrics.csv"),
        "field_review_template": str(template_path),
        "paper_fig_dir": str(paper_fig_dir) if paper_fig_dir is not None else "",
    }
    (output_dir / "independent_resampling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
