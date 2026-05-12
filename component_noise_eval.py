import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from DnResUnet_code import (
    IndependentGravityGenerator,
    compute_gfc,
    compute_mse,
    compute_psnr,
    predict_clean_signal,
    set_seed,
)
from forward_v2 import (
    ACQUISITION_TEMPLATES,
    CHALLENGING_KEEP_PROB,
    SIGNAL_AMPLITUDE_THRESHOLD,
    apply_depth_misalignment,
    create_random_model_v2,
    get_simulation_from_cache,
    resample_to_target,
    smooth_random_curve,
)
from independent_resampling_eval import load_checkpoint_model, tensor_from_numpy


COMPONENT_LABELS = {
    "heteroscedastic": "Heteroscedastic random noise",
    "drift": "Linear and non-linear drift",
    "colored": "Correlated colored noise",
    "step": "Step-like drift",
    "spike": "Impulsive spikes",
    "misalignment": "Mild depth-axis misalignment",
}


def make_component_noise(component, length, spacing, noise_std, rng):
    if component == "heteroscedastic":
        white_noise = rng.normal(0.0, noise_std, size=length).astype(np.float32)
        hetero_scale = np.ones(length, dtype=np.float32)
        n_segments = int(rng.integers(2, 6))
        segment_edges = np.linspace(0, length, n_segments + 1, dtype=int)
        for idx in range(n_segments):
            hetero_scale[segment_edges[idx] : segment_edges[idx + 1]] = rng.uniform(0.5, 1.8)
        return white_noise * hetero_scale

    if component == "drift":
        drift_amplitude = noise_std * rng.uniform(0.05, 0.25)
        linear_drift = np.linspace(
            rng.uniform(-drift_amplitude, drift_amplitude),
            rng.uniform(-drift_amplitude, drift_amplitude),
            length,
            dtype=np.float32,
        )
        nonlinear_drift = smooth_random_curve(
            length,
            rng,
            amplitude=noise_std * rng.uniform(0.08, 0.30),
            n_ctrl=int(rng.integers(4, 8)),
        )
        return linear_drift + nonlinear_drift

    if component == "colored":
        window_size = int(rng.integers(5, max(7, min(length // 6, 31))))
        raw = rng.normal(0.0, 1.0, size=length + window_size)
        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        filtered = np.convolve(raw, kernel, mode="valid")[:length]
        filtered_std = np.std(filtered)
        if filtered_std <= 1e-6:
            return np.zeros(length, dtype=np.float32)
        return (filtered / filtered_std * noise_std * rng.uniform(0.08, 0.25)).astype(np.float32)

    if component == "step":
        step_drift = np.zeros(length, dtype=np.float32)
        n_steps = int(rng.integers(1, 3))
        for _ in range(n_steps):
            step_idx = int(rng.integers(max(3, length // 10), max(4, length - length // 10)))
            step_amp = noise_std * rng.uniform(-0.4, 0.4)
            step_drift[step_idx:] += step_amp
        return step_drift

    if component == "spike":
        spikes = np.zeros(length, dtype=np.float32)
        n_spikes = int(rng.integers(1, 4))
        spacing_factor = np.clip(spacing / 1.0, 0.5, 5.0)
        for _ in range(n_spikes):
            spike_idx = int(rng.integers(0, length))
            spike_width = int(rng.integers(1, 4))
            spike_amp = noise_std * rng.uniform(1.5, 4.5) * rng.choice([-1.0, 1.0])
            end_idx = min(length, spike_idx + spike_width)
            spikes[spike_idx:end_idx] += spike_amp
        return (spikes * (0.8 + 0.1 * spacing_factor)).astype(np.float32)

    raise ValueError(f"Unsupported additive component: {component}")


def generate_component_sample(generator, component, noise_std, rng, target_length):
    while True:
        template = ACQUISITION_TEMPLATES[int(rng.integers(0, len(ACQUISITION_TEMPLATES)))]
        cached = get_simulation_from_cache(generator.sim_cache, generator.mesh, generator.ind_active, template)
        z_raw = cached["z_axis"]
        simulation = cached["simulation"]
        model, scenario_name = create_random_model_v2(generator.active_cc, rng)
        clean_raw = simulation.dpred(model).astype(np.float32)
        clean_peak = float(np.max(np.abs(clean_raw)))
        is_challenging = clean_peak < SIGNAL_AMPLITUDE_THRESHOLD
        if is_challenging and rng.random() > CHALLENGING_KEEP_PROB:
            continue

        if component == "misalignment":
            noisy_raw = apply_depth_misalignment(clean_raw, z_raw, template["spacing"], rng)
        else:
            noise = make_component_noise(component, len(z_raw), template["spacing"], noise_std, rng)
            noisy_raw = clean_raw + noise

        clean_norm, _ = resample_to_target(clean_raw, z_raw, target_length=target_length)
        noisy_norm, z_target = resample_to_target(noisy_raw, z_raw, target_length=target_length)
        scale = float(np.max(np.abs(noisy_norm)))
        if scale < 1e-6:
            scale = 1.0
        return (
            noisy_norm.astype(np.float32) / scale,
            clean_norm.astype(np.float32) / scale,
            {
                "depth_axis_m": z_target.astype(np.float32),
                "sample_interval_m": float(template["spacing"]),
                "raw_length": int(template["n_points"]),
                "scenario": scenario_name,
                "challenging": bool(is_challenging),
            },
        )


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def write_summary_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DnResUnet on isolated non-stationary noise components.")
    parser.add_argument("--checkpoint", default="Models_v3_realistic/dnresunet_v2_realistic_checkpoint.pt")
    parser.add_argument("--output-dir", default="component_noise_results")
    parser.add_argument("--samples-per-component", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--signal-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = load_checkpoint_model(Path(args.checkpoint), device)
    generator = IndependentGravityGenerator(length=args.signal_length, use_v2=True, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    raw_rows = []
    for component, label in COMPONENT_LABELS.items():
        noisy_metrics = {"mse": [], "psnr": [], "gfc": []}
        denoised_metrics = {"mse": [], "psnr": [], "gfc": []}
        for sample_index in range(args.samples_per_component):
            noisy_norm, clean_norm, metadata = generate_component_sample(
                generator,
                component,
                args.noise_std,
                rng,
                args.signal_length,
            )
            noisy_tensor = tensor_from_numpy(noisy_norm, device)
            clean_tensor = tensor_from_numpy(clean_norm, device)
            with torch.no_grad():
                denoised_tensor = predict_clean_signal(model, noisy_tensor)

            noisy_values = {
                "mse": compute_mse(clean_tensor, noisy_tensor),
                "psnr": compute_psnr(clean_tensor, noisy_tensor, max_i=1.0),
                "gfc": compute_gfc(clean_tensor, noisy_tensor),
            }
            denoised_values = {
                "mse": compute_mse(clean_tensor, denoised_tensor),
                "psnr": compute_psnr(clean_tensor, denoised_tensor, max_i=1.0),
                "gfc": compute_gfc(clean_tensor, denoised_tensor),
            }
            for key in noisy_metrics:
                noisy_metrics[key].append(noisy_values[key])
                denoised_metrics[key].append(denoised_values[key])
            raw_rows.append(
                {
                    "component": component,
                    "component_label": label,
                    "sample_index": sample_index,
                    "sample_interval_m": metadata["sample_interval_m"],
                    "raw_length": metadata["raw_length"],
                    "scenario": metadata["scenario"],
                    "noisy_mse": noisy_values["mse"],
                    "denoised_mse": denoised_values["mse"],
                    "noisy_psnr": noisy_values["psnr"],
                    "denoised_psnr": denoised_values["psnr"],
                    "noisy_gfc": noisy_values["gfc"],
                    "denoised_gfc": denoised_values["gfc"],
                }
            )

        noisy_mse_mean, noisy_mse_std = mean_std(noisy_metrics["mse"])
        den_mse_mean, den_mse_std = mean_std(denoised_metrics["mse"])
        noisy_psnr_mean, noisy_psnr_std = mean_std(noisy_metrics["psnr"])
        den_psnr_mean, den_psnr_std = mean_std(denoised_metrics["psnr"])
        noisy_gfc_mean, noisy_gfc_std = mean_std(noisy_metrics["gfc"])
        den_gfc_mean, den_gfc_std = mean_std(denoised_metrics["gfc"])
        reduction = 100.0 * (1.0 - den_mse_mean / max(noisy_mse_mean, 1e-12))
        summary_rows.append(
            {
                "component": component,
                "component_label": label,
                "samples": args.samples_per_component,
                "noise_std_mgal": args.noise_std,
                "noisy_mse_mean": noisy_mse_mean,
                "noisy_mse_std": noisy_mse_std,
                "denoised_mse_mean": den_mse_mean,
                "denoised_mse_std": den_mse_std,
                "mse_reduction_percent": reduction,
                "noisy_psnr_mean": noisy_psnr_mean,
                "noisy_psnr_std": noisy_psnr_std,
                "denoised_psnr_mean": den_psnr_mean,
                "denoised_psnr_std": den_psnr_std,
                "noisy_gfc_mean": noisy_gfc_mean,
                "noisy_gfc_std": noisy_gfc_std,
                "denoised_gfc_mean": den_gfc_mean,
                "denoised_gfc_std": den_gfc_std,
            }
        )

    write_summary_csv(summary_rows, output_dir / "component_noise_summary.csv")
    write_summary_csv(raw_rows, output_dir / "component_noise_raw.csv")
    print(f"Wrote {output_dir / 'component_noise_summary.csv'}")
    print(f"Wrote {output_dir / 'component_noise_raw.csv'}")


if __name__ == "__main__":
    main()
