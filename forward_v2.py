import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from discretize import TensorMesh
from discretize.utils import active_from_xyz, mkvc
from simpeg import maps
from simpeg.potential_fields import gravity


DEFAULT_NOISE_LEVELS = [0.01, 0.025, 0.05, 0.1, 0.2]
TARGET_LENGTH = 512
TARGET_CHANNELS = 1
SIGNAL_AMPLITUDE_THRESHOLD = 0.01
CHALLENGING_KEEP_PROB = 0.35
TARGET_LENGTH_DESCRIPTION = (
    "All samples are resampled to 512 points for network training after "
    "variable-spacing forward modeling."
)

# Finite acquisition templates keep simulation caching practical while still
# exposing the model to multiple logging spacings and interval lengths.
ACQUISITION_TEMPLATES = [
    {"spacing": 0.5, "n_points": 257, "z_start": -10.0},
    {"spacing": 0.5, "n_points": 385, "z_start": -20.0},
    {"spacing": 0.5, "n_points": 513, "z_start": -30.0},
    {"spacing": 1.0, "n_points": 257, "z_start": -10.0},
    {"spacing": 1.0, "n_points": 385, "z_start": -20.0},
    {"spacing": 1.0, "n_points": 513, "z_start": -30.0},
    {"spacing": 2.0, "n_points": 129, "z_start": -10.0},
    {"spacing": 2.0, "n_points": 257, "z_start": -20.0},
    {"spacing": 2.0, "n_points": 385, "z_start": -30.0},
    {"spacing": 5.0, "n_points": 97, "z_start": -10.0},
    {"spacing": 5.0, "n_points": 129, "z_start": -20.0},
    {"spacing": 5.0, "n_points": 161, "z_start": -30.0},
]

WORKER_CONTEXT = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a more realistic borehole-gravity synthetic dataset."
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=20000,
        help="Number of valid samples to generate for each noise level.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_LEVELS,
        help="Noise standard deviations in mGal.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gravity_dataset_100k_V2_REALISTIC.pt",
        help="Output .pt filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260312,
        help="Random seed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of worker processes for parallel generation.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Samples per shard task before merge.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="",
        help="Temporary directory for shard files. Defaults to <output_stem>_shards.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep intermediate shard files after the final dataset is merged.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip shard tasks that already exist in the shard directory.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing shard files into the final dataset.",
    )
    return parser.parse_args()


def build_mesh_and_active_cells():
    dh = 5.0
    hx = [(dh, 8, -1.3), (dh, 48), (dh, 8, 1.3)]
    hy = [(dh, 8, -1.3), (dh, 48), (dh, 8, 1.3)]
    hz = [(dh, 8, -1.3), (dh, 28)]
    mesh = TensorMesh([hx, hy, hz], "CCN")

    x_topo, y_topo = np.meshgrid(
        np.linspace(-260, 260, 53),
        np.linspace(-260, 260, 53),
    )
    z_topo = (
        -12.0 * np.exp(-(x_topo**2 + y_topo**2) / 85.0**2)
        - 0.015 * x_topo
        + 0.008 * y_topo
    )
    topo_xyz = np.c_[mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)]

    ind_active = active_from_xyz(mesh, topo_xyz)
    return mesh, ind_active


def build_simulation(mesh, ind_active, receiver_locations):
    receiver = gravity.receivers.Point(receiver_locations, components=["gz"])
    source_field = gravity.sources.SourceField(receiver_list=[receiver])
    survey = gravity.survey.Survey(source_field)
    model_map = maps.IdentityMap(nP=int(ind_active.sum()))
    return gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind_active,
        store_sensitivities="forward_only",
        engine="choclo",
    )


def sample_density(rng, min_abs=0.03, max_abs=0.5):
    value = rng.uniform(-max_abs, max_abs)
    while abs(value) < min_abs:
        value = rng.uniform(-max_abs, max_abs)
    return value


def resample_to_target(signal, z_raw, target_length=TARGET_LENGTH):
    z_target = np.linspace(z_raw[0], z_raw[-1], target_length, dtype=np.float32)
    signal_interp = np.interp(z_target[::-1], z_raw[::-1], signal[::-1])[::-1]
    return signal_interp.astype(np.float32), z_target


def smooth_random_curve(length, rng, amplitude, n_ctrl):
    ctrl_x = np.linspace(0, length - 1, n_ctrl)
    ctrl_y = rng.uniform(-amplitude, amplitude, size=n_ctrl)
    curve = np.interp(np.arange(length), ctrl_x, ctrl_y)
    return curve.astype(np.float32)


def block_mask(cell_centers, center, sizes):
    x1, x2 = center[0] - sizes[0] / 2.0, center[0] + sizes[0] / 2.0
    y1, y2 = center[1] - sizes[1] / 2.0, center[1] + sizes[1] / 2.0
    z1, z2 = center[2] - sizes[2] / 2.0, center[2] + sizes[2] / 2.0
    return (
        (cell_centers[:, 0] >= x1)
        & (cell_centers[:, 0] <= x2)
        & (cell_centers[:, 1] >= y1)
        & (cell_centers[:, 1] <= y2)
        & (cell_centers[:, 2] >= z1)
        & (cell_centers[:, 2] <= z2)
    )


def sphere_mask(cell_centers, center, radius):
    distance_sq = np.sum((cell_centers - np.asarray(center)) ** 2, axis=1)
    return distance_sq <= radius**2


def ellipsoid_mask(cell_centers, center, radii):
    norm_dist_sq = (
        ((cell_centers[:, 0] - center[0]) / radii[0]) ** 2
        + ((cell_centers[:, 1] - center[1]) / radii[1]) ** 2
        + ((cell_centers[:, 2] - center[2]) / radii[2]) ** 2
    )
    return norm_dist_sq <= 1.0


def assign_layer_values(layer_indicator, values):
    model = np.zeros(layer_indicator.shape[0], dtype=np.float32)
    for idx, value in enumerate(values):
        model[layer_indicator == idx] = value
    return model


def evaluate_layered_model(shifted_depth, boundaries, values):
    n_layers = len(values)
    tops = np.concatenate(([np.inf], boundaries, [-np.inf]))
    indicator = np.zeros(shifted_depth.shape[0], dtype=np.int32)
    for idx in range(n_layers):
        mask = (shifted_depth < tops[idx]) & (shifted_depth >= tops[idx + 1])
        indicator[mask] = idx
    return assign_layer_values(indicator, values)


def create_horizontal_layers(active_cc, rng):
    n_layers = int(rng.integers(2, 6))
    boundaries = np.sort(rng.uniform(-520.0, -40.0, size=n_layers - 1))[::-1]
    values = np.array([sample_density(rng, min_abs=0.04, max_abs=0.35) for _ in range(n_layers)])
    return evaluate_layered_model(active_cc[:, 2], boundaries, values)


def create_dipping_layers(active_cc, rng):
    n_layers = int(rng.integers(3, 6))
    boundaries = np.sort(rng.uniform(-520.0, -50.0, size=n_layers - 1))[::-1]
    dip_x = rng.uniform(-0.18, 0.18)
    dip_y = rng.uniform(-0.12, 0.12)
    shifted_depth = active_cc[:, 2] - dip_x * active_cc[:, 0] - dip_y * active_cc[:, 1]
    values = np.array([sample_density(rng, min_abs=0.04, max_abs=0.32) for _ in range(n_layers)])
    return evaluate_layered_model(shifted_depth, boundaries, values)


def create_faulted_layers(active_cc, rng):
    n_layers = int(rng.integers(3, 6))
    boundaries = np.sort(rng.uniform(-520.0, -50.0, size=n_layers - 1))[::-1]
    values = np.array([sample_density(rng, min_abs=0.04, max_abs=0.32) for _ in range(n_layers)])
    dip_x = rng.uniform(-0.18, 0.18)
    dip_y = rng.uniform(-0.12, 0.12)
    shifted_depth = active_cc[:, 2] - dip_x * active_cc[:, 0] - dip_y * active_cc[:, 1]
    base = evaluate_layered_model(shifted_depth, boundaries, values)

    fault_angle = rng.uniform(0.0, np.pi)
    throw = rng.uniform(20.0, 120.0)
    offset = rng.uniform(-60.0, 60.0)
    signed_distance = (
        np.cos(fault_angle) * active_cc[:, 0]
        + np.sin(fault_angle) * active_cc[:, 1]
        - offset
    )
    shifted_depth_faulted = shifted_depth.copy()
    shifted_depth_faulted[signed_distance > 0.0] -= throw
    shifted_layers = evaluate_layered_model(shifted_depth_faulted, boundaries, values)
    return np.where(signed_distance > 0.0, shifted_layers, base).astype(np.float32)


def create_thin_beds(active_cc, rng):
    model = np.full(active_cc.shape[0], rng.uniform(-0.03, 0.03), dtype=np.float32)
    n_beds = int(rng.integers(4, 9))
    center_depth = rng.uniform(-420.0, -120.0)
    current_top = center_depth + rng.uniform(40.0, 120.0)
    sign = 1.0
    for _ in range(n_beds):
        thickness = rng.uniform(5.0, 25.0)
        bottom = current_top - thickness
        density = sign * rng.uniform(0.04, 0.18)
        mask = (active_cc[:, 2] <= current_top) & (active_cc[:, 2] > bottom)
        model[mask] += density
        current_top = bottom - rng.uniform(4.0, 20.0)
        sign *= -1.0
    return np.clip(model, -0.45, 0.45).astype(np.float32)


def create_background_gradient(active_cc, rng):
    grad_z = rng.uniform(-0.0008, 0.0008)
    grad_x = rng.uniform(-0.0005, 0.0005)
    grad_y = rng.uniform(-0.0005, 0.0005)
    intercept = rng.uniform(-0.05, 0.05)
    model = (
        intercept
        + grad_z * (active_cc[:, 2] + 250.0)
        + grad_x * active_cc[:, 0]
        + grad_y * active_cc[:, 1]
    )
    return np.clip(model, -0.22, 0.22).astype(np.float32)


def add_random_anomaly(model, active_cc, rng, near_borehole=None, weak=False, strong=False):
    anomaly_type = rng.choice(["block", "sphere", "ellipsoid"])

    if near_borehole is True:
        center_xy = rng.uniform(-25.0, 25.0, size=2)
    elif near_borehole is False:
        radius = rng.uniform(80.0, 180.0)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        center_xy = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    else:
        center_xy = rng.uniform(-190.0, 190.0, size=2)

    center = np.array(
        [center_xy[0], center_xy[1], rng.uniform(-620.0, -40.0)],
        dtype=np.float32,
    )

    if weak:
        density = sample_density(rng, min_abs=0.03, max_abs=0.12)
    elif strong:
        density = sample_density(rng, min_abs=0.12, max_abs=0.5)
    else:
        density = sample_density(rng, min_abs=0.04, max_abs=0.45)

    if anomaly_type == "block":
        sizes = np.array(
            [
                rng.uniform(10.0, 80.0),
                rng.uniform(10.0, 80.0),
                rng.uniform(10.0, 60.0),
            ]
        )
        mask = block_mask(active_cc, center, sizes)
    elif anomaly_type == "sphere":
        radius = rng.uniform(10.0, 55.0)
        mask = sphere_mask(active_cc, center, radius)
    else:
        radii = np.array(
            [
                rng.uniform(12.0, 70.0),
                rng.uniform(12.0, 70.0),
                rng.uniform(10.0, 55.0),
            ]
        )
        mask = ellipsoid_mask(active_cc, center, radii)

    model[mask] += density
    return model


def create_random_model_v2(active_cc, rng):
    scenario = rng.choice(
        [
            "layered",
            "dipping",
            "faulted",
            "thin_beds",
            "gradient",
            "hybrid",
            "near_far_combo",
        ]
    )

    if scenario == "layered":
        model = create_horizontal_layers(active_cc, rng)
        if rng.random() < 0.55:
            for _ in range(int(rng.integers(1, 3))):
                model = add_random_anomaly(model, active_cc, rng)
    elif scenario == "dipping":
        model = create_dipping_layers(active_cc, rng)
        if rng.random() < 0.65:
            model = add_random_anomaly(model, active_cc, rng)
    elif scenario == "faulted":
        model = create_faulted_layers(active_cc, rng)
        if rng.random() < 0.70:
            model = add_random_anomaly(model, active_cc, rng)
    elif scenario == "thin_beds":
        model = create_thin_beds(active_cc, rng)
        if rng.random() < 0.55:
            model = add_random_anomaly(model, active_cc, rng)
    elif scenario == "gradient":
        model = create_background_gradient(active_cc, rng)
        for _ in range(int(rng.integers(1, 4))):
            model = add_random_anomaly(model, active_cc, rng)
    elif scenario == "hybrid":
        background_choice = rng.choice(
            [create_horizontal_layers, create_dipping_layers, create_background_gradient]
        )
        model = background_choice(active_cc, rng)
        for _ in range(int(rng.integers(2, 5))):
            model = add_random_anomaly(model, active_cc, rng)
    else:
        model = create_background_gradient(active_cc, rng)
        model = add_random_anomaly(model, active_cc, rng, near_borehole=True, weak=True)
        model = add_random_anomaly(model, active_cc, rng, near_borehole=False, strong=True)

    return np.clip(model, -0.6, 0.6).astype(np.float32), scenario


def apply_depth_misalignment(signal, depth_axis, spacing, rng):
    if rng.random() > 0.5:
        return signal.astype(np.float32)

    max_shift = min(0.45 * spacing, 1.0)
    ctrl_depth = np.linspace(depth_axis[0], depth_axis[-1], 5)
    ctrl_shift = rng.uniform(-max_shift, max_shift, size=5)
    shift_curve = np.interp(depth_axis[::-1], ctrl_depth[::-1], ctrl_shift[::-1])[::-1]
    warped_axis = depth_axis + shift_curve
    warped_signal = np.interp(
        depth_axis[::-1],
        warped_axis[::-1],
        signal[::-1],
        left=signal[-1],
        right=signal[0],
    )[::-1]
    return warped_signal.astype(np.float32)


def generate_complex_noise_v2(length, spacing, noise_std, rng):
    white_noise = rng.normal(0.0, noise_std, size=length).astype(np.float32)

    hetero_scale = np.ones(length, dtype=np.float32)
    n_segments = int(rng.integers(2, 6))
    segment_edges = np.linspace(0, length, n_segments + 1, dtype=int)
    for idx in range(n_segments):
        hetero_scale[segment_edges[idx] : segment_edges[idx + 1]] = rng.uniform(0.5, 1.8)
    heteroscedastic_noise = white_noise * hetero_scale

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

    colored_noise = np.zeros(length, dtype=np.float32)
    if rng.random() < 0.7:
        window_size = int(rng.integers(5, max(7, min(length // 6, 31))))
        raw = rng.normal(0.0, 1.0, size=length + window_size)
        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        filtered = np.convolve(raw, kernel, mode="valid")[:length]
        filtered_std = np.std(filtered)
        if filtered_std > 1e-6:
            colored_noise = (
                filtered / filtered_std * noise_std * rng.uniform(0.08, 0.25)
            ).astype(np.float32)

    step_drift = np.zeros(length, dtype=np.float32)
    if rng.random() < 0.6:
        n_steps = int(rng.integers(1, 3))
        for _ in range(n_steps):
            step_idx = int(rng.integers(max(3, length // 10), max(4, length - length // 10)))
            step_amp = noise_std * rng.uniform(-0.4, 0.4)
            step_drift[step_idx:] += step_amp

    spikes = np.zeros(length, dtype=np.float32)
    if rng.random() < 0.55:
        n_spikes = int(rng.integers(1, 4))
        for _ in range(n_spikes):
            spike_idx = int(rng.integers(0, length))
            spike_width = int(rng.integers(1, 4))
            spike_amp = noise_std * rng.uniform(1.5, 4.5) * rng.choice([-1.0, 1.0])
            end_idx = min(length, spike_idx + spike_width)
            spikes[spike_idx:end_idx] += spike_amp

    spacing_factor = np.clip(spacing / 1.0, 0.5, 5.0)
    total_noise = (
        heteroscedastic_noise
        + linear_drift
        + nonlinear_drift
        + colored_noise
        + step_drift
        + spikes * (0.8 + 0.1 * spacing_factor)
    )
    return total_noise.astype(np.float32)


def choose_acquisition_template(rng):
    return ACQUISITION_TEMPLATES[int(rng.integers(0, len(ACQUISITION_TEMPLATES)))]


def get_simulation_from_cache(cache, mesh, ind_active, template):
    key = (template["spacing"], template["n_points"], template["z_start"])
    if key not in cache:
        z_end = template["z_start"] - template["spacing"] * (template["n_points"] - 1)
        z_axis = np.linspace(template["z_start"], z_end, template["n_points"], dtype=np.float32)
        receiver_locations = np.c_[
            np.zeros(template["n_points"], dtype=np.float32),
            np.zeros(template["n_points"], dtype=np.float32),
            z_axis,
        ]
        cache[key] = {
            "simulation": build_simulation(mesh, ind_active, receiver_locations),
            "z_axis": z_axis,
        }
    return cache[key]


def init_worker():
    global WORKER_CONTEXT
    mesh, ind_active = build_mesh_and_active_cells()
    WORKER_CONTEXT = {
        "mesh": mesh,
        "ind_active": ind_active,
        "active_cc": mesh.gridCC[ind_active],
        "sim_cache": {},
    }


def generate_shard(task):
    global WORKER_CONTEXT

    if WORKER_CONTEXT is None:
        init_worker()

    mesh = WORKER_CONTEXT["mesh"]
    ind_active = WORKER_CONTEXT["ind_active"]
    active_cc = WORKER_CONTEXT["active_cc"]
    sim_cache = WORKER_CONTEXT["sim_cache"]

    shard_index = int(task["shard_index"])
    noise_std = float(task["noise_std"])
    target_count = int(task["target_count"])
    shard_path = Path(task["shard_path"])
    seed = int(task["seed"])
    rng = np.random.default_rng(seed)

    clean_signals = []
    noisy_signals = []
    sample_maxs = []
    noise_level_meta = []
    spacing_meta = []
    raw_length_meta = []
    challenging_mask = []
    clean_peak_meta = []
    scenario_meta = []
    depth_start_meta = []
    depth_end_meta = []

    generated = 0
    attempts = 0
    while generated < target_count:
        attempts += 1
        template = choose_acquisition_template(rng)
        cached = get_simulation_from_cache(sim_cache, mesh, ind_active, template)
        z_raw = cached["z_axis"]
        simulation = cached["simulation"]

        model_vector, scenario_name = create_random_model_v2(active_cc, rng)
        clean_raw = simulation.dpred(model_vector).astype(np.float32)
        clean_peak = float(np.max(np.abs(clean_raw)))
        is_challenging = clean_peak < SIGNAL_AMPLITUDE_THRESHOLD

        if is_challenging and rng.random() > CHALLENGING_KEEP_PROB:
            continue

        observed_signal = apply_depth_misalignment(
            clean_raw,
            z_raw,
            template["spacing"],
            rng,
        )
        noise_vector = generate_complex_noise_v2(
            len(z_raw),
            template["spacing"],
            noise_std,
            rng,
        )
        noisy_raw = observed_signal + noise_vector

        clean_resampled, z_target = resample_to_target(clean_raw, z_raw)
        noisy_resampled, _ = resample_to_target(noisy_raw, z_raw)

        sample_max = float(np.max(np.abs(noisy_resampled)))
        if sample_max < 1e-6:
            sample_max = 1.0

        clean_signals.append(clean_resampled / sample_max)
        noisy_signals.append(noisy_resampled / sample_max)
        sample_maxs.append(sample_max)
        noise_level_meta.append(noise_std)
        spacing_meta.append(template["spacing"])
        raw_length_meta.append(template["n_points"])
        challenging_mask.append(is_challenging)
        clean_peak_meta.append(clean_peak)
        scenario_meta.append(scenario_name)
        depth_start_meta.append(float(z_target[0]))
        depth_end_meta.append(float(z_target[-1]))

        generated += 1

    shard_payload = {
        "X_noisy": np.asarray(noisy_signals, dtype=np.float32),
        "Y_clean": np.asarray(clean_signals, dtype=np.float32),
        "sample_maxs": np.asarray(sample_maxs, dtype=np.float32),
        "noise_levels": np.asarray(noise_level_meta, dtype=np.float32),
        "sample_intervals": np.asarray(spacing_meta, dtype=np.float32),
        "raw_lengths": np.asarray(raw_length_meta, dtype=np.int32),
        "challenging_mask": np.asarray(challenging_mask, dtype=np.bool_),
        "clean_peaks_mgal": np.asarray(clean_peak_meta, dtype=np.float32),
        "depth_start_m": np.asarray(depth_start_meta, dtype=np.float32),
        "depth_end_m": np.asarray(depth_end_meta, dtype=np.float32),
        "scenario_labels": scenario_meta,
        "attempts": attempts,
        "generated": generated,
        "noise_std": noise_std,
        "shard_index": shard_index,
    }
    torch.save(shard_payload, shard_path)

    return {
        "shard_index": shard_index,
        "shard_path": str(shard_path),
        "generated": generated,
        "attempts": attempts,
        "noise_std": noise_std,
    }


def build_tasks(noise_levels, samples_per_level, shard_size, seed, shard_dir):
    tasks = []
    shard_index = 0
    for noise_level_index, noise_std in enumerate(noise_levels):
        remaining = int(samples_per_level)
        local_chunk = 0
        while remaining > 0:
            chunk_size = min(int(shard_size), remaining)
            shard_path = shard_dir / f"shard_{shard_index:05d}_noise_{noise_level_index}.pt"
            tasks.append(
                {
                    "shard_index": shard_index,
                    "noise_std": float(noise_std),
                    "target_count": chunk_size,
                    "seed": int(seed + 10000 * noise_level_index + local_chunk),
                    "shard_path": str(shard_path),
                }
            )
            remaining -= chunk_size
            shard_index += 1
            local_chunk += 1
    return tasks


def merge_shards(output_path, shard_paths, total_samples, noise_levels, samples_per_level):
    merged = {
        "X_noisy": [],
        "Y_clean": [],
        "sample_maxs": [],
        "noise_levels": [],
        "sample_intervals": [],
        "raw_lengths": [],
        "challenging_mask": [],
        "clean_peaks_mgal": [],
        "depth_start_m": [],
        "depth_end_m": [],
        "scenario_labels": [],
    }
    total_attempts = 0

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        merged["X_noisy"].append(shard["X_noisy"])
        merged["Y_clean"].append(shard["Y_clean"])
        merged["sample_maxs"].append(shard["sample_maxs"])
        merged["noise_levels"].append(shard["noise_levels"])
        merged["sample_intervals"].append(shard["sample_intervals"])
        merged["raw_lengths"].append(shard["raw_lengths"])
        merged["challenging_mask"].append(shard["challenging_mask"])
        merged["clean_peaks_mgal"].append(shard["clean_peaks_mgal"])
        merged["depth_start_m"].append(shard["depth_start_m"])
        merged["depth_end_m"].append(shard["depth_end_m"])
        merged["scenario_labels"].extend(shard["scenario_labels"])
        total_attempts += int(shard["attempts"])

    x_noisy = np.concatenate(merged["X_noisy"], axis=0)
    y_clean = np.concatenate(merged["Y_clean"], axis=0)
    sample_maxs = np.concatenate(merged["sample_maxs"], axis=0)
    noise_level_meta = np.concatenate(merged["noise_levels"], axis=0)
    spacing_meta = np.concatenate(merged["sample_intervals"], axis=0)
    raw_length_meta = np.concatenate(merged["raw_lengths"], axis=0)
    challenging_mask = np.concatenate(merged["challenging_mask"], axis=0)
    clean_peak_meta = np.concatenate(merged["clean_peaks_mgal"], axis=0)
    depth_start_meta = np.concatenate(merged["depth_start_m"], axis=0)
    depth_end_meta = np.concatenate(merged["depth_end_m"], axis=0)

    payload = {
        "X_noisy": torch.from_numpy(x_noisy.reshape(total_samples, TARGET_CHANNELS, TARGET_LENGTH)),
        "Y_clean": torch.from_numpy(y_clean.reshape(total_samples, TARGET_CHANNELS, TARGET_LENGTH)),
        "sample_maxs": torch.from_numpy(sample_maxs),
        "noise_levels": torch.from_numpy(noise_level_meta),
        "sample_intervals": torch.from_numpy(spacing_meta),
        "raw_lengths": torch.from_numpy(raw_length_meta),
        "challenging_mask": torch.from_numpy(challenging_mask),
        "clean_peaks_mgal": torch.from_numpy(clean_peak_meta),
        "depth_start_m": torch.from_numpy(depth_start_meta),
        "depth_end_m": torch.from_numpy(depth_end_meta),
        "scenario_labels": merged["scenario_labels"],
        "normalization": {
            "training_domain": "sample-wise normalization by noisy-signal peak",
            "evaluation_domain": "normalized domain plus denormalized physical-domain reporting",
            "physical_unit": "mGal",
            "target_length": TARGET_LENGTH,
        },
    }
    torch.save(payload, output_path)

    summary = {
        "output_file": str(output_path),
        "total_samples": total_samples,
        "samples_per_level": samples_per_level,
        "noise_levels_mgal": noise_levels,
        "target_length": TARGET_LENGTH,
        "challenging_fraction": float(np.mean(challenging_mask)),
        "sampling_intervals_m": sorted({float(v) for v in spacing_meta.tolist()}),
        "scenario_counts": {
            name: int(merged["scenario_labels"].count(name))
            for name in sorted(set(merged["scenario_labels"]))
        },
        "normalization_note": TARGET_LENGTH_DESCRIPTION,
        "total_generation_attempts": total_attempts,
    }
    return summary


def main():
    args = parse_args()
    noise_levels = [float(v) for v in args.noise_levels]
    samples_per_level = int(args.samples_per_level)
    total_samples = len(noise_levels) * samples_per_level
    num_workers = max(1, int(args.num_workers))
    shard_size = max(1, int(args.shard_size))
    output_path = Path(args.output)
    shard_dir = Path(args.temp_dir) if args.temp_dir else output_path.with_suffix("")
    shard_dir = Path(f"{shard_dir}_shards") if shard_dir.suffix == "" and not str(shard_dir).endswith("_shards") else shard_dir

    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Preparing to generate {total_samples} samples with {num_workers} worker(s)...")

    shard_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(
        noise_levels=noise_levels,
        samples_per_level=samples_per_level,
        shard_size=shard_size,
        seed=int(args.seed),
        shard_dir=shard_dir,
    )
    existing_task_results = []
    pending_tasks = []
    for task in tasks:
        shard_path = Path(task["shard_path"])
        if args.resume and shard_path.exists():
            existing_task_results.append(
                {
                    "shard_index": int(task["shard_index"]),
                    "shard_path": str(shard_path),
                    "generated": int(task["target_count"]),
                    "attempts": None,
                    "noise_std": float(task["noise_std"]),
                }
            )
        else:
            pending_tasks.append(task)

    print(f"Shard directory: {shard_dir}")
    print(f"Total shard tasks: {len(tasks)}")
    if args.resume:
        print(f"Existing shard tasks reused: {len(existing_task_results)}")
        print(f"Pending shard tasks: {len(pending_tasks)}")

    results = list(existing_task_results)
    if not args.merge_only and pending_tasks:
        if num_workers == 1:
            init_worker()
            for task in tqdm(pending_tasks, desc="Generating shards"):
                results.append(generate_shard(task))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers, initializer=init_worker) as pool:
                iterator = pool.imap_unordered(generate_shard, pending_tasks)
                for result in tqdm(iterator, total=len(pending_tasks), desc="Generating shards"):
                    results.append(result)
    elif args.merge_only:
        print("Merge-only mode: no new shards will be generated.")

    results.sort(key=lambda item: item["shard_index"])
    shard_paths = [item["shard_path"] for item in results]
    if len(shard_paths) != len(tasks):
        raise RuntimeError(
            f"Expected {len(tasks)} shard files, but found {len(shard_paths)}. "
            "Use --resume to continue generating missing shards before merging."
        )
    summary = merge_shards(
        output_path=output_path,
        shard_paths=shard_paths,
        total_samples=total_samples,
        noise_levels=noise_levels,
        samples_per_level=samples_per_level,
    )
    summary["num_workers"] = num_workers
    summary["shard_size"] = shard_size
    summary["num_shards"] = len(tasks)

    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)

    print(f"Dataset saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
