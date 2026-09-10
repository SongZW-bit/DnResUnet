from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pdfplumber
import torch
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter, wiener
from skimage.restoration import denoise_wavelet


WELLS = {
    11: {
        "well": "Dixie Valley 24-37-10DDCD",
        "short": "Dixie Valley",
        "expected_rows": 30,
        "bands": ((70, 110), (110, 165), (165, 205), (350, 395), (440, 505)),
    },
    12: {
        "well": "Wash O'Neil 2AA",
        "short": "Wash O'Neil",
        "expected_rows": 59,
        "bands": ((175, 205), (205, 245), (245, 280), (385, 420), (450, 500)),
    },
    13: {
        "well": "Winn Farms 23BDA",
        "short": "Winn Farms",
        "expected_rows": 30,
        "bands": ((70, 110), (110, 165), (165, 205), (350, 395), (440, 505)),
    },
}


LITHOLOGY_BOUNDARIES_M = {
    "Dixie Valley 24-37-10DDCD": [2.7, 15.8, 23.2, 26.8, 29.3, 31.1, 42.7, 47.6, 54.9, 58.5, 67.1],
    "Wash O'Neil 2AA": [5.2, 23.5, 29.6, 32.6, 38.7, 69.2, 72.2, 78.3, 84.4, 87.5, 102.7],
    "Winn Farms 23BDA": [3.0, 3.9, 35.1, 46.6, 57.3],
}


def _band_text(words, lo: float, hi: float) -> str:
    selected = sorted((word for word in words if lo <= word["x0"] < hi), key=lambda word: word["x0"])
    return "".join(word["text"] for word in selected)


def _parse_number(text: str) -> float | None:
    cleaned = (
        text.replace("J", "3")
        .replace("O", "0")
        .replace("Q", "0")
        .replace("^", "")
        .replace(":", ".")
        .replace(",", "")
        .replace('"', "")
        .replace(" ", "")
    )
    cleaned = re.sub(r"[^0-9+.-]", "", cleaned).replace("..", ".")
    if cleaned in {"", "+", "-", "."}:
        return None
    if cleaned.startswith("+."):
        cleaned = "+0" + cleaned[1:]
    elif cleaned.startswith("-."):
        cleaned = "-0" + cleaned[1:]
    elif cleaned.startswith("."):
        cleaned = "0" + cleaned
    try:
        return float(cleaned)
    except ValueError:
        return None


def _cluster_lines(words: list[dict], tolerance: float = 2.0) -> list[tuple[float, list[dict]]]:
    lines: list[list] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if lines and abs(lines[-1][0] - word["top"]) <= tolerance:
            lines[-1][1].append(word)
            lines[-1][0] = float(np.mean([item["top"] for item in lines[-1][1]]))
        else:
            lines.append([word["top"], [word]])
    return [(float(top), line_words) for top, line_words in lines]


def extract_principal_facts(pdf_path: Path) -> list[dict]:
    rows: list[dict] = []
    with pdfplumber.open(pdf_path) as document:
        for page_index, config in WELLS.items():
            words = document.pages[page_index].extract_words(x_tolerance=1, y_tolerance=1.5)
            lines = _cluster_lines(words)
            pending: tuple[int, float, list[dict]] | None = None
            page_rows: list[dict] = []
            for top, line_words in lines:
                values = [_band_text(line_words, *band) for band in config["bands"]]
                reading_text = values[0].strip()
                if reading_text == "U":
                    reading_text = "4"
                if reading_text.isdigit():
                    if pending is not None:
                        reading, pending_top, pending_words = pending
                        page_rows.append(_row_from_words(config, reading, pending_words))
                    pending = (int(reading_text), top, list(line_words))
                elif pending is not None and top - pending[1] <= 8.0:
                    pending[2].extend(line_words)
            if pending is not None:
                reading, _, pending_words = pending
                page_rows.append(_row_from_words(config, reading, pending_words))

            page_rows = [row for row in page_rows if row["depth_m"] is not None and row["corrected_gravity_mgal"]]
            page_rows.sort(key=lambda row: row["reading"])
            if len(page_rows) != config["expected_rows"]:
                raise RuntimeError(
                    f"Extracted {len(page_rows)} rows for {config['well']}; expected {config['expected_rows']}."
                )
            rows.extend(page_rows)
    return rows


def _row_from_words(config: dict, reading: int, words: list[dict]) -> dict:
    bands = config["bands"]
    values = [_band_text(words, *band) for band in bands]
    return {
        "well": config["well"],
        "well_short": config["short"],
        "reading": reading,
        "depth_ft": _parse_number(values[1]),
        "depth_m": _parse_number(values[2]),
        "drift_correction_mgal": _parse_number(values[3]),
        "corrected_gravity_mgal": _parse_number(values[4]),
    }


def load_model(code_dir: Path, checkpoint_path: Path):
    sys.path.insert(0, str(code_dir))
    from independent_resampling_eval import load_checkpoint_model  # noqa: PLC0415

    device = torch.device("cpu")
    return load_checkpoint_model(checkpoint_path, device), device


def predict_dnresunet(model, device, signal: np.ndarray) -> np.ndarray:
    from DnResUnet_code import predict_clean_signal  # noqa: PLC0415

    tensor = torch.from_numpy(signal.astype(np.float32)).view(1, 1, -1).to(device)
    with torch.no_grad():
        return predict_clean_signal(model, tensor).detach().cpu().numpy().reshape(-1)


def classical_predictions(signal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Savitzky-Golay": savgol_filter(signal, window_length=31, polyorder=5, mode="mirror"),
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


def gfc(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(estimate))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(reference, estimate) / denominator)


def interval_density(depth_ft: np.ndarray, gravity: np.ndarray, free_air_gradient: float = 0.09406):
    order = np.argsort(depth_ft)
    depth_ft = depth_ft[order]
    gravity = gravity[order]
    delta_z = np.diff(depth_ft)
    delta_g = np.diff(gravity)
    density = (free_air_gradient - delta_g / delta_z) / 0.02556
    mid_depth_m = 0.3048 * (depth_ft[:-1] + depth_ft[1:]) / 2.0
    return mid_depth_m, density


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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    facts: list[dict],
    model,
    device,
    repeats: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    rng = np.random.default_rng(seed)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    representative: dict[str, dict] = {}

    by_well: dict[str, list[dict]] = defaultdict(list)
    for row in facts:
        by_well[row["well"]].append(row)

    for well, well_rows in by_well.items():
        by_depth: dict[float, list[dict]] = defaultdict(list)
        for row in well_rows:
            by_depth[float(row["depth_ft"])].append(row)
        depths_ft = np.asarray(sorted(by_depth), dtype=float)
        depths_m = depths_ft * 0.3048
        corrected = np.asarray(
            [np.median([item["corrected_gravity_mgal"] for item in by_depth[depth]]) for depth in depths_ft]
        )
        grid_m = np.linspace(depths_m.min(), depths_m.max(), 512)
        grid_ft = grid_m / 0.3048
        reference_grid = np.interp(grid_ft, depths_ft, corrected)

        for repeat_index in range(repeats):
            selected = [group[int(rng.integers(0, len(group)))] for group in (by_depth[depth] for depth in depths_ft)]
            pre_drift = np.asarray(
                [item["corrected_gravity_mgal"] - item["drift_correction_mgal"] for item in selected], dtype=float
            )
            input_grid = np.interp(grid_ft, depths_ft, pre_drift)
            trend_coefficients = np.polyfit(grid_m, input_grid, deg=1)
            trend_grid = np.polyval(trend_coefficients, grid_m)
            input_residual = input_grid - trend_grid
            scale = float(np.max(np.abs(input_residual)))
            if scale < 1e-8:
                scale = 1.0
            normalized = input_residual / scale

            methods = {"Uncorrected input": input_grid}
            methods.update(
                {
                    name: prediction * scale + trend_grid
                    for name, prediction in classical_predictions(normalized).items()
                }
            )
            methods["DnResUnet"] = predict_dnresunet(model, device, normalized) * scale + trend_grid

            _, reference_density = interval_density(depths_ft, corrected)
            for method, estimate_grid in methods.items():
                estimate_at_stations = np.interp(depths_m, grid_m, estimate_grid)
                _, estimate_density = interval_density(depths_ft, estimate_at_stations)
                error = estimate_at_stations - corrected
                density_error = estimate_density - reference_density
                metric_rows.append(
                    {
                        "well": well,
                        "repeat": repeat_index,
                        "method": method,
                        "gravity_mse_mgal2": float(np.mean(error**2)),
                        "gravity_mae_mgal": float(np.mean(np.abs(error))),
                        "gravity_gfc": gfc(corrected - corrected.mean(), estimate_at_stations - estimate_at_stations.mean()),
                        "density_rmse_g_cm3": float(np.sqrt(np.mean(density_error**2))),
                        "density_mae_g_cm3": float(np.mean(np.abs(density_error))),
                        "density_correlation": float(np.corrcoef(reference_density, estimate_density)[0, 1]),
                    }
                )

            if repeat_index == 0:
                representative[well] = {
                    "depths_m": depths_m,
                    "depths_ft": depths_ft,
                    "grid_m": grid_m,
                    "corrected": corrected,
                    "reference_grid": reference_grid,
                    "methods": methods,
                }
                for index, depth_m in enumerate(grid_m):
                    prediction_rows.append(
                        {
                            "well": well,
                            "depth_m": float(depth_m),
                            "reference_corrected_mgal": float(reference_grid[index]),
                            **{f"{name}_mgal": float(values[index]) for name, values in methods.items()},
                        }
                    )
    return metric_rows, prediction_rows, representative


def aggregate_metrics(metric_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in metric_rows:
        groups[(row["well"], row["method"])].append(row)
    aggregated: list[dict] = []
    metric_names = [
        "gravity_mse_mgal2",
        "gravity_mae_mgal",
        "gravity_gfc",
        "density_rmse_g_cm3",
        "density_mae_g_cm3",
        "density_correlation",
    ]
    for (well, method), rows in sorted(groups.items()):
        result = {"well": well, "method": method}
        for metric_name in metric_names:
            stats = summarize([float(row[metric_name]) for row in rows])
            result.update({f"{metric_name}_{key}": value for key, value in stats.items()})
        aggregated.append(result)
    for method in sorted({row["method"] for row in metric_rows}):
        rows = [row for row in metric_rows if row["method"] == method]
        result = {"well": "All wells (pooled)", "method": method}
        for metric_name in metric_names:
            stats = summarize([float(row[metric_name]) for row in rows])
            result.update({f"{metric_name}_{key}": value for key, value in stats.items()})
        aggregated.append(result)
    return aggregated


def plot_validation(representative: dict[str, dict], metric_rows: list[dict], output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    wells = list(representative)
    colors = {
        "Uncorrected input": "#F8766D",
        "DnResUnet": "#00A6A6",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.35))
    fig.subplots_adjust(left=0.075, right=0.992, top=0.90, bottom=0.20, wspace=0.28)
    for column, well in enumerate(wells):
        ax = axes[column]
        record = representative[well]
        reference = record["reference_grid"]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#202020")
        ax.tick_params(top=False, right=False)
        ax.grid(True, color="#D8D8D8", linestyle=(0, (2.5, 2.5)), lw=0.5, alpha=0.72)
        ax.set_axisbelow(True)
        for boundary in LITHOLOGY_BOUNDARIES_M.get(well, []):
            if record["grid_m"].min() <= boundary <= record["grid_m"].max():
                ax.axhline(
                    boundary,
                    color="#A9824A",
                    lw=0.65,
                    linestyle=(0, (3.0, 2.2)),
                    alpha=0.48,
                    zorder=1,
                )
        ax.plot(
            1000.0 * (record["methods"]["Uncorrected input"] - reference),
            record["grid_m"],
            color=colors["Uncorrected input"],
            lw=0.95,
            alpha=0.82,
            label="Pre-drift input",
            zorder=3,
        )
        ax.axvline(
            0.0,
            color="black",
            lw=0.95,
            linestyle="--",
            label="Published corrected reference",
            zorder=2,
        )
        ax.plot(
            1000.0 * (record["methods"]["DnResUnet"] - reference),
            record["grid_m"],
            color=colors["DnResUnet"],
            lw=1.35,
            label="DnResUnet output",
            zorder=4,
        )
        ax.invert_yaxis()
        ax.set_title(well, pad=7, fontsize=8.5)
        if column == 0:
            ax.set_ylabel("Depth (m)")
        ax.text(
            0.02,
            0.98,
            chr(ord("a") + column),
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            fontsize=8,
        )

    fig.supxlabel(r"Difference from published corrected reference ($\mu$Gal)", y=0.105)
    legend_handles = [
        Line2D([0], [0], color=colors["Uncorrected input"], lw=0.95, alpha=0.82),
        Line2D([0], [0], color="black", lw=0.95, linestyle="--"),
        Line2D([0], [0], color=colors["DnResUnet"], lw=1.35),
        Line2D([0], [0], color="#A9824A", lw=0.65, linestyle=(0, (3.0, 2.2)), alpha=0.65),
    ]
    legend_labels = [
        "Pre-drift input",
        "Published corrected reference",
        "DnResUnet output",
        "Reported lithologic boundary",
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        fontsize=6.0,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.25,
    )

    prefix = output_dir / "field_validation"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    facts = extract_principal_facts(args.pdf)
    write_csv(args.output_dir / "usgs_principal_facts_extracted.csv", facts)
    model, device = load_model(args.code_dir, args.checkpoint)
    metric_rows, prediction_rows, representative = evaluate(facts, model, device, args.repeats, args.seed)
    aggregate_rows = aggregate_metrics(metric_rows)
    write_csv(args.output_dir / "field_validation_raw_metrics.csv", metric_rows)
    write_csv(args.output_dir / "field_validation_summary.csv", aggregate_rows)
    write_csv(args.output_dir / "field_validation_representative_profiles.csv", prediction_rows)
    plot_validation(representative, metric_rows, args.output_dir)

    report = {
        "source": "USGS Open-File Report 85-426",
        "source_url": "https://pubs.usgs.gov/of/1985/0426/report.pdf",
        "source_doi": "https://doi.org/10.3133/ofr85426",
        "interpretation_boundary": (
            "The published drift-corrected gravity is a conventional field reference, not a noise-free label."
        ),
        "wells": sorted({row["well"] for row in facts}),
        "station_readings": len(facts),
        "repeat_resamples_per_well": args.repeats,
        "summary": aggregate_rows,
    }
    (args.output_dir / "field_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"station_readings": len(facts), "summary_rows": len(aggregate_rows)}, indent=2))


if __name__ == "__main__":
    main()
