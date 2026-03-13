import argparse
import json
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DnResUnet_code import TrainingConfig, run_training


BASELINE_PRESETS = {
    "basiccnn": {
        "experiment_name": "BasicCNN_v2_realistic",
        "architecture": "basiccnn",
        "predict_noise": False,
        "kernel_size": 3,
        "tv_weight": 0.0,
        "norm": "batch",
        "activation": "relu",
        "dropout": 0.0,
    },
    "dncnn": {
        "experiment_name": "DnCNN_v2_realistic",
        "architecture": "dncnn",
        "predict_noise": True,
        "kernel_size": 3,
        "tv_weight": 0.0,
        "norm": "batch",
        "activation": "relu",
        "dropout": 0.0,
    },
    "unet1d": {
        "experiment_name": "UNet1D_v2_realistic",
        "architecture": "unet1d",
        "predict_noise": False,
        "kernel_size": 5,
        "tv_weight": 0.0,
        "norm": "group",
        "activation": "relu",
        "dropout": 0.05,
    },
    "tcn": {
        "experiment_name": "TCN_v2_realistic",
        "architecture": "tcn",
        "predict_noise": True,
        "kernel_size": 5,
        "tv_weight": 0.0,
        "norm": "group",
        "activation": "relu",
        "dropout": 0.05,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train all deep learning baselines on the realistic v2 dataset.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "gravity_dataset_100k_V2_REALISTIC.pt"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "Models_Benchmark_v2"))
    parser.add_argument("--models", nargs="+", default=["basiccnn", "dncnn", "unet1d", "tcn"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=4460)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def build_config(args, model_key):
    preset = BASELINE_PRESETS[model_key]
    return TrainingConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        experiment_name=preset["experiment_name"],
        architecture=preset["architecture"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        tv_weight=preset["tv_weight"],
        kernel_size=preset["kernel_size"],
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        predict_noise=preset["predict_noise"],
        norm=preset["norm"],
        activation=preset["activation"],
        dropout=preset["dropout"],
        device=args.device,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_models = []
    for model_key in args.models:
        key = model_key.lower()
        if key not in BASELINE_PRESETS:
            raise SystemExit(f"Unsupported baseline '{model_key}'. Choose from: {', '.join(BASELINE_PRESETS)}")
        selected_models.append(key)

    run_report = {
        "dataset": args.dataset,
        "output_dir": str(output_dir),
        "device": args.device,
        "models": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for model_key in selected_models:
        config = build_config(args, model_key)
        checkpoint_path = output_dir / f"{config.experiment_name}_checkpoint.pt"
        history_path = output_dir / f"{config.experiment_name}_history.csv"
        summary_path = output_dir / f"{config.experiment_name}_summary.json"

        if args.skip_existing and checkpoint_path.exists() and history_path.exists() and summary_path.exists():
            print(f"[SKIP] {config.experiment_name} already exists.")
            run_report["models"].append(
                {
                    "model": model_key,
                    "experiment_name": config.experiment_name,
                    "status": "skipped_existing",
                    "checkpoint": str(checkpoint_path),
                }
            )
            continue

        print("=" * 90)
        print(f"Starting baseline: {config.experiment_name}")
        print("=" * 90)
        start_time = time.time()

        try:
            run_training(config)
            elapsed = time.time() - start_time
            run_report["models"].append(
                {
                    "model": model_key,
                    "experiment_name": config.experiment_name,
                    "status": "completed",
                    "elapsed_seconds": round(elapsed, 2),
                    "checkpoint": str(checkpoint_path),
                    "history_csv": str(history_path),
                    "summary_json": str(summary_path),
                }
            )
            print(f"[DONE] {config.experiment_name} finished in {elapsed / 60.0:.2f} minutes.")
        except Exception as exc:  # pragma: no cover - defensive for workstation use
            elapsed = time.time() - start_time
            print(f"[FAILED] {config.experiment_name} failed after {elapsed / 60.0:.2f} minutes.")
            traceback.print_exc()
            run_report["models"].append(
                {
                    "model": model_key,
                    "experiment_name": config.experiment_name,
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 2),
                    "error": repr(exc),
                }
            )
            if args.stop_on_error:
                break

    run_report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report_path = output_dir / "baseline_batch_run_report.json"
    report_path.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
    print("=" * 90)
    print(f"Batch baseline run summary saved to: {report_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
