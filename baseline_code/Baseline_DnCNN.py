import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DnResUnet_code import TrainingConfig, run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DnCNN baseline on the realistic v2 dataset.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "gravity_dataset_100k_V2_REALISTIC.pt"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "Models_Benchmark_v2"))
    parser.add_argument("--experiment-name", default="DnCNN_v2_realistic")
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
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        architecture="dncnn",
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        tv_weight=0.0,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        predict_noise=True,
        device=args.device,
    )
    run_training(config)


if __name__ == "__main__":
    main()
