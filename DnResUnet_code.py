import argparse
import csv
import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError as exc:
    raise SystemExit("PyTorch is required to run DnResUnet_code.py.") from exc

try:
    from discretize import TensorMesh
    from discretize.base import BaseMesh
    from discretize.utils import active_from_xyz, mkvc
    from simpeg import maps
    from simpeg.potential_fields import gravity
    from simpeg.utils import model_builder
except ImportError:
    TensorMesh = None
    BaseMesh = None
    active_from_xyz = None
    mkvc = None
    maps = None
    gravity = None
    model_builder = None

try:
    from forward_v2 import (
        ACQUISITION_TEMPLATES,
        SIGNAL_AMPLITUDE_THRESHOLD,
        CHALLENGING_KEEP_PROB,
        apply_depth_misalignment,
        build_mesh_and_active_cells as build_mesh_and_active_cells_v2,
        create_random_model_v2,
        generate_complex_noise_v2,
        get_simulation_from_cache as get_simulation_from_cache_v2,
        resample_to_target as resample_to_target_v2,
    )
except ImportError:
    ACQUISITION_TEMPLATES = None
    SIGNAL_AMPLITUDE_THRESHOLD = 0.01
    CHALLENGING_KEEP_PROB = 0.35
    apply_depth_misalignment = None
    build_mesh_and_active_cells_v2 = None
    create_random_model_v2 = None
    generate_complex_noise_v2 = None
    get_simulation_from_cache_v2 = None
    resample_to_target_v2 = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mse(target, prediction):
    return float(torch.mean((target - prediction) ** 2).detach().cpu().item())


def compute_psnr(target, prediction, max_i=1.0):
    mse_value = compute_mse(target, prediction)
    if mse_value <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(max_i / math.sqrt(mse_value)))


def compute_gfc(target, prediction):
    target_flat = target.reshape(target.shape[0], -1)
    pred_flat = prediction.reshape(prediction.shape[0], -1)
    numerator = torch.sum(target_flat * pred_flat, dim=1)
    denominator = torch.sqrt(torch.sum(target_flat ** 2, dim=1)) * torch.sqrt(
        torch.sum(pred_flat ** 2, dim=1)
    )
    return float((numerator / (denominator + 1e-9)).mean().detach().cpu().item())


class TVLoss(nn.Module):
    def __init__(self, weight=1e-3):
        super().__init__()
        self.weight = weight

    def forward(self, signal):
        if signal.shape[-1] < 2:
            return signal.new_tensor(0.0)
        return self.weight * torch.mean((signal[..., 1:] - signal[..., :-1]) ** 2)


def make_norm(kind, channels):
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "instance":
        return nn.InstanceNorm1d(channels, affine=True)
    if kind == "group":
        groups = 8
        while groups > 1 and channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {kind}")


def make_activation(kind):
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "silu":
        return nn.SiLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {kind}")


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm, activation, dropout=0.0, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode="replicate",
                bias=False,
            ),
            make_norm(norm, out_channels),
            make_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm, activation, dropout):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size, norm, activation, dropout),
            ConvNormAct(out_channels, out_channels, kernel_size, norm, activation, dropout),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm, activation, dropout, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size, norm, activation, dropout, dilation)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode="replicate",
                bias=False,
            ),
            make_norm(norm, out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False), make_norm(norm, out_channels))
        )
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.activation(self.dropout(self.conv2(self.conv1(x))) + self.shortcut(x))


class DownBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.block = block

    def forward(self, x):
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, block_cls, kernel_size, norm, activation, dropout):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.block = block_cls(out_channels + skip_channels, out_channels, kernel_size, norm, activation, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=True)
        return self.block(torch.cat([x, skip], dim=1))


class BottleneckBlock(nn.Module):
    def __init__(self, channels, kernel_size, norm, activation, dropout):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(channels, channels, kernel_size, norm, activation, dropout, dilation=1),
            ResidualBlock(channels, channels, kernel_size, norm, activation, dropout, dilation=2),
            ResidualBlock(channels, channels, kernel_size, norm, activation, dropout, dilation=4),
        )

    def forward(self, x):
        return self.block(x)


class PlainBottleneckBlock(nn.Module):
    def __init__(self, channels, kernel_size, norm, activation, dropout):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size, norm, activation, dropout),
            ConvNormAct(channels, channels, kernel_size, norm, activation, dropout),
        )

    def forward(self, x):
        return self.block(x)


class LegacyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode="replicate"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LegacyResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode="replicate"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode="replicate"),
        )
        self.block2 = LegacyConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.block1(x) + self.block2(x)


class LegacyDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool1d(2), LegacyResBlock(in_channels, out_channels, kernel_size))

    def forward(self, x):
        return self.block(x)


class LegacyUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )
        self.conv = LegacyResBlock(out_channels * 2, out_channels, kernel_size)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class LegacyDnResUNet(nn.Module):
    def __init__(self, kernel_size=7, n_channels=1, n_out=1):
        super().__init__()
        self.predict_noise = True
        self.inputL = LegacyConvBlock(n_channels, 32, kernel_size)
        self.down1 = LegacyDown(32, 64, kernel_size)
        self.down2 = LegacyDown(64, 128, kernel_size)
        self.down3 = LegacyDown(128, 256, kernel_size)
        self.conv = LegacyConvBlock(256, 256, kernel_size)
        self.up3 = LegacyUp(256, 128, kernel_size)
        self.up4 = LegacyUp(128, 64, kernel_size)
        self.up5 = LegacyUp(64, 32, kernel_size)
        self.out = nn.Conv1d(32, n_out, kernel_size=1)

    def forward(self, x):
        x1 = self.inputL(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        b = self.conv(self.down3(x3))
        return self.out(self.up5(self.up4(self.up3(b, x3), x2), x1))


class ImprovedDnResUNet(nn.Module):
    def __init__(
        self,
        kernel_size=7,
        feature_sizes=(32, 64, 128, 256),
        norm="group",
        activation="relu",
        dropout=0.05,
        use_residual_blocks=True,
        predict_noise=True,
    ):
        super().__init__()
        block_cls = ResidualBlock if use_residual_blocks else PlainBlock
        self.predict_noise = predict_noise
        self.stem = block_cls(1, feature_sizes[0], kernel_size, norm, activation, dropout)
        self.downs = nn.ModuleList(
            [
                DownBlock(block_cls(in_c, out_c, kernel_size, norm, activation, dropout))
                for in_c, out_c in zip(feature_sizes[:-1], feature_sizes[1:])
            ]
        )
        self.bottleneck = BottleneckBlock(feature_sizes[-1], kernel_size, norm, activation, dropout)
        rev_features = list(reversed(feature_sizes))
        current_channels = rev_features[0]
        self.ups = nn.ModuleList()
        for skip_channels in rev_features[1:]:
            self.ups.append(
                UpBlock(current_channels, skip_channels, skip_channels, block_cls, kernel_size, norm, activation, dropout)
            )
            current_channels = skip_channels
        self.out = nn.Conv1d(feature_sizes[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(skips[-1])
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.out(x)


class BasicCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.predict_noise = False
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class DnCNN1D(nn.Module):
    def __init__(self, depth=17, channels=64):
        super().__init__()
        self.predict_noise = True
        layers = [nn.Conv1d(1, channels, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv1d(channels, 1, kernel_size=3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet1DBaseline(nn.Module):
    def __init__(self, kernel_size=5, feature_sizes=(32, 64, 128, 256), norm="group", activation="relu", dropout=0.05):
        super().__init__()
        self.predict_noise = False
        block_cls = PlainBlock
        self.stem = block_cls(1, feature_sizes[0], kernel_size, norm, activation, dropout)
        self.downs = nn.ModuleList(
            [
                DownBlock(block_cls(in_c, out_c, kernel_size, norm, activation, dropout))
                for in_c, out_c in zip(feature_sizes[:-1], feature_sizes[1:])
            ]
        )
        self.bottleneck = PlainBottleneckBlock(feature_sizes[-1], kernel_size, norm, activation, dropout)
        rev_features = list(reversed(feature_sizes))
        current_channels = rev_features[0]
        self.ups = nn.ModuleList()
        for skip_channels in rev_features[1:]:
            self.ups.append(
                UpBlock(current_channels, skip_channels, skip_channels, block_cls, kernel_size, norm, activation, dropout)
            )
            current_channels = skip_channels
        self.out = nn.Conv1d(feature_sizes[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(skips[-1])
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.out(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, norm, activation, dropout):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode="replicate",
                bias=False,
            ),
            make_norm(norm, channels),
            make_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode="replicate",
                bias=False,
            ),
            make_norm(norm, channels),
        )
        self.activation = make_activation(activation)

    def forward(self, x):
        return self.activation(self.conv2(self.conv1(x)) + x)


class TCN1D(nn.Module):
    def __init__(self, channels=64, kernel_size=5, dilations=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8), norm="group", activation="relu", dropout=0.05):
        super().__init__()
        self.predict_noise = True
        self.stem = ConvNormAct(1, channels, kernel_size, norm, activation, dropout)
        self.blocks = nn.Sequential(
            *[TCNResidualBlock(channels, kernel_size, dilation, norm, activation, dropout) for dilation in dilations]
        )
        self.head = nn.Sequential(
            ConvNormAct(channels, channels // 2, 3, norm, activation, dropout),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def predict_clean_signal(model, noisy):
    raw_output = model(noisy)
    if getattr(model, "predict_noise", False):
        return noisy - raw_output
    return raw_output


def load_pt_dataset(dataset_path):
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    return TensorDataset(data["X_noisy"].float(), data["Y_clean"].float())


@dataclass
class TrainingConfig:
    dataset: str
    output_dir: str
    experiment_name: str
    architecture: str = "improved"
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    tv_weight: float = 1e-3
    kernel_size: int = 7
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 4460
    num_workers: int = 0
    patience: int = 12
    use_residual_blocks: bool = True
    predict_noise: bool = True
    norm: str = "group"
    activation: str = "relu"
    dropout: float = 0.05
    device: str = "cuda"


def build_model(config):
    if config.architecture == "legacy":
        model = LegacyDnResUNet(kernel_size=config.kernel_size)
        model.predict_noise = config.predict_noise
        return model
    if config.architecture == "improved":
        return ImprovedDnResUNet(
            kernel_size=config.kernel_size,
            norm=config.norm,
            activation=config.activation,
            dropout=config.dropout,
            use_residual_blocks=config.use_residual_blocks,
            predict_noise=config.predict_noise,
        )
    if config.architecture == "basiccnn":
        model = BasicCNN1D()
        model.predict_noise = False
        return model
    if config.architecture == "dncnn":
        model = DnCNN1D()
        model.predict_noise = True
        return model
    if config.architecture == "unet1d":
        model = UNet1DBaseline(
            kernel_size=max(3, config.kernel_size),
            norm=config.norm,
            activation=config.activation,
            dropout=config.dropout,
        )
        model.predict_noise = False
        return model
    if config.architecture == "tcn":
        model = TCN1D(
            kernel_size=max(3, config.kernel_size),
            norm=config.norm,
            activation=config.activation,
            dropout=config.dropout,
        )
        model.predict_noise = True
        return model
    raise ValueError(f"Unsupported architecture: {config.architecture}")


def build_loss(tv_weight):
    mse_loss = nn.MSELoss()
    tv_loss = TVLoss(tv_weight)

    def hybrid_loss(prediction, target):
        return mse_loss(prediction, target) + tv_loss(prediction)

    return hybrid_loss


def evaluate_epoch(model, data_loader, loss_fn, device):
    model.eval()
    metrics = {"loss": 0.0, "mse": 0.0, "psnr": 0.0, "gfc": 0.0}
    batches = 0
    with torch.no_grad():
        for noisy, clean in data_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            prediction = predict_clean_signal(model, noisy)
            loss = loss_fn(prediction, clean)
            metrics["loss"] += float(loss.detach().cpu().item())
            metrics["mse"] += compute_mse(clean, prediction)
            metrics["psnr"] += compute_psnr(clean, prediction, max_i=1.0)
            metrics["gfc"] += compute_gfc(clean, prediction)
            batches += 1
    if batches == 0:
        return metrics
    return {key: value / batches for key, value in metrics.items()}


def run_training(config):
    set_seed(config.seed)
    dataset = load_pt_dataset(Path(config.dataset))
    total_samples = len(dataset)
    test_size = int(round(total_samples * config.test_ratio))
    valid_size = int(round(total_samples * config.valid_ratio))
    train_size = total_samples - test_size - valid_size
    split_generator = torch.Generator().manual_seed(config.seed)
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size], generator=split_generator)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=pin_memory, persistent_workers=config.num_workers > 0)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=pin_memory, persistent_workers=config.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = build_loss(config.tv_weight)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{config.experiment_name}_history.csv"
    checkpoint_path = output_dir / f"{config.experiment_name}_checkpoint.pt"
    history_rows = []
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = predict_clean_signal(model, noisy)
                loss = loss_fn(prediction, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu().item())
            batches += 1

        valid_metrics = evaluate_epoch(model, valid_loader, loss_fn, device)
        scheduler.step(valid_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "valid_loss": valid_metrics["loss"],
            "valid_mse": valid_metrics["mse"],
            "valid_psnr": valid_metrics["psnr"],
            "valid_gfc": valid_metrics["gfc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.6f} | "
            f"valid_loss={row['valid_loss']:.6f} | valid_psnr={row['valid_psnr']:.3f} | "
            f"valid_gfc={row['valid_gfc']:.4f}"
        )

        if valid_metrics["loss"] < best_val:
            best_val = valid_metrics["loss"]
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "config": vars(config), "best_valid_loss": best_val}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_epoch(model, test_loader, loss_fn, device)
    summary = {
        "experiment_name": config.experiment_name,
        "architecture": config.architecture,
        "kernel_size": config.kernel_size,
        "use_residual_blocks": config.use_residual_blocks,
        "predict_noise": config.predict_noise,
        "tv_weight": config.tv_weight,
        "test_loss": test_metrics["loss"],
        "test_mse": test_metrics["mse"],
        "test_psnr": test_metrics["psnr"],
        "test_gfc": test_metrics["gfc"],
    }
    (output_dir / f"{config.experiment_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def build_ablation_suite(base_config):
    configs = []
    for kernel_size in (3, 5, 7):
        cfg = deepcopy(base_config)
        cfg.experiment_name = f"{base_config.experiment_name}_kernel_{kernel_size}"
        cfg.kernel_size = kernel_size
        configs.append(cfg)
    cfg = deepcopy(base_config)
    cfg.experiment_name = f"{base_config.experiment_name}_wo_tv"
    cfg.tv_weight = 0.0
    configs.append(cfg)
    cfg = deepcopy(base_config)
    cfg.experiment_name = f"{base_config.experiment_name}_wo_resblock"
    cfg.use_residual_blocks = False
    configs.append(cfg)
    cfg = deepcopy(base_config)
    cfg.experiment_name = f"{base_config.experiment_name}_direct_clean"
    cfg.predict_noise = False
    configs.append(cfg)
    return configs


def ensure_simpeg_available():
    if TensorMesh is None or gravity is None or model_builder is None:
        raise RuntimeError("SimPEG/discretize dependencies are required for synthetic resampling.")


def get_indices_ellipsoid(center, radii, cell_centers):
    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.gridCC
    norm_dist_sq = (((cell_centers[:, 0] - center[0]) / radii[0]) ** 2 + ((cell_centers[:, 1] - center[1]) / radii[1]) ** 2 + ((cell_centers[:, 2] - center[2]) / radii[2]) ** 2)
    return norm_dist_sq < 1.0


class IndependentGravityGenerator:
    def __init__(self, length=512, spacing_m=1.0, use_v2=True, seed=45):
        ensure_simpeg_available()
        self.rng = np.random.default_rng(seed)
        self.length = length
        self.spacing_m = spacing_m
        self.use_v2 = use_v2 and build_mesh_and_active_cells_v2 is not None
        if self.use_v2:
            self.mesh, self.ind_active = build_mesh_and_active_cells_v2()
            self.active_cc = self.mesh.gridCC[self.ind_active]
            self.sim_cache = {}
            self.nC_active = int(self.ind_active.sum())
            return
        self.mesh = TensorMesh([[(5.0, 5, -1.3), (5.0, 40), (5.0, 5, 1.3)], [(5.0, 5, -1.3), (5.0, 40), (5.0, 5, 1.3)], [(5.0, 5, -1.3), (5.0, 15)]], "CCN")
        x_topo, y_topo = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-200, 200, 41))
        z_topo = -15.0 * np.exp(-(x_topo ** 2 + y_topo ** 2) / 80.0 ** 2)
        topo_xyz = np.c_[mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)]
        self.ind_active = active_from_xyz(self.mesh, topo_xyz)
        self.nC_active = int(self.ind_active.sum())
        z_stations = np.linspace(-20.0, -20.0 - (length - 1) * spacing_m, length)
        receiver_locations = np.c_[np.zeros(length), np.zeros(length), z_stations]
        receiver = gravity.receivers.Point(receiver_locations, components=["gz"])
        source_field = gravity.sources.SourceField(receiver_list=[receiver])
        survey = gravity.survey.Survey(source_field)
        self.simulation = gravity.simulation.Simulation3DIntegral(survey=survey, mesh=self.mesh, rhoMap=maps.IdentityMap(nP=self.nC_active), active_cells=self.ind_active, store_sensitivities="forward_only", engine="choclo")

    def create_random_model(self):
        if np.random.randint(0, 2) == 1:
            num_layers = np.random.randint(2, 6)
            layer_values = np.random.uniform(-0.5, 0.5, size=num_layers)
            for index in range(num_layers):
                while abs(layer_values[index]) < 0.05:
                    layer_values[index] = np.random.uniform(-0.5, 0.5)
            tops = np.sort(np.random.uniform(-450, -30, size=num_layers - 1))
            layer_tops = np.concatenate(([np.inf], tops[::-1]))
            return model_builder.create_layers_model(self.mesh.gridCC, layer_tops, layer_values)[self.ind_active].astype(np.float32)
        model = np.zeros(self.nC_active, dtype=np.float32)
        for _ in range(np.random.randint(1, 4)):
            anomaly_type = np.random.randint(0, 4)
            density = np.random.uniform(-0.5, 0.5)
            while abs(density) < 0.05:
                density = np.random.uniform(-0.5, 0.5)
            center = [np.random.uniform(-50, 50), np.random.uniform(-50, 50), np.random.uniform(-500, -30)]
            if anomaly_type == 0:
                wx, wy, hz = np.random.uniform(10, 50), np.random.uniform(10, 50), np.random.uniform(10, 40)
                x1, x2 = center[0] - wx / 2, center[0] + wx / 2
                y1, y2 = center[1] - wy / 2, center[1] + wy / 2
                z1, z2 = center[2] - hz / 2, center[2] + hz / 2
                mask = ((self.mesh.gridCC[self.ind_active, 0] > x1) & (self.mesh.gridCC[self.ind_active, 0] < x2) & (self.mesh.gridCC[self.ind_active, 1] > y1) & (self.mesh.gridCC[self.ind_active, 1] < y2) & (self.mesh.gridCC[self.ind_active, 2] > z1) & (self.mesh.gridCC[self.ind_active, 2] < z2))
            elif anomaly_type == 1:
                mask = model_builder.get_indices_sphere(center, np.random.uniform(10, 40), self.mesh.gridCC)[self.ind_active]
            elif anomaly_type == 2:
                mask = get_indices_ellipsoid(center, [np.random.uniform(10, 50), np.random.uniform(10, 50), np.random.uniform(10, 40)], self.mesh.gridCC)[self.ind_active]
            else:
                pts = np.zeros((np.random.randint(5, 11), 3))
                for axis in range(3):
                    pts[:, axis] = center[axis] + np.random.uniform(-30, 30, size=len(pts))
                try:
                    mask = model_builder.get_indices_polygon(self.mesh, pts)[self.ind_active]
                except Exception:
                    mask = np.zeros(self.nC_active, dtype=bool)
            if np.any(mask):
                model[mask] = density
        return model.astype(np.float32)

    def generate_complex_noise(self, length, noise_std):
        white_noise = (np.random.randn(length) * noise_std).astype(np.float32)
        drift_mag = noise_std * np.random.uniform(0.05, 0.2)
        linear_drift = np.linspace(np.random.uniform(-drift_mag, drift_mag), np.random.uniform(-drift_mag, drift_mag), length).astype(np.float32)
        colored_noise = np.zeros(length, dtype=np.float32)
        if np.random.rand() > 0.5:
            window_size = np.random.randint(5, 15)
            filtered = np.convolve(np.random.randn(length + window_size), np.ones(window_size) / window_size, "valid")[:length]
            colored_noise = (filtered / (np.std(filtered) + 1e-9) * noise_std * np.random.uniform(0.1, 0.3)).astype(np.float32)
        return white_noise + linear_drift + colored_noise

    def generate_sample(self, noise_std, return_metadata=False):
        if self.use_v2:
            while True:
                template = ACQUISITION_TEMPLATES[int(self.rng.integers(0, len(ACQUISITION_TEMPLATES)))]
                cached = get_simulation_from_cache_v2(self.sim_cache, self.mesh, self.ind_active, template)
                z_raw = cached["z_axis"]
                simulation = cached["simulation"]
                model, scenario_name = create_random_model_v2(self.active_cc, self.rng)
                clean_raw = simulation.dpred(model).astype(np.float32)
                clean_peak = float(np.max(np.abs(clean_raw)))
                is_challenging = clean_peak < SIGNAL_AMPLITUDE_THRESHOLD
                if is_challenging and self.rng.random() > CHALLENGING_KEEP_PROB:
                    continue
                observed_signal = apply_depth_misalignment(clean_raw, z_raw, template["spacing"], self.rng)
                noisy_raw = observed_signal + generate_complex_noise_v2(len(z_raw), template["spacing"], noise_std, self.rng)
                clean_norm, _ = resample_to_target_v2(clean_raw, z_raw, target_length=self.length)
                noisy_norm, z_target = resample_to_target_v2(noisy_raw, z_raw, target_length=self.length)
                scale = float(np.max(np.abs(noisy_norm)))
                if scale < 1e-6:
                    scale = 1.0
                result = (
                    noisy_norm.astype(np.float32) / scale,
                    clean_norm.astype(np.float32) / scale,
                    scale,
                )
                if return_metadata:
                    metadata = {
                        "depth_axis_m": z_target.astype(np.float32),
                        "sample_interval_m": float(template["spacing"]),
                        "raw_length": int(template["n_points"]),
                        "scenario": scenario_name,
                        "challenging": bool(is_challenging),
                    }
                    return result + (metadata,)
                return result
        while True:
            model = self.create_random_model()
            clean = self.simulation.dpred(model)
            if np.max(np.abs(clean)) >= 0.01:
                break
        noisy = clean + self.generate_complex_noise(self.length, noise_std)
        scale = float(np.max(np.abs(noisy)))
        if scale < 1e-6:
            scale = 1.0
        result = (noisy.astype(np.float32) / scale, clean.astype(np.float32) / scale, scale)
        if return_metadata:
            depth_axis_m = np.linspace(-20.0, -20.0 - (self.length - 1) * self.spacing_m, self.length).astype(np.float32)
            metadata = {
                "depth_axis_m": depth_axis_m,
                "sample_interval_m": float(self.spacing_m),
                "raw_length": int(self.length),
                "scenario": "legacy_generator",
                "challenging": False,
            }
            return result + (metadata,)
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Improved DnResUnet training and ablation script.")
    parser.add_argument("--dataset", default="gravity_dataset_100k_V2_REALISTIC.pt")
    parser.add_argument("--output-dir", default="Models_v3")
    parser.add_argument("--experiment-name", default="dnresunet_improved")
    parser.add_argument("--architecture", default="improved", choices=["legacy", "improved", "basiccnn", "dncnn"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--kernel-size", type=int, default=7, choices=[3, 5, 7])
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=4460)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--norm", default="group", choices=["group", "batch", "instance", "none"])
    parser.add_argument("--activation", default="relu", choices=["relu", "silu", "gelu"])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--direct-clean", action="store_true")
    parser.add_argument("--disable-residual-blocks", action="store_true")
    parser.add_argument("--run-ablation-suite", action="store_true")
    return parser.parse_args()


def config_from_args(args):
    return TrainingConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        architecture=args.architecture,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        tv_weight=args.tv_weight,
        kernel_size=args.kernel_size,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        use_residual_blocks=not args.disable_residual_blocks,
        predict_noise=not args.direct_clean,
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout,
        device=args.device,
    )


def main():
    args = parse_args()
    config = config_from_args(args)
    if args.run_ablation_suite:
        results = [run_training(cfg) for cfg in build_ablation_suite(config)]
        (Path(config.output_dir) / f"{config.experiment_name}_ablation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        run_training(config)


if __name__ == "__main__":
    main()
