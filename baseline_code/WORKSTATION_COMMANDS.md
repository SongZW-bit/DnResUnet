# Baseline v2 Training Commands

The traditional baselines (`S-G Filter`, `Wiener`, and `Wavelet`) do not require model training. They are already evaluated directly inside `independent_resampling_eval.py` on the realistic v2 independent test set.

The deep learning baselines that should be trained on the realistic v2 dataset are:

- `BasicCNN_v2_realistic`
- `DnCNN_v2_realistic`
- `UNet1D_v2_realistic`
- `TCN_v2_realistic`

## One-command option

If you prefer to launch all four deep learning baselines in one run, use:

```bash
cd /your/project/path
conda activate pytorch
CUDA_VISIBLE_DEVICES=7 python baseline_code/Run_All_Baselines.py \
  --dataset gravity_dataset_100k_V2_REALISTIC.pt \
  --output-dir Models_Benchmark_v2 \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --skip-existing
```

This script will train the following models in sequence:

- `BasicCNN_v2_realistic`
- `DnCNN_v2_realistic`
- `UNet1D_v2_realistic`
- `TCN_v2_realistic`

It also writes a batch summary file:

- `Models_Benchmark_v2/baseline_batch_run_report.json`

## Recommended environment

- Dataset: `gravity_dataset_100k_V2_REALISTIC.pt`
- GPU: use an idle A10 such as `GPU 1`, `GPU 6`, or `GPU 7`
- Batch size: `64` first; reduce to `32` if memory or throughput is unstable

## 1. Train BasicCNN on the realistic v2 dataset

```bash
cd /your/project/path
conda activate pytorch
CUDA_VISIBLE_DEVICES=7 python baseline_code/Baseline_1D_CNN.py \
  --dataset gravity_dataset_100k_V2_REALISTIC.pt \
  --output-dir Models_Benchmark_v2 \
  --experiment-name BasicCNN_v2_realistic \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

Estimated runtime on a single NVIDIA A10:

- about `20-40 min` if early stopping is triggered relatively early
- about `40-60 min` in a slower run or with smaller batch size

## 2. Train DnCNN on the realistic v2 dataset

```bash
cd /your/project/path
conda activate pytorch
CUDA_VISIBLE_DEVICES=7 python baseline_code/Baseline_DnCNN.py \
  --dataset gravity_dataset_100k_V2_REALISTIC.pt \
  --output-dir Models_Benchmark_v2 \
  --experiment-name DnCNN_v2_realistic \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

Estimated runtime on a single NVIDIA A10:

- about `45-90 min` in most runs
- potentially longer if the batch size must be reduced to `32`

## 3. Train a plain 1D U-Net baseline

```bash
cd /your/project/path
conda activate pytorch
CUDA_VISIBLE_DEVICES=7 python baseline_code/Baseline_UNet1D.py \
  --dataset gravity_dataset_100k_V2_REALISTIC.pt \
  --output-dir Models_Benchmark_v2 \
  --experiment-name UNet1D_v2_realistic \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

Estimated runtime on a single NVIDIA A10:

- about `30-60 min`

## 4. Train a dilated TCN baseline

```bash
cd /your/project/path
conda activate pytorch
CUDA_VISIBLE_DEVICES=7 python baseline_code/Baseline_TCN.py \
  --dataset gravity_dataset_100k_V2_REALISTIC.pt \
  --output-dir Models_Benchmark_v2 \
  --experiment-name TCN_v2_realistic \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

Estimated runtime on a single NVIDIA A10:

- about `35-70 min`

## 5. Files to bring back after training

For `BasicCNN_v2_realistic`:

- `Models_Benchmark_v2/BasicCNN_v2_realistic_checkpoint.pt`
- `Models_Benchmark_v2/BasicCNN_v2_realistic_history.csv`
- `Models_Benchmark_v2/BasicCNN_v2_realistic_summary.json`

For `DnCNN_v2_realistic`:

- `Models_Benchmark_v2/DnCNN_v2_realistic_checkpoint.pt`
- `Models_Benchmark_v2/DnCNN_v2_realistic_history.csv`
- `Models_Benchmark_v2/DnCNN_v2_realistic_summary.json`

For `UNet1D_v2_realistic`:

- `Models_Benchmark_v2/UNet1D_v2_realistic_checkpoint.pt`
- `Models_Benchmark_v2/UNet1D_v2_realistic_history.csv`
- `Models_Benchmark_v2/UNet1D_v2_realistic_summary.json`

For `TCN_v2_realistic`:

- `Models_Benchmark_v2/TCN_v2_realistic_checkpoint.pt`
- `Models_Benchmark_v2/TCN_v2_realistic_history.csv`
- `Models_Benchmark_v2/TCN_v2_realistic_summary.json`

## 6. What happens after you move them back

Once the new checkpoints are copied back into the local project, rerun:

- `independent_resampling_eval.py`

with the new checkpoint paths. That will regenerate:

- the baseline statistics
- the grouped comparison figures
- the deep-learning comparison table values used in the paper
