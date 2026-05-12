# DnResUnet

Code package for the manuscript **"Deep Residual U-Net Denoising of Borehole Gravity Data Under Non-stationary Noise"**, prepared for submission to *Journal of Applied Geophysics*.

This release is organized as a public-facing reproducibility package rather than a working lab directory. It contains the final scripts used by the current manuscript, trained checkpoints for the reported deep-learning models, independent-evaluation outputs, component-wise noise diagnostics, inference-efficiency benchmarking, an executable baseline suite, environment specifications, and licensing information.

## Scope of This Release

This repository is intended to support transparent review and reproducibility for the *Journal of Applied Geophysics* submission:

- the code is distributed under an open-source license;
- the main training, data-generation, and evaluation workflows are executable from the repository;
- pretrained model checkpoints used by the paper are included;
- representative output tables and grouped figures are included for reproducibility;
- environment requirements and usage instructions are documented in a persistent text form.

For the manuscript, the repository URL is reported in the `Computer Code Availability` and `Data Availability` sections.

## Repository Layout

- `DnResUnet_code.py`
  Final configurable training pipeline for DnResUnet and all deep-learning baselines.
- `forward_v2.py`
  Realistic synthetic borehole-gravity dataset generator with randomized acquisition regimes, expanded geological scenarios, and compound non-stationary noise.
- `independent_resampling_eval.py`
  Independent resampling benchmark, grouped-figure export, and no-ground-truth field-review template export.
- `component_noise_eval.py`
  Diagnostic experiment for isolated non-stationary noise components, including heteroscedastic noise, drift, colored noise, step offsets, spikes, and mild depth-axis misalignment.
- `efficiency_eval.py`
  CPU/GPU inference-efficiency benchmark for DnResUnet, deep-learning baselines, and classical filters.
- `data.py`
  Shared data utilities.
- `inspect_v2_dataset.py`
  Dataset inspection and plotting utility for checking generated samples.
- `baseline_code/`
  Standalone wrappers for BasicCNN, DnCNN, UNet1D, TCN, and batch baseline training.
- `checkpoints/main_model/`
  Final DnResUnet checkpoint and training summaries used by the manuscript.
- `checkpoints/baselines/`
  Final deep-learning baseline checkpoints and training summaries used by the manuscript.
- `results/independent_resampling_v3/`
  Exported statistics and grouped comparison figures corresponding to the latest independent benchmark.
- `component_noise_results/` and `efficiency_results/`
  Optional output locations for the additional diagnostic experiments introduced during revision.
- `sample_data/`
  A smoke-test dataset, metadata for the full realistic dataset, and related JSON descriptors.

## What Is Included and What Is Not

Included:

- all source code required to generate data, train models, run baseline experiments, and reproduce the independent evaluation workflow;
- scripts for the component-wise noise analysis and inference-efficiency comparison added in the revised manuscript;
- the trained DnResUnet model used in the paper;
- the trained deep-learning baselines used in the final comparison;
- the latest independent benchmark statistics and paper-ready grouped figures;
- a small smoke-test dataset for quick environment checks.

Not included:

- the full `gravity_dataset_100k_V2_REALISTIC.pt` training corpus, because large binary datasets are better archived separately from the code repository.

To reproduce the full benchmark from scratch, first generate the realistic dataset with `forward_v2.py`, then train the models, and finally run `independent_resampling_eval.py`. The component-wise and efficiency analyses can be run from the provided checkpoints.

## Installation

Create an isolated Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

An example Conda environment file is also provided in `environment.yml`.

## Quick Start

### 1. Smoke-test the environment

```bash
python inspect_v2_dataset.py --dataset sample_data/gravity_dataset_v2_smoketest.pt --output-dir smoke_test_inspection
```

### 2. Generate the full realistic dataset

```bash
python forward_v2.py --samples-per-level 20000 --output gravity_dataset_100k_V2_REALISTIC.pt
```

### 3. Train the main DnResUnet model

```bash
python DnResUnet_code.py \
  --dataset gravity_dataset_100k_V2_REALISTIC.pt \
  --output-dir Models_v3_realistic \
  --experiment-name dnresunet_v2_realistic \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda
```

### 4. Train all deep-learning baselines

```bash
python baseline_code/Run_All_Baselines.py \
  --dataset gravity_dataset_100k_V2_REALISTIC.pt \
  --output-dir Models_Benchmark_v2 \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --skip-existing
```

### 5. Run the independent benchmark

```bash
python independent_resampling_eval.py \
  --dnresunet-path checkpoints/main_model/dnresunet_v2_realistic_checkpoint.pt \
  --basiccnn-path checkpoints/baselines/BasicCNN_v2_realistic_checkpoint.pt \
  --dncnn-path checkpoints/baselines/DnCNN_v2_realistic_checkpoint.pt \
  --unet1d-path checkpoints/baselines/UNet1D_v2_realistic_checkpoint.pt \
  --tcn-path checkpoints/baselines/TCN_v2_realistic_checkpoint.pt \
  --samples-per-noise 200 \
  --output-dir reproducibility_eval \
  --device cuda
```

### 6. Run the component-wise noise diagnostic

This experiment evaluates the trained DnResUnet checkpoint under isolated non-stationary noise components without retraining.

```bash
python component_noise_eval.py \
  --checkpoint checkpoints/main_model/dnresunet_v2_realistic_checkpoint.pt \
  --samples-per-component 20 \
  --noise-std 0.05 \
  --output-dir component_noise_results \
  --device cuda
```

The script writes:

- `component_noise_results/component_noise_summary.csv`
- `component_noise_results/component_noise_raw.csv`

Use `--device cpu` if CUDA is unavailable.

### 7. Run the inference-efficiency benchmark

This experiment reports parameter counts, checkpoint sizes, single-profile latency, and batched latency.

```bash
python efficiency_eval.py \
  --device cpu \
  --threads 1 \
  --warmup 10 \
  --repeats 50 \
  --batch-size 64 \
  --output-dir efficiency_results
```

The script writes:

- `efficiency_results/deep_model_efficiency.csv`
- `efficiency_results/classical_efficiency.csv`

## Reproducibility Notes

- All paper results correspond to the realistic synthetic data pipeline implemented in `forward_v2.py`.
- The main network reported in the manuscript is the GroupNorm-based DnResUnet configuration in `DnResUnet_code.py`.
- The deep-learning baseline comparison uses BasicCNN, DnCNN, UNet1D, and TCN checkpoints contained in `checkpoints/baselines/`.
- The grouped benchmark figures and summary tables used in the manuscript are derived from the independent resampling workflow.
- The component-wise diagnostic table is produced by `component_noise_eval.py`.
- The computational-efficiency table is produced by `efficiency_eval.py`.

## Journal of Applied Geophysics Release Notes

Before making the public repository live, complete these final release steps:

1. Push this folder to a permanent public repository such as GitHub or GitLab.
2. Verify that the public repository contains `component_noise_eval.py`, `efficiency_eval.py`, the checkpoints, and the smoke-test data.
3. Keep the repository URL synchronized with the manuscript: `https://github.com/SongZW-bit/DnResUnet`.
4. If a versioned archive or dataset DOI is minted later, add it to the manuscript during proof or future release updates.

## Contact

For correspondence related to the code package, please update this section with the final corresponding-author information before public release.

- Contact email: `songzw24@mails.jlu.edu.cn`

## License

This project is distributed under the MIT License. See `LICENSE` for details.
