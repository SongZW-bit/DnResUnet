# DnResUnet

Code package for the manuscript **"An Intelligent Denoising Method for Borehole Gravity Data Based on Deep Residual Networks"**, prepared for submission to *Journal of Applied Geophysics*.

This release is organized as a public-facing reproducibility package rather than a working lab directory. It contains the final scripts used by the current manuscript, trained checkpoints for the reported deep-learning models, independent-evaluation outputs, an executable baseline suite, environment specifications, and licensing information.

## Scope of This Release

This repository is intended to satisfy the core software-sharing expectations of *Computers & Geosciences*:

- the code is distributed under an open-source license;
- the main training, data-generation, and evaluation workflows are executable from the repository;
- pretrained model checkpoints used by the paper are included;
- representative output tables and grouped figures are included for reproducibility;
- environment requirements and usage instructions are documented in a persistent text form.

For a public release, the repository URL and archival DOI should be added to the manuscript's `Computer Code Availability` section.

## Repository Layout

- `DnResUnet_code.py`
  Final configurable training pipeline for DnResUnet and all deep-learning baselines.
- `forward_v2.py`
  Realistic synthetic borehole-gravity dataset generator with randomized acquisition regimes, expanded geological scenarios, and compound non-stationary noise.
- `independent_resampling_eval.py`
  Independent resampling benchmark, grouped-figure export, and no-ground-truth field-review template export.
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
- `sample_data/`
  A smoke-test dataset, metadata for the full realistic dataset, and related JSON descriptors.

## What Is Included and What Is Not

Included:

- all source code required to generate data, train models, run baseline experiments, and reproduce the independent evaluation workflow;
- the trained DnResUnet model used in the paper;
- the trained deep-learning baselines used in the final comparison;
- the latest independent benchmark statistics and paper-ready grouped figures;
- a small smoke-test dataset for quick environment checks.

Not included:

- the full `gravity_dataset_100k_V2_REALISTIC.pt` training corpus, because large binary datasets are better archived separately from the code repository.

To reproduce the full benchmark from scratch, first generate the realistic dataset with `forward_v2.py`, then train the models, and finally run `independent_resampling_eval.py`.

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

## Reproducibility Notes

- All paper results correspond to the realistic synthetic data pipeline implemented in `forward_v2.py`.
- The main network reported in the manuscript is the GroupNorm-based DnResUnet configuration in `DnResUnet_code.py`.
- The deep-learning baseline comparison uses BasicCNN, DnCNN, UNet1D, and TCN checkpoints contained in `checkpoints/baselines/`.
- The grouped benchmark figures and summary tables used in the manuscript are derived from the independent resampling workflow.

## Computers & Geosciences Release Notes

Before making the public repository live, complete these final release steps:

1. Push this folder to a permanent public repository such as GitHub or GitLab.
2. Create a versioned archive release and mint a DOI through Zenodo or another archival service.
3. Add the final repository URL, release tag, DOI, and access date to the manuscript's `Computer Code Availability` section.
4. If the full realistic dataset is released separately, cite its archival DOI in a separate `Data Availability` statement.

## Contact

For correspondence related to the code package, please update this section with the final corresponding-author information before public release.

- Contact email: `songzw24@mails.jlu.edu.cn`

## License

This project is distributed under the MIT License. See `LICENSE` for details.
