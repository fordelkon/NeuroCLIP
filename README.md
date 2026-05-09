<div align="center">

# EEGDL

EEG deep learning workflows with Lightning, Hydra, uv, and THINGS-EEG2
preprocessing and feature extraction.

[![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/Lightning-2.0%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/Hydra-1.3-89B8CD)](https://hydra.cc/)
[![uv](https://img.shields.io/badge/uv-managed-261230?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit&logoColor=black)](https://pre-commit.com/)
[![dataset](https://img.shields.io/badge/dataset-THINGS--EEG2-2E7D32)](https://osf.io/3jk45/)

[Overview](#overview) |
[Quickstart](#quickstart) |
[Data Paths](#data-paths) |
[Data Preparation](#data-preparation) |
[Training](#training) |
[Evaluation](#evaluation) |
[Checks](#checks)

</div>

## Overview

EEGDL is a Lightning + Hydra project for EEG data preprocessing, CLIP feature
extraction, model training, and evaluation. It keeps the original train/eval
organization from `lightning-hydra-template` and adds THINGS-EEG2 preprocessing
and feature extraction workflows driven by Hydra configs.

This repository was initially copied from the `main` branch of
[`ashleve/lightning-hydra-template`](https://github.com/ashleve/lightning-hydra-template)
and then adapted for EEG data processing and experimentation.

### At A Glance

| Area        | Entry                  | Config                                      | Purpose                            |
| ----------- | ---------------------- | ------------------------------------------- | ---------------------------------- |
| Environment | `uv sync`              | `pyproject.toml`, `uv.lock`                 | Reproducible dependency management |
| Preprocess  | `src/preprocess.py`    | `configs/preprocess.yaml`                   | Build THINGS-EEG2 `.pt` tensors    |
| Extract     | `src/extract.py`       | `configs/extract.yaml`                      | Build THINGS-EEG2 CLIP features    |
| Train       | `src/train.py`         | `configs/train.yaml`                        | Run Lightning training             |
| Evaluate    | `src/eval.py`          | `configs/eval.yaml`                         | Evaluate a checkpoint              |
| Paths       | Hydra composition      | `configs/paths/default.yaml`                | Share dataset and output locations |
| Quality     | `pre-commit`, `pytest` | `.pre-commit-config.yaml`, `pyproject.toml` | Lint, format, and test             |

### Stack

| Layer              | Tools                           |
| ------------------ | ------------------------------- |
| Runtime            | Python 3.10+, uv                |
| Deep learning      | PyTorch, TorchVision, Lightning |
| Configuration      | Hydra                           |
| EEG preprocessing  | MNE, SciPy, scikit-learn        |
| Feature extraction | Transformers, Hugging Face CLIP |
| Quality            | pytest, pre-commit              |

## Quickstart

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already (e.g., via `curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux/macOS or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows).

```bash
uv sync --extra cpu
uv run --no-sync python src/preprocess.py preprocess.subject_id=1
uv run --no-sync python src/extract.py
uv run --no-sync python src/train.py
uv run --no-sync pytest
```

Use a CUDA extra when needed:

```bash
uv sync --extra cu126
```

Other configured CUDA extras are `cu128` and `cu129`.

Install Git hooks after syncing the environment:

```bash
uv run --no-sync pre-commit install
```

## Data Paths

> 🚨 <span style="color:red; font-size:1.1em; font-weight:bold;">CRITICAL: PATH CONFIGURATION</span> 🚨
>
> Shared paths live in `configs/paths/default.yaml`. **It is STRONGLY RECOMMENDED to edit this file directly** to map your data locations. This ensures a persistent configuration for the complete `unzip -> preprocess -> extract` flow.
>
> ⚠️ <span style="color:#d97706; font-weight:bold;">IF YOU DO NOT USE THE DEFAULT PATHS:</span>
> You **MUST** maintain strict consistency. Any path override via CLI in one step **MUST BE EXACTLY PRESERVED AND PASSED** to **EVERY subsequent step** that follows. If you fail to supply consistent overrides, Hydra will fallback to the default directories and the scripts **WILL FAIL to find your data**!

| Key                            | Default                                      | Used by             |
| ------------------------------ | -------------------------------------------- | ------------------- |
| `thingseeg2_raw_dir`           | `${paths.data_dir}/thingseeg2-eegs`          | unzip, preprocess   |
| `thingseeg2_img_dir`           | `${paths.data_dir}/thingseeg2-images`        | unzip, preprocess   |
| `thingseeg2_preprocessed_dir`  | `${paths.data_dir}/thingseeg2-eeg2-250hz`    | preprocess, extract |
| `thingseeg2_clip_features_dir` | `${paths.data_dir}/thingseeg2-clip-features` | extract, train      |
| `clip_model_cache_dir`         | `${paths.data_dir}/clip-cache`               | extract             |

When using CLI overrides instead of editing `configs/paths/default.yaml`, you **MUST** repeat the exact same `paths.*` overrides across all steps to maintain unity:

```bash
# ❌ INCORRECT (Inconsistent overrides, next step will look in default path)
bash unzip.sh /downloads/eeg /downloads/img paths.thingseeg2_raw_dir=/custom/eeg
uv run src/preprocess.py # FAIL: Missing paths.thingseeg2_raw_dir override!

# ✅ CORRECT (Strictly uniform overrides across the entire pipeline)
# 1. Unzip
bash scripts/linux/unzip.sh /zips/eeg /zips/img paths.thingseeg2_raw_dir=/custom/eeg paths.thingseeg2_preprocessed_dir=/custom/prep
# 2. Preprocess (Requires the EXACT same paths from Unzip, plus any new ones)
bash scripts/linux/preprocess.sh 1 10 paths.thingseeg2_raw_dir=/custom/eeg paths.thingseeg2_preprocessed_dir=/custom/prep
# 3. Extract (Requires the EXACT same paths from Preprocess)
bash scripts/linux/extract.sh paths.thingseeg2_preprocessed_dir=/custom/prep paths.thingseeg2_clip_features_dir=/custom/clip
```

## Data Preparation

The THINGS-EEG2 data preparation flow is:

```text
downloaded zip files -> unzip -> preprocess EEG -> extract CLIP features
```

### 1. Unzip

Expected downloaded zip layout:

```text
<eeg_zip_dir>/
  sub-01.zip
  sub-02.zip
  ...
<image_zip_dir>/
  training_images.zip
  test_images.zip
```

Unzip targets are resolved from `paths.thingseeg2_raw_dir` and
`paths.thingseeg2_img_dir`.

Target layout after unzip:

```text
data/thingseeg2-eegs/
  sub-01/
    ses-01/
      raw_eeg_training.npy
      raw_eeg_test.npy
data/thingseeg2-images/
  training_images/
  test_images/
```

Commands:

| Platform | Default Source & Target       | Custom Source & Target Dir (Override Example)                                                                                       |
| -------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Windows  | `.\scripts\windows\unzip.ps1` | `.\scripts\windows\unzip.ps1 D:\zips\eeg D:\zips\img paths.thingseeg2_raw_dir=D:\output\eeg paths.thingseeg2_img_dir=D:\output\img` |
| Linux    | `bash scripts/linux/unzip.sh` | `bash scripts/linux/unzip.sh /zips/eeg /zips/img paths.thingseeg2_raw_dir=/output/eeg paths.thingseeg2_img_dir=/output/img`         |
| macOS    | `bash scripts/macos/unzip.sh` | `bash scripts/macos/unzip.sh /zips/eeg /zips/img paths.thingseeg2_raw_dir=/output/eeg paths.thingseeg2_img_dir=/output/img`         |

Hydra overrides for destination targets come after the optional zip source directories:

```bash
# Default paths (Reads from config, unzips from and to default directories)
bash scripts/linux/unzip.sh

# Custom paths (Specify both zip source dirs and target output dirs)
bash scripts/linux/unzip.sh /downloads/eeg-zips /downloads/image-zips \
  paths.thingseeg2_raw_dir=/data/thingseeg2-eegs \
  paths.thingseeg2_img_dir=/data/thingseeg2-images
```

### 2. Preprocess

Main entrypoint:

```bash
uv run --no-sync python src/preprocess.py
```

Common overrides:

```bash
uv run --no-sync python src/preprocess.py preprocess.subject_id=1
uv run --no-sync python src/preprocess.py preprocess.subject_id=1 preprocess.sfreq=250
uv run --no-sync python src/preprocess.py preprocess.subject_id=1 preprocess.mvnn_dim=null
```

Default THINGS-EEG2 options live in `configs/preprocess/thingseeg2.yaml`.

| Option         | Meaning                                        |
| -------------- | ---------------------------------------------- |
| `subject_id`   | Subject identifier                             |
| `num_sessions` | Number of sessions to process                  |
| `sfreq`        | Target sampling frequency                      |
| `tmin`, `tmax` | Epoch window around stimulus onset             |
| `mvnn_dim`     | Normalization mode: `time`, `epoch`, or `null` |
| `raw_data_dir` | Raw EEG directory                              |
| `img_data_dir` | Image metadata directory                       |
| `save_dir`     | Preprocessed output directory                  |
| `seed`         | Trial selection seed                           |

The saved EEG tensors keep the post-stimulus `[0, 1)` second window. Training
and test partitions are normalized independently per subject and session:

| `mvnn_dim` | Normalization behavior                                           |
| ---------- | ---------------------------------------------------------------- |
| `time`     | Apply MVNN whitening with covariance averaged across time points |
| `epoch`    | Apply MVNN whitening with covariance averaged across epochs      |
| `null`     | Disable MVNN and apply per-session channel z-score               |

For MVNN modes, both training and test data use whitening statistics estimated
from the training partition. For `null`, z-score mean and standard deviation are
also estimated from the training partition, then applied to both partitions.

Preprocessed output layout:

```text
data/thingseeg2-eeg2-250hz/
  sub-01/
    training.pt
    test.pt
```

Scripts:

| Platform | Default paths (Recommended)        | Custom paths (Override Example)                                                                                                 | With Options Override                                        |
| -------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Windows  | `.\scripts\windows\preprocess.ps1` | `.\scripts\windows\preprocess.ps1 1 10 paths.thingseeg2_raw_dir=D:\output\eeg paths.thingseeg2_preprocessed_dir=D:\output\prep` | `.\scripts\windows\preprocess.ps1 1 10 preprocess.sfreq=250` |
| Linux    | `bash scripts/linux/preprocess.sh` | `bash scripts/linux/preprocess.sh 1 10 paths.thingseeg2_raw_dir=/output/eeg paths.thingseeg2_preprocessed_dir=/output/prep`     | `bash scripts/linux/preprocess.sh 1 10 preprocess.sfreq=250` |
| macOS    | `bash scripts/macos/preprocess.sh` | `bash scripts/linux/preprocess.sh 1 10 paths.thingseeg2_raw_dir=/output/eeg paths.thingseeg2_preprocessed_dir=/output/prep`     | `bash scripts/macos/preprocess.sh 1 10 preprocess.sfreq=250` |

The first two script arguments are the start and end subject IDs. Remaining
arguments are passed through as Hydra overrides.

### 3. Extract

Main entrypoint:

```bash
uv run --no-sync python src/extract.py
```

Common overrides:

```bash
uv run --no-sync python src/extract.py extract.reference_subject_id=1
uv run --no-sync python src/extract.py extract.model_name=ViT-B-32
uv run --no-sync python src/extract.py extract.feature_mode=pooled
uv run --no-sync python src/extract.py extract.device=cpu
uv run --no-sync python src/extract.py extract.extract_text=false
```

Default THINGS-EEG2 extraction options live in
`configs/extract/thingseeg2.yaml`.

| Option                 | Meaning                                             |
| ---------------------- | --------------------------------------------------- |
| `prep_eeg_data_dir`    | Preprocessed THINGS-EEG2 EEG directory              |
| `save_dir`             | CLIP feature output directory                       |
| `reference_subject_id` | Subject used to read image paths, texts, and labels |
| `model_name`           | Short CLIP model name                               |
| `model_id`             | Optional Hugging Face model id override             |
| `model_cache_dir`      | Hugging Face model cache directory                  |
| `partitions`           | Preprocessed partitions to extract                  |
| `batch_size`           | Inference batch size                                |
| `device`               | Inference device selection                          |
| `feature_mode`         | Saved CLIP feature type                             |
| `extract_image`        | Toggle image feature extraction                     |
| `extract_text`         | Toggle text feature extraction                      |

Supported `feature_mode` values:

| Mode                       | Output                               |
| -------------------------- | ------------------------------------ |
| `pooled`                   | Projected global CLIP features       |
| `last_hidden_state_no_cls` | Token features without the CLS token |

Device behavior:

| `device` value | Behavior                                                      |
| -------------- | ------------------------------------------------------------- |
| `auto`         | Use all visible CUDA devices, or CPU when CUDA is unavailable |
| `cuda`         | Use all visible CUDA devices and fail if CUDA is unavailable  |
| `cuda:N`       | Use only the selected CUDA device                             |
| `cpu`          | Use CPU inference                                             |

The saved feature rows follow the preprocessed EEG image order, so
`image_features[i]` aligns with `eeg[i, rep]` for every repetition.

Extracted CLIP feature layout:

```text
data/thingseeg2-clip-features/
  ViT-B-32/
    laion-CLIP-ViT-B-32-laion2B-s34B-b79K/
      pooled/
        training.pt
        test.pt
```

Scripts:

| Platform | Default paths (Recommended)     | Custom paths (Override Example)                                                                                                    | With Options Override                                          |
| -------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Windows  | `.\scripts\windows\extract.ps1` | `.\scripts\windows\extract.ps1 paths.thingseeg2_preprocessed_dir=D:\output\prep paths.thingseeg2_clip_features_dir=D:\output\clip` | `.\scripts\windows\extract.ps1 extract.reference_subject_id=1` |
| Linux    | `bash scripts/linux/extract.sh` | `bash scripts/linux/extract.sh paths.thingseeg2_preprocessed_dir=/output/prep paths.thingseeg2_clip_features_dir=/output/clip`     | `bash scripts/linux/extract.sh extract.reference_subject_id=1` |
| macOS    | `bash scripts/macos/extract.sh` | `bash scripts/linux/extract.sh paths.thingseeg2_preprocessed_dir=/output/prep paths.thingseeg2_clip_features_dir=/output/clip`     | `bash scripts/macos/extract.sh extract.reference_subject_id=1` |

Extraction scripts do not loop over subject ranges. All arguments are passed
through as Hydra overrides.

## Training

Main entrypoint:

```bash
uv run --no-sync python src/train.py
```

Common examples:

```bash
uv run --no-sync python src/train.py trainer=cpu
uv run --no-sync python src/train.py trainer=gpu
uv run --no-sync python src/train.py logger=tensorboard
uv run --no-sync python src/train.py experiment=example
uv run --no-sync python src/train.py trainer.max_epochs=20
uv run --no-sync python src/train.py ckpt_path="/path/to/checkpoint.ckpt"
```

The default training config is `configs/train.yaml`.

## Evaluation

Main entrypoint:

```bash
uv run --no-sync python src/eval.py ckpt_path="/path/to/checkpoint.ckpt"
```

The default evaluation config is `configs/eval.yaml`. A checkpoint path is
required.

## Checks

| Task              | Command                                             |
| ----------------- | --------------------------------------------------- |
| Full tests        | `uv run --no-sync pytest`                           |
| Preprocess tests  | `uv run --no-sync pytest tests/test_preprocess.py`  |
| Extract tests     | `uv run --no-sync pytest tests/test_extract.py`     |
| Train smoke tests | `uv run --no-sync pytest tests/test_train.py`       |
| All hooks         | `uv run --no-sync pre-commit run --all-files`       |
| README hooks      | `uv run --no-sync pre-commit run --files README.md` |

## Project Layout

```text
configs/
  train.yaml
  eval.yaml
  preprocess.yaml
  extract.yaml
  data/thingseeg2.yaml
  preprocess/thingseeg2.yaml
  extract/thingseeg2.yaml
  paths/default.yaml
scripts/
  windows/unzip.ps1
  windows/preprocess.ps1
  windows/extract.ps1
  linux/unzip.sh
  linux/preprocess.sh
  linux/extract.sh
  macos/unzip.sh
  macos/preprocess.sh
  macos/extract.sh
src/
  train.py
  eval.py
  preprocess.py
  extract.py
  preprocessors/thingseeg2.py
  extractors/thingseeg2.py
  data/
    thingseeg2_datamodule.py
    components/thingseeg2_dataset.py
  models/
  utils/
tests/
  test_configs.py
  test_preprocess.py
  test_extract.py
  test_thingseeg2_dataset.py
  test_thingseeg2_datamodule.py
  test_train.py
```

## Notes

- Use `uv run --no-sync ...` after syncing the environment.
- Use Hydra command-line overrides for one-off experiments.
- Keep machine-specific overrides in `configs/local/default.yaml` if needed.
