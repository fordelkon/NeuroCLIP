<div align="center">

# EEGDL

EEG deep learning workflows with Lightning, Hydra, uv, and THINGS-EEG2
preprocessing.

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
[Preprocessing](#preprocessing) |
[Extraction](#extraction) |
[Training](#training) |
[Evaluation](#evaluation) |
[Checks](#checks)

</div>

## Overview

EEGDL is a Lightning + Hydra project for EEG data preprocessing, CLIP feature
extraction, model training, and evaluation. It keeps the original train/eval
organization from `lightning-hydra-template` and adds THINGS-EEG2 preprocessing
and extraction workflows driven by Hydra configs.

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

Shared paths live in `configs/paths/default.yaml`. Change these defaults when
multiple workflows should use the same dataset location.

| Key                            | Default                                      | Purpose                            |
| ------------------------------ | -------------------------------------------- | ---------------------------------- |
| `data_dir`                     | `${paths.root_dir}/data/`                    | Base data directory                |
| `thingseeg2_raw_dir`           | `${paths.data_dir}/thingseeg2-eegs`          | Raw THINGS-EEG2 EEG files          |
| `thingseeg2_img_dir`           | `${paths.data_dir}/thingseeg2-images`        | THINGS-EEG2 image metadata folders |
| `thingseeg2_preprocessed_dir`  | `${paths.data_dir}/thingseeg2-eeg2-250hz`    | Preprocessed subject outputs       |
| `thingseeg2_clip_features_dir` | `${paths.data_dir}/thingseeg2-clip-features` | Extracted CLIP feature outputs     |
| `clip_model_cache_dir`         | `${paths.data_dir}/clip-cache`               | Hugging Face CLIP model cache      |

The THINGS-EEG2 preprocessing config uses those shared keys:

| Preprocess option | Default                                |
| ----------------- | -------------------------------------- |
| `raw_data_dir`    | `${paths.thingseeg2_raw_dir}`          |
| `img_data_dir`    | `${paths.thingseeg2_img_dir}`          |
| `save_dir`        | `${paths.thingseeg2_preprocessed_dir}` |

The THINGS-EEG2 extraction config uses these shared keys:

| Extract option      | Default                                 |
| ------------------- | --------------------------------------- |
| `prep_eeg_data_dir` | `${paths.thingseeg2_preprocessed_dir}`  |
| `save_dir`          | `${paths.thingseeg2_clip_features_dir}` |
| `model_cache_dir`   | `${paths.clip_model_cache_dir}`         |

Expected raw EEG layout:

```text
data/thingseeg2-eegs/
  sub-01/
    ses-01/
      raw_eeg_training.npy
      raw_eeg_test.npy
```

Expected image layout:

```text
data/thingseeg2-images/
  training_images/
  test_images/
```

Preprocessed output layout:

```text
data/thingseeg2-eeg2-250hz/
  sub-01/
    training.pt
    test.pt
```

Extracted CLIP feature layout:

```text
data/thingseeg2-clip-features/
  ViT-B-32/
    laion-CLIP-ViT-B-32-laion2B-s34B-b79K/
      pooled/
        training.pt
        test.pt
```

Override paths for a single run:

```bash
uv run --no-sync python src/preprocess.py \
  paths.thingseeg2_raw_dir="/path/to/thingseeg2-eegs" \
  paths.thingseeg2_img_dir="/path/to/thingseeg2-images" \
  paths.thingseeg2_preprocessed_dir="/path/to/output"
```

## Preprocessing

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

### Platform Scripts

| Platform | Command                            | Example Range                          | With Override                                                |
| -------- | ---------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Windows  | `.\scripts\windows\preprocess.ps1` | `.\scripts\windows\preprocess.ps1 1 3` | `.\scripts\windows\preprocess.ps1 1 10 preprocess.sfreq=250` |
| Linux    | `bash scripts/linux/preprocess.sh` | `bash scripts/linux/preprocess.sh 1 3` | `bash scripts/linux/preprocess.sh 1 10 preprocess.sfreq=250` |
| macOS    | `bash scripts/macos/preprocess.sh` | `bash scripts/macos/preprocess.sh 1 3` | `bash scripts/macos/preprocess.sh 1 10 preprocess.sfreq=250` |

The first two script arguments are the start and end subject IDs. Remaining
arguments are passed through as Hydra overrides.

## Extraction

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

### Extraction Scripts

| Platform | Command                         | With Override                                                                |
| -------- | ------------------------------- | ---------------------------------------------------------------------------- |
| Windows  | `.\scripts\windows\extract.ps1` | `.\scripts\windows\extract.ps1 extract.reference_subject_id=1`               |
| Linux    | `bash scripts/linux/extract.sh` | `bash scripts/linux/extract.sh extract.reference_subject_id=1`               |
| macOS    | `bash scripts/macos/extract.sh` | `bash scripts/macos/extract.sh extract.reference_subject_id=1`               |
| Windows  | `.\scripts\windows\extract.ps1` | `.\scripts\windows\extract.ps1 extract.device=cpu extract.extract_text=true` |
| Linux    | `bash scripts/linux/extract.sh` | `bash scripts/linux/extract.sh extract.device=cpu extract.extract_text=true` |
| macOS    | `bash scripts/macos/extract.sh` | `bash scripts/macos/extract.sh extract.device=cpu extract.extract_text=true` |

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
  preprocess/thingseeg2.yaml
  extract/thingseeg2.yaml
  paths/default.yaml
scripts/
  windows/preprocess.ps1
  windows/extract.ps1
  linux/preprocess.sh
  linux/extract.sh
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
  models/
  utils/
tests/
  test_configs.py
  test_preprocess.py
  test_extract.py
  test_train.py
```

## Notes

- Use `uv run --no-sync ...` after syncing the environment.
- Use Hydra command-line overrides for one-off experiments.
- Keep machine-specific overrides in `configs/local/default.yaml` if needed.
