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
[Training](#training) |
[Evaluation](#evaluation) |
[Checks](#checks)

</div>

## Overview

EEGDL is a Lightning + Hydra project for EEG data preprocessing, model
training, and evaluation. It keeps the original train/eval organization from
`lightning-hydra-template` and adds a THINGS-EEG2 preprocessing workflow driven
by Hydra configs.

This repository was initially copied from the `main` branch of
[`ashleve/lightning-hydra-template`](https://github.com/ashleve/lightning-hydra-template)
and then adapted for EEG data processing and experimentation.

### At A Glance

| Area        | Entry                  | Config                                      | Purpose                            |
| ----------- | ---------------------- | ------------------------------------------- | ---------------------------------- |
| Environment | `uv sync`              | `pyproject.toml`, `uv.lock`                 | Reproducible dependency management |
| Preprocess  | `src/preprocess.py`    | `configs/preprocess.yaml`                   | Build THINGS-EEG2 `.pt` tensors    |
| Train       | `src/train.py`         | `configs/train.yaml`                        | Run Lightning training             |
| Evaluate    | `src/eval.py`          | `configs/eval.yaml`                         | Evaluate a checkpoint              |
| Paths       | Hydra composition      | `configs/paths/default.yaml`                | Share dataset and output locations |
| Quality     | `pre-commit`, `pytest` | `.pre-commit-config.yaml`, `pyproject.toml` | Lint, format, and test             |

### Stack

| Layer             | Tools                           |
| ----------------- | ------------------------------- |
| Runtime           | Python 3.10+, uv                |
| Deep learning     | PyTorch, TorchVision, Lightning |
| Configuration     | Hydra                           |
| EEG preprocessing | MNE, SciPy, scikit-learn        |
| Quality           | pytest, pre-commit              |

## Quickstart

```bash
uv sync --extra cpu
uv run --no-sync python src/preprocess.py preprocess.subject_id=1
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

| Key                           | Default                                   | Purpose                            |
| ----------------------------- | ----------------------------------------- | ---------------------------------- |
| `data_dir`                    | `${paths.root_dir}/data/`                 | Base data directory                |
| `thingseeg2_raw_dir`          | `${paths.data_dir}/thingseeg2-eegs`       | Raw THINGS-EEG2 EEG files          |
| `thingseeg2_img_dir`          | `${paths.data_dir}/thingseeg2-images`     | THINGS-EEG2 image metadata folders |
| `thingseeg2_preprocessed_dir` | `${paths.data_dir}/thingseeg2-eeg2-250hz` | Preprocessed subject outputs       |

The THINGS-EEG2 preprocessing config uses those shared keys:

| Preprocess option | Default                                |
| ----------------- | -------------------------------------- |
| `raw_data_dir`    | `${paths.thingseeg2_raw_dir}`          |
| `img_data_dir`    | `${paths.thingseeg2_img_dir}`          |
| `save_dir`        | `${paths.thingseeg2_preprocessed_dir}` |

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

| Option         | Meaning                                      |
| -------------- | -------------------------------------------- |
| `subject_id`   | Subject identifier                           |
| `num_sessions` | Number of sessions to process                |
| `sfreq`        | Target sampling frequency                    |
| `tmin`, `tmax` | Epoch window around stimulus onset           |
| `mvnn_dim`     | MVNN covariance mode, or `null` to skip MVNN |
| `raw_data_dir` | Raw EEG directory                            |
| `img_data_dir` | Image metadata directory                     |
| `save_dir`     | Preprocessed output directory                |
| `seed`         | Trial selection seed                         |

### Platform Scripts

| Platform | Command                            | Example Range                          | With Override                                                |
| -------- | ---------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Windows  | `.\scripts\windows\preprocess.ps1` | `.\scripts\windows\preprocess.ps1 1 3` | `.\scripts\windows\preprocess.ps1 1 10 preprocess.sfreq=250` |
| Linux    | `bash scripts/linux/preprocess.sh` | `bash scripts/linux/preprocess.sh 1 3` | `bash scripts/linux/preprocess.sh 1 10 preprocess.sfreq=250` |
| macOS    | `bash scripts/macos/preprocess.sh` | `bash scripts/macos/preprocess.sh 1 3` | `bash scripts/macos/preprocess.sh 1 10 preprocess.sfreq=250` |

The first two script arguments are the start and end subject IDs. Remaining
arguments are passed through as Hydra overrides.

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
| Train smoke tests | `uv run --no-sync pytest tests/test_train.py`       |
| All hooks         | `uv run --no-sync pre-commit run --all-files`       |
| README hooks      | `uv run --no-sync pre-commit run --files README.md` |

## Project Layout

```text
configs/
  train.yaml
  eval.yaml
  preprocess.yaml
  preprocess/thingseeg2.yaml
  paths/default.yaml
scripts/
  windows/preprocess.ps1
  linux/preprocess.sh
  macos/preprocess.sh
src/
  train.py
  eval.py
  preprocess.py
  preprocessors/thingseeg2.py
  data/
  models/
  utils/
tests/
  test_configs.py
  test_preprocess.py
  test_train.py
```

## Notes

- Use `uv run --no-sync ...` after syncing the environment.
- Use Hydra command-line overrides for one-off experiments.
- Keep machine-specific overrides in `configs/local/default.yaml` if needed.
