<div align="center">

# NeuroCLIP

EEG-to-CLIP representation learning workflows for THINGS-EEG2, built with
Lightning, Hydra, uv, and reproducible preprocessing, feature extraction,
training, and evaluation.

[![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/Lightning-2.0%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/Hydra-1.3-89B8CD)](https://hydra.cc/)
[![uv](https://img.shields.io/badge/uv-managed-261230?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit&logoColor=black)](https://pre-commit.com/)
[![dataset](https://img.shields.io/badge/dataset-THINGS--EEG2-2E7D32)](https://osf.io/3jk45/)

[Overview](#overview) |
[Quickstart](#quickstart) |
[Detailed Run Guide](#detailed-run-guide) |
[Checks](#checks)

</div>

______________________________________________________________________

## Workflow

```text
uv sync -> unzip datasets -> preprocess EEG -> extract CLIP features -> train -> evaluate
```

| Stage | What happens                                       | Main command                               |
| ----- | -------------------------------------------------- | ------------------------------------------ |
| Setup | Create the Python environment with uv.             | `uv sync --extra cpu`                      |
| Data  | Unzip THINGS-EEG2 EEG and image archives.          | `scripts/*/unzip.*`                        |
| EEG   | Convert raw EEG into normalized `.pt` tensors.     | `scripts/*/preprocess.*`                   |
| CLIP  | Extract image/text features aligned with EEG rows. | `scripts/*/extract.*`                      |
| Train | Fit a Lightning model with Hydra overrides.        | `uv run --no-sync python src/train.py ...` |
| Eval  | Evaluate a saved checkpoint.                       | `uv run --no-sync python src/eval.py ...`  |

**Default data root:** `./data/`
**Primary configs:** `configs/train.yaml`, `configs/eval.yaml`,
`configs/preprocess.yaml`, `configs/extract.yaml`

## Overview

NeuroCLIP is a Lightning + Hydra project for THINGS-EEG2 EEG preprocessing,
CLIP feature extraction, model training, and evaluation. It keeps the original
train/eval organization from `lightning-hydra-template` and adds reproducible
EEG-to-CLIP workflows driven by Hydra configs.

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

This is the shortest end-to-end path from a fresh checkout to THINGS-EEG2
training. Later sections expand each step with platform-specific commands and
Hydra overrides.

| Step | Goal                                        | Output                                     |
| ---- | ------------------------------------------- | ------------------------------------------ |
| 1-2  | Install uv and dependencies.                | A synced project environment.              |
| 3-5  | Download, configure, and unzip THINGS-EEG2. | Raw EEG and image folders under `./data/`. |
| 6    | Preprocess EEG.                             | `training.pt` and `test.pt` per subject.   |
| 7    | Extract CLIP features.                      | CLIP feature tensors aligned to EEG rows.  |
| 8-9  | Train and evaluate.                         | Checkpoints, metrics, and logs.            |
| 10   | Run checks.                                 | Test and hook results.                     |

### 1. Install uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is
not already available:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies

Use the CPU environment when GPU support is not needed:

```bash
uv sync --extra cpu
```

Use a CUDA extra for GPU training:

```bash
uv sync --extra cu126
```

Other configured CUDA extras are `cu128` and `cu129`.

### 3. Download Datasets

Download the three THINGS-EEG2 assets and keep the zip files separated by EEG
archives and image archives before unzipping.

| Dataset                     | Download Link                                                                            | Storage Path                |
| --------------------------- | ---------------------------------------------------------------------------------------- | --------------------------- |
| THINGS-EEG2 EEG data        | [Google Drive](https://drive.google.com/drive/folders/1KnOcV38RthPcpZR2vtiSm0jtZ6p63RNt) | `./data/thingseeg2-eegs/`   |
| THINGS-EEG2 training images | [OSF](https://osf.io/y63gw/files/3v527)                                                  | `./data/thingseeg2-images/` |
| THINGS-EEG2 test images     | [OSF](https://osf.io/y63gw/files/znu7b)                                                  | `./data/thingseeg2-images/` |

After download, keep the zip files grouped as EEG zips and image zips:

```text
<eeg_zip_dir>/
  sub-01.zip
  sub-02.zip
  ...
<image_zip_dir>/
  training_images.zip
  test_images.zip
```

### 4. Configure Paths

> Recommended: edit `configs/paths/default.yaml` once, then keep those paths
> for the full `unzip -> preprocess -> extract -> train -> eval` flow.

The default THINGS-EEG2 paths are:

| Key                                  | Default                           |
| ------------------------------------ | --------------------------------- |
| `paths.thingseeg2_raw_dir`           | `./data/thingseeg2-eegs`          |
| `paths.thingseeg2_img_dir`           | `./data/thingseeg2-images`        |
| `paths.thingseeg2_preprocessed_dir`  | `./data/thingseeg2-eeg2-250hz`    |
| `paths.thingseeg2_clip_features_dir` | `./data/thingseeg2-clip-features` |

You can also pass `paths.*` overrides on the command line. If you do, reuse the
same overrides in every later step.

### 5. Unzip Datasets

Run the unzip script for your platform. The first two arguments are optional
source directories for downloaded EEG zips and image zips.

| Platform | Command                                               |
| -------- | ----------------------------------------------------- |
| Windows  | `.\scripts\windows\unzip.ps1 D:\zips\eeg D:\zips\img` |
| Linux    | `bash scripts/linux/unzip.sh /zips/eeg /zips/img`     |
| macOS    | `bash scripts/macos/unzip.sh /zips/eeg /zips/img`     |

### 6. Preprocess EEG

Use the platform wrapper to preprocess one or more THINGS-EEG2 subjects. The
first two arguments are the inclusive subject range. Any remaining arguments are
forwarded to `src/preprocess.py` as Hydra overrides.

| Platform | Subjects 1-10                           |
| -------- | --------------------------------------- |
| Windows  | `.\scripts\windows\preprocess.ps1 1 10` |
| Linux    | `bash scripts/linux/preprocess.sh 1 10` |
| macOS    | `bash scripts/macos/preprocess.sh 1 10` |

Pass preprocessing options after the subject range:

```bash
bash scripts/linux/preprocess.sh 1 10 preprocess.sfreq=250 preprocess.mvnn_dim=null
```

For a single subject, the direct Python entrypoint is also available:

```bash
uv run --no-sync python src/preprocess.py preprocess.subject_id=1
```

### 7. Extract CLIP Features

Use the platform wrapper to extract CLIP image/text features from the
preprocessed THINGS-EEG2 tensors. Extraction scripts do not loop over subject
ranges; all arguments are forwarded to `src/extract.py` as Hydra overrides.

| Platform | Command                         |
| -------- | ------------------------------- |
| Windows  | `.\scripts\windows\extract.ps1` |
| Linux    | `bash scripts/linux/extract.sh` |
| macOS    | `bash scripts/macos/extract.sh` |

Pass extraction options directly to the wrapper:

```bash
bash scripts/linux/extract.sh extract.reference_subject_id=1 extract.model_name=ViT-B-32 extract.feature_mode=pooled
```

The direct Python entrypoint is also available:

```bash
uv run --no-sync python src/extract.py
```

### 8. Train

Start with the THINGS-EEG2 NICE training config:

```bash
uv run --no-sync python src/train.py data=thingseeg2 model=nice trainer=gpu
```

Example with 5-fold validation, CLIP loss, longer training, and early stopping:

```bash
uv run --no-sync python src/train.py data=thingseeg2 data.k_fold=5 data.fold_idx=1 model=nice model.loss_type=cliploss trainer=gpu trainer.max_epochs=100 callbacks.early_stopping.patience=20
```

### 9. Evaluate

Evaluate with the same data/model overrides used by the checkpoint:

```bash
uv run --no-sync python src/eval.py data=thingseeg2 data.k_fold=5 data.fold_idx=1 model=nice model.loss_type=cliploss trainer=gpu ckpt_path="/path/to/checkpoint.ckpt"
```

### 10. Run Checks

```bash
uv run --no-sync pytest
```

## Detailed Run Guide

This section expands the Quickstart into the full operating guide. The sequence
is the same:

```text
configure paths -> unzip datasets -> preprocess EEG -> extract CLIP features -> train -> evaluate
```

Use it when you need platform-specific scripts, custom paths, or Hydra override
details.

### Step 4 Details: Configure Paths

Shared paths live in `configs/paths/default.yaml`. For a stable local setup,
edit this file once and keep those values for the complete
`unzip -> preprocess -> extract -> train -> eval` flow.

CLI `paths.*` overrides are useful for one-off runs, but they are not persisted.
If you override a path in one step, pass the same relevant override again in
every later step that reads that location. Otherwise Hydra falls back to
`configs/paths/default.yaml`.

| Key                                  | Default                                      | Flow role                                                                                                                                           |
| ------------------------------------ | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `paths.thingseeg2_raw_dir`           | `${paths.data_dir}/thingseeg2-eegs`          | **Unzip** writes raw EEG folders here. **Preprocess** reads raw EEG from here through `preprocess.raw_data_dir`.                                    |
| `paths.thingseeg2_img_dir`           | `${paths.data_dir}/thingseeg2-images`        | **Unzip** writes image folders here. **Preprocess** reads image metadata from here through `preprocess.img_data_dir`.                               |
| `paths.thingseeg2_preprocessed_dir`  | `${paths.data_dir}/thingseeg2-eeg2-250hz`    | **Preprocess** writes `sub-XX/training.pt` and `sub-XX/test.pt` here. **Extract**, **Train**, and **Eval** read preprocessed EEG tensors from here. |
| `paths.thingseeg2_clip_features_dir` | `${paths.data_dir}/thingseeg2-clip-features` | **Extract** writes CLIP feature files here under model-specific subdirectories. **Train** and **Eval** read offline CLIP features from here.        |
| `paths.clip_model_cache_dir`         | `${paths.data_dir}/clip-cache`               | **Extract** passes this to Hugging Face `from_pretrained(..., cache_dir=...)` for CLIP model, processor, and tokenizer cache.                       |

There are two different kinds of paths during unzip:

| Path kind       | How it is passed                                                | Example                 |
| --------------- | --------------------------------------------------------------- | ----------------------- |
| Zip source dirs | Positional arguments to `scripts/*/unzip.*`                     | `/downloads/eeg-zips`   |
| Output dirs     | Hydra `paths.thingseeg2_raw_dir` and `paths.thingseeg2_img_dir` | `/data/thingseeg2-eegs` |

When using CLI overrides instead of editing `configs/paths/default.yaml`, repeat
the same relevant `paths.*` overrides across the pipeline:

```bash
# INCORRECT (Preprocess will read the default raw EEG path)
bash scripts/linux/unzip.sh /downloads/eeg /downloads/img paths.thingseeg2_raw_dir=/custom/eeg
uv run --no-sync python src/preprocess.py preprocess.subject_id=1

# CORRECT (Path overrides are repeated where the path is read)
# 1. Unzip
bash scripts/linux/unzip.sh /zips/eeg /zips/img paths.thingseeg2_raw_dir=/custom/eeg paths.thingseeg2_img_dir=/custom/img
# 2. Preprocess
bash scripts/linux/preprocess.sh 1 10 paths.thingseeg2_raw_dir=/custom/eeg paths.thingseeg2_img_dir=/custom/img paths.thingseeg2_preprocessed_dir=/custom/prep
# 3. Extract
bash scripts/linux/extract.sh paths.thingseeg2_preprocessed_dir=/custom/prep paths.thingseeg2_clip_features_dir=/custom/clip
```

### Step 5 Details: Unzip Datasets

Quickstart step 5 turns downloaded zip files into raw EEG and image folders:

```text
downloaded zip files -> raw EEG directory + image directory
```

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

| Platform | Default config                | Custom source and target example                                                                                                    |
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

### Step 6 Details: Preprocess EEG

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
| macOS    | `bash scripts/macos/preprocess.sh` | `bash scripts/macos/preprocess.sh 1 10 paths.thingseeg2_raw_dir=/output/eeg paths.thingseeg2_preprocessed_dir=/output/prep`     | `bash scripts/macos/preprocess.sh 1 10 preprocess.sfreq=250` |

The first two script arguments are the start and end subject IDs. Remaining
arguments are passed through as Hydra overrides.

### Step 7 Details: Extract CLIP Features

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
| macOS    | `bash scripts/macos/extract.sh` | `bash scripts/macos/extract.sh paths.thingseeg2_preprocessed_dir=/output/prep paths.thingseeg2_clip_features_dir=/output/clip`     | `bash scripts/macos/extract.sh extract.reference_subject_id=1` |

Extraction scripts do not loop over subject ranges. All arguments are passed
through as Hydra overrides.

### Step 8 Details: Train

Main entrypoint:

```bash
uv run --no-sync python src/train.py
```

Train NICE on THINGS-EEG2 with a 5-fold validation split:

```bash
uv run --no-sync python src/train.py data=thingseeg2 data.k_fold=5 data.fold_idx=1 model=nice model.loss_type=cliploss trainer=gpu trainer.max_epochs=100 callbacks.early_stopping.patience=20
```

Common examples:

```bash
uv run --no-sync python src/train.py trainer=cpu
uv run --no-sync python src/train.py trainer=gpu
uv run --no-sync python src/train.py trainer=ddp trainer.devices=4
uv run --no-sync python src/train.py logger=tensorboard
uv run --no-sync python src/train.py experiment=example
uv run --no-sync python src/train.py data=thingseeg2 model=nice
uv run --no-sync python src/train.py trainer.max_epochs=20
uv run --no-sync python src/train.py ckpt_path="/path/to/checkpoint.ckpt"
```

The default training config is `configs/train.yaml`. Hydra overrides can select
config groups, for example `data=thingseeg2`, `model=nice`, `trainer=gpu`,
`logger=tensorboard`, or `experiment=example`. Values inside the selected
configs can be overridden with dotted keys.

Common training overrides:

| Override                                | Purpose                                                             | Examples                                                                         |
| --------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `data`                                  | Select the datamodule config group.                                 | `data=mnist`, `data=thingseeg2`                                                  |
| `model`                                 | Select the Lightning module config group.                           | `model=mnist`, `model=nice`                                                      |
| `trainer`                               | Select the Lightning trainer preset.                                | `trainer=cpu`, `trainer=gpu`, `trainer=ddp`                                      |
| `logger`                                | Select a logger config group.                                       | `logger=tensorboard`, `logger=wandb`, `logger=csv`                               |
| `experiment`                            | Load an experiment config that overrides multiple groups.           | `experiment=example`                                                             |
| `data.subjects`                         | Select THINGS-EEG2 subjects.                                        | `data.subjects=[sub-01]`, `data.subjects=[sub-01,sub-02]`                        |
| `data.experiment_setting`               | Choose intra-subject or cross-subject splitting.                    | `data.experiment_setting=intra-subject`, `data.experiment_setting=cross-subject` |
| `data.train_val_split`                  | Set train/validation split when K-Fold is disabled.                 | `data.train_val_split=[0.9,0.1]`                                                 |
| `data.k_fold`                           | Enable deterministic K-Fold validation over the training partition. | `data.k_fold=5`                                                                  |
| `data.fold_idx`                         | Select the active fold. It is zero-based.                           | `data.fold_idx=0`, `data.fold_idx=1`                                             |
| `data.train_batch_size`                 | Set global training batch size.                                     | `data.train_batch_size=128`                                                      |
| `data.val_batch_size`                   | Set validation batch size.                                          | `data.val_batch_size=200`                                                        |
| `data.test_batch_size`                  | Set test batch size.                                                | `data.test_batch_size=200`                                                       |
| `data.num_workers`                      | Set dataloader worker count.                                        | `data.num_workers=4`                                                             |
| `data.average_reps`                     | Average repeated EEG trials before training.                        | `data.average_reps=true`, `data.average_reps=false`                              |
| `data.selected_channels`                | Override the EEG channel list used by THINGS-EEG2.                  | `data.selected_channels=[P7,P5,P3]`                                              |
| `model.loss_type`                       | Select CLIP alignment loss for NICE.                                | `model.loss_type=cliploss`, `model.loss_type=sigliploss`                         |
| `model.retrieval_k_list`                | Set test retrieval k values.                                        | `model.retrieval_k_list=[2,4,10,200]`                                            |
| `model.optimizer.lr`                    | Set optimizer learning rate.                                        | `model.optimizer.lr=0.0001`                                                      |
| `model.optimizer.weight_decay`          | Set optimizer weight decay.                                         | `model.optimizer.weight_decay=0.0001`                                            |
| `model.eegnet.*`                        | Tune NICE EEG network parameters.                                   | `model.eegnet.emb_size=40`, `model.eegnet.p=0.5`                                 |
| `trainer.max_epochs`                    | Set maximum training epochs.                                        | `trainer.max_epochs=100`                                                         |
| `trainer.devices`                       | Set number of devices.                                              | `trainer.devices=1`, `trainer.devices=4`                                         |
| `trainer.precision`                     | Enable mixed precision when supported.                              | `trainer.precision=16`                                                           |
| `trainer.check_val_every_n_epoch`       | Set validation frequency in epochs.                                 | `trainer.check_val_every_n_epoch=1`                                              |
| `callbacks.early_stopping.patience`     | Set early stopping patience.                                        | `callbacks.early_stopping.patience=20`                                           |
| `callbacks.early_stopping.monitor`      | Set early stopping metric.                                          | `callbacks.early_stopping.monitor=val/top1_acc`                                  |
| `callbacks.model_checkpoint.monitor`    | Set checkpoint selection metric.                                    | `callbacks.model_checkpoint.monitor=val/top1_acc`                                |
| `callbacks.model_checkpoint.save_top_k` | Keep more best checkpoints.                                         | `callbacks.model_checkpoint.save_top_k=3`                                        |
| `paths.thingseeg2_preprocessed_dir`     | Point training at preprocessed EEG tensors.                         | `paths.thingseeg2_preprocessed_dir=/data/thingseeg2-preprocessed`                |
| `paths.thingseeg2_clip_features_dir`    | Point training at extracted CLIP features.                          | `paths.thingseeg2_clip_features_dir=/data/thingseeg2-clip-features`              |
| `ckpt_path`                             | Resume from a checkpoint.                                           | `ckpt_path=/path/to/last.ckpt`                                                   |
| `seed`                                  | Set RNG seed.                                                       | `seed=42`                                                                        |
| `train`                                 | Skip fitting when only test/eval behavior is needed.                | `train=false`                                                                    |
| `test`                                  | Enable or disable the final test pass after training.               | `test=true`, `test=false`                                                        |

When using `data.k_fold`, `data.fold_idx` must be in `[0, data.k_fold - 1]`.
For distributed training, keep batch sizes divisible by the total number of
devices.

### Step 9 Details: Evaluate

Main entrypoint:

```bash
uv run --no-sync python src/eval.py ckpt_path="/path/to/checkpoint.ckpt"
```

Evaluate a NICE checkpoint on THINGS-EEG2:

```bash
uv run --no-sync python src/eval.py data=thingseeg2 data.k_fold=5 data.fold_idx=1 model=nice model.loss_type=cliploss trainer=gpu ckpt_path="/path/to/checkpoint.ckpt"
```

Common examples:

```bash
uv run --no-sync python src/eval.py trainer=cpu ckpt_path="/path/to/checkpoint.ckpt"
uv run --no-sync python src/eval.py trainer=gpu ckpt_path="/path/to/checkpoint.ckpt"
uv run --no-sync python src/eval.py data=thingseeg2 model=nice ckpt_path="/path/to/checkpoint.ckpt"
uv run --no-sync python src/eval.py logger=tensorboard ckpt_path="/path/to/checkpoint.ckpt"
```

The default evaluation config is `configs/eval.yaml`. A checkpoint path is
required, and the `data` and `model` overrides should match the checkpoint that
is being evaluated.

Common evaluation overrides:

| Override                             | Purpose                                            | Examples                                                                         |
| ------------------------------------ | -------------------------------------------------- | -------------------------------------------------------------------------------- |
| `ckpt_path`                          | Load model weights for evaluation. Required.       | `ckpt_path=/path/to/best.ckpt`, `ckpt_path=/path/to/last.ckpt`                   |
| `data`                               | Select the datamodule config group.                | `data=mnist`, `data=thingseeg2`                                                  |
| `model`                              | Select the Lightning module config group.          | `model=mnist`, `model=nice`                                                      |
| `trainer`                            | Select the Lightning trainer preset.               | `trainer=cpu`, `trainer=gpu`, `trainer=ddp`                                      |
| `logger`                             | Select a logger config group.                      | `logger=tensorboard`, `logger=wandb`, `logger=csv`                               |
| `data.subjects`                      | Select THINGS-EEG2 test subjects.                  | `data.subjects=[sub-01]`, `data.subjects=[sub-01,sub-02]`                        |
| `data.experiment_setting`            | Choose intra-subject or cross-subject evaluation.  | `data.experiment_setting=intra-subject`, `data.experiment_setting=cross-subject` |
| `data.k_fold`                        | Recreate the validation fold split when needed.    | `data.k_fold=5`                                                                  |
| `data.fold_idx`                      | Select the same validation fold used in training.  | `data.fold_idx=0`, `data.fold_idx=1`                                             |
| `data.test_batch_size`               | Set test batch size.                               | `data.test_batch_size=200`                                                       |
| `data.num_workers`                   | Set dataloader worker count.                       | `data.num_workers=4`                                                             |
| `data.average_reps`                  | Average repeated EEG trials before evaluation.     | `data.average_reps=true`, `data.average_reps=false`                              |
| `data.selected_channels`             | Override the EEG channel list used by THINGS-EEG2. | `data.selected_channels=[P7,P5,P3]`                                              |
| `model.loss_type`                    | Match the loss config used by NICE checkpoint.     | `model.loss_type=cliploss`, `model.loss_type=sigliploss`                         |
| `model.retrieval_k_list`             | Set test retrieval k values.                       | `model.retrieval_k_list=[2,4,10,200]`                                            |
| `trainer.devices`                    | Set number of devices.                             | `trainer.devices=1`, `trainer.devices=4`                                         |
| `trainer.precision`                  | Enable mixed precision when supported.             | `trainer.precision=16`                                                           |
| `paths.thingseeg2_preprocessed_dir`  | Point eval at preprocessed EEG tensors.            | `paths.thingseeg2_preprocessed_dir=/data/thingseeg2-preprocessed`                |
| `paths.thingseeg2_clip_features_dir` | Point eval at extracted CLIP features.             | `paths.thingseeg2_clip_features_dir=/data/thingseeg2-clip-features`              |
| `seed`                               | Set RNG seed.                                      | `seed=42`                                                                        |

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
