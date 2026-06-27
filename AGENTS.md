# AGENTS.md

Repository-specific instructions for Codex-style agents working in NeuroCLIP.

## General Workflow

- Read the relevant code and configuration before making assumptions.
- Keep edits scoped to the requested task and follow existing project patterns.
- Use `rg`/`rg --files` for search when available.
- Use `apply_patch` for manual file edits.
- Do not revert user changes or unrelated dirty worktree changes.
- Prefer English comments in configuration and code documentation unless the user asks otherwise.
- This project uses Lightning, Hydra, uv, THINGS-EEG2, and EEG-to-CLIP training workflows. Prefer existing Hydra config groups, script wrappers, model components, and utility modules over duplicating logic.
- Use Hydra command-line overrides for one-off runs, and keep machine-specific defaults in `configs/local/default.yaml` when needed.

## Project Workflow

At a glance, the main project flow is:

```text
uv sync -> unzip datasets -> preprocess EEG -> extract CLIP features -> train -> evaluate
```

Primary entrypoints and configs:

| Area        | Entry                  | Config                                      | Purpose                            |
| ----------- | ---------------------- | ------------------------------------------- | ---------------------------------- |
| Environment | `uv sync`              | `pyproject.toml`, `uv.lock`                 | Reproducible dependency management |
| Preprocess  | `src/preprocess.py`    | `configs/preprocess.yaml`                   | Build THINGS-EEG2 `.pt` tensors    |
| Extract     | `src/extract.py`       | `configs/extract.yaml`                      | Build THINGS-EEG2 CLIP features    |
| Train       | `src/train.py`         | `configs/train.yaml`                        | Run Lightning training             |
| Evaluate    | `src/eval.py`          | `configs/eval.yaml`                         | Evaluate a checkpoint              |
| Paths       | Hydra composition      | `configs/paths/default.yaml`                | Share dataset and output locations |
| Quality     | `pre-commit`, `pytest` | `.pre-commit-config.yaml`, `pyproject.toml` | Lint, format, and test             |

Use `uv run --no-sync ...` after the environment has been synced.

Environment setup:

```powershell
uv sync --extra cpu
uv sync --extra cu126
```

Other configured CUDA extras are `cu128` and `cu129`.

## THINGS-EEG2 Data Flow

- Data preparation order is `downloaded zip files -> unzip -> preprocess EEG -> extract CLIP features`.
- Keep downloaded EEG zips and image zips separated before unzipping.
- Shared data paths live in `configs/paths/default.yaml`; prefer updating that file for persistent local path mappings.
- If using CLI `paths.*` overrides instead, pass the exact same relevant overrides through every later step. Hydra will otherwise fall back to default directories and scripts may fail to find generated data.
- Unzip targets are controlled by `paths.thingseeg2_raw_dir` and `paths.thingseeg2_img_dir`.
- Preprocessing reads raw EEG/images and writes preprocessed tensors under `paths.thingseeg2_preprocessed_dir`.
- Extraction reads preprocessed tensors and writes CLIP features under `paths.thingseeg2_clip_features_dir`.
- CLIP model cache location is controlled by `paths.clip_model_cache_dir`.

Common commands:

```powershell
uv run --no-sync python src/preprocess.py preprocess.subject_id=1
uv run --no-sync python src/extract.py
uv run --no-sync python src/train.py data=thingseeg2 model=nice trainer=gpu
uv run --no-sync python src/eval.py data=thingseeg2 model=nice ckpt_path="path/to/checkpoint.ckpt"
```

Platform script wrappers:

| Task       | Windows                            | Linux                              | macOS                              |
| ---------- | ---------------------------------- | ---------------------------------- | ---------------------------------- |
| Unzip      | `.\scripts\windows\unzip.ps1`      | `bash scripts/linux/unzip.sh`      | `bash scripts/macos/unzip.sh`      |
| Preprocess | `.\scripts\windows\preprocess.ps1` | `bash scripts/linux/preprocess.sh` | `bash scripts/macos/preprocess.sh` |
| Extract    | `.\scripts\windows\extract.ps1`    | `bash scripts/linux/extract.sh`    | `bash scripts/macos/extract.sh`    |

- Preprocess script wrappers take start and end subject IDs as the first two arguments; remaining arguments are Hydra overrides.
- Extraction script wrappers do not loop over subject ranges; all arguments are passed through as Hydra overrides.

## Training And Model Configs

- The default training config uses `data=mnist` and `model=mnist`.
- The default callback config is tuned for NICE/CLIP metrics and monitors `val/top1_acc`; MNIST runs usually need `callbacks.model_checkpoint.monitor=val/acc callbacks.early_stopping.monitor=val/acc`.
- THINGS-EEG2 CLIP training uses model configs under `configs/model/`, such as `nice.yaml`, `atms.yaml`, `flatnet.yaml`
- THINGS-EEG2 experiment entrypoints include `experiment=nice_experiment`, `experiment=atms_experiment`, `experiment=flatnet_experiment`
- `model=nice` uses `src.models.components.simple_nice.NICE`.
- `model=atms` uses `src.models.components.simple_atms.ATMS`.
- ATMS defaults to `model.eegnet.use_subject_embedding=false`, matching intra-subject runs. For cross-subject ATMS experiments, enable it explicitly with `model.eegnet.use_subject_embedding=true`.
- `src.models.clipv1_module.ClipV1LitModule` owns CLIP-style loss, retrieval metrics, and optimizer/scheduler wiring for NICE and ATMS.
- Dataset batches include `subject_id`; only subject-aware EEG nets should consume it.
- Hyperparameter sweeps live in `configs/hparams_search/` and should be paired with the matching experiment config, for example `-m hparams_search=nice_optuna experiment=nice_experiment`.
- Keep sweep parameters source-backed and shape-safe. Do not add NICE/ATMS `model.eegnet.emb_size`, `temporal_kernel`, `pool_kernel`, or `pool_stride` to a sweep unless `model.eegnet.emb_dim` is updated consistently. ATMS `model.eegnet.nhead` choices must divide `model.eegnet.sequence_length`.

Useful training examples:

```powershell
uv run --no-sync python src/train.py callbacks.model_checkpoint.monitor=val/acc callbacks.early_stopping.monitor=val/acc
uv run --no-sync python src/train.py data=thingseeg2 model=nice trainer=gpu
uv run --no-sync python src/train.py data=thingseeg2 model=atms trainer=gpu
uv run --no-sync python src/train.py data=thingseeg2 data.experiment_setting=cross-subject data.subjects=[sub-01] model=atms model.eegnet.use_subject_embedding=true trainer=gpu
uv run --no-sync python src/train.py data=thingseeg2 data.k_fold=5 data.fold_idx=1 model=nice model.loss_type=cliploss trainer=gpu trainer.max_epochs=100 callbacks.early_stopping.patience=20
uv run --no-sync python src/train.py -m hparams_search=nice_optuna experiment=nice_experiment trainer=gpu
uv run --no-sync python src/train.py -m hparams_search=atms_optuna experiment=atms_experiment trainer=gpu
```

Useful evaluation examples:

```powershell
uv run --no-sync python src/eval.py ckpt_path="path/to/checkpoint.ckpt"
uv run --no-sync python src/eval.py data=thingseeg2 model=nice ckpt_path="path/to/checkpoint.ckpt"
uv run --no-sync python src/eval.py data=thingseeg2 model=atms ckpt_path="path/to/checkpoint.ckpt"
```

## Adding A New EEG Model

Start by identifying the integration boundary. Some models are only a new EEG
encoder component; others need a new Hydra config, Lightning module behavior,
loss wiring, dataset fields, or evaluation logic. Keep the smallest boundary
that satisfies the model design, and reuse existing project contracts when they
fit.

1. Read the current model path first.

   - Check related configs under `configs/model/`.
   - Check existing components under `src/models/components/`.
   - Check the owning Lightning module, usually `src/models/clipv1_module.py` for EEG-to-CLIP models.
   - Check tests that cover the closest existing model.

2. Choose the model contract.

   - For EEG-to-CLIP encoders used by `ClipV1LitModule`, return a dictionary with `eeg_clip` shaped `[batch, proj_dim]`.
   - For models with different outputs, losses, or evaluation semantics, prefer a new Lightning module or a clearly named extension instead of forcing the model into the NICE/ATMS contract.
   - Keep constructor parameters Hydra-friendly: primitives, lists, dictionaries, or optional values.
   - Keep comments and docstrings in English.

3. Add the component, model config, experiment config, and sweep config.

   - Put reusable neural network pieces under `src/models/components/`.
   - If the model is a standalone encoder with a narrow project-specific wrapper, a direct module under `src/models/` is also acceptable; follow the closest existing pattern.
   - Add `configs/model/<name>.yaml` when the model should be selectable with `model=<name>`.
   - For CLIP alignment models, `configs/model/nice.yaml` and `configs/model/atms.yaml` are good references for wrapper structure, optimizer/scheduler fields, retrieval settings, modality, loss, alpha, and compile.
   - Use a different top-level `_target_` only when the model needs a different Lightning module.
   - Add `configs/experiment/<name>_experiment.yaml` when the model should have a reusable THINGS-EEG2 experiment entrypoint. Override `/data: thingseeg2`, `/model: <name>`, `/callbacks: default`, and `/trainer: default`; set tags, logger group names, seed, and `optimized_metric`.
   - Add `configs/hparams_search/<name>_optuna.yaml` only for source-backed, shape-safe sweep parameters. Pair it with the matching experiment config, for example `-m hparams_search=<name>_optuna experiment=<name>_experiment`.
   - Avoid sweep parameters that require coupled derived dimensions unless the config updates every dependent value. Examples: hidden sizes must divide attention heads; patch or pooling choices that change flatten dimensions need corresponding projection dimensions.

4. Handle batch metadata explicitly.

   - Dataset batches include `subject_id` as a 1-based THINGS-EEG2 id.
   - If a model needs metadata such as subject ids, expose an explicit opt-in or clearly named model parameter.
   - Update batch routing only where the metadata is consumed, and keep existing models compatible.
   - Do not make the datamodule depend on a specific model strategy.

5. Add focused tests before behavior changes.

   - Add or extend component tests for tensor shapes, required inputs, and error cases.
   - Extend `tests/test_configs.py` for the new Hydra model config, experiment entrypoint, Optuna search space, CLIP projection dim resolver, and selected-channel resolver when relevant.
   - Extend Lightning module tests when routing, loss behavior, metrics, or compatibility changes.
   - Extend dataset/datamodule tests only when the model truly needs new data fields or split behavior.

6. Verify the smallest relevant surface.

```powershell
uv run --no-sync pytest tests/test_<model>.py tests/test_configs.py
uv run --no-sync pre-commit run --files configs/model/<name>.yaml configs/experiment/<name>_experiment.yaml configs/hparams_search/<name>_optuna.yaml src/models/components/<model>.py tests/test_<model>.py tests/test_configs.py
```

Adjust the file list to the actual files touched. If the model lives directly
under `src/models/`, use that path instead of `src/models/components/<model>.py`.
Include `tests/test_clipv1_module.py` only when the Lightning routing or loss
contract changed. For reviewer-friendly commits, keep tightly coupled
component, config, experiment, sweep, and test changes together when they form
one inseparable integration; split unrelated refactors, dataset changes, or
training workflow changes. Exclude unrelated dirty files.

## Verification

- Use `uv run --no-sync ...` for project commands.
- If `uv` cannot access the user cache because of sandbox permissions, rerun the same command with escalated approval.
- Before claiming work is complete, run the relevant tests or checks and report the exact result.
- For README-only changes, run:

```powershell
uv run --no-sync pre-commit run --files README.md
```

- For AGENTS-only changes, run:

```powershell
uv run --no-sync pre-commit run --files AGENTS.md
```

- For broad changes, run:

```powershell
uv run --no-sync pre-commit run --all-files
```

- For preprocessing changes, run:

```powershell
uv run --no-sync pytest tests/test_preprocess.py
```

- For extraction changes, run:

```powershell
uv run --no-sync pytest tests/test_extract.py
```

- For THINGS-EEG2 dataset/datamodule changes, also run the focused tests when relevant:

```powershell
uv run --no-sync pytest tests/test_thingseeg2_datamodule.py tests/test_thingseeg2_dataset.py tests/test_clip_utils.py tests/test_configs.py
```

- For training workflow changes, run:

```powershell
uv run --no-sync pytest tests/test_train.py
```

- For model config changes, run:

```powershell
uv run --no-sync pytest tests/test_configs.py
```

- For NICE/ATMS model component changes, run the focused model tests when relevant:

```powershell
uv run --no-sync pytest tests/test_simple_nice.py tests/test_simple_atms.py tests/test_clipv1_module.py tests/test_configs.py
```

## THINGS-EEG2 Conventions

- `cross-subject` datamodule mode treats configured `subjects` as held-out test subjects.
- In `cross-subject`, train/validation use the remaining THINGS-EEG2 subjects.
- `intra-subject` mode uses configured `subjects` for train/validation/test.
- For cross-subject ATMS, enable subject embeddings with `model.eegnet.use_subject_embedding=true`.
- THINGS-EEG2 subject IDs are 1-based in dataset samples.
- CLIP model name/id resolution should go through `src.utils.clip` instead of duplicating maps.
- Dataset loading uses preprocessed EEG and extracted CLIP features; z-score and MVNN normalization belong in preprocessing, not dataset loading.
- CLIP feature rows align with preprocessed EEG image order: `image_features[i]` aligns with `eeg[i, rep]`.
- Saved EEG tensors keep the post-stimulus `[0, 1)` second window.
- Training and test partitions are normalized independently per subject and session.
- For MVNN modes, train and test data use whitening statistics estimated from the training partition.
- For `mvnn_dim=null`, z-score mean and standard deviation are estimated from the training partition and applied to both partitions.

## Project Layout Notes

- THINGS-EEG2 data config: `configs/data/thingseeg2.yaml`.
- NICE model config: `configs/model/nice.yaml`.
- ATMS model config: `configs/model/atms.yaml`.
- THINGS-EEG2 dataset and datamodule: `src/data/components/thingseeg2_dataset.py`, `src/data/thingseeg2_datamodule.py`.
- EEG-to-CLIP Lightning module: `src/models/clipv1_module.py`.
- NICE/ATMS components: `src/models/components/simple_nice.py`, `src/models/components/simple_atms.py`.

## Git Rules

- All `git add` and `git commit` operations require explicit user approval.
- When the user references `$git-commit-guideline`, read and follow the repository-local skill file when present:

```text
.agents/skills/git-commit-guideline/SKILL.md
```

If the file is not present in a different checkout, use the available `git-commit-guideline` skill content from the current agent environment.

- Keep commits reviewer-friendly:
  - one logical change per commit,
  - concise imperative subject,
  - body for non-obvious behavior, risk, or workflow changes.
- Do not rewrite shared history or use destructive git commands unless the user explicitly asks.

## Merge Workflow

- When merging a feature branch into `dev`, use a no-fast-forward merge unless the user says otherwise:

```powershell
git checkout dev
git merge --no-ff <branch>
```

- Prefer detailed merge commit messages that match the existing style:
  - subject: `merge: integrate <workflow>`,
  - first paragraph: what is being merged into `dev`,
  - following paragraphs: major features, behavior changes, tests/config/docs.
- Run relevant tests and `pre-commit` before and after the merge.
- If `pre-commit` changes files after a merge:
  - inspect the diff,
  - ask before `git add`,
  - amend the merge commit only if there is an actual staged content change.
