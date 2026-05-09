# AGENTS.md

Repository-specific instructions for Codex-style agents working in this project.

## General Workflow

- Read the relevant code and configuration before making assumptions.
- Keep edits scoped to the requested task and follow existing project patterns.
- Use `rg`/`rg --files` for search when available.
- Use `apply_patch` for manual file edits.
- Do not revert user changes or unrelated dirty worktree changes.
- Prefer English comments in configuration and code documentation unless the user asks otherwise.
- This project uses Lightning, Hydra, uv, and THINGS-EEG2 workflows. Prefer existing Hydra config groups, script wrappers, and utility modules over duplicating logic.
- Use Hydra command-line overrides for one-off runs, and keep machine-specific defaults in `configs/local/default.yaml` when needed.

## Project Workflow

At a glance, the main project flow is:

```text
uv sync -> unzip THINGS-EEG2 data -> preprocess EEG -> extract CLIP features -> train -> evaluate
```

Primary entrypoints and configs:

| Area        | Entry               | Config                       | Purpose                         |
| ----------- | ------------------- | ---------------------------- | ------------------------------- |
| Environment | `uv sync`           | `pyproject.toml`, `uv.lock`  | Reproducible dependency setup   |
| Preprocess  | `src/preprocess.py` | `configs/preprocess.yaml`    | Build THINGS-EEG2 `.pt` tensors |
| Extract     | `src/extract.py`    | `configs/extract.yaml`       | Build THINGS-EEG2 CLIP features |
| Train       | `src/train.py`      | `configs/train.yaml`         | Run Lightning training          |
| Evaluate    | `src/eval.py`       | `configs/eval.yaml`          | Evaluate a checkpoint           |
| Paths       | Hydra paths config  | `configs/paths/default.yaml` | Share data/output locations     |
| Quality     | `pytest`, hooks     | `pyproject.toml`, hooks      | Test, lint, and format          |

Use `uv run --no-sync ...` after the environment has been synced.

## THINGS-EEG2 Data Flow

- Data preparation order is `downloaded zip files -> unzip -> preprocess EEG -> extract CLIP features`.
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
uv run --no-sync python src/train.py
uv run --no-sync python src/eval.py ckpt_path="path/to/checkpoint.ckpt"
```

Platform script wrappers:

| Task       | Windows                            | Linux                              | macOS                              |
| ---------- | ---------------------------------- | ---------------------------------- | ---------------------------------- |
| Unzip      | `.\scripts\windows\unzip.ps1`      | `bash scripts/linux/unzip.sh`      | `bash scripts/macos/unzip.sh`      |
| Preprocess | `.\scripts\windows\preprocess.ps1` | `bash scripts/linux/preprocess.sh` | `bash scripts/macos/preprocess.sh` |
| Extract    | `.\scripts\windows\extract.ps1`    | `bash scripts/linux/extract.sh`    | `bash scripts/macos/extract.sh`    |

- Preprocess script wrappers take start and end subject IDs as the first two arguments; remaining arguments are Hydra overrides.
- Extraction script wrappers do not loop over subject ranges; all arguments are passed through as Hydra overrides.

## Verification

- Use `uv run --no-sync ...` for project commands.
- If `uv` cannot access the user cache because of sandbox permissions, rerun the same command with escalated approval.
- Before claiming work is complete, run the relevant tests or checks and report the exact result.
- For README-only changes, run:

```powershell
uv run --no-sync pre-commit run --files README.md
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

## THINGS-EEG2 Conventions

- `cross-subject` datamodule mode treats configured `subjects` as held-out test subjects.
- In `cross-subject`, train/validation use the remaining THINGS-EEG2 subjects.
- `intra-subject` mode uses configured `subjects` for train/validation/test.
- CLIP model name/id resolution should go through `src.utils.clip` instead of duplicating maps.
- Dataset loading uses preprocessed EEG and extracted CLIP features; z-score and MVNN normalization belong in preprocessing, not dataset loading.
- Saved EEG tensors keep the post-stimulus `[0, 1)` second window.
- Training and test partitions are normalized independently per subject and session.
- For MVNN modes, train and test data use whitening statistics estimated from the training partition.
- For `mvnn_dim=null`, z-score mean and standard deviation are estimated from the training partition and applied to both partitions.

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
