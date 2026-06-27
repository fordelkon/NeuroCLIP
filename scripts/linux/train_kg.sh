#!/usr/bin/env bash

# Usage:
#   bash scripts/linux/train_kg.sh <subject_id> <model> <k_fold> <fold_idx> [hydra_overrides...]
#
# Description:
#   Train model WITH knowledge graph smoothing.
#
# Arguments:
#   subject_id          Subject ID (e.g., sub-01, sub-02, ..., sub-10)
#   model               Model name (e.g., flatnet, nice, atms)
#   k_fold              Number of folds (default: 5)
#   fold_idx            Fold index for k-fold cross-validation (e.g., 0, 1, 2, 3, 4)
#   hydra_overrides     Optional Hydra overrides forwarded to src/train.py.
#
# Examples:
#   bash scripts/linux/train_kg.sh sub-01 flatnet 5 0
#   bash scripts/linux/train_kg.sh sub-02 nice 10 1
#   bash scripts/linux/train_kg.sh sub-03 atms 5 0 model.kgnet.alpha=0.5

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "Error: Missing required arguments"
  echo "Usage: bash scripts/linux/train_kg.sh <subject_id> <model> <k_fold> <fold_idx> [hydra_overrides...]"
  exit 1
fi

SUBJECT_ID=$1
MODEL=$2
K_FOLD=$3
FOLD_IDX=$4
shift 4

RUN_DIR="logs/train/runs/${MODEL}/${MODEL}_kg_k${K_FOLD}_fold${FOLD_IDX}_${SUBJECT_ID}"

uv run --no-sync python src/train.py \
  data=thingseeg2 \
  "data.subjects=[${SUBJECT_ID}]" \
  data.k_fold="${K_FOLD}" \
  data.fold_idx="${FOLD_IDX}" \
  model="${MODEL}" \
  model.loss_type=cliploss \
  model.enable_kg_smooth=true \
  trainer=cpu \
  trainer.max_epochs=200 \
  callbacks.early_stopping.patience=20 \
  callbacks.early_stopping.min_delta=0 \
  logger=csv \
  hydra.run.dir="${RUN_DIR}" \
  extract=thingseeg2 \
  build_kg=thingseeg2 \
  "$@"
