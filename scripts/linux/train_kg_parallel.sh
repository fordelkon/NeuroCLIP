#!/usr/bin/env bash

# Usage:
#   bash scripts/linux/train_kg_parallel.sh <subject_id> <model> <num_folds> [hydra_overrides...]
#
# Description:
#   Run parallel KG training jobs for all folds (0 to num_folds-1) in background.
#
# Arguments:
#   subject_id          Subject ID (e.g., sub-01, sub-02, ..., sub-10)
#   model               Model name (e.g., flatnet, nice, atms)
#   num_folds           Number of folds for k-fold cross-validation
#   hydra_overrides     Optional Hydra overrides forwarded to each training job
#
# Examples:
#   bash scripts/linux/train_kg_parallel.sh sub-10 flatnet 5
#   bash scripts/linux/train_kg_parallel.sh sub-01 nice 10 model.kgnet.alpha=0.5

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Error: Missing required arguments"
  echo "Usage: bash scripts/linux/train_kg_parallel.sh <subject_id> <model> <num_folds> [hydra_overrides...]"
  exit 1
fi

SUBJECT_ID=$1
MODEL=$2
NUM_FOLDS=$3
shift 3

echo "Starting $NUM_FOLDS background KG training jobs for $SUBJECT_ID with $MODEL..."

for ((fold=0; fold<NUM_FOLDS; fold++)); do
  bash scripts/linux/train_kg.sh "$SUBJECT_ID" "$MODEL" "$NUM_FOLDS" "$fold" "$@" &
  echo "Started fold $fold in background (PID: $!)"
done

echo ""
echo "All jobs started. Waiting for completion..."
echo "Check logs at: logs/train/runs/$MODEL/"

wait

echo ""
echo "All jobs finished."
