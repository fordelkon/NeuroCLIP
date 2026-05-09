#!/usr/bin/env bash

# Usage:
#   bash scripts/linux/preprocess.sh [start_subject] [end_subject] [hydra_overrides...]
#
# Arguments:
#   start_subject       First THINGS-EEG2 subject id to preprocess. Defaults to 1.
#   end_subject         Last THINGS-EEG2 subject id to preprocess. Defaults to 10.
#   hydra_overrides     Optional Hydra overrides forwarded to src/preprocess.py.
#
# Examples:
#   bash scripts/linux/preprocess.sh
#   bash scripts/linux/preprocess.sh 1 10
#   bash scripts/linux/preprocess.sh 1 10 preprocess.sfreq=250
#   bash scripts/linux/preprocess.sh 1 10 paths.thingseeg2_raw_dir=/data/eeg paths.thingseeg2_preprocessed_dir=/data/prep

set -euo pipefail

start_subject="${1:-1}"
end_subject="${2:-10}"

if [[ $# -gt 0 ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  shift
fi

hydra_overrides=("$@")

for subject_id in $(seq "${start_subject}" "${end_subject}"); do
  uv run --no-sync python src/preprocess.py \
    "preprocess.subject_id=${subject_id}" \
    "${hydra_overrides[@]}"
done
