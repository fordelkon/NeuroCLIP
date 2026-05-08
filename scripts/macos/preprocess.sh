#!/usr/bin/env bash

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
