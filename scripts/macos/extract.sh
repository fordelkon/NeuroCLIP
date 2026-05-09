#!/usr/bin/env bash

# Usage:
#   bash scripts/macos/extract.sh [hydra_overrides...]
#
# Arguments:
#   hydra_overrides     Optional Hydra overrides forwarded to src/extract.py.
#
# Examples:
#   bash scripts/macos/extract.sh
#   bash scripts/macos/extract.sh extract.reference_subject_id=1
#   bash scripts/macos/extract.sh extract.model_name=ViT-B-32 extract.feature_mode=pooled
#   bash scripts/macos/extract.sh paths.thingseeg2_preprocessed_dir=/data/prep paths.thingseeg2_clip_features_dir=/data/clip

set -euo pipefail

uv run --no-sync python src/extract.py "$@"
