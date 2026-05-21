#!/usr/bin/env bash

# Usage:
#   bash scripts/linux/build_kg.sh [hydra_overrides...]
#
# Arguments:
#   hydra_overrides     Optional Hydra overrides forwarded to src/build_kg.py.
#
# Examples:
#   bash scripts/linux/build_kg.sh
#   bash scripts/linux/build_kg.sh build_kg.k=20
#   bash scripts/linux/build_kg.sh build_kg.feature_mode=no_blur
#   bash scripts/linux/build_kg.sh build_kg.k=15 build_kg.partition=training

set -euo pipefail

uv run --no-sync python src/build_kg.py "$@"
