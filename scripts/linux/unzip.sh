#!/usr/bin/env bash

# Expected downloaded zip layout:
#   <eeg_zip_dir>/
#     sub-01.zip
#     sub-02.zip
#     ...
#   <image_zip_dir>/
#     training_images.zip
#     test_images.zip
#
# Extraction target layout is resolved from Hydra paths:
#   paths.thingseeg2_raw_dir/
#     sub-01/
#     sub-02/
#     ...
#   paths.thingseeg2_img_dir/
#     training_images/
#     test_images/
#
# Usage:
#   bash scripts/linux/unzip.sh [eeg_zip_dir] [image_zip_dir] [hydra_overrides...]
#
# Examples:
#   bash scripts/linux/unzip.sh
#   bash scripts/linux/unzip.sh /zips/eeg /zips/img
#   bash scripts/linux/unzip.sh /zips/eeg /zips/img paths.thingseeg2_raw_dir=/data/eeg paths.thingseeg2_img_dir=/data/img

set -euo pipefail

is_hydra_override() {
  [[ "$1" == *"="* || "$1" == +* || "$1" == ~* ]]
}

eeg_zip_dir=""
image_zip_dir=""

if [[ $# -gt 0 ]] && ! is_hydra_override "$1"; then
  eeg_zip_dir="$1"
  shift
fi

if [[ $# -gt 0 ]] && ! is_hydra_override "$1"; then
  image_zip_dir="$1"
  shift
fi

hydra_overrides=("$@")

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  export PROJECT_ROOT
  PROJECT_ROOT="$(pwd)"
fi

resolve_dataset_dirs() {
  uv run --no-sync python -c '
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

config_dir = str(Path("configs").resolve())
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="preprocess", overrides=list(sys.argv[1:]))

print(cfg.paths.thingseeg2_raw_dir)
print(cfg.paths.thingseeg2_img_dir)
' "$@"
}

unzip_archives() {
  local source_dir="$1"
  local target_dir="$2"
  local label="$3"
  local found=0

  if [[ ! -d "${source_dir}" ]]; then
    echo "Skipping ${label}: source directory not found: ${source_dir}"
    return
  fi

  mkdir -p "${target_dir}"

  shopt -s nullglob
  local archive
  for archive in "${source_dir}"/*.zip; do
    found=1
    echo "Extracting ${archive} -> ${target_dir}"
    unzip -q "${archive}" -d "${target_dir}"
    rm "${archive}"
  done
  shopt -u nullglob

  if [[ "${found}" -eq 0 ]]; then
    echo "No ${label} zip archives found in ${source_dir}"
  fi
}

dataset_dirs="$(resolve_dataset_dirs "${hydra_overrides[@]}")"
eeg_target_dir="$(printf "%s\n" "${dataset_dirs}" | sed -n "1p")"
image_target_dir="$(printf "%s\n" "${dataset_dirs}" | sed -n "2p")"

if [[ -z "${eeg_zip_dir}" ]]; then
  eeg_zip_dir="${eeg_target_dir}"
fi

if [[ -z "${image_zip_dir}" ]]; then
  image_zip_dir="${image_target_dir}"
fi

unzip_archives "${eeg_zip_dir}" "${eeg_target_dir}" "EEG"
unzip_archives "${image_zip_dir}" "${image_target_dir}" "image"
