#!/usr/bin/env bash

set -euo pipefail

uv run --no-sync python src/extract.py "$@"
