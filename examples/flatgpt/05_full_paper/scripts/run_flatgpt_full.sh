#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

conda run -n brain-gen python run.py \
  --mode train \
  --args examples/flatgpt/05_full_paper/configs/flatgpt/flat/train_ds.yaml
