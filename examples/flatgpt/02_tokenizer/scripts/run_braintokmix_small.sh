#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

conda run -n brain-gen python run.py \
  --mode train \
  --args examples/flatgpt/02_tokenizer/configs/braintokmix_train_small.yaml
