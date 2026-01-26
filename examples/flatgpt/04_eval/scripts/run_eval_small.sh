#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

conda run -n brain-gen python evals.py \
  --args examples/flatgpt/04_eval/configs/flatgpt_eval_small.yaml
