#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

conda run -n brain-gen python evals.py \
  --args examples/flatgpt/05_full_paper/configs/flatgpt/flat/test_aud.yaml
