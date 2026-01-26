#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

STAGE1_DIR="data/camcan/small/1to50hz_cont"
if find "$STAGE1_DIR" -type f -name "*_preproc-raw.fif" -print -quit | grep -q .; then
  echo "Skipping CamCAN stage 1; existing outputs found in $STAGE1_DIR."
else
  conda run -n brain-gen python preprocess.py \
    --dataset camcan \
    --stage stage_1 \
    --args examples/flatgpt/01_preprocess/configs/camcan_stage1_small.yaml
fi

conda run -n brain-gen python preprocess.py \
  --dataset camcan \
  --stage stage_2 \
  --args examples/flatgpt/01_preprocess/configs/camcan_stage2_small.yaml
