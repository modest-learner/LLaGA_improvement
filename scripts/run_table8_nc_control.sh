#!/bin/bash
set -euo pipefail

# Matched source graphs/nodes, HO/linear/SimTeG, seed42, one epoch, batch16.
# Changes source training task nd -> nc. Never loads the nd or official projector.
checkpoint="${NC_CKPT:-./checkpoints/table8/llaga-HO-arxiv-pubmed-nc-linear-simteg-lr2e-3-seed42-bs16}"
control_root="${CONTROL_ROOT:-./results/table8/HO_linear_simteg_nctrainseed42}"
log_dir="${CONTROL_LOG_DIR:-./logs/table8/nc_control_seed42}"
eval_gpu="${EVAL_GPU:-3}"
mkdir -p "$control_root/source" "$log_dir"
if [[ -e "$checkpoint" ]]; then
  echo "Checkpoint path already exists: $checkpoint. Use a new NC_CKPT; do not overwrite or silently retrain." >&2
  exit 1
fi

(
  unset CUDA_VISIBLE_DEVICES
  MODEL_BASE=./models/vicuna-7b-v1.5-16k \
  GPU_INCLUDE="${GPU_INCLUDE:-localhost:0,3}" \
  LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61006}" \
  LEARNING_RATE=2e-3 GRADIENT_ACCUMULATION_STEPS=2 OUTPUT_DIR="$checkpoint" \
  bash scripts/train_deepspeed.sh vicuna_4hop nc arxiv-pubmed 4 simteg --seed 42 \
    2>&1 | tee -a "$log_dir/train_nc.log"
)
test -s "$checkpoint/mm_projector.bin"
python - "$checkpoint" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
state = json.loads((root / 'trainer_state.json').read_text())
config = json.loads((root / 'config.json').read_text())
assert state['global_step'] == state['max_steps'] == 6423, state['global_step']
assert config['mm_projector_type'] == 'linear'
PY

for group in E_official E ET; do
  case "$group" in
    E_official) task=nc; prompt_style=official ;;
    E) task=nc; prompt_style=matched ;;
    ET) task=nctext; prompt_style=matched ;;
  esac
  CUDA_VISIBLE_DEVICES="$eval_gpu" \
  MODEL_BASE=./models/vicuna-7b-v1.5-16k MODEL_PATH="$checkpoint" \
  DATASET=cora TASK="$task" TEMPLATE=HO USE_HOP=4 EMB=simteg \
  INPUT_MODE=graph PROMPT_STYLE="$prompt_style" \
  OUTPUT_PATH="$control_root/source/cora_${group}_inferseed0.jsonl" \
  bash scripts/eval.sh --seed 0 --fail_on_error \
    2>&1 | tee -a "$log_dir/cora_${group}_sampled_seed0.log"
done

# T has no projector: reuse the identical frozen-Vicuna text baseline, not a new training replicate.
python - "$control_root/source/cora_T_inferseed0.jsonl" <<'PY'
from pathlib import Path
import sys
source = Path('results/table8/HO_linear_simteg_trainseed42/cora_T_inferseed0.jsonl')
target = Path(sys.argv[1])
payload = source.read_bytes()
if target.exists():
    assert target.read_bytes() == payload, 'Existing T file differs.'
else:
    with target.open('xb') as handle:
        handle.write(payload)
PY

unset LIMIT
DIAG_DATASET=cora SOURCE_PREFIX=0 \
EVAL_GPU="$eval_gpu" SOURCE_DIR="$control_root/source" \
OUTPUT_ROOT="$control_root/output_protocol_v2" LOG_DIR="$log_dir" \
bash scripts/run_table8_output_protocol.sh
