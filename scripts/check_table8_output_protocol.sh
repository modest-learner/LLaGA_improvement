#!/bin/bash
set -euo pipefail

# Fixed source-held-out prefix: interface check, never tune the protocol by its ACC.
eval_gpu="${EVAL_GPU:-3}"
check_n="${CHECK_N:-8}"
check_root="${CHECK_ROOT:-./results/table8/output_protocol_v2_sourcecheck_pubmed}"
checkpoint="${CKPT:-./checkpoints/table8/llaga-HO-arxiv-pubmed-nd-linear-simteg-lr2e-3-seed42-bs16}"
mkdir -p "$check_root/source" "$check_root/logs"
python - "$check_n" <<'PY'
import json, sys, torch
from pathlib import Path
n = int(sys.argv[1])
assert n > 0
rows = [json.loads(line) for line in Path('dataset/pubmed/sampled_2_10_test.jsonl').read_text().splitlines()][:n]
assert len(rows) == n
data = torch.load('dataset/pubmed/processed_data.pt')
assert all(not bool(data.train_mask[row['id']]) for row in rows), 'Source check contains a training node.'
PY

for group in E_official E ET T; do
  case "$group" in
    E_official) task=nc; input_mode=graph; prompt_style=official ;;
    E) task=nc; input_mode=graph; prompt_style=matched ;;
    ET) task=nctext; input_mode=graph; prompt_style=matched ;;
    T) task=nctext; input_mode=text_only; prompt_style=matched ;;
  esac
  CUDA_VISIBLE_DEVICES="$eval_gpu" \
  MODEL_BASE=./models/vicuna-7b-v1.5-16k MODEL_PATH="$checkpoint" \
  DATASET=pubmed TASK="$task" TEMPLATE=HO USE_HOP=4 EMB=simteg \
  INPUT_MODE="$input_mode" PROMPT_STYLE="$prompt_style" \
  OUTPUT_PATH="$check_root/source/pubmed_${group}_inferseed0.jsonl" \
  bash scripts/eval.sh --seed 0 --end "$check_n" --fail_on_error \
    2>&1 | tee -a "$check_root/logs/pubmed_${group}_source.log"
done

EVAL_GPU="$eval_gpu" DIAG_DATASET=pubmed SOURCE_PREFIX=1 LIMIT="$check_n" \
SOURCE_DIR="$check_root/source" OUTPUT_ROOT="$check_root/diagnostics" \
LOG_DIR="$check_root/logs" bash scripts/run_table8_output_protocol.sh
