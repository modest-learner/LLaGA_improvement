#!/bin/bash
set -euo pipefail
checkpoint="${LP_CKPT:?Set LP_CKPT}"
output="${OUTPUT_PATH:?Set OUTPUT_PATH}"
dataset="${DATASET:-cora}"
split="${LP_SPLIT:-test}"
ablation="${LP_ABLATION:-full}"
data_root="${DATA_ROOT:-dataset_research/lp_bert_disjoint_v1}"
base="${MODEL_BASE:-./models/vicuna-7b-v1.5-16k}"
log="${EVAL_LOG:-${output%.jsonl}.log}"
mkdir -p "$(dirname "$output")" "$(dirname "$log")"
if python research/check_lp_prediction.py --checkpoint "$checkpoint" --output "$output" \
    --dataset "$dataset" --split "$split" --ablation "$ablation" \
    --data-root "$data_root" --model-base "$base"; then
  echo "Skip completed evaluation: $output"
else
  status=$?
  if [[ "$status" != 10 && "$status" != 20 ]]; then exit "$status"; fi
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" \
  MODEL_PATH="$checkpoint" MODEL_BASE="$base" DATASET="$dataset" TASK=lp \
  TEMPLATE=HO USE_HOP=4 EMB=bert INPUT_MODE=graph PROMPT_STYLE=official OUTPUT_PATH="$output" \
  bash scripts/eval.sh --lp_diagnostics --lp_ablation "$ablation" \
    --lp_context clean_cached --lp_split "$split" --data_root "$data_root" \
    --ablation_seed 42 --lp_text_max_chars 2000 --greedy --seed 0 --fail_on_error \
    2>&1 | tee -a "$log"
fi
python eval/eval_res.py --dataset "$dataset" --task lp --res_path "$output" \
  | tee "${output%.jsonl}.official.score.txt"
python research/audit_lp_outputs.py --inputs "$output" --data-root "$data_root" \
  --output "${output%.jsonl}.audit"
