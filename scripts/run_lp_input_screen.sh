#!/bin/bash
set -euo pipefail

# Uses the author's evaluation entry; runs five conditions sequentially on one GPU.
gpu="${EVAL_GPU:-0}"
checkpoint="${LP_CKPT:-./checkpoints/table5/llaga-HO-arxiv-pubmed-lp-linear-lr2e-3-seed42-bs16}"
base="${MODEL_BASE:-./models/vicuna-7b-v1.5-16k}"
root="${OUTPUT_ROOT:-./results/lp_input_screen/HO_linear_simteg_trainseed42_greedy}"
log="${LOG_ROOT:-./logs/lp_input_screen/HO_linear_simteg_trainseed42_greedy}"
dataset="${DATASET:-cora}"
data_root="${DATA_ROOT:-dataset}"
embedding="${EMB:-simteg}"
context="${LP_CONTEXT:-official}"
extra=()
if [[ -n "${LIMIT:-}" ]]; then
  [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo 'LIMIT must be positive.' >&2; exit 1; }
  extra+=(--end "$LIMIT")
fi
mkdir -p "$root" "$log"
test -s "$checkpoint/mm_projector.bin"
for group in full center_only center_repeat hop_mismatch endpoint_text; do
  mode=graph
  ablation="$group"
  if [[ "$group" == endpoint_text ]]; then mode=text_only; ablation=full; fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  MODEL_PATH="$checkpoint" MODEL_BASE="$base" \
  DATASET="$dataset" TASK=lp TEMPLATE=HO USE_HOP=4 EMB="$embedding" \
  INPUT_MODE="$mode" PROMPT_STYLE=official \
  OUTPUT_PATH="$root/${dataset}_${group}.jsonl" \
  bash scripts/eval.sh --lp_diagnostics --lp_ablation "$ablation" \
    --lp_context "$context" --data_root "$data_root" --ablation_seed 42 \
    --lp_text_max_chars 2000 --greedy --seed 0 --fail_on_error "${extra[@]}" \
    2>&1 | tee -a "$log/${dataset}_${group}.log"
  python eval/eval_res.py --dataset "$dataset" --task lp \
    --res_path "$root/${dataset}_${group}.jsonl" | tee "$root/${dataset}_${group}.official.score.txt"
done
files=("$root/${dataset}_full.jsonl" "$root/${dataset}_center_only.jsonl" \
       "$root/${dataset}_center_repeat.jsonl" "$root/${dataset}_hop_mismatch.jsonl" \
       "$root/${dataset}_endpoint_text.jsonl")
python research/audit_lp_outputs.py --inputs "${files[@]}" --data-root "$data_root" --output "$root/audit"
