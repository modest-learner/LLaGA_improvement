#!/bin/bash
set -euo pipefail

# Run from repository root. Original sampling and candidate-score files are inputs only.
eval_gpu="${EVAL_GPU:-3}"
dataset="${DIAG_DATASET:-cora}"
source_dir="${SOURCE_DIR:-./results/table8/HO_linear_simteg_trainseed42}"
output_root="${OUTPUT_ROOT:-./results/table8/output_protocol_v2_HO_linear_simteg_trainseed42}"
log_dir="${LOG_DIR:-./logs/table8/output_protocol_v2}"
extra_args=()
if [[ -n "${LIMIT:-}" ]]; then extra_args+=(--limit "$LIMIT"); fi
if [[ "${SOURCE_PREFIX:-0}" = 1 ]]; then extra_args+=(--source_prefix); fi
mkdir -p "$output_root" "$log_dir"

# Preflight every source before starting any model load.
for group in E_official E ET T; do
  test -s "$source_dir/${dataset}_${group}_inferseed0.jsonl"
done

for group in E_official E ET T; do
  for decoding in greedy constrained_greedy; do
    name="${dataset}_${group}_${decoding}"
    CUDA_VISIBLE_DEVICES="$eval_gpu" \
    bash scripts/eval_nc_diagnostics.sh \
      --dataset "$dataset" \
      --source_results "$source_dir/${dataset}_${group}_inferseed0.jsonl" \
      --answers_file "$output_root/$name.jsonl" \
      --decoding "$decoding" --ablation full "${extra_args[@]}" \
      2>&1 | tee -a "$log_dir/$name.log"

    python eval/eval_res.py --dataset "$dataset" --task nc \
      --res_path "$output_root/$name.jsonl" \
      | tee "$output_root/$name.score.txt"
  done
done

python research/summarize_output_protocol.py \
  --dataset "$dataset" --source_dir "$source_dir" --output_dir "$output_root"
