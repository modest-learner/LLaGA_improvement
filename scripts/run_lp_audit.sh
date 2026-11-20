#!/bin/bash
set -euo pipefail
root="${AUDIT_ROOT:-./results/lp_protocol_audit/full_v1}"
mkdir -p "$root"
python research/audit_data.py --datasets cora pubmed ogbn-arxiv --output "$root/data.json" \
  2>&1 | tee "$root/data.log"
python research/audit_ho_cache.py --datasets cora pubmed ogbn-arxiv \
  --output "$root/ho_cache.json" 2>&1 | tee "$root/ho_cache.log"
files=()
for seed in 42 43 44; do
  for repeat in 1 2 3; do
    path="results/table5/cora_HO_linear_lr2e-3_seed${seed}_bs16_lp_rep${repeat}.jsonl"
    test -s "$path"
    files+=("$path")
  done
done
python research/audit_lp_outputs.py --inputs "${files[@]}" --output "$root/existing_predictions" \
  2>&1 | tee "$root/existing_predictions.log"
