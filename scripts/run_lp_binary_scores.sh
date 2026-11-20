#!/bin/bash
set -euo pipefail
stage="${STAGE:-all}"
root="${PHASE2_ROOT:-./results/lp_phase2}"
data_root="${DATA_ROOT:-dataset_research/lp_bert_disjoint_v1}"
base="${MODEL_BASE:-./models/vicuna-7b-v1.5-16k}"
[[ "$stage" =~ ^(all|pilot|score|summary)$ ]] || { echo 'Invalid STAGE'; exit 1; }
mkdir -p "$root/binary_scores" logs/lp_binary_scores
exec 9>>"$root/binary_scores/.runner.lock"
flock -n 9 || { echo 'Another binary scorer is active for this output root.' >&2; exit 1; }
unset LIMIT
if [[ "$stage" != summary ]]; then
 python research/lp_phase2_manifest.py --root "$root" --data-root "$data_root" --model-base "$base"
 python research/verify_clean_lp.py --root "$data_root"
 python research/summarize_lp_pubmed_seeds.py --root "$root" --data-root "$data_root" --model-base "$base"
 # Required source-only pilot, including when starting directly at score stage.
 for mode in full center_repeat; do
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-3}" python eval/score_lp_binary.py \
   --checkpoint "checkpoints/lp_phase2/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed42-bs16-${mode}" \
   --seed 42 --ablation "$mode" --root "$root" --data-root "$data_root" --model-base "$base" --pilot \
   2>&1 | tee -a "logs/lp_binary_scores/pilot_${mode}.log"
 done
fi
if [[ "$stage" == all || "$stage" == score ]]; then
 for seed in 42 43 44; do
  for mode in full center_repeat; do
   CUDA_VISIBLE_DEVICES="${EVAL_GPU:-3}" python eval/score_lp_binary.py \
    --checkpoint "checkpoints/lp_phase2/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed${seed}-bs16-${mode}" \
    --seed "$seed" --ablation "$mode" --root "$root" --data-root "$data_root" --model-base "$base" \
    2>&1 | tee -a "logs/lp_binary_scores/seed${seed}_${mode}.log"
  done
 done
fi
if [[ "$stage" != pilot ]]; then
 python research/summarize_lp_binary_scores.py --root "$root"
fi
