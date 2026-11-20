#!/bin/bash
set -euo pipefail
stage="${STAGE:-all}"
[[ "$stage" =~ ^(all|pubmed|cora|summary)$ ]] || { echo 'Invalid STAGE'; exit 1; }
mkdir -p results/lp_cora_scores logs/lp_cross_direction
exec 9>>results/lp_cora_scores/.runner.lock
flock -n 9 || { echo 'Another cross-direction runner is active.' >&2; exit 1; }
unset LIMIT
if [[ "$stage" == all || "$stage" == pubmed || "$stage" == summary ]]; then
 python research/lp_threshold_stability.py --direction pubmed --repeats 500 \
  2>&1 | tee logs/lp_cross_direction/pubmed_threshold.log
fi
if [[ "$stage" == all || "$stage" == cora ]]; then
 python research/verify_clean_lp.py --root dataset_research/lp_bert_disjoint_v1
 for seed in 42 43 44; do
  for mode in full center_repeat; do
   checkpoint="checkpoints/lp_clean/llaga-HO-arxiv-pubmed-lp-linear-bert-lr2e-3-seed${seed}-bs16-${mode}"
   for ds in arxiv pubmed cora; do
    split=val
    output="results/lp_next_round/source_validation/seed${seed}_${mode}/${ds}_val_${mode}.jsonl"
    if [[ "$ds" == cora ]]; then
     split=test; output="results/lp_clean/arxiv-pubmed_to_cora_trainseed${seed}_${mode}/screen/cora_${mode}.jsonl"
    fi
    python research/check_lp_prediction.py --checkpoint "$checkpoint" --output "$output" \
     --dataset "$ds" --split "$split" --ablation "$mode"
   done
   if [[ "$seed" == 42 ]]; then
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-3}" python eval/score_lp_binary_cora.py \
      --checkpoint "$checkpoint" --seed "$seed" --ablation "$mode" --pilot \
      2>&1 | tee -a "logs/lp_cross_direction/pilot_${mode}.log"
   fi
   CUDA_VISIBLE_DEVICES="${EVAL_GPU:-3}" python eval/score_lp_binary_cora.py \
     --checkpoint "$checkpoint" --seed "$seed" --ablation "$mode" \
     2>&1 | tee -a "logs/lp_cross_direction/seed${seed}_${mode}.log"
  done
 done
fi
if [[ "$stage" != pubmed ]]; then
 python research/lp_threshold_stability.py --direction cora --repeats 500 \
   2>&1 | tee logs/lp_cross_direction/cora_threshold.log
fi
