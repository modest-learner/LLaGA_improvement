#!/bin/bash
set -euo pipefail
stage="${STAGE:-all}"
[[ "$stage" =~ ^(all|prepare|train_eval|summary)$ ]] || { echo 'Invalid STAGE'; exit 1; }
gpus="${GPU_INCLUDE:-localhost:2,3}"
[[ "$gpus" =~ ^localhost:([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || { echo 'Exactly two distinct GPUs required for effective batch16'; exit 1; }
mkdir -p results/lp_weighted logs/lp_weighted checkpoints/lp_weighted
exec 9>>checkpoints/lp_weighted/.runner.lock
flock -n 9 || { echo 'Another weighted trial is active'; exit 1; }
unset LIMIT
if [[ "$stage" != summary ]]; then
 python research/verify_clean_lp.py --root dataset_research/lp_bert_disjoint_v1
 for sources in arxiv-pubmed arxiv-cora; do python research/prepare_lp_weights.py --sources "$sources"; done
 python research/test_lp_weighted_loss.py
fi
for target in cora pubmed; do
 sources=arxiv-pubmed; [[ "$target" != pubmed ]] || sources=arxiv-cora
 root="results/lp_weighted/${sources}_to_${target}"
 checkpoint="checkpoints/lp_weighted/llaga-HO-${sources}-lp-linear-bert-lr2e-3-seed42-bs16-full-negweight2"
 log="logs/lp_weighted/${sources}_seed42.log"
 python research/pin_lp_weighted.py --sources "$sources" --root "$root"
 if [[ "$stage" == all || "$stage" == train_eval ]]; then
  check=(python research/check_lp_training.py --checkpoint "$checkpoint" --log "$log" --seed 42 --ablation full --sources "$sources")
  if "${check[@]}"; then echo "Reuse $checkpoint"; else
   status=$?; [[ "$status" == 20 ]] || exit "$status"
   (
    unset CUDA_VISIBLE_DEVICES
    LP_WEIGHT_MANIFEST="results/lp_weighted/weights/${sources}/weights.json" \
    MODEL_BASE=./models/vicuna-7b-v1.5-16k GPU_INCLUDE="$gpus" \
    LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61014}" LEARNING_RATE=2e-3 GRADIENT_ACCUMULATION_STEPS=2 OUTPUT_DIR="$checkpoint" \
     bash scripts/train_deepspeed_lp_weighted.sh vicuna_4hop lp "$sources" 4 bert \
      --seed 42 --data_root dataset_research/lp_bert_disjoint_v1 --embedding_dim 768 --graph_input_ablation full \
      2>&1 | tee "$log"
   )
   "${check[@]}"
  fi
  for ds in arxiv pubmed cora; do
   split=val; [[ "$ds" != "$target" ]] || split=test
   LP_CKPT="$checkpoint" DATASET="$ds" LP_SPLIT="$split" LP_ABLATION=full EVAL_GPU="${EVAL_GPU:-3}" \
    OUTPUT_PATH="$root/predictions/${ds}_${split}_full.jsonl" bash scripts/eval_lp_matched.sh
  done
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-3}" python eval/score_lp_binary_weighted.py \
   --checkpoint "$checkpoint" --seed 42 --ablation full --target "$target" --root "$root" --pilot \
    2>&1 | tee "$root/score_pilot.log"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-3}" python eval/score_lp_binary_weighted.py \
   --checkpoint "$checkpoint" --seed 42 --ablation full --target "$target" --root "$root" \
    2>&1 | tee -a "$root/score.log"
 fi
 if [[ "$stage" != prepare ]]; then python research/summarize_lp_weighted.py --target "$target"; fi
done
