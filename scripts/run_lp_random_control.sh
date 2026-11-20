#!/bin/bash
set -euo pipefail
stage="${STAGE:-all}"
[[ "$stage" =~ ^(all|check|train_eval|summary)$ ]] || { echo 'Invalid STAGE'; exit 1; }
gpus="${GPU_INCLUDE:-localhost:1,2}"
[[ "$gpus" =~ ^localhost:([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || { echo 'Exactly two distinct training GPUs required'; exit 1; }
eval_gpu="${EVAL_GPU:-2}"
root=results/lp_random_control
mkdir -p "$root" logs/lp_random_control checkpoints/lp_random_control
exec 9>>checkpoints/lp_random_control/.runner.lock
flock -n 9 || { echo 'Another random-control runner is active'; exit 1; }
unset LIMIT
python research/prepare_lp_random_control.py
python research/pin_lp_random_control.py
if [[ "$stage" != summary ]]; then
 python research/verify_clean_lp.py --root dataset_research/lp_bert_disjoint_v1
 # Always enforce successful real-model equivalence before full training.
 for branch in baseline ones; do
  cp="checkpoints/lp_random_control/equivalence/$branch"
  if [[ -s "$cp/trainer_state.json" && -s "$cp/mm_projector.bin" ]]; then
   echo "Reuse short check: $cp"
  else
   [[ ! -d "$cp" ]] || { echo "Incomplete short check: $cp; inspect before restarting"; exit 1; }
   entry=scripts/train_deepspeed.sh
   [[ "$branch" != ones ]] || entry=scripts/train_deepspeed_lp_weighted.sh
   (
    unset CUDA_VISIBLE_DEVICES
    LP_WEIGHT_MANIFEST="$root/ones_weights.json" MODEL_BASE=./models/vicuna-7b-v1.5-16k \
    GPU_INCLUDE="localhost:$eval_gpu" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61015}" \
    LEARNING_RATE=2e-3 GRADIENT_ACCUMULATION_STEPS=4 OUTPUT_DIR="$cp" \
     bash "$entry" vicuna_4hop lp arxiv-cora 4 bert \
      --seed 42 --data_root dataset_research/lp_bert_disjoint_v1 --embedding_dim 768 --graph_input_ablation full \
      --max_steps 4 --report_to none 2>&1 | tee "logs/lp_random_control/check_${branch}.log"
   )
  fi
 done
 python research/check_lp_real_equivalence.py
fi
if [[ "$stage" == all || "$stage" == train_eval ]]; then
 cp=checkpoints/lp_random_control/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed42-bs16-randomneg
 log=logs/lp_random_control/train_seed42.log
 check=(python research/check_lp_training.py --checkpoint "$cp" --log "$log" --seed 42 --ablation full --sources arxiv-cora)
 if "${check[@]}"; then echo "Reuse $cp"; else
  status=$?; [[ "$status" == 20 ]] || exit "$status"
  (
   unset CUDA_VISIBLE_DEVICES
   LP_WEIGHT_MANIFEST="$root/random_weights.json" MODEL_BASE=./models/vicuna-7b-v1.5-16k \
   GPU_INCLUDE="$gpus" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61015}" LEARNING_RATE=2e-3 \
   GRADIENT_ACCUMULATION_STEPS=2 OUTPUT_DIR="$cp" \
    bash scripts/train_deepspeed_lp_weighted.sh vicuna_4hop lp arxiv-cora 4 bert \
     --seed 42 --data_root dataset_research/lp_bert_disjoint_v1 --embedding_dim 768 --graph_input_ablation full \
      2>&1 | tee "$log"
  )
  "${check[@]}"
 fi
 for ds in arxiv cora pubmed; do
  split=val; [[ "$ds" != pubmed ]] || split=test
  LP_CKPT="$cp" DATASET="$ds" LP_SPLIT="$split" LP_ABLATION=full EVAL_GPU="$eval_gpu" \
   OUTPUT_PATH="$root/predictions/${ds}_${split}_full.jsonl" bash scripts/eval_lp_matched.sh
 done
 CUDA_VISIBLE_DEVICES="$eval_gpu" python eval/score_lp_binary_weighted.py --checkpoint "$cp" \
  --seed 42 --ablation full --target pubmed --root "$root" --pilot 2>&1 | tee "$root/score_pilot.log"
 CUDA_VISIBLE_DEVICES="$eval_gpu" python eval/score_lp_binary_weighted.py --checkpoint "$cp" \
  --seed 42 --ablation full --target pubmed --root "$root" 2>&1 | tee -a "$root/score.log"
fi
if [[ "$stage" != check ]]; then python research/summarize_lp_random_control.py; fi
