#!/bin/bash
set -euo pipefail
stage="${STAGE:-all}"
[[ "$stage" =~ ^(all|audit|train_eval|summary)$ ]] || exit 1
gpus="${GPU_INCLUDE:-localhost:1,2}"; eval_gpu="${EVAL_GPU:-2}"
[[ "$gpus" =~ ^localhost:([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || { echo 'Exactly two distinct GPUs required'; exit 1; }
root=results/lp_weight_control_seed43
mkdir -p "$root" logs/lp_weight_control_seed43 checkpoints/lp_weight_control_seed43
exec 9>>checkpoints/lp_weight_control_seed43/.runner.lock
flock -n 9 || { echo 'Another seed43 control runner is active'; exit 1; }
unset CUDA_VISIBLE_DEVICES LIMIT
python research/prepare_lp_random_control.py
python research/pin_lp_random_control.py
python research/pin_lp_weight_control_seed43.py
python research/check_lp_real_equivalence.py
if [[ "$stage" == all || "$stage" == audit ]]; then
 python research/verify_clean_lp.py --root dataset_research/lp_bert_disjoint_v1
 python research/audit_lp_weight_selection.py
fi
if [[ "$stage" == all || "$stage" == train_eval ]]; then
 for branch in semantic random; do
  cp="checkpoints/lp_weight_control_seed43/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed43-bs16-$branch"
  log="logs/lp_weight_control_seed43/train_${branch}.log"
  output="$root/$branch"
  mkdir -p "$output"
  manifest=results/lp_weighted/weights/arxiv-cora/weights.json
  [[ "$branch" != random ]] || manifest=results/lp_random_control/random_weights.json
  check=(python research/check_lp_training.py --checkpoint "$cp" --log "$log" --seed 43 --ablation full --sources arxiv-cora)
  if "${check[@]}"; then echo "Reuse $cp"; else
   status=$?; [[ "$status" == 20 ]] || exit "$status"
   LP_WEIGHT_MANIFEST="$manifest" MODEL_BASE=./models/vicuna-7b-v1.5-16k \
   GPU_INCLUDE="$gpus" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61016}" LEARNING_RATE=2e-3 \
   GRADIENT_ACCUMULATION_STEPS=2 OUTPUT_DIR="$cp" \
    bash scripts/train_deepspeed_lp_weighted.sh vicuna_4hop lp arxiv-cora 4 bert \
    --seed 43 --data_root dataset_research/lp_bert_disjoint_v1 --embedding_dim 768 --graph_input_ablation full \
    2>&1 | tee "$log"
   "${check[@]}"
  fi
  for ds in arxiv cora pubmed; do
   split=val; [[ "$ds" != pubmed ]] || split=test
   LP_CKPT="$cp" DATASET="$ds" LP_SPLIT="$split" LP_ABLATION=full EVAL_GPU="$eval_gpu" \
    OUTPUT_PATH="$output/predictions/${ds}_${split}_full.jsonl" bash scripts/eval_lp_matched.sh
  done
  CUDA_VISIBLE_DEVICES="$eval_gpu" python eval/score_lp_binary_weighted.py --checkpoint "$cp" --seed 43 --ablation full --target pubmed --root "$output" --pilot 2>&1 | tee "$output/score_pilot.log"
  CUDA_VISIBLE_DEVICES="$eval_gpu" python eval/score_lp_binary_weighted.py --checkpoint "$cp" --seed 43 --ablation full --target pubmed --root "$output" 2>&1 | tee -a "$output/score.log"
 done
fi
if [[ "$stage" == all || "$stage" == summary ]]; then
 python research/summarize_lp_weight_control.py --output-dir "$root/summary"
fi
