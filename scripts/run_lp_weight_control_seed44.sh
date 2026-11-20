#!/bin/bash
# Seed44 semantic/random control; reuse original baseline and completed models.
set -euo pipefail
stage="${STAGE:-all}"
[[ "$stage" =~ ^(all|check|summary)$ ]] || { echo 'STAGE must be all, check or summary'; exit 1; }
gpus="${GPU_INCLUDE:-localhost:1,2}"
eval_gpu="${EVAL_GPU:-2}"
[[ "$gpus" =~ ^localhost:([0-9]+)(,([0-9]+))?$ ]] || { echo 'Use localhost:N or localhost:N,M'; exit 1; }
IFS=, read -r -a devices <<< "${gpus#localhost:}"
world=${#devices[@]}
[[ "$world" == 1 || "${devices[0]}" != "${devices[1]}" ]] || { echo 'Duplicate GPUs'; exit 1; }
micro="${MICRO_BATCH_SIZE:-4}"
[[ "$micro" =~ ^[1-9][0-9]*$ ]] || { echo 'Positive integer MICRO_BATCH_SIZE required'; exit 1; }
(( 16 % (world * micro) == 0 )) || { echo 'GPU count x microbatch must divide 16'; exit 1; }
accum=$((16 / (world * micro)))
root=results/lp_weight_control_seed44
mkdir -p "$root" logs/lp_weight_control_seed44 checkpoints/lp_weight_control_seed44
exec 9>>checkpoints/lp_weight_control_seed44/.runner.lock
flock -n 9 || { echo 'Another seed44 runner is active; do not launch concurrently.'; exit 1; }
unset CUDA_VISIBLE_DEVICES LIMIT
python research/prepare_lp_random_control.py
python research/pin_lp_random_control.py
python research/pin_lp_weight_control_seed44.py
python research/check_lp_real_equivalence.py
if [[ "$stage" != summary ]]; then
 for branch in semantic random; do
  cp="checkpoints/lp_weight_control_seed44/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed44-bs16-$branch"
  log="logs/lp_weight_control_seed44/train_${branch}.log"
  output="$root/$branch"
  mkdir -p "$output"
  manifest=results/lp_weighted/weights/arxiv-cora/weights.json
  [[ "$branch" != random ]] || manifest=results/lp_random_control/random_weights.json
  check=(python research/check_lp_training.py --checkpoint "$cp" --log "$log" --seed 44 --ablation full --sources arxiv-cora --effective-batch-size 16)
  if "${check[@]}"; then echo "Reuse completed $branch training"; else
   status=$?; [[ "$status" == 20 ]] || exit "$status"
   if [[ "$stage" == check ]]; then echo "Ready: $branch; GPUs=$gpus microbatch=$micro accumulation=$accum effective_batch=16"; continue; fi
   echo "Start $branch: GPUs=$gpus microbatch=$micro accumulation=$accum effective_batch=16"
   LP_WEIGHT_MANIFEST="$manifest" MODEL_BASE=./models/vicuna-7b-v1.5-16k \
   GPU_INCLUDE="$gpus" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61016}" LEARNING_RATE=2e-3 \
   GRADIENT_ACCUMULATION_STEPS="$accum" OUTPUT_DIR="$cp" \
    bash scripts/train_deepspeed_lp_weighted.sh vicuna_4hop lp arxiv-cora "$micro" bert \
    --seed 44 --data_root dataset_research/lp_bert_disjoint_v1 --embedding_dim 768 --graph_input_ablation full \
    2>&1 | tee "$log"
   "${check[@]}"
  fi
  [[ "$stage" != check ]] || continue
  for ds in arxiv cora pubmed; do
   split=val; [[ "$ds" != pubmed ]] || split=test
   LP_CKPT="$cp" DATASET="$ds" LP_SPLIT="$split" LP_ABLATION=full EVAL_GPU="$eval_gpu" \
    OUTPUT_PATH="$output/predictions/${ds}_${split}_full.jsonl" bash scripts/eval_lp_matched.sh
  done
  CUDA_VISIBLE_DEVICES="$eval_gpu" python eval/score_lp_binary_weighted.py --checkpoint "$cp" --seed 44 --ablation full --target pubmed --root "$output" --pilot 2>&1 | tee "$output/score_pilot.log"
  CUDA_VISIBLE_DEVICES="$eval_gpu" python eval/score_lp_binary_weighted.py --checkpoint "$cp" --seed 44 --ablation full --target pubmed --root "$output" 2>&1 | tee -a "$output/score.log"
 done
fi
if [[ "$stage" != check ]]; then
 python research/summarize_lp_weight_control.py --seed 44 --semantic-root "$root/semantic" --random-root "$root/random" --output-dir "$root/summary"
 python research/summarize_lp_weight_three_seeds.py
fi
