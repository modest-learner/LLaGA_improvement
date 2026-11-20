#!/bin/bash
# Extend the locked phase2 recipe to training seeds 43/44; reuse seed42.
set -euo pipefail
stage="${STAGE:-all}"
root="${PHASE2_ROOT:-./results/lp_phase2}"
data_root="${DATA_ROOT:-dataset_research/lp_bert_disjoint_v1}"
base="${MODEL_BASE:-./models/vicuna-7b-v1.5-16k}"
gpu_include="${GPU_INCLUDE:-localhost:2,3}"
eval_gpu="${EVAL_GPU:-2}"
[[ "$stage" =~ ^(all|diagnostics|seeds|summary)$ ]] || { echo 'Invalid STAGE'; exit 1; }
mkdir -p "$root" logs/lp_phase2 checkpoints/lp_phase2
exec 9>>checkpoints/lp_phase2/.runner.lock
flock -n 9 || { echo 'Another phase2 runner is active; wait before using these shared outputs.' >&2; exit 1; }
unset LIMIT
python research/lp_phase2_manifest.py --root "$root" --data-root "$data_root" --model-base "$base"
if [[ "$stage" == all || "$stage" == diagnostics ]]; then
  python research/diagnose_lp_score_shift.py --root "$root"
fi
if [[ "$stage" == all || "$stage" == seeds ]]; then
  [[ "$gpu_include" =~ ^localhost:([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || {
    echo 'Use exactly two distinct GPUs for effective batch16.' >&2; exit 1;
  }
  python research/verify_clean_lp.py --root "$data_root"
  for seed in 43 44; do
    for mode in full center_repeat; do
      checkpoint="./checkpoints/lp_phase2/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed${seed}-bs16-${mode}"
      log="./logs/lp_phase2/arxiv-cora_seed${seed}_${mode}"
      mkdir -p "$log"
      check=(python research/check_lp_training.py --checkpoint "$checkpoint" --log "$log/train.log" \
        --seed "$seed" --ablation "$mode" --sources arxiv-cora --data-root "$data_root")
      if "${check[@]}"; then
        echo "Reuse completed training: $checkpoint"
      else
        status=$?
        [[ "$status" == 20 ]] || exit "$status"
        (
          unset CUDA_VISIBLE_DEVICES
          MODEL_BASE="$base" GPU_INCLUDE="$gpu_include" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61010}" \
          LEARNING_RATE=2e-3 GRADIENT_ACCUMULATION_STEPS=2 OUTPUT_DIR="$checkpoint" \
            bash scripts/train_deepspeed.sh vicuna_4hop lp arxiv-cora 4 bert \
            --seed "$seed" --data_root "$data_root" --embedding_dim 768 --graph_input_ablation "$mode" \
            2>&1 | tee "$log/train.log"
        )
        "${check[@]}"
      fi
      for dataset in arxiv cora pubmed; do
        split=val
        [[ "$dataset" != pubmed ]] || split=test
        LP_CKPT="$checkpoint" MODEL_BASE="$base" DATA_ROOT="$data_root" \
        DATASET="$dataset" LP_SPLIT="$split" LP_ABLATION="$mode" EVAL_GPU="$eval_gpu" \
        OUTPUT_PATH="$root/arxiv-cora_to_pubmed/seed${seed}_${mode}/${dataset}_${split}_${mode}.jsonl" \
          bash scripts/eval_lp_matched.sh
      done
    done
  done
fi
if [[ "$stage" != diagnostics ]]; then
  python research/summarize_lp_pubmed_seeds.py --root "$root" --data-root "$data_root" --model-base "$base"
fi
