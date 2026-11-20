#!/bin/bash
# Sequential diagnostics, pair baselines, and held-out Pubmed LP training/evaluation.
set -euo pipefail
stage="${STAGE:-all}"
root="${PHASE2_ROOT:-./results/lp_phase2}"
data_root="${DATA_ROOT:-dataset_research/lp_bert_disjoint_v1}"
base="${MODEL_BASE:-./models/vicuna-7b-v1.5-16k}"
gpu_include="${GPU_INCLUDE:-localhost:2,3}"
eval_gpu="${EVAL_GPU:-2}"
baseline_gpu="${BASELINE_GPU:-2}"
[[ "$stage" =~ ^(all|strata|baselines|pubmed|summary)$ ]] || { echo 'Invalid STAGE'; exit 1; }
mkdir -p "$root" logs/lp_phase2 checkpoints/lp_phase2
# All stages are deliberately sequential; shared checkpoint paths have one global lock.
exec 9>>checkpoints/lp_phase2/.runner.lock
flock -n 9 || { echo 'An LP phase2 runner is already active. Wait for it; do not duplicate checkpoint writers.' >&2; exit 1; }
unset LIMIT
if [[ "$stage" != summary ]]; then
  python research/verify_clean_lp.py --root "$data_root"
  python research/lp_phase2_manifest.py --root "$root" --data-root "$data_root" --model-base "$base"
fi
if [[ "$stage" == all || "$stage" == strata ]]; then
  python research/stratify_lp_errors.py --data-root "$data_root" --output "$root/error_strata" \
    2>&1 | tee logs/lp_phase2/error_strata.log
fi
if [[ "$stage" == all || "$stage" == baselines ]]; then
  # Existing endpoint MLP is retained as the reference for Cora; fit two added controls.
  CUDA_VISIBLE_DEVICES="$baseline_gpu" python research/run_lp_pair_baselines.py \
    --sources arxiv pubmed --target cora --data-root "$data_root" \
    --output "$root/arxiv-pubmed_to_cora/pair_baselines" --models structure fusion \
    --seeds 42 43 44 --device cuda:0 \
    2>&1 | tee -a logs/lp_phase2/cora_pair_baselines.log
  # Pubmed is now a held-out target: all fitting and threshold selection use Arxiv+Cora.
  CUDA_VISIBLE_DEVICES="$baseline_gpu" python research/run_lp_pair_baselines.py \
    --sources arxiv cora --target pubmed --data-root "$data_root" \
    --output "$root/arxiv-cora_to_pubmed/pair_baselines" --models endpoint structure fusion \
    --seeds 42 43 44 --device cuda:0 \
    2>&1 | tee -a logs/lp_phase2/pubmed_pair_baselines.log
fi
if [[ "$stage" == all || "$stage" == pubmed ]]; then
  [[ "$gpu_include" =~ ^localhost:([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || {
    echo 'Use exactly two distinct GPUs to preserve effective batch16.' >&2; exit 1;
  }
  for mode in full center_repeat; do
    checkpoint="./checkpoints/lp_phase2/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed42-bs16-${mode}"
    log="./logs/lp_phase2/arxiv-cora_seed42_${mode}"
    mkdir -p "$log"
    check=(python research/check_lp_training.py --checkpoint "$checkpoint" --log "$log/train.log" \
      --seed 42 --ablation "$mode" --sources arxiv-cora --data-root "$data_root")
    if "${check[@]}"; then
      echo "Reuse completed training: $checkpoint"
    else
      status=$?
      [[ "$status" == 20 ]] || exit "$status"
      (
        unset CUDA_VISIBLE_DEVICES
        MODEL_BASE="$base" GPU_INCLUDE="$gpu_include" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61009}" \
        LEARNING_RATE=2e-3 GRADIENT_ACCUMULATION_STEPS=2 OUTPUT_DIR="$checkpoint" \
          bash scripts/train_deepspeed.sh vicuna_4hop lp arxiv-cora 4 bert \
          --seed 42 --data_root "$data_root" --embedding_dim 768 --graph_input_ablation "$mode" \
          2>&1 | tee "$log/train.log"
      )
      "${check[@]}"
    fi
    for dataset in arxiv cora pubmed; do
      split=val
      [[ "$dataset" != pubmed ]] || split=test
      LP_CKPT="$checkpoint" MODEL_BASE="$base" DATA_ROOT="$data_root" \
      DATASET="$dataset" LP_SPLIT="$split" LP_ABLATION="$mode" EVAL_GPU="$eval_gpu" \
      OUTPUT_PATH="$root/arxiv-cora_to_pubmed/seed42_${mode}/${dataset}_${split}_${mode}.jsonl" \
        bash scripts/eval_lp_matched.sh
    done
  done
fi
extra=()
if [[ "$stage" == all || "$stage" == summary ]]; then extra+=(--require-complete); fi
python research/summarize_lp_phase2.py --root "$root" --data-root "$data_root" "${extra[@]}"
