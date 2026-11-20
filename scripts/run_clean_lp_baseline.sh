#!/bin/bash
set -euo pipefail

data_root="${DATA_ROOT:-dataset_research/lp_bert_disjoint_v1}"
seed="${TRAIN_SEED:-42}"
sources="${SOURCES:-arxiv-pubmed}"
ablation="${TRAIN_ABLATION:-full}"
checkpoint="${LP_CKPT:-./checkpoints/lp_clean/llaga-HO-${sources}-lp-linear-bert-lr2e-3-seed${seed}-bs16-${ablation}}"
root="${OUTPUT_ROOT:-./results/lp_clean/${sources}_to_cora_trainseed${seed}_${ablation}}"
log="${LOG_ROOT:-./logs/lp_clean/${sources}_seed${seed}_${ablation}}"
[[ "$ablation" == full || "$ablation" == center_repeat ]] || exit 1
[[ "$sources" == arxiv || "$sources" == pubmed || "$sources" == arxiv-pubmed ]] || exit 1
if [[ -e "$checkpoint" ]]; then
  echo "Checkpoint exists: $checkpoint. Inspect it; use a new LP_CKPT for a new training run." >&2
  exit 1
fi
python research/verify_clean_lp.py --root "$data_root"
mkdir -p "$root" "$log"
(
  unset CUDA_VISIBLE_DEVICES
  MODEL_BASE="${MODEL_BASE:-./models/vicuna-7b-v1.5-16k}" \
  GPU_INCLUDE="${GPU_INCLUDE:-localhost:0,3}" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61007}" \
  LEARNING_RATE=2e-3 GRADIENT_ACCUMULATION_STEPS=2 OUTPUT_DIR="$checkpoint" \
  bash scripts/train_deepspeed.sh vicuna_4hop lp "$sources" 4 bert \
    --seed "$seed" --data_root "$data_root" --embedding_dim 768 \
    --graph_input_ablation "$ablation" \
    2>&1 | tee "$log/train.log"
)
test -s "$checkpoint/mm_projector.bin"
unset LIMIT
EVAL_GPU="${EVAL_GPU:-0}" LP_CKPT="$checkpoint" DATASET=cora EMB=bert \
DATA_ROOT="$data_root" LP_CONTEXT=clean_cached OUTPUT_ROOT="$root/screen" LOG_ROOT="$log/screen" \
bash scripts/run_lp_input_screen.sh
