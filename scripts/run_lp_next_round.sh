#!/bin/bash
set -euo pipefail

# STAGE=all is sequential; independent individual stages may run on separate GPUs.
stage="${STAGE:-all}"
gpu_include="${GPU_INCLUDE:-localhost:0,3}"
eval_gpu="${EVAL_GPU:-0}"
data_root="${DATA_ROOT:-dataset_research/lp_bert_disjoint_v1}"
root="${ROUND_ROOT:-./results/lp_next_round}"
base="${MODEL_BASE:-./models/vicuna-7b-v1.5-16k}"
[[ "$stage" =~ ^(all|source|baselines|seeds|summary)$ ]] || { echo 'Invalid STAGE' >&2; exit 1; }
mkdir -p "$root" logs/lp_next_round
# Keep compatibility with an already running legacy runner holding an exclusive lock.
# New runners share this lock and exclude duplicate writers using per-stage locks.
exec 9>>"$root/.runner.lock"
if ! flock -sn 9; then
  python research/check_lp_legacy_lock.py --lock "$root/.runner.lock" --stage "$stage"
fi
stage_lock_fds=()
for part in source baselines seeds; do
  if [[ "$stage" == all || "$stage" == "$part" ]]; then
    exec {stage_fd}>>"$root/.stage_${part}.lock"
    flock -n "$stage_fd" || {
      echo "Stage '$part' is already running for $root; use a different stage or wait." >&2
      exit 1
    }
    stage_lock_fds+=("$stage_fd")
  fi
done
if [[ "$stage" != summary ]]; then
  python research/verify_clean_lp.py --root "$data_root"
  exec 8>>"$root/.recipe.lock"
  flock 8
  python - "$data_root" "$base" "$root/round_recipe.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
data_root,base,target=map(Path,sys.argv[1:])
recipe=dict(protocol='matched_lp_next_round_v1',source_graphs=['arxiv','pubmed'],target='cora',
            training_seeds=[42,43,44],inputs=['full','center_repeat'],
            source_manifests={n:sha(data_root/n/'clean_lp_manifest.json') for n in ['ogbn-arxiv','pubmed','cora']},
            llm_config=sha(base/'config.json'),tokenizer=sha(base/'tokenizer.model'),
            code={f:sha(f) for f in ['train/train.py','train/llaga_trainer.py','model/llaga_arch.py','eval/eval_pretrain.py','eval/lp_inputs.py']})
if target.exists():
    assert json.loads(target.read_text())==recipe,'Round data/model/code recipe changed; inspect before mixing results.'
else:
    target.write_text(json.dumps(recipe,indent=2)+'\n')
PY
  exec 8>&-
fi

evaluate_model() {
  local seed="$1" mode="$2" dataset output split
  local checkpoint="./checkpoints/lp_clean/llaga-HO-arxiv-pubmed-lp-linear-bert-lr2e-3-seed${seed}-bs16-${mode}"
  local train_log="./logs/lp_clean/arxiv-pubmed_seed${seed}_${mode}/train.log"
  python research/check_lp_training.py --checkpoint "$checkpoint" --log "$train_log" --seed "$seed" --ablation "$mode" --data-root "$data_root"
  for dataset in arxiv pubmed cora; do
    split=val
    output="$root/source_validation/seed${seed}_${mode}/${dataset}_val_${mode}.jsonl"
    if [[ "$dataset" == cora ]]; then
      split=test
      output="./results/lp_clean/arxiv-pubmed_to_cora_trainseed${seed}_${mode}/screen/cora_${mode}.jsonl"
    fi
    LP_CKPT="$checkpoint" OUTPUT_PATH="$output" DATASET="$dataset" LP_SPLIT="$split" LP_ABLATION="$mode" \
    EVAL_GPU="$eval_gpu" DATA_ROOT="$data_root" MODEL_BASE="$base" \
      bash scripts/eval_lp_matched.sh
  done
}

if [[ "$stage" == all || "$stage" == source ]]; then
  for mode in full center_repeat; do evaluate_model 42 "$mode"; done
fi

if [[ "$stage" == all || "$stage" == baselines ]]; then
  CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-$eval_gpu}" \
    python research/run_lp_simple_baselines.py --data-root "$data_root" \
      --output "$root/simple_baselines" --device "${BASELINE_DEVICE:-cuda:0}" --seeds 42 43 44 \
      2>&1 | tee -a logs/lp_next_round/simple_baselines.log
fi

if [[ "$stage" == all || "$stage" == seeds ]]; then
  [[ "$gpu_include" =~ ^localhost:([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || {
    echo 'Use exactly two distinct training GPUs, e.g. GPU_INCLUDE=localhost:0,3, to keep batch16.' >&2; exit 1;
  }
  for seed in 43 44; do
    for mode in full center_repeat; do
      checkpoint="./checkpoints/lp_clean/llaga-HO-arxiv-pubmed-lp-linear-bert-lr2e-3-seed${seed}-bs16-${mode}"
      train_log="./logs/lp_clean/arxiv-pubmed_seed${seed}_${mode}/train.log"
      if python research/check_lp_training.py --checkpoint "$checkpoint" --log "$train_log" --seed "$seed" --ablation "$mode" --data-root "$data_root"; then
        echo "Skip completed training: $checkpoint"
      else
        status=$?
        [[ "$status" == 20 ]] || exit "$status"
        mkdir -p "$(dirname "$train_log")"
        (
          unset CUDA_VISIBLE_DEVICES
          MODEL_BASE="$base" GPU_INCLUDE="$gpu_include" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61008}" \
          LEARNING_RATE=2e-3 GRADIENT_ACCUMULATION_STEPS=2 OUTPUT_DIR="$checkpoint" \
          bash scripts/train_deepspeed.sh vicuna_4hop lp arxiv-pubmed 4 bert \
            --seed "$seed" --data_root "$data_root" --embedding_dim 768 --graph_input_ablation "$mode" \
            2>&1 | tee "$train_log"
        )
      fi
      evaluate_model "$seed" "$mode"
    done
  done
fi

extra=()
if [[ "$stage" == all || "$stage" == summary ]]; then extra+=(--require-complete); fi
python research/summarize_lp_next_round.py --root "$root" --data-root "$data_root" "${extra[@]}"
