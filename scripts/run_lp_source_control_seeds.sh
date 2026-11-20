#!/bin/bash
set -euo pipefail
stage="${STAGE:-all}"
[[ "$stage" =~ ^(all|check|summary)$ ]] || exit 1
gpus="${GPU_INCLUDE:-localhost:1,2}"; eval_gpu="${EVAL_GPU:-2}"
[[ "$gpus" =~ ^localhost:([0-9]+)(,([0-9]+))?$ ]] || { echo 'Use one or two GPUs'; exit 1; }
IFS=, read -r -a devices <<< "${gpus#localhost:}"
world=${#devices[@]}
[[ "$world" == 1 || "${devices[0]}" != "${devices[1]}" ]] || exit 1
micro="${MICRO_BATCH_SIZE:-4}"
[[ "$micro" =~ ^[1-9][0-9]*$ ]] || exit 1
((16 % (world * micro) == 0)) || exit 1
accum=$((16 / (world * micro)))
root=results/lp_source_control/arxiv_only
mkdir -p "$root" logs/lp_source_control checkpoints/lp_source_control
exec 9>>checkpoints/lp_source_control/.runner.lock
flock -n 9 || { echo 'Another source-control runner is active.'; exit 1; }
unset CUDA_VISIBLE_DEVICES LIMIT LP_WEIGHT_MANIFEST
python research/prepare_lp_source_control.py
python research/pin_lp_source_seeds.py
if [[ "$stage" != summary ]]; then
 python research/verify_clean_lp.py --root dataset_research/lp_bert_disjoint_v1
 python research/diagnose_lp_stable_pairs.py
fi
if [[ "$stage" == all ]]; then
 for seed in 43 44; do
 root=results/lp_source_control/seed${seed}/arxiv_only
 mkdir -p "$root"
 for mode in full center_repeat; do
  cp="checkpoints/lp_source_control/llaga-HO-arxiv-lp-linear-bert-lr2e-3-seed${seed}-bs16-budget81849-$mode"
  log="logs/lp_source_control/train_seed${seed}_${mode}.log"
  check=(python research/check_lp_training.py --checkpoint "$cp" --log "$log" --seed "$seed" --ablation "$mode" --sources arxiv --effective-batch-size 16)
  if "${check[@]}"; then echo "Reuse $cp"; else
   status=$?; [[ "$status" == 20 ]] || exit "$status"
   MODEL_BASE=./models/vicuna-7b-v1.5-16k GPU_INCLUDE="$gpus" LLAGA_MASTER_PORT="${LLAGA_MASTER_PORT:-61019}" \
   LEARNING_RATE=2e-3 GRADIENT_ACCUMULATION_STEPS="$accum" OUTPUT_DIR="$cp" \
    bash scripts/train_deepspeed_lp_source_control.sh vicuna_4hop lp arxiv "$micro" bert \
    --seed "$seed" --data_root dataset_research/lp_bert_disjoint_v1 --embedding_dim 768 --graph_input_ablation "$mode" \
    2>&1 | tee "$log"
   "${check[@]}"
  fi
  python - "$cp" "$log" <<'PY'
import sys,json
from pathlib import Path
state=json.loads((Path(sys.argv[1])/'trainer_state.json').read_text())
assert state['global_step']==5116,'Equal-budget optimizer-step count differs.'
assert 'SOURCE_CONTROL_BUDGET 81849 unique 80468' in Path(sys.argv[2]).read_text(),'Budget marker missing.'
PY
  for ds in arxiv cora pubmed; do
   split=val; [[ "$ds" != pubmed ]] || split=test
   LP_CKPT="$cp" DATASET="$ds" LP_SPLIT="$split" LP_ABLATION="$mode" EVAL_GPU="$eval_gpu" \
    OUTPUT_PATH="$root/predictions/${ds}_${split}_${mode}.jsonl" bash scripts/eval_lp_matched.sh
  done
  CUDA_VISIBLE_DEVICES="$eval_gpu" python eval/score_lp_binary_weighted.py --checkpoint "$cp" --seed "$seed" --ablation "$mode" --target pubmed --root "$root" --pilot 2>&1 | tee "$root/score_${mode}_pilot.log"
  CUDA_VISIBLE_DEVICES="$eval_gpu" python eval/score_lp_binary_weighted.py --checkpoint "$cp" --seed "$seed" --ablation "$mode" --target pubmed --root "$root" 2>&1 | tee -a "$root/score_${mode}.log"
 done
 python research/summarize_lp_source_control_seed.py --seed "$seed" --single-score-root "$root" --output "results/lp_source_control/seed${seed}/summary"
 done
fi
if [[ "$stage" != check ]]; then
 for seed in 43 44; do
  python research/summarize_lp_source_control_seed.py --seed "$seed" --single-score-root "results/lp_source_control/seed${seed}/arxiv_only" --output "results/lp_source_control/seed${seed}/summary"
 done
 python research/summarize_lp_source_three_seeds.py
fi
