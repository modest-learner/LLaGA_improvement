# 补齐 Table 1 的 General Model 结果

核查时指定目录已完成13/24项，缺General ND的Arxiv/Pubmed NC及三个数据集LP，共5项；缺General HO的全部6项。下面沿用原脚本和采样设置，只补缺失推理，并补评分。不会重跑完整的General ND Cora NC。Classification Expert的12个完整结果不在本循环中。

运行前确认GPU3空闲；按需修改EVAL_GPU。执行期间不要修改scripts/eval.sh。此命令尚未代为启动。

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
nvidia-smi

(
  set -euo pipefail
  EVAL_GPU=3
  RUN=official_20261114_192146

  for template in ND HO; do
    if [ "$template" = ND ]; then
      hop=2
    else
      hop=4
    fi
    group="general_model_${template}"
    model="./models/llaga-vicuna-7b-simteg-${template}-general_model-2-layer-mlp-projector"
    mkdir -p "results/table1/$RUN/$group" "logs/table1/$RUN/$group"

    for dataset in arxiv pubmed cora; do
      for task in nc lp; do
        case "${dataset}_${task}" in
          arxiv_nc) expected=48603 ;;
          arxiv_lp) expected=80086 ;;
          pubmed_nc) expected=3944 ;;
          pubmed_lp) expected=5368 ;;
          cora_nc) expected=542 ;;
          cora_lp) expected=680 ;;
        esac
        result="./results/table1/$RUN/$group/${dataset}_${task}.jsonl"
        lines=0
        if [ -f "$result" ]; then
          lines=$(wc -l < "$result")
        fi
        if [ "$lines" -gt "$expected" ]; then
          echo "Unexpected sample count: $result ($lines > $expected)" >&2
          exit 1
        fi

        if [ "$lines" -lt "$expected" ]; then
          CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
          MODEL_BASE=./models/vicuna-7b-v1.5-16k \
          MODEL_PATH="$model" DATASET="$dataset" TASK="$task" \
          TEMPLATE="$template" USE_HOP="$hop" EMB=simteg \
          INPUT_MODE=graph PROMPT_STYLE=official OUTPUT_PATH="$result" \
          bash scripts/eval.sh --fail_on_error \
            2>&1 | tee -a "logs/table1/$RUN/$group/${dataset}_${task}_resume_infer.log"
        fi

        test "$(wc -l < "$result")" -eq "$expected"
        python eval/eval_res.py \
          --dataset "$dataset" --task "$task" --res_path "$result" \
          | tee "logs/table1/$RUN/$group/${dataset}_${task}_score.txt"
      done
    done
  done
)
```

保持原采样设置且不指定新推理种子。断点续跑只保证继续补足样本，不保证与一次连续运行产生逐条相同的随机预测。完成后重新核对所有ID、GT和空回答；现有audit_summary只记录补跑前状态。
