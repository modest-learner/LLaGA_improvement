# Table 8：源图节点描述训练与 Cora 零样本分类

更新：2026-11-15。使用原有 `scripts/train_deepspeed.sh`、`scripts/eval.sh`、`eval/eval_res.py`，不另建训练/推理入口。本阶段只做论文 Table 8 的 Cora 部分；Products 部分需要完整数据及 Arxiv+Pubmed+Cora 的另一轮源图训练。

## 固定方案与边界

- 从 Vicuna 基座初始化新的 Projector，以 Arxiv+Pubmed 的 **nd（节点描述任务）** 训练，不能使用 Table 5 的 LP Projector 代替。小写任务 `nd` 与大写图模板 `ND` 不同；本次模板仍为 **HO**。
- 沿用已确定的工作配置：HO、4-hop、linear、SimTeG、LR=2e-3、1 epoch、seed42、双卡×每卡4×梯度累积2=有效batch16、BF16、ZeRO-2。Table 5 上选定的配置不意味着 Table 8 已证明最优，也未证实是作者 Table 8 的全部原始超参数。
- 作者 nd 的回答包含**源图类别名及标题**。本地检查 Arxiv 90941 条、Pubmed 11830 条，共102771条，ID唯一且全部来自相应 train_mask；不包含 Cora 节点。
- 所有分类组使用同一 Cora 测试集542节点、同一冻结 Vicuna 和类别候选列表。T 使用 `raw_texts[id][:2000]`，与原始 nctext 一致，不额外拼入该节点标签。
- SimTeG 特征的目标标签使用情况仍需追溯；这些实验属于使用现有特征的复现线，暂不能作为整个管线无目标标签监督的证明。

## 组别

| 输出代号 | 信息与提示 | 是否使用新的 nd Projector |
|---|---|---|
| E_official | 图embedding序列，原始 nc 提示 | 是 |
| E | 图embedding序列，与 E+T 对齐类别列表和输出要求 | 是 |
| ET | 图embedding序列+中心原始文本，原始 nctext 提示 | 是 |
| T | 中心原始文本，删除图输入提示，保留 ET 的文本与分类指令 | 否，直接加载冻结 Vicuna |

这是三种信息条件，加一组提示措辞控制，无需额外训练。原始 nc 和 nctext 的提示不完全一致：用 E_official 与 ET 对照原文，用 E 与 ET/T 做后续信息贡献分析。E 与 ET 仍须分别描述有无文本，T 删除无意义的图前缀；“matched”不意味着三个完整字符串相同。

## 修改与兼容

`eval/eval_pretrain.py` 新增 `--input_mode text_only`，只允许与 `--task nctext` 使用。它加载普通 Vicuna，跳过 Projector、图特征和图张量。新增 `--prompt_style matched` 用于统一 NC 与 nctext 的类别列表及输出要求。新增可选的 `--seed`（逐样本推理随机种子）、`--greedy`、`--max_new_tokens`、`--fail_on_error`，在输出的 `eval_config` 中记录设置。

`scripts/eval.sh` 新增 `INPUT_MODE`、`PROMPT_STYLE` 环境变量，并透传附加 CLI 参数。默认仍为 graph + official；未指定 seed 时保持原采样行为，temperature=0.2、max_new_tokens=1024。原评分器与训练算法没有因本次 Table 8 改动而更改。

已在 GPU3 用真实权重验证：修改前后 HO 的 nc/lp/nctext 各2个样本，在相同 RNG 状态下提示、图输入记录、预测和标签一致；T-only 在不存在的 Projector/embedding 路径下仍正常预测，模型没有 mm_projector，且特征加载函数设为报错也没有被调用；同推理种子重复2个样本输出一致。匹配提示的类别列表/输出要求一致。训练数据抽查每个源图16条，监督token非空，HO输入为(1,5,2432)，数值有限。以上为小样本回归检查，不代表已完成 Table 8 全量实验。

另外通过原始 train_deepspeed.sh 在 GPU3 单卡执行了2步 nd 训练（试跑microbatch=1），loss为4.6875、3.625，进程正常退出并保存Projector。随后通过 scripts/eval.sh 成功加载该权重，完成1条 E+T 推理。试跑权重在 /tmp，未启动下方双卡全量训练，也不用于正式结果。检查日志存于 `logs/table8/validation_20261115/`。

## 1. 准备

以下命令在仓库根目录执行。GPU0、3须先确认可用；不要覆盖正在运行的 Table 1 进程。DeepSpeed 用 GPU_INCLUDE 指定物理GPU，评估用 CUDA_VISIBLE_DEVICES 单卡即可。若更换GPU，请同步修改训练GPU_INCLUDE和评估EVAL_GPU。

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
nvidia-smi
mkdir -p logs/table8 results/table8/HO_linear_simteg_trainseed42
```

## 2. 完整训练：Arxiv+Pubmed nd → Cora NC

这是新的训练，不加载官方 general/classification expert 或 Table 5 的 LP 权重。不要在已用目录中改配置重跑；已有非空目录时训练脚本会拒绝继续。当前只先跑一个训练种子，确认实验行为后再决定补43/44，T组不随训练种子重复训练。

```bash
(
  set -euo pipefail
  unset CUDA_VISIBLE_DEVICES
  MODEL_BASE=./models/vicuna-7b-v1.5-16k \
  GPU_INCLUDE=localhost:0,3 \
  LLAGA_MASTER_PORT=61002 \
  LEARNING_RATE=2e-3 \
  GRADIENT_ACCUMULATION_STEPS=2 \
  OUTPUT_DIR=./checkpoints/table8/llaga-HO-arxiv-pubmed-nd-linear-simteg-lr2e-3-seed42-bs16 \
  bash scripts/train_deepspeed.sh \
    vicuna_4hop nd arxiv-pubmed 4 simteg --seed 42 \
    2>&1 | tee logs/table8/train_HO_nd_linear_simteg_lr2e-3_seed42_bs16.log

  test -s checkpoints/table8/llaga-HO-arxiv-pubmed-nd-linear-simteg-lr2e-3-seed42-bs16/mm_projector.bin
  test -s checkpoints/table8/llaga-HO-arxiv-pubmed-nd-linear-simteg-lr2e-3-seed42-bs16/config.json
)
```

`vicuna_4hop`选择HO+linear；不要误改为`vicuna`（ND），也不要把任务nd写成nc/lp。预计优化器步数约6423；节点描述比LP回答长，训练耗时以本轮开始后的进度条估计，不直接套用LP耗时。

## 3. 评估：四行、每行三次独立推理

确认上一步成功后执行。推理 seed=0/1/2 与训练 seed42 是不同概念。保留官方采样设置；未加 `--greedy`。每个输出文件应为542行。所有组最后均用 `eval_res.py --task nc` 评分，包括推理任务名nctext的组。

```bash
(
  set -euo pipefail
  EVAL_GPU=3
  CKPT=./checkpoints/table8/llaga-HO-arxiv-pubmed-nd-linear-simteg-lr2e-3-seed42-bs16
  OUT=./results/table8/HO_linear_simteg_trainseed42
  mkdir -p "$OUT" logs/table8
  test -s "$CKPT/mm_projector.bin"

  for infer_seed in 0 1 2; do
    for group in E_official E ET T; do
      case "$group" in
        E_official) task=nc;     input_mode=graph;     prompt_style=official ;;
        E)          task=nc;     input_mode=graph;     prompt_style=matched ;;
        ET)         task=nctext; input_mode=graph;     prompt_style=matched ;;
        T)          task=nctext; input_mode=text_only; prompt_style=matched ;;
      esac
      result="$OUT/cora_${group}_inferseed${infer_seed}.jsonl"

      CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
      MODEL_BASE=./models/vicuna-7b-v1.5-16k \
      MODEL_PATH="$CKPT" \
      DATASET=cora TASK="$task" TEMPLATE=HO USE_HOP=4 EMB=simteg \
      INPUT_MODE="$input_mode" PROMPT_STYLE="$prompt_style" \
      OUTPUT_PATH="$result" \
      bash scripts/eval.sh --seed "$infer_seed" --fail_on_error \
        2>&1 | tee "logs/table8/cora_${group}_trainseed42_inferseed${infer_seed}.log"

      python eval/eval_res.py --dataset cora --task nc --res_path "$result" \
        | tee "$OUT/cora_${group}_inferseed${infer_seed}.score.txt"
    done
  done
)
```

不要把旧参数的结果放进上述目录。同名完整结果会被跳过，部分结果按行数续跑；改变条件或种子必须更换文件名。逐样本种子使续跑不依赖此前生成了多少条，但不保证不同硬件/依赖版本之间逐位一致。输出异常时 fail_on_error 终止；空回答或无效类别仍应统计并保留，不能只删除错误样本。

## 4. 完整性与精确统计

原评分器仅打印4位小数。以下命令按同一个 Cora 官方匹配规则算精确正确数、三次推理均值、样本标准差，并检查542个唯一ID与测试集相符；不修改原预测文件。std仅反映一次训练下的生成波动，不是跨训练随机种子的std。

```bash
python - <<'PY'
import csv, json, statistics
from pathlib import Path
import torch

root = Path('results/table8/HO_linear_simteg_trainseed42')
data = torch.load('dataset/cora/processed_data.pt')
labels = [s.split('_')[0].strip().lower() for s in data.label_texts]
test = [json.loads(s) for s in Path('dataset/cora/sampled_2_10_test.jsonl').read_text().splitlines()]
expected = [r['id'] for r in test]
assert len(expected) == len(set(expected)) == 542
summary = []
for group in ['E_official', 'E', 'ET', 'T']:
    accuracies = []
    for seed in [0, 1, 2]:
        path = root / f'cora_{group}_inferseed{seed}.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert [r['question_id'] for r in rows] == expected, path
        assert all(r['eval_config']['seed'] == seed for r in rows), path
        correct = empty = invalid = 0
        for row in rows:
            answer = row['text'].strip().lower()
            hits = [label in answer for label in labels]
            empty += not bool(answer)
            invalid += sum(hits) != 1
            correct += hits[int(data.y[row['question_id']])] and sum(hits) == 1
        acc = 100 * correct / len(rows)
        accuracies.append(acc)
        summary.append(dict(group=group, inference_seed=seed, n=len(rows), correct=correct,
                            accuracy_pct=acc, empty=empty, invalid=invalid))
        print(f'{group:10s} seed={seed}: {correct}/542 = {acc:.6f}%, empty={empty}, invalid={invalid}')
    print(f'  inference mean ± sample std: {statistics.mean(accuracies):.6f} ± {statistics.stdev(accuracies):.6f} percentage points')
with (root / 'summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
PY
```

## 如何解释及下一阶段

1. E_official vs ET：原始 Table 8 的两种提示条件。E_official vs E：提示措辞效应。
2. ET vs T：在已有中心文本的条件下，增加训练后的图embedding通道带来的净变化。这里同时包含中心embedding、邻域聚合、特征监督来源和Projector的影响，不能直接称为“纯结构贡献”。
3. ET vs E：提示要求对齐后，增加中心原始文本的净变化。记录整体准确率、类别指标、ET/T错误重合、无效输出比例；按节点比较“ET正确/T错误”和“ET错误/T正确”。
4. 后续补中心embedding+T、邻域打乱/重连等对照，区分中心语义与邻域贡献；若作为训练设置改变的结论，需要匹配重训，区分训练期消融与推理期扰动。
5. 干净特征线：选固定版本、未在这些目标图标签上微调的预训练文本编码器，冻结参数，仅从原始文本生成所有图的新特征；按现有传播公式重新生成1–4hop缓存，另起名字保存，并按新输入维度重新训练nd Projector。原SimTeG与新特征分目录/命名保存，使用相同源训练集和Cora测试集比较。不能把旧SimTeG向量再映射一次就称为消除了标签信息，也不能只换评估EMB却复用旧Projector。

可选模型示例：[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)（384维）。目前未下载或实现这条特征生成线，等本轮对照结果后开展。该方案能控制直接的目标图标签微调通道；不能据此保证预训练语料从未含相关论文或其它信息来源完全没有污染。当前SimTeG特征是否具体使用了Cora标签仍需查明，不先作事实断言。
