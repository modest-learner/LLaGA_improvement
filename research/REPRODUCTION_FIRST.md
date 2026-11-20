# 第一阶段：先复现作者指标

## 2026-11-15：配置决定更新

四组 HO 对照已核对完成。后续本地主基线固定为 **HO + linear + lr=2e-3**；两层 MLP/lr=2e-5 保留为论文文字配置对照。结果、评分大小写问题、决策边界及后续步骤见 [TABLE5_CONFIGURATION_DECISION.md](TABLE5_CONFIGURATION_DECISION.md)。该决定替代下文历史记录中将两层 MLP 作为本地主配置的安排；官方权重评估仍采用各自原有架构。

## 2026-11-14：当前结果与 Table 5 配置

资源约束更新：用户最多使用两张 GPU 同时训练。后续采用 `GPU_INCLUDE=localhost:0,1`、每卡 batch 4、`GRADIENT_ACCUMULATION_STEPS=2`，有效 batch 为16。作者 DeepSpeed 脚本的默认 GPU 列表现为0、1；累积步数继续由命令显式设置。原 smoke 使用两卡、每卡4、累积1，有效batch为8，只用于检查流程。

Projector 配置说明更新：论文正文称其为 simple MLP，公开 HO/ND General 权重均为 `2-layer-mlp`。按论文的 MLP 描述开展主实验时，采用作者原脚本的 `vicuna_4hop_2layer`（HO）和 `vicuna_2layer`（ND）分支，另存 `mlp2` checkpoint/结果。此前给出的 `vicuna_4hop`/`vicuna` 是发布脚本默认的单层线性配置，保留作脚本对照；已完成的 linear smoke 只验证该架构的训练流程。Table 5 没有单独说明深度，公开 General 的两层结构不能独立证明 Table 5 使用完全相同结构。下文旧的 linear 命令仍表示脚本默认对照，不应与两层 MLP 主实验混写。

- 已核对 `official_ho_general`：Cora NC 542 条、LP 680 条，ID 顺序完整且无空输出；用户实测为 NC 86.90%、LP 83.38%。这来自公开 HO General 权重，属于 Table 1 推理复现。
- 已读取项目根目录用户上传 PDF：Table 5 的 Arxiv+Pubmed → Cora 为 ND 86.47%、HO 87.35%；Products 分支为 ND 92.65%、HO 92.99%。
- 已查询作者公开模型列表，没有找到上述源图 LP-only 专用权重；公开 General/Classification Expert 不能冒充 Table 5 留图模型。
- 先评估公开 HO/ND General，再用作者 `scripts/train_deepspeed.sh` 从随机 Projector 开始训练 `lp`、`arxiv-pubmed`，不使用 Cora 训练样本或公开 General Projector 初始化。Products 分支延后。
- 当前四张 3090 的 NVML 均正常且各约 24 GB 空闲；正式多卡任务由用户执行，启动前重新确认资源。
- 发布训练脚本默认 `linear` Projector、学习率 `2e-3`；PDF 正文写 `2e-5`，没有单独给出 Table 5 Projector 深度。因此先运行发布脚本配置，必要时将 `LEARNING_RATE=2e-5` 作为明确区分的正文配置分别报告；不能声称完整原始训练配方已确认。
- 原有脚本现在支持环境变量覆盖，默认算法参数不变：评估 `MODEL_PATH/MODEL_BASE/DATASET/TASK/EMB/USE_HOP/SAMPLE_SIZE/TEMPLATE/OUTPUT_PATH`；训练 `MODEL_BASE/LEARNING_RATE/OUTPUT_DIR/GRADIENT_ACCUMULATION_STEPS`；DeepSpeed 另支持 `GPU_INCLUDE/LLAGA_MASTER_PORT`。第六个及后续训练脚本参数转发给原训练入口，例如 `--max_steps 2 --seed 42`。
- 修正 `train_deepspeed.sh` 将 embedding 从第六参数读取的问题，现在按 README 使用第五参数；训练按 `--seed` 在 Projector 初始化与数据 shuffle 前设种子，使不同 rank 的样本列表一致。
- 已检查脚本语法、无训练的参数捕获、源图各32条LP样本的有效监督token及输入文件。尚未执行正式训练，也未实测新训练 checkpoint。

Table 5 发布脚本配置的训练命令（两卡，每卡4条、累积2步，有效batch=16）：

```bash
MODEL_BASE=./models/vicuna-7b-v1.5-16k \
GPU_INCLUDE=localhost:0,1 \
GRADIENT_ACCUMULATION_STEPS=2 \
OUTPUT_DIR=./checkpoints/table5/llaga-HO-arxiv-pubmed-lp-linear-lr2e-3-seed42 \
LEARNING_RATE=2e-3 \
bash scripts/train_deepspeed.sh vicuna_4hop lp arxiv-pubmed 4 simteg --seed 42

MODEL_BASE=./models/vicuna-7b-v1.5-16k \
GPU_INCLUDE=localhost:0,1 \
GRADIENT_ACCUMULATION_STEPS=2 \
OUTPUT_DIR=./checkpoints/table5/llaga-ND-arxiv-pubmed-lp-linear-lr2e-3-seed42 \
LEARNING_RATE=2e-3 \
bash scripts/train_deepspeed.sh vicuna lp arxiv-pubmed 4 simteg --seed 42
```

训练完成后将对应 `OUTPUT_DIR` 传给原 `scripts/eval.sh` 的 `MODEL_PATH`，设置 `DATASET=cora TASK=lp`；HO 为 `TEMPLATE=HO USE_HOP=4`，ND 为 `TEMPLATE=ND USE_HOP=2`，使用独立的 `OUTPUT_PATH`。随后使用原始 `eval/eval_res.py --dataset cora --task lp --res_path ...` 评分。

**执行入口修订：按用户要求，直接使用 README 与作者原有的 `scripts/eval.sh`。下文早先提供的 `research/reproduce_official.py` 命令不再作为推荐运行入口。**

当前 `scripts/eval.sh` 已填写本地 Vicuna 与 HO General Projector 路径，设置 Cora、NC、SimTeG、HO、4 hop。GPU 恢复且显存充足后，在项目根目录执行：

```bash
conda activate llaga
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh
python eval/eval_res.py --dataset cora --task nc --res_path ./results/official_ho_general/cora_nc.jsonl
```

LP 实验在同一个作者脚本中把 `task="nc"` 改为 `task="lp"`，把输出路径改为 `./results/official_ho_general/cora_lp.jsonl`，再执行脚本与对应的 `--task lp` 评分命令。NC 与 LP 使用不同输出文件。以上仅填写官方脚本配置，推理及评分实现沿用作者原始代码。

当前优先级：官方权重推理复现 → 原始协议从头训练复现 → 清洁协议对照 → 机制研究。
本阶段使用原始数据、SimTeG、官方提示与评分。已发现的数据疑点保留记录，暂不清洗或改写它们。

## 1. 当前已知状态

- 核心环境导入、CUDA/FlashAttention/PyG 小规模测试此前已通过，但尚未加载完整 Vicuna 或产生正式预测结果。
- Cora/Pubmed 的官方 HO+SimTeG 所需文件在当前清单中齐全；普通 SBERT 多跳缓存缺失与本轮实验无关。
- 默认缓存只发现 OPT-2.7B；下面命令准备独立的 Vicuna 和匹配 Projector。
- 2026-11-13 本轮检查：GPU 0/1 各约 1.6 GiB 空闲，GPU 2 约 3.4 GiB，GPU 3 的 NVML 查询报 Unknown Error。状态会变化；执行前重新检查。先申请一张正常且最好有约 20 GiB 空闲显存的卡；不终止他人进程，不直接重置 GPU。
- Products 不完整不影响使用官方多图模型在 Cora/Pubmed 上推理；从头重现包含 Products 的完整 General Model 训练则需要它。

## 2. 对齐哪一行指标

使用 `Runjin/llaga-vicuna-7b-simteg-HO-general_model-2-layer-mlp-projector`，基座 `lmsys/vicuna-7b-v1.5-16k`，HO，`use_hop=4`，SimTeG，`conv_mode=v1`。

ICML 终稿 Table 1 General Model / LLaGA-HO-7B：

| 数据集 | NC Accuracy | LP Accuracy |
|---|---:|---:|
| Cora | 87.82% | 84.41% |
| Pubmed | 94.45% | 89.81% |
| Arxiv | 75.01% | 92.04% |

来源：[PMLR 论文页面](https://proceedings.mlr.press/v235/chen24bh.html)、[终稿 PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/chen24bh/chen24bh.pdf)。上述表格已与下载的终稿核对。早期 arXiv v2 的部分 LP 数值不同，勿混用。

这验证官方已训练模型的推理与评估链路。它不是 Arxiv+Pubmed → Cora 的零样本结果，也不等于已复现训练过程。公开权重名称与配置对应这一模型类型，但没有完整训练日志，不能保证某个随机种子逐样本精确恢复论文分数。

## 3. 下载固定版本

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
python research/reproduce_official.py download
```

将下载到当前项目的 `models/`：

- Vicuna revision：`c8df3ca4436a3bce5c4b5877e0117032081852b4`。
- HO General Projector revision：`5fbc78bde96eadefeaaf4abd3bf8f070a194148c`。

基座权重约十几 GB，建议准备至少约 25 GB 可用磁盘空间。下载步骤不需要 GPU。使用现有 huggingface_hub，不升级环境。

现已将用户已有的 Vicuna 从 `/data2/workspace/hy/llaga/lmsys/vicuna-7b-v1.5-16k` 移至本项目 `models/vicuna-7b-v1.5-16k`，下载元数据的 revision 与上面固定版本一致。`download` 会检查并复用本地完整基座；明确只下载 Projector 可执行：

```bash
python research/reproduce_official.py download --only-projector
```

## 4. 小规模运行

先查询计划使用的单张卡。例如 GPU 0（应替换为实际获配的健康 GPU）：

```bash
nvidia-smi -i 0 --query-gpu=index,name,memory.total,memory.free --format=csv
```

准备好空闲 GPU 后：

```bash
mkdir -p results/official_ho_general
set -o pipefail
python research/reproduce_official.py run --gpu 0 --task nc --limit 32 --seed 42 \
  2>&1 | tee results/official_ho_general/cora_nc_seed42_n32.log
```

入口调用原始 `eval/eval_pretrain.py` 和 `eval/eval_res.py`。额外工作只有固定随机种子、固定路径/配置、检查文件与结果完整性、记录 manifest。原始采样设置保持 `do_sample=True`、temperature=0.2、num_beams=1、max_new_tokens=1024。固定种子有利于重复，但不保证跨硬件逐位一致。

通过条件：32 条输出、按原始顺序对应32个ID、没有空输出、无未处理异常。小样本准确率不是正式论文指标。出现空输出时，入口停止，不让原始评分器把异常空串按负例计分；预测文件保留以便排错。

## 5. 完整 Cora NC 和 LP

确认小规模成功后，在同一个已激活的终端执行：

```bash
python research/reproduce_official.py run --gpu 0 --task nc --seed 42 \
  2>&1 | tee results/official_ho_general/cora_nc_seed42_full.log

python research/reproduce_official.py run --gpu 0 --task lp --seed 42 \
  2>&1 | tee results/official_ho_general/cora_lp_seed42_full.log
```

NC 应完整覆盖当前数据的542条，LP应完整覆盖680条。预测、模型/代码版本、参数和完整率保存在 `results/official_ho_general/`。脚本使用原始评分器输出 accuracy；打印的0.8782代表87.82%。

预测文件已存在时入口会拒绝重跑，避免作者原始脚本的自动续写混用不同配置。重试用新目录，例如 `--output-dir results/official_ho_general_retry1`；不需要删除旧实验。

## 6. 结果核对和复跑

保存论文分数、实测分数、差值（百分点）、样本数、空输出数和日志。论文结果为比较目标，不预先保证精确一致。

- 先排查覆盖不全、空输出、模型/特征/模板不匹配、论文版本不匹配。
- 出现明显差距时检查原始预测及加载警告。尤其需确认 Projector 真正加载，不是随机初始化；检查 checkpoint 的 key/shape 和实际加载参数。输出正常但明显随机时优先查这一点。
- 使用 seed 42/43/44 重复推理以测量解码随机性；这是推理随机性，不是三个独立训练种子。比较均值、范围和逐样本变化，不挑最优种子作为“复现成功”。
- “相差1个百分点”可以用作排查提示，但不是普适验收标准；Cora NC每个样本约0.1845个百分点，LP每个样本约0.1471个百分点。误差应结合推理随机性、样本数和配置差异解释。
- LP 同时人工抽查非标准、含糊答案，记录原始评分规则的影响。正式复现仍报告原始分数；改进评分单列，不覆盖原始结果。

随后对 Pubmed 重复：

```bash
python research/reproduce_official.py run --gpu 0 --dataset pubmed --task nc --seed 42
python research/reproduce_official.py run --gpu 0 --dataset pubmed --task lp --seed 42
```

## 7. 推理复现通过后，才开始从头训练复现

先选择无需 Products 的 `Arxiv+Pubmed → Cora` LP。终稿 Table 5 的目标为 ND 86.47%、HO 87.35%；不能用 General Model 的84.41%或旧版论文90.44%混为对照。

训练只使用源图的LP任务；保留原始SimTeG和原始数据协议，Projector随机初始化、LLM冻结。训练配置需要单独确定，因为现有材料确有差异：终稿正文学习率写2e-5，而 `scripts/train.sh` 写2e-3；README和脚本默认模型类型/Projector深度也不能直接视为表5的完整训练配方。把“按发布脚本”与“按论文正文”标为不同配置，用源域验证选择并报告，不能靠Cora测试分数反推超参数。

从头训练完整四图 General Model需要Products；当前先重现官方权重及两源跨图LP即可逐步验证系统。只有训练复现结果可信后，再进入 `RESEARCH_PLAN.md` 的清洁协议和机制研究。

## 当前执行状态

已完成：官方模型配置/版本核对、终稿指标核对、入口脚本语法及命令参数检查。尚未执行大型权重下载、完整模型推理或训练；当前没有可报告的实测复现准确率。
