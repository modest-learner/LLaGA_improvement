# LP 初筛、审计和冻结编码器命令

本轮准备的入口继续调用作者 scripts/eval.sh 与 scripts/train_deepspeed.sh。未启动全量推理、BERT下载或Projector训练。五组推理已各验证2对；诊断Full与默认原路径在这2对上的提示、图ID、回答逐字一致；CPU测试及Cora真实清洁划分检查通过。另用临时本地微型BERT（768维、单层、合成词表）通过了编码→masked pooling→HO→哈希/划分校验端到端检查，不代表官方BERT的实验性能。新BERT全量编码/训练尚未运行，不能声称已有新协议结果。

## A. 当前先执行：审计与旧权重五组初筛

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
(
  set -euo pipefail
  unset LIMIT
  bash scripts/run_lp_audit.sh
  EVAL_GPU=0 bash scripts/run_lp_input_screen.sh
)
```

单卡0顺序运行五组，不重新训练。Full/Center-only/Center-repeat/Hop-mismatch使用原Table5 HO-linear/lr2e-3/seed42；Endpoint-text直接使用原Vicuna，无Projector。每组Cora原680条，用do_sample=False,num_beams=1；seed0为推理种子，错配种子42，文本每端点最多2000字符。该贪心Full不是历史87.17%的同一协议数字。保留原官方Accuracy，另报固定词法yes/no解析的Overall/Valid/Conditional与有效回答混淆矩阵；词法解析不自动判断复杂否定语义。

输出：
- results/lp_protocol_audit/full_v1/data.json
- results/lp_protocol_audit/full_v1/ho_cache.json
- results/lp_protocol_audit/full_v1/existing_predictions/metrics.csv
- results/lp_input_screen/HO_linear_simteg_trainseed42_greedy/cora_{full,center_only,center_repeat,hop_mismatch,endpoint_text}.jsonl
- 同目录audit/metrics.csv、invalid_answers.jsonl与每组official.score.txt

数据/缓存审计在CPU上运行，对三图三个SimTeG组件逐一检查与哈希；可能耗时，不能以无输出判断卡死。缓存审计检验的是“当前完整图+对称归一化+自环”这一具体假设，不证明实际缓存来源。不吻合也不能直接断言标签泄露。SimTeG训练标签来源仍需上游权重和生成记录。

断点重跑相同命令时检查已写预测配置及顺序，再续写；换参数/代码需新OUTPUT_ROOT，不能往旧文件混写。每个采样重复作为原复现，不能当独立训练种子。

## B. 建立新LP视图与统一冻结BERT特征

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
(
  set -euo pipefail
  mkdir -p logs/lp_clean
  python research/prepare_clean_lp.py \
    --stage graph \
    --output-root dataset_research/lp_bert_disjoint_v1 \
    2>&1 | tee logs/lp_clean/prepare_graph.log
  python research/verify_clean_lp.py \
    --root dataset_research/lp_bert_disjoint_v1 --graph-only
  CUDA_VISIBLE_DEVICES=0 python research/prepare_clean_lp.py \
    --stage features \
    --output-root dataset_research/lp_bert_disjoint_v1 \
    --encoder google-bert/bert-base-uncased \
    --revision 86b5e0934494bd15c9632b12f734a8a67f723594 \
    --device cuda:0 --batch-size 16 --max-length 256 \
    2>&1 | tee logs/lp_clean/prepare_bert.log
  python research/verify_clean_lp.py \
    --root dataset_research/lp_bert_disjoint_v1
)
```

仅首次需要下载固定revision的BERT权重和tokenizer（缓存models/encoder_cache）；Vicuna/官方Projector/Products不会因此重复下载。保存encoder_lock.json锁定三图相同模型版本。不微调BERT，只用title+abstract、256token截断、排除special/pad的masked mean pooling、无L2归一化；生成768维bert_x.pt和1–4hop特征。

独立目录中修正候选、规范无向对、去重、自连接与已知伪负例检查，原始训练候选中分层划出10%源验证。三个split所有监督正边整体从同一上下文图移除（包括源训练正查询），再计算HO；因此训练查询在自己的输入中也不可见。源验证仅用于源选择，不把Cora标签用于调参。构图及负例核验使用GT作为离线benchmark制备，不是推理输入。保持原测试剩余候选及其正负比；尚未增加难负例集，也尚未形成固定预算单源/多源比较。

在临时目录的真实Cora测试得到651条清洁测试候选（原680），这不是重新运行的模型指标。清洁protocol与旧SimTeG不同；未来新旧分差不能单独归因BERT或新方法。

## C. 新协议准备后才运行：两组匹配LP重训

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
(
  set -euo pipefail
  unset CUDA_VISIBLE_DEVICES LIMIT
  for mode in full center_repeat; do
    GPU_INCLUDE=localhost:0,3 EVAL_GPU=0 LLAGA_MASTER_PORT=61007 \
    TRAIN_SEED=42 SOURCES=arxiv-pubmed TRAIN_ABLATION="$mode" \
    bash scripts/run_clean_lp_baseline.sh
  done
)
```

两个训练顺序执行，都为HO4/linear/lr2e-3/1epoch/有效batch16；冻结LLM，仅训练新Projector，输入为共享BERT特征。沿用原作者训练入口，额外传data_root/embedding_dim/graph_input_ablation。训练后每模型顺序跑Cora五组评估。两卡都空闲时再提交；不要将已有占用的设备硬塞进训练。脚本对已有checkpoint拒绝覆盖，退出时需检查日志而不是直接换名字反复重训。

Full→Full 与 Center-repeat→Center-repeat才是匹配训练比较；另一格仅作输入干预诊断。每个模式共用全部已清洁源监督，因此这是全量源数据基线，还不是固定监督预算下的单源/多源机制矩阵。新split记录量与6423步可能不同，不检查旧固定步数。

新checkpoint：checkpoints/lp_clean/llaga-HO-arxiv-pubmed-lp-linear-bert-lr2e-3-seed42-bs16-{full,center_repeat}。
新结果：results/lp_clean/arxiv-pubmed_to_cora_trainseed42_{full,center_repeat}/screen/audit/metrics.csv。

## 当前证据边界

没有运行新的全量BERT编码或训练。CPU训练数据入口已用临时合成768维特征检查Full/Center-repeat形状及行为；这不等于新权重训练验证。排序AUC/AP需要连续分数，本轮脚本不会从自由生成硬标签伪造；固定候选打分和难负例将在源验证协议确定后增加。先完成A核查结果，再按B/C重建研究基线，不把它们称为复现原论文同一设置。

BERT来源：https://huggingface.co/google-bert/bert-base-uncased 。
