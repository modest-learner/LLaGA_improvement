# LP下一批：错误分层、成对融合基线、Pubmed新目标

本批仅LP→LP。先做已有结果的CPU分析，再单卡2训练小型基线，最后双卡2/3训练Arxiv+Cora→Pubmed的Full与Center-repeat（各seed42），单卡2评估源val及目标test。批内顺序执行，不同时加载两个Vicuna。仍调用作者训练与评估入口。

## 一条命令执行

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
(
  set -euo pipefail
  unset CUDA_VISIBLE_DEVICES LIMIT
  STAGE=all GPU_INCLUDE=localhost:2,3 EVAL_GPU=2 BASELINE_GPU=2 \
  LLAGA_MASTER_PORT=61009 bash scripts/run_lp_phase2.sh
)
```

## 分阶段（与all二选一）

```bash
STAGE=strata bash scripts/run_lp_phase2.sh
STAGE=baselines BASELINE_GPU=2 bash scripts/run_lp_phase2.sh
STAGE=pubmed GPU_INCLUDE=localhost:2,3 EVAL_GPU=2 LLAGA_MASTER_PORT=61009 bash scripts/run_lp_phase2.sh
STAGE=summary bash scripts/run_lp_phase2.sh
```

脚本保留批次锁，避免两个runner重复写相同权重。重复提交会复用已完成训练和小MLP，评估按完整性检查跳过或续写；不完整Projector训练因原保存缺优化器状态而停止，不能宣称能从优化步续训。不删除旧结果，不自动重启不完整训练。

## 实验设置

- 新LLM方向：Arxiv训练80468对+Cora训练1381对，共81849；目标Pubmed测试5111对。目标Pubmed标签不参与训练、早停或阈值选择。源验证为Arxiv8939对+Cora153对。
- 共享BERT与Vicuna冻结，HO4/Linear/lr2e-3/epoch1，两卡×micro4×accum2=16；输入分别full和center_repeat。原有Cora方向结果保留。
- 原Cora方向新增structure_MLP与fusion_MLP，各seed42/43/44；端点MLP参照已有结果，不重复训练。
- 新Pubmed方向重做5种启发式及endpoint/structure/fusion三种MLP，各MLP三种子；所有阈值与epoch只按Arxiv/Cora源验证等权Balanced ACC选择。
- 小MLP与此前同配方：hidden256、GELU、AdamW lr1e-3、batch512、10epochs。语义为abs(xu−xv)拼xu*xv；结构为log1p(CN,AA,RA,PA,min_degree,max_degree)六维；融合拼接两者。标准化仅源训练拟合。
- 加入结构维度会稍增参数（1542维对1536维），不是严格等参数对照。结构MLP与融合MLP均为诊断性强基线，不预先称为创新。
- 错误分层用源负例等权分位数决定语义四分位、最大端点度中位数，结合CN=0/>0联合分组；每类样本不足20标记。分析是描述性分层，不能据此声称因果识别。

## 输出

- `results/lp_phase2/error_strata/`：strata.csv、cora_pairs.csv、protocol.json。
- `results/lp_phase2/arxiv-pubmed_to_cora/pair_baselines/`：新增简单基线、连续分数、权重。
- `results/lp_phase2/arxiv-cora_to_pubmed/pair_baselines/`：新方向简单基线。
- `checkpoints/lp_phase2/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed42-bs16-{full,center_repeat}/`：新Projector。
- `results/lp_phase2/arxiv-cora_to_pubmed/seed42_{full,center_repeat}/`：两源val与目标test预测。
- `logs/lp_phase2/`：日志；`results/lp_phase2/summary/`：汇总及完整性状态。

## 如何解释旧简单基线

不经过LLaGA的五种评分：BERT余弦是端点语义相似；CN为共同邻居数；AA按1/log(邻居度)加权共同邻居；RA按1/邻居度加权；PA为端点度乘积。它们不学习评分参数，但分类阈值只在源验证选择，所以并非毫无源数据参与。

端点MLP学习两个中心BERT表示与源LP标签的关系，不读邻域，不用LLM。ACC为全体预测正确比例；Balanced ACC=(正例召回+负例召回)/2。CN/AA/RA当前Cora预测相同是阈值下的结果，连续分数与排序指标仍不同。

旧报告高语义相似负例按两个源验证负例合并后的75%分位数0.9002075分组；Cora108个高相似负例，Full每种子平均误报47个，47/108=43.52%。其余269个负例平均误报63.67，比例23.67%。这是负例内部误报率，不是ACC或所有错误的占比。此次联合分层改用源等权分位数，阈值记录在protocol.json，不覆盖旧统计。

## 已完成验证

本次已完成真实数据的CPU分层（264行分组指标、651对Cora样本）。微型数据上三种MLP端到端完成；翻转目标test标签且移除目标train/val文件后，权重、源选epoch、阈值、预测分数均不变。成对结构特征交换端点不变，目标混入训练源会被拒绝。

原作者训练数据类在Arxiv+Cora微型数据上通过Full/Center-repeat检查，两端点各5个768维图Token且监督Token非空。已有Arxiv+Pubmed完成权重仍通过兼容检查。新脚本语法与Python编译通过，尚未启动全量GPU基线或Projector训练。
