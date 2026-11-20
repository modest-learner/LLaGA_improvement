# LP下一轮：源验证、简单基线与匹配训练种子

本轮只运行同任务LP→LP。复用dataset_research/lp_bert_disjoint_v1与已有seed42模型。冻结BERT、Vicuna、HO4、Linear、lr2e-3、有效batch16、1epoch；不重新生成特征，不再重复纯文本/单中心/错配五组全矩阵，不启动NC或新目标方向。

## 顺序与目的

1. 已有seed42两模型在Arxiv(8939对)、Pubmed(1125对)源验证集上做匹配评估；Cora两个已完成结果跳过加载。了解源验证与目标表现，但候选难度/正负比不同，不能把全部差距归因图分布变化。
2. 同一合法可见图上建立端点BERT余弦、CN、AA、RA、PA与端点MLP。前五个不训练，只在两个源验证上选择阈值；小MLP用相同源训练候选，3个种子。阈值/MLP最佳epoch按等权源图的Balanced Accuracy选择，Cora只用于最后计分。MLP默认对称abs(xu-xv)与xu*xv，hidden256、GELU、AdamW lr1e-3、batch512、10epochs；标准化统计只在源训练拟合。这不是与LLM相同计算预算的比较，需如实报告。
3. Full和Center-repeat各补seed43/44，共4轮Projector训练。每模型评估Arxiv/Pubmed源val和Cora目标test，均匹配自己的训练输入。LLM合计18份结果（其中已有2份Cora直接复用）。源/目标两种输入模型都比较，不只补表现最好的一组。
4. 按训练种子汇总mean±sample SD（ddof1）和同seed Full减Center的成对差值。贪心推理不机械重复3次。AUC/AP仅简单基线的连续分数可用，LLM自由生成仍报告官方/辅助ACC、Balanced ACC与有效率。

## 一次性顺序执行

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
nvidia-smi

(
  set -euo pipefail
  unset CUDA_VISIBLE_DEVICES LIMIT
  STAGE=all \
  GPU_INCLUDE=localhost:0,3 \
  EVAL_GPU=0 \
  BASELINE_GPU=0 \
  LLAGA_MASTER_PORT=61008 \
  bash scripts/run_lp_next_round.sh
)
```

所有阶段依次执行，不同时启动两个LLaGA模型。训练双卡0/3，其余单卡0。训练严格两卡×micro4×accum2=16；换卡可以用其他两个空闲编号，但不能直接改为单卡而仍称相同有效batch。根据上轮实测每次训练约2h55，四轮训练本身约12小时，另加源验证、目标推理与基线时间；共享负载可改变耗时。

## 分阶段运行同样的任务

如果希望先看源验证和基线再提交长训练：

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga

STAGE=source EVAL_GPU=0 bash scripts/run_lp_next_round.sh

STAGE=baselines BASELINE_GPU=0 bash scripts/run_lp_next_round.sh

STAGE=seeds GPU_INCLUDE=localhost:0,3 EVAL_GPU=0 LLAGA_MASTER_PORT=61008 \
  bash scripts/run_lp_next_round.sh

STAGE=summary bash scripts/run_lp_next_round.sh
```

以上分段不必在一次性命令之后再运行。分段source结束时summary标为尚未完整是预期状态，因为43/44和小模型还没完成。summary阶段要求文件齐全才返回成功；完整性以summary/status.json为准。

如果source已在GPU0运行，可以在另一个终端并行启动seeds：

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
(
  unset CUDA_VISIBLE_DEVICES LIMIT
  STAGE=seeds GPU_INCLUDE=localhost:1,3 EVAL_GPU=1 LLAGA_MASTER_PORT=61008 \
    bash scripts/run_lp_next_round.sh
)
```

seeds内部依次完成四轮训练与对应评估；不要重复启动另一个seeds，也不要同时启动all。source与seeds只要使用不同显卡即可并行；baselines另选空闲卡或等source结束后使用GPU0。

## 续跑与覆盖规则

- 重复同一命令会核对数据/代码/模型版本。已完成训练跳过；已完成推理跳过GPU加载；部分JSONL在配置与顺序一致时续写。换评估卡只改EVAL_GPU。
- 不同单独阶段可以分卡并行：例如source使用GPU0，seeds使用GPU1/3。脚本按source、baselines、seeds分别加锁，同一阶段禁止重复启动；STAGE=all占用全部阶段锁。锁不负责分配显卡，需自己确保显卡没有重叠。
- 兼容修复前已启动的独立source进程，允许新seeds阶段并行；若旧进程是all，则必须等待，避免它后续也启动seeds。不要删除.runner.lock来绕过检查。
- 汇总写入会串行，正在写入的预测文件标为未完成，不计入均值。所有阶段结束后运行STAGE=summary，检查完整性。
- 原作者Projector保存不包含完整优化器状态。训练中断后若留下不完整目录，脚本停止，不冒充从原优化步恢复，也不删除旧文件。该情况需检查并用保留旧目录的方式重新开始该轮训练。已完成训练后评估中断可以直接续跑。
- 基线小MLP完成后会保存权重并复用；未完成的小MLP训练重启该种子。阈值/超参/feature recipe变化需新输出目录，不能混写结果。
- 核心eval/eval_pretrain.py与eval/lp_inputs.py本轮未改动，seed42预测的配置哈希继续匹配。

## 输出路径

- 源验证：results/lp_next_round/source_validation/seed{42,43,44}_{full,center_repeat}/
- 简单基线：results/lp_next_round/simple_baselines/metrics.csv（另存每个方法连续分数npz、MLP权重与来源配置）
- 新权重：checkpoints/lp_clean/llaga-HO-arxiv-pubmed-lp-linear-bert-lr2e-3-seed{43,44}-bs16-{full,center_repeat}/
- Cora新结果：results/lp_clean/arxiv-pubmed_to_cora_trainseed{43,44}_{full,center_repeat}/screen/
- 最终汇总：results/lp_next_round/summary/REPORT.md、llm_per_seed.csv、llm_mean_sd.csv、paired_delta_mean_sd.csv、status.json

最终需要看：Full增益是否跨种子稳定；源图也有多少邻域增益；简单端点MLP能否接近Full；FP是否主要集中在语义相似非边；结构启发式在当前候选上的强弱。之后再决定新目标训练和难负例实验，不依据Cora测试标签重选当前阈值。

## 本轮实现检查

源验证Pubmed分支用已有Full模型仅测试2对；源选择阈值算法和对称MLP特征的CPU检查通过；临时微型数据上六种基线端到端完成；旧Cora结果跳过模型加载检查通过。没有运行真实数据的全量基线、源验证或seed43/44训练。脚本继续调用原作者scripts/eval.sh和scripts/train_deepspeed.sh。
