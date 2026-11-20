# 单源/多源 × Full/Center-repeat，seed42

先阅读 results/lp_weight_control_seed43/summary/seed_configuration_audit.json：语义/随机加权分别比较seed42/43，启动参数只差seed、物理GPU、端口、输出目录。历史清单中的训练核心、加权代码、权重清单哈希与当前文件一致。没有历史完整环境快照，不能声称除seed外所有底层数值条件完全相同。

新增训练为Arxiv-only Full和Center-repeat，复用Arxiv+Cora的两个seed42模型与连续评分。全部Arxiv训练80468条，再用固定选择种子20261120无放回选1381条重复一次，总81849次呈现，5116优化步。相同loss、冻结BERT与Vicuna、HO4 Linear、lr2e-3、有效batch16、1epoch。补齐样本清单在results/lp_source_control/budget.json。所有模型阈值比较仅使用Arxiv验证集；Cora对单源模型仅为辅助迁移评估。

双卡预计4–6小时，单卡预计6–9小时，共享显卡会更久。主入口先运行CPU数据校验和已有三种子错误定位，然后顺序Full训练/评估、Center训练/评估、汇总。尚未启动GPU训练。

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
(
  set -euo pipefail
  unset CUDA_VISIBLE_DEVICES LIMIT
  STAGE=all GPU_INCLUDE=localhost:1,2 EVAL_GPU=2 MICRO_BATCH_SIZE=4 LLAGA_MASTER_PORT=61017 \
    bash scripts/run_lp_source_control.sh
)
```

单卡将GPU_INCLUDE改为localhost:2；脚本自动把梯度累积从2改为4，保持有效batch16。一次只运行一个入口。物理1号卡如仍被占用，可使用空闲0,2双卡（设置localhost:0,2）。

完成模型会验证后复用；完整预测跳过、部分预测/连续评分按原配置续跑。中断训练不支持优化器续跑，不完整模型目录会停止并要求检查。不得手动把不完整checkpoint当完成结果。

输出：results/lp_source_control/summary/REPORT.md、comparison.csv；diagnostics/pairs.csv、strata.csv；logs/lp_source_control/train_{full,center_repeat}.log；checkpoints/lp_source_control/。

验证：shell语法；真实数据CPU加载与collator、重复数量/唯一性/中心输入一致性；真实已有分数端到端24行汇总、相同输入产生相同指标；完整CPU入口预检。尚未进行新入口的GPU训练测试。
