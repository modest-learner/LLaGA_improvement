# 单源seed43/44：四组训练完整命令

依次训练Arxiv-only Full43、Center-repeat43、Full44、Center-repeat44，每组完成Arxiv/Cora验证和Pubmed测试、yes/no连续评分。Cora为单源模型的辅助迁移评估，主阈值只用Arxiv验证集。复用原单源seed42和多源seed42/43/44。

设置：HO4 Linear、冻结BERT/Vicuna、lr2e-3、有效batch16、1epoch、81849次样本呈现、5116优化步。重复清单沿用budget.json，选择种子20261120固定，不因训练种子改变。

双卡预计8–12小时，单卡预计12–18小时。显卡共享时可能更久；先等现有占用相同卡的训练结束。

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
(
  set -euo pipefail
  unset CUDA_VISIBLE_DEVICES LIMIT
  STAGE=all GPU_INCLUDE=localhost:1,2 EVAL_GPU=2 MICRO_BATCH_SIZE=4 LLAGA_MASTER_PORT=61019 \
    bash scripts/run_lp_source_control_seeds.sh
)
```

单卡使用GPU_INCLUDE=localhost:2，自动累积4；双卡自动累积2。四组顺序执行，一次只启动一个入口。

完成模型验证后复用、完整预测跳过、部分预测和连续评分按匹配配置续跑；训练不保存可恢复的优化器状态，未完成训练需要检查目录后重跑该组。

总报告：results/lp_source_control/three_seed_summary/REPORT.md；target_mean_std.csv；paired_by_seed.csv；paired_mean_std.csv。
各种子报告：results/lp_source_control/seed43/summary/REPORT.md及seed44对应路径。
训练日志：logs/lp_source_control/train_seed43_full.log等。

验证范围：shell/Python语法、seed42真实分数重算与原表一致、临时三种子统计完整性/标准差/交互量校验、18份多源评分完整性及CPU数据预检。未启动新的GPU训练。
