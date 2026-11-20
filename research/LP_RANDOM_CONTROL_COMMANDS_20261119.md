# Pubmed方向随机加权对照

固定设置：Arxiv+Cora→Pubmed，Full HO4、Linear、BERT，训练seed42，lr2e-3，batch16，1 epoch。随机选择种子20261119；各源高权重负例数量、权重数值、负例总权重与语义加权完全一致。仅选择哪些负例被加权不同。

完整命令（预计3–4.5小时；1、2双卡训练，2号单卡评估）：

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
(
  set -euo pipefail
  unset CUDA_VISIBLE_DEVICES LIMIT
  STAGE=all GPU_INCLUDE=localhost:1,2 EVAL_GPU=2 LLAGA_MASTER_PORT=61015 \
    bash scripts/run_lp_random_control.sh
)
```

脚本先做两次单GPU四步真实训练：原入口与全1权重入口。seed、batch16、学习率与样本来源相同，比较四步loss及最终Projector参数；未通过则停止，不能跳过检查继续完整实验。单GPU短程一致性不等于完整双GPU训练逐位一致性证明。检查完整时复用其结果。

检查后新增一次随机加权完整训练，随后原生成评估、候选评分、源验证阈值与三组配对汇总。不会重训原基线或语义加权组。

结果在results/lp_random_control：equivalence.json、REPORT.md、comparison.csv、negative_strata.csv、predictions、binary_scores。日志在logs/lp_random_control。

中断后重执行可复用完整训练和已完成推理；不完整训练没有优化器恢复状态，需检查目录后才能从头重跑。不要删除中断目录或覆盖日志来冒充续训。
