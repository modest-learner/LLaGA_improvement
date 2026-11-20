# Table 5 配置核对与后续基线（2026-11-15）

## 核对范围

Arxiv + Pubmed LP 训练 → Cora LP 测试；HO、4-hop、SimTeG、冻结 Vicuna-7B-v1.5-16k，仅训练 Projector。四组均为 seed=42、1 epoch、有效 batch=16（双卡×每卡4×累积2）、102773 条训练样本、6423 optimizer steps。

四组 trainer_state 均记录完整训练；日志均有两个 rank 正常退出，无 Traceback；最终权重均存在且无 NaN/Inf。680 条测试节点对、标签和顺序一致，均无空答案。

| Projector | LR | 官方正确数/680 | 官方 Accuracy | 最后1000步平均训练loss | FP | FN |
|---|---|---|---|---|---|---|
| linear | 2e-3 | 597 | 87.79% | 0.0258523 | 51 | 32 |
| linear | 2e-5 | 579 | 85.15% | 0.0354105 | 75 | 26 |
| 2-layer-mlp | 2e-3 | 571 | 83.97% | 0.0319927 | 85 | 24 |
| 2-layer-mlp | 2e-5 | 515 | 75.74% | 0.0471072 | 150 | 15 |

结果位置：results/table5/cora_HO_{linear,mlp2}_lr{2e-3,2e-5}_seed42_bs16_lp.jsonl。
检查点位置：checkpoints/table5/llaga-HO-arxiv-pubmed-lp-{linear,mlp2}-lr{2e-3,2e-5}-seed42-bs16。

官方 eval/eval_res.py 的 LP 判别是大小写敏感的 yes 子串匹配。Linear/2e-5 有两个正样本回答为 “Yes, the two center nodes in the subgraphs should connect to each other.”，被算作负类。仅忽略大小写的辅助核对为 581/680=85.44%，不改变排序。本次没有修改评分代码或预测文件。

额外已有 ND 单种子结果：linear/2e-3 为605/680=88.97%；mlp2/2e-3 为565/680=83.09%。这些仍是单次训练与采样解码结果。

## 固定配置决定

后续 Table 5 与本地机制分析主基线固定为 HO、4-hop、linear、LR=2e-3、SimTeG、Vicuna-7B-v1.5-16k、1 epoch、有效batch=16、warmup_ratio=0.03、cosine scheduler、weight_decay=0、BF16、DeepSpeed ZeRO-2。

理由：采用公开脚本默认架构和学习率，有明确实现依据、参数量较少，本轮训练后期loss与目标图表现也支持它作为可用工作基线。它不代表已证实作者 Table 5 的原始配方，也不代表普遍最优。保留 MLP/2e-5 为论文文字配置对照；不再扩大本轮架构和学习率搜索。

已经观察了 Cora 四组测试结果，须将这轮如实报告为探索性配置核对。多种子重训不会消除目标测试集参与决策的影响。未来方法改动和超参数选择使用源图验证协议；在尚未用于选择配置的迁移目标上做确认。

该决定不修改官方 checkpoint 架构，不意味着源图节点描述训练（Table 8）也已验证了相同配置。

## 下一步

1. 继续完成 Table 1 已启动的官方评估。检查时 Classification Expert ND 六项已输出完整样本数，HO Arxiv NC 尚未完整，未发现 summary.csv。
2. 对主基线已有 seed42 权重补两次全量推理，使用独立输出文件，估计当前 temperature=0.2、do_sample=True 的生成波动。不要将推理重复数当作训练种子数。
3. 主基线补 seed43、44 两次独立训练；数据划分、训练样本、超参数固定。每个种子采用一致的推理重复次数。以每个训练种子的推理平均分为单位，报告三个训练种子间的均值和样本标准差，并另报生成波动。约需两次HO训练；不自动启动训练。
4. 固定官方 Classification Expert 权重开展 Table 6 推理期扰动分析。论文 NONE 行需要独立训练；跨图结论应在本地源图训练权重上另行验证。
5. Table 8 单独以源图 nd（节点描述）任务训练，再比较 E、E+T、T、中心embedding+T。原生 nctext 仍含图embedding，不能充当T-only；文本组需真正移除图token通道。先核查文本特征训练的目标图标签使用情况。

单种子结果支持当前配置决策，尚不足以声称架构/学习率优势具有统计稳定性。若要将这一比较作为论文结论，须对参与比较的配置补匹配训练种子。
