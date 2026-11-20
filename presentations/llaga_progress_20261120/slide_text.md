<!-- Slide number: 1 -->
让图语言模型在未见图上稳定预测链接
LLaGA 跨图泛化研究
从可信复现，到邻域贡献与迁移失效机制
源图
LP 监督训练
冻结 LLM
训练 Projector
未见目标图
LP 零样本测试
核心问题：邻域信息何时帮助跨图预测，何时放大错误？
研究进展汇报 · 2026年11月20日
1

### Notes:
让图语言模型在未见图上稳定预测链接

主线已明确为同任务LP→LP，节点分类作为后续扩展。汇报呈现已完成的实验与证据边界，不将诊断性尝试包装为最终方法。

数据与出处：研究进展汇报 · 2026年11月20日
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 2 -->
研究目标是跨图稳定迁移，而不只是提高一次准确率
固定任务
LP → LP
改变图分布
源图 → 未见图
固定输出
yes / no

| 研究问题 | 需要区分的因素 |
| --- | --- |
| RQ1：邻域为什么有效？ | 端点语义 vs 邻域语义与传播 |
| RQ2：迁移为何不稳定？ | 排序能力 vs 决策阈值／分数偏移 |
| RQ3：多源是否可靠？ | 训练预算 vs 源图组成与多样性 |

研究问题与范围界定
2

### Notes:
研究目标是跨图稳定迁移，而不只是提高一次准确率

同任务LP固定输出空间yes/no，避免把节点描述到分类的任务变化、类别空间变化与图分布变化混为一谈。目标图不使用LP标签训练或选阈值，允许使用协议定义的未标注上下文图。跨领域时边的关系语义变化仍需单独解释。

数据与出处：研究问题与范围界定
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 3 -->
研究路线已从复现与任务诊断，收敛到 LP 机制研究
01
02
03
04
建立可信基线
排除任务混淆
建立统一协议
定位失效机制
TEA-GLM → LLaGA
官方权重与 Table 5
ND → NC 诊断
回归 LP → LP
冻结 BERT、边隔离
Full / Center 对照
种子波动与误报
加权、单源／多源对照
目前阶段：机制验证；尚未形成稳定优于基线的新方法。
阶段记录；详见附录中的本地结果索引
3

### Notes:
研究路线已从复现与任务诊断，收敛到 LP 机制研究

TEA-GLM复现未成功是本项目的实践背景，不据此评价原论文有效性。随后完成LLaGA环境与官方权重验证、Table5参数控制和多种子、Table8生成与任务迁移诊断，再明确LP主线。当前进入清洁协议下的邻域、加权和单多源控制。

数据与出处：阶段记录；详见附录中的本地结果索引
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 4 -->
LLaGA 将图表示映射到冻结 LLM，只训练 Projector
节点文本
冻结 BERT
HO 表示
中心 + 1–4 hop
Linear Projector
唯一训练模块
节点对的图 Token + LP 指令
冻结 Vicuna-7B
“两个节点之间是否存在链接？” → yes / no
保持语言模型固定，研究可迁移的图信息及其决策行为。
Chen et al., LLaGA (2024)；本地清洁 LP 配置
4

### Notes:
LLaGA 将图表示映射到冻结 LLM，只训练 Projector

当前主基线为HO模板、0至4-hop、Linear Projector、冻结BERT与Vicuna-7B-v1.5-16k。HO包含结构传播与邻域文本语义，不能等同纯拓扑。官方general模型本次使用两层MLP；本地Linear来自已完成参数对照，二者不是同一配置。LP输入有两个端点，每个端点包含5个图token。

数据与出处：Chen et al., LLaGA (2024)；本地清洁 LP 配置
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 5 -->
官方权重评估基本复现，仍有一个明显偏差项

| HO General LP | 论文 ACC | 本地 ACC | 差值 |
| --- | --- | --- | --- |
| Arxiv | 92.04% | 92.00% | −0.04 pp |
| Pubmed | 89.81% | 89.96% | +0.15 pp |
| Cora | 84.41% | 82.79% | −1.62 pp |

11 / 12 项
绝对偏差小于 1 pp

推理链路基本可信；
不宣称严格逐项复现。
历史 Table 5：HO + Linear + SimTeG，87.17% ± 0.64%
该数值属于原协议；后续换特征与边划分后，必须重新建立基线。
Table 1：general_model_audit_20261116.csv；原论文 Table 1
5

### Notes:
官方权重评估基本复现，仍有一个明显偏差项

在Arxiv、Pubmed、Cora上评估官方ND/HO General两层MLP权重，NC/LP共12项；11项绝对偏差小于1pp。HO Cora LP为82.7941%，论文84.41%，差−1.6159pp。Products未评估。官方General不是严格零样本跨图训练设置。本地SimTeG的Table5主基线三训练种子、每种子三次推理为87.17±0.64%，不能与后续清洁协议直接比较。

数据与出处：Table 1：general_model_audit_20261116.csv；原论文 Table 1
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 6 -->
任务对齐改善了回答格式，却没有解决迁移判别问题

### Chart

| Category | 准确率 | 有效回答率 |
| --- | --- | --- |
| ND 训练 → NC | 28.41 | 45.76 |
| NC 训练 → NC | 28.04 | 94.1 |

格式 ≠ 判别
有效回答率大幅提升，
准确率基本未变。

保留为诊断支线，
主线回归 LP → LP。
Table 8：response_validity_audit_20261116；nctrainseed42/audit_20261116
6

### Notes:
任务对齐改善了回答格式，却没有解决迁移判别问题

此页为单训练种子的诊断，不是最终LP结论。相同SimTeG设置下greedy E官方提示：节点描述训练→NC准确率28.41%、有效率45.76%；NC训练→NC准确率28.04%、有效率94.10%。回答合法不等于类别正确。早期E+T优于E也混入原文语义与指令差异，因此主线转为固定任务和二元输出空间的LP。

数据与出处：Table 8：response_validity_audit_20261116；nctrainseed42/audit_20261116
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 7 -->
清洁协议隔离标签与边信息，重建可比较的基线
特征统一
图上下文隔离
选择规则固定
冻结 BERT
不以目标标签微调
监督正边不参与聚合
固定节点对与负例
只用源验证选阈值
目标图只报告结果

| 数据集 | 源训练节点对 | 源验证节点对 | 测试节点对 |
| --- | --- | --- | --- |
| Arxiv | 80,468 | 8,939 | 78,456 |
| Pubmed | 10,132 | 1,125 | 5,111 |
| Cora | 1,381 | 153 | 651 |

dataset_research/lp_bert_disjoint_v1；清洁协议与缓存审计
7

### Notes:
清洁协议隔离标签与边信息，重建可比较的基线

冻结google-bert/bert-base-uncased，以文本token的masked mean提取768维特征，max_length256，重新训练Projector。监督正边从聚合上下文图中移除，包括训练、验证与测试；无向节点对去重，跨划分节点对隔离。负例排除数据中已知正边，不意味着现实中绝对无关系。节点本身可跨划分共享，属于节点对级LP协议。目标图未标注上下文可见，无目标LP标签训练或选阈值。SimTeG标签来源尚未完全确认，不能声称已证实泄露。

数据与出处：dataset_research/lp_bert_disjoint_v1；清洁协议与缓存审计
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 8 -->
匹配重训 Full 与 Center-repeat，检验邻域的增量价值
中心
1-hop
2-hop
3-hop
4-hop
Full
中心
中心
中心
中心
中心
Center-repeat
相同 token 数、源图、训练预算、编码器与 LLM
分别训练 × 3 个种子  →  相同目标节点对评估
解释边界：这里测量“邻域信息整体”的收益，不能直接命名为纯拓扑收益。
清洁 HO 基线：lr=2e−3，Linear，有效 batch=16，1 epoch
8

### Notes:
匹配重训 Full 与 Center-repeat，检验邻域的增量价值

Full输入为每端点0至4-hop五个图表示；Center-repeat将同一中心表示重复五次，维持token数和模型架构一致。两组分别训练，避免仅在测试删除邻域造成训练测试分布偏移。邻域收益仍包含邻域语义、传播和图结构共同作用，不是纯结构因果估计。

数据与出处：清洁 HO 基线：lr=2e−3，Linear，有效 batch=16，1 epoch
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 9 -->
邻域收益在两个迁移方向均存在，Pubmed 波动更大

### Chart

| Category | Full | Center-repeat |
| --- | --- | --- |
| Arxiv + Pubmed → Cora | 78.65 | 71.02 |
| Arxiv + Cora → Pubmed | 72.88 | 64.47 |

邻域平均增益
Cora：+7.63 pp
Pubmed：+8.41 pp

Full 的种子标准差：
2.00 vs 6.62 pp
柱高为三种子均值；完整均值 ± 标准差见附录。
lp_next_round/summary；lp_phase2/pubmed_three_seed_summary
9

### Notes:
邻域收益在两个迁移方向均存在，Pubmed 波动更大

均为原自由生成ACC、三训练种子42/43/44，±样本标准差。CoraFull78.6482±1.9969、Center71.0189±1.692；PubmedFull72.88±6.62、Center64.47±1.71。配对增益Cora7.63±3.52pp、Pubmed8.41±6.53pp。两个方向足以支持当前范围中的邻域收益，尚不足以证明跨所有图稳定。

数据与出处：lp_next_round/summary；lp_phase2/pubmed_three_seed_summary
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 10 -->
Pubmed 的 Full 结果对训练种子明显敏感

### Chart

| Category | Full | Center-repeat |
| --- | --- | --- |
| seed42 | 74.5842 | 66.4449 |
| seed43 | 65.5645 | 63.5492 |
| seed44 | 78.4778 | 63.4122 |

极差 12.91 pp
同一配方，
不同训练种子。

研究重点转向：
泛化收益与稳定性
能否同时提高？
lp_phase2/pubmed_three_seed_summary/REPORT.md
10

### Notes:
Pubmed 的 Full 结果对训练种子明显敏感

Full原生成ACC seed42=74.5842、43=65.5645、44=78.4778；Center66.4449、63.5492、63.4122。最大最小差12.9133pp。种子改变训练随机过程，不能靠重复同一权重推理替代训练种子验证。三种子统计有限，不能用极值选最优种子作为最终方法结果。

数据与出处：lp_phase2/pubmed_three_seed_summary/REPORT.md
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 11 -->
阈值能改变决策，但不能解释全部种子波动

| Pubmed · Full | 指标 |
| --- | --- |
| ROC-AUC | 0.9094 ± 0.0253 |
| AP | 0.8959 ± 0.0282 |
| 默认候选 ACC | 72.81% ± 6.62 |
| 源验证阈值 ACC | 75.69% ± 5.74 |

分开观察两件事
排序：谁更像正例？
决策：在哪里判有边？

源阈值未消除波动，
不能只做阈值补救。
lp_cross_direction_analysis/REPORT_20261119.md
11

### Notes:
阈值能改变决策，但不能解释全部种子波动

使用统一yes/no连续分数，ROC-AUC/AP刻画排序，ACC刻画给定阈值决策。PubmedFullAUC.9094±.0253，AP.8959±.0282；候选默认ACC72.81±6.62，源等权验证阈值75.69±5.74。源阈值仅由源验证标签确定，目标标签不得用于调阈值。该阈值与后续单多源比较中Arxiv-only阈值不同。阈值不能改变同一组分数的AUC，排序也有波动，不能全部归因于校准。

数据与出处：lp_cross_direction_analysis/REPORT_20261119.md
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 12 -->
高语义相似负例更易误报，但相关性不等于原因

### Chart

| Category | Full 的误报率 |
| --- | --- |
| 高语义相似负例 | 62.42 |
| 其他负例 | 29.24 |

62.42% vs 29.24%
“内容相似”
不保证“存在链接”。

下一步用受控干预
检验训练中的作用。
lp_cross_direction_analysis/REPORT_20261119.md · Pubmed seed42
12

### Notes:
高语义相似负例更易误报，但相关性不等于原因

以源验证负例语义相似度Q75确定高相似阈值（源等权，约.898629），应用到Pubmed测试负例，1357高相似、1344其他。Fullseed42误报率62.42%对29.24%；seed43为71.33/53.12，seed44为55.42/19.64。此分层只用于描述，目标标签不可参与训练权重或阈值选择。高相似性与结构、度等变量可能相关，不能仅凭分组推断语义导致误报。

数据与出处：lp_cross_direction_analysis/REPORT_20261119.md · Pubmed seed42
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 13 -->
语义加权的效果随种子反转，尚无稳定改进证据

### Chart

| Category | 原基线 | 语义加权 | 随机加权 |
| --- | --- | --- | --- |
| seed42 | 74.5842 | 63.9601 | 76.306 |
| seed43 | 65.5645 | 68.9885 | 67.7167 |

−10.62 → +3.42 pp
语义加权相对原基线
在两个种子上反转。

先补齐 seed44；
暂不继续调权重。
lp_weight_control_seed43/summary；seed42 对照；seed44 总表尚未完成
13

### Notes:
语义加权的效果随种子反转，尚无稳定改进证据

语义加权对源训练负例按各源相似度top quartile加权，原始权重2:1，负类归一化均值1；随机对照匹配每源高权重数量、权重和总权重，固定选择种子20261119。seed42原生成ACC baseline74.5842 semantic63.9601 random76.3060；43 baseline65.5645 semantic68.9885 random67.7167。语义AUCdelta−.023314与+.013270，同样反转。已审计42/43核心配置相同，种子不同；物理GPU/输出目录等不同，不代表全历史环境完全相同。未读取到seed44完整三种子总表，不能提前下最终结论。

数据与出处：lp_weight_control_seed43/summary；seed42 对照；seed44 总表尚未完成
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 14 -->
多源提高了 ACC，Full 的 AUC 却几乎不变

| 输入 | 源图 | ACC | ROC-AUC |
| --- | --- | --- | --- |
| Full | Arxiv | 68.54% | 0.9164 |
| Full | Arxiv + Cora | 74.58% | 0.9152 |
| Center | Arxiv | 61.61% | 0.6771 |
| Center | Arxiv + Cora | 66.44% | 0.7251 |

Full：+6.05 pp
AUC：−0.0012
误报数：减少 337

仅一个训练种子，
需要配对验证。
等样本呈现预算；源图与唯一样本多样性仍同时变化。
lp_source_control/summary/ANALYSIS_20261120.md · seed42
14

### Notes:
多源提高了 ACC，Full 的 AUC 却几乎不变

等预算实验：单源80468唯一Arxiv对+1381固定重复；多源80468Arxiv+1381Cora，共81849呈现、5116步。重复清单seed20261120。匹配样本呈现预算，但唯一对多样性和来源同时改变。Full原生成ACC+6.05pp，AUC−.0012；CenterACC+4.83pp，AUC+.048。候选混淆矩阵Full单源TP2375FP1577FN35TN1124，多源2347/1240/63/1461：FP减少337、TP减少28。仅说明当前seed42决策改善，不能证明稳定排序改进或全由阈值引起。43/44汇总未完成。

数据与出处：lp_source_control/summary/ANALYSIS_20261120.md · seed42
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 15 -->
先完成种子验证，再决定研究校准还是表示迁移
① 补齐配对证据
单源／多源 × Full／Center × 3 seeds
ACC 改善，AUC 近似
研究决策偏移与校准
ACC、AUC 均改善
研究表示与源图互补
收益随种子反转
优先解决训练稳定性
② 固定方法 → 未调参目标图确认 → E-commerce 跨领域扩展
Cora、Pubmed 已用于开发；最终结论需要新的保留目标图。
下一阶段计划 · 不新增目标图调参；单源43/44与加权44待最终汇总
15

### Notes:
先完成种子验证，再决定研究校准还是表示迁移

优先完成等预算单源Full/Center seed43/44四轮，与已有多源42/43/44形成2×2×3配对比较。固定当前数据、重复清单、编码器、预算、评分和Arxiv-only阈值。语义/随机加权seed44作为既有假设收尾。若ACC稳定改善而AUC/AP不变，研究决策偏移与源可选校准；若排序也改善，研究源组成及表示；若反转优先稳定性。随后在不使用其标签调参的新目标图确认，最后拓展电商跨领域，边语义变化单独报告。

数据与出处：下一阶段计划 · 不新增目标图调参；单源43/44与加权44待最终汇总
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 16 -->
现阶段已建立证据链，下一步寻找稳定且可解释的改进
已支持
匹配重训后，邻域信息在两个迁移方向都有平均收益。
未解决
Full 的泛化波动较大；语义加权尚未稳定改善结果。
待确认
多源收益能否跨种子保持，以及改善来自哪条信息路径。
论文目标：机制证据 → 针对性方法 → 未见图上的稳定收益
结论与讨论 · 当前证据范围：引文网络，LP → LP
16

### Notes:
现阶段已建立证据链，下一步寻找稳定且可解释的改进

可向导师讨论：是否认可先收尾单多源种子验证再选方法切入点；是否确定独立保留目标图；后续论文以邻域迁移稳定性为统一核心。当前不是最终创新方法完成，不保证会议录用。若目标IJCAI2027，需机制假设、针对性方法、多目标/多种子、强简单基线与成本、公平目标隔离和可复现协议形成完整证据，而非仅一张ACC提升表。

数据与出处：结论与讨论 · 当前证据范围：引文网络，LP → LP
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 17 -->
附录 A｜官方 General 的 12 项结果与论文对照

| 模板／任务 | Arxiv | Pubmed | Cora |
| --- | --- | --- | --- |
| ND · NC | 74.29 → 74.32 | 92.42 → 91.94 | 87.82 → 87.08 |
| HO · NC | 75.01 → 75.12 | 94.45 → 94.27 | 87.82 → 88.01 |
| ND · LP | 90.53 → 90.65 | 86.31 → 86.20 | 81.91 → 82.50 |
| HO · LP | 92.04 → 92.00 | 89.81 → 89.96 | 84.41 → 82.79 |

单位：ACC（%）；箭头表示“论文 → 本地”。
results/table1/official_20261114_192146/general_model_audit_20261116.csv
17

### Notes:
附录 A｜官方 General 的 12 项结果与论文对照

每格为论文→本地ACC百分比。使用官方SimTeG、Vicuna、两层MLP General权重，ND与HO。Products未完成，不在表中。官方模型并非本地重训模型；该表不能直接证明严格未见目标图的泛化。

数据与出处：results/table1/official_20261114_192146/general_model_audit_20261116.csv
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 18 -->
附录 B｜原协议与清洁协议的结果不能直接相减

| 比较项 | 历史 Table 5 | 清洁 LP |
| --- | --- | --- |
| 特征 | SimTeG | 冻结 BERT |
| Cora 测试节点对 | 680 | 651 |
| 上下文／划分 | 原始协议 | 监督正边隔离 |
| Full ACC | 87.17% ± 0.64 | 78.65% ± 2.00 |

差异来自多项协议变化，不能解释为“BERT 导致下降 8.52 pp”。
results/table5/ho_linear_multiseed_summary.json；dataset_research/lp_bert_disjoint_v1
18

### Notes:
附录 B｜原协议与清洁协议的结果不能直接相减

原协议seed42/43/44各3次推理，先对同种子取均值，再统计三种子样本标准差，87.1732±.6398。清洁协议使用统一冻结BERT且上下文图、LP节点对划分改变，Cora651而原680，因此差异不是单独替换编码器的因果效果。要研究编码器影响需固定同一cleanLP数据协议后分别重训。

数据与出处：results/table5/ho_linear_multiseed_summary.json；dataset_research/lp_bert_disjoint_v1
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 19 -->
附录 C｜有效回答率帮助区分格式与分类错误

| 条件 | 整体 ACC | 有效回答率 | 有效内 ACC |
| --- | --- | --- | --- |
| E 原始提示 | 27.06 ± 1.05 | 45.02 ± 0.67 | 60.10 ± 1.43 |
| E 对齐提示 | 25.15 ± 0.70 | 41.39 ± 0.11 | 60.78 ± 1.84 |
| E + 文本 | 57.69 ± 0.77 | 89.24 ± 1.31 | 64.65 ± 0.14 |
| 仅中心文本 | 55.78 ± 0.11 | 95.39 ± 0.55 | 58.48 ± 0.36 |

单位：%；此处 ± 是重复推理波动，不是多训练种子的不确定性。
results/table8/response_validity_audit_20261116/REPORT.md
19

### Notes:
附录 C｜有效回答率帮助区分格式与分类错误

单训练seed42，每条件3次采样推理，各542样本。OverallAccuracy=C/N，ValidResponseRate=V/N，ConditionalAccuracy=C/V；只有正确回答一定在有效集合中时乘法恒等式成立。跨推理均值的乘积不一定严格等于总体均值。不能把无效回答都算作潜在正确答案；约束解码会改变序列路径，并非修复格式而保持决策不变。

数据与出处：results/table8/response_validity_audit_20261116/REPORT.md
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 20 -->
附录 D｜简单基线是判断 LLM 增益与成本的必要参照

| 方法 | Cora ACC | Pubmed ACC |
| --- | --- | --- |
| Full LLaGA | 78.65 ± 2.00 | 72.88 ± 6.62 |
| Center-repeat LLaGA | 71.02 ± 1.69 | 64.47 ± 1.71 |
| 端点 BERT + MLP | 72.25 ± 2.00 | 47.64 ± 0.16 |
| 语义 + 结构 MLP | 81.77 ± 1.02 | 48.27 ± 0.55 |
| 结构 MLP | 77.73 ± 0.00 | 72.95 ± 0.05 |

单位：%；既要报告排序指标，也要报告参数量、显存与推理成本。
lp_phase2/analysis_20261118；pubmed_three_seed_summary/analysis
20

### Notes:
附录 D｜简单基线是判断 LLM 增益与成本的必要参照

端点BERT+MLP使用冻结端点文本表示的组合进行监督二分类；融合MLP加入CN/AA/RA/PA等结构特征。具体组合以实现和runconfig为准，不在此捏造拼接公式。CN公共邻居数；AA以log度折减公共邻居；RA以度折减；PA度乘积。对源训练监督拟合，源验证选阈值，目标测试无调参。表中为原报告ACC口径；LLM自由生成与MLP固定阈值不是同一输出机制，因此同时报告AUC/AP及阈值诊断。不可宣称LLM全面优于简单基线。

数据与出处：lp_phase2/analysis_20261118；pubmed_three_seed_summary/analysis
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 21 -->
附录 E｜主基线配置与统计口径固定后再做干预

| 项目 | 固定设置 |
| --- | --- |
| 任务与模型 | LP → LP；HO；Linear；冻结 Vicuna-7B |
| 文本编码器 | 冻结 bert-base-uncased；768维；max length 256 |
| 训练 | lr=2e−3；有效 batch=16；1 epoch |
| 随机性 | 训练种子 42 / 43 / 44；报告配对收益 |
| 评估 | 原生成 ACC + 连续分数 AUC / AP + 源阈值 |

三种子标准差 ≠ 置信区间；同模型多次推理 ≠ 独立训练重复。
清洁 LP 配置与各阶段 run_config；细节见演讲者备注
21

### Notes:
附录 E｜主基线配置与统计口径固定后再做干预

所有主基线：HO Full或Centerrepeat、Linear、lr.002、batch16、1epoch，冻结BERT与Vicuna；seed42/43/44。Cora源ArxivPubmed，Pubmed源ArxivCora。源训练集用于梯度，源验证用于阈值选择；不拿源测试或目标测试调参。当前只有3种子，报告样本标准差不是置信区间，配对增益用同seed差异计算，不将同权重推理重复作为独立训练样本。Cora/Pubmed经多轮分析已为开发目标。

数据与出处：清洁 LP 配置与各阶段 run_config；细节见演讲者备注
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。

<!-- Slide number: 22 -->
附录 F｜参考文献与本地证据均可追溯
Chen et al. (2024). LLaGA: Large Language and Graph Assistant.
代码：github.com/VITA-Group/LLaGA

| 证据 | 项目内结果目录 |
| --- | --- |
| 官方复现 | results/table1/official_20261114_192146 |
| 清洁三种子 | results/lp_next_round；results/lp_phase2 |
| 连续评分与分层 | results/lp_cross_direction_analysis |
| 加权控制 | results/lp_weight_control_seed43 |
| 单源／多源 | results/lp_source_control |

制作：academic-pptx + Anthropic pptx；数据快照：2026-11-20。
完整路径、文件 SHA-256 与逐页讲稿随 PPT 一同保存
22

### Notes:
附录 F｜参考文献与本地证据均可追溯

论文：Runjin Chen, Tong Zhao, Ajay Jaiswal, Neil Shah, Zhangyang Wang. LLaGA: Large Language and Graph Assistant. 2024. https://github.com/VITA-Group/LLaGA 。技能academic-pptx来自https://github.com/Gabberflast/academic-pptx-skill；Anthropic pptx来自https://github.com/anthropics/skills/tree/main/skills/pptx。报告指标是本项目数据，不是论文发表结果。

数据与出处：完整路径、文件 SHA-256 与逐页讲稿随 PPT 一同保存
统计口径：除特别说明外，±为三个独立训练种子的样本标准差；pp为百分点。数据快照2026-11-20。
