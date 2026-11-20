# 当前执行顺序（2026-11-16）

2026-11-17下一轮命令已准备，见 [LP_NEXT_ROUND_COMMANDS_20261117.md](LP_NEXT_ROUND_COMMANDS_20261117.md)：scripts/run_lp_next_round.sh按source→baselines→seeds→summary顺序执行。复用seed42与冻结BERT，新增四轮匹配LP训练（43/44×Full/Center）和源验证，不再重跑纯文本矩阵。阶段可拆开运行，部分推理支持续写，完成训练可跳过；原始Projector保存缺优化器，训练中断不承诺精确步数恢复。新目标图与预算控制单/多源放在本轮之后。未代用户启动全量任务。

**2026-11-17状态更新：用户已完成A/B/C三阶段。** 综合分析见 [结果报告](../results/analysis_lp_20261117/REPORT.md)。15份结果完整，新BERT Full与Center-repeat匹配训练均5662步；Cora匹配ACC为80.65%对70.20%（单seed42）。补核已按作者无额外自环规则复现SimTeG HO缓存，最初自环假设造成的误差不应解读为缓存异常。纯文本原评分出现Bayes/Bayesian内yes子串触发，不用于结构能力增益比较。下一顺序：冻结本轮工作配置→现有模型源验证与简单强基线→两匹配模型补seed43/44→新目标匹配训练验证→预算控制单源/多源及针对性方法。NC旁支继续暂缓；不要重复启动已完成的特征生成或seed42训练。

执行入口已准备并做小样本检查，完整命令见 [LP_COMMANDS_20261116.md](LP_COMMANDS_20261116.md)：A为CPU数据/缓存/评分审计与旧权重五组LP初筛；B为新数据视图及固定BERT特征；C才启动新协议Full/Center-repeat匹配重训。未启动任何全量任务。固定预算单源/多源采样、连续候选打分和难负例矩阵尚未实现，不能把C的全量基线描述为这些对照已完成。

本轮进一步确认：用户选择“保持 LP→LP 主线，NC 作为后续扩展”。单源/多源作为统一实验变量，详见 [SOURCE_PROTOCOL_RESEARCH_PLAN_20261116.md](SOURCE_PROTOCOL_RESEARCH_PLAN_20261116.md)。保留现有Table5多源复现；先旧权重LP输入初筛，再在清洁协议下比较Arxiv、Pubmed、Arxiv+Pubmed（固定监督预算），之后做匹配中心重训与多目标验证。已完成的NC control归档，不重复训练；截图NC+nd方案当前不启动。

**最新且优先的用户决定：收回到 Table 5 同任务跨图 LP→LP。** 完整执行方案见 [LP_SAME_TASK_PLAN_20261116.md](LP_SAME_TASK_PLAN_20261116.md)。当前顺序：现有LP协议/数据审计→旧LP权重输入干预初筛→清洁LP视图与共享冻结文本编码器→Full/Center匹配LP重训→同领域多目标与关键种子→针对性改进→跨领域LP。Table8 nd→nc、G1/G2与NC control均不再是当前前置任务；保留已有成果。本次只更新计划与只读审计，没有启动或终止训练进程。

以下为历史计划，不再作为执行指令；其中BERT上先训nd、先跑NC control等均被上述决定取代。

最新研究决策：先研究同领域跨图泛化，再研究跨领域跨图泛化。两阶段可分别发展为论文，但需各自独立的机制证据和方法贡献。此前“先五张电商跨领域目标、后论文图轮换”的建议调整为：G1/G2结果分析→同特征NC训练任务对照→统一冻结BERT基线→同领域多方向/多种子验证与方法改进→跨领域五张电商目标验证。电商数据准备可并行，正式跨领域实验作为第二阶段。以下较早排序保留作历史计划，不覆盖本段最新决定。

当前G1/G2已经完成，分析见results/table8/output_protocol_v2_HO_linear_simteg_trainseed42/RESULTS_20261116.md；下一项是scripts/run_table8_nc_control.sh，尚未代为启动训练。

此文件统一当前优先级，取代此前“优先补另外两个现有SimTeG特征留一图方向”的建议。当前G1/G2正在运行，不修改其入口、解码、提示或权重，不从未完成文件提前挑选协议。

更新：用户已提出新增Computers、Children、Sports-Fitness、Photo、History五张电商目标图。它们优先于下表第4步的论文图轮换；统一冻结BERT特征线完成后，先固定源权重逐图测试五张电商图。电商源图训练和留一验证为后续独立实验，不是这轮新增目标测试的前置条件。详细方案见ECOMMERCE_EXTENSION_PLAN.md。

|顺序|工作|完成标准|
|---|---|---|
|1，正在执行|现有nd Projector，Cora四组G1/G2输出诊断|8文件各542条，完整性核对，metrics/paired/class_counts生成；保留旧G0和失败候选打分|
|2|现有SimTeG上新训nc Projector，Arxiv+Pubmed→Cora NC|seed42，HO/linear/lr2e-3/有效batch16/1epoch；用相同G1/G2比较nd与nc；仅作为机制对照|
|3|建立统一冻结BERT特征线，先重训nd Projector|一个固定版本的BERT-base，原始文本不拼标签，固定池化/截断/归一化，重算HO缓存；Arxiv+Pubmed→Cora，复用固定评估|
|4|在这套冻结BERT特征上增加另外两个留一图方向|Arxiv+Cora→Pubmed、Pubmed+Cora→Arxiv；先seed42，固定nd主线及关键输入组，不把训练任务或编码器一起换来换去|
|5|对主要结论补独立训练/扰动种子|训练seed43/44，错配映射也补独立种子；G1/G2不机械重复推理种子；T不随Projector重复训练|
|6|冻结方法后加入独立或不同领域目标图|Products或另一可追溯文本图；明确监督来源、源目标重叠与划分，最终确认泛化|

现有SimTeG另外两个方向暂缓，避免在监督来源未明的特征上扩充完整实验矩阵后又在BERT特征上重复一遍。它们有复现参考价值，但不是进入下一阶段的必要条件。BERT下nd是Table8机制研究主线，nc是训练任务对照，不替换论文nd→nc行；新编码器下的nc匹配对照在需要验证该任务效应时再补。

G1/G2实际使用do_sample=False和num_beams=1，不是把temperature设成0。G1对全部词表贪心，G2对合法类别token路径贪心，完整类名后EOS；无采样不保证跨硬件/内核/精度逐位复现，也不保证判断正确。原采样G0继续保留。

Cora当前为2708节点、7类、542测试节点。CoraFull是不同图版本，不能与Planetoid的split='full'混淆，也不能直接替换论文Cora计算复现差值。新增Cora版本需核对原始文本、ID与类别映射、与已有图的重复。七类数量本身不能说明跨图分类已足够简单。

暂缓：继续搜索lr/架构、ND模板全组合、Table6完整重训、同时试多个文本编码器、直接开始复杂新方法。Table1的HO/Cora LP小规模复测可低优先级单独安排，不阻塞当前主线。

已有入口：步骤1 scripts/run_table8_output_protocol.sh；步骤2 scripts/run_table8_nc_control.sh（双卡0/3，训练后3卡评估）；步骤3及后续尚需实现/审计新特征管线，不能直接把EMB改成bert运行。当前Cora已参与方法开发，最终外部确认需另留目标图。
