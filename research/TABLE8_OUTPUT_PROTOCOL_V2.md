# Table 8 输出诊断与同任务训练对照

固定日期：2026-11-16。新方案在全量目标评估前固定，不按Cora ACC选择类别别名、前缀、打分归一化或搜索宽度。当前旧mean候选打分失败结果保留。

## 要回答的问题与边界

“信息不足”和“输出/任务适配不足”可以同时存在，任何一种强制选择协议都不能保证测出模型潜在知识。约束后恢复正确答案支持输出限制能改善决策；未恢复不证明embedding没有信息。跨任务/跨图/类别空间迁移、描述训练目标、Projector优化都可能是剩余原因。

## 固定的配对协议

|协议|提示/模型/图输入|解码|
|---|---|---|
|G0，已有结果|原始四组E_official/E/ET/T|temperature0.2，自由采样，已有3次重复保留|
|G1，新增|逐字复用对应G0文件中的提示、相同权重和图特征|无约束greedy，1 beam，最多1024 token|
|G2，新增|与G1完全相同|greedy，仅允许原始完整类别字符串及EOS，1 beam|

G1→G2是核心配对，只改变合法token限制；不能只比较采样G0与确定性G2。G2通过原模型logits逐步贪心选择合法路径，不算mean/sum候选分数、不加长度归一化，不修改类别名称/顺序/提示，不额外加自然答案前缀，不使用节点GT决定候选。类别集合是任务公开信息。

每个完整prompt+空格+类别整体分词，并检查原prompt的token前缀逐位一致。所有类别路径组成trie，完整类别后必须EOS；共享前缀和类别本身为另一类别前缀的情形均处理。图token插入不改变generate返回的原始输入ID前缀，运行中也检查。G2最大生成长度由最长合法路径+EOS决定，绝不截断合法类名。保留生成token及各类别长度供审计。长度和类别先验仍可能影响贪心分支选择，此协议不承诺概率校准，也不是全局最优序列搜索。

原始eval/eval_pretrain.py、scripts/eval.sh和eval_res.py未为此新协议改动；扩展独立eval_nc_diagnostics.py的可选分支，旧generate/label_score仍可使用。新增检查和汇总脚本不会改写预测输入和权重。

## 1. 源图保留节点工程检查

Pubmed官方test前8个节点不属于用于当前Projector训练的train集合。这里是源域保留节点的工程检查，不是新的零样本目标结果，也不使用其ACC调协议。不筛选“容易样本”，不把8节点结果作统计证据。

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga
EVAL_GPU=3 CHECK_N=8 bash scripts/check_table8_output_protocol.sh
```

依次生成原始四组小样本源结果，再各跑G1/G2。检查：无异常、token边界一致、G2全部为合法类、完整终止、ID/GT/提示一致，metrics/paired.csv生成。低准确率或少数类集中必须如实保留，不能根据它改名或换评分规则后挑高分。

## 2. Cora全量诊断（无需训练）

```bash
EVAL_GPU=3 bash scripts/run_table8_output_protocol.sh
```

共4组×2协议=8次推理，每次542节点，顺序使用一张卡。G1/G2确定性，不再重复3个采样种子。G0三次结果保留；自动配对先以seed0作逐节点参照，核心比较仍是G1→G2。输出：results/table8/output_protocol_v2_HO_linear_simteg_trainseed42/{cora_*_greedy.jsonl,cora_*_constrained_greedy.jsonl,metrics.csv,paired.csv,class_counts.json}。日志：logs/table8/output_protocol_v2/。

已有文件按配置/源文件哈希/权重哈希/实现哈希检查后续跑；不完整或不同设置不能静默混用。工程检查可用LIMIT=2并改OUTPUT_ROOT，正式全量不要设置LIMIT，也不要复用小样本输出目录。

自动汇总ACC、Macro-F1、无效回答数、类别分布、无效→正确、原正确→错误、净正确数变化。只消除无效而不提高ACC，不支持输出格式是主要原因。G2若连T也异常下降，先把它视为失败决策协议，不能用它否定图通道。G1/G2的改善都需要逐节点检查，不能仅看总体ACC。

## 3. nc→nc同任务源训练对照

固定Arxiv+Pubmed、相同训练节点、HO4hop、linear、SimTeG、lr2e-3、seed42、1epoch、有效batch16；从Vicuna重新初始化Projector，任务改为nc，不加载nd训练/官方权重。目标仍为从未参与该Projector训练的Cora。原始nd任务回答本身包含源图真实类别，故两组都有源标签监督。两任务样本数相同但监督token数/回答长度不同，所以这是任务训练配方对照，不是只改一个输出格式字符的纯因果实验。

确认0/3卡可用后执行（不要与占用3号卡的上面诊断同时启动）：

```bash
GPU_INCLUDE=localhost:0,3 EVAL_GPU=3 LLAGA_MASTER_PORT=61006 \
bash scripts/run_table8_nc_control.sh
```

此命令使用原scripts/train_deepspeed.sh训练，完成检查后使用原scripts/eval.sh生成E_official/E/ET的seed0结果；T是完全相同的冻结Vicuna文本基线，直接复用。然后自动执行G1/G2和汇总。新增训练权重目录checkpoints/table8/llaga-HO-arxiv-pubmed-nc-linear-simteg-lr2e-3-seed42-bs16，结果目录results/table8/HO_linear_simteg_nctrainseed42/。这是独立机制对照，不能替换论文Table8的nd→nc复现。

中断后若训练权重目录已存在，脚本会拒绝覆盖。训练已完成时无需重训，可先用原scripts/eval.sh补缺失源结果，再指定SOURCE_DIR/OUTPUT_ROOT重跑run_table8_output_protocol.sh；不要为续跑删除已有权重。

对照解释：nd/G1差而nd/G2恢复→输出限制有可恢复收益；nd/G2仍差但nc/G1及nc/G2明显改善→源分类训练比描述训练更适配跨图分类；只有无效率改善而ACC不改善→形式遵循不足以解释分类错误；两任务在G2都差→继续查表示和跨图类别语义迁移，不可认定图信息不存在。若要更细分指令/类别语义/答案长度，需要后续匹配回答模板或混合任务实验。

本轮先seed42验证方向，固定后补独立训练种子43/44和目标图；G1/G2无需机械重复推理种子。不能用当前Cora反馈反复选协议后再称其为独立确认集。

## 已完成的实现验证

2026-11-16在GPU0使用真实基座/nd权重完成Cora四组各2节点的G1/G2，以及Pubmed源图保留前2节点的原采样+G1/G2端到端检查。8个配对组输入、GT一致，约束结果全部为合法完整类别并EOS结束，原官方评分和新增汇总均正常。共享token前缀、类别路径互为前缀、重复路径拒绝、token边界改变拒绝的3项单元测试通过；shell语法和Python编译检查通过。验证记录logs/table8/output_protocol_v2_validation/validation.json，Cora逐组日志同目录；小样本预测在/tmp下，不作为正式结果。未执行全量诊断或nc训练，nc包装脚本仅完成语法与参数流程审查。
