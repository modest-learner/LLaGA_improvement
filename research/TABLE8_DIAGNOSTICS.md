# Table 8 下一步：中心表示、邻域配对、分类输出协议

日期：2026-11-15。当前固定HO/4-hop/linear/lr2e-3/SimTeG、Arxiv+Pubmed节点描述训练seed42的权重。此阶段只做推理诊断，不重训。目标是区分信息通道与输出任务问题，不能据此直接断言差距的根本原因。

## 实现与验证

新增 `eval/eval_nc_diagnostics.py` 和 `scripts/eval_nc_diagnostics.sh`。原Table 1评估仍在运行，本次未改动已有评估/训练入口。新入口复用原模型加载器、HO特征加载、Vicuna对话模板；逐字复用已有预测文件中保存的提示。原文件中的预测答案不作为模型输入，GT仅用于一致性检查及保存结果。候选打分为每个节点提供同一份完整类别集合，不根据该节点GT选择候选。

入口要求输入完整、有eval_config的542条Cora/HO测试结果，并校验ID、GT和配置。生成沿用temperature=0.2、max_new_tokens=1024、逐节点随机种子规则。输出放入新目录，断点续跑会检查源文件哈希、Projector哈希、提示、顺序与诊断配置是否一致。首次运行不覆盖源结果。T直接加载冻结Vicuna。

GPU3小样本验证已通过：full条件下E/ET各2条预测与已有seed0结果相同；三个消融各完成1条真实推理；四组候选打分各完成1条；图输入变化和donor无自配对/一一置换通过检查；已有完整文件跳过及参数不匹配拒绝续跑通过检查。候选打分的平均负对数概率与模型原生teacher-forced loss交叉核对：1.496174 vs 1.496094（后者半精度）。尚未启动全量诊断。日志保存在 `logs/table8/diagnostic_validation_20261115/`。

## 首先限定结论

原始E均值27.06%、约54.98%输出无法按官方规则解析为唯一类别，是明确问题；尚不能认定它解释了全部论文差距。论文未提供同协议下的无效回答率，本地工作配置也未证实与论文全部相同。节点描述式回答、候选类别之外的主题、复述多类等可以与语义信息不足、任务遵循不佳、源目标标签空间差异同时发生。

本轮保留自由生成和官方评分的原结果。新候选打分是另一种决策协议，不能用其结果替换论文复现行；其提升也不等同于“仅修复解析器”的收益。

## A. 中心表示与邻域配对诊断

原HO输入记为 `[h0,h1,h2,h3,h4]`，h0是中心SimTeG表示，后四项是多步传播缓存。HO各跳表示并非互不重叠的邻居环，通常混合中心及邻域信息。

|名称|实际输入|作用/限制|
|---|---|---|
|full（已有ET）|[h0,h1,h2,h3,h4]+中心文本|参考条件|
|center_only|[h0]+相同中心文本|移除多跳输入，但token数从5变1|
|center_repeat|[h0,h0,h0,h0,h0]+相同中心文本|保持5个token，不新增邻域信息；重复中心也会改变分布|
|neighbor_shuffle|[h0(v),h1(u),h2(u),h3(u),h4(u)]+v的文本|保留中心，破坏中心与后四项的正确配对|
|T（已有）|仅相同中心文本|无图通道参考|

neighbor_shuffle使用固定ablation_seed=42在完整542个测试节点间生成随机循环置换，每个节点恰好被用作一次donor，无自身配对，四跳作为整体一起置换。映射在limit/断点续跑前建立，不受推理种子影响，不使用标签。它是HO聚合表示的配对扰动，不是图边重连实验，也不能单独分离拓扑和邻居文本的贡献。

三种扰动是推理期压力测试，不是重新训练中心-only模型，也不是论文Table 6的完整训练期消融。性能下降可能有分布偏移的因素。后续若要发表稳定的扰动结论，需补独立置换种子及训练种子，并考虑匹配重训的消融。

在GPU3空闲时执行（若0、3卡正在跑ND训练，先等待释放，或另选空闲卡）。共9次生成，每次542节点，无需训练。

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga

(
  set -euo pipefail
  EVAL_GPU=3
  SOURCE=./results/table8/HO_linear_simteg_trainseed42/cora_ET_inferseed0.jsonl
  OUT=./results/table8/diagnostics_HO_linear_simteg_trainseed42/generation
  LOG=./logs/table8/diagnostics_HO_linear_simteg_trainseed42
  mkdir -p "$OUT" "$LOG"

  for ablation in center_only center_repeat neighbor_shuffle; do
    for infer_seed in 0 1 2; do
      name="cora_ET_${ablation}_aseed42_inferseed${infer_seed}"
      CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
      bash scripts/eval_nc_diagnostics.sh \
        --source_results "$SOURCE" \
        --answers_file "$OUT/$name.jsonl" \
        --ablation "$ablation" \
        --ablation_seed 42 \
        --decoding generate \
        --inference_seed "$infer_seed" \
        2>&1 | tee -a "$LOG/$name.log"

      python eval/eval_res.py --dataset cora --task nc \
        --res_path "$OUT/$name.jsonl" \
        | tee "$OUT/$name.score.txt"
    done
  done
)
```

aseed42仅对neighbor_shuffle生效，其余组文件名中的该部分只是统一命名。SOURCE虽然是inferseed0文件，但这里只复用其输入提示、数据和模型配置；实际生成种子分别为0/1/2。对照原来的ET/T三次结果，不能只挑最好一次。

解释顺序：先比ET与center_repeat，再结合center_only和T判断；若ET优于两种中心条件且neighbor_shuffle更差，支持正确匹配的多跳表示有帮助，但仍需分布偏移/编码器监督来源控制。若center_repeat优于ET，说明本权重下多跳输入可能带来干扰；若置换几乎不改变结果，应检查中心文本支配输出或模型未有效使用对应多跳信息等假设。

## B. 固定候选类别打分

自由生成是让模型自行生成答案，再由官方规则判断是否唯一匹配类别。候选打分则保持同一输入，依次计算七个完整类别字符串（如Case_Based、Neural_Networks、Theory）的条件分数，选择最高者；不会从模型原来的自由回答中人工猜类别。

当前固定主分数为候选字符串的平均token对数概率：score(c)=sum(log P(c_t | prompt,c_<t>))/token_count。不计prompt loss、不计EOS，不采样。使用原类别名（含下划线），七类全部参与每个样本的决策。以平均分减少不同token长度的直接影响，但它仍受类别措辞和先验偏好影响，不能理解为校准的类别概率。

同时保存每类的mean/sum logprob、token数及sum规则的预测，供敏感性检查；主结果预先固定为mean规则，不能根据Cora准确率事后择优。多token类别会对所有token打分，不只比较首token。新text字段是所选合法类别，因此能继续用原eval_res.py评分。

每组只需执行一次；这是确定性候选打分，重复三个采样种子没有与自由生成相同的意义。相同环境下通常可复现，但不承诺跨硬件逐位一致。仍须对E_official、E、ET、T全部做，不能只改变E的输出协议后与旧T比较。

```bash
cd /data2/workspace/hy/LLaGA
conda activate llaga

(
  set -euo pipefail
  EVAL_GPU=3
  SOURCE_DIR=./results/table8/HO_linear_simteg_trainseed42
  OUT=./results/table8/diagnostics_HO_linear_simteg_trainseed42/label_score
  LOG=./logs/table8/diagnostics_HO_linear_simteg_trainseed42
  mkdir -p "$OUT" "$LOG"

  for group in E_official E ET T; do
    name="cora_${group}_full_labelscore"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
    bash scripts/eval_nc_diagnostics.sh \
      --source_results "$SOURCE_DIR/cora_${group}_inferseed0.jsonl" \
      --answers_file "$OUT/$name.jsonl" \
      --ablation full \
      --decoding label_score \
      2>&1 | tee -a "$LOG/$name.log"

    python eval/eval_res.py --dataset cora --task nc \
      --res_path "$OUT/$name.jsonl" \
      | tee "$OUT/$name.score.txt"
  done
)
```

合法输出率会因为七选一机制达到100%，这不等于准确率提高，更不等于根因已经确定。需按节点统计：自由生成无效且候选打分正确的数量、无效但候选仍错误的数量，以及自由生成原本正确却被新协议损坏的数量。同时比较新协议下ET-T的差值。如果E恢复明显，支持自由输出/决策协议是瓶颈之一；若仍差，继续检查图表示、跨标签空间语义映射及类别先验。约束解码是另一种实现：只允许生成合法类别字符串；本次先做可保存七类分数的候选打分，不额外堆叠另一协议。

## 下一步顺序

1. 优先跑B（4次确定性候选打分），回答输出协议是否掩盖了部分信息。
2. 跑A（9次生成），回答中心/多跳通道带来的帮助和干扰。两段命令使用相同GPU时顺序执行。
3. 若自由生成无效比例让A难解释，将同样三个消融也放到label_score协议下比较。脚本支持直接将对应命令的decoding换为label_score、每组只跑一次，并使用新文件名。
4. 跟进无目标标签微调的冻结编码器特征线，重新生成缓存并重训；原SimTeG来源仍需追溯。避免用Cora测试反馈反复选提示、标签别名或归一化方式。
5. 在固定协议下补训练seed43/44和新的目标图；要解释模板差异再补ND，不在这些诊断前直接扩展所有组合。
