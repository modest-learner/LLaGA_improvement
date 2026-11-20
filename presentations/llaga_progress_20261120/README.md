# LLaGA 中文研究进展汇报

成品位于项目根目录：
- `LLaGA_跨图LP研究进展汇报_20261120.pptx`：可编辑幻灯片，22页（16页正文+6页附录），包含逐页演讲者备注。
- `LLaGA_跨图LP研究进展汇报_20261120.pdf`：跨设备预览版本。

## 内容与排版

以同任务跨图LP为主线，覆盖官方复现、任务迁移诊断、清洁协议、邻域匹配重训、种子稳定性、连续评分、语义/随机加权和单源/多源控制。正文建议20–25分钟，附录用于答疑。

采用academic-pptx的学术汇报原则与Anthropic pptx制作流程。中文字体Noto Sans CJK SC，标题27pt、正文主要20–24pt、图表标签16pt，白底深蓝。图表、表格和流程图均可编辑。其他电脑缺少该字体时可用PDF保留版式。

## 数据快照与边界

快照日期2026-11-20；来源共17个本地报告/CSV/JSON，保存在`source_snapshot`；原路径、SHA-256和时间戳见`source_manifest.json`。未修改实验数据、训练配置或代码，未启动GPU任务。

三种子均值与样本标准差、单种子三次推理、自由生成ACC、候选分数阈值ACC分别标识。原SimTeG协议与清洁BERT协议不直接比较性能差。尚无完整汇总的加权seed44与单源seed43/44，不当作完成结果引用。

## 配套文件

- `讲稿与逐页大纲.md`：逐页解释、限制和数据出处。
- `outline_and_notes.json`：结构化逐页内容。
- `build_deck.js`：PptxGenJS制作源码。
- `audit_sources.py`、`data_checks.json`：关键数字复核与来源快照。
- `validation.txt`：PPTX结构校验结果。
- `renders/`：22页最终渲染图。
- `slide_text.md`：PPTX内容提取结果。

制作工具仅用于CPU排版和渲染：PptxGenJS、LibreOffice、PyMuPDF。已逐页检查最终渲染，无可见文本溢出；PPTX结构校验通过。数值断言覆盖官方12项偏差、加权反转、单源多源ACC差与FP差，其余指标与本地报告核对。

技能来源：
- academic-pptx：https://github.com/Gabberflast/academic-pptx-skill
- Anthropic pptx：https://github.com/anthropics/skills/tree/main/skills/pptx
