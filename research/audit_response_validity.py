"""Audit existing Table 8 sampled answers; no inference or primary scorer changes."""
import ast
from collections import Counter
import contextlib
import csv
import io
import json
from pathlib import Path
import re
import statistics
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/table8/HO_linear_simteg_trainseed42'
OUT = ROOT / 'results/table8/response_validity_audit_20261116'
GROUPS = ('E_official', 'E', 'ET', 'T')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    data = torch.load(ROOT / 'dataset/cora/processed_data.pt', map_location='cpu')
    labels = list(data.label_texts)
    short = [label.split('_')[0].strip().lower() for label in labels]
    refs = [json.loads(line) for line in (ROOT / 'dataset/cora/sampled_2_10_test.jsonl').read_text().splitlines()]
    ids = [r['id'] for r in refs]
    tree = ast.parse((ROOT / 'eval/eval_res.py').read_text())
    function = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_cora_nc']
    env = dict(torch=SimpleNamespace(load=lambda _: data), json=json)
    exec(compile(ast.Module(body=function, type_ignores=[]), 'original_cora_scorer', 'exec'), env)
    per_run, pooled, means, invalid_rows, incidental_cases, invalid_top = [], [], [], [], [], {}
    for group in GROUPS:
        group_runs = []
        top = Counter()
        for seed in (0, 1, 2):
            path = SOURCE / f'cora_{group}_inferseed{seed}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == len({r['question_id'] for r in rows}) == 542
            assert [r['question_id'] for r in rows] == ids
            correct = valid = zero = multi = 0
            for r, ref in zip(rows, refs):
                assert r['gt'] == ref['conversations'][1]['value']
                assert r['eval_config']['seed'] == seed
                answer = r['text'].strip()
                hits = [i for i, name in enumerate(short) if name in answer.lower()]
                truth = int(data.y[r['question_id']])
                if len(hits) == 1:
                    valid += 1
                    correct += hits[0] == truth
                    continue
                zero += not hits
                multi += len(hits) > 1
                item = dict(group=group, inference_seed=seed, question_id=r['question_id'],
                            source_file=str(path.relative_to(ROOT)),
                            failure_type='zero_keyword_match' if not hits else 'multiple_keyword_matches',
                            matched_categories=[labels[i] for i in hits], text=answer,
                            gt=labels[truth])
                invalid_rows.append(item)
                top[answer] += 1
                # One explicit lexical diagnostic, chosen independently of the GT.
                clean = re.sub(r'\bin this case\b', '', answer.lower())
                clean_hits = [i for i, name in enumerate(short) if name in clean]
                if len(hits) > 1 and len(clean_hits) == 1 and not re.search(r'case[\s_\\-]*based', answer.lower()):
                    incidental_cases.append(dict(**item, normalized_prediction=labels[clean_hits[0]],
                                                 matches_gt=clean_hits[0] == truth))
            n = len(rows)
            assert valid + zero + multi == n
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                env['eval_cora_nc'](str(path))
            original_acc = float(re.search(r'acc: ([0-9.]+)', capture.getvalue()).group(1))
            assert abs(correct / n - original_acc) < .000051
            row = dict(group=group, inference_seed=seed, n=n, correct=correct, valid=valid,
                       valid_wrong=valid-correct, zero_match=zero, multiple_matches=multi,
                       overall_accuracy_pct=correct/n*100, valid_response_rate_pct=valid/n*100,
                       conditional_accuracy_pct=correct/valid*100)
            per_run.append(row)
            group_runs.append(row)
        sums = {name: sum(r[name] for r in group_runs) for name in
                ('n', 'correct', 'valid', 'valid_wrong', 'zero_match', 'multiple_matches')}
        pooled.append(dict(group=group, **sums,
                           overall_accuracy_pct=sums['correct']/sums['n']*100,
                           valid_response_rate_pct=sums['valid']/sums['n']*100,
                           conditional_accuracy_pct=sums['correct']/sums['valid']*100))
        summary = dict(group=group)
        for metric in ('overall_accuracy_pct', 'valid_response_rate_pct', 'conditional_accuracy_pct'):
            values = [r[metric] for r in group_runs]
            summary[metric+'_mean'] = statistics.mean(values)
            summary[metric+'_sample_sd'] = statistics.stdev(values)
        means.append(summary)
        invalid_top[group] = [dict(text=s, count=n) for s, n in top.most_common(20)]
    write_csv(OUT / 'per_inference.csv', per_run)
    write_csv(OUT / 'mean_sd.csv', means)
    write_csv(OUT / 'pooled_counts.csv', pooled)
    for name, records in [('invalid_answers', invalid_rows), ('incidental_case_diagnostic', incidental_cases)]:
        with (OUT / f'{name}.jsonl').open('w') as handle:
            for r in records:
                handle.write(json.dumps(r, ensure_ascii=False) + '\n')
    details = dict(per_inference=per_run, means=means, pooled=pooled, invalid_top=invalid_top,
                   incidental_case_counts={group: dict(candidates=sum(r['group']==group for r in incidental_cases),
                       matches_gt=sum(r['group']==group and r['matches_gt'] for r in incidental_cases)) for group in GROUPS},
                   limitations=['Three inference repeats share one training checkpoint and the same 542 nodes.',
                                'Conditional accuracy is on a selected subset and is not a causal estimate of format loss.',
                                'Keyword hits are lexical observations, not mutually exclusive semantic causes.',
                                'Incidental-case normalization is an auxiliary diagnostic, not a replacement score.'])
    (OUT / 'summary.json').write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Table 8：已有回答有效性审计（2026-11-16）', '',
             '固定同一个训练seed42权重，原G0采样推理seed0/1/2；未重训、未重新生成、未修改原评分或原预测。每组每次542节点。', '',
             'Overall Accuracy=C/N；Valid Response Rate=V/N；Conditional Accuracy=C/V。有效定义严格沿用原Cora评分器：七个类别短关键词中恰有一个在回答中出现。解释性文字本身不导致无效。', '',
             '|条件|Overall ACC%|Valid Response Rate%|Conditional ACC%|',
             '|---|---:|---:|---:|']
    for r in means:
        values = [f"{r[m+'_mean']:.2f} ± {r[m+'_sample_sd']:.2f}" for m in
                  ('overall_accuracy_pct', 'valid_response_rate_pct', 'conditional_accuracy_pct')]
        lines.append('|'+r['group']+'|'+'|'.join(values)+'|')
    lines += ['', '三列均先按每次推理计算，再取三次均值与样本SD。SD不是跨训练种子波动。', '',
              '|条件|总回答|有效且正确|有效但错误|零关键词|多个关键词|',
              '|---|---:|---:|---:|---:|---:|']
    for r in pooled:
        lines.append(f"|{r['group']}|{r['n']}|{r['correct']}|{r['valid_wrong']}|{r['zero_match']}|{r['multiple_matches']}|")
    lines += ['', '合并计数包含同542节点的三次生成，不是1626个独立测试节点。合并后的C/N=(V/N)×(C/V)严格成立；各列均值的乘积不一定精确相等。', '',
              '## 无效回答真实案例', '',
              '- 无候选类别/宽泛描述：E_official seed0，节点9：“This is a paper in machine learning.”。',
              '- 源标签空间残留：E_official seed0，节点369：“This is a paper in cs.CV.”。',
              '- 拒答：E seed0，节点1，开头“I’m sorry, but I cannot directly tell you which class the center node belongs to.”（原文使用ASCII撇号）。',
              '- 多类别复述：E_official seed0，节点845：“The center node in this graph is a mixture of the classes Case\\_Based, Genetic\\_Algorithms, Neural\\_Networks, Probabilistic\\_Methods, Reinforcement\\_Learning, Rule\\_Learning, and Theory.”。',
              '- 候选外类别/粒度不一致：T seed2，节点833：“The center node belongs to the class \"Evolutionary Algorithms (EA)\".”，GT为Genetic_Algorithms；这不能自动视为严格同义或单纯标点错误。',
              '- 评分器词义混淆：E seed0，节点143：“The center node in this case is a \"Neural Networks\" node.”，GT为Neural_Networks；in this case中的case误触发Case_Based短关键词，形成两个匹配。', '',
              '解释性文字只要包含唯一类别，原规则通常仍判有效；大小写、空格/下划线/连字符变化也常因短词匹配而被接受。零关键词不等于无语义，多关键词不等于确实作了多个类别判断。', '',
              '## 辅助词义消歧检查', '',
              '只在原无效回答中删除固定短语in this case，且要求没有显式Case-Based名称、剩余恰一类。该规则不读取GT决定预测。筛出60条，28条消歧后的类别与GT一致；按E_official/E/ET/T分别为19/36/4/1条，匹配GT为10/13/4/1条。它证明存在评分器假阴性，但不构成完整语义标注，也不能用28条推算全部解析误差。', '',
              '## 解释边界', '',
              'E组有效率低是主要可观测现象，但条件准确率只覆盖模型选择作答的子集，不能外推到无效节点，不能把Conditional ACC−Overall ACC解释为可恢复的格式损失。',
              '需要区分：合法错误选择；无目标类别决策（描述/拒答/源标签）；表达可恢复但评分拒绝。前两者仍不能单凭输出证明图Token的信息不足，也可能是Projector读出、标签语义映射或任务对齐失败。',
              'E+T相对T的Conditional ACC更高、有效率更低，但两个有效子集不同；不能据此宣称图表示在相同节点上一定更准确。',
              '后续以固定输入/评估进行nd→nc训练任务对照，保留原指标。纯解析辅助实验只处理既有文本，不重新生成、不按GT从多个类别中挑正确项。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(means, ensure_ascii=False, indent=2))
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
