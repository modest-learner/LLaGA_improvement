"""Read-only prediction/weight audit of the completed NC vs ND control."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/table8'
NEW = BASE / 'HO_linear_simteg_nctrainseed42'
OUT = NEW / 'audit_20261116'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    data = torch.load(ROOT / 'dataset/cora/processed_data.pt', map_location='cpu')
    labels = list(data.label_texts)
    short = [s.split('_')[0].strip().lower() for s in labels]
    refs = read(ROOT / 'dataset/cora/sampled_2_10_test.jsonl')
    ids = [r['id'] for r in refs]
    truth = [int(data.y[i]) for i in ids]
    assert len(ids) == len(set(ids)) == 542
    metrics, comparisons, counts, records, predictions = [], [], {}, {}, {}
    weight_audits = {}
    for training, source, diag in [
        ('nd', BASE / 'HO_linear_simteg_trainseed42', BASE / 'output_protocol_v2_HO_linear_simteg_trainseed42'),
        ('nc', NEW / 'source', NEW / 'output_protocol_v2'),
    ]:
        existing = {(r['group'], r['protocol']): r for r in csv.DictReader((diag / 'metrics.csv').open())}
        for group in ('E_official', 'E', 'ET', 'T'):
            src = source / f'cora_{group}_inferseed0.jsonl'
            source_rows = read(src)
            for protocol in ('sampled_seed0', 'greedy', 'constrained_greedy'):
                path = src if protocol == 'sampled_seed0' else diag / f'cora_{group}_{protocol}.jsonl'
                rows = read(path)
                assert [r['question_id'] for r in rows] == ids
                assert all(r['gt'] == q['conversations'][1]['value'] for r, q in zip(rows, refs))
                assert all(r['prompt'] == s['prompt'] and r['graph'] == s['graph'] for r, s in zip(rows, source_rows))
                if protocol != 'sampled_seed0':
                    cfg = rows[0]['diagnostic_config']
                    assert all(r['diagnostic_config'] == cfg for r in rows)
                    assert cfg['source_sha256'] == sha(src)
                    assert cfg['decoding'] == protocol
                    model = ROOT / cfg['source_eval_config']['model_path']
                    if group != 'T':
                        weight = model / 'mm_projector.bin'
                        assert sha(weight) == cfg['checkpoint_sha256']
                        if str(model) not in weight_audits:
                            tensors = torch.load(weight, map_location='cpu')
                            assert all(torch.isfinite(t).all() for t in tensors.values())
                            state = json.loads((model / 'trainer_state.json').read_text())
                            assert state['global_step'] == state['max_steps'] == 6423
                            weight_audits[str(model)] = {'sha256': sha(weight), 'steps': 6423, 'finite': True}
                pred = []
                for r in rows:
                    hits = [i for i, stem in enumerate(short) if stem in r['text'].strip().lower()]
                    pred.append(hits[0] if len(hits) == 1 else -1)
                n = len(pred)
                correct = sum(p == t for p, t in zip(pred, truth))
                valid = sum(p >= 0 for p in pred)
                old = existing[group, protocol]
                assert int(old['correct']) == correct and int(old['invalid']) == n - valid
                if protocol == 'constrained_greedy':
                    assert all(r['text'] in labels for r in rows)
                row = dict(training=training, group=group, protocol=protocol, n=n,
                           correct=correct, valid=valid, invalid=n-valid, valid_wrong=valid-correct,
                           accuracy_pct=100*correct/n, valid_rate_pct=100*valid/n,
                           conditional_accuracy_pct=100*correct/valid,
                           macro_f1_pct=100*f1_score(truth, pred, labels=list(range(7)), average='macro', zero_division=0))
                metrics.append(row)
                key = (training, group, protocol)
                records[key], predictions[key] = rows, pred
                counts['/'.join(key)] = dict(Counter(labels[p] if p >= 0 else '<invalid>' for p in pred))
    for group in ('E_official', 'E', 'ET', 'T'):
        for protocol in ('sampled_seed0', 'greedy', 'constrained_greedy'):
            before, after = records['nd', group, protocol], records['nc', group, protocol]
            assert all(a['prompt'] == b['prompt'] and a['graph'] == b['graph'] for a, b in zip(before, after))
            if protocol != 'sampled_seed0':
                a, b = before[0]['diagnostic_config'], after[0]['diagnostic_config']
                for k in ('implementation_sha256', 'constraint_implementation_sha256', 'decoding', 'source_prefix', 'max_new_tokens'):
                    assert a[k] == b[k]
            if group == 'T':
                assert all(a['text'] == b['text'] for a, b in zip(before, after))
            p, q = predictions['nd', group, protocol], predictions['nc', group, protocol]
            comparisons.append(dict(group=group, protocol=protocol,
                wrong_to_correct=sum(a != t and b == t for a,b,t in zip(p,q,truth)),
                correct_to_wrong=sum(a == t and b != t for a,b,t in zip(p,q,truth)),
                invalid_to_correct=sum(a < 0 and b == t for a,b,t in zip(p,q,truth)),
                invalid_to_valid_wrong=sum(a < 0 and b >= 0 and b != t for a,b,t in zip(p,q,truth))))
    for name, values in [('metrics', metrics), ('nd_to_nc_paired', comparisons)]:
        with (OUT / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = dict(checks='542 unique ordered IDs, GT, prompts, graph IDs, source hashes, projector hashes/finite values, 6423 steps, same diagnostic code; T answers identical',
                   truth_counts=dict(Counter(labels[t] for t in truth)), weights=weight_audits, prediction_counts=counts)
    (OUT / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    lines = ['# NC→Cora NC 控制实验审计（2026-11-16）', '',
             '一个训练种子42；每条件一次采样seed0、一次G1贪心和一次G2约束贪心。T不使用Projector，不是独立重训样本。', '',
             '|训练任务|条件|协议|正确/542|ACC %|有效率 %|条件ACC %|Macro-F1 %|',
             '|---|---|---|---|---|---|---|---|']
    for r in metrics:
        lines.append(f"|{r['training']}|{r['group']}|{r['protocol']}|{r['correct']}|{r['accuracy_pct']:.2f}|{r['valid_rate_pct']:.2f}|{r['conditional_accuracy_pct']:.2f}|{r['macro_f1_pct']:.2f}|")
    lines += ['', '结论：NC训练明显改善E_official输出有效性，但未改善整体准确率；G1出现Neural_Networks/Case_Based集中预测。ET低于旧nd模型和T。不能把当前结果归结为单纯生成格式问题，也不能据此断言embedding不含判别信息或类别空间变化是唯一原因。', '',
              '限制：单训练种子；条件准确率对应不同选中子集；源图NC与Cora类别语义不一致；nd监督也包含源类别且保留标题，任务改变同时改变提示及监督内容。SimTeG上游来源仍需核查。T只是冻结LLM参考；ET-T也涉及提示及输入长度，不能直接归因为拓扑有害。', '',
              '主线仍为Table5 LP→LP。本实验归档为同任务NC也未自动恢复泛化的控制证据，不自动扩展训练矩阵。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    for r in metrics:
        if r['protocol'] == 'greedy':
            print(r)
    print('Paired:', comparisons)
    print('Truth:', summary['truth_counts'])
    print('Report:', OUT / 'REPORT.md')


if __name__ == '__main__':
    main()
