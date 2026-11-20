"""Audit LP predictions without running a model; original ACC is preserved."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.lp_inputs import parse_binary


def audit(path, data_root):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError(f'Empty predictions: {path}')
    dataset = rows[0].get('lp_diagnostic_config', {}).get('dataset', 'cora')
    directory = {'arxiv': 'ogbn-arxiv', 'cora': 'cora', 'pubmed': 'pubmed'}[dataset]
    cfg = rows[0].get('lp_diagnostic_config')
    split = cfg.get('split', 'test') if cfg else 'test'
    refs = [json.loads(s) for s in (data_root / directory / f'edge_sampled_2_10_only_{split}.jsonl').read_text().splitlines()]
    start = max(0, cfg['start']) if cfg else 0
    end = cfg['end'] if cfg and cfg['end'] > 0 else len(refs)
    refs = refs[start:end]
    if len(rows) != len(refs):
        raise ValueError(f'Incomplete file {path}: {len(rows)}/{len(refs)}')
    counts = Counter()
    invalid = []
    pairs = set()
    for i, (row, ref) in enumerate(zip(rows, refs)):
        assert row['question_id'] == ref['id'], (path, i, 'ID')
        assert row['gt'] == ref['conversations'][1]['value'], (path, i, 'GT')
        if cfg:
            assert row['lp_diagnostic_config'] == cfg
        gt = row['gt'].strip().lower()
        assert gt in ('yes', 'no')
        text = row['text'].strip()
        # Exactly the author's case-sensitive yes-substring rule.
        original = 'yes' if 'yes' in text else 'no'
        counts['official_correct'] += original == gt
        pred = parse_binary(text)
        counts['valid'] += pred is not None
        counts['correct'] += pred == gt
        counts['empty'] += not text
        counts['positive'] += gt == 'yes'
        if pred is None:
            counts['invalid'] += 1
            reason = 'multiple_binary_keywords' if re.search(r'\byes\b', text, re.I) and re.search(r'\bno\b', text, re.I) else 'no_binary_keyword'
            invalid.append(dict(file=str(path), record_index=i, pair=row['question_id'], gt=gt, reason=reason, text=text))
        else:
            counts[{('yes','yes'):'TP', ('no','yes'):'FP', ('yes','no'):'FN', ('no','no'):'TN'}[gt,pred]] += 1
        pairs.add(tuple(sorted(row['question_id'])))
    n = len(rows)
    result = dict(file=str(path), n=n, unique_undirected_pairs=len(pairs), positive=counts['positive'],
                  official_accuracy_pct=100*counts['official_correct']/n,
                  overall_accuracy_pct=100*counts['correct']/n,
                  valid_rate_pct=100*counts['valid']/n,
                  conditional_accuracy_pct=100*counts['correct']/counts['valid'] if counts['valid'] else None,
                  invalid=counts['invalid'], empty=counts['empty'],
                  **{k: counts[k] for k in ('TP','FP','FN','TN')})
    return result, invalid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', required=True, type=Path)
    p.add_argument('--data-root', type=Path, default=Path('dataset'))
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, invalid = [], []
    for path in args.inputs:
        r, bad = audit(path, args.data_root)
        results.append(r)
        invalid.extend(bad)
        print(f"{path.name}: official={r['official_accuracy_pct']:.2f}%, auxiliary={r['overall_accuracy_pct']:.2f}%, valid={r['valid_rate_pct']:.2f}%, n={r['n']}")
    with (args.output / 'metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    (args.output / 'invalid_answers.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in invalid))
    (args.output / 'README.md').write_text('Original ACC uses the author’s case-sensitive yes-substring rule. Auxiliary parser accepts exactly one distinct whole-word yes/no (case insensitive); it is lexical, not a semantic judge. Invalid outputs count wrong for auxiliary overall ACC. Conditional ACC uses a selected subset. FP/FN cover valid answers only; invalid outcomes are separate. No logits are present, so AUC/AP are not computed.\n')


if __name__ == '__main__':
    main()
