"""Paired output-protocol audit; GT is read only after prediction files exist."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main(args):
    directory = {'cora': 'cora', 'pubmed': 'pubmed', 'arxiv': 'ogbn-arxiv'}[args.dataset]
    data = torch.load(Path('dataset') / directory / 'processed_data.pt')
    official = read(Path('dataset') / directory / 'sampled_2_10_test.jsonl')
    labels = list(data.label_texts)
    if args.dataset == 'cora':
        short = [label.split('_')[0].lower().strip() for label in labels]
    elif args.dataset == 'pubmed':
        short = [label[18:].lower().strip() for label in labels]
    else:
        short = [label[:5].lower().strip() for label in labels]

    def parse(text):
        answer = text.strip().lower()
        exact = [i for i, label in enumerate(labels) if label.strip().lower() == answer]
        if exact:
            return exact[0]
        hits = [i for i, name in enumerate(short) if name in answer]
        return hits[0] if len(hits) == 1 else -1

    metrics, paired, class_counts, invalid_answers = [], [], {}, []
    for group in ('E_official', 'E', 'ET', 'T'):
        rows = {mode: read(args.output_dir / f'{args.dataset}_{group}_{mode}.jsonl')
                for mode in ('greedy', 'constrained_greedy')}
        left = rows['greedy'][0]['diagnostic_config']
        right = rows['constrained_greedy'][0]['diagnostic_config']
        for key in ('source_sha256', 'checkpoint_sha256', 'source_eval_config'):
            assert left[key] == right[key], f'Paired models/inputs differ: {key}'
        ids = [r['question_id'] for r in rows['greedy']]
        assert ids and len(set(ids)) == len(ids)
        assert ids == [r['id'] for r in official[:len(ids)]]
        truth = np.array([int(data.y[idx]) for idx in ids])
        source = read(args.source_dir / f'{args.dataset}_{group}_inferseed0.jsonl')[:len(ids)]
        assert ids == [r['question_id'] for r in source]
        rows['sampled_seed0'] = source
        predictions = {}
        for mode, records in rows.items():
            assert [r['question_id'] for r in records] == ids
            assert all(r['prompt'] == s['prompt'] and r['gt'] == s['gt'] for r, s in zip(records, source))
            assert all(r['gt'] == ref['conversations'][1]['value'] for r, ref in zip(records, official))
            pred = np.array([parse(r['text']) for r in records])
            predictions[mode] = pred
            for record, category in zip(records, pred):
                if category < 0:
                    invalid_answers.append(dict(group=group, protocol=mode,
                                                question_id=record['question_id'], text=record['text']))
            if mode == 'constrained_greedy':
                assert all(r['text'] in labels for r in records)
                assert np.all(pred >= 0)
            counts = Counter(labels[i] if i >= 0 else '<invalid>' for i in pred)
            class_counts[f'{group}/{mode}'] = dict(counts)
            row = dict(group=group, protocol=mode, n=len(ids), correct=int((pred == truth).sum()),
                       accuracy_pct=float((pred == truth).mean() * 100), invalid=int((pred < 0).sum()),
                       macro_f1_pct=float(f1_score(truth, pred, labels=list(range(len(labels))),
                                                 average='macro', zero_division=0) * 100))
            metrics.append(row)
            print(f"{group:10s} {mode:20s}: {row['correct']}/{len(ids)}, "
                  f"ACC={row['accuracy_pct']:.2f}%, invalid={row['invalid']}, Macro-F1={row['macro_f1_pct']:.2f}%")
        after = predictions['constrained_greedy']
        for before_mode in ('greedy', 'sampled_seed0'):
            before = predictions[before_mode]
            row = dict(group=group, before=before_mode, after='constrained_greedy',
                       old_invalid=int((before < 0).sum()),
                       invalid_to_correct=int(((before < 0) & (after == truth)).sum()),
                       valid_wrong_to_correct=int(((before >= 0) & (before != truth) & (after == truth)).sum()),
                       old_correct_to_wrong=int(((before == truth) & (after != truth)).sum()),
                       changed_prediction=int((before != after).sum()),
                       net_correct_change=int((after == truth).sum() - (before == truth).sum()))
            paired.append(row)
            print(f"  vs {before_mode}: invalid→correct={row['invalid_to_correct']}, "
                  f"correct→wrong={row['old_correct_to_wrong']}, net={row['net_correct_change']:+d}")
    for name, values in [('metrics', metrics), ('paired', paired)]:
        with (args.output_dir / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (args.output_dir / 'class_counts.json').write_text(json.dumps(class_counts, ensure_ascii=False, indent=2) + '\n')
    # Deliberately omit GT so descriptive/refusal/multi-class answers can be audited blind.
    with (args.output_dir / 'invalid_answers_blind.jsonl').open('w') as handle:
        for record in invalid_answers:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cora', 'pubmed', 'arxiv'], default='cora')
    parser.add_argument('--source_dir', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    main(parser.parse_args())
