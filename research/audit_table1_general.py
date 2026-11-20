"""Recompute the 12 completed Table 1 general-model cells without inference."""
import ast
from collections import Counter
import contextlib
import csv
import gc
import io
from itertools import zip_longest
import json
from pathlib import Path
import re
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/table1/official_20261114_192146'
PAPER = {
    'ND': {'nc': {'arxiv': 74.29, 'pubmed': 92.42, 'cora': 87.82},
           'lp': {'arxiv': 90.53, 'pubmed': 86.31, 'cora': 81.91}},
    'HO': {'nc': {'arxiv': 75.01, 'pubmed': 94.45, 'cora': 87.82},
           'lp': {'arxiv': 92.04, 'pubmed': 89.81, 'cora': 84.41}},
}


def main():
    tree = ast.parse((ROOT / 'eval/eval_res.py').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in ('eval_arxiv_nc', 'eval_pubmed_nc', 'eval_cora_nc', 'eval_lp')]
    rows, details = [], []
    for dataset, directory in [('arxiv', 'ogbn-arxiv'), ('pubmed', 'pubmed'), ('cora', 'cora')]:
        data = torch.load(ROOT / 'dataset' / directory / 'processed_data.pt', map_location='cpu')
        labels = list(data.label_texts)
        short = ([label[:5] for label in labels] if dataset == 'arxiv' else
                 [label[18:] for label in labels] if dataset == 'pubmed' else
                 [label.split('_')[0] for label in labels])
        env = dict(torch=SimpleNamespace(load=lambda _: data), json=json, args=SimpleNamespace(sample=0))
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'official_scorers', 'exec'), env)
        for template in ('ND', 'HO'):
            for task in ('nc', 'lp'):
                path = RUN / f'general_model_{template}' / f'{dataset}_{task}.jsonl'
                reference = ROOT / 'dataset' / directory / ('sampled_2_10_test.jsonl' if task == 'nc'
                                                            else 'edge_sampled_2_10_only_test.jsonl')
                n = correct = empty = invalid = strict = ci_correct = missing_config = 0
                ids, ref_ids, configs = set(), set(), Counter()
                invalid_examples = []
                expected_path = f'./models/llaga-vicuna-7b-simteg-{template}-general_model-2-layer-mlp-projector'
                with path.open() as predictions, reference.open() as originals:
                    for line, original in zip_longest(predictions, originals):
                        assert line is not None and original is not None, f'Wrong sample count: {path}'
                        row, ref = json.loads(line), json.loads(original)
                        assert row['question_id'] == ref['id'], (path, n, 'ID/order')
                        assert row['gt'] == ref['conversations'][1]['value'], (path, n, 'GT')
                        idx = row['question_id']
                        key = tuple(idx) if isinstance(idx, list) else idx
                        ids.add(key)
                        ref_ids.add(tuple(ref['id']) if isinstance(ref['id'], list) else ref['id'])
                        if template == 'ND':
                            expected_graph = ref['graph'] if isinstance(ref['graph'][0], list) else [ref['graph']]
                            assert row['graph'] == expected_graph, (path, n, 'ND graph')
                        else:
                            centers = idx if isinstance(idx, list) else [idx]
                            assert row['graph'] == [[center] * 5 for center in centers], (path, n, 'HO graph')
                        config = row.get('eval_config')
                        if config is None:
                            missing_config += 1
                        else:
                            assert config['model_path'] == expected_path, (path, n, 'Projector')
                            assert config['model_base'] == './models/vicuna-7b-v1.5-16k'
                            for field, value in dict(task=task, input_mode='graph', prompt_style='official',
                                                     template=template, embedding='simteg',
                                                     use_hop=2 if template == 'ND' else 4,
                                                     do_sample=True, temperature=0.2, max_new_tokens=1024).items():
                                assert config[field] == value, (path, n, field)
                            configs[json.dumps(config, sort_keys=True)] += 1
                        answer = row['text'].strip()
                        empty += not bool(answer)
                        if task == 'lp':
                            truth = 'yes' in row['gt'].strip()
                            correct += ('yes' in answer) == truth
                            ci_correct += ('yes' in answer.lower()) == truth
                        else:
                            target = int(data.y[idx])
                            hits = [i for i, label in enumerate(short) if label.strip().lower() in answer.lower()]
                            exact = answer.lower() == labels[target].strip().lower()
                            valid = len(hits) == 1
                            hit = (valid and hits[0] == target) or (dataset != 'cora' and exact)
                            correct += hit
                            strict += exact
                            invalid += not (valid or any(answer.lower() == label.strip().lower() for label in labels))
                            if not valid and len(invalid_examples) < 3:
                                invalid_examples.append(dict(id=idx, text=answer[:1000]))
                        n += 1
                assert ids == ref_ids
                if task == 'nc':
                    assert len(ids) == n
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture):
                    env['eval_lp' if task == 'lp' else f'eval_{dataset}_nc'](str(path))
                original_output = capture.getvalue()
                name = 'overall_acc' if task == 'nc' and dataset != 'cora' else 'acc'
                printed = float(re.search(rf'\b{name}: ([0-9.]+)', original_output).group(1))
                assert abs(correct / n - printed) < .000051, (path, correct, original_output)
                paper = PAPER[template][task][dataset]
                result = dict(template=template, dataset=dataset, task=task, correct=correct, n=n,
                              accuracy_pct=correct/n*100, paper_pct=paper, delta_pp=correct/n*100-paper,
                              empty=empty, invalid=invalid if task == 'nc' else None,
                              strict_accuracy_pct=strict/n*100 if task == 'nc' else None,
                              case_insensitive_correct=ci_correct if task == 'lp' else None,
                              unique_ids=len(ids), missing_eval_config=missing_config,
                              config_variants=len(configs), ids_gt_graph_checks_passed=True,
                              official_score_matches=True)
                rows.append(result)
                details.append(dict(**result, file=str(path.relative_to(ROOT)),
                                    config_counts={k: v for k, v in configs.items()},
                                    invalid_examples=invalid_examples, official_stdout=original_output))
                print(f"{template} {dataset:6s} {task}: {correct}/{n}={correct/n*100:.4f}% "
                      f"paper={paper:.2f} delta={result['delta_pp']:+.4f}pp "
                      f"empty={empty} invalid={result['invalid']} missing_config={missing_config}", flush=True)
        del env, data
        gc.collect()
    order = {'arxiv': 0, 'pubmed': 1, 'cora': 2}
    rows.sort(key=lambda row: (row['template'] != 'ND', row['task'] != 'nc', order[row['dataset']]))
    with (RUN / 'general_model_audit_20261116.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RUN / 'general_model_audit_20261116.json').write_text(json.dumps(details, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
