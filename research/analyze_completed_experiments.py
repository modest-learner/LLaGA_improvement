"""Audit Table 5's eight cells and Table 8 diagnostics; no model inference.

Run from repository root using the llaga Python environment.
"""
import ast
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/table8/HO_linear_simteg_trainseed42'
DIAG = ROOT / 'results/table8/diagnostics_HO_linear_simteg_trainseed42'
OUT = ROOT / 'results/analysis_20261116'


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    data = torch.load(ROOT / 'dataset/cora/processed_data.pt')
    labels = list(data.label_texts)
    short = [label.split('_')[0].strip().lower() for label in labels]
    gold = {task: read(ROOT / 'dataset/cora' / name) for task, name in
            [('nc', 'sampled_2_10_test.jsonl'), ('lp', 'edge_sampled_2_10_only_test.jsonl')]}
    y = np.array([int(data.y[row['id']]) for row in gold['nc']])
    tree = ast.parse((ROOT / 'eval/eval_res.py').read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('eval_lp', 'eval_cora_nc')]
    env = dict(torch=SimpleNamespace(load=lambda _: data), json=json, args=SimpleNamespace(sample=0))
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'official_scorer', 'exec'), env)
    audits = []

    def audit(path, task):
        rows = read(path)
        expected = gold[task]
        key = lambda value: tuple(value) if isinstance(value, list) else value
        ids = [key(r['question_id']) for r in rows]
        assert len(set(ids)) == len(expected) == len(rows), path
        assert ids == [key(r['id']) for r in expected], path
        assert all(r['gt'] == ref['conversations'][1]['value'] for r, ref in zip(rows, expected)), path
        if task == 'nc':
            hits = [[i for i, name in enumerate(short) if name in r['text'].strip().lower()] for r in rows]
            pred = np.array([h[0] if len(h) == 1 else -1 for h in hits])
            correct = pred == y
        else:
            pred = np.array(['yes' in r['text'].strip() for r in rows])
            truth = np.array(['yes' in r['gt'].strip() for r in rows])
            correct = pred == truth
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            env['eval_cora_nc' if task == 'nc' else 'eval_lp'](str(path))
        score = float(re.search(r'\bacc: ([0-9.]+)', buffer.getvalue()).group(1))
        assert abs(score - correct.mean()) < .000051, path
        audits.append(dict(file=str(path.relative_to(ROOT)), n=len(rows), unique=len(set(ids)),
                           empty=sum(not r['text'].strip() for r in rows),
                           correct=int(correct.sum()), accuracy_pct=float(correct.mean()*100),
                           invalid=int((pred == -1).sum()) if task == 'nc' else None))
        return rows, correct.astype(int), pred

    table5 = []
    for template in ('HO', 'ND'):
        for arch in ('linear', 'mlp2'):
            for lr in ('2e-3', '2e-5'):
                path = ROOT / f'results/table5/cora_{template}_{arch}_lr{lr}_seed42_bs16_lp.jsonl'
                rows, correct, pred = audit(path, 'lp')
                truth = np.array(['yes' in r['gt'].strip() for r in rows])
                ckpt = ROOT / f'checkpoints/table5/llaga-{template}-arxiv-pubmed-lp-{arch}-lr{lr}-seed42-bs16'
                state = json.loads((ckpt/'trainer_state.json').read_text())
                config = json.loads((ckpt/'config.json').read_text())
                assert state['global_step'] == state['max_steps'] == 6423, ckpt
                assert config['mm_projector_type'] == ('linear' if arch == 'linear' else '2-layer-mlp')
                losses = [entry['loss'] for entry in state['log_history'] if 'loss' in entry]
                rates = [entry['learning_rate'] for entry in state['log_history'] if 'learning_rate' in entry]
                assert abs(max(rates)-float(lr)) < 1e-10
                weights = torch.load(ckpt/'mm_projector.bin', map_location='cpu')
                assert all(bool(torch.isfinite(v).all()) for v in weights.values())
                table5.append(dict(template=template, projector=arch, learning_rate=lr,
                                   seed=42, n=680, correct=int(correct.sum()), accuracy_pct=float(correct.mean()*100),
                                   case_insensitive_correct=sum(('yes' in r['text'].lower()) == ('yes' in r['gt'].lower()) for r in rows),
                                   false_positive=int((pred & ~truth).sum()), false_negative=int((~pred & truth).sum()),
                                   last1000_mean_loss=statistics.mean(losses[-1000:]),
                                   training_hours=state['log_history'][-1]['train_runtime']/3600,
                                   paper_reference_pct=87.35 if template == 'HO' else 86.47,
                                   source=str(path.relative_to(ROOT))))

    arrays, predictions, records, configs = {}, {}, {}, {}
    for group in ('E_official', 'E', 'ET', 'T', 'center_only', 'center_repeat', 'neighbor_shuffle'):
        arr, preds, recs = [], [], []
        for seed in range(3):
            path = BASE/f'cora_{group}_inferseed{seed}.jsonl' if group in ('E_official','E','ET','T') else (
                DIAG/f'generation/cora_ET_{group}_aseed42_inferseed{seed}.jsonl')
            rows, correct, pred = audit(path, 'nc')
            if group in ('center_only','center_repeat','neighbor_shuffle'):
                conf = rows[0]['diagnostic_config']
                assert all(row['diagnostic_config']==conf for row in rows)
                assert conf['inference_seed']==seed and conf['ablation']==group and conf['decoding']=='generate'
                assert conf['source_sha256']==digest(conf['source'])
                assert conf['checkpoint_sha256']==digest(Path(conf['source_eval_config']['model_path'])/'mm_projector.bin')
                source_rows=read(conf['source'])
                assert all(r['prompt']==s['prompt'] for r,s in zip(rows,source_rows))
                assert all(r['graph']==[[r['question_id']]*(1 if group=='center_only' else 5)] for r in rows)
                if group=='neighbor_shuffle':
                    assert conf['ablation_seed']==42
                    ids=[r['question_id'] for r in rows]
                    order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(42)).tolist()
                    donor={ids[order[i]]:ids[order[(i+1)%len(ids)]] for i in range(len(ids))}
                    assert all(r['donor_id']==donor[r['question_id']] for r in rows)
                configs[f'{group}_{seed}']=conf
            arr.append(correct);preds.append(pred);recs.append(rows)
        arrays[group]=np.array(arr);predictions[group]=np.array(preds);records[group]=recs
    generation=[]
    for group, arr in arrays.items():
        acc=arr.mean(axis=1)*100
        generation.append(dict(group=group,correct_per_run=arr.sum(axis=1).tolist(),mean_pct=float(acc.mean()),
                               sample_std_pp=float(acc.std(ddof=1)),invalid_per_run=(predictions[group]==-1).sum(axis=1).tolist(),
                               invalid_mean_pct=float((predictions[group]==-1).mean()*100)))
    rng=np.random.default_rng(42)
    comparisons=[]
    for a,b in [('ET','center_only'),('ET','center_repeat'),('ET','neighbor_shuffle'),
                ('center_only','T'),('center_repeat','T'),('neighbor_shuffle','T'),('ET','T')]:
        delta=(arrays[a]-arrays[b]).mean(axis=0)
        boot=np.concatenate([delta[rng.integers(0,len(y),(1000,len(y)))].mean(axis=1)*100 for _ in range(10)])
        comparisons.append(dict(a=a,b=b,delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
             gained_per_run=[int(((arrays[a][i]==1)&(arrays[b][i]==0)).sum()) for i in range(3)],
             lost_per_run=[int(((arrays[a][i]==0)&(arrays[b][i]==1)).sum()) for i in range(3)],
             changed_prediction_per_run=(predictions[a]!=predictions[b]).sum(axis=1).tolist()))

    scoring=[];recovery=[];classwise=[]
    for group in ('E_official','E','ET','T'):
        path=DIAG/f'label_score/cora_{group}_full_labelscore.jsonl'
        rows,correct,pred=audit(path,'nc')
        conf=rows[0]['diagnostic_config']
        assert all(row['diagnostic_config']==conf for row in rows)
        assert conf['decoding']=='label_score' and conf['ablation']=='full'
        assert conf['source_sha256']==digest(conf['source'])
        src_conf=conf['source_eval_config']
        weight_file=Path(src_conf['model_path'])/('config.json' if group=='T' else 'mm_projector.bin')
        assert conf['checkpoint_sha256']==digest(weight_file)
        assert all(r['prompt']==s['prompt'] for r,s in zip(rows,records[group][0]))
        for row in rows:
            scores=row['candidate_scores']
            assert set(scores)==set(labels)
            assert row['text']==max(scores,key=lambda name:scores[name]['mean_logprob'])
            assert row['prediction_sum_logprob']==max(scores,key=lambda name:scores[name]['sum_logprob'])
            for score in scores.values():
                assert np.isfinite(score['mean_logprob']) and score['tokens']>0
                assert abs(score['mean_logprob']*score['tokens']-score['sum_logprob'])<1e-4
        sum_pred=[r['prediction_sum_logprob'] for r in rows]
        scoring.append(dict(group=group,correct=int(correct.sum()),accuracy_pct=float(correct.mean()*100),
                            sum_score_correct=sum(p==labels[t] for p,t in zip(sum_pred,y)),
                            sum_score_accuracy_pct=100*sum(p==labels[t] for p,t in zip(sum_pred,y))/len(y),
                            mean_prediction_counts=dict(Counter(r['text'] for r in rows)),
                            sum_prediction_counts=dict(Counter(sum_pred)),
                            token_counts={k:v['tokens'] for k,v in rows[0]['candidate_scores'].items()},
                            mean_token_logprob_by_class={k:statistics.mean(r['candidate_scores'][k]['mean_logprob'] for r in rows) for k in labels}))
        for seed in range(3):
            old_correct=arrays[group][seed].astype(bool)
            old_invalid=predictions[group][seed]==-1
            new_correct=correct.astype(bool)
            recovery.append(dict(group=group,inference_seed=seed,old_invalid=int(old_invalid.sum()),
                invalid_recovered=int((old_invalid&new_correct).sum()),invalid_still_wrong=int((old_invalid&~new_correct).sum()),
                valid_wrong_recovered=int((~old_invalid&~old_correct&new_correct).sum()),
                old_correct_lost=int((old_correct&~new_correct).sum()),
                both_correct=int((old_correct&new_correct).sum()),net_correct_change=int(new_correct.sum()-old_correct.sum())))
        for i,label in enumerate(labels):
            classwise.append(dict(group=group,protocol='label_score_mean',label=label,n=int((y==i).sum()),
                                  recall_pct=float(correct[y==i].mean()*100),predicted_nodes=int((pred==i).sum())))
    for group in arrays:
        for i,label in enumerate(labels):
            classwise.append(dict(group=group,protocol='generate_3run_mean',label=label,n=int((y==i).sum()),
                                   recall_pct=float(arrays[group][:,y==i].mean()*100),
                                   predicted_nodes=float((predictions[group]==i).sum(axis=1).mean())))
    report=dict(date='2026-11-16',table5=table5,generation=generation,paired_comparisons=comparisons,
                candidate_scoring=scoring,recovery=recovery,classwise=classwise,audits=audits,
                diagnostic_configs=configs,notes=['Table5 uses the original seed42 single-inference file for every cell.',
                    'Generation std is across inference repeats, not training seeds.',
                    'Bootstrap resamples nodes after averaging repeats; exploratory, no graph correlation modeled.',
                    'Candidate mean and sum protocols remain separate; no post-hoc best-rule selection.'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    write_csv(OUT/'table5_eight_configs.csv',table5)
    write_csv(OUT/'table8_generation.csv',generation)
    write_csv(OUT/'table8_candidate_recovery.csv',recovery)
    write_csv(OUT/'table8_classwise.csv',classwise)
    write_csv(OUT/'file_audit.csv',audits)
    for name in ('table5','generation','paired_comparisons','candidate_scoring','recovery'):
        print(name,json.dumps(report[name],ensure_ascii=False))
    print(f'ALL {len(audits)} FILES PASSED ID/GT/SCORE CHECKS. Summary: {OUT}/summary.json')


if __name__=='__main__':
    main()
