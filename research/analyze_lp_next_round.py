"""Audit and analyze completed matched LP runs without loading models on a GPU."""
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.lp_inputs import parse_binary
from research.run_lp_simple_baselines import metrics, select_threshold

ROOT = Path('results/lp_next_round')
OUT = ROOT / 'analysis_20261117'
DATA = Path('dataset_research/lp_bert_disjoint_v1')
SEEDS = (42, 43, 44)
DATASETS = ('arxiv', 'pubmed', 'cora')


def write_csv(name, rows):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    audit = [];predictions = {};references = {};rows = [];pair_rows = []
    for dataset in DATASETS:
        split = 'test' if dataset == 'cora' else 'val'
        refs = [json.loads(s) for s in (DATA / ('ogbn-arxiv' if dataset == 'arxiv' else dataset) /
                f'edge_sampled_2_10_only_{split}.jsonl').read_text().splitlines()]
        references[dataset] = (np.array([r['id'] for r in refs]),
                               np.array([r['conversations'][1]['value'] == 'yes' for r in refs]))
    for seed in SEEDS:
        for mode in ('full', 'center_repeat'):
            checkpoint = Path(f'checkpoints/lp_clean/llaga-HO-arxiv-pubmed-lp-linear-bert-lr2e-3-seed{seed}-bs16-{mode}')
            # Reuse the production preflight for exact code/data/weight provenance.
            cmd = [sys.executable, 'research/check_lp_training.py', '--checkpoint', str(checkpoint),
                   '--log', f'logs/lp_clean/arxiv-pubmed_seed{seed}_{mode}/train.log',
                   '--seed', str(seed), '--ablation', mode]
            audit.append(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip())
            for dataset in DATASETS:
                split = 'test' if dataset == 'cora' else 'val'
                path = (Path(f'results/lp_clean/arxiv-pubmed_to_cora_trainseed{seed}_{mode}/screen/cora_{mode}.jsonl')
                        if dataset == 'cora' else ROOT / f'source_validation/seed{seed}_{mode}/{dataset}_val_{mode}.jsonl')
                cmd = [sys.executable, 'research/check_lp_prediction.py', '--checkpoint', str(checkpoint),
                       '--output', str(path), '--dataset', dataset, '--split', split, '--ablation', mode]
                audit.append(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip())
                records = [json.loads(s) for s in path.read_text().splitlines()]
                parsed = [parse_binary(r['text']) for r in records]
                assert all(p in ('yes', 'no') for p in parsed)
                pred = np.array([p == 'yes' for p in parsed])
                official = np.array(['yes' in r['text'].strip() for r in records])
                assert np.array_equal(pred, official)
                labels = references[dataset][1]
                predictions[seed, mode, dataset] = pred
                tp = int((pred & labels).sum());tn = int((~pred & ~labels).sum())
                fp = int((pred & ~labels).sum());fn = int((~pred & labels).sum())
                rows.append(dict(seed=seed,mode=mode,dataset=dataset,n=len(labels),TP=tp,FP=fp,FN=fn,TN=tn,
                    accuracy_pct=100*float((pred==labels).mean()),
                    balanced_accuracy_pct=50*(tp/(tp+fn)+tn/(tn+fp)),
                    positive_recall_pct=100*tp/(tp+fn),negative_recall_pct=100*tn/(tn+fp),
                    positive_precision_pct=100*tp/(tp+fp),valid_rate_pct=100.0))
    (OUT/'audit.txt').write_text('\n'.join(audit)+'\n')
    baseline_rows = list(csv.DictReader((ROOT/'simple_baselines/metrics.csv').open()))
    assert len(baseline_rows) == 24
    scores = {}
    for row in baseline_rows:
        name = row['method'];dataset = row['dataset'];seed = row['seed']
        suffix = f'_seed{seed}' if seed else ''
        data = np.load(ROOT/f'simple_baselines/{dataset}_{name}{suffix}_scores.npz')
        pairs, labels = references[dataset]
        assert np.array_equal(data['pairs'], pairs) and np.array_equal(data['labels'], labels)
        result = metrics(labels,data['scores'],float(data['threshold']))
        for key, val in result.items():
            assert np.isclose(float(row[key]),val), (name,dataset,seed,key)
        scores[name,seed,dataset] = data['scores'],float(data['threshold'])
    for method,seed in sorted({(r['method'],r['seed']) for r in baseline_rows}):
        xs = np.concatenate([scores[method,seed,d][0] for d in ('arxiv','pubmed')])
        ys = np.concatenate([references[d][1] for d in ('arxiv','pubmed')])
        gs = np.concatenate([np.full(len(references[d][1]),i) for i,d in enumerate(('arxiv','pubmed'))])
        threshold,_ = select_threshold(xs,ys,gs)
        assert np.isclose(threshold,scores[method,seed,'cora'][1])
    means = []
    for dataset in DATASETS:
        for mode in ('full','center_repeat'):
            selected = [r for r in rows if r['dataset']==dataset and r['mode']==mode]
            item = dict(dataset=dataset,mode=mode)
            for key in ('accuracy_pct','balanced_accuracy_pct','positive_recall_pct','negative_recall_pct','FP','FN'):
                values = [r[key] for r in selected]
                item[key+'_mean']=statistics.mean(values);item[key+'_sd']=statistics.stdev(values)
            means.append(item)
        for seed in SEEDS:
            labels=references[dataset][1]
            f=predictions[seed,'full',dataset];c=predictions[seed,'center_repeat',dataset]
            gain=int(((f==labels)&(c!=labels)).sum());loss=int(((f!=labels)&(c==labels)).sum())
            pair_rows.append(dict(seed=seed,dataset=dataset,gain=gain,loss=loss,delta_pp=100*(gain-loss)/len(labels)))
    # All strata are exploratory. Cosine cutoffs come only from source validation negatives.
    source_cos=np.concatenate([scores['cosine','',d][0][~references[d][1]] for d in ('arxiv','pubmed')])
    cutoff=float(np.quantile(source_cos,.75))
    stratum_rows=[];complement=[];consensus={}
    for dataset in DATASETS:
        pairs,labels=references[dataset];cn=scores['CN','',dataset][0]
        cosine=scores['cosine','',dataset][0];cn_pred=cn>=scores['CN','',dataset][1]
        for seed in SEEDS:
            for mode in ('full','center_repeat'):
                pred=predictions[seed,mode,dataset]
                for key,mask in [('CN=0',cn==0),('CN>0',cn>0),
                                 ('negative_high_cosine',(cosine>=cutoff)&~labels),
                                 ('negative_other',(cosine<cutoff)&~labels)]:
                    if not mask.any():continue
                    positives=int(labels[mask].sum());negatives=int((~labels[mask]).sum())
                    fp=int((pred[mask]&~labels[mask]).sum());fn=int((~pred[mask]&labels[mask]).sum())
                    stratum_rows.append(dict(dataset=dataset,seed=seed,mode=mode,stratum=key,n=int(mask.sum()),
                        positives=positives,negatives=negatives,FP=fp,FN=fn,
                        accuracy_pct=100*float((pred[mask]==labels[mask]).mean()),
                        fpr_pct=100*fp/negatives if negatives else None,
                        fnr_pct=100*fn/positives if positives else None))
            f=predictions[seed,'full',dataset]
            complement.append(dict(dataset=dataset,seed=seed,
                full_correct_CN_wrong=int(((f==labels)&(cn_pred!=labels)).sum()),
                full_wrong_CN_correct=int(((f!=labels)&(cn_pred==labels)).sum()),
                both_wrong=int(((f!=labels)&(cn_pred!=labels)).sum())))
        full=np.stack([predictions[s,'full',dataset] for s in SEEDS])
        center=np.stack([predictions[s,'center_repeat',dataset] for s in SEEDS])
        consensus[dataset]=dict(full_wrong_all_seeds=int((full!=labels).all(axis=0).sum()),
            full_FP_all_seeds=int((full.all(axis=0)&~labels).sum()),
            full_FN_all_seeds=int(((~full).all(axis=0)&labels).sum()),
            full_prediction_disagreement=int((full.any(axis=0)!=full.all(axis=0)).sum()),
            center_prediction_disagreement=int((center.any(axis=0)!=center.all(axis=0)).sum()),
            full_benefit_all_seeds=int(((full==labels)&(center!=labels)).all(axis=0).sum()),
            full_harm_all_seeds=int(((full!=labels)&(center==labels)).all(axis=0).sum()))
        if dataset=='cora':
            examples=[]
            for i,(u,v) in enumerate(pairs):
                examples.append(dict(u=int(u),v=int(v),label=int(labels[i]),CN=float(cn[i]),cosine=float(cosine[i]),
                    full_yes_count=int(full[:,i].sum()),center_yes_count=int(center[:,i].sum()),
                    full_wrong_all_seeds=bool((full[:,i]!=labels[i]).all()),
                    **{f'{mode}_seed{s}':int(predictions[s,mode,dataset][i]) for s in SEEDS for mode in ('full','center_repeat')}))
            write_csv('cora_pair_analysis.csv',examples)
    write_csv('llm_confusion.csv',rows);write_csv('llm_mean_sd.csv',means)
    write_csv('paired_deltas.csv',pair_rows);write_csv('exploratory_strata.csv',stratum_rows)
    write_csv('full_vs_CN.csv',complement)
    summary=dict(completed_checkpoints=6,validated_prediction_files=18,
        validated_prediction_records=sum(r['n'] for r in rows),validated_baseline_rows=24,
        cosine_source_negative_p75=cutoff,consensus=consensus,
        cora_delta_mean=statistics.mean(r['delta_pp'] for r in pair_rows if r['dataset']=='cora'),
        cora_delta_sd=statistics.stdev(r['delta_pp'] for r in pair_rows if r['dataset']=='cora'))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('LLM MEANS',json.dumps(means))
    for method in ('cosine','CN','AA','RA','PA','endpoint_MLP'):
        selected=[r for r in baseline_rows if r['method']==method and r['dataset']=='cora']
        print('CORA BASELINE',method,{key:(statistics.mean(float(r[key]) for r in selected),
             statistics.stdev(float(r[key]) for r in selected) if len(selected)>1 else None)
             for key in ('accuracy_pct','balanced_accuracy_pct','roc_auc','average_precision')})
    for dataset in DATASETS:
        for group in ('CN=0','CN>0','negative_high_cosine','negative_other'):
            for mode in ('full','center_repeat'):
                selected=[r for r in stratum_rows if r['dataset']==dataset and r['stratum']==group and r['mode']==mode]
                print('STRATUM',dataset,group,mode,'n=',selected[0]['n'],'positive=',selected[0]['positives'],
                      'mean_FP=',statistics.mean(r['FP'] for r in selected),
                      'mean_FN=',statistics.mean(r['FN'] for r in selected))
    print('COMPLEMENT',json.dumps(complement))


if __name__=='__main__':
    main()
