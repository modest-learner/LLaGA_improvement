"""Summarize matched LP source/target evaluations and independent training seeds."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.lp_inputs import parse_binary


def read(path, allow_partial_tail=False):
    data=path.read_text()
    if allow_partial_tail and data and not data.endswith('\n'):
        data=data.rsplit('\n',1)[0]+'\n' if '\n' in data else ''
    return [json.loads(s) for s in data.splitlines()]


def csv_write(path,rows):
    if rows:
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(a):
    out=a.root/'summary'
    metrics=[];missing=[];raw={}
    for seed in a.seeds:
        for mode in ('full','center_repeat'):
            for dataset in ('arxiv','pubmed','cora'):
                split='test' if dataset=='cora' else 'val'
                path=(Path(f'results/lp_clean/arxiv-pubmed_to_cora_trainseed{seed}_{mode}/screen/cora_{mode}.jsonl') if dataset=='cora' else
                      a.root/f'source_validation/seed{seed}_{mode}/{dataset}_val_{mode}.jsonl')
                if not path.exists():
                    missing.append(str(path));continue
                records=read(path,allow_partial_tail=True)
                refs=read(a.data_root/('ogbn-arxiv' if dataset=='arxiv' else dataset)/f'edge_sampled_2_10_only_{split}.jsonl')
                assert len(records)<=len(refs),f'Too many predictions: {path}'
                if len(records)<len(refs):
                    missing.append(f'{path} (in progress: {len(records)}/{len(refs)})');continue
                assert all(r['question_id']==q['id'] and r['gt']==q['conversations'][1]['value'] for r,q in zip(records,refs))
                cfg=records[0]['lp_diagnostic_config']
                assert all(r['lp_diagnostic_config']==cfg for r in records)
                assert cfg['dataset']==dataset and cfg['split']==split and cfg['ablation']==mode and cfg['greedy']
                assert f'seed{seed}-bs16-{mode}' in cfg['model_path']
                pred=[parse_binary(r['text']) for r in records];gt=[r['gt'] for r in records]
                correct=sum(p==y for p,y in zip(pred,gt));valid=sum(p is not None for p in pred)
                positive=gt.count('yes');negative=len(gt)-positive
                tp=sum(p=='yes' and y=='yes' for p,y in zip(pred,gt))
                tn=sum(p=='no' and y=='no' for p,y in zip(pred,gt))
                metrics.append(dict(mode=mode,seed=seed,dataset=dataset,split=split,n=len(gt),correct=correct,
                    official_accuracy_pct=100*sum(('yes' if 'yes' in r['text'].strip() else 'no')==r['gt'] for r in records)/len(gt),
                    accuracy_pct=100*correct/len(gt),valid_rate_pct=100*valid/len(gt),
                    balanced_accuracy_pct=50*(tp/positive+tn/negative),invalid=len(gt)-valid))
                raw[seed,mode,dataset]=(pred,gt)
    means=[]
    for dataset in ('arxiv','pubmed','cora'):
        for mode in ('full','center_repeat'):
            rows=[r for r in metrics if r['mode']==mode and r['dataset']==dataset]
            if not rows:
                continue
            item=dict(mode=mode,dataset=dataset,training_seeds=len(rows))
            for key in ('accuracy_pct','balanced_accuracy_pct','valid_rate_pct'):
                values=[r[key] for r in rows]
                item[key+'_mean']=statistics.mean(values)
                item[key+'_sample_sd']=statistics.stdev(values) if len(values)>1 else None
            means.append(item)
    deltas=[]
    for seed in a.seeds:
        for dataset in ('arxiv','pubmed','cora'):
            if (seed,'full',dataset) not in raw or (seed,'center_repeat',dataset) not in raw:
                continue
            x,truth=raw[seed,'full',dataset];y,t=raw[seed,'center_repeat',dataset]
            assert truth==t
            gain=sum(p==z and q!=z for p,q,z in zip(x,y,truth));loss=sum(p!=z and q==z for p,q,z in zip(x,y,truth))
            deltas.append(dict(seed=seed,dataset=dataset,full_gain=gain,full_loss=loss,delta_accuracy_pp=100*(gain-loss)/len(t)))
    delta_summary=[]
    for dataset in ('arxiv','pubmed','cora'):
        values=[r['delta_accuracy_pp'] for r in deltas if r['dataset']==dataset]
        if values:
            delta_summary.append(dict(dataset=dataset,training_seeds=len(values),mean_delta_pp=statistics.mean(values),
                                      sample_sd_pp=statistics.stdev(values) if len(values)>1 else None))
    baseline_path=a.root/'simple_baselines/metrics.csv'
    baseline_rows=list(csv.DictReader(baseline_path.open())) if baseline_path.exists() else []
    if not baseline_rows:
        missing.append(str(baseline_path))
    csv_write(out/'llm_per_seed.csv',metrics);csv_write(out/'llm_mean_sd.csv',means)
    csv_write(out/'paired_deltas.csv',deltas);csv_write(out/'paired_delta_mean_sd.csv',delta_summary)
    (out/'status.json').write_text(json.dumps(dict(complete=not missing,missing=missing,llm_files=len(metrics)),indent=2)+'\n')
    lines=['# 匹配LP下一轮结果','',f'状态：{"完整" if not missing else "尚未完整"}；LLM评估文件{len(metrics)}份。',
           '', '|输入模型|数据集|训练种子数|ACC均值|样本标准差（百分点）|', '|---|---|---:|---:|---:|']
    for r in means:
        sd=r['accuracy_pct_sample_sd'];sd_text=f'{sd:.2f}' if sd is not None else '尚无'
        lines.append(f"|{r['mode']}|{r['dataset']}|{r['training_seeds']}|{r['accuracy_pct_mean']:.2f}%|{sd_text}|")
    lines+=['','源验证与目标候选难度/正负比例不相同，不能把分差全部归因跨图。使用Balanced ACC辅助比较，仍非因果隔离。基线阈值和MLP选择使用源验证，因此其源验证成绩有选择偏差。',
            'LLM仍为自由生成ACC，没有连续打分，AUC/AP仅在simple_baselines中可用。无效输出计错；词法解析不是语义裁判。',
            '训练种子为统计单位；没有给同一贪心推理人为重复次数。目标图只有Cora，多种子稳定不等于多目标泛化。']
    if missing:
        lines+=['','缺失项：']+['- '+s for s in missing]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Summary:',out/'REPORT.md','; complete=',not missing)
    if a.require_complete and missing:
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/lp_next_round'))
    p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args()
    out=a.root/'summary';out.mkdir(parents=True,exist_ok=True)
    with (out/'.summary.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        summarize(a)


if __name__=='__main__':
    main()
