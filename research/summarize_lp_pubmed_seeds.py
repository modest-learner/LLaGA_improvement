"""Validate all three training seeds before reporting matched Full/Center results."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit_lp_outputs import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/lp_phase2'))
    p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--model-base',type=Path,default=Path('models/vicuna-7b-v1.5-16k'))
    a=p.parse_args(); out=a.root/'pubmed_three_seed_summary';out.mkdir(parents=True,exist_ok=True)
    rows=[]; missing=[]
    for seed in [42,43,44]:
        for mode in ['full','center_repeat']:
            cp=Path(f'checkpoints/lp_phase2/llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed{seed}-bs16-{mode}')
            for ds in ['arxiv','cora','pubmed']:
                split='test' if ds=='pubmed' else 'val'
                path=a.root/f'arxiv-cora_to_pubmed/seed{seed}_{mode}/{ds}_{split}_{mode}.jsonl'
                if not path.exists():missing.append(str(path));continue
                result=subprocess.run([sys.executable,'research/check_lp_prediction.py',
                    '--checkpoint',str(cp),'--output',str(path),'--dataset',ds,'--split',split,
                    '--ablation',mode,'--data-root',str(a.data_root),'--model-base',str(a.model_base)],capture_output=True,text=True)
                if result.returncode in [10,20]:missing.append(str(path));continue
                if result.returncode:raise RuntimeError(result.stdout+result.stderr)
                score,_=audit(path,a.data_root)
                rows.append(dict(seed=seed,mode=mode,dataset=ds,split=split,**score,
                    balanced_accuracy_pct=50*(score['TP']/score['positive']+score['TN']/(score['n']-score['positive']))))
    (out/'status.json').write_text(json.dumps(dict(complete=not missing,files=len(rows),missing=missing),indent=2)+'\n')
    if rows:
        with (out/'metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if missing:
        (out/'REPORT.md').write_text('# 尚未完成\n\n完整文件：'+str(len(rows))+'/18。请查看status.json。\n')
        raise SystemExit(f'Incomplete: {len(rows)}/18 files; see {out}/status.json')
    lines=['# Pubmed方向三训练种子汇总','', '平均值±训练种子样本标准差；每个模型一次固定greedy推理。',
        '', '|数据集|模式|ACC (%)|Balanced ACC (%)|有效率 (%)|','|---|---|---:|---:|---:|']
    deltas=[]
    for ds in ['arxiv','cora','pubmed']:
        for mode in ['full','center_repeat']:
            z=[r for r in rows if r['dataset']==ds and r['mode']==mode]
            cells=[]
            for k in ['overall_accuracy_pct','balanced_accuracy_pct','valid_rate_pct']:
                v=[r[k] for r in z];cells.append(f'{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}')
            lines.append('|'+ '|'.join([ds,mode]+cells)+'|')
        delta=[next(r['overall_accuracy_pct'] for r in rows if r['dataset']==ds and r['mode']=='full' and r['seed']==s)-
            next(r['overall_accuracy_pct'] for r in rows if r['dataset']==ds and r['mode']=='center_repeat' and r['seed']==s) for s in [42,43,44]]
        deltas.append(f'\n{ds}配对Full−Center ACC增益：{statistics.mean(delta):.2f} ± {statistics.stdev(delta):.2f}个百分点。\n')
    (out/'REPORT.md').write_text('\n'.join(lines+deltas)+'\n')
    print('Complete:',out/'REPORT.md')


if __name__=='__main__':main()
