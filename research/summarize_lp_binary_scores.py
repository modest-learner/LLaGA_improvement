"""Report fixed candidate and source-selected thresholds separately from generation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,json,statistics
import numpy as np
from research.run_lp_simple_baselines import metrics,select_threshold
from research.check_lp_prediction import sha


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('results/lp_phase2'));a=p.parse_args()
 rows=[];distributions=[];provenance={}
 for seed in [42,43,44]:
  for mode in ['full','center_repeat']:
   sets={}
   for ds in ['arxiv','cora','pubmed']:
    split='test' if ds=='pubmed' else 'val'
    path=a.root/'binary_scores'/f'seed{seed}_{mode}'/f'{ds}_{split}.jsonl'
    ref=a.root/f'arxiv-cora_to_pubmed/seed{seed}_{mode}/{ds}_{split}_{mode}.jsonl'
    z=[json.loads(l) for l in path.open()];orig=[json.loads(l) for l in ref.open()]
    recipe=json.loads(path.with_suffix('.recipe.json').read_text())
    assert not recipe['pilot'] and recipe['input_sha256']==sha(ref)
    assert recipe['seed']==seed and recipe['ablation']==mode and recipe['dataset']==ds
    assert len(z)==len(orig) and all(x['pair']==r['question_id'] and x['gt']==r['gt'] and x['original_text']==r['text'] for x,r in zip(z,orig))
    y=np.array([int(r['gt']=='yes') for r in z]);score=np.array([r['score'] for r in z])
    assert np.isfinite(score).all() and set(y)=={0,1}
    sets[ds]=(y,score,z);provenance[str(path)]=sha(path)
   # Target labels/scores never enter this threshold selection call.
   threshold,_=select_threshold(np.concatenate([sets[g][1] for g in ['arxiv','cora']]),
     np.concatenate([sets[g][0] for g in ['arxiv','cora']]),
     np.concatenate([np.full(len(sets[g][0]),i) for i,g in enumerate(['arxiv','cora'])]))
   for ds,(y,score,z) in sets.items():
    for protocol,t in [('candidate_zero',0.),('source_threshold',threshold)]:
     m=metrics(y,score,t)
     gen=np.array(['yes' in r['original_text'] for r in z])
     pred=score>=t
     rows.append(dict(seed=seed,mode=mode,dataset=ds,protocol=protocol,**m,
      original_accuracy_pct=float((gen==y).mean()*100),
      original_disagreement=int((pred!=gen).sum()),
      corrected=int(((pred==y)&(gen!=y)).sum()),harmed=int(((pred!=y)&(gen==y)).sum())))
    for label in [0,1]:
     x=score[y==label];q=np.quantile(x,[.05,.25,.5,.75,.95])
     distributions.append(dict(seed=seed,mode=mode,dataset=ds,label=label,n=len(x),
      q05=q[0],q25=q[1],median=q[2],q75=q[3],q95=q[4],source_threshold=threshold,
      positive_pct_zero=float((x>=0).mean()*100),positive_pct_source=float((x>=threshold).mean()*100)))
 out=a.root/'binary_scores_summary';out.mkdir(parents=True,exist_ok=True)
 for name,data in [('metrics.csv',rows),('distributions.csv',distributions)]:
  with (out/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 lines=['# 固定候选分数诊断','', '原生成ACC单独保留在metrics.csv；candidate_zero为logP(yes)-logP(no)>=0；source_threshold只用Arxiv/Cora验证集等权Balanced ACC选择。',
  '候选只评分第一个yes/no token，不计EOS，不做长度归一化；不是完整回答正确性的概率。阈值边界相等时判yes。源验证结果有选择偏差。',
  '', '|模式|协议|Pubmed ACC|Balanced ACC|ROC-AUC|AP|','|---|---|---:|---:|---:|---:|']
 for mode in ['full','center_repeat']:
  for protocol in ['candidate_zero','source_threshold']:
   rs=[r for r in rows if r['mode']==mode and r['protocol']==protocol and r['dataset']=='pubmed']
   cells=[]
   for key in ['accuracy_pct','balanced_accuracy_pct','roc_auc','average_precision']:
    v=[r[key] for r in rs];cells.append(f'{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}')
   lines.append('|'+ '|'.join([mode,protocol]+cells)+'|')
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (out/'status.json').write_text(json.dumps(dict(complete=True,files=18,inputs=provenance),indent=2)+'\n')
 print('Saved',out)
if __name__=='__main__':main()
