"""Paired seed42 baseline comparison, with fixed source-only threshold selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,csv,numpy as np
from research.run_lp_simple_baselines import select_threshold,metrics
from research.check_lp_prediction import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--target',choices=['cora','pubmed'],required=True);a=p.parse_args()
 sources=['arxiv','pubmed'] if a.target=='cora' else ['arxiv','cora'];root=Path('results/lp_weighted')/('-'.join(sources)+'_to_'+a.target)
 base=Path('results/lp_cora_scores' if a.target=='cora' else 'results/lp_phase2');rows=[];hashes={}
 reference={}
 # Reuse the predeclared source-validation semantic grouping; never select cuts on target.
 cosine_root=Path('results/lp_next_round/simple_baselines' if a.target=='cora' else 'results/lp_phase2/arxiv-cora_to_pubmed/pair_baselines')
 values=[];mass=[]
 for g in sources:
  z=np.load(cosine_root/f'{g}_cosine_scores.npz');v=z['scores'][z['labels']==0];values.extend(v);mass.extend(np.full(len(v),1/(len(sources)*len(v))))
 values=np.array(values);mass=np.array(mass);order=np.argsort(values);cut=values[order][np.searchsorted(np.cumsum(mass[order]),.75)]
 cosine=np.load(cosine_root/f'{a.target}_cosine_scores.npz');strata=[]
 for name,d in [('baseline',base),('weighted',root)]:
  sets={}
  for ds in sources+[a.target]:
   split='test' if ds==a.target else 'val';f=d/f'binary_scores/seed42_full/{ds}_{split}.jsonl'
   z=[json.loads(l) for l in f.open()];recipe=json.loads(f.with_suffix('.recipe.json').read_text());assert not recipe['pilot']
   hashes[str(f)]=sha(f)
   pairs=[r['pair'] for r in z];y=np.array([int(r['gt']=='yes') for r in z]);score=np.array([r['score'] for r in z]);assert np.isfinite(score).all()
   if name=='baseline':reference[ds]=(pairs,y)
   else:assert pairs==reference[ds][0] and np.array_equal(y,reference[ds][1])
   sets[ds]=(y,score,z)
  t,_=select_threshold(np.concatenate([sets[g][1] for g in sources]),np.concatenate([sets[g][0] for g in sources]),np.concatenate([np.full(len(sets[g][0]),i) for i,g in enumerate(sources)]))
  for ds,(y,s,z) in sets.items():
   original=np.array(['yes' in r['original_text'] for r in z]);genacc=100*(original==y).mean()
   for protocol,threshold in [('candidate_zero',0),('equal_source',t)]:
    if ds==a.target:
     assert np.array_equal(np.array([r['pair'] for r in z]),cosine['pairs']), 'Target pair order differs'
     assert np.array_equal(y,cosine['labels']), 'Target labels differ'
     for group,selected in [('high_similarity',cosine['scores']>=cut),('other',cosine['scores']<cut)]:
      mask=selected&(y==0)
      count=int(mask.sum())
      fp=int((s[mask]>=threshold).sum())
      strata.append(dict(model=name,protocol=protocol,group=group,cutoff=float(cut),
                         negatives=count,FP=fp,FPR_pct=100*fp/count if count else None))
    m=metrics(y,s,threshold);rows.append(dict(model=name,seed=42,dataset=ds,protocol=protocol,original_accuracy_pct=float(genacc),**m,
      FPR_pct=100*m['FP']/(m['FP']+m['TN']),TPR_pct=100*m['TP']/(m['TP']+m['FN'])))
 assert len(strata)==8, 'Expected two models x two protocols x two negative groups'
 with (root/'comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (root/'negative_strata.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(strata[0]));w.writeheader();w.writerows(strata)
 lines=['# 源负例加权试验：seed42配对比较','', '单训练种子筛查，不能宣称稳定提升；原生成、候选默认阈值、源选阈值分开报告。','',
 '|模型|协议|原生成ACC|候选ACC|BA|AUC|AP|FPR|TPR|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  if r['dataset']==a.target:lines.append('|'+ '|'.join([r['model'],r['protocol']]+[f'{r[k]:.4f}' for k in ['original_accuracy_pct','accuracy_pct','balanced_accuracy_pct','roc_auc','average_precision','FPR_pct','TPR_pct']])+'|')
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n');(root/'comparison_inputs.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print('Saved',root/'REPORT.md')
if __name__=='__main__':main()
