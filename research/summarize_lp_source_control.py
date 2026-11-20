"""Matched source-composition x neighborhood comparison; calibrate on Arxiv only."""
import sys,json,csv,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.run_lp_simple_baselines import metrics,select_threshold
from research.check_lp_prediction import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--single-score-root',type=Path,default=Path('results/lp_source_control/arxiv_only'));p.add_argument('--output',type=Path,default=Path('results/lp_source_control/summary'));a=p.parse_args()
 rows=[];refs={};inputs={}
 for source,root in [('arxiv_cora',Path('results/lp_phase2')),('arxiv_only',a.single_score_root)]:
  for mode in ['full','center_repeat']:
   sets={}
   for ds in ['arxiv','cora','pubmed']:
    split='test' if ds=='pubmed' else 'val';f=root/f'binary_scores/seed42_{mode}/{ds}_{split}.jsonl'
    recipe=json.loads(f.with_suffix('.recipe.json').read_text());assert not recipe['pilot'] and recipe['seed']==42 and recipe['ablation']==mode
    z=[json.loads(l) for l in f.open()];pairs=[r['pair'] for r in z];y=np.array([r['gt']=='yes' for r in z],dtype=int);s=np.array([r['score'] for r in z]);assert np.isfinite(s).all()
    if ds not in refs:refs[ds]=(pairs,y)
    else:assert pairs==refs[ds][0] and np.array_equal(y,refs[ds][1])
    sets[ds]=(z,y,s);inputs[str(f)]=sha(f)
   _,vy,vs=sets['arxiv'];threshold,_=select_threshold(vs,vy,np.zeros(len(vy),dtype=int))
   for ds,(z,y,s) in sets.items():
    for protocol,t in [('candidate_zero',0.),('arxiv_val_threshold',threshold)]:
     m=metrics(y,s,t);rows.append(dict(sources=source,mode=mode,seed=42,dataset=ds,protocol=protocol,original_accuracy_pct=100*np.mean(np.array(['yes' in r['original_text'] for r in z])==y),**m))
 assert len(rows)==24
 a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/'comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# 等预算单源/多源 × 邻域对照，seed42','', 'Cora对单源模型是未参与训练的辅助图；两种源配置的校准均只用Arxiv验证集。原生成指标独立报告。','', '|sources|mode|protocol|original ACC|candidate ACC|BA|AUC|AP|','|---|---|---|---:|---:|---:|---:|---:|']
 for r in rows:
  if r['dataset']=='pubmed':lines.append('|'+ '|'.join([r['sources'],r['mode'],r['protocol']]+[f'{r[k]:.4f}' for k in ['original_accuracy_pct','accuracy_pct','balanced_accuracy_pct','roc_auc','average_precision']])+'|')
 lines+=['','## Neighborhood gains on Pubmed (Full minus Center-repeat)']
 for source in ['arxiv_cora','arxiv_only']:
  d={r['mode']:r for r in rows if r['dataset']=='pubmed' and r['sources']==source and r['protocol']=='candidate_zero'}
  lines.append(f"{source}: original ACC {d['full']['original_accuracy_pct']-d['center_repeat']['original_accuracy_pct']:+.4f} pp; AUC {d['full']['roc_auc']-d['center_repeat']['roc_auc']:+.4f}.")
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n');(a.output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n');print('Saved',a.output/'REPORT.md')
if __name__=='__main__':main()
