"""Descriptive source-training-only audit; no model fitting or target tuning."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from research.run_lp_simple_baselines import read_pairs,heuristic_scores
from research.check_lp_prediction import sha

def main():
 torch.set_num_threads(4)
 out=Path('results/lp_weight_control_seed43');out.mkdir(parents=True,exist_ok=True)
 paths={'semantic':Path('results/lp_weighted/weights/arxiv-cora/weights.json'),'random':Path('results/lp_random_control/random_weights.json')}
 specs={k:json.loads(p.read_text()) for k,p in paths.items()}
 rows=[]
 for ds in ['arxiv','cora']:
  for spec in specs.values():
   for f,h in spec['inputs'][ds].items():assert sha(Path(f))==h,f
  d=Path('dataset_research/lp_bert_disjoint_v1')/('ogbn-arxiv' if ds=='arxiv' else ds)
  pairs,y=read_pairs(d/'edge_sampled_2_10_only_train.jsonl');pairs=pairs[y==0]
  data=torch.load(d/'processed_data.pt',map_location='cpu');x=torch.load(d/'bert_x.pt',map_location='cpu').float()
  features=heuristic_scores(data,x,pairs)
  degree=torch.bincount(data.edge_index[0],minlength=data.num_nodes).numpy()
  u,v=degree[pairs[:,0]],degree[pairs[:,1]]
  features.update(min_degree=np.minimum(u,v),max_degree=np.maximum(u,v))
  features['Jaccard']=np.divide(features['CN'],u+v-features['CN'],out=np.zeros(len(pairs)),where=u+v-features['CN']>0)
  keys=[','.join(map(str,p)) for p in pairs]
  masks={'all_negatives':np.ones(len(pairs),dtype=bool)}
  for name,spec in specs.items():
   masks[name+'_selected']=np.array([spec['weights'][ds][k]>1 for k in keys])
   masks[name+'_unselected']=~masks[name+'_selected']
  assert masks['semantic_selected'].sum()==masks['random_selected'].sum()
  for group,mask in masks.items():
   for feature,values in features.items():
    a=values[mask];assert len(a) and np.isfinite(a).all()
    rows.append(dict(dataset=ds,group=group,n=len(a),feature=feature,mean=float(a.mean()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),q75=float(np.quantile(a,.75)),q90=float(np.quantile(a,.9)),zero_pct=float(100*(a==0).mean())))
  print('Audited source train negatives:',ds,len(pairs),flush=True)
 assert len(rows)==2*5*8
 with (out/'selection_audit.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'selection_audit_inputs.json').write_text(json.dumps({str(p):sha(p) for p in paths.values()},indent=2)+'\n')
 print('Saved',out/'selection_audit.csv')
if __name__=='__main__':main()
