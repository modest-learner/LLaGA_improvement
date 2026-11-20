"""Locate stable neighborhood corrections/harm using existing three-seed scores."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.check_lp_prediction import sha

def main():
 out=Path('results/lp_source_control/diagnostics');out.mkdir(parents=True,exist_ok=True)
 summary=[];pairrows=[];inputs={}
 for ds,root,features,sources in [('cora',Path('results/lp_cora_scores'),Path('results/lp_phase2/arxiv-pubmed_to_cora/pair_baselines'),['arxiv','pubmed']),('pubmed',Path('results/lp_phase2'),Path('results/lp_phase2/arxiv-cora_to_pubmed/pair_baselines'),['arxiv','cora'])]:
  modes={};ref=None
  for mode in ['full','center_repeat']:
   scores=[]
   for seed in [42,43,44]:
    f=root/f'binary_scores/seed{seed}_{mode}/{ds}_test.jsonl';z=[json.loads(l) for l in f.open()];pairs=[r['pair'] for r in z];y=np.array([r['gt']=='yes' for r in z]);s=np.array([r['score'] for r in z]);inputs[str(f)]=sha(f)
    if ref is None:ref=(pairs,y)
    else:assert pairs==ref[0] and np.array_equal(y,ref[1])
    assert np.isfinite(s).all();scores.append(s)
   modes[mode]=np.array(scores)>=0
  cosine=np.load(features/f'{ds}_cosine_scores.npz');cn=np.load(features/f'{ds}_CN_scores.npz');assert np.array_equal(cosine['pairs'],ref[0]) and np.array_equal(cn['pairs'],ref[0])
  v=[];w=[]
  for source in sources:
   z=np.load(features/f'{source}_cosine_scores.npz');a=z['scores'][z['labels']==0];v.extend(a);w.extend(np.full(len(a),1/(len(sources)*len(a))))
  v=np.array(v);w=np.array(w);idx=np.argsort(v);cut=v[idx][np.searchsorted(w[idx].cumsum(),.75)]
  full=modes['full'];center=modes['center_repeat'];fc=full==y;cc=center==y
  masks={'stable_corrected':np.all(fc&~cc,axis=0),'stable_harmed':np.all(~fc&cc,axis=0),'both_always_wrong':np.all(~fc&~cc,axis=0),'either_prediction_varies':np.any(full!=full[0],axis=0)|np.any(center!=center[0],axis=0)}
  for category,mask in masks.items():
   for label in [0,1]:
    for high in [False,True]:
     for common in [False,True]:
      selected=mask&(y==label)&((cosine['scores']>=cut)==high)&((cn['scores']>0)==common)
      summary.append(dict(dataset=ds,category=category,label=label,high_semantic=high,common_neighbor=common,n=int(selected.sum()),source_cosine_cut=float(cut)))
  for i,pair in enumerate(ref[0]):
   pairrows.append(dict(dataset=ds,u=pair[0],v=pair[1],label=int(y[i]),cosine=float(cosine['scores'][i]),CN=float(cn['scores'][i]),**{k:bool(m[i]) for k,m in masks.items()}))
  print(ds,{k:int(m.sum()) for k,m in masks.items()})
 for name,rows in [('strata.csv',summary),('pairs.csv',pairrows)]:
  with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
if __name__=='__main__':main()
