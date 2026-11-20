"""Select hard negatives from SOURCE TRAIN ONLY; keep all examples and source/class mass."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,numpy as np,torch
from research.check_lp_prediction import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--sources',required=True,choices=['arxiv-pubmed','arxiv-cora']);a=p.parse_args()
 root=Path('dataset_research/lp_bert_disjoint_v1');out=Path('results/lp_weighted/weights')/a.sources;out.mkdir(parents=True,exist_ok=True)
 payload={'sources':a.sources,'rule':'top quarter negatives by frozen BERT cosine; raw2 vs1; per-source negative mean1','inputs':{},'stats':{},'weights':{}}
 for ds in a.sources.split('-'):
  d=root/('ogbn-arxiv' if ds=='arxiv' else ds);f=d/'edge_sampled_2_10_only_train.jsonl';feat=d/'bert_x.pt'
  rows=[json.loads(l) for l in f.open()];x=torch.nn.functional.normalize(torch.load(feat,map_location='cpu').float(),dim=-1)
  pairs=np.array([r['id'] for r in rows]);negative=np.array([r['conversations'][1]['value']=='no' for r in rows]);scores=[]
  for start in range(0,len(rows),4096):
   z=pairs[start:start+4096];scores.extend((x[z[:,0]]*x[z[:,1]]).sum(-1).tolist())
  scores=np.array(scores);idx=np.flatnonzero(negative);order=np.lexsort((idx,-scores[idx]));hard=idx[order[:int(np.ceil(len(idx)/4))]]
  w=np.ones(len(rows),dtype=float);w[hard]=2;w[negative]/=w[negative].mean()
  assert np.isclose(w[negative].mean(),1) and (w[~negative]==1).all()
  table={','.join(map(str,r['id'])):float(v) for r,v in zip(rows,w)};assert len(table)==len(rows)
  payload['weights'][ds]=table;payload['inputs'][ds]={str(p):sha(p) for p in [f,feat,d/'clean_lp_manifest.json']}
  payload['stats'][ds]=dict(pairs=len(rows),negatives=len(idx),hard=len(hard),hard_min_cosine=float(scores[hard].min()),low_weight=float(w[idx[order[-1]]]),high_weight=float(w[hard[0]]))
 path=out/'weights.json';text=json.dumps(payload,sort_keys=True,indent=2)+'\n'
 if path.exists():assert path.read_text()==text,'Weight recipe changed'
 else:path.write_text(text)
 print(path,payload['stats'])
if __name__=='__main__':main()
