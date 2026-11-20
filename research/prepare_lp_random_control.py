"""Prepare fixed random-negative control and all-ones implementation check."""
import sys,json,copy,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.check_lp_prediction import sha

def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(obj,sort_keys=True,indent=2)+'\n'
 if p.exists():assert p.read_text()==text,'Recipe changed: '+str(p)
 else:p.write_text(text)

def main():
 source=Path('results/lp_weighted/weights/arxiv-cora/weights.json');spec=json.loads(source.read_text())
 for files in spec['inputs'].values():
  for f,h in files.items():assert sha(Path(f))==h
 ones=copy.deepcopy(spec);random=copy.deepcopy(spec)
 ones['rule']='all ones implementation check';random['rule']='random negatives; same counts and weight multiset as semantic treatment';random['selection_seed']=20261119
 random['semantic_manifest_sha256']=sha(source)
 for ds,weights in spec['weights'].items():
  f=next(f for f in spec['inputs'][ds] if f.endswith('_train.jsonl'));rows=[json.loads(l) for l in open(f)]
  negatives=[','.join(map(str,r['id'])) for r in rows if r['conversations'][1]['value']=='no']
  hard=sum(weights[k]>1 for k in negatives);high=max(weights[k] for k in negatives);low=min(weights[k] for k in negatives)
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'20261119:{ds}'.encode()).digest()[:8],'big'))
  chosen=set(rng.choice(negatives,size=hard,replace=False).tolist())
  ones['weights'][ds]={k:1. for k in weights}
  negative_set=set(negatives)
  random['weights'][ds]={k:(high if k in chosen else low) if k in negative_set else 1. for k in weights}
  assert sorted(random['weights'][ds].values())==sorted(weights.values())
  random['stats'][ds]=dict(pairs=len(rows),negatives=len(negatives),hard=hard,high_weight=high,low_weight=low,overlap_with_semantic=sum(weights[k]>1 for k in chosen))
 save(Path('results/lp_random_control/ones_weights.json'),ones)
 save(Path('results/lp_random_control/random_weights.json'),random)
 print('Prepared all-ones and random control; identical per-source weight multiset.')
if __name__=='__main__':main()
