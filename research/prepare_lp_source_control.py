"""Freeze source-only repetition budget without looking at target outcomes."""
from pathlib import Path
import json,random,hashlib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 root=Path('results/lp_source_control');root.mkdir(parents=True,exist_ok=True)
 base=Path('dataset_research/lp_bert_disjoint_v1')
 a=base/'ogbn-arxiv/edge_sampled_2_10_only_train.jsonl';c=base/'cora/edge_sampled_2_10_only_train.jsonl'
 rows=[json.loads(l) for l in a.open()];other=[json.loads(l) for l in c.open()]
 rng=random.Random(20261120);selected=rng.sample(range(len(rows)),len(other))
 spec=dict(selection_seed=20261120,original_count=len(rows),extra_count=len(other),total=len(rows)+len(other),extra_pairs=[rows[i]['id'] for i in selected],inputs={str(p):sha(p) for p in [a,c]})
 assert spec['total']==81849 and spec['extra_count']==1381
 # The repetition set is shared by Full and Center-repeat.
 files=['train/train_mem_lp_source_control.py','scripts/train_deepspeed_lp_source_control.sh','train/train.py','train/llaga_trainer.py','model/llaga_arch.py','model/language_model/llaga_llama.py','models/vicuna-7b-v1.5-16k/config.json','models/vicuna-7b-v1.5-16k/tokenizer.model']
 spec['code']={p:sha(p) for p in files}
 for ds in ['ogbn-arxiv','cora','pubmed']:
  p=base/ds/'clean_lp_manifest.json';spec['inputs'][str(p)]=sha(p)
 p=root/'budget.json'
 if p.exists():assert json.loads(p.read_text())==spec,'Budget or training implementation changed.'
 else:p.write_text(json.dumps(spec,indent=2)+'\n')
 print('Verified fixed budget:',spec['original_count'],'+',spec['extra_count'],'=',spec['total'])
if __name__=='__main__':main()
