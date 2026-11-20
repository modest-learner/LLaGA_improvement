"""Under the runner lock, archive an empty failed checkpoint and its log only."""
from pathlib import Path
from datetime import datetime
import json

def main():
 base=Path('checkpoints/lp_weight_control_seed43')
 cp=base/'llaga-HO-arxiv-cora-lp-linear-bert-lr2e-3-seed43-bs16-random'
 if not cp.exists():
  print('Random checkpoint absent: ready for fresh training.');return
 if any(cp.iterdir()):
  print('Random checkpoint contains files; leave intact for completion validation.');return
 out=Path('results/lp_weight_control_seed43/random')
 saved=list(out.rglob('*.jsonl')) if out.exists() else []
 assert not any(p.stat().st_size for p in saved),'Saved random predictions exist; inspect before replacing their checkpoint.'
 stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
 log=Path('logs/lp_weight_control_seed43/train_random.log')
 record=dict(reason='Empty directory from failed initialization; no weights or optimizer steps can be resumed.',timestamp=stamp,moves=[])
 for p in (cp,log):
  if p.exists():
   destination=p.with_name(p.name+'.failed_'+stamp)
   p.rename(destination);record['moves'].append(dict(source=str(p),archive=str(destination)))
 root=Path('results/lp_weight_control_seed43/recovery');root.mkdir(parents=True,exist_ok=True)
 (root/f'{stamp}.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
