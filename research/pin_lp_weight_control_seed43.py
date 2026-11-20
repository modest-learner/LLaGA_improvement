import json
from pathlib import Path
from check_lp_prediction import sha
files=['scripts/run_lp_weight_control_seed43.sh','research/audit_lp_weight_selection.py','research/summarize_lp_weight_control.py','train/train_mem_lp_weighted.py','research/lp_weighted_loss.py','scripts/train_deepspeed_lp_weighted.sh','eval/score_lp_binary_weighted.py','scripts/eval_lp_matched.sh','results/lp_weighted/weights/arxiv-cora/weights.json','results/lp_random_control/random_weights.json']
recipe=dict(seed=43,sources='arxiv-cora',target='pubmed',ablation='full',lr=.002,epochs=1,batch=16,random_selection_seed=20261119,files={f:sha(Path(f)) for f in files})
p=Path('results/lp_weight_control_seed43/recipe.json')
if p.exists():assert json.loads(p.read_text())==recipe,'Seed43 recipe changed; inspect before proceeding.'
else:p.write_text(json.dumps(recipe,indent=2)+'\n')
print('Seed43 paired recipe verified.')
