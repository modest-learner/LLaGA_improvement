import json
from pathlib import Path
from check_lp_prediction import sha
files=['scripts/run_lp_source_control_seeds.sh','research/summarize_lp_source_control_seed.py','research/summarize_lp_source_three_seeds.py','results/lp_source_control/budget.json','eval/score_lp_binary_weighted.py','scripts/eval_lp_matched.sh','research/check_lp_training.py']
recipe=dict(seeds=[43,44],source='arxiv',target='pubmed',modes=['full','center_repeat'],batch=16,steps=5116,epochs=1,lr=.002,threshold_validation='arxiv',files={p:sha(Path(p)) for p in files})
p=Path('results/lp_source_control/seed_extension_recipe.json')
if p.exists():assert json.loads(p.read_text())==recipe,'Seed extension configuration changed.'
else:p.write_text(json.dumps(recipe,indent=2)+'\n')
print('Verified seed43/44 source-control recipe.')
