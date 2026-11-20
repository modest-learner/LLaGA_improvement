import json
from pathlib import Path
from check_lp_prediction import sha
files=['train/train_mem.py','train/train_mem_lp_weighted.py','train/train.py','train/llaga_trainer.py','research/lp_weighted_loss.py','scripts/train_deepspeed.sh','scripts/train_deepspeed_lp_weighted.sh','model/llaga_arch.py','model/language_model/llaga_llama.py','results/lp_random_control/random_weights.json','results/lp_random_control/ones_weights.json','models/vicuna-7b-v1.5-16k/config.json','models/vicuna-7b-v1.5-16k/tokenizer.model']
recipe=dict(seed=42,sources='arxiv-cora',target='pubmed',lr=.002,batch=16,epochs=1,selection_seed=20261119,code={p:sha(Path(p)) for p in files})
p=Path('results/lp_random_control/recipe.json')
if p.exists():assert json.loads(p.read_text())==recipe,'Recipe changed'
else:p.write_text(json.dumps(recipe,indent=2)+'\n')
print('Random control recipe verified.')
