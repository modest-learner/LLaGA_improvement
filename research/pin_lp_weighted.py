import argparse,json
from pathlib import Path
from check_lp_prediction import sha
p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
files=['train/train_mem_lp_weighted.py','research/lp_weighted_loss.py','scripts/train_deepspeed_lp_weighted.sh','train/train.py','train/llaga_trainer.py','model/language_model/llaga_llama.py','model/llaga_arch.py',f'results/lp_weighted/weights/{a.sources}/weights.json','models/vicuna-7b-v1.5-16k/config.json','models/vicuna-7b-v1.5-16k/tokenizer.model']
recipe=dict(sources=a.sources,seed=42,template='HO4',projector='linear',lr=.002,batch=16,epochs=1,weights={f:sha(Path(f)) for f in files})
a.root.mkdir(parents=True,exist_ok=True);p=a.root/'training_recipe.json'
if p.exists():assert json.loads(p.read_text())==recipe,'Weighted recipe changed; inspect before resuming.'
else:p.write_text(json.dumps(recipe,indent=2)+'\n')
print('Verified',p)
