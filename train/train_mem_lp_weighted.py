"""Opt-in weighted LP launcher; original train/model source files remain unchanged."""
import sys,os,json
from pathlib import Path
sys.path.extend(['.','./utils'])
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
import torch
import train as original
from research.check_lp_prediction import sha
from research.lp_weighted_loss import ACTIVE_WEIGHTS,PairWeightedCE
import model.language_model.llaga_llama as llama

manifest=Path(os.environ['LP_WEIGHT_MANIFEST']);spec=json.loads(manifest.read_text())
for files in spec['inputs'].values():
 for f,h in files.items():assert sha(Path(f))==h,'Weight input changed'
BaseDataset=original.LazySupervisedGraphDataset
class WeightedDataset(BaseDataset):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  assert '-'.join(self.use_dataset)==spec['sources']
  assert self.data_args.use_task=='lp' and self.template=='HO' and self.graph_input_ablation=='full'
  self.weights=[]
  for r in self.list_data_dict:
   self.weights.append(spec['weights'][r['dataset']][','.join(map(str,r['id']))])
 def __getitem__(self,i):
  row=super().__getitem__(i);row['pair_weight']=self.weights[i];return row
BaseCollator=original.DataCollatorForSupervisedDataset
class WeightedCollator(BaseCollator):
 def __call__(self,instances):
  batch=super().__call__(instances);batch['pair_weight']=torch.tensor([r['pair_weight'] for r in instances],dtype=torch.float32);return batch
BaseTrainer=original.LLaGATrainer
class WeightedTrainer(BaseTrainer):
 def _set_signature_columns_if_needed(self):
  super()._set_signature_columns_if_needed()
  if 'pair_weight' not in self._signature_columns:self._signature_columns.append('pair_weight')
 def compute_loss(self,model,inputs,return_outputs=False):
  weights=inputs.pop('pair_weight');token=ACTIVE_WEIGHTS.set(weights)
  try:return super().compute_loss(model,inputs,return_outputs=return_outputs)
  finally:ACTIVE_WEIGHTS.reset(token)
def weighted_data_module(tokenizer,data_args):
 return dict(train_dataset=WeightedDataset(tokenizer=tokenizer,data_args=data_args),eval_dataset=None,data_collator=WeightedCollator(tokenizer=tokenizer))
original.make_supervised_data_module=weighted_data_module
original.LLaGATrainer=WeightedTrainer
llama.CrossEntropyLoss=PairWeightedCE
if __name__=='__main__':original._train()
