"""Original training loss with a frozen Arxiv-only presentation budget."""
import sys,json,copy,random,os
from pathlib import Path
sys.path.extend(['.','./utils'])
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
import train as original
from research.check_lp_prediction import sha
spec=json.loads(Path('results/lp_source_control/budget.json').read_text())
for p,h in {**spec['inputs'],**spec['code']}.items():assert sha(Path(p))==h,p
Base=original.LazySupervisedGraphDataset
class BudgetDataset(Base):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  assert self.use_dataset==['arxiv'] and self.use_task==['lp'] and self.template=='HO'
  assert len(self.list_data_dict)==spec['original_count']
  lookup={tuple(r['id']):r for r in self.list_data_dict};assert len(lookup)==len(self.list_data_dict)
  self.list_data_dict.extend(copy.deepcopy(lookup[tuple(p)]) for p in spec['extra_pairs'])
  random.shuffle(self.list_data_dict)
  assert len(self.list_data_dict)==spec['total']
  print('SOURCE_CONTROL_BUDGET',spec['total'],'unique',spec['original_count'],flush=True)
def module(tokenizer,data_args):
 return dict(train_dataset=BudgetDataset(tokenizer=tokenizer,data_args=data_args),eval_dataset=None,data_collator=original.DataCollatorForSupervisedDataset(tokenizer=tokenizer))
original.make_supervised_data_module=module
if __name__=='__main__':original._train()
