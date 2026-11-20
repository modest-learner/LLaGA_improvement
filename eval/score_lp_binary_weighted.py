"""Independent fixed yes/no continuation scoring; original generation outputs stay intact."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.append('./utils')
import argparse,json,hashlib
import torch
from tqdm import tqdm
from model.builder import load_pretrained_model
from utils.utils import tokenizer_graph_token,get_model_name_from_path,disable_torch_init
from utils.conversation import conv_templates
from eval.lp_inputs import LP_PROMPT,intervene
from research.check_lp_prediction import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--target',choices=['cora','pubmed'],required=True)
 p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--ablation',choices=['full','center_repeat'],required=True)
 p.add_argument('--seed',type=int,required=True)
 p.add_argument('--root',type=Path,default=Path('results/lp_phase2'))
 p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
 p.add_argument('--model-base',type=Path,default=Path('models/vicuna-7b-v1.5-16k'))
 p.add_argument('--pilot',action='store_true',help='First four source-validation pairs per source; verify generate first token.')
 a=p.parse_args()
 torch.manual_seed(0)
 disable_torch_init()
 tok,model,_=load_pretrained_model(str(a.checkpoint),str(a.model_base),get_model_name_from_path(str(a.checkpoint)),device_map={'':torch.cuda.current_device()})
 model=model.to(torch.float16).cuda().eval()
 c=conv_templates['v1'].copy();c.append_message(c.roles[0],LP_PROMPT);c.append_message(c.roles[1],None)
 prompt=c.get_prompt();base=tokenizer_graph_token(prompt,tok)
 ids={}
 for name in ['yes','no']:
  tokens=tokenizer_graph_token(prompt+' '+name,tok)
  assert tokens[:len(base)]==base and len(tokens)==len(base)+1,'Candidate boundary changed; stop instead of silently changing protocol.'
  ids[name]=tokens[-1]
 input_ids=torch.tensor([base],device='cuda')
 sources=['arxiv','pubmed'] if a.target=='cora' else ['arxiv','cora']
 for ds in (sources if a.pilot else sources+[a.target]):
  split='test' if ds==a.target else 'val'
  d=a.data_root/('ogbn-arxiv' if ds=='arxiv' else ds)
  ref=a.root/f'predictions/{ds}_{split}_{a.ablation}.jsonl'
  original=[json.loads(l) for l in ref.open()]
  raw=[json.loads(l) for l in (d/f'edge_sampled_2_10_only_{split}.jsonl').open()]
  assert len(raw)==len(original)
  for q,r in zip(raw,original):
   assert q['id']==r['question_id'] and q['conversations'][1]['value']==r['gt']
   assert r['prompt']==LP_PROMPT
   cfg=r['lp_diagnostic_config']
   assert cfg['ablation']==a.ablation and cfg['context']=='clean_cached' and cfg['embedding']=='bert' and cfg['use_hop']==4
  # Hash once, outside the sample loop.
  weight_hash=sha(a.checkpoint/'mm_projector.bin')
  assert all(r['lp_diagnostic_config']['projector_sha256']==weight_hash for r in original)
  files=[d/'bert_x.pt']+[d/f'bert_{h}hop_x.pt' for h in range(1,5)]
  recipe=dict(protocol='lp_yes_no_single_token_v1',seed=a.seed,ablation=a.ablation,dataset=ds,
    checkpoint=str(a.checkpoint.resolve()),model_base=str(a.model_base.resolve()),
    projector_sha256=weight_hash,manifest_sha256=sha(d/'clean_lp_manifest.json'),
    input_sha256=sha(ref),feature_sha256={f.name:sha(f) for f in files},
    code_sha256=sha(Path(__file__)),core_sha256={f:sha(Path(f)) for f in ['model/llaga_arch.py','model/language_model/llaga_llama.py','eval/lp_inputs.py']},
    prompt=prompt,token_ids=ids,score='logP(yes|prompt,graph)-logP(no|prompt,graph); no EOS; no length normalization',pilot=a.pilot)
  out=a.root/('binary_scores_pilot' if a.pilot else 'binary_scores')/f'seed{a.seed}_{a.ablation}'
  out.mkdir(parents=True,exist_ok=True);path=out/f'{ds}_{split}.jsonl';rp=path.with_suffix('.recipe.json')
  if rp.exists():assert json.loads(rp.read_text())==recipe,'Recipe changed: use a separate output root.'
  else:
   assert not path.exists();rp.write_text(json.dumps(recipe,indent=2)+'\n')
  selected=original[:4] if a.pilot else original
  previous=[json.loads(l) for l in path.open()] if path.exists() else []
  assert len(previous)<=len(selected)
  for i,r in enumerate(previous):assert r['pair']==selected[i]['question_id'] and r['gt']==selected[i]['gt']
  if len(previous)==len(selected):print('Complete',path);continue
  features=[torch.load(f,map_location='cpu') for f in files]
  with path.open('a') as f,torch.inference_mode():
   for r in tqdm(selected[len(previous):],desc=f'{a.seed} {a.ablation} {ds}'):
    pair=r['question_id'];graph=torch.tensor(pair).unsqueeze(1).expand(-1,5).clone().cuda()
    emb=intervene(torch.stack([x[pair] for x in features],dim=1),a.ablation).half().cuda()
    logits=model(input_ids=input_ids,graph=graph,graph_emb=emb,use_cache=False).logits[0,-1].float()
    lp=logits.log_softmax(-1);top=int(logits.argmax())
    if a.pilot:
     generated=model.generate(input_ids,graph=graph,graph_emb=emb,do_sample=False,max_new_tokens=1,use_cache=True)
     assert int(generated[0,-1])==top,'Forward/generate first-token mismatch'
     # Existing complete yes/no outputs must begin with the same token under this input.
     if r['text'].strip() in ids:assert top==ids[r['text'].strip()],'Original generation differs'
    rec=dict(pair=pair,gt=r['gt'],logp_yes=float(lp[ids['yes']]),logp_no=float(lp[ids['no']]),
      score=float(lp[ids['yes']]-lp[ids['no']]),top_token_id=top,top_token=tok.decode([top]),
      original_text=r['text'])
    f.write(json.dumps(rec)+'\n');f.flush()
  del features
 print('Scoring complete; original generation results unchanged.')
if __name__=='__main__':main()
