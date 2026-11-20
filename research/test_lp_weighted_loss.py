"""CPU loss/gradient equivalence and source weight invariants."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from research.lp_weighted_loss import ACTIVE_WEIGHTS,PairWeightedCE

def main():
 for dtype in [torch.float32,torch.bfloat16]:
  x=torch.randn(12,7,dtype=dtype,requires_grad=True)
  y=torch.tensor([-100,-100,1,2,-100,-100,3,4,-100,-100,5,6]);f=PairWeightedCE(ignore_index=-100)
  t=ACTIVE_WEIGHTS.set(torch.ones(3));a=f(x,y);g=torch.autograd.grad(a,x,retain_graph=True)[0]
  b=torch.nn.functional.cross_entropy(x,y);h=torch.autograd.grad(b,x,retain_graph=True)[0]
  assert torch.equal(a,b) and torch.equal(g,h);ACTIVE_WEIGHTS.reset(t)
  t=ACTIVE_WEIGHTS.set(torch.tensor([1.,1.6,.8]));a=f(x,y)
  expected=(torch.nn.functional.cross_entropy(x,y,reduction='none').reshape(3,4).float().sum(1)/2*torch.tensor([1.,1.6,.8])).mean()
  assert torch.allclose(a,expected) and torch.isfinite(torch.autograd.grad(a,x)[0]).all();ACTIVE_WEIGHTS.reset(t)
 for path in Path('results/lp_weighted/weights').glob('*/weights.json'):
  spec=json.loads(path.read_text())
  for ds,w in spec['weights'].items():
   train=next(f for f in spec['inputs'][ds] if f.endswith('_train.jsonl'));rows=[json.loads(l) for l in open(train)]
   pos=[];neg=[]
   for r in rows:(neg if r['conversations'][1]['value']=='no' else pos).append(w[','.join(map(str,r['id']))])
   assert all(v==1 for v in pos) and abs(sum(neg)/len(neg)-1)<1e-10
 print('PASS FP32/BF16 all-ones loss+gradient equivalence; weighted loss; per-source class mass.')
if __name__=='__main__':main()
