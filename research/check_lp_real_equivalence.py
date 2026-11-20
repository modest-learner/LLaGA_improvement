"""Compare actual short training loss and final projector; fail before full trial."""
import json,torch
from pathlib import Path
root=Path('checkpoints/lp_random_control/equivalence')
a=torch.load(root/'baseline/mm_projector.bin',map_location='cpu');b=torch.load(root/'ones/mm_projector.bin',map_location='cpu')
assert a.keys()==b.keys();details={}
for k in a:
 assert torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all()
 details[k]=float((a[k].float()-b[k].float()).abs().max())
 assert torch.allclose(a[k].float(),b[k].float(),atol=1e-5,rtol=1e-5),(k,details[k])
states=[json.loads((root/x/'trainer_state.json').read_text()) for x in ['baseline','ones']]
assert all(s['global_step']==s['max_steps']==4 for s in states)
losses=[[r['loss'] for r in s['log_history'] if 'loss' in r] for s in states]
assert len(losses[0])==len(losses[1])==4
assert max(abs(x-y) for x,y in zip(*losses))<=1e-4,losses
out=Path('results/lp_random_control/equivalence.json')
out.write_text(json.dumps(dict(passed=True,steps=4,losses=losses,max_abs_parameter_difference=details,
 scope='Real model, original vs all-ones weighted launcher, single GPU, same seed and arguments; final projector and logged loss compared. Does not prove long-run bitwise or two-GPU equivalence.'),indent=2)+'\n')
print('PASS actual model 4-step equivalence:',details)
