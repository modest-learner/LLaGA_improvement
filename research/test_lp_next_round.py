"""CPU checks for source-only threshold selection and symmetric pair baselines."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np
import torch
from research.run_lp_simple_baselines import select_threshold,metrics,pair_features,fit_mlp,make_mlp,predict_mlp


def main():
    torch.set_num_threads(2)
    score=np.array([.9,.8,.8,.2,.7,.6,.3,.1])
    label=np.array([1,0,1,0,1,0,0,1]);domain=np.array([0,0,0,0,1,1,1,1])
    threshold,objective=select_threshold(score,label,domain)
    candidates=np.r_[np.nextafter(score.max(),np.inf),np.unique(score)]
    def value(t):
        return np.mean([metrics(label[domain==d],score[domain==d],t)['balanced_accuracy_pct']/100 for d in (0,1)])
    assert abs(objective-max(value(t) for t in candidates))<1e-12
    assert abs(objective-value(threshold))<1e-12
    t,v=select_threshold(np.zeros(8),label,domain)
    assert t>0 and abs(v-.5)<1e-12
    x=torch.arange(24,dtype=torch.float32).reshape(6,4)
    pairs=np.array([[0,2],[1,5]])
    assert torch.equal(pair_features(x,pairs),pair_features(x,pairs[:,::-1].copy()))
    torch.manual_seed(4)
    tx=torch.randn(24,8);ty=np.array([0,1]*12)
    vx=torch.randn(8,8)
    args=SimpleNamespace(device='cpu',hidden=8,lr=.001,epochs=2,batch_size=8)
    best=fit_mlp(tx,ty,vx,label,domain,args,42)
    model=make_mlp(8,8);model.load_state_dict(best['state_dict'])
    s=predict_mlp(model,(vx-best['feature_mean'])/best['feature_scale'],'cpu')
    t,value=select_threshold(s,label,domain)
    assert abs(value-best['source_selection_score'])<1e-12 and t==best['threshold']
    assert all(torch.isfinite(w).all() for w in best['state_dict'].values())
    print('PASS: exact source-macro threshold sweep, constant-score ties, symmetric pairs, finite MLP fit and saved source-selected checkpoint.')


if __name__=='__main__':
    main()
