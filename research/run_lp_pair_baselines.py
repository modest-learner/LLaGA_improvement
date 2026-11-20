"""Source-only selected endpoint, structural and fused pair MLP baselines."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.run_lp_simple_baselines import (
    sha,read_pairs,pair_features,heuristic_scores,select_threshold,metrics,
    fit_mlp,make_mlp,predict_mlp,
)


def directory(root,name):
    return root/('ogbn-arxiv' if name=='arxiv' else name)


def structure_features(data,pairs,heuristics):
    degree=torch.bincount(data.edge_index[0],minlength=data.num_nodes).numpy()
    a,b=degree[pairs[:,0]],degree[pairs[:,1]]
    raw=np.column_stack([heuristics[k] for k in ('CN','AA','RA','PA')]+[np.minimum(a,b),np.maximum(a,b)])
    assert np.isfinite(raw).all() and (raw>=0).all()
    return torch.from_numpy(np.log1p(raw).astype(np.float32))


def feature_block(s,model):
    if model=='endpoint':return s['semantic']
    if model=='structure':return s['structure']
    return torch.cat([s['semantic'],s['structure']],dim=1)


def run(a):
    names=a.sources+[a.target]
    manifests={n:json.loads((directory(a.data_root,n)/'clean_lp_manifest.json').read_text()) for n in names}
    assert all(m['features_complete'] and m['encoder']==manifests[names[0]]['encoder'] for m in manifests.values())
    recipe=dict(protocol='source_selected_pair_baselines_v1',sources=a.sources,target=a.target,
        manifests={n:sha(directory(a.data_root,n)/'clean_lp_manifest.json') for n in names},
        data_root=str(a.data_root.resolve()),epochs=a.epochs,hidden=a.hidden,batch_size=a.batch_size,lr=a.lr,
        models=a.models,seeds=a.seeds,semantic='abs(xu-xv),xu*xv',
        structure='log1p(CN,AA,RA,PA,min_degree,max_degree)',
        normalization='source training mean and std only',
        selection='equal-source validation balanced ACC; best epoch and shared threshold',
        implementation=sha(Path(__file__)),helper=sha(Path('research/run_lp_simple_baselines.py')))
    config=a.output/'run_config.json'
    if config.exists():
        assert json.loads(config.read_text())==recipe,'Configuration changed: use a new output directory.'
    else:config.write_text(json.dumps(recipe,indent=2)+'\n')
    # Regenerate metrics from saved scores/weights on reruns. No target labels enter fitting.
    sets={};training={}
    for name in names:
        d=directory(a.data_root,name)
        data=torch.load(d/'processed_data.pt',map_location='cpu')
        x=torch.load(d/'bert_x.pt',map_location='cpu').float()
        for split in (['train','val'] if name in a.sources else ['test']):
            pairs,labels=read_pairs(d/f'edge_sampled_2_10_only_{split}.jsonl')
            print(f'Prepare {name} {split}: {len(pairs)} pairs',flush=True)
            heuristics=heuristic_scores(data,x,pairs)
            record=dict(pairs=pairs,labels=labels,split=split,heuristics=heuristics,
                        semantic=pair_features(x,pairs),structure=structure_features(data,pairs,heuristics))
            (training if split=='train' else sets)[name]=record
        del data,x
    vy=np.concatenate([sets[n]['labels'] for n in a.sources])
    vg=np.concatenate([np.full(len(sets[n]['labels']),i) for i,n in enumerate(a.sources)])
    ty=np.concatenate([training[n]['labels'] for n in a.sources])
    rows=[]
    for method in ('cosine','CN','AA','RA','PA'):
        scores=np.concatenate([sets[n]['heuristics'][method] for n in a.sources])
        threshold,_=select_threshold(scores,vy,vg)
        for name,s in sets.items():
            values=s['heuristics'][method]
            rows.append(dict(method=method,seed='',dataset=name,split=s['split'],**metrics(s['labels'],values,threshold)))
            np.savez_compressed(a.output/f'{name}_{method}_scores.npz',pairs=s['pairs'],labels=s['labels'],scores=values,threshold=threshold)
    for kind in a.models:
        tx=torch.cat([feature_block(training[n],kind) for n in a.sources])
        vx=torch.cat([feature_block(sets[n],kind) for n in a.sources])
        for seed in a.seeds:
            checkpoint=a.output/f'{kind}_mlp_seed{seed}.pt'
            if checkpoint.exists():
                best=torch.load(checkpoint,map_location='cpu')
                assert best['recipe']==recipe and best['kind']==kind and best['seed']==seed
                print('Reuse',checkpoint,flush=True)
            else:
                print('Fit',kind,'seed',seed,flush=True)
                best=fit_mlp(tx,ty,vx,vy,vg,a,seed)
                best.update(recipe=recipe,kind=kind,seed=seed)
                tmp=checkpoint.with_suffix('.part');torch.save(best,tmp);tmp.replace(checkpoint)
            model=make_mlp(tx.shape[1],a.hidden).to(a.device);model.load_state_dict(best['state_dict'])
            for name,s in sets.items():
                features=(feature_block(s,kind)-best['feature_mean'])/best['feature_scale']
                values=predict_mlp(model,features,a.device)
                rows.append(dict(method=kind+'_MLP',seed=seed,dataset=name,split=s['split'],**metrics(s['labels'],values,best['threshold'])))
                np.savez_compressed(a.output/f'{name}_{kind}_MLP_seed{seed}_scores.npz',
                    pairs=s['pairs'],labels=s['labels'],scores=values,threshold=best['threshold'])
            del model
        del tx,vx
    temporary=a.output/'metrics.csv.part'
    with temporary.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temporary.replace(a.output/'metrics.csv')
    expected=(5+len(a.models)*len(a.seeds))*len(names)
    assert len(rows)==expected
    (a.output/'status.json').write_text(json.dumps(dict(complete=True,rows=len(rows),recipe_sha256=sha(config)),indent=2)+'\n')
    print('Complete:',a.output/'metrics.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',required=True,choices=['arxiv','pubmed','cora'])
    p.add_argument('--target',required=True,choices=['arxiv','pubmed','cora'])
    p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=['endpoint','structure','fusion'],default=['endpoint','structure','fusion'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--lr',type=float,default=.001)
    a=p.parse_args()
    assert a.target not in a.sources and len(set(a.sources))==len(a.sources),'Target must be held out.'
    assert len(set(a.models))==len(a.models) and len(set(a.seeds))==len(a.seeds)
    assert a.epochs>0 and a.batch_size>0
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This baseline output is already in use.')
        run(a)


if __name__=='__main__':main()
