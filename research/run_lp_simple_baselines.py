"""Source-selected structural/text baselines on the existing frozen-BERT LP view.

Thresholds maximize equal-source, class-balanced validation accuracy using ONLY
Arxiv/Pubmed validation labels. MLP checkpoint selection uses that same criterion.
Cora labels are used for final reporting only. No BERT/LLM weights are updated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import csv

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            d.update(b)
    return d.hexdigest()


def read_pairs(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    return np.array([r['id'] for r in rows],dtype=np.int64),np.array([r['conversations'][1]['value']=='yes' for r in rows],dtype=np.int64)


def select_threshold(scores,labels,graphs):
    """Exact sweep, score>=threshold. Tie: prefer higher threshold deterministically."""
    assert len(scores)==len(labels)==len(graphs) and len(scores)>0
    assert np.isfinite(scores).all() and set(np.unique(labels))<={0,1}
    weights=np.zeros(len(labels),dtype=np.float64)
    domains=np.unique(graphs)
    for domain in domains:
        for label in (0,1):
            mask=(graphs==domain)&(labels==label)
            if not mask.any():
                raise ValueError('Each source validation graph requires both classes.')
            weights[mask]=1/(len(domains)*2*int(mask.sum()))
    order=np.argsort(-scores,kind='stable')
    s=scores[order]
    increments=np.where(labels[order]==1,weights[order],-weights[order])
    ends=np.r_[np.nonzero(s[:-1]!=s[1:])[0],len(s)-1]
    base=weights[labels==0].sum()
    values=np.r_[base,base+np.cumsum(increments)[ends]]
    thresholds=np.r_[np.nextafter(float(s[0]),np.inf),s[ends].astype(float)]
    # First maximum is the highest threshold; tolerate numerical summation ties.
    index=int(np.flatnonzero(values>=values.max()-1e-12)[0])
    return float(thresholds[index]),float(values[index])


def metrics(labels,scores,threshold):
    pred=scores>=threshold
    tp=int(((labels==1)&pred).sum());tn=int(((labels==0)&~pred).sum())
    fp=int(((labels==0)&pred).sum());fn=int(((labels==1)&~pred).sum())
    return dict(n=len(labels),accuracy_pct=float((pred==labels).mean()*100),
                balanced_accuracy_pct=50*(tp/(tp+fn)+tn/(tn+fp)),
                roc_auc=float(roc_auc_score(labels,scores)),average_precision=float(average_precision_score(labels,scores)),
                TP=tp,FP=fp,FN=fn,TN=tn,threshold=float(threshold))


def pair_features(x,pairs,batch=4096):
    chunks=[]
    for start in range(0,len(pairs),batch):
        a,b=x[torch.from_numpy(pairs[start:start+batch]).T]
        chunks.append(torch.cat([(a-b).abs(),a*b],dim=1))
    return torch.cat(chunks)


def heuristic_scores(data,x,pairs):
    adjacency=[set() for _ in range(data.num_nodes)]
    for u,v in data.edge_index.T.tolist():
        adjacency[u].add(v)
    deg=np.array([len(s) for s in adjacency],dtype=float)
    norm=torch.nn.functional.normalize(x,dim=1)
    arrays={k:np.empty(len(pairs),dtype=np.float64) for k in ['cosine','CN','AA','RA','PA']}
    for i,(u,v) in enumerate(pairs):
        common=adjacency[u]&adjacency[v]
        arrays['cosine'][i]=float((norm[u]*norm[v]).sum())
        arrays['CN'][i]=len(common)
        arrays['AA'][i]=sum(1/np.log(deg[k]) for k in common if deg[k]>1)
        arrays['RA'][i]=sum(1/deg[k] for k in common if deg[k]>0)
        arrays['PA'][i]=deg[u]*deg[v]
    return arrays


@torch.inference_mode()
def predict_mlp(model,x,device,batch_size=2048):
    model.eval()
    return torch.cat([model(part.to(device)).squeeze(-1).cpu() for part in x.split(batch_size)]).numpy().astype(np.float64)


def make_mlp(dim,hidden):
    return torch.nn.Sequential(torch.nn.Linear(dim,hidden),torch.nn.GELU(),torch.nn.Linear(hidden,1))


def fit_mlp(train_x,train_y,val_x,val_y,val_graphs,args,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    # Statistics fitted on source training pairs only.
    mean=train_x.mean(0);scale=train_x.std(0,unbiased=False).clamp_min(1e-6)
    train_x=(train_x-mean)/scale;val_x=(val_x-mean)/scale
    model=make_mlp(train_x.shape[1],args.hidden).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    target=torch.tensor(train_y,dtype=torch.float32)
    rng=torch.Generator().manual_seed(seed)
    best=None;best_score=-1;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0
        for idx in torch.randperm(len(train_x),generator=rng).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits=model(train_x[idx].to(args.device)).squeeze(-1)
            loss=loss_fn(logits,target[idx].to(args.device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite source MLP loss.')
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(idx)
        val_scores=predict_mlp(model,val_x,args.device)
        threshold,score=select_threshold(val_scores,val_y,val_graphs)
        history.append(dict(epoch=epoch,training_loss=loss_sum/len(train_x),source_macro_balanced_accuracy=score,threshold=threshold))
        if score>best_score+1e-12:
            best_score=score
            best=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                      feature_mean=mean,feature_scale=scale,threshold=threshold,epoch=epoch,source_selection_score=score)
        print(f'MLP seed={seed}, epoch={epoch}, source-macro balanced ACC={100*score:.2f}%',flush=True)
    del model
    best['history']=history
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--output',type=Path,default=Path('results/lp_next_round/simple_baselines'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--lr',type=float,default=1e-3)
    a=p.parse_args()
    torch.set_num_threads(4)
    if a.epochs<1:
        raise ValueError('epochs must be positive.')
    a.output.mkdir(parents=True,exist_ok=True)
    manifests={n:json.loads((a.data_root/d/'clean_lp_manifest.json').read_text()) for n,d in [('arxiv','ogbn-arxiv'),('pubmed','pubmed'),('cora','cora')]}
    assert all(m['features_complete'] for m in manifests.values())
    assert all(m['encoder']==manifests['arxiv']['encoder'] for m in manifests.values())
    recipe=dict(protocol='source_selected_baselines_v1',source_graphs=['arxiv','pubmed'],target='cora',
                manifests={n:sha(a.data_root/('ogbn-arxiv' if n=='arxiv' else n)/'clean_lp_manifest.json') for n in manifests},
                threshold_policy='equal-source class-balanced validation accuracy; score>=threshold; high-threshold tie break',
                mlp_input='symmetric abs(xu-xv) concat xu*xv, raw frozen BERT; feature stats from source training only',
                epochs=a.epochs,batch_size=a.batch_size,hidden=a.hidden,learning_rate=a.lr,
                implementation_sha256=sha(Path(__file__)))
    manifest_file=a.output/'run_config.json'
    if manifest_file.exists():
        assert json.loads(manifest_file.read_text())==recipe,'Baseline recipe changed; use a new output directory.'
    else:
        manifest_file.write_text(json.dumps(recipe,indent=2)+'\n')
    sets={};train_x=[];train_y=[];source_labels=[];source_graphs=[]
    for name in ('arxiv','pubmed','cora'):
        directory=a.data_root/('ogbn-arxiv' if name=='arxiv' else name)
        data=torch.load(directory/'processed_data.pt',map_location='cpu')
        x=torch.load(directory/'bert_x.pt',map_location='cpu').float()
        split='test' if name=='cora' else 'val'
        pairs,labels=read_pairs(directory/f'edge_sampled_2_10_only_{split}.jsonl')
        sets[name]=dict(split=split,pairs=pairs,labels=labels,features=pair_features(x,pairs),heuristics=heuristic_scores(data,x,pairs))
        if name!='cora':
            tp,ty=read_pairs(directory/'edge_sampled_2_10_only_train.jsonl')
            train_x.append(pair_features(x,tp));train_y.append(ty)
            source_labels.append(labels);source_graphs.append(np.full(len(labels),len(source_graphs),dtype=int))
        del data,x
    validation_labels=np.concatenate(source_labels);validation_graphs=np.concatenate(source_graphs)
    result=[]
    for method in ('cosine','CN','AA','RA','PA'):
        validation_scores=np.concatenate([sets[n]['heuristics'][method] for n in ('arxiv','pubmed')])
        threshold,criterion=select_threshold(validation_scores,validation_labels,validation_graphs)
        for name,s in sets.items():
            scores=s['heuristics'][method]
            result.append(dict(method=method,seed='',dataset=name,split=s['split'],**metrics(s['labels'],scores,threshold)))
            np.savez_compressed(a.output/f'{name}_{method}_scores.npz',pairs=s['pairs'],labels=s['labels'],scores=scores,threshold=threshold)
    tx=torch.cat(train_x);ty=np.concatenate(train_y)
    del train_x,train_y
    vx=torch.cat([sets[n]['features'] for n in ('arxiv','pubmed')])
    for seed in a.seeds:
        checkpoint=a.output/f'endpoint_mlp_seed{seed}.pt'
        if checkpoint.exists():
            best=torch.load(checkpoint,map_location='cpu')
            assert best['recipe']==recipe and best['seed']==seed
            print('Reuse completed MLP:',checkpoint,flush=True)
        else:
            best=fit_mlp(tx,ty,vx,validation_labels,validation_graphs,a,seed)
            best.update(seed=seed,recipe=recipe)
            temporary=checkpoint.with_suffix('.part')
            torch.save(best,temporary);temporary.replace(checkpoint)
        model=make_mlp(tx.shape[1],a.hidden).to(a.device)
        model.load_state_dict(best['state_dict'])
        for name,s in sets.items():
            features=(s['features']-best['feature_mean'])/best['feature_scale']
            scores=predict_mlp(model,features,a.device)
            result.append(dict(method='endpoint_MLP',seed=seed,dataset=name,split=s['split'],**metrics(s['labels'],scores,best['threshold'])))
            np.savez_compressed(a.output/f'{name}_endpoint_MLP_seed{seed}_scores.npz',pairs=s['pairs'],labels=s['labels'],scores=scores,threshold=best['threshold'])
        del model
    with (a.output/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    (a.output/'README.md').write_text('CN/AA/RA/PA/cosine use the same clean context or frozen BERT. All thresholds and MLP selection use Arxiv+Pubmed source validation only. Validation performance is used for selection and is optimistic; Cora performance is held out from baseline fitting. MLP uses 10 default source epochs and fewer parameters; it is an information/capacity baseline, not an equal-compute LLM comparison. AUC/AP apply to these continuous scores only, not the LLM generation files. If all MLP fits finish, rerun reuses complete weights. Partial MLP fits restart from source training, not from optimizer state.\n')
    print('Metrics:',a.output/'metrics.csv',flush=True)


if __name__=='__main__':
    main()
