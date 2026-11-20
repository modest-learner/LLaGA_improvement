"""Exploratory error strata with semantic/degree cutoffs fitted on source negatives only."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.lp_inputs import parse_binary
from research.run_lp_simple_baselines import sha,read_pairs


def weighted_cutoffs(values,quantiles):
    x=np.concatenate(values)
    w=np.concatenate([np.full(len(v),1/(len(values)*len(v))) for v in values])
    order=np.argsort(x,kind='stable');x=x[order];cdf=np.cumsum(w[order])
    return [float(x[min(np.searchsorted(cdf,q),len(x)-1)]) for q in quantiles]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--round-root',type=Path,default=Path('results/lp_next_round'))
    p.add_argument('--output',type=Path,default=Path('results/lp_phase2/error_strata'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    sets={};inputs={}
    for name in ('arxiv','pubmed','cora'):
        split='test' if name=='cora' else 'val'
        d=a.data_root/('ogbn-arxiv' if name=='arxiv' else name)
        pairs,y=read_pairs(d/f'edge_sampled_2_10_only_{split}.jsonl');y=y.astype(bool)
        graph=torch.load(d/'processed_data.pt',map_location='cpu')
        degree=torch.bincount(graph.edge_index[0],minlength=graph.num_nodes).numpy()
        item=dict(pairs=pairs,y=y,degree_max=degree[pairs].max(axis=1),pred={})
        inputs[str(d/'clean_lp_manifest.json')]=sha(d/'clean_lp_manifest.json')
        for key in ('cosine','CN'):
            path=a.round_root/f'simple_baselines/{name}_{key}_scores.npz'
            npz=np.load(path);assert np.array_equal(npz['pairs'],pairs) and np.array_equal(npz['labels'],y)
            item[key]=npz['scores'];inputs[str(path)]=sha(path)
        for seed,mode in itertools.product((42,43,44),('full','center_repeat')):
            path=(Path(f'results/lp_clean/arxiv-pubmed_to_cora_trainseed{seed}_{mode}/screen/cora_{mode}.jsonl') if name=='cora' else
                  a.round_root/f'source_validation/seed{seed}_{mode}/{name}_val_{mode}.jsonl')
            records=[json.loads(s) for s in path.read_text().splitlines()]
            assert len(records)==len(y)
            assert all(r['question_id']==pair.tolist() and (r['gt']=='yes')==bool(label) for r,pair,label in zip(records,pairs,y))
            parsed=[parse_binary(r['text']) for r in records];assert all(v in ('yes','no') for v in parsed)
            item['pred'][seed,mode]=np.array([v=='yes' for v in parsed]);inputs[str(path)]=sha(path)
        sets[name]=item
    cuts={key:weighted_cutoffs([sets[n][key][~sets[n]['y']] for n in ('arxiv','pubmed')],qs)
          for key,qs in [('cosine',[.25,.5,.75]),('degree_max',[.5])]}
    rows=[];pairs_out=[]
    for name,s in sets.items():
        sb=np.searchsorted(cuts['cosine'],s['cosine'],side='right')
        db=np.searchsorted(cuts['degree_max'],s['degree_max'],side='right')
        cb=(s['CN']>0).astype(int)
        for semantic,degree,cn in itertools.product(range(4),range(2),range(2)):
            mask=(sb==semantic)&(db==degree)&(cb==cn);n=int(mask.sum())
            if not n:continue
            y=s['y'][mask];neg=int((~y).sum());pos=int(y.sum())
            for (seed,mode),pred in s['pred'].items():
                v=pred[mask];fp=int((v&~y).sum());fn=int((~v&y).sum())
                rows.append(dict(dataset=name,semantic_bin=semantic,degree_bin=degree,CN_positive=cn,
                    seed=seed,mode=mode,n=n,positives=pos,negatives=neg,FP=fp,FN=fn,
                    accuracy_pct=100*float((v==y).mean()),FPR_pct=100*fp/neg if neg else '',
                    FNR_pct=100*fn/pos if pos else '',small_negative_group=neg<20,small_positive_group=pos<20))
        if name=='cora':
            for i,pair in enumerate(s['pairs']):
                full=np.array([s['pred'][seed,'full'][i] for seed in (42,43,44)])
                center=np.array([s['pred'][seed,'center_repeat'][i] for seed in (42,43,44)])
                y=bool(s['y'][i])
                pairs_out.append(dict(u=int(pair[0]),v=int(pair[1]),label=int(y),cosine=float(s['cosine'][i]),
                    CN=float(s['CN'][i]),max_degree=int(s['degree_max'][i]),semantic_bin=int(sb[i]),degree_bin=int(db[i]),
                    full_yes_count=int(full.sum()),center_yes_count=int(center.sum()),
                    full_stable_FP=bool(full.all() and not y),full_stable_FN=bool((~full).all() and y),
                    stable_neighbor_harm=bool(((full!=y)&(center==y)).all()),
                    stable_neighbor_benefit=bool(((full==y)&(center!=y)).all())))
    for filename,data in [('strata.csv',rows),('cora_pairs.csv',pairs_out)]:
        with (a.output/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (a.output/'protocol.json').write_text(json.dumps(dict(
        source_graphs=['arxiv','pubmed'],target='cora',cutoffs=cuts,
        cutoff_rule='Equal-source weighted negative quantiles. Right-open bins; ties go to higher bin.',
        interpretation='Descriptive stratification, not causal adjustment. Small cells explicitly flagged.',
        inputs=inputs,code_sha256=sha(Path(__file__))),indent=2)+'\n')
    (a.output/'status.json').write_text(json.dumps(dict(complete=True,stratum_rows=len(rows),target_pairs=len(pairs_out)),indent=2)+'\n')
    print('Completed error strata:',a.output,'cutoffs:',cuts,flush=True)


if __name__=='__main__':main()
