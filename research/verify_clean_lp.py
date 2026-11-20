"""Verify clean LP split separation, hidden edges and cached feature hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            d.update(b)
    return d.hexdigest()


def verify(root,name,require_features=True):
    out=root/name
    m=json.loads((out/'clean_lp_manifest.json').read_text())
    assert m['protocol']=='disjoint_context_lp_v1'
    assert sha(out/'processed_data.pt')==m['processed_sha256']
    data=torch.load(out/'processed_data.pt',map_location='cpu')
    context={tuple(sorted(e)) for e in data.edge_index.t().tolist()}
    assert all(a!=b for a,b in context)
    seen=set()
    counts={}
    for split in ('train','val','test'):
        path=out/f'edge_sampled_2_10_only_{split}.jsonl'
        assert sha(path)==m['pair_sha256'][split]
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        for r in rows:
            pair=tuple(r['id'])
            assert pair==tuple(sorted(pair)) and pair[0]!=pair[1]
            assert 0<=pair[0]<pair[1]<data.num_nodes
            assert pair not in seen and pair not in context
            seen.add(pair)
        counts[split]=len(rows)
    if require_features:
        assert m['features_complete']
        for filename,digest in m['feature_sha256'].items():
            path=out/filename
            assert sha(path)==digest
            x=torch.load(path,map_location='cpu')
            assert x.shape==(data.num_nodes,768) and torch.isfinite(x).all()
        assert m['encoder']['revision']
    print(name,counts,'PASS',flush=True)
    return m.get('encoder') if m.get('features_complete') else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--datasets',nargs='+',default=['ogbn-arxiv','pubmed','cora'])
    p.add_argument('--graph-only',action='store_true')
    a=p.parse_args()
    encoders=[verify(a.root,n,not a.graph_only) for n in a.datasets]
    if not a.graph_only:
        assert all(e==encoders[0] for e in encoders),'Cross-graph encoder recipes differ.'


if __name__=='__main__':
    main()
