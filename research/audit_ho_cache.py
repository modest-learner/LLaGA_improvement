"""Compare supplied HO caches with a stated full-graph normalization hypothesis.

A match is consistency evidence, not proof of the upstream cache provenance.
Uses CPU; does not overwrite features or change the graph.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, degree


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['cora', 'pubmed', 'ogbn-arxiv'])
    p.add_argument('--components', nargs='+', default=['simteg_sbert', 'simteg_roberta', 'simteg_e5'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--self-loops', choices=['add', 'keep'], default='add',
                   help='add: original audit hypothesis; keep: match utils/data_process.py as stored.')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    report = dict(hypothesis=f'stored edge_index, self_loops={args.self_loops}, symmetric degree(col) normalization, incoming sum; full graph', datasets={})
    for dataset in args.datasets:
        root = Path('dataset') / dataset
        data = torch.load(root / 'processed_data.pt', map_location='cpu')
        edge = data.edge_index
        if args.self_loops == 'add':
            edge, _ = remove_self_loops(edge)
            edge, _ = add_self_loops(edge)
        inv = degree(edge[1], data.num_nodes).pow(-0.5)
        inv[~torch.isfinite(inv)] = 0
        mat = torch.sparse_coo_tensor(edge.flip(0), inv[edge[0]]*inv[edge[1]], (data.num_nodes, data.num_nodes)).coalesce()
        info = dict(graph_sha256=sha(root / 'processed_data.pt'), components={})
        sample = torch.randperm(data.num_nodes, generator=torch.Generator().manual_seed(42))[:128]
        for component in args.components:
            path = root / f'{component}_x.pt'
            x = torch.load(path, map_location='cpu').float()
            item = dict(shape=list(x.shape), base_sha256=sha(path), base_finite=bool(torch.isfinite(x).all()), hops=[])
            for h in range(1,5):
                x = torch.sparse.mm(mat, x)
                cache_path = root / f'{component}_{h}hop_x.pt'
                if not cache_path.exists():
                    item['hops'].append(dict(hop=h, missing=True))
                    continue
                cache = torch.load(cache_path, map_location='cpu')
                assert cache.shape == x.shape
                delta = cache[sample].float()-x[sample]
                item['hops'].append(dict(hop=h, sha256=sha(cache_path), finite=bool(torch.isfinite(cache).all()),
                                         sampled_relative_l2=float(delta.norm()/x[sample].norm().clamp_min(1e-12)),
                                         sampled_max_abs=float(delta.abs().max())))
                del cache
            info['components'][component] = item
            del x
        report['datasets'][dataset] = info
        print(dataset, json.dumps(info, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
