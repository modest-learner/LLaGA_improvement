"""CPU integration check: held-out labels cannot change learned pair models."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.run_lp_pair_baselines import structure_features
from research.run_lp_simple_baselines import heuristic_scores,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,default=Path('/tmp/llaga_lp_baseline_fixture_20261117'))
    a=p.parse_args();torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='lp_phase2_check_') as temporary:
        root=Path(temporary)
        for name in ('original','flipped'):
            shutil.copytree(a.fixture,root/name)
        target=root/'flipped/pubmed/edge_sampled_2_10_only_test.jsonl'
        rows=[json.loads(s) for s in target.read_text().splitlines()]
        for row in rows:
            answer=row['conversations'][1]
            answer['value']='no' if answer['value']=='yes' else 'yes'
        target.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        manifest=target.parent/'clean_lp_manifest.json'
        m=json.loads(manifest.read_text());m['pair_sha256']['test']=sha(target)
        manifest.write_text(json.dumps(m))
        # No target train/val input is available in this test.
        for split in ('train','val'):(target.parent/f'edge_sampled_2_10_only_{split}.jsonl').unlink()
        for name in ('original','flipped'):
            cmd=[sys.executable,'research/run_lp_pair_baselines.py','--sources','arxiv','cora','--target','pubmed',
                 '--data-root',str(root/name),'--output',str(root/(name+'_out')),'--device','cpu',
                 '--seeds','42','--epochs','2','--hidden','8','--batch-size','8']
            subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for kind in ('endpoint','structure','fusion'):
            left=torch.load(root/f'original_out/{kind}_mlp_seed42.pt',map_location='cpu')
            right=torch.load(root/f'flipped_out/{kind}_mlp_seed42.pt',map_location='cpu')
            assert left['threshold']==right['threshold'] and left['epoch']==right['epoch']
            assert all(torch.equal(v,right['state_dict'][k]) for k,v in left['state_dict'].items())
            x=np.load(root/f'original_out/pubmed_{kind}_MLP_seed42_scores.npz')
            y=np.load(root/f'flipped_out/pubmed_{kind}_MLP_seed42_scores.npz')
            assert np.array_equal(x['scores'],y['scores'])
        assert len(list(csv.DictReader((root/'original_out/metrics.csv').open())))==24
        graph=torch.load(a.fixture/'cora/processed_data.pt',map_location='cpu')
        x=torch.load(a.fixture/'cora/bert_x.pt',map_location='cpu')
        pairs=np.array([[0,1],[1,2]])
        forward=structure_features(graph,pairs,heuristic_scores(graph,x,pairs))
        backward=structure_features(graph,pairs[:,::-1].copy(),heuristic_scores(graph,x,pairs[:,::-1].copy()))
        assert torch.allclose(forward,backward)
        rejected=subprocess.run([sys.executable,'research/run_lp_pair_baselines.py','--sources','arxiv','pubmed',
            '--target','pubmed','--output',str(root/'bad'),'--device','cpu'],capture_output=True,text=True)
        assert rejected.returncode!=0 and 'Target must be held out' in rejected.stderr
    print('PASS: source/target isolation, target-label flip invariance of weights/threshold/score, symmetric structural features, complete metrics.')


if __name__=='__main__':main()
