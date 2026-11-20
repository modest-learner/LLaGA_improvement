"""Validate a matched clean-LP result before GPU loading.

Exit 0: complete, 10: validated partial, 20: absent/empty. Other failures must
not be silently resumed. No output is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            d.update(b)
    return d.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',choices=['arxiv','pubmed','cora'],required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--ablation',choices=['full','center_repeat'],required=True)
    p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--model-base',type=Path,default=Path('models/vicuna-7b-v1.5-16k'))
    a=p.parse_args()
    if not a.output.exists() or not a.output.stat().st_size:
        print('Missing:',a.output)
        raise SystemExit(20)
    directory='ogbn-arxiv' if a.dataset=='arxiv' else a.dataset
    prompt=a.data_root/directory/f'edge_sampled_2_10_only_{a.split}.jsonl'
    expected=dict(protocol='lp_input_screen_v1',dataset=a.dataset,data_root=str(a.data_root.resolve()),
                  context='clean_cached',split=a.split,input_mode='graph',ablation=a.ablation,ablation_seed=42,
                  greedy=True,seed=0,temperature=0.2,max_new_tokens=1024,text_max_chars=2000,
                  use_hop=4,embedding='bert',model_path=str(a.checkpoint.resolve()),model_base=str(a.model_base.resolve()),
                  prompt_sha256=sha(prompt),code_sha256=sha('eval/eval_pretrain.py'),input_code_sha256=sha('eval/lp_inputs.py'),
                  start=-1,end=-1,projector_sha256=sha(a.checkpoint/'mm_projector.bin'),
                  manifest_sha256=sha(a.data_root/directory/'clean_lp_manifest.json'))
    refs=[json.loads(s) for s in prompt.read_text().splitlines()]
    rows=[json.loads(s) for s in a.output.read_text().splitlines()]
    assert len(rows)<=len(refs),'Output longer than split.'
    for i,(r,ref) in enumerate(zip(rows,refs)):
        assert r['lp_diagnostic_config']==expected,(a.output,i,'configuration differs')
        assert r['question_id']==ref['id'] and r['gt']==ref['conversations'][1]['value'],(a.output,i,'sample differs')
    print('Complete:' if len(rows)==len(refs) else 'Partial:',a.output,f'{len(rows)}/{len(refs)}')
    raise SystemExit(0 if len(rows)==len(refs) else 10)


if __name__=='__main__':
    main()
