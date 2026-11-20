"""Check completed source LP training; this does not resume optimizer state."""
import argparse
import json
from pathlib import Path
import re
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--ablation',choices=['full','center_repeat'],required=True)
    p.add_argument('--sources',default='arxiv-pubmed',choices=['arxiv','pubmed','cora','arxiv-pubmed','arxiv-cora','pubmed-cora'])
    p.add_argument('--data-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--effective-batch-size',type=int,help='Verify GPU count x microbatch x accumulation from the logged command; permits single-GPU execution.')
    a=p.parse_args()
    if not a.checkpoint.exists():
        raise SystemExit(20)
    required=['mm_projector.bin','config.json','trainer_state.json']
    if any(not (a.checkpoint/f).is_file() for f in required):
        raise RuntimeError(f'Incomplete training directory: {a.checkpoint}. Original adapter saves lack optimizer state; inspect before restarting in a fresh checkpoint path.')
    state=json.loads((a.checkpoint/'trainer_state.json').read_text())
    assert state['global_step']==state['max_steps'] and state['epoch']>.99,'Training incomplete.'
    config=json.loads((a.checkpoint/'config.json').read_text())
    assert config['mm_projector_type']=='linear' and config['mm_hidden_size']==768
    weights=torch.load(a.checkpoint/'mm_projector.bin',map_location='cpu')
    assert all(torch.isfinite(v).all() for v in weights.values())
    assert a.log.is_file(),'Training command log missing; cannot verify recipe.'
    text=a.log.read_text(errors='replace')
    paths=re.findall(r'--data_root\s+(\S+)',text)
    assert paths and all(Path(s).resolve()==a.data_root.resolve() for s in paths),'Training data_root differs.'
    for flag,value in [('seed',str(a.seed)),('use_task','lp'),('use_dataset',a.sources),
                       ('graph_input_ablation',a.ablation),('embedding_dim','768'),('pretrained_embedding_type','bert'),
                       ('learning_rate','2e-3')]:
        assert re.search(r'--'+flag+r'\s+'+re.escape(value)+r'(?:\s|$)',text),(flag,value)
    if a.effective_batch_size is None:
        assert re.search(r'--gradient_accumulation_steps\s+2(?:\s|$)',text),'gradient_accumulation_steps'
    else:
        layouts=re.findall(r'--include\s+[\"\']?(localhost:[0-9,]+)',text)
        batches=re.findall(r'--per_device_train_batch_size\s+(\d+)',text)
        accumulations=re.findall(r'--gradient_accumulation_steps\s+(\d+)',text)
        assert layouts and batches and accumulations,'Missing batch layout in training command log.'
        assert len(set(layouts))==len(set(batches))==len(set(accumulations))==1,'Mixed training layouts in one log.'
        devices=layouts[0].split(':')[1].split(',')
        assert len(set(devices))==len(devices),'Duplicate training GPUs.'
        effective=len(devices)*int(batches[0])*int(accumulations[0])
        assert effective==a.effective_batch_size,('effective batch size',effective,a.effective_batch_size)
    print('Completed training:',a.checkpoint,'steps',state['global_step'])


if __name__=='__main__':
    main()
