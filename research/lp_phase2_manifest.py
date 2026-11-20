"""Pin phase2 data/model/core implementation before running or resuming."""
import argparse
import json
from pathlib import Path
from check_lp_prediction import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--model-base',type=Path,required=True)
    a=p.parse_args()
    recipe=dict(protocol='lp_phase2_v1',data_root=str(a.data_root.resolve()),model_base=str(a.model_base.resolve()),
        directions=['arxiv-pubmed_to_cora','arxiv-cora_to_pubmed'],projector_seed=42,
        projector=dict(template='HO4',type='linear',embedding='bert',lr=.002,batch=16,epoch=1),
        manifests={n:sha(a.data_root/n/'clean_lp_manifest.json') for n in ['ogbn-arxiv','pubmed','cora']},
        model_files={n:sha(a.model_base/n) for n in ['config.json','tokenizer.model']},
        code={n:sha(n) for n in ['train/train.py','train/llaga_trainer.py','model/llaga_arch.py',
            'eval/eval_pretrain.py','eval/lp_inputs.py','scripts/train_deepspeed.sh',
            'research/run_lp_pair_baselines.py','research/run_lp_simple_baselines.py']})
    # Checkpoint names are shared regardless of result root: pin the recipe there too.
    for path in [a.root/'recipe.json',Path('checkpoints/lp_phase2/recipe.json')]:
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert json.loads(path.read_text())==recipe,f'Recipe changed: {path}; inspect before resuming.'
        else:path.write_text(json.dumps(recipe,indent=2)+'\n')
    print('Phase2 data/model/training recipe verified.',flush=True)


if __name__=='__main__':main()
