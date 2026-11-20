"""Build independent LP contexts and frozen, shared BERT HO features.

Does not train a Projector. Original graph files are read only. Every supervised
positive (train/val/test) is removed from the shared message-passing graph.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, degree


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            d.update(chunk)
    return d.hexdigest()


def load_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def clean_splits(data, train, test, seed):
    """Filter original candidates without labels influencing model selection."""
    known = {tuple(sorted(e)) for e in data.edge_index.t().tolist() if e[0] != e[1]}
    stats = {}
    accepted = {}
    seen = set()
    for split, rows in [('train', train), ('test', test)]:
        out = []
        counts = Counter()
        # Any conflicting labels on the same pair are removed in this split.
        label_sets = {}
        for r in rows:
            pair = tuple(sorted(map(int,r['id'])))
            label_sets.setdefault(pair, set()).add(r['conversations'][1]['value'].strip().lower())
        for r in rows:
            pair = tuple(sorted(map(int,r['id'])))
            label = r['conversations'][1]['value'].strip().lower()
            if label not in ('yes','no') or min(pair) < 0 or max(pair) >= data.num_nodes:
                counts['invalid'] += 1
            elif pair[0] == pair[1]:
                counts['self_pair'] += 1
            elif len(label_sets[pair]) != 1:
                counts['conflicting_labels'] += 1
            elif (pair in known) != (label == 'yes'):
                counts['inconsistent_with_known_graph'] += 1
            elif pair in seen:
                counts['duplicate_or_cross_split'] += 1
            else:
                seen.add(pair)
                out.append((pair, label))
        accepted[split] = out
        stats[split] = dict(counts)
    rng = random.Random(seed)
    train_groups = {label:[r for r in accepted['train'] if r[1] == label] for label in ('yes','no')}
    val = []
    final_train = []
    for items in train_groups.values():
        rng.shuffle(items)
        k = max(1, int(len(items)*0.1))
        if len(items) <= k:
            raise ValueError('Too few source training pairs.')
        val.extend(items[:k])
        final_train.extend(items[k:])
    rng.shuffle(val)
    rng.shuffle(final_train)
    splits = dict(train=final_train, val=val, test=accepted['test'])
    forbidden = {p for rows in splits.values() for p,y in rows if y=='yes'}
    context = sorted(known-forbidden)
    context_set = set(context)
    assert all(p not in context_set for p in forbidden)
    all_pairs = [p for rows in splits.values() for p,_ in rows]
    assert len(all_pairs)==len(set(all_pairs))
    stats['split_counts'] = {s:dict(Counter(y for _,y in rows)) for s,rows in splits.items()}
    stats['context_undirected_edges'] = len(context)
    stats['removed_supervised_positive_edges'] = len(forbidden)
    return splits, context, stats


def node_texts(data):
    # Exactly the same policy on all graphs. No y or label_texts access.
    texts = []
    for i in range(data.num_nodes):
        title = str(data.title[i] or '').strip()
        abstract = str(data.abs[i] or '').strip()
        text = '\n'.join(s for s in (title, abstract) if s)
        texts.append(text or '[EMPTY TEXT]')
    return texts


def graph_stage(args, name):
    source = args.source_root/name
    out = args.output_root/name
    inputs = {s:sha(source/f'edge_sampled_2_10_only_{s}.jsonl') for s in ('train','test')}
    recipe = dict(protocol='disjoint_context_lp_v1', seed=args.seed, source_graph_sha256=sha(source/'processed_data.pt'),
                  source_pair_sha256=inputs, graph_policy='undirected_unique_no_self_loops; remove union of all supervised positives; source train 10% stratified val',
                  embedding='bert', embedding_dim=768)
    manifest_path = out/'clean_lp_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['recipe'] == recipe, 'Existing dataset has a different recipe; use a new output root.'
        assert sha(out/'processed_data.pt') == manifest['processed_sha256']
        for split,digest in manifest['pair_sha256'].items():
            assert sha(out/f'edge_sampled_2_10_only_{split}.jsonl') == digest
        return manifest
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Partial output exists: {out}; inspect it and use a fresh output root.')
    out.mkdir(parents=True,exist_ok=True)
    data = torch.load(source/'processed_data.pt',map_location='cpu')
    splits,context,stats = clean_splits(data, load_rows(source/'edge_sampled_2_10_only_train.jsonl'),
                                       load_rows(source/'edge_sampled_2_10_only_test.jsonl'), args.seed)
    edges = torch.tensor(context,dtype=torch.long).t().contiguous()
    data.edge_index = torch.cat([edges,edges.flip(0)],dim=1)
    torch.save(data,out/'processed_data.pt')
    for split,rows in splits.items():
        with (out/f'edge_sampled_2_10_only_{split}.jsonl').open('w') as f:
            for (u,v),label in rows:
                f.write(json.dumps(dict(id=[u,v],graph=[[u],[v]], conversations=[
                    dict(**{'from':'human','value':'Given two node-centered subgraphs: <graph> and <graph>, predict whether their center nodes connect.'}),
                    dict(**{'from':'gpt','value':label})]))+'\n')
    manifest = dict(**recipe, recipe=recipe, features_complete=False, stats=stats,
                    processed_sha256=sha(out/'processed_data.pt'),
                    pair_sha256={s:sha(out/f'edge_sampled_2_10_only_{s}.jsonl') for s in splits})
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(name,stats,flush=True)
    return manifest


def feature_stage(args):
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer, AutoModel
    lock_path = args.output_root/'encoder_lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        assert lock['repo'] == args.encoder
        revision = lock['revision']
        if args.revision != 'main' and args.revision != revision:
            raise ValueError('Encoder revision differs from lock.')
    else:
        revision = HfApi().model_info(args.encoder,revision=args.revision).sha
        lock = dict(repo=args.encoder,revision=revision)
        args.output_root.mkdir(parents=True,exist_ok=True)
        lock_path.write_text(json.dumps(lock,indent=2)+'\n')
    path = snapshot_download(args.encoder,revision=revision,cache_dir=str(args.encoder_cache),
                             allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json'])
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval().to(args.device)
    model.requires_grad_(False)
    assert model.config.hidden_size == 768
    weight_hash = sha(Path(path)/'pytorch_model.bin')
    encoder = dict(**lock,weight_sha256=weight_hash,pooling='masked mean excluding special/pad tokens; no L2 normalization',
                   text='title + newline + abstract; empty fallback [EMPTY TEXT]; no label appended',max_length=args.max_length,
                   torch_version=torch.__version__,dtype='float32',device_type=str(args.device).split(':')[0])
    for name in args.datasets:
        out=args.output_root/name
        manifest=json.loads((out/'clean_lp_manifest.json').read_text())
        if manifest['features_complete']:
            assert manifest['encoder']==encoder,'Existing features have a different encoder recipe.'
            for file,digest in manifest['feature_sha256'].items():
                assert sha(out/file)==digest
            print(name,'features already complete',flush=True)
            continue
        data=torch.load(out/'processed_data.pt',map_location='cpu')
        texts=node_texts(data)
        recipe = dict(encoder=encoder,text_sha256=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest())
        cache_meta=out/'bert_base_manifest.json'
        cache_x=out/'bert_x.pt'
        if cache_x.exists() and cache_meta.exists():
            meta=json.loads(cache_meta.read_text())
            assert meta['recipe']==recipe and meta['sha256']==sha(cache_x)
            x=torch.load(cache_x,map_location='cpu')
        else:
            batches=[]
            from tqdm import tqdm
            with torch.inference_mode():
                for start in tqdm(range(0,len(texts),args.batch_size),desc=name+' BERT'):
                    batch=tokenizer(texts[start:start+args.batch_size],padding=True,truncation=True,
                                    max_length=args.max_length,return_special_tokens_mask=True,return_tensors='pt')
                    special=batch.pop('special_tokens_mask').to(args.device)
                    batch={k:v.to(args.device) for k,v in batch.items()}
                    hidden=model(**batch).last_hidden_state
                    mask=(batch['attention_mask'].bool() & ~special.bool()).unsqueeze(-1)
                    pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                    batches.append(pooled.cpu())
            x=torch.cat(batches)
            assert x.shape==(data.num_nodes,768) and torch.isfinite(x).all()
            temp=cache_x.with_suffix('.part')
            torch.save(x,temp)
            temp.replace(cache_x)
            cache_meta.write_text(json.dumps(dict(recipe=recipe,sha256=sha(cache_x)),indent=2)+'\n')
        edge,_=remove_self_loops(data.edge_index)
        edge,_=add_self_loops(edge)
        inv=degree(edge[1],data.num_nodes).pow(-0.5)
        inv[~torch.isfinite(inv)]=0
        mat=torch.sparse_coo_tensor(edge.flip(0),inv[edge[0]]*inv[edge[1]],(data.num_nodes,data.num_nodes)).coalesce()
        for h in range(1,5):
            x=torch.sparse.mm(mat,x)
            assert torch.isfinite(x).all()
            file=out/f'bert_{h}hop_x.pt'
            temp=file.with_suffix('.part')
            torch.save(x,temp)
            temp.replace(file)
        manifest.update(features_complete=True,encoder=encoder,
                        feature_sha256={f.name:sha(f) for f in [cache_x]+[out/f'bert_{h}hop_x.pt' for h in range(1,5)]})
        (out/'clean_lp_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(name,'features complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['graph','features','all'],default='all')
    p.add_argument('--source-root',type=Path,default=Path('dataset'))
    p.add_argument('--output-root',type=Path,default=Path('dataset_research/lp_bert_disjoint_v1'))
    p.add_argument('--datasets',nargs='+',default=['ogbn-arxiv','pubmed','cora'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--encoder',default='google-bert/bert-base-uncased')
    p.add_argument('--revision',default='86b5e0934494bd15c9632b12f734a8a67f723594')
    p.add_argument('--encoder-cache',type=Path,default=Path('models/encoder_cache'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-length',type=int,default=256)
    args=p.parse_args()
    torch.set_num_threads(4)
    if not 1<=args.max_length<=512:
        raise ValueError('BERT max length must be 1..512.')
    for name in args.datasets:
        graph_stage(args,name)
    if args.stage in ('features','all'):
        feature_stage(args)


if __name__=='__main__':
    main()
