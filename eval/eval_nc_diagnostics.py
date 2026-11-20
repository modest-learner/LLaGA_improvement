"""Replay saved Cora/HO prompts for inference ablations and candidate scoring.

Uses the original loader, HO features and conversation template. Does not alter
the original evaluator, predictions, graph caches, or trained Projector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tqdm import tqdm
from transformers import set_seed
from eval.constrained_labels import build_label_trie
from eval.eval_pretrain import load_pretrain_embedding_hop
from model.builder import load_pretrained_model
from utils.constants import DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
from utils.conversation import conv_templates, SeparatorStyle
from utils.utils import disable_torch_init, get_model_name_from_path, tokenizer_graph_token


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def donor_map(ids, seed):
    """Random cycle: bijective, no self-donors; independent of labels/resume."""
    if len(ids) < 2:
        raise ValueError('Neighbor permutation requires at least two nodes.')
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ids), generator=generator).tolist()
    return {ids[order[i]]: ids[order[(i + 1) % len(ids)]] for i in range(len(ids))}


def ablate_embedding(embedding, ablation, donor=None):
    if ablation == 'full':
        return embedding
    if ablation == 'center_only':
        return embedding[:1]
    if ablation == 'center_repeat':
        return embedding[:1].expand_as(embedding).clone()
    if ablation == 'neighbor_shuffle':
        if donor is None or donor.shape != embedding.shape:
            raise ValueError('Missing or incompatible donor embedding.')
        return torch.cat([embedding[:1], donor[1:]], dim=0)
    raise ValueError(ablation)


@torch.inference_mode()
def score_candidates(model, tokenizer, prompt, prefix_ids, candidates, graph_kwargs):
    """Teacher-forced mean/sum log P of each complete label; no EOS scored."""
    scores = {}
    for label in candidates:
        full_ids = tokenizer_graph_token(prompt + ' ' + label, tokenizer,
                                        GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        prefix_length = prefix_ids.shape[1]
        if not torch.equal(full_ids[:, :prefix_length], prefix_ids):
            raise ValueError('Candidate changes the prompt tokenization boundary.')
        target = full_ids[:, prefix_length:]
        count = target.shape[1]
        if count < 1:
            raise ValueError('Empty candidate.')
        output = model(input_ids=full_ids, **graph_kwargs, use_cache=False, return_dict=True)
        # Graph insertion changes the prefix length but not the answer suffix.
        logits = output.logits[:, -(count + 1):-1, :].float()
        log_probs = logits.log_softmax(dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        if not torch.isfinite(log_probs).all():
            raise ValueError('Non-finite candidate score.')
        scores[label] = {'mean_logprob': log_probs.mean().item(),
                         'sum_logprob': log_probs.sum().item(), 'tokens': count}
    best_mean = max(candidates, key=lambda label: scores[label]['mean_logprob'])
    best_sum = max(candidates, key=lambda label: scores[label]['sum_logprob'])
    return best_mean, best_sum, scores


def main(args):
    source = Path(args.source_results)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    dataset_dir = Path('dataset') / {'cora': 'cora', 'pubmed': 'pubmed', 'arxiv': 'ogbn-arxiv'}[args.dataset]
    official = [json.loads(line) for line in (dataset_dir / 'sampled_2_10_test.jsonl').read_text().splitlines()]
    ids = [row['question_id'] for row in rows]
    expected_ids = [row['id'] for row in official]
    if args.source_prefix:
        expected_ids = expected_ids[:len(ids)]
    if not ids or ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError('Source must contain the ordered test set (or its prefix with --source_prefix).')
    if any(row['gt'] != ref['conversations'][1]['value'] for row, ref in zip(rows, official)):
        raise ValueError('Source labels do not match the original test file.')
    config = rows[0]['eval_config']
    if any(row['eval_config'] != config for row in rows):
        raise ValueError('Source mixes evaluation configurations.')
    text_only = config['input_mode'] == 'text_only'
    if config['task'] not in ('nc', 'nctext') or config['template'] != 'HO':
        raise ValueError('This diagnostic supports HO NC/nctext only.')
    if text_only and args.ablation != 'full':
        raise ValueError('Graph ablations do not apply to T-only.')
    if not text_only and (config['use_hop'] != 4 or config['embedding'] != 'simteg'):
        raise ValueError('Expected the existing 4-hop SimTeG baseline.')
    for row in rows:
        if row['prompt'].count(DEFAULT_GRAPH_TOKEN) != (0 if text_only else 1):
            raise ValueError('Unexpected number of graph tokens in source prompt.')
    if not config['do_sample'] or config['temperature'] != 0.2 or config['max_new_tokens'] != 1024:
        raise ValueError('Expected the original Table 8 sampling configuration.')

    model_path, model_base = config['model_path'], config['model_base']
    checkpoint_file = Path(model_path) / ('config.json' if text_only else 'mm_projector.bin')
    constrained = args.decoding == 'constrained_greedy'
    greedy = args.decoding in ('greedy', 'constrained_greedy')
    run_config = dict(protocol='ho_output_diagnostics_v2' if greedy else 'cora_ho_diagnostics_v1', source=str(source.resolve()),
                      source_sha256=sha256_file(source), source_eval_config=config,
                      checkpoint_sha256=sha256_file(checkpoint_file), ablation=args.ablation,
                      ablation_seed=args.ablation_seed if args.ablation == 'neighbor_shuffle' else None,
                      decoding=args.decoding, inference_seed=args.inference_seed if args.decoding == 'generate' else None,
                      candidate_score='mean_token_logprob_no_eos' if args.decoding == 'label_score' else None,
                      conv_mode='v1', max_new_tokens=args.max_new_tokens if args.decoding in ('generate', 'greedy') else None)
    if greedy or args.dataset != 'cora' or args.source_prefix:
        run_config.update(dataset=args.dataset, source_prefix=args.source_prefix,
                          implementation_sha256=sha256_file(__file__),
                          constraint_implementation_sha256=sha256_file(Path(__file__).with_name('constrained_labels.py')),
                          label_policy='original_label_strings_with_leading_space_and_eos' if constrained else None,
                          search='greedy_one_beam_no_length_normalization' if greedy else None)
    # Build against the full source set before slicing/resuming.
    donors = donor_map(ids, args.ablation_seed) if args.ablation == 'neighbor_shuffle' else {}
    selected = rows if args.limit is None else rows[:args.limit]
    output_path = Path(args.answers_file)
    if output_path.resolve() == source.resolve():
        raise ValueError('Output must not overwrite source predictions.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = []
    if output_path.exists():
        done = [json.loads(line) for line in output_path.read_text().splitlines()]
        if len(done) > len(selected):
            raise ValueError('Output contains more samples than requested.')
        for prev, ref in zip(done, selected):
            if (prev['question_id'] != ref['question_id'] or prev['prompt'] != ref['prompt']
                    or prev['gt'] != ref['gt'] or prev['diagnostic_config'] != run_config):
                raise ValueError('Existing output has a different protocol or sample order; use a new filename.')
    if len(done) == len(selected):
        print(f'Already complete: {len(done)} samples in {output_path}')
        return

    data = torch.load(dataset_dir / 'processed_data.pt')
    candidates = list(data.label_texts)  # All labels, identical for every node.
    features = positions = None
    if not text_only:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[ids] = True
        hops = load_pretrain_embedding_hop(str(dataset_dir), 'simteg', 4, mask)
        features = torch.stack(hops, dim=1)
        positions = {idx: i for i, idx in enumerate(mask.nonzero().flatten().tolist())}
        if not torch.isfinite(features).all():
            raise ValueError('Non-finite input features.')
        del hops
    disable_torch_init()
    name = 'text_only_base' if text_only else get_model_name_from_path(model_path)
    tokenizer, model, _ = load_pretrained_model(model_path, model_base, name,
                                               device_map={'': torch.cuda.current_device()},
                                               cache_dir='../../checkpoint')
    model = model.eval()
    print(f'{args.decoding}; {args.ablation}; {len(selected)} nodes; model={model_path}', flush=True)
    with output_path.open('a') as handle:
        for row in tqdm(selected[len(done):]):
            idx = row['question_id']
            conv = conv_templates['v1'].copy()
            conv.append_message(conv.roles[0], row['prompt'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()
            graph_kwargs, graph_record = {}, []
            if not text_only:
                donor = features[positions[donors[idx]]] if donors else None
                emb = ablate_embedding(features[positions[idx]], args.ablation, donor)
                graph = torch.full((1, emb.shape[0]), idx, dtype=torch.long, device='cuda')
                graph_kwargs = {'graph': graph, 'graph_emb': emb.unsqueeze(0).half().cuda()}
                graph_record = graph.tolist()
            extra = {}
            if args.decoding in ('generate', 'greedy', 'constrained_greedy'):
                seed_key = json.dumps([args.inference_seed, args.dataset, idx], separators=(',', ':'))
                set_seed(int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], 'big'))
                generation_kwargs = dict(do_sample=not greedy, num_beams=1,
                                         max_new_tokens=args.max_new_tokens, use_cache=True)
                if not greedy:
                    generation_kwargs.update(temperature=0.2, top_p=None)
                trie = None
                if constrained:
                    trie = build_label_trie(prompt, input_ids, candidates, tokenizer,
                                             lambda text, tok: tokenizer_graph_token(text, tok, GRAPH_TOKEN_INDEX))
                    prefix_length = input_ids.shape[1]
                    generation_kwargs.update(
                        prefix_allowed_tokens_fn=lambda batch, tokens: trie.allowed(tokens[prefix_length:].tolist()),
                        max_new_tokens=trie.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id)
                with torch.inference_mode():
                    generated = model.generate(input_ids, **graph_kwargs, **generation_kwargs)
                if not torch.equal(generated[:, :input_ids.shape[1]], input_ids):
                    raise ValueError('Generation changed input token IDs.')
                text = tokenizer.batch_decode(generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                if text.endswith(stop_str):
                    text = text[:-len(stop_str)].strip()
                if trie is not None:
                    suffix = generated[0, input_ids.shape[1]:].tolist()
                    label = trie.finish(suffix)
                    if text != label:
                        raise ValueError('Decoded output differs from completed label path.')
                    extra = dict(generated_token_ids=suffix, constraint_max_new_tokens=trie.max_new_tokens,
                                 candidate_token_counts={label: len(tokens) for tokens, label in trie.labels.items()})
            else:
                text, best_sum, scores = score_candidates(model, tokenizer, prompt, input_ids, candidates, graph_kwargs)
                extra = {'candidate_scores': scores, 'prediction_sum_logprob': best_sum}
            result = dict(question_id=idx, prompt=row['prompt'], graph=graph_record,
                          text=text, gt=row['gt'], diagnostic_config=run_config,
                          donor_id=donors.get(idx), **extra)
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_results', required=True)
    parser.add_argument('--dataset', choices=['cora', 'pubmed', 'arxiv'], default='cora')
    parser.add_argument('--source_prefix', action='store_true', help='Allow a saved prefix for engineering checks, not a full evaluation.')
    parser.add_argument('--answers_file', required=True)
    parser.add_argument('--ablation', choices=['full', 'center_only', 'center_repeat', 'neighbor_shuffle'], default='full')
    parser.add_argument('--decoding', choices=['generate', 'label_score', 'greedy', 'constrained_greedy'], default='generate')
    parser.add_argument('--inference_seed', type=int, default=0)
    parser.add_argument('--ablation_seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--limit', type=int, default=None, help='Smoke tests only; omit for all 542 nodes.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.max_new_tokens < 1:
        parser.error('--max_new_tokens must be positive')
    main(args)
