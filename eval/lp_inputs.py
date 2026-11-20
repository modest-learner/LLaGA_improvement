"""LP-only input interventions; never reads labels to choose donors."""
import hashlib
import json
import re

import torch
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops


LP_PROMPT = ('Given two node-centered subgraphs: <graph> and <graph>, we need to predict '
             'whether these two nodes connect with each other. Please tell me whether two '
             'center nodes in the subgraphs should connect to each other.')


def donors_for_pair(pair, n, seed):
    if n < 3:
        raise ValueError('Donors require at least 3 nodes.')
    excluded = set(map(int, pair))
    result = []
    for side in range(2):
        key = json.dumps([seed, list(map(int, pair)), side], separators=(',', ':'))
        node = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % n
        while node in excluded:
            node = (node + 1) % n
        result.append(node)
    return result


def pair_ho(mp, edge_index, features, centers, remove_pair, hop, device):
    """Same local normalization as official small-graph LP; also removes original query for donors."""
    subset, edges, mapping, _ = k_hop_subgraph(centers, hop, edge_index, relabel_nodes=True)
    a, b = map(int, remove_pair)
    original = subset[edges]
    remove = ((original[0] == a) & (original[1] == b)) | ((original[0] == b) & (original[1] == a))
    edges = edges[:, ~remove]
    local = features[subset].to(device)
    out = [features[torch.as_tensor(centers)].to(device)]
    edges, _ = remove_self_loops(edges)
    edges, _ = add_self_loops(edges)
    edges = edges.to(device)
    row, col = edges
    inv = degree(col, len(subset), dtype=features.dtype).pow(-0.5)
    inv[inv == float('inf')] = 0
    norm = inv[row] * inv[col]
    for _ in range(hop):
        local = mp.propagate(edges, x=local, norm=norm)
        out.append(local[mapping])
    return torch.stack(out, dim=1)


def intervene(embedding, mode, donor=None):
    if mode == 'full':
        return embedding
    if mode == 'center_only':
        return embedding[:, :1]
    if mode == 'center_repeat':
        return embedding[:, :1].expand_as(embedding).clone()
    if mode == 'hop_mismatch':
        if donor is None or donor.shape != embedding.shape:
            raise ValueError('Incompatible donor embeddings.')
        return torch.cat([embedding[:, :1], donor[:, 1:]], dim=1)
    raise ValueError(mode)


def endpoint_prompt(texts, pair, max_chars):
    a, b = (str(texts[int(i)])[:max_chars] for i in pair)
    # Raw text is data, not an additional graph-token placeholder.
    if '<graph>' in a or '<graph>' in b:
        raise ValueError('Reserved graph token in node text.')
    return ('Given the textual attributes of two nodes:\n'
            f'First node: {a}\nSecond node: {b}\n'
            'We need to predict whether these two nodes connect with each other. '
            'Please tell me whether these two nodes should connect to each other.')


def parse_binary(text):
    """Auxiliary lexical protocol: exactly one distinct whole-word yes/no, case insensitive."""
    hits = set(re.findall(r'\b(?:yes|no)\b', text.lower()))
    return next(iter(hits)) if len(hits) == 1 else None
