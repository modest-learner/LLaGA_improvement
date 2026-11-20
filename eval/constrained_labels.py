"""A token trie for exact label generation, independent of node ground truth."""


class LabelTrie:
    def __init__(self, label_tokens, eos_token_id):
        if eos_token_id is None or not label_tokens:
            raise ValueError('Labels and EOS are required.')
        self.eos_token_id = int(eos_token_id)
        self.transitions = {}
        self.labels = {}
        for label, tokens in label_tokens.items():
            tokens = tuple(int(token) for token in tokens)
            if not tokens or self.eos_token_id in tokens or tokens in self.labels:
                raise ValueError('Empty, duplicate, or EOS-containing label path.')
            self.labels[tokens] = label
            path = tokens + (self.eos_token_id,)
            for position, token in enumerate(path):
                self.transitions.setdefault(path[:position], set()).add(token)
        self.max_new_tokens = max(map(len, self.labels)) + 1

    def allowed(self, suffix):
        suffix = tuple(int(token) for token in suffix)
        if suffix not in self.transitions:
            raise ValueError(f'Generation left the label trie: {suffix}')
        return sorted(self.transitions[suffix])

    def finish(self, suffix):
        suffix = tuple(int(token) for token in suffix)
        if not suffix or suffix[-1] != self.eos_token_id:
            raise ValueError('Constrained answer did not terminate with EOS.')
        return self.labels[suffix[:-1]]


def build_label_trie(prompt, prefix_ids, candidates, tokenizer, tokenize_prompt):
    """Tokenize complete prompt+label, checking the exact continuation boundary."""
    prefix = prefix_ids[0].tolist()
    label_tokens = {}
    for label in candidates:
        tokens = tokenize_prompt(prompt + ' ' + label, tokenizer)
        if tokens[:len(prefix)] != prefix:
            raise ValueError('Label changes prompt tokenization boundary.')
        suffix = tokens[len(prefix):]
        if tokenizer.decode(suffix, skip_special_tokens=True).strip() != label:
            raise ValueError('Label tokenization does not round trip.')
        label_tokens[label] = suffix
    return LabelTrie(label_tokens, tokenizer.eos_token_id)
