"""Meaningful grammar checks: prefix labels, EOS, and tokenizer boundary failures."""
import unittest

from eval.constrained_labels import LabelTrie, build_label_trie


class TrieTests(unittest.TestCase):
    def test_shared_prefix_and_terminal_prefix(self):
        trie = LabelTrie({'a': [10], 'ab': [10, 11], 'c': [12, 13]}, 2)
        self.assertEqual(trie.allowed([]), [10, 12])
        self.assertEqual(trie.allowed([10]), [2, 11])
        for path, label in [([10, 2], 'a'), ([10, 11, 2], 'ab'), ([12, 13, 2], 'c')]:
            for i, token in enumerate(path):
                self.assertIn(token, trie.allowed(path[:i]))
            self.assertEqual(trie.finish(path), label)
        with self.assertRaises(ValueError):
            trie.finish([10])
        with self.assertRaises(ValueError):
            trie.allowed([99])

    def test_duplicate_path_is_rejected(self):
        with self.assertRaises(ValueError):
            LabelTrie({'a': [10], 'b': [10]}, 2)

    def test_context_boundary_is_checked(self):
        import torch
        class Tokenizer:
            eos_token_id = 2
            def decode(self, tokens, **kwargs):
                return ' a'
        with self.assertRaises(ValueError):
            build_label_trie('prompt', torch.tensor([[1, 5]]), ['a'], Tokenizer(),
                             lambda text, tok: [1, 9, 10])


if __name__ == '__main__':
    unittest.main()
