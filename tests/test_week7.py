"""CPU-only protocol tests; run with unittest discover."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from week7_common import (
    fuse, fusion_weights, ranking_metrics, scheduler_ratio, train_items,
    unit_rows, write_json, write_jsonl,
)
from week7_report import paired_bootstrap


class Week7Tests(unittest.TestCase):
    def test_unit_rows_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            unit_rows(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            unit_rows([[np.nan, 1]])

    def test_four_groups_and_matched_energy(self):
        text = np.arange(1, 41).reshape(10, 4)
        collab = np.arange(1, 21).reshape(10, 2)
        weights, meta = fusion_weights(np.arange(1, 11))
        self.assertTrue(np.all(np.diff(weights["adaptive"]) >= 0))
        for name, weight in weights.items():
            value = fuse(text, collab, weight)
            self.assertEqual(value.shape, (10, 6))
            np.testing.assert_allclose(np.linalg.norm(value, axis=1), 1, atol=1e-6)
            if name == "text":
                self.assertTrue(np.all(value[:, 4:] == 0))
        adaptive_energy = np.mean(weights["adaptive"] ** 2 /
                                  (1 + weights["adaptive"] ** 2))
        w = meta["matched_weight"]
        self.assertAlmostEqual(adaptive_energy, w * w / (1 + w * w))

    def test_train_reader_never_needs_valid_or_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "semantic_ids.json", {"0": [0, 0], "1": [1, 0]})
            write_jsonl(root / "train.semantic.jsonl",
                        [{"user": 7, "items": [[0, 0], [1, 0]]}])
            # Invalid evaluation files must not affect a train-only feature reader.
            (root / "valid.semantic.jsonl").write_text("not JSON")
            (root / "test.semantic.jsonl").write_text("not JSON")
            rows, _ = train_items(root)
            self.assertEqual(rows, [{"user": 7, "items": [0, 1]}])

    def test_duplicate_semantic_ids_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "semantic_ids.json", {"0": [0], "1": [0]})
            with self.assertRaises(ValueError):
                train_items(root)

    def test_metrics_by_hand(self):
        value = ranking_metrics([9, 8, 7, 6, 5, 4], 4)
        self.assertEqual(value["recall@5"], 0)
        self.assertEqual(value["recall@10"], 1)
        self.assertAlmostEqual(value["ndcg@10"], 1 / math.log2(7))

    def test_scheduler(self):
        self.assertEqual(scheduler_ratio(0, 1500, 60000), 0)
        self.assertEqual(scheduler_ratio(1500, 1500, 60000), 1)
        self.assertAlmostEqual(scheduler_ratio(60000, 1500, 60000), 0.05)

    def test_bootstrap_reproducible(self):
        first = paired_bootstrap([0.1, 0.2, 0.3], resamples=128)
        second = paired_bootstrap([0.1, 0.2, 0.3], resamples=128)
        self.assertEqual(first, second)
        self.assertGreater(first["ci95"][0], 0)
        self.assertIsNone(paired_bootstrap([])["ci95"])

    def test_svd_determinism(self):
        try:
            from scipy.sparse import csr_matrix
            from sklearn.decomposition import TruncatedSVD
        except ImportError:
            self.skipTest("requires scipy and scikit-learn")
        matrix = csr_matrix(np.eye(8, dtype=np.float32) +
                            np.ones((8, 8), dtype=np.float32))
        first = TruncatedSVD(n_components=3, random_state=42).fit_transform(matrix)
        second = TruncatedSVD(n_components=3, random_state=42).fit_transform(matrix)
        np.testing.assert_allclose(first, second)

    def test_token_roundtrip_and_trie(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")
        from generative_common import SemanticArtifacts, SemanticTrie
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "semantic_ids.json",
                       {"0": [0, 0, 0, 0], "1": [0, 0, 1, 0], "2": [1, 1, 1, 0]})
            write_json(root / "token_vocab.json",
                       {"codebook_size": 2, "semantic_id_length": 4, "vocab_size": 11})
            a = SemanticArtifacts(root)
            for code in a.item_to_code.values():
                self.assertEqual(a.tokens_to_code(a.code_to_tokens(code)), code)
            trie = SemanticTrie(a)
            seen = {0, 1}
            self.assertEqual(trie.allowed([], seen), [a.code_to_tokens([1, 1, 1, 0])[0]])
            self.assertEqual(trie.allowed(a.code_to_tokens([0, 0, 0, 0]), seen), [])

    def test_resumable_batch_sequence(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")
        from week7_train import RestartableBatches
        import itertools
        uninterrupted = list(itertools.islice(iter(RestartableBatches(11, 3, 42)), 12))
        resumed = list(itertools.islice(iter(RestartableBatches(11, 3, 42, start=5)), 7))
        self.assertEqual(uninterrupted[5:], resumed)

    def test_teacher_forcing_shift_and_logprob(self):
        try:
            import torch
            from transformers import T5Config, T5ForConditionalGeneration
        except ImportError:
            self.skipTest("requires torch and transformers")
        torch.manual_seed(42)
        model = T5ForConditionalGeneration(T5Config(
            vocab_size=19, d_model=16, d_ff=32, num_layers=1,
            num_decoder_layers=1, num_heads=2, d_kv=8,
            pad_token_id=0, decoder_start_token_id=1, eos_token_id=2,
        )).eval()
        labels = torch.tensor([[3, 7, 11, 15, 2]])
        self.assertEqual(model._shift_right(labels).tolist(), [[1, 3, 7, 11, 15]])
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([[3, 7, 11, 15, 2]]),
                           labels=labels).logits
            vector_score = logits.log_softmax(-1).gather(-1, labels[..., None]).sum()
            hand_score = sum(logits[0, t].log_softmax(-1)[int(labels[0, t])]
                             for t in range(5))
        self.assertAlmostEqual(float(vector_score), float(hand_score), places=5)


if __name__ == "__main__":
    unittest.main()