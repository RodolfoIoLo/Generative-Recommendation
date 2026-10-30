"""CPU-only protocol tests; run with unittest discover."""
from __future__ import annotations

import json
import errno
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

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


class GPUGuardTests(unittest.TestCase):
    @staticmethod
    def samples(*snapshots):
        results = []
        for memory, utilization, processes in snapshots:
            results += [Mock(stdout=f"{memory}, {utilization}\n"),
                        Mock(stdout=processes)]
        return results

    def test_utilization_alone_does_not_block_idle_gpu(self):
        from run_week7 import check_gpu
        with patch("run_week7.subprocess.run", side_effect=self.samples(
            (11, 95, ""), (11, 13, ""),
        )), patch("run_week7.time.sleep"):
            check_gpu(1)

    def test_live_process_blocks_even_with_low_memory(self):
        from run_week7 import check_gpu
        with patch("run_week7.subprocess.run", side_effect=self.samples(
            (11, 0, "12345\n"),
        )), patch("run_week7.time.sleep") as sleep:
            with self.assertRaisesRegex(SystemExit, "occupied"):
                check_gpu(1)
            sleep.assert_not_called()

    def test_process_appearing_on_second_sample_blocks(self):
        from run_week7 import check_gpu
        with patch("run_week7.subprocess.run", side_effect=self.samples(
            (11, 0, ""), (11, 0, "12345\n"),
        )), patch("run_week7.time.sleep"):
            with self.assertRaisesRegex(SystemExit, "occupied"):
                check_gpu(1)

    def test_memory_cleanup_is_retried(self):
        from run_week7 import check_gpu
        with patch("run_week7.subprocess.run", side_effect=self.samples(
            (2048, 0, ""), (11, 13, ""), (11, 0, ""),
        )), patch("run_week7.time.sleep") as sleep:
            check_gpu(1)
            self.assertEqual(sleep.call_count, 2)

    def test_persistent_memory_usage_is_rejected(self):
        from run_week7 import check_gpu
        with patch("run_week7.subprocess.run", side_effect=self.samples(
            *((2048, 0, ""),) * 6,
        )), patch("run_week7.time.sleep") as sleep:
            with self.assertRaisesRegex(SystemExit, "did not become stably idle"):
                check_gpu(1)
            self.assertEqual(sleep.call_count, 5)

    def test_idle_samples_must_be_consecutive(self):
        from run_week7 import check_gpu
        with patch("run_week7.subprocess.run", side_effect=self.samples(
            (11, 0, ""), (2048, 0, ""), (11, 0, ""), (11, 0, ""),
        )), patch("run_week7.time.sleep") as sleep:
            check_gpu(1)
            self.assertEqual(sleep.call_count, 3)


class CheckpointRecoveryTests(unittest.TestCase):
    def test_failed_checkpoint_write_keeps_previous(self):
        from week7_train import commit_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last"
            path.mkdir()
            (path / "marker").write_text("previous")

            def fail(temp):
                (temp / "marker").write_text("partial")
                raise OSError("synthetic write failure")

            with self.assertRaisesRegex(OSError, "synthetic"):
                commit_checkpoint(path, fail)
            self.assertEqual((path / "marker").read_text(), "previous")
            self.assertFalse(path.with_name("last.tmp").exists())

    def test_full_disk_retries_without_losing_checkpoint(self):
        from week7_train import commit_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last"
            path.mkdir()
            (path / "marker").write_text("previous")
            attempts = []

            def populate(temp):
                attempts.append(1)
                self.assertEqual((path / "marker").read_text(), "previous")
                (temp / "marker").write_text("replacement")
                if len(attempts) == 1:
                    raise OSError(errno.ENOSPC, "No space left on device")

            with patch("week7_train.time.sleep") as sleep:
                commit_checkpoint(path, populate)
            self.assertEqual(len(attempts), 2)
            sleep.assert_called_once_with(30)
            self.assertEqual((path / "marker").read_text(), "replacement")
            self.assertFalse(path.with_name("last.previous").exists())

    def test_low_space_waits_before_writing(self):
        from week7_train import commit_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            with patch("week7_train.shutil.disk_usage", side_effect=[
                Mock(free=0), Mock(free=2 * 1024**3),
            ]), patch("week7_train.time.sleep") as sleep:
                path = Path(folder) / "last"
                commit_checkpoint(path, lambda temp: (temp / "marker").write_text("saved"))
            sleep.assert_called_once_with(30)
            self.assertEqual((path / "marker").read_text(), "saved")

    def make_recovery(self, folder, last_step=5000, existing_best=False):
        import torch
        from week7_common import object_hash
        destination = Path(folder) / "run"
        last = destination / "last"
        last.mkdir(parents=True)
        identity = {"protocol": "test", "variant": "text", "seed": 44,
                    "artifact_hash": "test-input", "smoke": False,
                    "config": {"train": {"max_steps": 60000, "patience": 4}}}
        state = {"signature": object_hash(identity), "global_step": last_step,
                 "best": 0.2, "bad": 0}
        torch.save(state, last / "trainer_state.pt")
        (last / "model.safetensors").write_bytes(b"last weights")
        history = [{"type": "valid", "step": 5000,
                    "metrics": {"all": {"ndcg@10": 0.2}}}]
        if existing_best:
            best = destination / "best"
            best.mkdir()
            torch.save({**state, "global_step": 5000}, best / "trainer_state.pt")
            (best / "model.safetensors").write_bytes(b"earlier best weights")
        return destination, Path(folder) / "best.json", identity, state, history

    def test_recover_first_best_from_matching_last(self):
        from week7_train import repair_best_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            args = self.make_recovery(folder)
            repair_best_checkpoint(*args)
            self.assertEqual((args[0] / "best/model.safetensors").read_bytes(), b"last weights")
            result = json.loads(args[1].read_text())
            self.assertEqual(result["best_step"], 5000)
            self.assertEqual(result["metrics"], args[4][0]["metrics"])
            self.assertFalse(result["finished"])

    def test_missing_earlier_best_cannot_be_invented_from_last(self):
        from week7_train import repair_best_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            args = self.make_recovery(folder, last_step=10000)
            with self.assertRaisesRegex(ValueError, "missing earlier best weights"):
                repair_best_checkpoint(*args)
            self.assertFalse(args[1].exists())

    def test_metadata_repair_preserves_existing_best_weights(self):
        from week7_train import repair_best_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            args = self.make_recovery(folder, last_step=10000, existing_best=True)
            repair_best_checkpoint(*args)
            self.assertEqual((args[0] / "best/model.safetensors").read_bytes(), b"earlier best weights")
            self.assertEqual(json.loads(args[1].read_text())["best_step"], 5000)


if __name__ == "__main__":
    unittest.main()
