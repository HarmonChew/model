from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location("stage_62_under_test", Path(__file__).with_name("62_attention_maps.py"))
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)

MODEL_SPEC = importlib.util.spec_from_file_location("stage_14_for_stage_62_test", Path(__file__).with_name("14_dropout.py"))
assert MODEL_SPEC is not None and MODEL_SPEC.loader is not None
STAGE_14 = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = STAGE_14
MODEL_SPEC.loader.exec_module(STAGE_14)


def one_hot_attention(keys: list[int]) -> torch.Tensor:
    """Attention of shape (1, 1, 1, T, T) where query t looks only at keys[t]."""
    weights = torch.zeros(1, 1, 1, len(keys), len(keys))
    for query, key in enumerate(keys):
        weights[0, 0, 0, query, key] = 1.0
    return weights


class Stage62Test(unittest.TestCase):
    def test_forward_with_attention_matches_model(self) -> None:
        torch.manual_seed(0)
        model = STAGE_14.GPTLanguageModel(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=2).eval()
        idx = torch.randint(0, 10, (3, 8))
        logits, weights = STAGE.forward_with_attention(model, idx)
        self.assertTrue(torch.allclose(logits, model(idx)[0], atol=1e-6))
        self.assertEqual(tuple(weights.shape), (3, 2, 2, 8, 8))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3, 2, 2, 8)))

    def test_last_position_of(self) -> None:
        idx = torch.tensor([[5, 0, 5, 5, 0, 5]])  # 0 marks a space
        self.assertEqual(STAGE.last_position_of(idx, [0]).tolist(), [[-1, -1, 1, 1, 1, 4]])

    def test_scores_find_a_previous_character_head(self) -> None:
        length = STAGE.MIN_QUERY + 4
        idx = torch.ones(1, length, dtype=torch.long)
        weights = one_hot_attention([max(t - 1, 0) for t in range(length)])
        scores = STAGE.pick_best_offset(STAGE.head_scores(weights, idx, [0], [0]))
        self.assertEqual(int(scores["best offset"]), 1)
        self.assertAlmostEqual(float(scores["offset"]), 1.0)
        self.assertAlmostEqual(float(scores["distance"]), 1.0)

    def test_scores_find_a_word_start_head(self) -> None:
        length = STAGE.MIN_QUERY + 4
        idx = torch.ones(1, length, dtype=torch.long)
        idx[0, 10] = 0  # one space; every later query looks at it
        weights = one_hot_attention([10 if t > 10 else t for t in range(length)])
        scores = STAGE.head_scores(weights, idx, [0], [9])
        self.assertAlmostEqual(float(scores["word start"]), 1.0)
        self.assertAlmostEqual(float(scores["line start"]), 0.0)

    def test_label(self) -> None:
        self.assertEqual(STAGE.label({"offset": 0.9, "best offset": 2, "word start": 0.1, "line start": 0.0}), "offset 2")
        self.assertEqual(STAGE.label({"offset": 0.1, "best offset": 1, "word start": 0.4, "line start": 0.0}), "word start")
        self.assertEqual(STAGE.label({"offset": 0.1, "best offset": 1, "word start": 0.2, "line start": 0.1}), "spread")


if __name__ == "__main__":
    unittest.main()
