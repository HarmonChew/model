from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location("stage_63_under_test", Path(__file__).with_name("63_induction_heads.py"))
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


def one_hot_attention(keys: list[int]) -> torch.Tensor:
    """Attention of shape (T, T) where query t looks only at keys[t]."""
    weights = torch.zeros(len(keys), len(keys))
    for query, key in enumerate(keys):
        weights[query, key] = 1.0
    return weights


class Stage63Test(unittest.TestCase):
    def test_sequences_repeat_exactly(self) -> None:
        sequences = STAGE.make_sequences([3, 5, 7], count=4, half=10, seed=0)
        self.assertEqual(tuple(sequences.shape), (4, 20))
        self.assertTrue(torch.equal(sequences[:, :10], sequences[:, 10:]))
        self.assertTrue(set(sequences.flatten().tolist()) <= {3, 5, 7})

    def test_scores_tell_induction_and_duplicate_heads_apart(self) -> None:
        half = 6
        length = 2 * half - 1  # the model reads all but the last character
        induction_head = one_hot_attention([0 if t < half else t - (half - 1) for t in range(length)])
        duplicate_head = one_hot_attention([0 if t < half else t - half for t in range(length)])
        # (batch 2, layer 1, head 2, T, T): head 1 does induction, head 2 duplicates
        weights = torch.stack([induction_head, duplicate_head]).expand(2, 1, 2, length, length)
        induction, duplicate = STAGE.repeat_scores(weights, half)
        self.assertEqual(tuple(induction.shape), (1, 2))
        self.assertTrue(torch.allclose(induction, torch.tensor([[1.0, 0.0]])))
        self.assertTrue(torch.allclose(duplicate, torch.tensor([[0.0, 1.0]])))
        self.assertEqual(STAGE.label(1.0, 0.0), "induction")

    def test_copy_losses_split_the_halves(self) -> None:
        half = 3
        sequences = torch.tensor([[0, 1, 2, 0, 1, 2]])
        logits = torch.zeros(1, 5, 4)
        logits[0, 3:, :] = torch.tensor([0.0, 0.0, 0.0, -1e9])  # second half: 3 choices left
        loss, first, second = STAGE.copy_losses(logits, sequences, half)
        self.assertEqual(len(loss), 5)
        self.assertAlmostEqual(first, math.log(4), places=5)
        self.assertAlmostEqual(second, math.log(3), places=5)


if __name__ == "__main__":
    unittest.main()
