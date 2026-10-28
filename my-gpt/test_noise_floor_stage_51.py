from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


SPEC = importlib.util.spec_from_file_location(
    "stage_51_under_test", Path(__file__).with_name("51_noise_floor.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)

CPU = torch.device("cpu")
FAKE_DATA = types.SimpleNamespace(vocabulary=types.SimpleNamespace(size=98))


def architecture(width: int, contexts: tuple[int | None, ...]) -> dict:
    return {"n_embd": width, "n_head": 4, "block_size": 256, "layer_contexts": contexts}


class ToyModel(torch.nn.Module):
    """Logits depend on the token only, so every window loss is easy to check."""

    def __init__(self, vocab: int = 5) -> None:
        super().__init__()
        self.table = torch.nn.Embedding(vocab, vocab)

    def forward(self, idx, targets=None):
        return self.table(idx), None


class FullSplitScoringTest(unittest.TestCase):
    def test_scores_every_target_once_in_windows(self) -> None:
        torch.manual_seed(0)
        tokens = torch.randint(0, 5, (43,))
        model = ToyModel()
        losses = STAGE.score_full_split(model, tokens, 4, CPU, batch_windows=3)

        self.assertEqual(len(losses), 10)  # (43 - 1) // 4
        with torch.no_grad():
            logits = model.table(tokens[:40]).double()
            expected = F.cross_entropy(logits, tokens[1:41]).item()
        self.assertAlmostEqual(sum(losses) / len(losses), expected, places=10)

    def test_batching_does_not_change_scores(self) -> None:
        torch.manual_seed(1)
        tokens = torch.randint(0, 5, (100,))
        model = ToyModel()
        one = STAGE.score_full_split(model, tokens, 8, CPU, batch_windows=1)
        many = STAGE.score_full_split(model, tokens, 8, CPU, batch_windows=64)
        self.assertEqual(one, many)

    def test_restores_training_mode(self) -> None:
        model = ToyModel().train()
        STAGE.score_full_split(model, torch.arange(20) % 5, 4, CPU)
        self.assertTrue(model.training)

    def test_rejects_split_shorter_than_two_windows(self) -> None:
        with self.assertRaises(ValueError):
            STAGE.score_full_split(ToyModel(), torch.arange(8) % 5, 4, CPU)


class StatisticsTest(unittest.TestCase):
    def test_paired_delta_of_a_constant_shift(self) -> None:
        baseline = [1.0, 2.0, 3.0, 4.0]
        result = STAGE.paired_delta([x + 0.5 for x in baseline], baseline)
        self.assertAlmostEqual(result["delta"], 0.5)
        self.assertAlmostEqual(result["se"], 0.0)
        self.assertAlmostEqual(result["low"], 0.5)

    def test_paired_delta_requires_matching_windows(self) -> None:
        with self.assertRaises(ValueError):
            STAGE.paired_delta([1.0, 2.0], [1.0])

    def test_seed_spread_threshold(self) -> None:
        spread = STAGE.seed_spread([1.0, 1.002, 1.004])
        self.assertAlmostEqual(spread["sd"], 0.002)
        self.assertAlmostEqual(spread["two_run_threshold"], 2 * math.sqrt(2) * 0.002)
        with self.assertRaises(ValueError):
            STAGE.seed_spread([1.0])


class ProtocolTest(unittest.TestCase):
    def test_replicate_zero_uses_the_lineage_seeds(self) -> None:
        self.assertEqual(
            STAGE.replicate_seeds(0),
            {"init_seed": 1337, "batch_seed": 1338, "rng_seed": 1340},
        )

    def test_replicate_seeds_are_distinct(self) -> None:
        seen = [value for r in STAGE.REPLICATES for value in STAGE.replicate_seeds(r).values()]
        self.assertEqual(len(seen), len(set(seen)))

    def test_rebuilds_recorded_architectures_at_recorded_sizes(self) -> None:
        cases = (
            (256, STAGE._STAGE_50.DEFAULT_CONTROL_LAYER_CONTEXTS, 4_850_274),
            (256, STAGE._STAGE_50.DEFAULT_TREATMENT_LAYER_CONTEXTS, 6_428_258),
            (320, STAGE._STAGE_50.DEFAULT_CONTROL_LAYER_CONTEXTS, 7_537_378),
        )
        for width, contexts, expected in cases:
            model = STAGE.build_endpoint(FAKE_DATA, architecture(width, contexts))
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_historical_calls_name_known_endpoints(self) -> None:
        for _, _, candidate, baseline in STAGE.HISTORICAL_CALLS:
            self.assertIn(candidate, STAGE.STORED_ENDPOINTS)
            self.assertIn(baseline, STAGE.STORED_ENDPOINTS)


if __name__ == "__main__":
    unittest.main()
