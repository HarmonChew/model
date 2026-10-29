from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location("stage_61_under_test", Path(__file__).with_name("61_token_model.py"))
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage61Test(unittest.TestCase):
    def test_bits_per_character_matches_hand_calculation(self) -> None:
        # Two windows of 4 targets at ln 2 nats each = 8 bits over 16 characters.
        self.assertAlmostEqual(STAGE.bits_per_character([math.log(2)] * 2, 4, 16), 0.5)
        # Characters: one character per target, so bpc is loss / ln 2.
        self.assertAlmostEqual(STAGE.bits_per_character([1.0, 1.0], 4, 8), 1 / math.log(2))

    def test_scored_characters_counts_target_symbols(self) -> None:
        tokens = torch.tensor([0, 1, 2, 1, 2, 0])
        lengths = torch.tensor([1, 2, 3])
        # block 2 -> 2 windows, targets are tokens[1:5] = 1,2,1,2 -> 2+3+2+3.
        self.assertEqual(STAGE.scored_characters(tokens, 2, lengths), 10)

    def test_sampler_and_builder_accept_token_data(self) -> None:
        config = STAGE._STAGE_50.canonical_config()
        data = STAGE.TokenData(
            vocabulary=STAGE.TokenVocabulary(STAGE.VOCAB_SIZE),
            train_data=torch.full((5_000,), 1000, dtype=torch.long),
            val_data=torch.zeros(1_000, dtype=torch.long),
            symbol_lengths=torch.ones(STAGE.VOCAB_SIZE, dtype=torch.long),
        )
        inputs, targets = STAGE._STAGE_50.get_batch(data, "train", config, torch.Generator().manual_seed(0), torch.device("cpu"))
        self.assertTrue(bool((inputs == 1000).all()) and bool((targets == 1000).all()))
        model = STAGE._STAGE_51.build_endpoint(data, STAGE.architecture())
        self.assertEqual(model.lm_head.out_features, STAGE.VOCAB_SIZE)

    def test_paths(self) -> None:
        self.assertEqual(STAGE.replicate_path(6).name, "stage_61_c320_l6_p015_bpe1024_seed6_final_checkpoint.pt")
        self.assertEqual(STAGE.replicate_path(6, "best").name, "stage_61_c320_l6_p015_bpe1024_seed6_best_checkpoint.pt")


if __name__ == "__main__":
    unittest.main()
