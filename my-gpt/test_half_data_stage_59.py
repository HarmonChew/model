from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location(
    "stage_59_under_test", Path(__file__).with_name("59_half_data.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage59Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full = STAGE.CharacterData.from_file(STAGE._STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
        cls.half = STAGE.halve_training_split(cls.full)

    def test_path_and_baseline(self) -> None:
        self.assertEqual(
            STAGE.replicate_path(6).name, "stage_59_c320_l6_p015_half_seed6_final_checkpoint.pt"
        )
        self.assertEqual(STAGE.WIDTH, STAGE._STAGE_53.WIDTH)
        self.assertEqual(STAGE.WEIGHT_DECAY, 0.01)

    def test_half_keeps_every_other_block(self) -> None:
        full, half, size = self.full.train_data, self.half.train_data, STAGE.BLOCK_CHARACTERS
        self.assertAlmostEqual(len(half) / len(full), 0.5, delta=0.02)
        self.assertTrue(torch.equal(half[:size], full[:size]))
        self.assertTrue(torch.equal(half[size : 2 * size], full[2 * size : 3 * size]))

    def test_vocabulary_and_val_unchanged(self) -> None:
        self.assertIs(self.half.vocabulary, self.full.vocabulary)
        self.assertTrue(torch.equal(self.half.val_data, self.full.val_data))

    def test_sampler_reads_only_the_halved_split(self) -> None:
        config = STAGE._STAGE_50.canonical_config()
        marked = STAGE.dataclasses.replace(
            self.half, train_data=torch.full_like(self.half.train_data, 7)
        )
        generator = torch.Generator().manual_seed(0)
        inputs, targets = STAGE._STAGE_50.get_batch(marked, "train", config, generator, torch.device("cpu"))
        self.assertTrue(bool((inputs == 7).all()) and bool((targets == 7).all()))


if __name__ == "__main__":
    unittest.main()
