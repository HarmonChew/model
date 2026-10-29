from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location("stage_65_under_test", Path(__file__).with_name("65_logit_lens.py"))
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)

MODEL_SPEC = importlib.util.spec_from_file_location("stage_14_for_stage_65_test", Path(__file__).with_name("14_dropout.py"))
assert MODEL_SPEC is not None and MODEL_SPEC.loader is not None
STAGE_14 = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = STAGE_14
MODEL_SPEC.loader.exec_module(STAGE_14)


class Stage65Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = STAGE_14.GPTLanguageModel(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=3).eval()
        self.idx = torch.randint(0, 10, (3, 8))

    def test_one_reading_for_the_embeddings_and_each_layer(self) -> None:
        readings = STAGE.lens(self.model, self.idx)
        self.assertEqual(len(readings), 4)
        for logits in readings:
            self.assertEqual(logits.shape, (3, 8, 10))

    def test_last_reading_equals_the_model_output(self) -> None:
        self.assertTrue(torch.equal(STAGE.lens(self.model, self.idx)[-1], self.model(self.idx)[0]))

    def test_each_layer_changes_the_reading(self) -> None:
        readings = STAGE.lens(self.model, self.idx)
        for before, after in zip(readings, readings[1:]):
            self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
