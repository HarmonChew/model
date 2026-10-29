from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location("stage_64_under_test", Path(__file__).with_name("64_head_ablation.py"))
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)

MODEL_SPEC = importlib.util.spec_from_file_location("stage_14_for_stage_64_test", Path(__file__).with_name("14_dropout.py"))
assert MODEL_SPEC is not None and MODEL_SPEC.loader is not None
STAGE_14 = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = STAGE_14
MODEL_SPEC.loader.exec_module(STAGE_14)


class Stage64Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = STAGE_14.GPTLanguageModel(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=2).eval()
        self.idx = torch.randint(0, 10, (3, 8))

    def test_switched_off_head_outputs_zeros(self) -> None:
        head = self.model.blocks[1].sa.heads[0]
        x = torch.randn(3, 8, 16)
        with STAGE.heads_off(self.model, [(1, 0)]):
            self.assertTrue(torch.equal(head(x), torch.zeros(3, 8, 8)))
        self.assertFalse(torch.equal(head(x), torch.zeros(3, 8, 8)))

    def test_switch_off_changes_output_and_restores_it(self) -> None:
        before = self.model(self.idx)[0]
        with STAGE.heads_off(self.model, [(0, 1)]):
            during = self.model(self.idx)[0]
        after = self.model(self.idx)[0]
        self.assertFalse(torch.allclose(before, during))
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
