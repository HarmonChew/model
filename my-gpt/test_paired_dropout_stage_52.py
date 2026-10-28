from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location(
    "stage_52_under_test", Path(__file__).with_name("52_paired_dropout.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class SetDropoutTest(unittest.TestCase):
    def test_sets_every_dropout_module(self) -> None:
        model = torch.nn.Sequential(torch.nn.Dropout(0.15), torch.nn.Linear(2, 2), torch.nn.Dropout(0.15))
        self.assertEqual(STAGE.set_dropout(model, 0.10), 2)
        self.assertEqual({m.p for m in model if isinstance(m, torch.nn.Dropout)}, {0.10})

    def test_leaves_weights_untouched(self) -> None:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.15))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        STAGE.set_dropout(model, 0.10)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_refuses_model_without_dropout(self) -> None:
        with self.assertRaises(RuntimeError):
            STAGE.set_dropout(torch.nn.Linear(2, 2), 0.10)


class LayoutTest(unittest.TestCase):
    def test_pairs_with_stage_51_seed_sets(self) -> None:
        self.assertEqual(STAGE.REPLICATES, STAGE._STAGE_51.REPLICATES)
        self.assertEqual(STAGE.replicate_path(2).name, "stage_52_c256_l6_p010_seed2_final_checkpoint.pt")


if __name__ == "__main__":
    unittest.main()
