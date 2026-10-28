from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location(
    "stage_55_under_test", Path(__file__).with_name("55_paired_dropout.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage55Test(unittest.TestCase):
    def test_paths_name_width_and_dropout(self) -> None:
        self.assertEqual(STAGE.replicate_path(0.10, 6).name, "stage_55_c320_l6_p010_seed6_final_checkpoint.pt")
        self.assertEqual(STAGE.replicate_path(0.20, 0).name, "stage_55_c320_l6_p020_seed0_final_checkpoint.pt")

    def test_baseline_is_the_stage_53_champion(self) -> None:
        self.assertEqual(STAGE.WIDTH, 320)
        self.assertEqual(STAGE._STAGE_53.replicate_path(1).name, "stage_53_c320_l6_p015_seed1_final_checkpoint.pt")

    def test_set_dropout_leaves_weights_untouched(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.15))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        STAGE.set_dropout(model, 0.20)
        self.assertEqual(model[1].p, 0.20)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))


if __name__ == "__main__":
    unittest.main()
