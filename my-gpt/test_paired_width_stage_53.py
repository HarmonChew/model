from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "stage_53_under_test", Path(__file__).with_name("53_paired_width.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class LayoutTest(unittest.TestCase):
    def test_pairs_with_stage_51_seed_sets(self) -> None:
        self.assertEqual(STAGE.REPLICATES, STAGE._STAGE_51.REPLICATES)
        self.assertEqual(STAGE.replicate_path(2).name, "stage_53_c320_l6_p015_seed2_final_checkpoint.pt")

    def test_only_width_differs_from_the_champion(self) -> None:
        self.assertEqual(STAGE.WIDTH, 320)
        self.assertEqual(STAGE._STAGE_50.canonical_config().n_embd, 256)
        self.assertEqual(STAGE._STAGE_50.DEFAULT_DROPOUT, 0.15)
        self.assertEqual(STAGE.STAGE_46_TREATMENT, "stage_46_treatment_c320_p015_step_50000_final_checkpoint.pt")


if __name__ == "__main__":
    unittest.main()
