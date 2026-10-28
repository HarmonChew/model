from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "stage_54_under_test", Path(__file__).with_name("54_paired_width.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class LayoutTest(unittest.TestCase):
    def test_pairs_with_stage_53_champion(self) -> None:
        self.assertEqual(STAGE.replicate_path(2).name, "stage_54_c384_l6_p015_seed2_final_checkpoint.pt")
        self.assertEqual(STAGE._STAGE_53.replicate_path(2).name, "stage_53_c320_l6_p015_seed2_final_checkpoint.pt")
        self.assertEqual(STAGE.REPORT_PATH.name, "stage_54_report.json")

    def test_width_is_the_only_change(self) -> None:
        self.assertEqual(STAGE.WIDTH, 384)
        self.assertEqual(STAGE.WIDTH % STAGE._STAGE_44.N_HEAD, 0)
        self.assertEqual(STAGE._STAGE_50.DEFAULT_DROPOUT, 0.15)

    def test_model_constructs_on_cpu(self) -> None:
        data = STAGE.CharacterData.from_file(STAGE._STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
        STAGE.check_architecture(data)


if __name__ == "__main__":
    unittest.main()
