from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "stage_58_under_test", Path(__file__).with_name("58_weight_decay.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage58Test(unittest.TestCase):
    def test_path_names_width_and_weight_decay(self) -> None:
        self.assertEqual(
            STAGE.replicate_path(6).name, "stage_58_c320_l6_p015_wd01_seed6_final_checkpoint.pt"
        )

    def test_baseline_is_stage_53_at_width_320(self) -> None:
        self.assertEqual(STAGE.WIDTH, 320)
        self.assertEqual(STAGE._STAGE_53.WIDTH, 320)
        self.assertEqual(STAGE._STAGE_53.replicate_path(1).name, "stage_53_c320_l6_p015_seed1_final_checkpoint.pt")

    def test_schedule_is_champion_schedule(self) -> None:
        old, new = STAGE._STAGE_50.make_schedule(), STAGE.make_schedule()
        for update in (0, 9_999, 10_000, 13_000, 18_000, 49_999):
            self.assertEqual(new.for_update(update), old.for_update(update))
        self.assertEqual(new.for_update(0), STAGE._STAGE_50.canonical_config().learning_rate)

    def test_weight_decay(self) -> None:
        self.assertEqual(STAGE.WEIGHT_DECAY, 0.1)


if __name__ == "__main__":
    unittest.main()
