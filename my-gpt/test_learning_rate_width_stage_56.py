from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "stage_56_under_test", Path(__file__).with_name("56_learning_rate_width.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage56Test(unittest.TestCase):
    def test_path_names_width_and_learning_rate(self) -> None:
        self.assertEqual(
            STAGE.replicate_path(6).name, "stage_56_c384_l6_p015_lr5e4_seed6_final_checkpoint.pt"
        )

    def test_baselines_are_stage_54_and_53(self) -> None:
        self.assertEqual(STAGE.WIDTH, 384)
        self.assertEqual(STAGE._STAGE_54.replicate_path(1).name, "stage_54_c384_l6_p015_seed1_final_checkpoint.pt")
        self.assertEqual(STAGE._STAGE_53.replicate_path(1).name, "stage_53_c320_l6_p015_seed1_final_checkpoint.pt")

    def test_schedule_halves_only_the_first_phase(self) -> None:
        old, new = STAGE._STAGE_50.make_schedule(), STAGE.make_schedule()
        self.assertEqual(old.for_update(0), 1e-3)
        self.assertEqual(new.for_update(0), 5e-4)
        self.assertEqual(new.for_update(9_999), 5e-4)
        for update in (10_000, 13_000, 18_000, 49_999):
            self.assertEqual(new.for_update(update), old.for_update(update))
        self.assertEqual(new.first_decay_step, old.first_decay_step)

    def test_config_learning_rate_matches_schedule(self) -> None:
        config = STAGE._STAGE_50.replace(
            STAGE._STAGE_50.canonical_config(), learning_rate=STAGE.LOW_LEARNING_RATE
        )
        self.assertEqual(config.learning_rate, STAGE.make_schedule().for_update(0))


if __name__ == "__main__":
    unittest.main()
