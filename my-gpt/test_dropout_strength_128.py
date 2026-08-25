from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from config import TrainingConfig
from test_width_128_dropout import (
    STAGE_21,
    create_tiny_stage_20_source,
    make_character_data,
    make_tiny_protocol,
)
from training import seed_everything


MODULE_PATH = Path(__file__).with_name("22_dropout_strength.py")
SPEC = importlib.util.spec_from_file_location("stage_22_dropout_strength", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE_22 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_22
SPEC.loader.exec_module(STAGE_22)


class Stage22DefaultsAndInitializationTests(unittest.TestCase):
    def test_defaults_pin_probability_artifacts_and_initialization(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_22.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 128)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.dropout, 0.10)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_21_dropout_005_best_checkpoint.pt",
        )
        self.assertEqual(
            args.checkpoint_path.name,
            "stage_22_dropout_010_best_checkpoint.pt",
        )
        self.assertEqual(
            STAGE_22._pin_for_default_path(args.control_checkpoint_path),
            (STAGE_22.DEFAULT_CONTROL_CHECKPOINT_SHA256, 13_500),
        )

        data = make_character_data(vocabulary_size=65)
        config = TrainingConfig(
            batch_size=32,
            block_size=64,
            n_embd=128,
            learning_rate=1e-3,
            max_iters=18_000,
            eval_interval=500,
            eval_iters=1,
            seed=1_337,
        )
        initial_state, initial_hash, parameter_count = (
            STAGE_22.create_shared_initial_state(data, config, 4, 4)
        )
        self.assertEqual(parameter_count, 816_705)
        self.assertEqual(initial_hash, STAGE_22.DEFAULT_INITIAL_STATE_SHA256)

        seed_everything(config.seed)
        treatment_model = STAGE_22._build_model(data, config, 4, 4, 0.10)
        treatment_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in treatment_model.state_dict().items()
        }
        self.assertEqual(initial_state.keys(), treatment_state.keys())
        for name in initial_state:
            torch.testing.assert_close(
                initial_state[name],
                treatment_state[name],
                rtol=0,
                atol=0,
            )
        dropout_modules = [
            module
            for module in treatment_model.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        self.assertEqual(len(dropout_modules), 8)
        self.assertTrue(all(module.p == 0.10 for module in dropout_modules))

    def test_probability_is_a_hard_protocol_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = TrainingConfig(
                batch_size=2,
                block_size=8,
                n_embd=16,
                learning_rate=2e-2,
                max_iters=3,
                eval_interval=3,
                eval_iters=1,
                seed=101,
            )
            schedule = STAGE_22.LearningRateSchedule(
                initial_learning_rate=2e-2,
                first_decay_step=1,
                middle_learning_rate=7e-3,
                second_decay_step=2,
                final_learning_rate=2e-3,
            )
            wrong = STAGE_22.DropoutSpec(
                "dropout-p009",
                0.09,
                directory / "treatment.pt",
            )
            with self.assertRaisesRegex(ValueError, "exactly 0.10"):
                STAGE_22.validate_protocol(
                    config,
                    2,
                    1,
                    wrong,
                    directory / "control.pt",
                    schedule,
                )


class Stage22WorkflowTests(unittest.TestCase):
    def test_read_only_stage_21_control_and_stage_22_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (
                data,
                narrow_config,
                config,
                narrow,
                wide,
                stage_21_treatment,
                schedule,
            ) = make_tiny_protocol(directory)
            stage_20 = create_tiny_stage_20_source(
                data,
                narrow_config,
                config,
                narrow,
                wide,
                schedule,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                stage_21 = STAGE_21.run_experiment(
                    data,
                    config,
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    schedule,
                    stage_21_treatment,
                    wide.checkpoint_path,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_control_sha256=STAGE_21.checkpoint_sha256(
                        wide.checkpoint_path
                    ),
                    expected_control_step=stage_20.treatment.best_step,
                )
            control_bytes = stage_21_treatment.checkpoint_path.read_bytes()
            control_hash = STAGE_22.checkpoint_sha256(
                stage_21_treatment.checkpoint_path
            )
            treatment = STAGE_22.DropoutSpec(
                "dropout-p010",
                0.10,
                directory / "stage22_dropout_010.pt",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                report = STAGE_22.run_experiment(
                    data,
                    config,
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    schedule,
                    treatment,
                    stage_21_treatment.checkpoint_path,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_control_sha256=control_hash,
                    expected_control_step=stage_21.treatment.best_step,
                )

            self.assertEqual(
                stage_21_treatment.checkpoint_path.read_bytes(),
                control_bytes,
            )
            self.assertTrue(treatment.checkpoint_path.is_file())
            self.assertEqual(report.control.reference.dropout, 0.05)
            self.assertEqual(report.treatment.spec.dropout, 0.10)
            self.assertEqual(
                report.initial_state_sha256,
                stage_21.initial_state_sha256,
            )
            payload = STAGE_22.validate_stage_22_checkpoint(
                treatment.checkpoint_path,
                treatment,
                data,
                config,
                2,
                1,
                schedule,
                report.bands,
                initial_state_sha256=report.initial_state_sha256,
                parameter_count=report.parameter_count,
                training_batch_seed=202,
                training_rng_seed=303,
                control_checkpoint_sha256=control_hash,
                control_checkpoint_step=stage_21.treatment.best_step,
            )
            experiment = payload["experiment"]
            self.assertEqual(experiment["stage"], 22)
            self.assertEqual(experiment["control_stage"], 21)
            self.assertEqual(experiment["control_branch"], "dropout-p005")
            self.assertEqual(experiment["control_residual_dropout"], 0.05)
            self.assertEqual(experiment["treatment_residual_dropout"], 0.10)
            self.assertEqual(
                payload["initialization"]["kind"],
                STAGE_22.INITIALIZATION_KIND,
            )

            treatment_bytes = treatment.checkpoint_path.read_bytes()
            precise = STAGE_22.run_precise_validation(
                data,
                config,
                2,
                1,
                torch.device("cpu"),
                schedule,
                treatment,
                report,
                eval_iters=3,
                seed=404,
            )
            self.assertEqual(
                [result.name for result in precise.results],
                ["p=0.05", "p=0.10"],
            )
            self.assertEqual(precise.delta.candidate, "p=0.10")
            self.assertEqual(precise.delta.baseline, "p=0.05")
            self.assertEqual(treatment.checkpoint_path.read_bytes(), treatment_bytes)
            self.assertEqual(
                stage_21_treatment.checkpoint_path.read_bytes(),
                control_bytes,
            )


if __name__ == "__main__":
    unittest.main()
