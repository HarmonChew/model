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


STAGE_22_PATH = Path(__file__).with_name("22_dropout_strength.py")
STAGE_22_SPEC = importlib.util.spec_from_file_location(
    "stage_22_dropout_strength_for_stage_23_tests",
    STAGE_22_PATH,
)
assert STAGE_22_SPEC is not None and STAGE_22_SPEC.loader is not None
STAGE_22 = importlib.util.module_from_spec(STAGE_22_SPEC)
sys.modules[STAGE_22_SPEC.name] = STAGE_22
STAGE_22_SPEC.loader.exec_module(STAGE_22)

MODULE_PATH = Path(__file__).with_name("23_dropout_strength.py")
SPEC = importlib.util.spec_from_file_location("stage_23_dropout_strength", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE_23 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_23
SPEC.loader.exec_module(STAGE_23)


class Stage23DefaultsAndInitializationTests(unittest.TestCase):
    def test_defaults_pin_probability_artifacts_and_initialization(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_23.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 128)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.dropout, 0.15)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertIsNone(args.control_checkpoint_sha256)
        self.assertIsNone(args.control_checkpoint_step)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_22_dropout_010_best_checkpoint.pt",
        )
        self.assertEqual(
            args.checkpoint_path.name,
            "stage_23_dropout_015_best_checkpoint.pt",
        )
        self.assertEqual(
            STAGE_23._pin_for_default_path(args.control_checkpoint_path),
            (STAGE_23.DEFAULT_CONTROL_CHECKPOINT_SHA256, 15_500),
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
            STAGE_23.create_shared_initial_state(data, config, 4, 4)
        )
        self.assertEqual(parameter_count, 816_705)
        self.assertEqual(initial_hash, STAGE_23.DEFAULT_INITIAL_STATE_SHA256)

        seed_everything(config.seed)
        treatment_model = STAGE_23._build_model(data, config, 4, 4, 0.15)
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
        self.assertTrue(all(module.p == 0.15 for module in dropout_modules))

    def test_custom_control_requires_an_explicit_hash_and_step(self) -> None:
        custom_path = Path("custom_stage_22.pt")
        with self.assertRaisesRegex(ValueError, "custom Stage 22 control"):
            STAGE_23.resolve_control_pin(custom_path, None, None)
        with self.assertRaisesRegex(ValueError, "provided together"):
            STAGE_23.resolve_control_pin(custom_path, "a" * 64, None)
        with self.assertRaisesRegex(ValueError, "64 hex digits"):
            STAGE_23.resolve_control_pin(custom_path, "not-a-hash", 3)
        self.assertEqual(
            STAGE_23.resolve_control_pin(custom_path, "A" * 64, 3),
            ("a" * 64, 3),
        )

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
            schedule = STAGE_23.LearningRateSchedule(
                initial_learning_rate=2e-2,
                first_decay_step=1,
                middle_learning_rate=7e-3,
                second_decay_step=2,
                final_learning_rate=2e-3,
            )
            wrong = STAGE_23.DropoutSpec(
                "dropout-p014",
                0.14,
                directory / "treatment.pt",
            )
            with self.assertRaisesRegex(ValueError, "exactly 0.15"):
                STAGE_23.validate_protocol(
                    config,
                    2,
                    1,
                    wrong,
                    directory / "control.pt",
                    schedule,
                )

    def test_fixed_training_panel_disables_dropout_and_restores_mode(self) -> None:
        data = make_character_data()
        config = TrainingConfig(
            batch_size=2,
            block_size=8,
            n_embd=16,
            learning_rate=2e-2,
            max_iters=3,
            eval_interval=3,
            eval_iters=2,
            seed=101,
        )
        model = STAGE_23._build_model(data, config, 2, 1, 0.15)
        model.train()
        observations: list[tuple[bool, bool]] = []
        handles = [
            module.register_forward_pre_hook(
                lambda observed, _inputs: observations.append(
                    (observed.training, torch.is_grad_enabled())
                )
            )
            for module in model.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        rng_before = torch.random.get_rng_state().clone()
        try:
            first = STAGE_23.evaluate_on_fixed_batches(
                model,
                data,
                config,
                torch.device("cpu"),
                STAGE_23.make_position_bands(config.block_size),
            )
            second = STAGE_23.evaluate_on_fixed_batches(
                model,
                data,
                config,
                torch.device("cpu"),
                STAGE_23.make_position_bands(config.block_size),
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(first, second)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertEqual(len(observations), 16)
        self.assertTrue(
            all(
                not dropout_training and not gradients_enabled
                for dropout_training, gradients_enabled in observations
            )
        )


class Stage23WorkflowTests(unittest.TestCase):
    def test_read_only_stage_22_control_and_stage_23_checkpoint(self) -> None:
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
            stage_21_hash = STAGE_22.checkpoint_sha256(
                stage_21_treatment.checkpoint_path
            )
            stage_22_treatment = STAGE_22.DropoutSpec(
                "dropout-p010",
                0.10,
                directory / "stage22_dropout_010.pt",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                stage_22 = STAGE_22.run_experiment(
                    data,
                    config,
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    schedule,
                    stage_22_treatment,
                    stage_21_treatment.checkpoint_path,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_control_sha256=stage_21_hash,
                    expected_control_step=stage_21.treatment.best_step,
                )

            control_bytes = stage_22_treatment.checkpoint_path.read_bytes()
            control_hash = STAGE_23.checkpoint_sha256(
                stage_22_treatment.checkpoint_path
            )
            treatment = STAGE_23.DropoutSpec(
                "dropout-p015",
                0.15,
                directory / "stage23_dropout_015.pt",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report = STAGE_23.run_experiment(
                    data,
                    config,
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    schedule,
                    treatment,
                    stage_22_treatment.checkpoint_path,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_control_sha256=control_hash,
                    expected_control_step=stage_22.treatment.best_step,
                )

            self.assertEqual(
                stage_22_treatment.checkpoint_path.read_bytes(),
                control_bytes,
            )
            self.assertIn("eval-train", output.getvalue())
            self.assertIn("dropout disabled", output.getvalue())
            self.assertTrue(treatment.checkpoint_path.is_file())
            self.assertEqual(report.control.reference.dropout, 0.10)
            self.assertEqual(report.treatment.spec.dropout, 0.15)
            self.assertEqual(
                report.treatment_checkpoint_sha256,
                STAGE_23.checkpoint_sha256(treatment.checkpoint_path),
            )
            self.assertEqual(
                report.initial_state_sha256,
                stage_22.initial_state_sha256,
            )
            payload = STAGE_23.validate_stage_23_checkpoint(
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
                control_checkpoint_step=stage_22.treatment.best_step,
            )
            experiment = payload["experiment"]
            self.assertEqual(experiment["stage"], 23)
            self.assertEqual(experiment["control_stage"], 22)
            self.assertEqual(experiment["control_branch"], "dropout-p010")
            self.assertEqual(experiment["control_residual_dropout"], 0.10)
            self.assertEqual(experiment["treatment_residual_dropout"], 0.15)
            self.assertTrue(experiment["identical_fixed_training_batches"])
            self.assertEqual(
                payload["initialization"]["kind"],
                STAGE_23.INITIALIZATION_KIND,
            )
            evaluation = payload["evaluation_protocol"]
            self.assertEqual(evaluation["fixed_panel_seed"], config.seed + 2)
            self.assertEqual(evaluation["model_mode"], "eval")
            self.assertFalse(evaluation["dropout_enabled"])
            self.assertFalse(evaluation["gradients_enabled"])
            self.assertEqual(evaluation["split_order"], ("train", "val"))
            self.assertEqual(
                evaluation["reported_train_loss"],
                "fixed_training_panel_eval_mode",
            )

            treatment_bytes = treatment.checkpoint_path.read_bytes()
            captured_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            original_get_batch = type(data).get_batch

            def capture_get_batch(
                source: object,
                split: str,
                **kwargs: object,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                inputs, targets = original_get_batch(source, split, **kwargs)
                if split == "val":
                    captured_batches.append(
                        (
                            inputs.detach().cpu().clone(),
                            targets.detach().cpu().clone(),
                        )
                    )
                return inputs, targets

            with mock.patch.object(type(data), "get_batch", new=capture_get_batch):
                precise = STAGE_23.run_precise_validation(
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
                ["p=0.10", "p=0.15"],
            )
            self.assertEqual(precise.delta.candidate, "p=0.15")
            self.assertEqual(precise.delta.baseline, "p=0.10")
            self.assertEqual(len(captured_batches), 6)
            for control_batch, treatment_batch in zip(
                captured_batches[:3],
                captured_batches[3:],
                strict=True,
            ):
                torch.testing.assert_close(
                    control_batch[0],
                    treatment_batch[0],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    control_batch[1],
                    treatment_batch[1],
                    rtol=0,
                    atol=0,
                )
            self.assertEqual(treatment.checkpoint_path.read_bytes(), treatment_bytes)
            self.assertEqual(
                stage_22_treatment.checkpoint_path.read_bytes(),
                control_bytes,
            )
            treatment.checkpoint_path.write_bytes(treatment_bytes + b"replaced")
            with self.assertRaisesRegex(
                RuntimeError,
                "changed after checkpoint selection",
            ):
                STAGE_23.run_precise_validation(
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


if __name__ == "__main__":
    unittest.main()
