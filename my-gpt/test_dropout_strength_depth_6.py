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
from test_depth_scaling import STAGE_22, STAGE_24
from test_width_128_dropout import (
    STAGE_21,
    create_tiny_stage_20_source,
    make_character_data,
    make_tiny_protocol,
)
from training import seed_everything


MODULE_PATH = Path(__file__).with_name("25_dropout_strength.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_25_dropout_strength_depth_6",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_25 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_25
SPEC.loader.exec_module(STAGE_25)


class Stage25DefaultsAndInitializationTests(unittest.TestCase):
    def test_defaults_pin_l6_hash_count_dropout_sites_and_shared_state(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_25.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 128)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 6)
        self.assertEqual(args.dropout, 0.15)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.initial_learning_rate, 1e-3)
        self.assertEqual(args.first_decay_step, 10_000)
        self.assertEqual(args.middle_learning_rate, 3e-4)
        self.assertEqual(args.second_decay_step, 13_000)
        self.assertEqual(args.final_learning_rate, 1e-4)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_24_depth_6_best_checkpoint.pt",
        )
        self.assertEqual(
            args.checkpoint_path.name,
            "stage_25_depth_6_dropout_015_best_checkpoint.pt",
        )
        self.assertEqual(
            STAGE_25._pin_for_default_path(args.control_checkpoint_path),
            (STAGE_25.DEFAULT_CONTROL_CHECKPOINT_SHA256, 10_500),
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
            STAGE_25.create_shared_initial_state(data, config, 4, 6)
        )
        self.assertEqual(parameter_count, 1_212_481)
        self.assertEqual(
            STAGE_25.expected_parameter_count(
                vocab_size=65,
                block_size=64,
                n_embd=128,
                n_layer=6,
            ),
            1_212_481,
        )
        self.assertEqual(
            initial_hash,
            "b4bf85310a542f248a51a8eb8e290bbb9274adcce6a2d9898f89cf2e78665a71",
        )
        self.assertEqual(initial_hash, STAGE_25.DEFAULT_INITIAL_STATE_SHA256)

        seed_everything(config.seed)
        control_model = STAGE_25._build_model(data, config, 4, 6, 0.10)
        control_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in control_model.state_dict().items()
        }
        seed_everything(config.seed)
        treatment_model = STAGE_25._build_model(data, config, 4, 6, 0.15)
        treatment_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in treatment_model.state_dict().items()
        }

        self.assertEqual(initial_state.keys(), control_state.keys())
        self.assertEqual(control_state.keys(), treatment_state.keys())
        for name in initial_state:
            torch.testing.assert_close(
                initial_state[name],
                control_state[name],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                control_state[name],
                treatment_state[name],
                rtol=0,
                atol=0,
            )

        control_dropouts = [
            module
            for module in control_model.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        treatment_dropouts = [
            module
            for module in treatment_model.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        self.assertEqual(len(control_dropouts), 12)
        self.assertEqual(len(treatment_dropouts), 12)
        self.assertTrue(all(module.p == 0.10 for module in control_dropouts))
        self.assertTrue(all(module.p == 0.15 for module in treatment_dropouts))
        self.assertEqual(
            len([name for name in initial_state if name.endswith(".tril")]),
            24,
        )


class Stage25ProtocolTests(unittest.TestCase):
    def test_custom_pin_and_dropout_depth_path_boundaries(self) -> None:
        custom_path = Path("custom_stage_24.pt")
        with self.assertRaisesRegex(ValueError, "custom Stage 24 control"):
            STAGE_25.resolve_control_pin(custom_path, None, None)
        with self.assertRaisesRegex(ValueError, "provided together"):
            STAGE_25.resolve_control_pin(custom_path, "a" * 64, None)
        with self.assertRaisesRegex(ValueError, "64 hex digits"):
            STAGE_25.resolve_control_pin(custom_path, "not-a-hash", 3)
        self.assertEqual(
            STAGE_25.resolve_control_pin(custom_path, "A" * 64, 3),
            ("a" * 64, 3),
        )

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            schedule = STAGE_25.LearningRateSchedule(
                initial_learning_rate=2e-2,
                first_decay_step=1,
                middle_learning_rate=7e-3,
                second_decay_step=2,
                final_learning_rate=2e-3,
            )
            config = TrainingConfig(
                batch_size=2,
                block_size=8,
                n_embd=16,
                learning_rate=schedule.initial_learning_rate,
                max_iters=3,
                eval_interval=3,
                eval_iters=1,
                seed=101,
            )
            control_path = directory / "stage24.pt"
            treatment = STAGE_25.DropoutSpec(
                "dropout-p015",
                0.15,
                directory / "stage25.pt",
            )
            self.assertEqual(
                STAGE_25.validate_protocol(
                    config,
                    2,
                    3,
                    treatment,
                    control_path,
                    schedule,
                ),
                ((0, 2), (2, 4), (4, 6), (6, 8)),
            )

            wrong_dropout = STAGE_25.DropoutSpec(
                "dropout-p010",
                0.10,
                directory / "wrong-dropout.pt",
            )
            with self.assertRaisesRegex(ValueError, "exactly 0.15"):
                STAGE_25.validate_protocol(
                    config,
                    2,
                    3,
                    wrong_dropout,
                    control_path,
                    schedule,
                )
            with self.assertRaisesRegex(ValueError, "at least 3"):
                STAGE_25.validate_protocol(
                    config,
                    2,
                    2,
                    treatment,
                    control_path,
                    schedule,
                )
            with self.assertRaisesRegex(ValueError, "canonical.*n_layer=6"):
                STAGE_25.validate_protocol(
                    config,
                    2,
                    3,
                    treatment,
                    STAGE_25.DEFAULT_CONTROL_CHECKPOINT_PATH,
                    schedule,
                )
            same_path = STAGE_25.DropoutSpec(
                "dropout-p015",
                0.15,
                control_path,
            )
            with self.assertRaisesRegex(ValueError, "paths must differ"):
                STAGE_25.validate_protocol(
                    config,
                    2,
                    3,
                    same_path,
                    control_path,
                    schedule,
                )


class Stage25WorkflowTests(unittest.TestCase):
    def test_stage24_control_to_stage25_checkpoint_fresh_panel_and_replacement(
        self,
    ) -> None:
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
                stage_22_treatment = STAGE_22.DropoutSpec(
                    "dropout-p010",
                    0.10,
                    directory / "stage22_dropout_010.pt",
                )
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
                    expected_control_sha256=STAGE_22.checkpoint_sha256(
                        stage_21_treatment.checkpoint_path
                    ),
                    expected_control_step=stage_21.treatment.best_step,
                )

                stage_24_control = STAGE_24.DepthSpec(
                    "depth-1",
                    1,
                    stage_22_treatment.checkpoint_path,
                )
                stage_24_treatment = STAGE_24.DepthSpec(
                    "depth-3",
                    3,
                    directory / "stage24_depth_3.pt",
                )
                stage_22_bytes = stage_24_control.checkpoint_path.read_bytes()
                stage_22_hash = STAGE_24.checkpoint_sha256(
                    stage_24_control.checkpoint_path
                )
                stage_24 = STAGE_24.run_experiment(
                    data,
                    config,
                    stage_24_control,
                    stage_24_treatment,
                    2,
                    torch.device("cpu"),
                    0,
                    schedule,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_control_sha256=stage_22_hash,
                    expected_control_step=stage_22.treatment.best_step,
                    benchmark=None,
                )

            self.assertEqual(
                stage_24_control.checkpoint_path.read_bytes(),
                stage_22_bytes,
            )
            stage_24_bytes = stage_24_treatment.checkpoint_path.read_bytes()
            stage_24_hash = STAGE_25.checkpoint_sha256(
                stage_24_treatment.checkpoint_path
            )
            treatment = STAGE_25.DropoutSpec(
                "dropout-p015",
                0.15,
                directory / "stage25_depth_3_dropout_015.pt",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                report = STAGE_25.run_experiment(
                    data,
                    config,
                    2,
                    3,
                    torch.device("cpu"),
                    0,
                    schedule,
                    treatment,
                    stage_24_treatment.checkpoint_path,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_control_sha256=stage_24_hash,
                    expected_control_step=stage_24.treatment.best_step,
                )

            self.assertEqual(
                stage_24_treatment.checkpoint_path.read_bytes(),
                stage_24_bytes,
            )
            self.assertTrue(treatment.checkpoint_path.is_file())
            self.assertEqual(report.control.reference.dropout, 0.10)
            self.assertEqual(report.treatment.spec.dropout, 0.15)
            self.assertEqual(
                report.initial_state_sha256,
                stage_24.initialization.treatment_state_sha256,
            )
            self.assertEqual(
                report.treatment_checkpoint_sha256,
                STAGE_25.checkpoint_sha256(treatment.checkpoint_path),
            )

            payload = STAGE_25.validate_stage_25_checkpoint(
                treatment.checkpoint_path,
                treatment,
                data,
                config,
                2,
                3,
                schedule,
                report.bands,
                initial_state_sha256=report.initial_state_sha256,
                parameter_count=report.parameter_count,
                training_batch_seed=202,
                training_rng_seed=303,
                control_checkpoint_sha256=stage_24_hash,
                control_checkpoint_step=stage_24.treatment.best_step,
            )
            experiment = payload["experiment"]
            self.assertEqual(experiment["stage"], 25)
            self.assertEqual(experiment["comparison_variable"], "residual_dropout")
            self.assertEqual(experiment["control_stage"], 24)
            self.assertEqual(experiment["control_branch"], "depth-3")
            self.assertEqual(experiment["control_residual_dropout"], 0.10)
            self.assertEqual(experiment["treatment_residual_dropout"], 0.15)
            self.assertEqual(experiment["fixed_transformer_depth"], 3)
            self.assertTrue(experiment["capacity_regularization_interaction"])
            self.assertTrue(experiment["identical_fixed_training_batches"])
            self.assertTrue(experiment["identical_complete_model_state"])
            self.assertTrue(experiment["same_training_rng_seed"])
            self.assertTrue(experiment["same_optimizer_configuration"])
            self.assertFalse(experiment["identical_dropout_masks"])
            self.assertTrue(experiment["reused_control"])
            self.assertEqual(payload["architecture"]["n_layer"], 3)
            self.assertEqual(payload["architecture"]["residual_dropout"], 0.15)
            self.assertEqual(
                payload["initialization"]["kind"],
                STAGE_25.INITIALIZATION_KIND,
            )
            self.assertEqual(payload["initialization"]["source_stage"], 24)
            self.assertEqual(payload["initialization"]["source_branch"], "depth-3")
            self.assertTrue(payload["initialization"]["identical_complete_model_state"])
            self.assertEqual(payload["evaluation_protocol"]["model_mode"], "eval")
            self.assertFalse(payload["evaluation_protocol"]["dropout_enabled"])

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
                precise = STAGE_25.run_precise_validation(
                    data,
                    config,
                    2,
                    3,
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
            self.assertEqual(
                stage_24_treatment.checkpoint_path.read_bytes(),
                stage_24_bytes,
            )
            self.assertEqual(treatment.checkpoint_path.read_bytes(), treatment_bytes)

            treatment.checkpoint_path.write_bytes(treatment_bytes + b"replaced")
            with self.assertRaisesRegex(
                RuntimeError,
                "changed after checkpoint selection",
            ):
                STAGE_25.run_precise_validation(
                    data,
                    config,
                    2,
                    3,
                    torch.device("cpu"),
                    schedule,
                    treatment,
                    report,
                    eval_iters=3,
                    seed=404,
                )

            treatment.checkpoint_path.write_bytes(treatment_bytes)
            stage_24_treatment.checkpoint_path.write_bytes(stage_24_bytes + b"replaced")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                STAGE_25.run_precise_validation(
                    data,
                    config,
                    2,
                    3,
                    torch.device("cpu"),
                    schedule,
                    treatment,
                    report,
                    eval_iters=3,
                    seed=404,
                )
            stage_24_treatment.checkpoint_path.write_bytes(stage_24_bytes)


if __name__ == "__main__":
    unittest.main()
