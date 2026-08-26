import argparse
import contextlib
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("26_context_length.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_26_context_length",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_26 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_26
SPEC.loader.exec_module(STAGE_26)


def make_character_data(
    *,
    vocabulary_size: int = 5,
    train_tokens: int = 400,
    val_tokens: int = 100,
) -> CharacterData:
    chars = "".join(chr(32 + index) for index in range(vocabulary_size))
    vocabulary = CharacterVocabulary.from_text(chars)
    token_ids = torch.arange(train_tokens + val_tokens) % vocabulary_size
    return CharacterData(
        vocabulary=vocabulary,
        train_data=token_ids[:train_tokens].to(torch.long),
        val_data=token_ids[train_tokens:].to(torch.long),
        num_characters=train_tokens + val_tokens,
    )


class Stage26ProtocolTests(unittest.TestCase):
    def test_defaults_and_preflight_define_the_stage_26_protocol(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_26.parse_args()

        self.assertEqual(args.control_batch_size, 32)
        self.assertEqual(args.control_block_size, 64)
        self.assertEqual(args.treatment_batch_size, 16)
        self.assertEqual(args.treatment_block_size, 128)
        self.assertEqual(args.n_embd, 128)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 6)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(STAGE_26.DEFAULT_DROPOUT, 0.10)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_26_context_64_best_checkpoint.pt",
        )
        self.assertEqual(
            args.treatment_checkpoint_path.name,
            "stage_26_context_128_best_checkpoint.pt",
        )

        control = STAGE_26.ContextSpec(
            "T=64 control",
            args.control_batch_size,
            args.control_block_size,
            args.control_checkpoint_path,
        )
        treatment = STAGE_26.ContextSpec(
            "T=128 treatment",
            args.treatment_batch_size,
            args.treatment_block_size,
            args.treatment_checkpoint_path,
        )
        self.assertEqual(
            STAGE_26.validate_context_protocol(control, treatment),
            2_048,
        )
        self.assertEqual(2_048 * args.max_iters, 36_864_000)

        bad_count = argparse.Namespace(precise_eval_iters=1, seed=1_337)
        with self.assertRaisesRegex(ValueError, "0 or at least 2"):
            STAGE_26._STAGE_17._validate_context_stage_options(
                bad_count,
                training_batch_seed=1_338,
                training_rng_seed=1_340,
                precise_eval_seed=1_344,
            )
        reused_seed = argparse.Namespace(precise_eval_iters=2, seed=1_337)
        with self.assertRaisesRegex(ValueError, "fixed-panel seed"):
            STAGE_26._STAGE_17._validate_context_stage_options(
                reused_seed,
                training_batch_seed=1_338,
                training_rng_seed=1_340,
                precise_eval_seed=1_339,
            )

    def test_long_segments_are_split_without_changing_target_order(self) -> None:
        source = torch.arange(50, dtype=torch.long)
        paired = STAGE_26.build_paired_context_batch(
            source,
            torch.tensor([3, 20]),
            control_block_size=4,
            treatment_block_size=8,
        )
        expected_long = torch.tensor(
            [
                [3, 4, 5, 6, 7, 8, 9, 10],
                [20, 21, 22, 23, 24, 25, 26, 27],
            ]
        )
        torch.testing.assert_close(paired.treatment_inputs, expected_long)
        torch.testing.assert_close(
            paired.control_inputs.reshape_as(expected_long),
            expected_long,
        )
        torch.testing.assert_close(
            paired.control_targets.reshape_as(paired.treatment_targets),
            paired.treatment_targets,
        )

        generator = torch.Generator().manual_seed(91)
        control_logits = torch.randn(4, 2, 5, generator=generator)
        control_targets = torch.tensor(
            [[0, 1], [2, 3], [1, 2], [3, 4]],
            dtype=torch.long,
        )
        treatment_logits = control_logits.reshape(2, 4, 5)
        treatment_targets = control_targets.reshape(2, 4)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            control = STAGE_26.ContextSpec("control", 4, 2, directory / "c.pt")
            treatment = STAGE_26.ContextSpec("treatment", 2, 4, directory / "t.pt")
            control_losses = STAGE_26.split_cross_entropy(
                control_logits,
                control_targets,
                control,
                control=control,
                treatment=treatment,
            )
            treatment_losses = STAGE_26.split_cross_entropy(
                treatment_logits,
                treatment_targets,
                treatment,
                control=control,
                treatment=treatment,
            )
        expected_first = F.cross_entropy(
            control_logits.reshape(2, 2, 2, 5)[:, 0].reshape(-1, 5),
            control_targets.reshape(2, 2, 2)[:, 0].reshape(-1),
        )
        expected_second = F.cross_entropy(
            control_logits.reshape(2, 2, 2, 5)[:, 1].reshape(-1, 5),
            control_targets.reshape(2, 2, 2)[:, 1].reshape(-1),
        )
        for losses in (control_losses, treatment_losses):
            torch.testing.assert_close(losses[1], expected_first)
            torch.testing.assert_close(losses[2], expected_second)
            torch.testing.assert_close(losses[0], (losses[1] + losses[2]) / 2)


class Stage26InitializationTests(unittest.TestCase):
    def test_canonical_initialization_is_pinned_and_overlap_matched(self) -> None:
        data = CharacterData.from_file(Path(__file__).parent / "data" / "input.txt")
        common = {
            "n_embd": 128,
            "learning_rate": 1e-3,
            "max_iters": 2,
            "eval_interval": 1,
            "eval_iters": 1,
            "seed": 1_337,
        }
        control_config = TrainingConfig(batch_size=32, block_size=64, **common)
        treatment_config = TrainingConfig(batch_size=16, block_size=128, **common)
        initialization = STAGE_26.create_paired_initial_states(
            data,
            control_config,
            treatment_config,
            4,
            6,
        )
        self.assertEqual(
            STAGE_26.fingerprint_data(data),
            STAGE_26.DEFAULT_DATA_FINGERPRINT,
        )

        self.assertEqual(
            initialization.control_state_sha256,
            STAGE_26.DEFAULT_CONTROL_INITIAL_STATE_SHA256,
        )
        self.assertEqual(
            initialization.common_state_sha256,
            STAGE_26.DEFAULT_CONTROL_INITIAL_STATE_SHA256,
        )
        self.assertEqual(
            initialization.treatment_state_sha256,
            STAGE_26.DEFAULT_TREATMENT_INITIAL_STATE_SHA256,
        )
        self.assertEqual(
            initialization.shared_parameter_initialization_sha256,
            STAGE_26.DEFAULT_SHARED_PARAMETER_INITIALIZATION_SHA256,
        )
        self.assertEqual(
            initialization.extra_position_rows_sha256,
            STAGE_26.DEFAULT_EXTRA_POSITION_ROWS_SHA256,
        )
        self.assertEqual(initialization.control_parameter_count, 1_212_481)
        self.assertEqual(initialization.treatment_parameter_count, 1_220_673)
        self.assertEqual(
            initialization.treatment_parameter_count
            - initialization.control_parameter_count,
            64 * 128,
        )

        control_position = initialization.control["position_embedding_table.weight"]
        treatment_position = initialization.treatment["position_embedding_table.weight"]
        torch.testing.assert_close(
            treatment_position[:64],
            control_position,
            rtol=0,
            atol=0,
        )
        self.assertNotEqual(control_position.data_ptr(), treatment_position.data_ptr())
        self.assertEqual(
            STAGE_26.tensor_sha256(treatment_position[64:]),
            STAGE_26.DEFAULT_EXTRA_POSITION_ROWS_SHA256,
        )
        for name, control_tensor in initialization.control.items():
            treatment_tensor = initialization.treatment[name]
            with self.subTest(state=name):
                if control_tensor.shape == treatment_tensor.shape:
                    torch.testing.assert_close(
                        treatment_tensor,
                        control_tensor,
                        rtol=0,
                        atol=0,
                    )
                elif name.endswith(".tril"):
                    torch.testing.assert_close(
                        treatment_tensor[:64, :64],
                        control_tensor,
                        rtol=0,
                        atol=0,
                    )

        control_model = STAGE_26._build_model(
            data, control_config, 4, 6, torch.device("cpu")
        )
        treatment_model = STAGE_26._build_model(
            data, treatment_config, 4, 6, torch.device("cpu")
        )
        control_model.load_state_dict(initialization.control, strict=True)
        treatment_model.load_state_dict(initialization.treatment, strict=True)
        self.assertEqual(
            sum(
                isinstance(module, torch.nn.Dropout)
                for module in control_model.modules()
            ),
            12,
        )
        self.assertEqual(
            sum(
                isinstance(module, torch.nn.Dropout)
                for module in treatment_model.modules()
            ),
            12,
        )
        self.assertEqual(
            sum(name.endswith(".tril") for name in initialization.control),
            24,
        )
        self.assertEqual(
            STAGE_26.inspect_attention_shape(
                control_model,
                STAGE_26.ContextSpec("c", 32, 64, Path("c.pt")),
                4,
                torch.device("cpu"),
            ),
            (32, 4, 64, 64),
        )
        self.assertEqual(
            STAGE_26.inspect_attention_shape(
                treatment_model,
                STAGE_26.ContextSpec("t", 16, 128, Path("t.pt")),
                4,
                torch.device("cpu"),
            ),
            (16, 4, 128, 128),
        )
        difference = STAGE_26.verify_initial_first_half_equivalence(
            control_model,
            treatment_model,
            data,
            STAGE_26.ContextSpec("c", 32, 64, Path("c.pt")),
            STAGE_26.ContextSpec("t", 16, 128, Path("t.pt")),
            torch.device("cpu"),
            seed=1_343,
        )
        self.assertLessEqual(difference, 1e-6)


class Stage26StatisticsTests(unittest.TestCase):
    @staticmethod
    def _result(name: str, values: tuple[object, ...]) -> object:
        return STAGE_26.PreciseBranchResult(
            name=name,
            checkpoint_path=Path(f"{name}.pt"),
            checkpoint_sha256="0" * 64,
            checkpoint_step=0,
            fixed_panel_loss=0.0,
            statistics=STAGE_26._split_statistics(values),
            batch_losses=values,
        )

    def test_paired_student_t_interval_uses_batch_differences(self) -> None:
        baseline_values = tuple(
            STAGE_26.SplitLosses(value, value + 1, value + 2)
            for value in (10.0, 20.0, 30.0)
        )
        candidate_values = tuple(
            STAGE_26.SplitLosses(
                baseline.overall + delta,
                baseline.first_half + 2 * delta,
                baseline.second_half - delta,
            )
            for baseline, delta in zip(
                baseline_values,
                (-1.0, 0.0, 1.0),
                strict=True,
            )
        )
        delta = STAGE_26._paired_delta(
            self._result("candidate", candidate_values),
            self._result("baseline", baseline_values),
        )
        critical = STAGE_26._paired_t_95_critical_value(3)
        self.assertAlmostEqual(critical, 4.302652730)
        self.assertAlmostEqual(delta.critical_value, critical)
        self.assertAlmostEqual(delta.mean_delta.overall, 0.0)
        self.assertAlmostEqual(delta.standard_error.overall, 1 / math.sqrt(3))
        self.assertAlmostEqual(
            delta.confidence_low.overall,
            -critical / math.sqrt(3),
        )
        self.assertAlmostEqual(
            delta.confidence_high.overall,
            critical / math.sqrt(3),
        )
        self.assertAlmostEqual(delta.standard_error.first_half, 2 / math.sqrt(3))
        with self.assertRaisesRegex(ValueError, "at least two"):
            STAGE_26._paired_delta(
                self._result("candidate", candidate_values[:1]),
                self._result("baseline", baseline_values[:1]),
            )


class Stage26TinyWorkflowTests(unittest.TestCase):
    def test_training_checkpoint_and_fresh_panel_are_fully_paired(self) -> None:
        data = make_character_data()
        device = torch.device("cpu")
        schedule = STAGE_26.LearningRateSchedule(
            initial_learning_rate=2e-2,
            first_decay_step=2,
            middle_learning_rate=7e-3,
            second_decay_step=3,
            final_learning_rate=2e-3,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            control = STAGE_26.ContextSpec(
                "T=4 control", 2, 4, directory / "control.pt"
            )
            treatment = STAGE_26.ContextSpec(
                "T=8 treatment", 1, 8, directory / "treatment.pt"
            )
            common = {
                "n_embd": 8,
                "learning_rate": schedule.initial_learning_rate,
                "max_iters": 4,
                "eval_interval": 2,
                "eval_iters": 1,
                "seed": 101,
            }
            control_config = TrainingConfig(
                batch_size=control.batch_size,
                block_size=control.block_size,
                **common,
            )
            treatment_config = TrainingConfig(
                batch_size=treatment.batch_size,
                block_size=treatment.block_size,
                **common,
            )

            observed: list[tuple[str, torch.Tensor, torch.Tensor]] = []
            original_get_batch = STAGE_26._STAGE_17.get_paired_context_batch

            def checked_get_batch(*args: object, **kwargs: object) -> object:
                batch = original_get_batch(*args, **kwargs)
                split = args[1]
                assert isinstance(split, str)
                torch.testing.assert_close(
                    batch.control_targets.reshape_as(batch.treatment_targets),
                    batch.treatment_targets,
                    rtol=0,
                    atol=0,
                )
                observed.append(
                    (
                        split,
                        batch.starts.clone(),
                        batch.treatment_targets.clone(),
                    )
                )
                return batch

            with (
                mock.patch.object(
                    STAGE_26._STAGE_17,
                    "get_paired_context_batch",
                    side_effect=checked_get_batch,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = STAGE_26.run_experiment(
                    data,
                    control_config,
                    treatment_config,
                    control,
                    treatment,
                    2,
                    1,
                    device,
                    0,
                    schedule,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    benchmark=None,
                )

            # The first call is the pre-training first-half equivalence audit.
            branch_observed = observed[1:]
            self.assertEqual(len(branch_observed) % 2, 0)
            branch_call_count = len(branch_observed) // 2
            for first, second in zip(
                branch_observed[:branch_call_count],
                branch_observed[branch_call_count:],
                strict=True,
            ):
                self.assertEqual(first[0], second[0])
                torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
                torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
            self.assertEqual(
                report.control.final_batch_generator_sha256,
                report.treatment.final_batch_generator_sha256,
            )

            payloads = []
            for branch, config, initial_hash in (
                (control, control_config, report.initialization.control_state_sha256),
                (
                    treatment,
                    treatment_config,
                    report.initialization.treatment_state_sha256,
                ),
            ):
                payload = STAGE_26.validate_stage_26_checkpoint(
                    branch.checkpoint_path,
                    data,
                    config,
                    branch,
                    control,
                    treatment,
                    2,
                    1,
                    schedule,
                    initialization_seed=101,
                    branch_initial_state_sha256=initial_hash,
                    common_state_sha256=report.initialization.common_state_sha256,
                    extra_position_rows_sha256=(
                        report.initialization.extra_position_rows_sha256
                    ),
                    shared_parameter_initialization_sha256=(
                        report.initialization.shared_parameter_initialization_sha256
                    ),
                    training_batch_seed=202,
                    training_rng_seed=303,
                )
                self.assertEqual(payload["experiment"]["stage"], 26)
                self.assertEqual(payload["architecture"]["residual_dropout"], 0.10)
                self.assertIn("best_training_losses", payload)
                self.assertTrue(payload["experiment"]["paired_target_characters"])
                self.assertTrue(payload["experiment"]["new_control_training_run"])
                self.assertFalse(
                    payload["experiment"]["reused_stage_24_trained_checkpoint"]
                )
                self.assertFalse(
                    payload["experiment"]["identical_dropout_masks_guaranteed"]
                )
                self.assertEqual(
                    payload["evaluation_protocol"]["first_half_panel_positions"],
                    (0, 3),
                )
                self.assertEqual(
                    payload["evaluation_protocol"]["second_half_panel_positions"],
                    (4, 7),
                )
                self.assertIsInstance(payload["rng_state"]["cpu"], torch.Tensor)
                payloads.append(payload)

            checkpoint_hashes = (
                STAGE_26.checkpoint_sha256(control.checkpoint_path),
                STAGE_26.checkpoint_sha256(treatment.checkpoint_path),
            )
            self.assertEqual(
                checkpoint_hashes,
                (
                    report.control_checkpoint_sha256,
                    report.treatment_checkpoint_sha256,
                ),
            )
            precise_observed: list[tuple[torch.Tensor, torch.Tensor]] = []

            def checked_precise_batch(*args: object, **kwargs: object) -> object:
                batch = original_get_batch(*args, **kwargs)
                precise_observed.append(
                    (batch.starts.clone(), batch.treatment_targets.clone())
                )
                return batch

            torch.manual_seed(987)
            rng_before = torch.get_rng_state().clone()
            with (
                mock.patch.object(
                    STAGE_26._STAGE_17,
                    "get_paired_context_batch",
                    side_effect=checked_precise_batch,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                precise = STAGE_26.run_precise_validation(
                    data,
                    control_config,
                    treatment_config,
                    control,
                    treatment,
                    2,
                    1,
                    device,
                    schedule,
                    report.initialization,
                    initialization_seed=101,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    eval_iters=3,
                    seed=404,
                    expected_control_checkpoint_sha256=(
                        report.control_checkpoint_sha256
                    ),
                    expected_treatment_checkpoint_sha256=(
                        report.treatment_checkpoint_sha256
                    ),
                )
            torch.testing.assert_close(
                torch.get_rng_state(), rng_before, rtol=0, atol=0
            )
            self.assertEqual(len(precise_observed), 6)
            for first, second in zip(
                precise_observed[:3], precise_observed[3:], strict=True
            ):
                torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
                torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
            self.assertEqual(precise.eval_iters, 3)
            critical = STAGE_26._paired_t_95_critical_value(3)
            self.assertAlmostEqual(
                precise.delta.confidence_low.overall,
                precise.delta.mean_delta.overall
                - critical * precise.delta.standard_error.overall,
            )
            self.assertEqual(
                checkpoint_hashes,
                (
                    STAGE_26.checkpoint_sha256(control.checkpoint_path),
                    STAGE_26.checkpoint_sha256(treatment.checkpoint_path),
                ),
            )
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                STAGE_26.run_precise_validation(
                    data,
                    control_config,
                    treatment_config,
                    control,
                    treatment,
                    2,
                    1,
                    device,
                    schedule,
                    report.initialization,
                    initialization_seed=101,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    eval_iters=2,
                    seed=404,
                    expected_control_checkpoint_sha256="0" * 64,
                    expected_treatment_checkpoint_sha256=(
                        report.treatment_checkpoint_sha256
                    ),
                )

            malformed_path = directory / "malformed.pt"
            malformed = dict(payloads[0])
            malformed.pop("rng_state")
            torch.save(malformed, malformed_path)
            malformed_control = STAGE_26.ContextSpec(
                control.name,
                control.batch_size,
                control.block_size,
                malformed_path,
            )
            with self.assertRaisesRegex(ValueError, "rng_state"):
                STAGE_26.validate_stage_26_checkpoint(
                    malformed_path,
                    data,
                    control_config,
                    malformed_control,
                    control,
                    treatment,
                    2,
                    1,
                    schedule,
                    initialization_seed=101,
                    branch_initial_state_sha256=(
                        report.initialization.control_state_sha256
                    ),
                    common_state_sha256=report.initialization.common_state_sha256,
                    extra_position_rows_sha256=(
                        report.initialization.extra_position_rows_sha256
                    ),
                    shared_parameter_initialization_sha256=(
                        report.initialization.shared_parameter_initialization_sha256
                    ),
                    training_batch_seed=202,
                    training_rng_seed=303,
                )


if __name__ == "__main__":
    unittest.main()
