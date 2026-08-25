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


MODULE_PATH = Path(__file__).with_name("17_context_length.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_17_context_length",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_17 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_17
SPEC.loader.exec_module(STAGE_17)


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


class Stage17DefaultsAndBatchingTests(unittest.TestCase):
    def test_defaults_define_the_controlled_context_protocol(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_17.parse_args()

        self.assertEqual(args.control_batch_size, 32)
        self.assertEqual(args.control_block_size, 64)
        self.assertEqual(args.treatment_batch_size, 16)
        self.assertEqual(args.treatment_block_size, 128)
        self.assertEqual(args.n_embd, 64)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.initial_learning_rate, 1e-3)
        self.assertEqual(args.first_decay_step, 10_000)
        self.assertEqual(args.middle_learning_rate, 3e-4)
        self.assertEqual(args.second_decay_step, 13_000)
        self.assertEqual(args.final_learning_rate, 1e-4)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertIsNone(args.precise_eval_seed)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_17_control_best_checkpoint.pt",
        )
        self.assertEqual(
            args.treatment_checkpoint_path.name,
            "stage_17_context_128_best_checkpoint.pt",
        )

        control = STAGE_17.ContextSpec(
            "T=64 control",
            args.control_batch_size,
            args.control_block_size,
            args.control_checkpoint_path,
        )
        treatment = STAGE_17.ContextSpec(
            "T=128 treatment",
            args.treatment_batch_size,
            args.treatment_block_size,
            args.treatment_checkpoint_path,
        )
        targets_per_update = STAGE_17.validate_context_protocol(
            control,
            treatment,
        )
        self.assertEqual(targets_per_update, 2_048)
        self.assertEqual(control.tokens_per_update, treatment.tokens_per_update)
        self.assertEqual(targets_per_update * args.max_iters, 36_864_000)
        self.assertEqual(STAGE_17.DEFAULT_DROPOUT, 0.0)

    def test_batches_are_interleaved_exactly_and_consume_one_start_draw(
        self,
    ) -> None:
        source = torch.arange(50, dtype=torch.long)
        starts = torch.tensor([3, 20])
        paired = STAGE_17.build_paired_context_batch(
            source,
            starts,
            control_block_size=4,
            treatment_block_size=8,
        )

        expected_treatment_inputs = torch.tensor(
            [
                [3, 4, 5, 6, 7, 8, 9, 10],
                [20, 21, 22, 23, 24, 25, 26, 27],
            ]
        )
        expected_control_inputs = torch.tensor(
            [
                [3, 4, 5, 6],
                [7, 8, 9, 10],
                [20, 21, 22, 23],
                [24, 25, 26, 27],
            ]
        )
        torch.testing.assert_close(
            paired.treatment_inputs,
            expected_treatment_inputs,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            paired.control_inputs,
            expected_control_inputs,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            paired.treatment_targets,
            expected_treatment_inputs + 1,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            paired.control_targets,
            expected_control_inputs + 1,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            paired.control_inputs.reshape_as(paired.treatment_inputs),
            paired.treatment_inputs,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            paired.control_targets.reshape_as(paired.treatment_targets),
            paired.treatment_targets,
            rtol=0,
            atol=0,
        )

        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            control = STAGE_17.ContextSpec(
                "control",
                4,
                4,
                directory / "control.pt",
            )
            treatment = STAGE_17.ContextSpec(
                "treatment",
                2,
                8,
                directory / "treatment.pt",
            )
            actual_generator = torch.Generator().manual_seed(91)
            expected_generator = torch.Generator().manual_seed(91)
            actual = STAGE_17.get_paired_context_batch(
                data,
                "train",
                control,
                treatment,
                actual_generator,
            )
            expected_starts = torch.randint(
                0,
                len(data.train_data) - treatment.block_size,
                (treatment.batch_size,),
                generator=expected_generator,
            )

        torch.testing.assert_close(
            actual.starts,
            expected_starts,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_generator.get_state(),
            expected_generator.get_state(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual.control_targets.reshape_as(actual.treatment_targets),
            actual.treatment_targets,
            rtol=0,
            atol=0,
        )


class Stage17ArchitectureAndLossTests(unittest.TestCase):
    def test_overlap_initialization_counts_logits_and_attention_shapes(
        self,
    ) -> None:
        data = make_character_data(vocabulary_size=65, val_tokens=200)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            control = STAGE_17.ContextSpec(
                "T=64 control",
                32,
                64,
                directory / "control.pt",
            )
            treatment = STAGE_17.ContextSpec(
                "T=128 treatment",
                16,
                128,
                directory / "treatment.pt",
            )
            common_config = {
                "n_embd": 64,
                "learning_rate": 1e-3,
                "max_iters": 18_000,
                "eval_interval": 500,
                "eval_iters": 1,
                "seed": 1_337,
            }
            control_config = TrainingConfig(
                batch_size=control.batch_size,
                block_size=control.block_size,
                **common_config,
            )
            treatment_config = TrainingConfig(
                batch_size=treatment.batch_size,
                block_size=treatment.block_size,
                **common_config,
            )
            initialization = STAGE_17.create_paired_initial_states(
                data,
                control_config,
                treatment_config,
                4,
                4,
            )

            self.assertEqual(initialization.control_parameter_count, 211_777)
            self.assertEqual(initialization.treatment_parameter_count, 215_873)
            self.assertEqual(
                initialization.treatment_parameter_count
                - initialization.control_parameter_count,
                4_096,
            )
            self.assertNotEqual(
                initialization.control_state_sha256,
                initialization.treatment_state_sha256,
            )
            self.assertEqual(
                initialization.common_state_sha256,
                initialization.control_state_sha256,
            )

            control_position = initialization.control["position_embedding_table.weight"]
            treatment_position = initialization.treatment[
                "position_embedding_table.weight"
            ]
            torch.testing.assert_close(
                treatment_position[:64],
                control_position,
                rtol=0,
                atol=0,
            )
            self.assertNotEqual(
                treatment_position.data_ptr(),
                control_position.data_ptr(),
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
                        self.assertNotEqual(
                            treatment_tensor.data_ptr(),
                            control_tensor.data_ptr(),
                        )
                    elif name.endswith(".tril"):
                        torch.testing.assert_close(
                            treatment_tensor[:64, :64],
                            control_tensor,
                            rtol=0,
                            atol=0,
                        )

            control_model = STAGE_17._build_model(
                data,
                control_config,
                4,
                4,
                torch.device("cpu"),
            )
            treatment_model = STAGE_17._build_model(
                data,
                treatment_config,
                4,
                4,
                torch.device("cpu"),
            )
            control_model.load_state_dict(initialization.control, strict=True)
            treatment_model.load_state_dict(
                initialization.treatment,
                strict=True,
            )
            difference = STAGE_17.verify_initial_first_half_equivalence(
                control_model,
                treatment_model,
                data,
                control,
                treatment,
                torch.device("cpu"),
                seed=1_343,
            )
            self.assertLessEqual(difference, 1e-6)
            self.assertEqual(
                STAGE_17.inspect_attention_shape(
                    control_model,
                    control,
                    4,
                    torch.device("cpu"),
                ),
                (32, 4, 64, 64),
            )
            self.assertEqual(
                STAGE_17.inspect_attention_shape(
                    treatment_model,
                    treatment,
                    4,
                    torch.device("cpu"),
                ),
                (16, 4, 128, 128),
            )
            self.assertEqual(math.prod((32, 4, 64, 64)), 524_288)
            self.assertEqual(math.prod((16, 4, 128, 128)), 1_048_576)
            self.assertEqual(32 * 64 * 64, 16 * 128 * 64)

    def test_split_cross_entropy_uses_example_pairs_for_the_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            control = STAGE_17.ContextSpec(
                "control",
                4,
                2,
                directory / "control.pt",
            )
            treatment = STAGE_17.ContextSpec(
                "treatment",
                2,
                4,
                directory / "treatment.pt",
            )
            generator = torch.Generator().manual_seed(17)
            control_logits = torch.randn(4, 2, 3, generator=generator)
            control_targets = torch.tensor(
                [[0, 1], [2, 0], [1, 2], [0, 2]],
                dtype=torch.long,
            )
            treatment_logits = control_logits.reshape(2, 4, 3)
            treatment_targets = control_targets.reshape(2, 4)

            control_losses = STAGE_17.split_cross_entropy(
                control_logits,
                control_targets,
                control,
                control=control,
                treatment=treatment,
            )
            treatment_losses = STAGE_17.split_cross_entropy(
                treatment_logits,
                treatment_targets,
                treatment,
                control=control,
                treatment=treatment,
            )

        reshaped_logits = control_logits.reshape(2, 2, 2, 3)
        reshaped_targets = control_targets.reshape(2, 2, 2)
        expected_overall = F.cross_entropy(
            control_logits.reshape(-1, 3),
            control_targets.reshape(-1),
        )
        expected_first = F.cross_entropy(
            reshaped_logits[:, 0].reshape(-1, 3),
            reshaped_targets[:, 0].reshape(-1),
        )
        expected_second = F.cross_entropy(
            reshaped_logits[:, 1].reshape(-1, 3),
            reshaped_targets[:, 1].reshape(-1),
        )
        for losses in (control_losses, treatment_losses):
            torch.testing.assert_close(losses[0], expected_overall)
            torch.testing.assert_close(losses[1], expected_first)
            torch.testing.assert_close(losses[2], expected_second)
            torch.testing.assert_close(
                losses[0],
                (losses[1] + losses[2]) / 2,
            )
        self.assertNotAlmostEqual(
            expected_first.item(),
            expected_second.item(),
        )


class Stage17TinyWorkflowTests(unittest.TestCase):
    def test_scheduled_training_checkpoints_validator_and_precise_pairing(
        self,
    ) -> None:
        data = make_character_data()
        device = torch.device("cpu")
        schedule = STAGE_17.LearningRateSchedule(
            initial_learning_rate=2e-2,
            first_decay_step=2,
            middle_learning_rate=7e-3,
            second_decay_step=4,
            final_learning_rate=2e-3,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            control = STAGE_17.ContextSpec(
                "T=4 control",
                2,
                4,
                directory / "control.pt",
            )
            treatment = STAGE_17.ContextSpec(
                "T=8 treatment",
                1,
                8,
                directory / "treatment.pt",
            )
            common_config = {
                "n_embd": 8,
                "learning_rate": schedule.initial_learning_rate,
                "max_iters": 6,
                "eval_interval": 6,
                "eval_iters": 1,
                "seed": 101,
            }
            control_config = TrainingConfig(
                batch_size=control.batch_size,
                block_size=control.block_size,
                **common_config,
            )
            treatment_config = TrainingConfig(
                batch_size=treatment.batch_size,
                block_size=treatment.block_size,
                **common_config,
            )
            recorded_rates: list[float] = []
            original_set_rate = STAGE_17.set_optimizer_learning_rate

            def record_rate(
                optimizer: torch.optim.Optimizer,
                learning_rate: float,
            ) -> None:
                recorded_rates.append(learning_rate)
                original_set_rate(optimizer, learning_rate)

            with (
                mock.patch.object(
                    STAGE_17,
                    "set_optimizer_learning_rate",
                    side_effect=record_rate,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = STAGE_17.run_experiment(
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

            expected_branch_rates = [
                2e-2,
                2e-2,
                7e-3,
                7e-3,
                2e-3,
                2e-3,
            ]
            self.assertEqual(
                recorded_rates,
                expected_branch_rates + expected_branch_rates,
            )
            self.assertEqual(report.targets_per_update, 8)
            self.assertEqual(
                [record.step for record in report.control.training.history],
                [0, 2, 4, 6],
            )
            self.assertEqual(
                [record.step for record in report.treatment.training.history],
                [0, 2, 4, 6],
            )
            self.assertEqual(
                report.control.final_batch_generator_sha256,
                report.treatment.final_batch_generator_sha256,
            )
            self.assertEqual(report.control.attention_shape, (2, 2, 4, 4))
            self.assertEqual(report.treatment.attention_shape, (1, 2, 8, 8))

            payloads = []
            for spec, config, initial_hash in (
                (
                    control,
                    control_config,
                    report.initialization.control_state_sha256,
                ),
                (
                    treatment,
                    treatment_config,
                    report.initialization.treatment_state_sha256,
                ),
            ):
                with self.subTest(branch=spec.name):
                    self.assertTrue(spec.checkpoint_path.is_file())
                    self.assertFalse(
                        spec.checkpoint_path.with_name(
                            f".{spec.checkpoint_path.name}.tmp"
                        ).exists()
                    )
                    payload = STAGE_17.validate_stage_17_checkpoint(
                        spec.checkpoint_path,
                        data,
                        config,
                        spec,
                        control,
                        treatment,
                        2,
                        1,
                        schedule,
                        initialization_seed=101,
                        branch_initial_state_sha256=initial_hash,
                        common_state_sha256=(report.initialization.common_state_sha256),
                        extra_position_rows_sha256=(
                            report.initialization.extra_position_rows_sha256
                        ),
                        training_batch_seed=202,
                        training_rng_seed=303,
                    )
                    self.assertEqual(payload["experiment"]["stage"], 17)
                    self.assertTrue(payload["experiment"]["paired_target_characters"])
                    self.assertEqual(
                        payload["experiment"]["targets_per_update"],
                        8,
                    )
                    self.assertFalse(
                        payload["experiment"]["identical_full_initialization"]
                    )
                    self.assertEqual(
                        payload["architecture"]["residual_dropout"],
                        0.0,
                    )
                    payloads.append(payload)

            checkpoint_hashes_before = (
                STAGE_17.checkpoint_sha256(control.checkpoint_path),
                STAGE_17.checkpoint_sha256(treatment.checkpoint_path),
            )
            precise = STAGE_17.run_precise_validation(
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
            )

            self.assertEqual(precise.eval_iters, 3)
            self.assertEqual(precise.seed, 404)
            self.assertEqual(
                [result.name for result in precise.results],
                [control.name, treatment.name],
            )
            self.assertTrue(
                all(len(result.batch_losses) == 3 for result in precise.results)
            )
            expected_differences = [
                STAGE_17.SplitLosses(
                    treatment_loss.overall - control_loss.overall,
                    treatment_loss.first_half - control_loss.first_half,
                    treatment_loss.second_half - control_loss.second_half,
                )
                for control_loss, treatment_loss in zip(
                    precise.results[0].batch_losses,
                    precise.results[1].batch_losses,
                    strict=True,
                )
            ]
            for field in ("overall", "first_half", "second_half"):
                expected_mean = sum(
                    getattr(value, field) for value in expected_differences
                ) / len(expected_differences)
                self.assertAlmostEqual(
                    getattr(precise.delta.mean_delta, field),
                    expected_mean,
                )
            self.assertEqual(precise.delta.candidate, treatment.name)
            self.assertEqual(precise.delta.baseline, control.name)
            self.assertEqual(
                checkpoint_hashes_before,
                (
                    STAGE_17.checkpoint_sha256(control.checkpoint_path),
                    STAGE_17.checkpoint_sha256(treatment.checkpoint_path),
                ),
            )


class Stage17BenchmarkTests(unittest.TestCase):
    def test_benchmark_pauses_for_evaluation_without_adding_updates(
        self,
    ) -> None:
        data = make_character_data()
        device = torch.device("cpu")
        schedule = STAGE_17.LearningRateSchedule(
            initial_learning_rate=2e-2,
            first_decay_step=2,
            middle_learning_rate=7e-3,
            second_decay_step=4,
            final_learning_rate=2e-3,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            control = STAGE_17.ContextSpec(
                "T=4 control",
                2,
                4,
                directory / "control.pt",
            )
            treatment = STAGE_17.ContextSpec(
                "T=8 treatment",
                1,
                8,
                directory / "treatment.pt",
            )
            common_config = {
                "n_embd": 8,
                "learning_rate": schedule.initial_learning_rate,
                "max_iters": 6,
                "eval_interval": 6,
                "eval_iters": 1,
                "seed": 101,
            }
            control_config = TrainingConfig(
                batch_size=control.batch_size,
                block_size=control.block_size,
                **common_config,
            )
            treatment_config = TrainingConfig(
                batch_size=treatment.batch_size,
                block_size=treatment.block_size,
                **common_config,
            )
            initialization = STAGE_17.create_paired_initial_states(
                data,
                control_config,
                treatment_config,
                2,
                1,
            )
            reference_model = STAGE_17._build_model(
                data,
                control_config,
                2,
                1,
                device,
            )
            benchmarked_model = STAGE_17._build_model(
                data,
                control_config,
                2,
                1,
                device,
            )
            reference_model.load_state_dict(
                initialization.control,
                strict=True,
            )
            benchmarked_model.load_state_dict(
                initialization.control,
                strict=True,
            )
            reference_optimizer = torch.optim.AdamW(
                reference_model.parameters(),
                lr=schedule.initial_learning_rate,
            )
            benchmarked_optimizer = torch.optim.AdamW(
                benchmarked_model.parameters(),
                lr=schedule.initial_learning_rate,
            )
            reference_generator = torch.Generator().manual_seed(202)
            benchmarked_generator = torch.Generator().manual_seed(202)
            reference_records: list[object] = []
            benchmarked_records: list[object] = []

            reference = STAGE_17.train_with_paired_schedule(
                reference_model,
                reference_optimizer,
                data,
                control_config,
                control,
                control,
                treatment,
                device,
                reference_generator,
                schedule,
                on_evaluation=reference_records.append,
            )
            peak_values = (
                (1_024, 2_048),
                (3_072, 4_096),
                (2_048, 8_192),
            )
            benchmark_config = STAGE_17.BenchmarkConfig(
                num_warmup=1,
                num_steps=4,
            )
            with (
                mock.patch.object(
                    benchmarked_optimizer,
                    "step",
                    wraps=benchmarked_optimizer.step,
                ) as optimizer_step,
                mock.patch.object(STAGE_17, "sync_device") as sync_device,
                mock.patch.object(
                    STAGE_17,
                    "reset_peak_memory_stats",
                ) as reset_peaks,
                mock.patch.object(
                    STAGE_17,
                    "get_peak_memory_stats",
                    side_effect=peak_values,
                ) as get_peaks,
            ):
                benchmarked = STAGE_17.train_with_paired_schedule(
                    benchmarked_model,
                    benchmarked_optimizer,
                    data,
                    control_config,
                    control,
                    control,
                    treatment,
                    device,
                    benchmarked_generator,
                    schedule,
                    on_evaluation=benchmarked_records.append,
                    benchmark=benchmark_config,
                )

        self.assertIsNone(reference.benchmark)
        self.assertIsNotNone(benchmarked.benchmark)
        stats = benchmarked.benchmark
        assert stats is not None
        self.assertGreater(stats.seconds, 0.0)
        self.assertGreater(stats.iterations_per_sec, 0.0)
        self.assertAlmostEqual(
            stats.tokens_per_sec,
            stats.iterations_per_sec * control.tokens_per_update,
        )
        self.assertEqual(
            stats.peak_allocated_mb,
            3_072 / 1024**2,
        )
        self.assertEqual(
            stats.peak_reserved_mb,
            8_192 / 1024**2,
        )

        expected_steps = [0, 2, 4, 6]
        self.assertEqual(
            [record.step for record in reference.history],
            expected_steps,
        )
        self.assertEqual(
            [record.step for record in benchmarked.history],
            expected_steps,
        )
        self.assertEqual(reference_records, list(reference.history))
        self.assertEqual(benchmarked_records, list(benchmarked.history))
        self.assertEqual(reference.initial, benchmarked.initial)
        self.assertEqual(reference.final, benchmarked.final)

        # The timed optimizer steps complete at steps 2..5. The schedule
        # boundaries at 2 and 4 are therefore evaluations inside that span.
        timed_evaluation_steps = [
            step
            for step in expected_steps
            if benchmark_config.num_warmup
            < step
            < benchmark_config.num_warmup + benchmark_config.num_steps
        ]
        self.assertEqual(timed_evaluation_steps, [2, 4])
        self.assertEqual(optimizer_step.call_count, control_config.max_iters)
        self.assertEqual(reset_peaks.call_count, 3)
        self.assertEqual(get_peaks.call_count, 3)
        self.assertEqual(sync_device.call_count, 6)

        torch.testing.assert_close(
            reference_generator.get_state(),
            benchmarked_generator.get_state(),
            rtol=0,
            atol=0,
        )
        for reference_parameter, benchmarked_parameter in zip(
            reference_model.parameters(),
            benchmarked_model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                reference_parameter,
                benchmarked_parameter,
                rtol=0,
                atol=0,
            )

    def test_counterbalanced_benchmarks_are_disposable_and_aggregated(
        self,
    ) -> None:
        data = make_character_data()
        device = torch.device("cpu")
        schedule = STAGE_17.LearningRateSchedule(
            initial_learning_rate=2e-2,
            first_decay_step=2,
            middle_learning_rate=7e-3,
            second_decay_step=4,
            final_learning_rate=2e-3,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            control = STAGE_17.ContextSpec(
                "T=4 control",
                2,
                4,
                directory / "control.pt",
            )
            treatment = STAGE_17.ContextSpec(
                "T=8 treatment",
                1,
                8,
                directory / "treatment.pt",
            )
            common_config = {
                "n_embd": 8,
                "learning_rate": schedule.initial_learning_rate,
                "max_iters": 6,
                "eval_interval": 6,
                "eval_iters": 1,
                "seed": 101,
            }
            control_config = TrainingConfig(
                batch_size=control.batch_size,
                block_size=control.block_size,
                **common_config,
            )
            treatment_config = TrainingConfig(
                batch_size=treatment.batch_size,
                block_size=treatment.block_size,
                **common_config,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                baseline = STAGE_17.run_experiment(
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

            baseline_checkpoint_states = []
            for spec in (control, treatment):
                payload = torch.load(
                    spec.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model_state = payload["model_state_dict"]
                baseline_checkpoint_states.append(
                    {
                        name: tensor.detach().clone()
                        for name, tensor in model_state.items()
                    }
                )
                baseline_checkpoint_states[-1]["__training_generator_state__"] = (
                    payload["training_generator_state"].detach().clone()
                )

            benchmark = STAGE_17.BenchmarkConfig(
                num_warmup=1,
                num_steps=2,
            )
            synthetic_runs = (
                STAGE_17.BenchmarkStats(2.0, 1.0, 8.0, 1.0, 10.0),
                STAGE_17.BenchmarkStats(1.0, 2.0, 16.0, 2.0, 20.0),
                STAGE_17.BenchmarkStats(
                    3.0,
                    2.0 / 3.0,
                    16.0 / 3.0,
                    3.0,
                    30.0,
                ),
                STAGE_17.BenchmarkStats(4.0, 0.5, 4.0, 4.0, 40.0),
            )
            observed_specs: list[object] = []
            observed_state_hashes: list[str] = []
            observed_seeds: list[tuple[int, int]] = []

            def fake_benchmark_context_branch(
                _data: CharacterData,
                _config: TrainingConfig,
                spec: object,
                _control: object,
                _treatment: object,
                _n_head: int,
                _n_layer: int,
                _device: torch.device,
                initial_state: object,
                *,
                training_batch_seed: int,
                training_rng_seed: int,
                benchmark: object,
            ) -> object:
                self.assertIs(benchmark, benchmark_config)
                observed_specs.append(spec)
                assert isinstance(initial_state, dict)
                observed_state_hashes.append(
                    STAGE_17.tensor_mapping_sha256(initial_state)
                )
                observed_seeds.append((training_batch_seed, training_rng_seed))
                return synthetic_runs[len(observed_specs) - 1]

            benchmark_config = benchmark
            with (
                mock.patch.object(
                    STAGE_17,
                    "benchmark_context_branch",
                    side_effect=fake_benchmark_context_branch,
                ) as benchmark_branch,
                mock.patch.object(
                    STAGE_17,
                    "run_branch",
                    wraps=STAGE_17.run_branch,
                ) as train_branch,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                measured = STAGE_17.run_experiment(
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
                    benchmark=benchmark_config,
                )

            self.assertEqual(
                observed_specs,
                [control, treatment, treatment, control],
            )
            self.assertEqual(benchmark_branch.call_count, 4)
            self.assertEqual(observed_seeds, [(202, 303)] * 4)
            self.assertEqual(
                observed_state_hashes,
                [
                    measured.initialization.control_state_sha256,
                    measured.initialization.treatment_state_sha256,
                    measured.initialization.treatment_state_sha256,
                    measured.initialization.control_state_sha256,
                ],
            )
            self.assertEqual(train_branch.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["benchmark"] is None
                    for call in train_branch.call_args_list
                )
            )

            control_stats = measured.control.training.benchmark
            treatment_stats = measured.treatment.training.benchmark
            assert control_stats is not None
            assert treatment_stats is not None
            self.assertEqual(control_stats.seconds, 6.0)
            self.assertAlmostEqual(control_stats.iterations_per_sec, 4.0 / 6.0)
            self.assertAlmostEqual(
                control_stats.tokens_per_sec,
                (4.0 / 6.0) * control.tokens_per_update,
            )
            self.assertEqual(control_stats.peak_allocated_mb, 4.0)
            self.assertEqual(control_stats.peak_reserved_mb, 40.0)
            self.assertEqual(treatment_stats.seconds, 4.0)
            self.assertEqual(treatment_stats.iterations_per_sec, 1.0)
            self.assertEqual(
                treatment_stats.tokens_per_sec,
                treatment.tokens_per_update,
            )
            self.assertEqual(treatment_stats.peak_allocated_mb, 3.0)
            self.assertEqual(treatment_stats.peak_reserved_mb, 30.0)

            for baseline_branch, measured_branch in (
                (baseline.control, measured.control),
                (baseline.treatment, measured.treatment),
            ):
                with self.subTest(branch=baseline_branch.spec.name):
                    self.assertEqual(
                        baseline_branch.training.initial,
                        measured_branch.training.initial,
                    )
                    self.assertEqual(
                        baseline_branch.training.final,
                        measured_branch.training.final,
                    )
                    self.assertEqual(
                        baseline_branch.training.history,
                        measured_branch.training.history,
                    )
                    self.assertEqual(
                        baseline_branch.final_batch_generator_sha256,
                        measured_branch.final_batch_generator_sha256,
                    )
            self.assertEqual(
                measured.control.final_batch_generator_sha256,
                measured.treatment.final_batch_generator_sha256,
            )

            for spec, expected_state in zip(
                (control, treatment),
                baseline_checkpoint_states,
                strict=True,
            ):
                payload = torch.load(
                    spec.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                actual_state = payload["model_state_dict"]
                self.assertEqual(
                    set(actual_state),
                    set(expected_state) - {"__training_generator_state__"},
                )
                for name, tensor in actual_state.items():
                    torch.testing.assert_close(
                        tensor,
                        expected_state[name],
                        rtol=0,
                        atol=0,
                    )
                torch.testing.assert_close(
                    payload["training_generator_state"],
                    expected_state["__training_generator_state__"],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
