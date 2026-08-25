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
from data_utils import CharacterData, CharacterVocabulary
from training import BenchmarkConfig, BenchmarkStats


MODULE_PATH = Path(__file__).with_name("20_scale_width.py")
SPEC = importlib.util.spec_from_file_location("stage_20_scale_width", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE_20 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_20
SPEC.loader.exec_module(STAGE_20)


def make_character_data(
    *,
    vocabulary_size: int = 5,
    train_tokens: int = 500,
    val_tokens: int = 200,
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


def make_specs(
    directory: Path,
    *,
    control_width: int = 8,
    treatment_width: int = 16,
) -> tuple[STAGE_20.WidthSpec, STAGE_20.WidthSpec]:
    return (
        STAGE_20.WidthSpec(
            f"width-{control_width}",
            control_width,
            directory / "control.pt",
        ),
        STAGE_20.WidthSpec(
            f"width-{treatment_width}",
            treatment_width,
            directory / "treatment.pt",
        ),
    )


def make_tiny_protocol(
    directory: Path,
) -> tuple[
    CharacterData,
    TrainingConfig,
    TrainingConfig,
    STAGE_20.WidthSpec,
    STAGE_20.WidthSpec,
    STAGE_20.LearningRateSchedule,
]:
    data = make_character_data()
    control, treatment = make_specs(directory)
    control_config = TrainingConfig(
        batch_size=2,
        block_size=8,
        n_embd=8,
        learning_rate=2e-2,
        max_iters=3,
        eval_interval=3,
        eval_iters=1,
        seed=101,
    )
    treatment_config = TrainingConfig(
        batch_size=2,
        block_size=8,
        n_embd=16,
        learning_rate=2e-2,
        max_iters=3,
        eval_interval=3,
        eval_iters=1,
        seed=101,
    )
    schedule = STAGE_20.LearningRateSchedule(
        initial_learning_rate=2e-2,
        first_decay_step=1,
        middle_learning_rate=7e-3,
        second_decay_step=2,
        final_learning_rate=2e-3,
    )
    return (
        data,
        control_config,
        treatment_config,
        control,
        treatment,
        schedule,
    )


class Stage20DefaultsAndArchitectureTests(unittest.TestCase):
    def test_defaults_and_exact_parameter_budgets(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_20.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.control_width, 64)
        self.assertEqual(args.treatment_width, 128)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.initial_learning_rate, 1e-3)
        self.assertEqual(args.first_decay_step, 10_000)
        self.assertEqual(args.middle_learning_rate, 3e-4)
        self.assertEqual(args.second_decay_step, 13_000)
        self.assertEqual(args.final_learning_rate, 1e-4)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_20_width_64_best_checkpoint.pt",
        )
        self.assertEqual(
            args.treatment_checkpoint_path.name,
            "stage_20_width_128_best_checkpoint.pt",
        )

        data = make_character_data(vocabulary_size=65)
        config = TrainingConfig(
            batch_size=32,
            block_size=64,
            n_embd=64,
            learning_rate=1e-3,
            max_iters=18_000,
            eval_interval=500,
            eval_iters=1,
            seed=1_337,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            control, treatment = make_specs(
                Path(directory_name),
                control_width=64,
                treatment_width=128,
            )
            targets, bands = STAGE_20.validate_width_protocol(
                config, control, treatment, 4, 4
            )
            control_model = STAGE_20._build_model(data, config, control, 4, 4)
            treatment_config = TrainingConfig(
                batch_size=32,
                block_size=64,
                n_embd=128,
                learning_rate=1e-3,
                max_iters=18_000,
                eval_interval=500,
                eval_iters=1,
                seed=1_337,
            )
            treatment_model = STAGE_20._build_model(
                data, treatment_config, treatment, 4, 4
            )

        self.assertEqual(targets, 2_048)
        self.assertEqual(targets * 18_000, 36_864_000)
        self.assertEqual(bands, ((0, 16), (16, 32), (32, 48), (48, 64)))
        self.assertEqual(STAGE_20.DEFAULT_DROPOUT, 0.0)
        self.assertEqual(control_model.blocks[0].sa.head_size, 16)
        self.assertEqual(treatment_model.blocks[0].sa.head_size, 32)
        self.assertEqual(
            control_model.blocks[0].ffwd.net[0].weight.shape,
            (256, 64),
        )
        self.assertEqual(
            treatment_model.blocks[0].ffwd.net[0].weight.shape,
            (512, 128),
        )
        self.assertEqual(STAGE_20.count_parameters(control_model), 211_777)
        self.assertEqual(STAGE_20.count_parameters(treatment_model), 816_705)
        self.assertEqual(
            STAGE_20.expected_parameter_count(
                vocab_size=65,
                block_size=64,
                n_embd=64,
                n_layer=4,
            ),
            211_777,
        )
        self.assertEqual(
            STAGE_20.expected_parameter_count(
                vocab_size=65,
                block_size=64,
                n_embd=128,
                n_layer=4,
            ),
            816_705,
        )

    def test_native_initializations_are_deterministic_but_not_claimed_equal(
        self,
    ) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            (
                _,
                control_config,
                treatment_config,
                control,
                treatment,
                schedule,
            ) = make_tiny_protocol(Path(directory_name))
            first = STAGE_20.create_paired_initial_states(
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                2,
                1,
            )
            second = STAGE_20.create_paired_initial_states(
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                2,
                1,
            )

        self.assertEqual(first.control_state_sha256, second.control_state_sha256)
        self.assertEqual(first.treatment_state_sha256, second.treatment_state_sha256)
        self.assertNotEqual(first.control_state_sha256, first.treatment_state_sha256)
        self.assertEqual(first.control_parameter_count, 1_013)
        self.assertEqual(first.treatment_parameter_count, 3_557)

    def test_cli_rejects_invalid_protocol_before_loading_data(self) -> None:
        cases = (
            (["--n-head", "0"], "n_head must be a positive integer"),
            (
                ["--precise-eval-iters", "1"],
                "precise-eval-iters must be 0 or at least 2",
            ),
            (
                ["--precise-eval-seed", "1339"],
                "must differ from the fixed-panel seed",
            ),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with (
                    mock.patch.object(
                        sys,
                        "argv",
                        [str(MODULE_PATH), *arguments],
                    ),
                    mock.patch.object(
                        CharacterData,
                        "from_file",
                    ) as load_data,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        STAGE_20.main()
                load_data.assert_not_called()


class Stage20PairedTrainingTests(unittest.TestCase):
    def test_training_batch_traces_and_lr_schedule_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            (
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                schedule,
            ) = make_tiny_protocol(Path(directory_name))
            initialization = STAGE_20.create_paired_initial_states(
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                2,
                1,
            )
            traces: list[list[tuple[int, float, torch.Tensor, torch.Tensor]]] = []
            final_generator_states: list[torch.Tensor] = []
            branches = (
                (
                    control_config,
                    control,
                    initialization.control,
                ),
                (
                    treatment_config,
                    treatment,
                    initialization.treatment,
                ),
            )
            for config, spec, state in branches:
                model = STAGE_20._build_model(data, config, spec, 2, 1)
                model.load_state_dict(state, strict=True)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=schedule.initial_learning_rate
                )
                generator = torch.Generator().manual_seed(202)
                trace: list[tuple[int, float, torch.Tensor, torch.Tensor]] = []

                def capture(
                    step: int,
                    learning_rate: float,
                    inputs: torch.Tensor,
                    targets: torch.Tensor,
                ) -> None:
                    trace.append(
                        (
                            step,
                            learning_rate,
                            inputs.detach().clone(),
                            targets.detach().clone(),
                        )
                    )

                STAGE_20.seed_everything(303)
                result = STAGE_20.train_with_schedule(
                    model,
                    optimizer,
                    data,
                    config,
                    torch.device("cpu"),
                    generator,
                    schedule,
                    STAGE_20.make_position_bands(8),
                    on_evaluation=lambda _record: None,
                    on_update=capture,
                )
                self.assertEqual(
                    tuple(record.step for record in result.history),
                    (0, 1, 2, 3),
                )
                traces.append(trace)
                final_generator_states.append(generator.get_state().clone())

        self.assertEqual([row[1] for row in traces[0]], [2e-2, 7e-3, 2e-3])
        for control_row, treatment_row in zip(traces[0], traces[1], strict=True):
            self.assertEqual(control_row[:2], treatment_row[:2])
            torch.testing.assert_close(control_row[2], treatment_row[2], rtol=0, atol=0)
            torch.testing.assert_close(control_row[3], treatment_row[3], rtol=0, atol=0)
        torch.testing.assert_close(
            final_generator_states[0],
            final_generator_states[1],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            final_generator_states[0],
            STAGE_20.expected_training_generator_state(
                data,
                control_config,
                training_batch_seed=202,
                completed_steps=3,
            ),
            rtol=0,
            atol=0,
        )


class Stage20WorkflowTests(unittest.TestCase):
    def test_tiny_workflow_checkpoints_and_fresh_panel_are_paired(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            (
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                schedule,
            ) = make_tiny_protocol(Path(directory_name))
            with contextlib.redirect_stdout(io.StringIO()):
                report = STAGE_20.run_experiment(
                    data,
                    control_config,
                    treatment_config,
                    control,
                    treatment,
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    schedule,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    benchmark=None,
                )

            self.assertEqual(report.targets_per_update, 16)
            self.assertEqual(report.bands, ((0, 2), (2, 4), (4, 6), (6, 8)))
            self.assertEqual(report.control.parameter_count, 1_013)
            self.assertEqual(report.treatment.parameter_count, 3_557)
            self.assertEqual(report.control.attention_shape, (2, 2, 8, 8))
            self.assertEqual(report.treatment.attention_shape, (2, 2, 8, 8))
            self.assertEqual(
                report.control.final_batch_generator_sha256,
                report.treatment.final_batch_generator_sha256,
            )

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
                checkpoint = STAGE_20.validate_stage_20_checkpoint(
                    spec.checkpoint_path,
                    data,
                    config,
                    spec,
                    control,
                    treatment,
                    2,
                    1,
                    schedule,
                    report.bands,
                    initialization_seed=101,
                    branch_initial_state_sha256=initial_hash,
                    control_parameter_count=1_013,
                    treatment_parameter_count=3_557,
                    training_batch_seed=202,
                    training_rng_seed=303,
                )
                architecture = checkpoint["architecture"]
                experiment = checkpoint["experiment"]
                self.assertEqual(architecture["n_embd"], spec.n_embd)
                self.assertEqual(architecture["head_size"], spec.n_embd // 2)
                self.assertEqual(architecture["feed_forward_width"], 4 * spec.n_embd)
                self.assertEqual(architecture["attention_mask"], "full_causal")
                self.assertEqual(architecture["attention_mask_scope"], "per_layer")
                self.assertEqual(experiment["stage"], 20)
                self.assertEqual(experiment["comparison_variable"], "embedding_width")
                self.assertFalse(experiment["identical_parameter_count"])
                self.assertFalse(experiment["identical_learned_initialization"])
                self.assertFalse(experiment["identical_position_embeddings"])
                self.assertTrue(experiment["identical_training_batches"])
                self.assertTrue(experiment["identical_attention_shapes"])
                self.assertTrue(experiment["identical_fixed_validation_batches"])
                self.assertTrue(experiment["fresh_paired_validation_protocol"])

            hashes_before = (
                STAGE_20.checkpoint_sha256(control.checkpoint_path),
                STAGE_20.checkpoint_sha256(treatment.checkpoint_path),
            )
            precise_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            original_get_batch = CharacterData.get_batch

            def capture_precise_batch(
                batch_data: CharacterData,
                split: str,
                **kwargs,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                inputs, targets = original_get_batch(batch_data, split, **kwargs)
                if split == "val":
                    precise_batches.append(
                        (inputs.detach().clone(), targets.detach().clone())
                    )
                return inputs, targets

            with mock.patch.object(
                CharacterData,
                "get_batch",
                new=capture_precise_batch,
            ):
                precise = STAGE_20.run_precise_validation(
                    data,
                    control_config,
                    treatment_config,
                    control,
                    treatment,
                    2,
                    1,
                    torch.device("cpu"),
                    schedule,
                    report.initialization,
                    report.bands,
                    initialization_seed=101,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    eval_iters=3,
                    seed=404,
                )
            hashes_after = (
                STAGE_20.checkpoint_sha256(control.checkpoint_path),
                STAGE_20.checkpoint_sha256(treatment.checkpoint_path),
            )

            control_payload = torch.load(
                control.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            treatment_payload = torch.load(
                treatment.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            treatment_payload["optimizer_state_dict"] = control_payload[
                "optimizer_state_dict"
            ]
            torch.save(treatment_payload, treatment.checkpoint_path)
            with self.assertRaisesRegex(ValueError, "optimizer .* incompatible"):
                STAGE_20.validate_stage_20_checkpoint(
                    treatment.checkpoint_path,
                    data,
                    treatment_config,
                    treatment,
                    control,
                    treatment,
                    2,
                    1,
                    schedule,
                    report.bands,
                    initialization_seed=101,
                    branch_initial_state_sha256=(
                        report.initialization.treatment_state_sha256
                    ),
                    control_parameter_count=1_013,
                    treatment_parameter_count=3_557,
                    training_batch_seed=202,
                    training_rng_seed=303,
                )

            corrupt = control_payload
            mask_name = next(
                name for name in corrupt["model_state_dict"] if name.endswith(".tril")
            )
            corrupt["model_state_dict"][mask_name] = torch.zeros((8, 8))
            torch.save(corrupt, control.checkpoint_path)
            with self.assertRaisesRegex(ValueError, "incorrect causal mask"):
                STAGE_20.validate_stage_20_checkpoint(
                    control.checkpoint_path,
                    data,
                    control_config,
                    control,
                    control,
                    treatment,
                    2,
                    1,
                    schedule,
                    report.bands,
                    initialization_seed=101,
                    branch_initial_state_sha256=(
                        report.initialization.control_state_sha256
                    ),
                    control_parameter_count=1_013,
                    treatment_parameter_count=3_557,
                    training_batch_seed=202,
                    training_rng_seed=303,
                )

        self.assertEqual(hashes_before, hashes_after)
        self.assertEqual(len(precise.results), 2)
        self.assertEqual(len(precise.results[0].batch_losses), 3)
        self.assertEqual(len(precise_batches), 6)
        for control_batch, treatment_batch in zip(
            precise_batches[:3], precise_batches[3:], strict=True
        ):
            torch.testing.assert_close(
                control_batch[0], treatment_batch[0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                control_batch[1], treatment_batch[1], rtol=0, atol=0
            )
        expected_differences = tuple(
            STAGE_20._STAGE_18._subtract_position_losses(candidate, baseline)
            for candidate, baseline in zip(
                precise.results[1].batch_losses,
                precise.results[0].batch_losses,
                strict=True,
            )
        )
        expected_statistics = STAGE_20._position_statistics(expected_differences)
        self.assertEqual(precise.delta.mean_delta, expected_statistics.mean)
        self.assertEqual(
            precise.delta.standard_error,
            expected_statistics.standard_error,
        )
        self.assertEqual(
            precise.delta.critical_value,
            STAGE_20._paired_t_95_critical_value(3),
        )

    def test_disposable_benchmark_cannot_change_quality_training(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_directory_name,
            tempfile.TemporaryDirectory() as second_directory_name,
        ):
            first_protocol = make_tiny_protocol(Path(first_directory_name))
            second_protocol = make_tiny_protocol(Path(second_directory_name))
            with contextlib.redirect_stdout(io.StringIO()):
                without_benchmark = STAGE_20.run_experiment(
                    *first_protocol[:5],
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    first_protocol[5],
                    training_batch_seed=202,
                    training_rng_seed=303,
                    benchmark=None,
                )
                with_benchmark = STAGE_20.run_experiment(
                    *second_protocol[:5],
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    second_protocol[5],
                    training_batch_seed=202,
                    training_rng_seed=303,
                    benchmark=BenchmarkConfig(num_warmup=1, num_steps=1),
                )

            for first_branch, second_branch in (
                (without_benchmark.control, with_benchmark.control),
                (without_benchmark.treatment, with_benchmark.treatment),
            ):
                first_checkpoint = torch.load(
                    first_branch.spec.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                second_checkpoint = torch.load(
                    second_branch.spec.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(
                    STAGE_20.tensor_mapping_sha256(
                        first_checkpoint["model_state_dict"]
                    ),
                    STAGE_20.tensor_mapping_sha256(
                        second_checkpoint["model_state_dict"]
                    ),
                )
                self.assertEqual(first_branch.training, second_branch.training)
                self.assertEqual(
                    first_branch.best_val_loss,
                    second_branch.best_val_loss,
                )
                self.assertEqual(first_branch.best_step, second_branch.best_step)

        self.assertIsNone(without_benchmark.control_benchmark)
        self.assertIsNotNone(with_benchmark.control_benchmark)
        self.assertIsNotNone(with_benchmark.treatment_benchmark)

    def test_disposable_benchmark_applies_the_lr_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            (
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                schedule,
            ) = make_tiny_protocol(Path(directory_name))
            initialization = STAGE_20.create_paired_initial_states(
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                2,
                1,
            )
            observed_rates: list[float] = []
            original_set_rate = STAGE_20.set_optimizer_learning_rate

            def capture_rate(optimizer, learning_rate):
                observed_rates.append(learning_rate)
                original_set_rate(optimizer, learning_rate)

            with mock.patch.object(
                STAGE_20,
                "set_optimizer_learning_rate",
                new=capture_rate,
            ):
                STAGE_20.benchmark_width_branch(
                    data,
                    control_config,
                    control,
                    2,
                    1,
                    schedule,
                    torch.device("cpu"),
                    initialization.control,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    benchmark=BenchmarkConfig(num_warmup=0, num_steps=3),
                )

        self.assertEqual(observed_rates, [2e-2, 7e-3, 2e-3])

    def test_counterbalanced_benchmark_order_and_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            (
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                schedule,
            ) = make_tiny_protocol(Path(directory_name))
            initialization = STAGE_20.create_paired_initial_states(
                data,
                control_config,
                treatment_config,
                control,
                treatment,
                2,
                1,
            )
            calls: list[tuple[str, int, int, bool]] = []
            elapsed_values = iter((1.0, 2.0, 4.0, 8.0))

            def fake_benchmark(
                _data,
                config,
                spec,
                _n_head,
                _n_layer,
                _schedule,
                _device,
                state,
                *,
                training_batch_seed,
                training_rng_seed,
                benchmark,
            ):
                expected_state = (
                    initialization.control
                    if spec == control
                    else initialization.treatment
                )
                is_expected_state = STAGE_20.tensor_mapping_sha256(
                    state
                ) == STAGE_20.tensor_mapping_sha256(expected_state)
                calls.append(
                    (
                        spec.name,
                        training_batch_seed,
                        training_rng_seed,
                        is_expected_state,
                    )
                )
                elapsed = next(elapsed_values)
                iterations_per_sec = benchmark.num_steps / elapsed
                return BenchmarkStats(
                    seconds=elapsed,
                    iterations_per_sec=iterations_per_sec,
                    tokens_per_sec=(
                        iterations_per_sec * config.batch_size * config.block_size
                    ),
                    peak_allocated_mb=elapsed,
                    peak_reserved_mb=2 * elapsed,
                )

            benchmark = BenchmarkConfig(num_warmup=1, num_steps=2)
            with mock.patch.object(
                STAGE_20,
                "benchmark_width_branch",
                new=fake_benchmark,
            ):
                control_stats, treatment_stats = (
                    STAGE_20.run_counterbalanced_benchmarks(
                        data,
                        control_config,
                        treatment_config,
                        control,
                        treatment,
                        2,
                        1,
                        schedule,
                        torch.device("cpu"),
                        initialization,
                        training_batch_seed=202,
                        training_rng_seed=303,
                        benchmark=benchmark,
                    )
                )

        self.assertEqual(
            [call[0] for call in calls],
            [control.name, treatment.name, treatment.name, control.name],
        )
        self.assertTrue(all(call[1:] == (202, 303, True) for call in calls))
        self.assertAlmostEqual(control_stats.iterations_per_sec, 4 / 9)
        self.assertAlmostEqual(treatment_stats.iterations_per_sec, 4 / 6)
        self.assertEqual(control_stats.peak_allocated_mb, 8.0)
        self.assertEqual(treatment_stats.peak_allocated_mb, 4.0)
        self.assertEqual(control_stats.peak_reserved_mb, 16.0)
        self.assertEqual(treatment_stats.peak_reserved_mb, 8.0)


if __name__ == "__main__":
    unittest.main()
