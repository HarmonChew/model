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
from training import BenchmarkConfig, BenchmarkStats


STAGE_22_PATH = Path(__file__).with_name("22_dropout_strength.py")
STAGE_22_SPEC = importlib.util.spec_from_file_location(
    "stage_22_dropout_strength_for_stage_24_tests",
    STAGE_22_PATH,
)
assert STAGE_22_SPEC is not None and STAGE_22_SPEC.loader is not None
STAGE_22 = importlib.util.module_from_spec(STAGE_22_SPEC)
sys.modules[STAGE_22_SPEC.name] = STAGE_22
STAGE_22_SPEC.loader.exec_module(STAGE_22)

MODULE_PATH = Path(__file__).with_name("24_scale_depth.py")
SPEC = importlib.util.spec_from_file_location("stage_24_scale_depth", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE_24 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_24
SPEC.loader.exec_module(STAGE_24)


def make_depth_specs(
    directory: Path,
    *,
    control_layers: int = 1,
    treatment_layers: int = 3,
) -> tuple[STAGE_24.DepthSpec, STAGE_24.DepthSpec]:
    return (
        STAGE_24.DepthSpec(
            f"depth-{control_layers}",
            control_layers,
            directory / "control.pt",
        ),
        STAGE_24.DepthSpec(
            f"depth-{treatment_layers}",
            treatment_layers,
            directory / "treatment.pt",
        ),
    )


def make_depth_protocol(
    directory: Path,
    *,
    eval_iters: int = 1,
) -> tuple[
    object,
    TrainingConfig,
    STAGE_24.DepthSpec,
    STAGE_24.DepthSpec,
    STAGE_24.LearningRateSchedule,
]:
    data = make_character_data()
    schedule = STAGE_24.LearningRateSchedule(
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
        eval_iters=eval_iters,
        seed=101,
    )
    control, treatment = make_depth_specs(directory)
    return data, config, control, treatment, schedule


class Stage24DefaultsAndInitializationTests(unittest.TestCase):
    def test_defaults_counts_native_hashes_shared_prefix_and_dropout_sites(
        self,
    ) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_24.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 128)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.control_layers, 4)
        self.assertEqual(args.treatment_layers, 6)
        self.assertEqual(args.dropout, 0.10)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.initial_learning_rate, 1e-3)
        self.assertEqual(args.first_decay_step, 10_000)
        self.assertEqual(args.middle_learning_rate, 3e-4)
        self.assertEqual(args.second_decay_step, 13_000)
        self.assertEqual(args.final_learning_rate, 1e-4)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(args.benchmark_warmup, 20)
        self.assertEqual(args.benchmark_steps, 100)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_22_dropout_010_best_checkpoint.pt",
        )
        self.assertEqual(
            args.checkpoint_path.name,
            "stage_24_depth_6_best_checkpoint.pt",
        )
        self.assertEqual(
            STAGE_24._pin_for_default_path(args.control_checkpoint_path),
            (STAGE_24.DEFAULT_CONTROL_CHECKPOINT_SHA256, 15_500),
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
        with tempfile.TemporaryDirectory() as directory_name:
            control, treatment = make_depth_specs(
                Path(directory_name),
                control_layers=4,
                treatment_layers=6,
            )
            initialization = STAGE_24.create_paired_initial_states(
                data,
                config,
                control,
                treatment,
                4,
            )
            control_model = STAGE_24._build_model(data, config, control, 4)
            treatment_model = STAGE_24._build_model(data, config, treatment, 4)

        self.assertEqual(initialization.control_parameter_count, 816_705)
        self.assertEqual(initialization.treatment_parameter_count, 1_212_481)
        self.assertEqual(
            STAGE_24.expected_parameter_count(
                vocab_size=65,
                block_size=64,
                n_embd=128,
                n_layer=4,
            ),
            816_705,
        )
        self.assertEqual(
            STAGE_24.expected_parameter_count(
                vocab_size=65,
                block_size=64,
                n_embd=128,
                n_layer=6,
            ),
            1_212_481,
        )
        self.assertEqual(
            initialization.control_state_sha256,
            STAGE_24.DEFAULT_CONTROL_INITIAL_STATE_SHA256,
        )
        self.assertEqual(
            initialization.treatment_state_sha256,
            STAGE_24.DEFAULT_TREATMENT_INITIAL_STATE_SHA256,
        )
        self.assertNotEqual(
            initialization.control_state_sha256,
            initialization.treatment_state_sha256,
        )

        # Resetting one seed preserves the architecture's shared construction
        # prefix. The final projection differs because L=6 constructs two more
        # blocks before constructing that projection.
        shared_prefix_names = [
            name
            for name in initialization.control
            if name in initialization.treatment and not name.startswith("lm_head.")
        ]
        self.assertTrue(shared_prefix_names)
        for name in shared_prefix_names:
            torch.testing.assert_close(
                initialization.control[name],
                initialization.treatment[name],
                rtol=0,
                atol=0,
            )
        self.assertFalse(
            torch.equal(
                initialization.control["lm_head.weight"],
                initialization.treatment["lm_head.weight"],
            )
        )
        extra_names = set(initialization.treatment).difference(initialization.control)
        self.assertTrue(any(name.startswith("blocks.4.") for name in extra_names))
        self.assertTrue(any(name.startswith("blocks.5.") for name in extra_names))

        self.assertEqual(len(control_model.blocks), 4)
        self.assertEqual(len(treatment_model.blocks), 6)
        self.assertEqual(control_model.blocks[0].sa.head_size, 32)
        self.assertEqual(treatment_model.blocks[0].sa.head_size, 32)
        self.assertEqual(
            treatment_model.blocks[0].ffwd.net[0].weight.shape,
            (512, 128),
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
        self.assertEqual(len(control_dropouts), 8)
        self.assertEqual(len(treatment_dropouts), 12)
        self.assertTrue(all(module.p == 0.10 for module in control_dropouts))
        self.assertTrue(all(module.p == 0.10 for module in treatment_dropouts))
        self.assertEqual(
            len(
                [name for name in control_model.state_dict() if name.endswith(".tril")]
            ),
            16,
        )
        self.assertEqual(
            len(
                [
                    name
                    for name in treatment_model.state_dict()
                    if name.endswith(".tril")
                ]
            ),
            24,
        )


class Stage24ProtocolTests(unittest.TestCase):
    def test_protocol_pin_and_fixed_eval_mode(self) -> None:
        custom_path = Path("custom_stage_22.pt")
        with self.assertRaisesRegex(ValueError, "custom Stage 22 control"):
            STAGE_24.resolve_control_pin(custom_path, None, None)
        with self.assertRaisesRegex(ValueError, "provided together"):
            STAGE_24.resolve_control_pin(custom_path, "a" * 64, None)
        with self.assertRaisesRegex(ValueError, "64 hex digits"):
            STAGE_24.resolve_control_pin(custom_path, "not-a-hash", 3)
        self.assertEqual(
            STAGE_24.resolve_control_pin(custom_path, "A" * 64, 3),
            ("a" * 64, 3),
        )

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            data, config, control, treatment, schedule = make_depth_protocol(
                directory,
                eval_iters=2,
            )
            targets, bands = STAGE_24.validate_depth_protocol(
                config,
                control,
                treatment,
                2,
                0.10,
                schedule,
            )
            self.assertEqual(targets, 16)
            self.assertEqual(bands, ((0, 2), (2, 4), (4, 6), (6, 8)))

            wrong_depth = STAGE_24.DepthSpec(
                "depth-2",
                2,
                directory / "wrong-depth.pt",
            )
            with self.assertRaisesRegex(ValueError, "exactly two layers"):
                STAGE_24.validate_depth_protocol(
                    config,
                    control,
                    wrong_depth,
                    2,
                    0.10,
                    schedule,
                )
            with self.assertRaisesRegex(ValueError, "exactly 0.10"):
                STAGE_24.validate_depth_protocol(
                    config,
                    control,
                    treatment,
                    2,
                    0.15,
                    schedule,
                )

            model = STAGE_24._build_model(data, config, treatment, 2)
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
                first = STAGE_24.evaluate_on_fixed_batches(
                    model,
                    data,
                    config,
                    torch.device("cpu"),
                    bands,
                )
                second = STAGE_24.evaluate_on_fixed_batches(
                    model,
                    data,
                    config,
                    torch.device("cpu"),
                    bands,
                )
            finally:
                for handle in handles:
                    handle.remove()

        self.assertEqual(first, second)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertEqual(len(observations), 48)
        self.assertTrue(
            all(
                not dropout_training and not gradients_enabled
                for dropout_training, gradients_enabled in observations
            )
        )


class Stage24BenchmarkTests(unittest.TestCase):
    def test_benchmark_schedule_order_state_and_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            (
                data,
                config,
                control,
                treatment,
                schedule,
            ) = make_depth_protocol(Path(directory_name))
            initialization = STAGE_24.create_paired_initial_states(
                data,
                config,
                control,
                treatment,
                2,
            )
            state_hash_before = STAGE_24.tensor_mapping_sha256(initialization.control)
            observed_rates: list[float] = []
            original_set_rate = STAGE_24.set_optimizer_learning_rate

            def capture_rate(optimizer, learning_rate):
                observed_rates.append(learning_rate)
                original_set_rate(optimizer, learning_rate)

            with mock.patch.object(
                STAGE_24,
                "set_optimizer_learning_rate",
                new=capture_rate,
            ):
                STAGE_24.benchmark_depth_branch(
                    data,
                    config,
                    control,
                    2,
                    schedule,
                    torch.device("cpu"),
                    initialization.control,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    benchmark=BenchmarkConfig(num_warmup=0, num_steps=3),
                )

            self.assertEqual(observed_rates, [2e-2, 7e-3, 2e-3])
            self.assertEqual(
                STAGE_24.tensor_mapping_sha256(initialization.control),
                state_hash_before,
            )

            calls: list[tuple[str, int, int, float, bool]] = []
            elapsed_values = iter((1.0, 2.0, 4.0, 8.0))

            def fake_benchmark(
                _data,
                _config,
                spec,
                _n_head,
                _schedule,
                _device,
                state,
                *,
                training_batch_seed,
                training_rng_seed,
                benchmark,
                dropout,
            ):
                expected_state = (
                    initialization.control
                    if spec == control
                    else initialization.treatment
                )
                state_matches = STAGE_24.tensor_mapping_sha256(
                    state
                ) == STAGE_24.tensor_mapping_sha256(expected_state)
                calls.append(
                    (
                        spec.name,
                        training_batch_seed,
                        training_rng_seed,
                        dropout,
                        state_matches,
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
                STAGE_24,
                "benchmark_depth_branch",
                new=fake_benchmark,
            ):
                control_stats, treatment_stats = (
                    STAGE_24.run_counterbalanced_benchmarks(
                        data,
                        config,
                        control,
                        treatment,
                        2,
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
        self.assertTrue(all(call[1:] == (202, 303, 0.10, True) for call in calls))
        self.assertAlmostEqual(control_stats.iterations_per_sec, 4 / 9)
        self.assertAlmostEqual(treatment_stats.iterations_per_sec, 4 / 6)
        self.assertEqual(control_stats.peak_allocated_mb, 8.0)
        self.assertEqual(treatment_stats.peak_allocated_mb, 4.0)
        self.assertEqual(control_stats.peak_reserved_mb, 16.0)
        self.assertEqual(treatment_stats.peak_reserved_mb, 8.0)

    def test_disposable_benchmark_cannot_change_treatment_training(self) -> None:
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
                    directory / "stage22_for_benchmark.pt",
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

            control = STAGE_24.DepthSpec(
                "depth-1",
                1,
                stage_22_treatment.checkpoint_path,
            )
            without_benchmark_spec = STAGE_24.DepthSpec(
                "depth-3",
                3,
                directory / "stage24_without_benchmark.pt",
            )
            with_benchmark_spec = STAGE_24.DepthSpec(
                "depth-3",
                3,
                directory / "stage24_with_benchmark.pt",
            )
            control_hash = STAGE_24.checkpoint_sha256(control.checkpoint_path)
            control_bytes = control.checkpoint_path.read_bytes()
            shared_arguments = {
                "training_batch_seed": 202,
                "training_rng_seed": 303,
                "expected_control_sha256": control_hash,
                "expected_control_step": stage_22.treatment.best_step,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                without_benchmark = STAGE_24.run_experiment(
                    data,
                    config,
                    control,
                    without_benchmark_spec,
                    2,
                    torch.device("cpu"),
                    0,
                    schedule,
                    benchmark=None,
                    **shared_arguments,
                )
                with_benchmark = STAGE_24.run_experiment(
                    data,
                    config,
                    control,
                    with_benchmark_spec,
                    2,
                    torch.device("cpu"),
                    0,
                    schedule,
                    benchmark=BenchmarkConfig(num_warmup=1, num_steps=1),
                    **shared_arguments,
                )

            without_checkpoint = torch.load(
                without_benchmark_spec.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            with_checkpoint = torch.load(
                with_benchmark_spec.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            control_bytes_after = control.checkpoint_path.read_bytes()

        self.assertEqual(control_bytes_after, control_bytes)
        self.assertEqual(
            STAGE_24.tensor_mapping_sha256(without_checkpoint["model_state_dict"]),
            STAGE_24.tensor_mapping_sha256(with_checkpoint["model_state_dict"]),
        )
        self.assertEqual(
            without_benchmark.treatment.training,
            with_benchmark.treatment.training,
        )
        self.assertEqual(
            without_benchmark.treatment.best_val_loss,
            with_benchmark.treatment.best_val_loss,
        )
        self.assertEqual(
            without_benchmark.treatment.best_step,
            with_benchmark.treatment.best_step,
        )
        self.assertIsNone(without_benchmark.control_benchmark)
        self.assertIsNone(without_benchmark.treatment_benchmark)
        self.assertIsNotNone(with_benchmark.control_benchmark)
        self.assertIsNotNone(with_benchmark.treatment_benchmark)


class Stage24WorkflowTests(unittest.TestCase):
    def test_custom_stage_22_control_treatment_checkpoint_and_fresh_panel(
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
                    expected_control_sha256=STAGE_22.checkpoint_sha256(
                        stage_21_treatment.checkpoint_path
                    ),
                    expected_control_step=stage_21.treatment.best_step,
                )

            control = STAGE_24.DepthSpec(
                "depth-1",
                1,
                stage_22_treatment.checkpoint_path,
            )
            treatment = STAGE_24.DepthSpec(
                "depth-3",
                3,
                directory / "stage24_depth_3.pt",
            )
            control_bytes = control.checkpoint_path.read_bytes()
            control_hash = STAGE_24.checkpoint_sha256(control.checkpoint_path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report = STAGE_24.run_experiment(
                    data,
                    config,
                    control,
                    treatment,
                    2,
                    torch.device("cpu"),
                    0,
                    schedule,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_control_sha256=control_hash,
                    expected_control_step=stage_22.treatment.best_step,
                    benchmark=None,
                )

            self.assertEqual(control.checkpoint_path.read_bytes(), control_bytes)
            self.assertIn("masks are not claimed identical", output.getvalue())
            self.assertIn("eval-train", output.getvalue())
            self.assertTrue(treatment.checkpoint_path.is_file())
            self.assertEqual(report.control.spec.n_layer, 1)
            self.assertEqual(report.treatment.spec.n_layer, 3)
            self.assertEqual(
                report.treatment_checkpoint_sha256,
                STAGE_24.checkpoint_sha256(treatment.checkpoint_path),
            )
            self.assertEqual(
                report.initialization.control_state_sha256,
                stage_22.initial_state_sha256,
            )
            payload = STAGE_24.validate_stage_24_checkpoint(
                treatment.checkpoint_path,
                data,
                config,
                treatment,
                control,
                treatment,
                2,
                schedule,
                report.bands,
                report.initialization,
                training_batch_seed=202,
                training_rng_seed=303,
                control_checkpoint_sha256=control_hash,
                control_checkpoint_step=stage_22.treatment.best_step,
            )
            experiment = payload["experiment"]
            self.assertEqual(experiment["stage"], 24)
            self.assertEqual(experiment["comparison_variable"], "transformer_depth")
            self.assertEqual(experiment["control_n_layer"], 1)
            self.assertEqual(experiment["treatment_n_layer"], 3)
            self.assertFalse(experiment["identical_dropout_masks"])
            self.assertTrue(experiment["reused_control"])
            self.assertEqual(experiment["control_stage"], 22)
            self.assertEqual(payload["architecture"]["n_layer"], 3)
            self.assertEqual(payload["architecture"]["residual_dropout"], 0.10)
            self.assertEqual(
                payload["initialization"]["kind"],
                STAGE_24.INITIALIZATION_KIND,
            )
            self.assertFalse(
                payload["initialization"]["identical_complete_model_state"]
            )
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
                precise = STAGE_24.run_precise_validation(
                    data,
                    config,
                    control,
                    treatment,
                    2,
                    torch.device("cpu"),
                    schedule,
                    report,
                    eval_iters=3,
                    seed=404,
                )
            self.assertEqual(
                [result.name for result in precise.results],
                ["L=1", "L=3"],
            )
            self.assertEqual(precise.delta.candidate, "L=3")
            self.assertEqual(precise.delta.baseline, "L=1")
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
            self.assertEqual(control.checkpoint_path.read_bytes(), control_bytes)
            self.assertEqual(treatment.checkpoint_path.read_bytes(), treatment_bytes)

            corrupted = torch.load(
                treatment.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            corrupted["optimizer_state_dict"]["param_groups"][0]["weight_decay"] = 0.02
            torch.save(corrupted, treatment.checkpoint_path)
            with self.assertRaisesRegex(ValueError, "weight decay"):
                STAGE_24.validate_stage_24_checkpoint(
                    treatment.checkpoint_path,
                    data,
                    config,
                    treatment,
                    control,
                    treatment,
                    2,
                    schedule,
                    report.bands,
                    report.initialization,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    control_checkpoint_sha256=control_hash,
                    control_checkpoint_step=stage_22.treatment.best_step,
                )
            treatment.checkpoint_path.write_bytes(treatment_bytes)

            treatment.checkpoint_path.write_bytes(treatment_bytes + b"replaced")
            with self.assertRaisesRegex(
                RuntimeError,
                "changed after checkpoint selection",
            ):
                STAGE_24.run_precise_validation(
                    data,
                    config,
                    control,
                    treatment,
                    2,
                    torch.device("cpu"),
                    schedule,
                    report,
                    eval_iters=3,
                    seed=404,
                )


if __name__ == "__main__":
    unittest.main()
