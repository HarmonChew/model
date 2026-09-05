from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("40_late_learning_rate_refinement.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_40_late_learning_rate_refinement_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_40 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_40
SPEC.loader.exec_module(STAGE_40)


def make_canonical_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=STAGE_40.DEFAULT_BATCH_SIZE,
        block_size=STAGE_40.DEFAULT_BLOCK_SIZE,
        n_embd=STAGE_40.DEFAULT_N_EMBD,
        learning_rate=STAGE_40.DEFAULT_SOURCE_LEARNING_RATE,
        max_iters=STAGE_40.DEFAULT_TARGET_STEP,
        eval_interval=STAGE_40.DEFAULT_EVAL_INTERVAL,
        eval_iters=STAGE_40.DEFAULT_EVAL_ITERS,
        seed=STAGE_40.DEFAULT_SEED,
    )


def make_small_data() -> CharacterData:
    vocabulary = CharacterVocabulary.from_text("abcde")
    token_ids = torch.arange(2_400, dtype=torch.long) % vocabulary.size
    return CharacterData(
        vocabulary=vocabulary,
        train_data=token_ids[:1_800],
        val_data=token_ids[1_800:],
        num_characters=len(token_ids),
    )


def make_stage_38_source() -> object:
    return STAGE_40._STAGE_39.Stage38Source(
        path=Path("synthetic_stage_38.pt"),
        sha256="8" * 64,
        checkpoint={},
        step=25_000,
        best_step=25_000,
        best_train_loss=1.6,
        best_val_loss=1.7,
        snapshot_train_loss=1.6,
        snapshot_val_loss=1.7,
        runtime_device_type="cpu",
        runtime_device_name="cpu",
        torch_version=str(torch.__version__),
        canonical=False,
    )


class ScalarTrainingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.completed_updates = 0

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del inputs, targets
        return self.weight.reshape(1, 1, 1), (self.weight - 1.0).square()


class RecordingOptimizer:
    def __init__(self, model: ScalarTrainingModel) -> None:
        self.model = model
        self.param_groups: list[dict[str, float]] = [{"lr": 0.0}]
        self.learning_rates: list[float] = []

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none
        self.model.weight.grad = None

    def step(self) -> None:
        self.learning_rates.append(self.param_groups[0]["lr"])
        self.model.completed_updates += 1


class PairedRecordingModel(torch.nn.Module):
    def __init__(self, loss_offset: float) -> None:
        super().__init__()
        self.loss_offset = loss_offset
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets
        self.inputs.append(inputs.detach().clone())
        # Deliberately consume the caller's Torch RNG. The paired evaluator
        # must still restore its caller's exact state in its finally block.
        torch.rand(1)
        loss = inputs.to(torch.float64).mean() + self.loss_offset
        return inputs.to(torch.float32).unsqueeze(-1), loss


class Stage40LateLearningRateRefinementTests(unittest.TestCase):
    def test_locked_defaults_schedules_cli_and_distinct_default_paths(self) -> None:
        self.assertEqual(STAGE_40.EXPERIMENT_STAGE, 40)
        self.assertEqual(
            (
                STAGE_40.DEFAULT_BATCH_SIZE,
                STAGE_40.DEFAULT_BLOCK_SIZE,
                STAGE_40.DEFAULT_N_EMBD,
                STAGE_40.DEFAULT_N_HEAD,
                STAGE_40.DEFAULT_HEAD_SIZE,
                STAGE_40.DEFAULT_FEED_FORWARD_WIDTH,
                STAGE_40.DEFAULT_N_LAYER,
                STAGE_40.DEFAULT_DROPOUT,
            ),
            (8, 256, 128, 4, 32, 512, 6, 0.10),
        )
        self.assertEqual(
            STAGE_40.DEFAULT_LAYER_CONTEXTS,
            (23, 23, 23, 23, None, None),
        )
        self.assertEqual(STAGE_40.DEFAULT_SOURCE_STEP, 32_000)
        self.assertEqual(STAGE_40.DEFAULT_TARGET_STEP, 38_000)
        self.assertEqual(STAGE_40.DEFAULT_SOURCE_LEARNING_RATE, 3e-5)
        self.assertEqual(STAGE_40.DEFAULT_CONTROL_LEARNING_RATE, 3e-5)
        self.assertEqual(STAGE_40.DEFAULT_TREATMENT_LEARNING_RATE, 1e-5)
        self.assertEqual(STAGE_40.DEFAULT_FRESH_VALIDATION_ITERS, 500)
        self.assertEqual(STAGE_40.DEFAULT_FRESH_VALIDATION_SEED, 1_354)
        self.assertEqual(STAGE_40.DEFAULT_TOTAL_TARGET_CHARACTERS, 77_824_000)
        self.assertEqual(
            STAGE_40.DEFAULT_ADDED_TARGET_CHARACTERS_PER_BRANCH,
            12_288_000,
        )
        self.assertEqual(
            STAGE_40.DEFAULT_EXPERIMENT_ADDED_TARGET_CHARACTERS,
            24_576_000,
        )

        control_schedule = STAGE_40.LateLearningRateSchedule(3e-5)
        treatment_schedule = STAGE_40.LateLearningRateSchedule(1e-5)
        expected_history = {
            0: 1e-3,
            9_999: 1e-3,
            10_000: 3e-4,
            12_999: 3e-4,
            13_000: 1e-4,
            17_999: 1e-4,
            18_000: 3e-5,
            31_999: 3e-5,
        }
        for update_index, expected_rate in expected_history.items():
            with self.subTest(update_index=update_index):
                self.assertEqual(
                    control_schedule.for_update(update_index), expected_rate
                )
                self.assertEqual(
                    treatment_schedule.for_update(update_index), expected_rate
                )
        self.assertEqual(control_schedule.for_update(32_000), 3e-5)
        self.assertEqual(treatment_schedule.for_update(32_000), 1e-5)
        self.assertEqual(treatment_schedule.for_update(37_999), 1e-5)
        self.assertEqual(
            treatment_schedule.as_metadata(target_step=38_000)[-1],
            {
                "start_update": 32_000,
                "end_update_exclusive": 38_000,
                "learning_rate": 1e-5,
            },
        )
        self.assertEqual(
            control_schedule.as_metadata(target_step=38_000)[-2:],
            [
                {
                    "start_update": 18_000,
                    "end_update_exclusive": 32_000,
                    "learning_rate": 3e-5,
                },
                {
                    "start_update": 32_000,
                    "end_update_exclusive": 38_000,
                    "learning_rate": 3e-5,
                },
            ],
        )

        with patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_40.parse_args()
        self.assertEqual(args.data, STAGE_40.DEFAULT_COMPLETE_DATA_PATH)
        self.assertEqual(
            args.source_checkpoint,
            STAGE_40.DEFAULT_SOURCE_CHECKPOINT_PATH,
        )
        self.assertEqual(
            args.stage_38_checkpoint,
            STAGE_40.DEFAULT_STAGE_38_CHECKPOINT_PATH,
        )
        self.assertEqual(args.target_step, 38_000)
        self.assertEqual(args.fresh_eval_iters, 500)
        self.assertEqual(args.fresh_eval_seed, 1_354)
        self.assertEqual(
            args.control_best_checkpoint_path,
            STAGE_40.DEFAULT_CONTROL_BEST_CHECKPOINT_PATH,
        )
        self.assertEqual(
            args.treatment_best_checkpoint_path,
            STAGE_40.DEFAULT_TREATMENT_BEST_CHECKPOINT_PATH,
        )
        self.assertIsNone(args.control_final_checkpoint_path)
        self.assertIsNone(args.treatment_final_checkpoint_path)
        self.assertFalse(hasattr(args, "control_learning_rate"))
        self.assertFalse(hasattr(args, "treatment_learning_rate"))

        self.assertEqual(
            STAGE_40._default_final_checkpoint_path("control", 38_000),
            STAGE_40.DEFAULT_CONTROL_FINAL_CHECKPOINT_PATH,
        )
        self.assertEqual(
            STAGE_40._default_final_checkpoint_path("treatment", 38_000),
            STAGE_40.DEFAULT_TREATMENT_FINAL_CHECKPOINT_PATH,
        )
        self.assertIn(
            "37000",
            STAGE_40._default_final_checkpoint_path("control", 37_000).name,
        )

        default_paths = {
            STAGE_40.DEFAULT_COMPLETE_DATA_PATH.resolve(),
            STAGE_40.DEFAULT_SOURCE_CHECKPOINT_PATH.resolve(),
            STAGE_40.DEFAULT_STAGE_38_CHECKPOINT_PATH.resolve(),
            STAGE_40.DEFAULT_CONTROL_BEST_CHECKPOINT_PATH.resolve(),
            STAGE_40.DEFAULT_CONTROL_FINAL_CHECKPOINT_PATH.resolve(),
            STAGE_40.DEFAULT_TREATMENT_BEST_CHECKPOINT_PATH.resolve(),
            STAGE_40.DEFAULT_TREATMENT_FINAL_CHECKPOINT_PATH.resolve(),
        }
        self.assertEqual(len(default_paths), 7)

    def test_every_artifact_and_corpus_path_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            keys = (
                "source_checkpoint_path",
                "stage_38_checkpoint_path",
                "control_best_checkpoint_path",
                "control_final_checkpoint_path",
                "treatment_best_checkpoint_path",
                "treatment_final_checkpoint_path",
                "source_data_path",
            )
            distinct = {
                key: root / f"{index}_{key}" for index, key in enumerate(keys)
            }
            STAGE_40.validate_artifact_paths(**distinct)

            for first_key, second_key in combinations(keys, 2):
                with self.subTest(first=first_key, second=second_key):
                    collided = dict(distinct)
                    collided[second_key] = collided[first_key]
                    with self.assertRaisesRegex(ValueError, "distinct paths"):
                        STAGE_40.validate_artifact_paths(**collided)

    def test_fresh_paired_statistics_batches_and_caller_rng_are_preserved(self) -> None:
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=3e-5,
            max_iters=32_001,
            eval_interval=1,
            eval_iters=1,
            seed=301,
        )
        control = PairedRecordingModel(loss_offset=0.0)
        candidate = PairedRecordingModel(loss_offset=-0.25)
        control.train()
        candidate.eval()

        def seeded_batch(
            data: object,
            split: str,
            batch_config: TrainingConfig,
            generator: torch.Generator,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del data, batch_config
            self.assertEqual(split, "val")
            values = torch.randint(
                0,
                20,
                (2, 3),
                generator=generator,
                device=device,
            )
            return values, values.clone()

        torch.manual_seed(302)
        caller_rng_state = torch.get_rng_state().clone()
        with patch.object(STAGE_40, "get_stage_40_batch", side_effect=seeded_batch):
            losses, final_generator_hash = STAGE_40.evaluate_fresh_paired_models(
                (control, candidate),
                object(),
                config,
                torch.device("cpu"),
                eval_iters=5,
                seed=303,
                fixed_panel_seed=304,
            )

        self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng_state))
        self.assertTrue(control.training)
        self.assertFalse(candidate.training)
        self.assertEqual(len(control.inputs), 5)
        self.assertEqual(len(candidate.inputs), 5)
        for control_inputs, candidate_inputs in zip(
            control.inputs, candidate.inputs, strict=True
        ):
            torch.testing.assert_close(
                control_inputs,
                candidate_inputs,
                rtol=0,
                atol=0,
            )
        for control_loss, candidate_loss in zip(*losses, strict=True):
            self.assertAlmostEqual(candidate_loss - control_loss, -0.25)
        self.assertEqual(len(final_generator_hash), 64)

        baseline = STAGE_40.FreshCheckpointResult(
            name="25k",
            checkpoint_path=Path("25k.pt"),
            checkpoint_sha256="2" * 64,
            checkpoint_step=25_000,
            fixed_panel_loss=1.4,
            mean_loss=1.0,
            standard_error=0.0,
            batch_losses=(1.0, 1.0, 1.0),
        )
        treatment = STAGE_40.FreshCheckpointResult(
            name="32k",
            checkpoint_path=Path("32k.pt"),
            checkpoint_sha256="3" * 64,
            checkpoint_step=32_000,
            fixed_panel_loss=1.3,
            mean_loss=1.2,
            standard_error=0.0,
            batch_losses=(0.9, 1.2, 1.5),
        )
        delta = STAGE_40._fresh_paired_delta(treatment, baseline)
        expected_standard_error = 0.3 / math.sqrt(3)
        expected_margin = 4.302652730 * expected_standard_error
        self.assertAlmostEqual(delta.mean_delta, 0.2)
        self.assertAlmostEqual(delta.standard_error, expected_standard_error)
        self.assertAlmostEqual(delta.critical_value, 4.302652730)
        self.assertAlmostEqual(delta.confidence_low, 0.2 - expected_margin)
        self.assertAlmostEqual(delta.confidence_high, 0.2 + expected_margin)

    def test_train_branch_uses_absolute_steps_and_branch_lr_only(self) -> None:
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=STAGE_40.DEFAULT_SOURCE_LEARNING_RATE,
            max_iters=32_002,
            eval_interval=1,
            eval_iters=1,
            seed=311,
        )
        dummy_batch = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
        )

        for learning_rate in (3e-5, 1e-5):
            with self.subTest(learning_rate=learning_rate):
                model = ScalarTrainingModel()
                optimizer = RecordingOptimizer(model)
                training_generator = torch.Generator().manual_seed(312)

                def fixed_evaluator(*args: object, **kwargs: object) -> object:
                    del args, kwargs
                    completed = float(model.completed_updates)
                    return STAGE_40.LossEstimate(
                        train=completed,
                        val=10.0 + completed,
                    )

                with (
                    patch.object(
                        STAGE_40,
                        "evaluate_on_fixed_batches",
                        side_effect=fixed_evaluator,
                    ),
                    patch.object(
                        STAGE_40,
                        "get_stage_40_batch",
                        return_value=dummy_batch,
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = STAGE_40.train_branch(
                        model,
                        optimizer,
                        object(),
                        config,
                        torch.device("cpu"),
                        training_generator,
                        STAGE_40.LateLearningRateSchedule(learning_rate),
                        start_step=32_000,
                        fixed_panel_seed=313,
                        on_evaluation=lambda record: None,
                    )

                self.assertEqual(
                    tuple(record.step for record in result.history),
                    (32_000, 32_001, 32_002),
                )
                self.assertEqual(
                    tuple(record.losses.train for record in result.history),
                    (0.0, 1.0, 2.0),
                )
                self.assertEqual(
                    optimizer.learning_rates,
                    [learning_rate, learning_rate],
                )

    def test_independent_full_state_restores_have_equal_fork_hashes_and_lr_only_delta(
        self,
    ) -> None:
        device = torch.device("cpu")
        torch.manual_seed(321)
        source_model = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(
            source_model.parameters(),
            lr=STAGE_40.DEFAULT_SOURCE_LEARNING_RATE,
        )
        inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        source_model(inputs).square().mean().backward()
        source_optimizer.step()
        source_generator = torch.Generator().manual_seed(322)
        torch.randint(0, 10, (7,), generator=source_generator)
        torch.manual_seed(323)
        source_rng_state = STAGE_40.capture_rng_state(device)
        checkpoint = {
            "model_state_dict": copy.deepcopy(source_model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(source_optimizer.state_dict()),
            "training_generator_state": source_generator.get_state().clone(),
            "rng_state": copy.deepcopy(source_rng_state),
        }
        source_optimizer_hash = STAGE_40.structured_state_sha256(
            checkpoint["optimizer_state_dict"]
        )
        source = STAGE_40.Stage39Source(
            path=Path("synthetic_stage_39.pt"),
            sha256="9" * 64,
            checkpoint=checkpoint,
            stage_38_source=make_stage_38_source(),
            step=32_000,
            best_step=32_000,
            best_train_loss=1.2,
            best_val_loss=1.4,
            snapshot_train_loss=1.2,
            snapshot_val_loss=1.4,
            runtime_device_type="cpu",
            runtime_device_name="cpu",
            torch_version=str(torch.__version__),
            canonical=False,
        )

        restored: list[
            tuple[
                torch.nn.Linear,
                torch.optim.AdamW,
                torch.Generator,
                tuple[str, str, str, str],
            ]
        ] = []
        for seed in (324, 325):
            torch.manual_seed(seed)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
            generator = torch.Generator().manual_seed(seed)
            torch.rand(11)
            with patch.object(STAGE_40, "verify_model_masks"):
                STAGE_40.restore_stage_39_source(
                    source,
                    model,
                    optimizer,
                    generator,
                    device,
                )
            hashes = (
                STAGE_40.tensor_mapping_sha256(model.state_dict()),
                STAGE_40.structured_state_sha256(optimizer.state_dict()),
                STAGE_40.tensor_sha256(generator.get_state()),
                STAGE_40.structured_state_sha256(
                    STAGE_40.capture_rng_state(device)
                ),
            )
            restored.append((model, optimizer, generator, hashes))

        self.assertEqual(restored[0][3], restored[1][3])
        model_a, optimizer_a, generator_a, fork_hashes = restored[0]
        model_b, optimizer_b, generator_b, _ = restored[1]
        fork_lineage = STAGE_40.optimizer_lineage_sha256(
            optimizer_a.state_dict()
        )
        self.assertEqual(
            fork_lineage,
            STAGE_40.optimizer_lineage_sha256(optimizer_b.state_dict()),
        )

        STAGE_40._set_optimizer_learning_rate(optimizer_a, 3e-5)
        STAGE_40._set_optimizer_learning_rate(optimizer_b, 1e-5)
        self.assertEqual(
            STAGE_40.optimizer_lineage_sha256(optimizer_a.state_dict()),
            STAGE_40.optimizer_lineage_sha256(optimizer_b.state_dict()),
        )
        self.assertNotEqual(
            STAGE_40.structured_state_sha256(optimizer_a.state_dict()),
            STAGE_40.structured_state_sha256(optimizer_b.state_dict()),
        )
        self.assertEqual(
            STAGE_40.tensor_mapping_sha256(model_a.state_dict()),
            fork_hashes[0],
        )
        self.assertEqual(
            STAGE_40.tensor_mapping_sha256(model_b.state_dict()),
            fork_hashes[0],
        )
        self.assertEqual(
            STAGE_40.tensor_sha256(generator_a.get_state()),
            STAGE_40.tensor_sha256(generator_b.get_state()),
        )
        optimizer_a.zero_grad(set_to_none=True)
        model_a(inputs).square().mean().backward()
        optimizer_a.step()
        self.assertEqual(
            STAGE_40.structured_state_sha256(
                source.checkpoint["optimizer_state_dict"]
            ),
            source_optimizer_hash,
        )

    def test_canonical_stage_39_source_rejects_cpu_runtime(self) -> None:
        source_path = STAGE_40.DEFAULT_SOURCE_CHECKPOINT_PATH
        if not source_path.is_file():
            self.skipTest("ignored canonical Stage 39 artifact is unavailable")
        self.assertEqual(
            STAGE_40._sha256_file(source_path),
            STAGE_40.DEFAULT_SOURCE_CHECKPOINT_SHA256,
        )
        data = CharacterData.from_file(STAGE_40.DEFAULT_COMPLETE_DATA_PATH)
        config = make_canonical_config()
        with self.assertRaisesRegex(ValueError, "recorded device/backend"):
            STAGE_40.validate_and_load_stage_39_source(
                source_path,
                data,
                config,
                STAGE_40.DEFAULT_N_HEAD,
                STAGE_40.DEFAULT_N_LAYER,
                torch.device("cpu"),
                training_batch_seed=STAGE_40.DEFAULT_TRAINING_BATCH_SEED,
                fixed_panel_seed=STAGE_40.DEFAULT_FIXED_PANEL_SEED,
                training_rng_seed=STAGE_40.DEFAULT_TRAINING_RNG_SEED,
            )

    def test_checkpoint_round_trip_for_best_and_final_branch_states(self) -> None:
        data = make_small_data()
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=STAGE_40.DEFAULT_SOURCE_LEARNING_RATE,
            max_iters=32_001,
            eval_interval=1,
            eval_iters=1,
            seed=331,
        )
        device = torch.device("cpu")
        runtime_type, runtime_name = STAGE_40._runtime_identity(device)
        torch.manual_seed(config.seed)
        model = STAGE_40.build_model(data, config, 2, 6, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=STAGE_40.DEFAULT_SOURCE_LEARNING_RATE,
        )
        inputs = torch.zeros((2, 24), dtype=torch.long)
        _, loss = model(inputs, inputs)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertEqual(len(optimizer.state), len(tuple(model.parameters())))

        def set_optimizer_step(step: int, learning_rate: float) -> None:
            STAGE_40._set_optimizer_learning_rate(optimizer, learning_rate)
            for state in optimizer.state.values():
                state["step"].fill_(step)

        training_batch_seed = 332
        fixed_panel_seed = 333
        training_rng_seed = 334
        training_generator = torch.Generator()
        training_generator.set_state(
            STAGE_40._expected_training_generator_state(
                data,
                config,
                training_batch_seed=training_batch_seed,
                completed_steps=32_000,
            )
        )
        set_optimizer_step(32_000, STAGE_40.DEFAULT_SOURCE_LEARNING_RATE)
        source_checkpoint = {
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "training_generator_state": training_generator.get_state().clone(),
            "rng_state": copy.deepcopy(STAGE_40.capture_rng_state(device)),
        }
        source = STAGE_40.Stage39Source(
            path=Path("synthetic_stage_39.pt"),
            sha256="9" * 64,
            checkpoint=source_checkpoint,
            stage_38_source=make_stage_38_source(),
            step=32_000,
            best_step=32_000,
            best_train_loss=1.2,
            best_val_loss=1.4,
            snapshot_train_loss=1.2,
            snapshot_val_loss=1.4,
            runtime_device_type=runtime_type,
            runtime_device_name=runtime_name,
            torch_version=str(torch.__version__),
            canonical=False,
        )

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            branch = STAGE_40.BranchSpec(
                key="treatment",
                name="B: refine at 1e-5",
                learning_rate=1e-5,
                best_checkpoint_path=root / "best.pt",
                final_checkpoint_path=root / "final.pt",
            )
            set_optimizer_step(32_000, STAGE_40.DEFAULT_SOURCE_LEARNING_RATE)
            fork_model_hash = STAGE_40.tensor_mapping_sha256(model.state_dict())
            fork_optimizer_hash = STAGE_40.structured_state_sha256(
                optimizer.state_dict()
            )
            fork_generator_hash = STAGE_40.tensor_sha256(
                training_generator.get_state()
            )
            fork_rng_hash = STAGE_40.structured_state_sha256(
                STAGE_40.capture_rng_state(device)
            )
            STAGE_40._set_optimizer_learning_rate(optimizer, branch.learning_rate)

            tracker = STAGE_40.Stage40ValidationCheckpoint(
                model=model,
                optimizer=optimizer,
                training_generator=training_generator,
                path=branch.best_checkpoint_path,
                device=device,
                config=config,
                n_head=2,
                n_layer=6,
                data_fingerprint=STAGE_40.fingerprint_data(data),
                optimizer_restart_step=None,
                optimizer_provenance_known=True,
                best_val_loss=source.best_val_loss,
                best_step=source.best_step,
                branch=branch,
                schedule=branch.schedule,
                target_step=config.max_iters,
                source=source,
                fresh_validation=None,
                num_characters=data.num_characters,
                train_characters=len(data.train_data),
                val_characters=len(data.val_data),
                vocabulary_size=data.vocabulary.size,
                train_fraction=len(data.train_data) / data.num_characters,
                targets_per_update=config.batch_size * config.block_size,
                training_batch_seed=training_batch_seed,
                fixed_panel_seed=fixed_panel_seed,
                training_rng_seed=training_rng_seed,
                parameter_count=sum(p.numel() for p in model.parameters()),
                initial_state_sha256=STAGE_40.tensor_mapping_sha256(
                    model.state_dict()
                ),
                best_train_loss=source.best_train_loss,
                source_audit_verified=True,
                fork_model_sha256=fork_model_hash,
                fork_optimizer_sha256=fork_optimizer_hash,
                fork_batch_generator_sha256=fork_generator_hash,
                fork_rng_state_sha256=fork_rng_hash,
            )
            tracker.save(32_000)
            best_payload = STAGE_40.validate_stage_40_checkpoint(
                branch.best_checkpoint_path,
                data,
                config,
                2,
                6,
                source,
                branch,
                None,
                expected_kind="best",
                training_batch_seed=training_batch_seed,
                fixed_panel_seed=fixed_panel_seed,
                training_rng_seed=training_rng_seed,
            )
            self.assertEqual(best_payload["step"], 32_000)
            self.assertEqual(
                best_payload["experiment"][
                    "only_intended_post_fork_difference"
                ],
                "learning_rate",
            )
            self.assertEqual(
                best_payload["fork_state"]["optimizer_sha256"],
                fork_optimizer_hash,
            )

            set_optimizer_step(32_001, branch.learning_rate)
            training_generator.set_state(
                STAGE_40._expected_training_generator_state(
                    data,
                    config,
                    training_batch_seed=training_batch_seed,
                    completed_steps=32_001,
                )
            )
            tracker.save_final(
                branch.final_checkpoint_path,
                STAGE_40.EvaluationRecord(
                    step=32_001,
                    losses=STAGE_40.LossEstimate(train=1.1, val=1.39),
                ),
            )
            final_payload = STAGE_40.validate_stage_40_checkpoint(
                branch.final_checkpoint_path,
                data,
                config,
                2,
                6,
                source,
                branch,
                None,
                expected_kind="final",
                training_batch_seed=training_batch_seed,
                fixed_panel_seed=fixed_panel_seed,
                training_rng_seed=training_rng_seed,
            )
            self.assertEqual(final_payload["step"], 32_001)
            self.assertEqual(
                STAGE_40._optimizer_rates(final_payload["optimizer_state_dict"]),
                {branch.learning_rate},
            )


if __name__ == "__main__":
    unittest.main()
