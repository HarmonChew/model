from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("39_extended_training.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_39_extended_training_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_39 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_39
SPEC.loader.exec_module(STAGE_39)


def make_canonical_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=STAGE_39.DEFAULT_BATCH_SIZE,
        block_size=STAGE_39.DEFAULT_BLOCK_SIZE,
        n_embd=STAGE_39.DEFAULT_N_EMBD,
        learning_rate=STAGE_39.DEFAULT_CONTINUATION_LEARNING_RATE,
        max_iters=STAGE_39.DEFAULT_TARGET_STEP,
        eval_interval=STAGE_39.DEFAULT_EVAL_INTERVAL,
        eval_iters=STAGE_39.DEFAULT_EVAL_ITERS,
        seed=STAGE_39.DEFAULT_SEED,
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


class RecordingCheckpoint:
    def __init__(
        self,
        *,
        target_step: int,
        training_generator: torch.Generator,
    ) -> None:
        self.target_step = target_step
        self.training_generator = training_generator
        self.best_step = STAGE_39.DEFAULT_SOURCE_STEP
        self.best_val_loss = 2.0
        self.source_audit_verified = False
        self.snapshots: list[tuple[str, Path | None, int]] = []

    def consider(self, record: object) -> bool:
        del record
        return False

    def save(self, step: int) -> None:
        self.snapshots.append(("best", None, step))

    def save_final(self, path: Path, record: object) -> None:
        self.snapshots.append(("final", path, record.step))


class Stage39ExtendedTrainingTests(unittest.TestCase):
    def test_locked_defaults_cli_schedule_and_distinct_artifacts(self) -> None:
        self.assertEqual(STAGE_39.EXPERIMENT_STAGE, 39)
        self.assertEqual(
            (
                STAGE_39.DEFAULT_BATCH_SIZE,
                STAGE_39.DEFAULT_BLOCK_SIZE,
                STAGE_39.DEFAULT_N_EMBD,
                STAGE_39.DEFAULT_N_HEAD,
                STAGE_39.DEFAULT_HEAD_SIZE,
                STAGE_39.DEFAULT_FEED_FORWARD_WIDTH,
                STAGE_39.DEFAULT_N_LAYER,
                STAGE_39.DEFAULT_DROPOUT,
            ),
            (8, 256, 128, 4, 32, 512, 6, 0.10),
        )
        self.assertEqual(
            STAGE_39.DEFAULT_LAYER_CONTEXTS,
            (23, 23, 23, 23, None, None),
        )
        self.assertEqual(STAGE_39.DEFAULT_SOURCE_STEP, 25_000)
        self.assertEqual(STAGE_39.DEFAULT_TARGET_STEP, 32_000)
        self.assertEqual(STAGE_39.DEFAULT_CONTINUATION_LEARNING_RATE, 3e-5)
        self.assertEqual(STAGE_39.DEFAULT_EVAL_INTERVAL, 500)
        self.assertEqual(STAGE_39.DEFAULT_EVAL_ITERS, 100)
        self.assertEqual(STAGE_39.DEFAULT_TOTAL_TARGET_CHARACTERS, 65_536_000)
        self.assertEqual(STAGE_39.DEFAULT_ADDED_TARGET_CHARACTERS, 14_336_000)

        schedule = STAGE_39.ExtendedLearningRateSchedule()
        self.assertEqual(schedule.for_update(24_999), 3e-5)
        self.assertEqual(schedule.for_update(25_000), 3e-5)
        self.assertEqual(schedule.for_update(31_999), 3e-5)
        self.assertEqual(
            schedule.as_metadata(target_step=32_000)[-1],
            {
                "start_update": 18_000,
                "end_update_exclusive": 32_000,
                "learning_rate": 3e-5,
            },
        )

        with patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_39.parse_args()
        self.assertEqual(args.data, STAGE_39.DEFAULT_COMPLETE_DATA_PATH)
        self.assertEqual(
            args.source_checkpoint,
            STAGE_39.DEFAULT_SOURCE_CHECKPOINT_PATH,
        )
        self.assertEqual(args.target_step, 32_000)
        self.assertEqual(
            args.best_checkpoint_path,
            STAGE_39.DEFAULT_BEST_CHECKPOINT_PATH,
        )
        self.assertIsNone(args.final_checkpoint_path)
        self.assertEqual(
            STAGE_39._default_final_checkpoint_path(32_000),
            STAGE_39.DEFAULT_FINAL_CHECKPOINT_PATH,
        )
        self.assertIn(
            "25500",
            STAGE_39._default_final_checkpoint_path(25_500).name,
        )

        default_paths = {
            STAGE_39.DEFAULT_COMPLETE_DATA_PATH.resolve(),
            STAGE_39.DEFAULT_SOURCE_CHECKPOINT_PATH.resolve(),
            STAGE_39.DEFAULT_BEST_CHECKPOINT_PATH.resolve(),
            STAGE_39.DEFAULT_FINAL_CHECKPOINT_PATH.resolve(),
        }
        self.assertEqual(len(default_paths), 4)

    def test_every_artifact_and_corpus_path_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            keys = (
                "source_checkpoint_path",
                "best_checkpoint_path",
                "final_checkpoint_path",
                "source_data_path",
            )
            distinct = {key: root / f"{index}_{key}" for index, key in enumerate(keys)}
            STAGE_39.validate_artifact_paths(**distinct)

            for first_key, second_key in combinations(keys, 2):
                with self.subTest(first=first_key, second=second_key):
                    collided = dict(distinct)
                    collided[second_key] = collided[first_key]
                    with self.assertRaisesRegex(ValueError, "distinct paths"):
                        STAGE_39.validate_artifact_paths(**collided)

    def test_canonical_stage_38_source_rejects_cpu_runtime(self) -> None:
        source_path = STAGE_39.DEFAULT_SOURCE_CHECKPOINT_PATH
        if not source_path.is_file():
            self.skipTest("ignored canonical Stage 38 artifact is unavailable")
        self.assertEqual(
            STAGE_39._sha256_file(source_path),
            STAGE_39.DEFAULT_SOURCE_CHECKPOINT_SHA256,
        )
        data = CharacterData.from_file(STAGE_39.DEFAULT_COMPLETE_DATA_PATH)
        config = make_canonical_config()
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
        generator_state = checkpoint["training_generator_state"]
        self.assertEqual(
            STAGE_39.tensor_sha256(generator_state),
            STAGE_39.DEFAULT_SOURCE_GENERATOR_SHA256,
        )
        self.assertTrue(
            torch.equal(
                generator_state,
                STAGE_39._expected_training_generator_state(
                    data,
                    config,
                    training_batch_seed=STAGE_39.DEFAULT_TRAINING_BATCH_SEED,
                    completed_steps=STAGE_39.DEFAULT_SOURCE_STEP,
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "recorded device/backend"):
            STAGE_39.validate_and_load_stage_38_source(
                source_path,
                data,
                config,
                STAGE_39.DEFAULT_N_HEAD,
                STAGE_39.DEFAULT_N_LAYER,
                STAGE_39.ExtendedLearningRateSchedule(),
                torch.device("cpu"),
                training_batch_seed=STAGE_39.DEFAULT_TRAINING_BATCH_SEED,
                fixed_panel_seed=STAGE_39.DEFAULT_FIXED_PANEL_SEED,
                training_rng_seed=STAGE_39.DEFAULT_TRAINING_RNG_SEED,
            )

    def test_absolute_steps_constant_lr_source_audit_and_final_snapshot(self) -> None:
        schedule = STAGE_39.ExtendedLearningRateSchedule()
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=STAGE_39.DEFAULT_CONTINUATION_LEARNING_RATE,
            max_iters=25_001,
            eval_interval=500,
            eval_iters=1,
            seed=101,
        )
        model = ScalarTrainingModel()
        optimizer = RecordingOptimizer(model)
        training_generator = torch.Generator().manual_seed(102)
        best = RecordingCheckpoint(
            target_step=config.max_iters,
            training_generator=training_generator,
        )
        source = STAGE_39.Stage38Source(
            path=Path("stage38.pt"),
            sha256="a" * 64,
            checkpoint={},
            step=25_000,
            best_step=25_000,
            best_train_loss=0.0,
            best_val_loss=2.0,
            snapshot_train_loss=0.0,
            snapshot_val_loss=2.0,
            runtime_device_type="cpu",
            runtime_device_name="cpu",
            torch_version=str(torch.__version__),
            canonical=False,
        )
        final_path = Path("final.pt")
        callback, source_audits = STAGE_39.make_evaluation_callback(
            best,
            source,
            final_checkpoint_path=final_path,
        )

        def fixed_evaluator(*args: object, **kwargs: object) -> object:
            del args, kwargs
            completed = float(model.completed_updates)
            return STAGE_39.LossEstimate(train=completed, val=2.0 + completed)

        dummy_batch = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
        )
        with (
            patch.object(
                STAGE_39,
                "evaluate_on_fixed_batches",
                side_effect=fixed_evaluator,
            ),
            patch.object(STAGE_39, "get_stage_38_batch", return_value=dummy_batch),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = STAGE_39.train_continuation(
                model,
                optimizer,
                object(),
                config,
                torch.device("cpu"),
                training_generator,
                schedule,
                start_step=25_000,
                fixed_panel_seed=103,
                on_evaluation=callback,
            )

        self.assertEqual(
            tuple(record.step for record in result.history),
            (25_000, 25_001),
        )
        self.assertEqual(
            tuple(record.losses.train for record in result.history),
            (0.0, 1.0),
        )
        self.assertEqual(optimizer.learning_rates, [3e-5])
        self.assertEqual(len(source_audits), 1)
        self.assertTrue(source_audits[0].verified)
        self.assertTrue(best.source_audit_verified)
        self.assertEqual(
            best.snapshots,
            [
                ("best", None, 25_000),
                ("final", final_path, 25_001),
            ],
        )

    def test_checkpoint_round_trip_for_best_and_final_states(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abcde")
        token_ids = torch.arange(2_400, dtype=torch.long) % vocabulary.size
        data = CharacterData(
            vocabulary=vocabulary,
            train_data=token_ids[:1_800],
            val_data=token_ids[1_800:],
            num_characters=len(token_ids),
        )
        schedule = STAGE_39.ExtendedLearningRateSchedule()
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=STAGE_39.DEFAULT_CONTINUATION_LEARNING_RATE,
            max_iters=25_001,
            eval_interval=1,
            eval_iters=1,
            seed=211,
        )
        device = torch.device("cpu")
        torch.manual_seed(config.seed)
        model = STAGE_39.build_model(data, config, 2, 6, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=STAGE_39.DEFAULT_CONTINUATION_LEARNING_RATE,
        )
        inputs = torch.zeros((2, 24), dtype=torch.long)
        _, loss = model(inputs, inputs)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertEqual(len(optimizer.state), len(tuple(model.parameters())))

        def set_optimizer_step(step: int) -> None:
            STAGE_39._set_optimizer_learning_rate(
                optimizer,
                STAGE_39.DEFAULT_CONTINUATION_LEARNING_RATE,
            )
            for state in optimizer.state.values():
                state["step"].fill_(step)

        training_generator = torch.Generator()
        training_generator.set_state(
            STAGE_39._expected_training_generator_state(
                data,
                config,
                training_batch_seed=212,
                completed_steps=25_000,
            )
        )
        source = STAGE_39.Stage38Source(
            path=Path("synthetic_stage_38.pt"),
            sha256="a" * 64,
            checkpoint={},
            step=25_000,
            best_step=25_000,
            best_train_loss=1.5,
            best_val_loss=1.6,
            snapshot_train_loss=1.5,
            snapshot_val_loss=1.6,
            runtime_device_type="cpu",
            runtime_device_name="cpu",
            torch_version=str(torch.__version__),
            canonical=False,
        )

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            best_path = root / "best.pt"
            final_path = root / "final.pt"
            tracker = STAGE_39.Stage39ValidationCheckpoint(
                model=model,
                optimizer=optimizer,
                training_generator=training_generator,
                path=best_path,
                device=device,
                config=config,
                n_head=2,
                n_layer=6,
                data_fingerprint=STAGE_39.fingerprint_data(data),
                optimizer_restart_step=None,
                optimizer_provenance_known=True,
                best_val_loss=source.best_val_loss,
                best_step=source.best_step,
                schedule=schedule,
                target_step=config.max_iters,
                source=source,
                num_characters=data.num_characters,
                train_characters=len(data.train_data),
                val_characters=len(data.val_data),
                vocabulary_size=data.vocabulary.size,
                train_fraction=len(data.train_data) / data.num_characters,
                targets_per_update=config.batch_size * config.block_size,
                training_batch_seed=212,
                fixed_panel_seed=213,
                training_rng_seed=214,
                parameter_count=sum(p.numel() for p in model.parameters()),
                initial_state_sha256=STAGE_39.tensor_mapping_sha256(model.state_dict()),
                best_train_loss=source.best_train_loss,
                source_audit_verified=True,
            )
            set_optimizer_step(25_000)
            tracker.save(25_000)
            STAGE_39.validate_stage_39_checkpoint(
                best_path,
                data,
                config,
                2,
                6,
                schedule,
                source,
                expected_kind="best",
                training_batch_seed=212,
                fixed_panel_seed=213,
                training_rng_seed=214,
            )

            set_optimizer_step(25_001)
            training_generator.set_state(
                STAGE_39._expected_training_generator_state(
                    data,
                    config,
                    training_batch_seed=212,
                    completed_steps=25_001,
                )
            )
            tracker.save_final(
                final_path,
                STAGE_39.EvaluationRecord(
                    step=25_001,
                    losses=STAGE_39.LossEstimate(train=1.4, val=1.65),
                ),
            )
            STAGE_39.validate_stage_39_checkpoint(
                final_path,
                data,
                config,
                2,
                6,
                schedule,
                source,
                expected_kind="final",
                training_batch_seed=212,
                fixed_panel_seed=213,
                training_rng_seed=214,
            )


if __name__ == "__main__":
    unittest.main()
