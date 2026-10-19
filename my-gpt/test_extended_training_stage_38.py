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


MODULE_PATH = Path(__file__).with_name("38_extended_training.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_38_extended_training_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_38 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_38
SPEC.loader.exec_module(STAGE_38)


def make_canonical_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=STAGE_38.DEFAULT_BATCH_SIZE,
        block_size=STAGE_38.DEFAULT_BLOCK_SIZE,
        n_embd=STAGE_38.DEFAULT_N_EMBD,
        learning_rate=STAGE_38.DEFAULT_EXTENSION_LEARNING_RATE,
        max_iters=STAGE_38.DEFAULT_TARGET_STEP,
        eval_interval=STAGE_38.DEFAULT_EVAL_INTERVAL,
        eval_iters=STAGE_38.DEFAULT_EVAL_ITERS,
        seed=STAGE_38.DEFAULT_SEED,
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
        self.best_step = STAGE_38.DEFAULT_SOURCE_STEP
        self.best_val_loss = 1.0
        self.replay_verified = False
        self.snapshots: list[tuple[Path, int, str]] = []
        self.saved_steps: list[int] = []

    def consider(self, record: object) -> bool:
        del record
        return False

    def save(self, step: int) -> None:
        self.saved_steps.append(step)

    def save_snapshot(
        self,
        path: Path,
        record: object,
        *,
        checkpoint_kind: str,
    ) -> None:
        self.snapshots.append((path, record.step, checkpoint_kind))


class Stage38ExtendedTrainingTests(unittest.TestCase):
    def test_locked_defaults_cli_and_distinct_default_artifacts(self) -> None:
        self.assertEqual(STAGE_38.EXPERIMENT_STAGE, 38)
        self.assertEqual(
            (
                STAGE_38.DEFAULT_BATCH_SIZE,
                STAGE_38.DEFAULT_BLOCK_SIZE,
                STAGE_38.DEFAULT_N_EMBD,
                STAGE_38.DEFAULT_N_HEAD,
                STAGE_38.DEFAULT_HEAD_SIZE,
                STAGE_38.DEFAULT_FEED_FORWARD_WIDTH,
                STAGE_38.DEFAULT_N_LAYER,
                STAGE_38.DEFAULT_DROPOUT,
            ),
            (8, 256, 128, 4, 32, 512, 6, 0.10),
        )
        self.assertEqual(
            STAGE_38.DEFAULT_LAYER_CONTEXTS,
            (23, 23, 23, 23, None, None),
        )
        # The CUDA Stage 37 reproduction selected its best at the final
        # evaluation, so the authenticated source already sits on the stage-38
        # reconstruction boundary and no updates are replayed.
        self.assertEqual(STAGE_38.DEFAULT_SOURCE_STEP, 18_000)
        self.assertEqual(STAGE_38.DEFAULT_EXTENSION_START_STEP, 18_000)
        self.assertEqual(STAGE_38.DEFAULT_TARGET_STEP, 25_000)
        self.assertEqual(STAGE_38.DEFAULT_EXTENSION_LEARNING_RATE, 3e-5)
        self.assertEqual(STAGE_38.DEFAULT_EVAL_INTERVAL, 500)
        self.assertEqual(STAGE_38.DEFAULT_EVAL_ITERS, 100)
        self.assertEqual(STAGE_38.DEFAULT_TOTAL_TARGET_CHARACTERS, 51_200_000)
        self.assertEqual(STAGE_38.DEFAULT_ADDED_TARGET_CHARACTERS, 14_336_000)

        with patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_38.parse_args()
        self.assertEqual(args.data, STAGE_38.DEFAULT_COMPLETE_DATA_PATH)
        self.assertEqual(
            args.source_checkpoint,
            STAGE_38.DEFAULT_SOURCE_CHECKPOINT_PATH,
        )
        self.assertEqual(args.target_step, 25_000)
        self.assertEqual(args.extension_learning_rate, 3e-5)
        self.assertEqual(
            args.best_checkpoint_path,
            STAGE_38.DEFAULT_BEST_CHECKPOINT_PATH,
        )
        self.assertEqual(
            args.replay_checkpoint_path,
            STAGE_38.DEFAULT_REPLAY_CHECKPOINT_PATH,
        )
        self.assertIsNone(args.final_checkpoint_path)
        self.assertEqual(
            STAGE_38._default_final_checkpoint_path(25_000),
            STAGE_38.DEFAULT_FINAL_CHECKPOINT_PATH,
        )
        self.assertIn(
            "24000",
            STAGE_38._default_final_checkpoint_path(24_000).name,
        )

        default_paths = {
            STAGE_38.DEFAULT_COMPLETE_DATA_PATH.resolve(),
            STAGE_38.DEFAULT_SOURCE_CHECKPOINT_PATH.resolve(),
            STAGE_38.DEFAULT_BEST_CHECKPOINT_PATH.resolve(),
            STAGE_38.DEFAULT_REPLAY_CHECKPOINT_PATH.resolve(),
            STAGE_38.DEFAULT_FINAL_CHECKPOINT_PATH.resolve(),
        }
        self.assertEqual(len(default_paths), 5)

    def test_four_phase_schedule_boundary_and_metadata(self) -> None:
        schedule = STAGE_38.ExtendedLearningRateSchedule()
        expected_rates = {
            0: 1e-3,
            9_999: 1e-3,
            10_000: 3e-4,
            12_999: 3e-4,
            13_000: 1e-4,
            17_999: 1e-4,
            18_000: 3e-5,
            24_999: 3e-5,
        }
        for update_index, expected in expected_rates.items():
            with self.subTest(update_index=update_index):
                self.assertEqual(schedule.for_update(update_index), expected)

        self.assertEqual(
            schedule.as_metadata(target_step=25_000),
            [
                {
                    "start_update": 0,
                    "end_update_exclusive": 10_000,
                    "learning_rate": 1e-3,
                },
                {
                    "start_update": 10_000,
                    "end_update_exclusive": 13_000,
                    "learning_rate": 3e-4,
                },
                {
                    "start_update": 13_000,
                    "end_update_exclusive": 18_000,
                    "learning_rate": 1e-4,
                },
                {
                    "start_update": 18_000,
                    "end_update_exclusive": 25_000,
                    "learning_rate": 3e-5,
                },
            ],
        )
        self.assertEqual(
            schedule.stage_37_schedule().as_metadata(max_iters=18_000),
            schedule.as_metadata(target_step=25_000)[:3],
        )

        shorter_schedule = STAGE_38.ExtendedLearningRateSchedule(
            extension_learning_rate=2e-5,
        )
        shorter_config = TrainingConfig(
            batch_size=STAGE_38.DEFAULT_BATCH_SIZE,
            block_size=STAGE_38.DEFAULT_BLOCK_SIZE,
            n_embd=STAGE_38.DEFAULT_N_EMBD,
            learning_rate=shorter_schedule.extension_learning_rate,
            max_iters=24_000,
            eval_interval=STAGE_38.DEFAULT_EVAL_INTERVAL,
            eval_iters=STAGE_38.DEFAULT_EVAL_ITERS,
            seed=STAGE_38.DEFAULT_SEED,
        )
        with patch.object(
            STAGE_38,
            "fingerprint_data",
            return_value=STAGE_38.DEFAULT_COMPLETE_DATA_FINGERPRINT,
        ):
            self.assertTrue(
                STAGE_38.is_canonical_stage_37_source_contract(
                    object(),
                    shorter_config,
                    STAGE_38.DEFAULT_N_HEAD,
                    STAGE_38.DEFAULT_N_LAYER,
                    shorter_schedule,
                )
            )
            self.assertFalse(
                STAGE_38.is_canonical_protocol(
                    object(),
                    shorter_config,
                    STAGE_38.DEFAULT_N_HEAD,
                    STAGE_38.DEFAULT_N_LAYER,
                    shorter_schedule,
                )
            )

    def test_every_artifact_and_corpus_path_must_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            keys = (
                "source_checkpoint_path",
                "best_checkpoint_path",
                "replay_checkpoint_path",
                "final_checkpoint_path",
                "source_data_path",
            )
            distinct = {
                key: root / f"{index}_{key}"
                for index, key in enumerate(keys)
            }
            STAGE_38.validate_artifact_paths(**distinct)

            for first_key, second_key in combinations(keys, 2):
                with self.subTest(first=first_key, second=second_key):
                    collided = dict(distinct)
                    collided[second_key] = collided[first_key]
                    with self.assertRaisesRegex(ValueError, "distinct paths"):
                        STAGE_38.validate_artifact_paths(**collided)

    def test_canonical_stage_37_source_rejects_cpu_runtime(self) -> None:
        source_path = STAGE_38.DEFAULT_SOURCE_CHECKPOINT_PATH
        if not source_path.is_file():
            self.skipTest("ignored canonical Stage 37 artifact is unavailable")
        self.assertEqual(
            STAGE_38._sha256_file(source_path),
            STAGE_38.DEFAULT_SOURCE_CHECKPOINT_SHA256,
        )
        data = CharacterData.from_file(STAGE_38.DEFAULT_COMPLETE_DATA_PATH)
        config = make_canonical_config()
        with self.assertRaisesRegex(ValueError, "recorded device/backend"):
            STAGE_38.validate_and_load_stage_37_source(
                source_path,
                data,
                config,
                STAGE_38.DEFAULT_N_HEAD,
                STAGE_38.DEFAULT_N_LAYER,
                STAGE_38.ExtendedLearningRateSchedule(),
                torch.device("cpu"),
                training_batch_seed=STAGE_38.DEFAULT_TRAINING_BATCH_SEED,
                fixed_panel_seed=STAGE_38.DEFAULT_FIXED_PANEL_SEED,
                training_rng_seed=STAGE_38.DEFAULT_TRAINING_RNG_SEED,
            )

    def test_absolute_steps_lr_boundary_and_unconditional_snapshots(self) -> None:
        schedule = STAGE_38.ExtendedLearningRateSchedule()
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=schedule.extension_learning_rate,
            max_iters=18_001,
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
        source = STAGE_38.Stage37Source(
            path=Path("stage37.pt"),
            sha256="a" * 64,
            checkpoint={},
            step=17_500,
            best_step=17_500,
            best_train_loss=1.0,
            best_val_loss=1.0,
            runtime_device_type="cpu",
            runtime_device_name="cpu",
            torch_version=str(torch.__version__),
            canonical=False,
        )
        replay_path = Path("replayed.pt")
        final_path = Path("final.pt")
        callback, replay_audits = STAGE_38.make_evaluation_callback(
            best,
            schedule,
            source,
            replay_checkpoint_path=replay_path,
            final_checkpoint_path=final_path,
        )

        def fixed_evaluator(*args: object, **kwargs: object) -> object:
            del args, kwargs
            completed = float(model.completed_updates)
            return STAGE_38.LossEstimate(train=completed, val=2.0 + completed)

        dummy_batch = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
        )
        with (
            patch.object(
                STAGE_38,
                "evaluate_on_fixed_batches",
                side_effect=fixed_evaluator,
            ),
            patch.object(STAGE_38, "get_stage_37_batch", return_value=dummy_batch),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = STAGE_38.train_extended(
                model,
                optimizer,
                object(),
                config,
                torch.device("cpu"),
                training_generator,
                schedule,
                start_step=17_500,
                fixed_panel_seed=103,
                on_evaluation=callback,
            )

        self.assertEqual(
            tuple(record.step for record in result.history),
            (17_500, 18_000, 18_001),
        )
        self.assertEqual(
            tuple(record.losses.train for record in result.history),
            (0.0, 500.0, 501.0),
        )
        self.assertEqual(len(optimizer.learning_rates), 501)
        learning_rate_by_update = dict(
            zip(
                range(17_500, 18_001),
                optimizer.learning_rates,
                strict=True,
            )
        )
        self.assertEqual(learning_rate_by_update[17_999], 1e-4)
        self.assertEqual(learning_rate_by_update[18_000], 3e-5)
        self.assertTrue(all(rate == 1e-4 for rate in optimizer.learning_rates[:500]))
        self.assertEqual(optimizer.learning_rates[-1], 3e-5)

        self.assertEqual(len(replay_audits), 1)
        self.assertEqual(replay_audits[0].record.step, 18_000)
        self.assertTrue(replay_audits[0].verified)
        self.assertTrue(best.replay_verified)
        self.assertEqual(
            best.snapshots,
            [
                (replay_path, 18_000, "reconstructed_18000"),
                (final_path, 18_001, "final"),
            ],
        )

    def test_checkpoint_round_trip_for_best_replay_and_final_states(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abcde")
        token_ids = torch.arange(2_400, dtype=torch.long) % vocabulary.size
        data = CharacterData(
            vocabulary=vocabulary,
            train_data=token_ids[:1_800],
            val_data=token_ids[1_800:],
            num_characters=len(token_ids),
        )
        schedule = STAGE_38.ExtendedLearningRateSchedule()
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=schedule.extension_learning_rate,
            max_iters=18_001,
            eval_interval=1,
            eval_iters=1,
            seed=211,
        )
        device = torch.device("cpu")
        torch.manual_seed(config.seed)
        model = STAGE_38.build_model(data, config, 2, 6, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        inputs = torch.zeros((2, 24), dtype=torch.long)
        _, loss = model(inputs, inputs)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertEqual(len(optimizer.state), len(tuple(model.parameters())))

        def set_optimizer_step(step: int, learning_rate: float) -> None:
            STAGE_38._set_optimizer_learning_rate(optimizer, learning_rate)
            for state in optimizer.state.values():
                state["step"].fill_(step)

        training_generator = torch.Generator()
        training_generator.set_state(
            STAGE_38._expected_training_generator_state(
                data,
                config,
                training_batch_seed=212,
                completed_steps=17_500,
            )
        )
        source = STAGE_38.Stage37Source(
            path=Path("synthetic_stage_37.pt"),
            sha256="a" * 64,
            checkpoint={},
            step=17_500,
            best_step=17_500,
            best_train_loss=1.5,
            best_val_loss=1.6,
            runtime_device_type="cpu",
            runtime_device_name="cpu",
            torch_version=str(torch.__version__),
            canonical=False,
        )

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            best_path = root / "best.pt"
            replay_path = root / "replay.pt"
            final_path = root / "final.pt"
            tracker = STAGE_38.Stage38ValidationCheckpoint(
                model=model,
                optimizer=optimizer,
                training_generator=training_generator,
                path=best_path,
                device=device,
                config=config,
                n_head=2,
                n_layer=6,
                data_fingerprint=STAGE_38.fingerprint_data(data),
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
                initial_state_sha256=STAGE_38.tensor_mapping_sha256(
                    model.state_dict()
                ),
                best_train_loss=source.best_train_loss,
            )
            set_optimizer_step(17_500, 1e-4)
            tracker.save(17_500)
            STAGE_38.validate_stage_38_checkpoint(
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

            set_optimizer_step(18_000, 1e-4)
            training_generator.set_state(
                STAGE_38._expected_training_generator_state(
                    data,
                    config,
                    training_batch_seed=212,
                    completed_steps=18_000,
                )
            )
            tracker.replay_verified = True
            tracker.save_snapshot(
                replay_path,
                STAGE_38.EvaluationRecord(
                    step=18_000,
                    losses=STAGE_38.LossEstimate(train=1.4, val=1.5),
                ),
                checkpoint_kind="reconstructed_18000",
            )
            STAGE_38.validate_stage_38_checkpoint(
                replay_path,
                data,
                config,
                2,
                6,
                schedule,
                source,
                expected_kind="reconstructed_18000",
                training_batch_seed=212,
                fixed_panel_seed=213,
                training_rng_seed=214,
            )

            set_optimizer_step(18_001, 3e-5)
            training_generator.set_state(
                STAGE_38._expected_training_generator_state(
                    data,
                    config,
                    training_batch_seed=212,
                    completed_steps=18_001,
                )
            )
            tracker.save_snapshot(
                final_path,
                STAGE_38.EvaluationRecord(
                    step=18_001,
                    losses=STAGE_38.LossEstimate(train=1.3, val=1.55),
                ),
                checkpoint_kind="final",
            )
            STAGE_38.validate_stage_38_checkpoint(
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
