from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("41_final_compute_extension.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_41_final_compute_extension_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_41 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_41
SPEC.loader.exec_module(STAGE_41)


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
        self.device = torch.device("cpu")
        self.best_step = STAGE_41.DEFAULT_SOURCE_STEP
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


class Stage41FinalComputeExtensionTests(unittest.TestCase):
    def test_locked_defaults_cli_schedule_and_distinct_default_paths(self) -> None:
        self.assertEqual(STAGE_41.EXPERIMENT_STAGE, 41)
        self.assertEqual(
            (
                STAGE_41.DEFAULT_BATCH_SIZE,
                STAGE_41.DEFAULT_BLOCK_SIZE,
                STAGE_41.DEFAULT_N_EMBD,
                STAGE_41.DEFAULT_N_HEAD,
                STAGE_41.DEFAULT_HEAD_SIZE,
                STAGE_41.DEFAULT_FEED_FORWARD_WIDTH,
                STAGE_41.DEFAULT_N_LAYER,
                STAGE_41.DEFAULT_DROPOUT,
            ),
            (8, 256, 128, 4, 32, 512, 6, 0.10),
        )
        self.assertEqual(
            STAGE_41.DEFAULT_LAYER_CONTEXTS,
            (23, 23, 23, 23, None, None),
        )
        self.assertEqual(STAGE_41.DEFAULT_SOURCE_STEP, 38_000)
        self.assertEqual(STAGE_41.DEFAULT_TARGET_STEP, 44_000)
        self.assertEqual(STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE, 3e-5)
        self.assertEqual(STAGE_41.DEFAULT_EVAL_INTERVAL, 500)
        self.assertEqual(STAGE_41.DEFAULT_EVAL_ITERS, 100)
        self.assertEqual(STAGE_41.DEFAULT_TOTAL_TARGET_CHARACTERS, 90_112_000)
        self.assertEqual(STAGE_41.DEFAULT_ADDED_TARGET_CHARACTERS, 12_288_000)
        self.assertEqual(
            STAGE_41.DEFAULT_SOURCE_CHECKPOINT_PATH,
            STAGE_41._STAGE_40.DEFAULT_CONTROL_FINAL_CHECKPOINT_PATH,
        )

        schedule = STAGE_41.FinalComputeExtensionSchedule()
        expected_rates = {
            0: 1e-3,
            9_999: 1e-3,
            10_000: 3e-4,
            12_999: 3e-4,
            13_000: 1e-4,
            17_999: 1e-4,
            18_000: 3e-5,
            31_999: 3e-5,
            32_000: 3e-5,
            37_999: 3e-5,
            38_000: 3e-5,
            43_999: 3e-5,
        }
        for update_index, expected_rate in expected_rates.items():
            with self.subTest(update_index=update_index):
                self.assertEqual(schedule.for_update(update_index), expected_rate)
        self.assertEqual(
            schedule.as_metadata(target_step=44_000)[-1],
            {
                "start_update": 38_000,
                "end_update_exclusive": 44_000,
                "learning_rate": 3e-5,
            },
        )

        with patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_41.parse_args()
        self.assertEqual(args.data, STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
        self.assertEqual(
            args.source_checkpoint,
            STAGE_41.DEFAULT_SOURCE_CHECKPOINT_PATH,
        )
        self.assertEqual(args.target_step, 44_000)
        self.assertEqual(
            args.best_checkpoint_path,
            STAGE_41.DEFAULT_BEST_CHECKPOINT_PATH,
        )
        self.assertIsNone(args.final_checkpoint_path)
        self.assertEqual(
            STAGE_41._default_final_checkpoint_path(44_000),
            STAGE_41.DEFAULT_FINAL_CHECKPOINT_PATH,
        )
        self.assertIn(
            "43000",
            STAGE_41._default_final_checkpoint_path(43_000).name,
        )

        default_paths = {
            STAGE_41.DEFAULT_COMPLETE_DATA_PATH.resolve(),
            STAGE_41.DEFAULT_SOURCE_CHECKPOINT_PATH.resolve(),
            STAGE_41.DEFAULT_BEST_CHECKPOINT_PATH.resolve(),
            STAGE_41.DEFAULT_FINAL_CHECKPOINT_PATH.resolve(),
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
            distinct = {
                key: root / f"{index}_{key}" for index, key in enumerate(keys)
            }
            STAGE_41.validate_artifact_paths(**distinct)

            for first_key, second_key in combinations(keys, 2):
                with self.subTest(first=first_key, second=second_key):
                    collided = dict(distinct)
                    collided[second_key] = collided[first_key]
                    with self.assertRaisesRegex(ValueError, "distinct paths"):
                        STAGE_41.validate_artifact_paths(**collided)

    def test_restore_preserves_exact_model_optimizer_batch_and_rng_lineage(
        self,
    ) -> None:
        device = torch.device("cpu")
        torch.manual_seed(411)
        source_model = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(
            source_model.parameters(),
            lr=STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE,
        )
        inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        source_model(inputs).square().mean().backward()
        source_optimizer.step()
        source_generator = torch.Generator().manual_seed(412)
        torch.randint(0, 10, (7,), generator=source_generator)
        torch.manual_seed(413)
        source_rng_state = STAGE_41.capture_rng_state(device)
        checkpoint = {
            "model_state_dict": copy.deepcopy(source_model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(source_optimizer.state_dict()),
            "training_generator_state": source_generator.get_state().clone(),
            "rng_state": copy.deepcopy(source_rng_state),
        }
        source_optimizer_hash = STAGE_41.structured_state_sha256(
            checkpoint["optimizer_state_dict"]
        )
        source = SimpleNamespace(checkpoint=checkpoint)

        restored_hashes: list[tuple[str, str, str, str]] = []
        restored_objects: list[
            tuple[torch.nn.Linear, torch.optim.AdamW, torch.Generator]
        ] = []
        for seed in (414, 415):
            torch.manual_seed(seed)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
            generator = torch.Generator().manual_seed(seed)
            torch.rand(11)
            with patch.object(STAGE_41, "verify_model_masks"):
                STAGE_41.restore_stage_40_source(
                    source,
                    model,
                    optimizer,
                    generator,
                    device,
                )
            hashes = (
                STAGE_41.tensor_mapping_sha256(model.state_dict()),
                STAGE_41.structured_state_sha256(optimizer.state_dict()),
                STAGE_41.tensor_sha256(generator.get_state()),
                STAGE_41.structured_state_sha256(
                    STAGE_41.capture_rng_state(device)
                ),
            )
            restored_hashes.append(hashes)
            restored_objects.append((model, optimizer, generator))

        self.assertEqual(restored_hashes[0], restored_hashes[1])
        self.assertEqual(
            restored_hashes[0][0],
            STAGE_41.tensor_mapping_sha256(checkpoint["model_state_dict"]),
        )
        self.assertEqual(restored_hashes[0][1], source_optimizer_hash)
        self.assertEqual(
            restored_hashes[0][2],
            STAGE_41.tensor_sha256(checkpoint["training_generator_state"]),
        )
        self.assertEqual(
            restored_hashes[0][3],
            STAGE_41.structured_state_sha256(checkpoint["rng_state"]),
        )
        self.assertEqual(
            STAGE_41._optimizer_rates(restored_objects[0][1].state_dict()),
            {STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE},
        )

        model, optimizer, _ = restored_objects[0]
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
        self.assertEqual(
            STAGE_41.structured_state_sha256(
                source.checkpoint["optimizer_state_dict"]
            ),
            source_optimizer_hash,
        )

    def test_absolute_500_step_cadence_source_audit_and_final_snapshot(
        self,
    ) -> None:
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE,
            max_iters=39_000,
            eval_interval=500,
            eval_iters=1,
            seed=421,
        )
        model = ScalarTrainingModel()
        optimizer = RecordingOptimizer(model)
        training_generator = torch.Generator().manual_seed(422)
        best = RecordingCheckpoint(
            target_step=config.max_iters,
            training_generator=training_generator,
        )
        source = SimpleNamespace(
            step=STAGE_41.DEFAULT_SOURCE_STEP,
            snapshot_train_loss=0.0,
            snapshot_val_loss=2.0,
            runtime_device_type="cpu",
            runtime_device_name="cpu",
        )
        final_path = Path("stage_41_final.pt")
        callback, source_audits = STAGE_41.make_evaluation_callback(
            best,
            source,
            final_checkpoint_path=final_path,
        )

        def fixed_evaluator(*args: object, **kwargs: object) -> object:
            del args, kwargs
            completed = float(model.completed_updates)
            return STAGE_41.LossEstimate(
                train=completed,
                val=2.0 + completed,
            )

        dummy_batch = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
        )
        with (
            patch.object(
                STAGE_41,
                "evaluate_on_fixed_batches",
                side_effect=fixed_evaluator,
            ),
            patch.object(STAGE_41, "get_stage_41_batch", return_value=dummy_batch),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = STAGE_41.train_continuation(
                model,
                optimizer,
                object(),
                config,
                torch.device("cpu"),
                training_generator,
                STAGE_41.FinalComputeExtensionSchedule(),
                start_step=STAGE_41.DEFAULT_SOURCE_STEP,
                fixed_panel_seed=423,
                on_evaluation=callback,
            )

        self.assertEqual(
            tuple(record.step for record in result.history),
            (38_000, 38_500, 39_000),
        )
        self.assertEqual(
            tuple(record.losses.train for record in result.history),
            (0.0, 500.0, 1_000.0),
        )
        self.assertEqual(len(optimizer.learning_rates), 1_000)
        self.assertEqual(
            set(optimizer.learning_rates),
            {STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE},
        )
        self.assertEqual(len(source_audits), 1)
        self.assertTrue(source_audits[0].verified)
        self.assertTrue(best.source_audit_verified)
        self.assertEqual(
            best.snapshots,
            [
                ("best", None, 38_000),
                ("final", final_path, 39_000),
            ],
        )

    def test_no_new_best_does_not_relabel_38k_state_as_inherited_37500_best(
        self,
    ) -> None:
        training_generator = torch.Generator().manual_seed(431)
        best = RecordingCheckpoint(
            target_step=44_000,
            training_generator=training_generator,
        )
        best.best_step = 37_500
        best.best_val_loss = 1.4204
        source = SimpleNamespace(
            step=38_000,
            best_step=37_500,
            best_train_loss=1.201,
            best_val_loss=1.4204,
            snapshot_train_loss=1.200,
            snapshot_val_loss=1.4210,
            runtime_device_type="cpu",
            runtime_device_name="cpu",
        )
        final_path = Path("stage_41_final.pt")
        callback, source_audits = STAGE_41.make_evaluation_callback(
            best,
            source,
            final_checkpoint_path=final_path,
        )

        callback(
            STAGE_41.EvaluationRecord(
                step=38_000,
                losses=STAGE_41.LossEstimate(train=1.200, val=1.4210),
            )
        )
        callback(
            STAGE_41.EvaluationRecord(
                step=44_000,
                losses=STAGE_41.LossEstimate(train=1.180, val=1.4220),
            )
        )

        self.assertEqual(len(source_audits), 1)
        self.assertTrue(source_audits[0].verified)
        self.assertTrue(best.source_audit_verified)
        self.assertEqual(
            best.snapshots,
            [("final", final_path, 44_000)],
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
        schedule = STAGE_41.FinalComputeExtensionSchedule()
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE,
            max_iters=38_001,
            eval_interval=1,
            eval_iters=1,
            seed=441,
        )
        device = torch.device("cpu")
        runtime_type, runtime_name = STAGE_41._runtime_identity(device)
        torch.manual_seed(config.seed)
        model = STAGE_41.build_model(data, config, 2, 6, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE,
        )
        inputs = torch.zeros((2, 24), dtype=torch.long)
        _, loss = model(inputs, inputs)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertEqual(len(optimizer.state), len(tuple(model.parameters())))

        def set_optimizer_step(step: int) -> None:
            STAGE_41._set_optimizer_learning_rate(
                optimizer,
                STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE,
            )
            for state in optimizer.state.values():
                state["step"].fill_(step)

        training_batch_seed = 442
        fixed_panel_seed = 443
        training_rng_seed = 444
        training_generator = torch.Generator()
        training_generator.set_state(
            STAGE_41._expected_training_generator_state(
                data,
                config,
                training_batch_seed=training_batch_seed,
                completed_steps=38_000,
            )
        )
        set_optimizer_step(38_000)
        source = STAGE_41.Stage40Source(
            path=Path("synthetic_stage_40_control_final.pt"),
            sha256="4" * 64,
            checkpoint={},
            step=38_000,
            best_step=38_000,
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
            best_path = root / "best.pt"
            final_path = root / "final.pt"
            fork_model_hash = STAGE_41.tensor_mapping_sha256(model.state_dict())
            fork_optimizer_hash = STAGE_41.structured_state_sha256(
                optimizer.state_dict()
            )
            fork_generator_hash = STAGE_41.tensor_sha256(
                training_generator.get_state()
            )
            fork_rng_hash = STAGE_41.structured_state_sha256(
                STAGE_41.capture_rng_state(device)
            )
            tracker = STAGE_41.Stage41ValidationCheckpoint(
                model=model,
                optimizer=optimizer,
                training_generator=training_generator,
                path=best_path,
                device=device,
                config=config,
                n_head=2,
                n_layer=6,
                data_fingerprint=STAGE_41.fingerprint_data(data),
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
                training_batch_seed=training_batch_seed,
                fixed_panel_seed=fixed_panel_seed,
                training_rng_seed=training_rng_seed,
                parameter_count=sum(p.numel() for p in model.parameters()),
                initial_state_sha256=fork_model_hash,
                fork_model_sha256=fork_model_hash,
                fork_optimizer_sha256=fork_optimizer_hash,
                fork_batch_generator_sha256=fork_generator_hash,
                fork_rng_state_sha256=fork_rng_hash,
                best_train_loss=source.best_train_loss,
                source_audit_verified=True,
            )
            tracker.save(38_000)
            best_payload = STAGE_41.validate_stage_41_checkpoint(
                best_path,
                data,
                config,
                2,
                6,
                schedule,
                source,
                expected_kind="best",
                training_batch_seed=training_batch_seed,
                fixed_panel_seed=fixed_panel_seed,
                training_rng_seed=training_rng_seed,
            )
            self.assertEqual(best_payload["step"], 38_000)
            self.assertEqual(best_payload["checkpoint_kind"], "best")
            self.assertEqual(
                best_payload["fork_state"]["optimizer_sha256"],
                fork_optimizer_hash,
            )

            set_optimizer_step(38_001)
            training_generator.set_state(
                STAGE_41._expected_training_generator_state(
                    data,
                    config,
                    training_batch_seed=training_batch_seed,
                    completed_steps=38_001,
                )
            )
            tracker.save_final(
                final_path,
                STAGE_41.EvaluationRecord(
                    step=38_001,
                    losses=STAGE_41.LossEstimate(train=1.19, val=1.41),
                ),
            )
            final_payload = STAGE_41.validate_stage_41_checkpoint(
                final_path,
                data,
                config,
                2,
                6,
                schedule,
                source,
                expected_kind="final",
                training_batch_seed=training_batch_seed,
                fixed_panel_seed=fixed_panel_seed,
                training_rng_seed=training_rng_seed,
            )
            self.assertEqual(final_payload["step"], 38_001)
            self.assertEqual(final_payload["checkpoint_kind"], "final")
            self.assertEqual(
                STAGE_41._optimizer_rates(
                    final_payload["optimizer_state_dict"]
                ),
                {STAGE_41.DEFAULT_CONTINUATION_LEARNING_RATE},
            )


if __name__ == "__main__":
    unittest.main()
