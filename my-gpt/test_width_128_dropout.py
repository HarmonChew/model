from __future__ import annotations

import contextlib
import copy
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
from training import seed_everything


MODULE_PATH = Path(__file__).with_name("21_dropout.py")
SPEC = importlib.util.spec_from_file_location("stage_21_dropout", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE_21 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_21
SPEC.loader.exec_module(STAGE_21)


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


def make_tiny_protocol(
    directory: Path,
) -> tuple[
    CharacterData,
    TrainingConfig,
    TrainingConfig,
    object,
    object,
    STAGE_21.DropoutSpec,
    STAGE_21.LearningRateSchedule,
]:
    data = make_character_data()
    schedule = STAGE_21.LearningRateSchedule(
        initial_learning_rate=2e-2,
        first_decay_step=1,
        middle_learning_rate=7e-3,
        second_decay_step=2,
        final_learning_rate=2e-3,
    )
    narrow_config = TrainingConfig(
        batch_size=2,
        block_size=8,
        n_embd=8,
        learning_rate=schedule.initial_learning_rate,
        max_iters=3,
        eval_interval=3,
        eval_iters=1,
        seed=101,
    )
    wide_config = TrainingConfig(
        batch_size=2,
        block_size=8,
        n_embd=16,
        learning_rate=schedule.initial_learning_rate,
        max_iters=3,
        eval_interval=3,
        eval_iters=1,
        seed=101,
    )
    narrow = STAGE_21._STAGE_20.WidthSpec(
        name="width-8",
        n_embd=8,
        checkpoint_path=directory / "stage20_width_8.pt",
    )
    wide = STAGE_21._STAGE_20.WidthSpec(
        name="width-16",
        n_embd=16,
        checkpoint_path=directory / "stage20_width_16.pt",
    )
    treatment = STAGE_21.DropoutSpec(
        name="dropout-p005",
        dropout=0.05,
        checkpoint_path=directory / "stage21_dropout_005.pt",
    )
    return (
        data,
        narrow_config,
        wide_config,
        narrow,
        wide,
        treatment,
        schedule,
    )


def create_tiny_stage_20_source(
    data: CharacterData,
    narrow_config: TrainingConfig,
    wide_config: TrainingConfig,
    narrow: object,
    wide: object,
    schedule: STAGE_21.LearningRateSchedule,
) -> object:
    with contextlib.redirect_stdout(io.StringIO()):
        return STAGE_21._STAGE_20.run_experiment(
            data,
            narrow_config,
            wide_config,
            narrow,
            wide,
            2,
            1,
            torch.device("cpu"),
            0,
            schedule,
            training_batch_seed=202,
            training_rng_seed=303,
            benchmark=None,
        )


class Stage21DefaultsAndInitializationTests(unittest.TestCase):
    def test_defaults_pins_parameter_budget_and_identical_initialization(
        self,
    ) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_21.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 128)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.initial_learning_rate, 1e-3)
        self.assertEqual(args.first_decay_step, 10_000)
        self.assertEqual(args.middle_learning_rate, 3e-4)
        self.assertEqual(args.second_decay_step, 13_000)
        self.assertEqual(args.final_learning_rate, 1e-4)
        self.assertEqual(args.dropout, 0.05)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_20_width_128_best_checkpoint.pt",
        )
        self.assertEqual(
            args.checkpoint_path.name,
            "stage_21_dropout_005_best_checkpoint.pt",
        )

        checkpoint_hash, checkpoint_step = STAGE_21._pin_for_default_path(
            STAGE_21.DEFAULT_CONTROL_CHECKPOINT_PATH
        )
        self.assertEqual(
            checkpoint_hash,
            "645f7dc1971bd7e790b6583564731acd7f8501e26ecea457d21f8f7cec8c03f2",
        )
        self.assertEqual(checkpoint_step, 10_500)

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
            STAGE_21.create_shared_initial_state(data, config, 4, 4)
        )
        self.assertEqual(parameter_count, 816_705)
        self.assertEqual(initial_hash, STAGE_21.DEFAULT_INITIAL_STATE_SHA256)

        seed_everything(config.seed)
        dropout_model = STAGE_21._build_model(data, config, 4, 4, 0.05)
        treatment_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in dropout_model.state_dict().items()
        }
        self.assertEqual(initial_state.keys(), treatment_state.keys())
        for name in initial_state:
            torch.testing.assert_close(
                initial_state[name], treatment_state[name], rtol=0, atol=0
            )
        self.assertEqual(
            STAGE_21.tensor_mapping_sha256(treatment_state),
            initial_hash,
        )
        dropout_modules = [
            module
            for module in dropout_model.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        self.assertEqual(len(dropout_modules), 8)
        self.assertTrue(all(module.p == 0.05 for module in dropout_modules))

    def test_probability_and_reference_paths_are_hard_protocol_boundaries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "stage20.pt"
            treatment_path = directory / "stage21.pt"
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
            schedule = STAGE_21.LearningRateSchedule(
                initial_learning_rate=2e-2,
                first_decay_step=1,
                middle_learning_rate=7e-3,
                second_decay_step=2,
                final_learning_rate=2e-3,
            )
            wrong_probability = STAGE_21.DropoutSpec(
                "dropout-p004",
                0.04,
                treatment_path,
            )
            with self.assertRaisesRegex(ValueError, "exactly 0.05"):
                STAGE_21.validate_protocol(
                    config,
                    2,
                    1,
                    wrong_probability,
                    source,
                    schedule,
                )

            with self.assertRaisesRegex(ValueError, "different paths"):
                STAGE_21.validate_distinct_paths(source, source)
            source_temporary = source.with_name(f".{source.name}.tmp")
            with self.assertRaisesRegex(ValueError, "different paths"):
                STAGE_21.validate_distinct_paths(source_temporary, source)


class Stage21TreatmentWorkflowTests(unittest.TestCase):
    def test_treatment_only_checkpoint_and_fresh_banded_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (
                data,
                narrow_config,
                config,
                narrow,
                wide,
                treatment,
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
            source_bytes = wide.checkpoint_path.read_bytes()
            source_hash = STAGE_21.checkpoint_sha256(wide.checkpoint_path)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                STAGE_21.validate_stage_20_reference(
                    wide.checkpoint_path,
                    data,
                    config,
                    2,
                    1,
                    schedule,
                    initial_state_sha256=(
                        stage_20.initialization.treatment_state_sha256
                    ),
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_checkpoint_sha256="0" * 64,
                )

            with contextlib.redirect_stdout(io.StringIO()):
                report = STAGE_21.run_experiment(
                    data,
                    config,
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    schedule,
                    treatment,
                    wide.checkpoint_path,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    expected_control_sha256=source_hash,
                    expected_control_step=stage_20.treatment.best_step,
                )

            self.assertEqual(wide.checkpoint_path.read_bytes(), source_bytes)
            self.assertTrue(treatment.checkpoint_path.is_file())
            self.assertEqual(report.parameter_count, 3_557)
            self.assertEqual(
                report.initial_state_sha256,
                stage_20.initialization.treatment_state_sha256,
            )
            self.assertEqual(report.control.reference.checkpoint_sha256, source_hash)
            self.assertEqual(report.treatment.spec.dropout, 0.05)
            self.assertEqual(report.bands, ((0, 2), (2, 4), (4, 6), (6, 8)))
            expected_final_generator = STAGE_21.expected_training_generator_state(
                data,
                config,
                training_batch_seed=202,
                completed_steps=config.max_iters,
            )
            self.assertEqual(
                report.treatment.final_batch_generator_sha256,
                STAGE_21.tensor_sha256(expected_final_generator),
            )

            payload = STAGE_21.validate_stage_21_checkpoint(
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
                control_checkpoint_sha256=source_hash,
                control_checkpoint_step=report.control.best_step,
            )
            step = int(payload["step"])
            best_record = next(
                record
                for record in report.treatment.training.history
                if record.step == step
            )
            self.assertEqual(
                report.treatment.best_training_losses,
                best_record.losses.train,
            )
            self.assertEqual(
                report.treatment.best_validation_losses,
                best_record.losses.val,
            )
            self.assertEqual(
                payload["best_training_losses"]["overall"],
                best_record.losses.train.overall,
            )
            self.assertEqual(
                payload["best_validation_losses"]["overall"],
                best_record.losses.val.overall,
            )
            self.assertEqual(payload["best_val_loss"], best_record.losses.val.overall)
            replayed = STAGE_21.replay_treatment_fixed_panel(
                treatment,
                STAGE_21.checkpoint_sha256(treatment.checkpoint_path),
                STAGE_21.PositionLossEstimate(
                    train=best_record.losses.train,
                    val=best_record.losses.val,
                ),
                data,
                config,
                2,
                1,
                torch.device("cpu"),
                report.bands,
            )
            self.assertEqual(replayed, best_record.losses)

            experiment = payload["experiment"]
            self.assertEqual(experiment["stage"], 21)
            self.assertTrue(experiment["from_scratch"])
            self.assertTrue(experiment["reused_control"])
            self.assertTrue(experiment["identical_parameter_count"])
            self.assertTrue(experiment["identical_learned_initialization"])
            self.assertTrue(experiment["identical_training_batches"])
            self.assertEqual(experiment["control_stage"], 20)
            self.assertEqual(experiment["control_checkpoint_sha256"], source_hash)
            self.assertEqual(payload["optimizer_restart_step"], None)
            self.assertTrue(payload["optimizer_provenance_known"])

            training = payload["training_config"]
            expected_rate = schedule.for_update(max(0, step - 1))
            self.assertEqual(training["learning_rate"], expected_rate)
            self.assertEqual(training["optimizer"], "AdamW")
            self.assertEqual(training["optimizer_weight_decay"], 0.01)
            optimizer = payload["optimizer_state_dict"]
            self.assertEqual(
                {float(group["lr"]) for group in optimizer["param_groups"]},
                {expected_rate},
            )
            if step:
                self.assertTrue(optimizer["state"])
                for state in optimizer["state"].values():
                    optimizer_step = state["step"]
                    if isinstance(optimizer_step, torch.Tensor):
                        optimizer_step = optimizer_step.item()
                    self.assertEqual(float(optimizer_step), step)
            else:
                self.assertFalse(optimizer["state"])
            expected_checkpoint_generator = (
                STAGE_21.expected_training_generator_state(
                    data,
                    config,
                    training_batch_seed=202,
                    completed_steps=step,
                )
            )
            torch.testing.assert_close(
                payload["training_generator_state"],
                expected_checkpoint_generator,
                rtol=0,
                atol=0,
            )

            self._assert_tampering_is_rejected(
                directory,
                payload,
                treatment,
                data,
                config,
                schedule,
                report,
                source_hash,
            )

            treatment_bytes = treatment.checkpoint_path.read_bytes()
            precise_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            original_get_batch = CharacterData.get_batch

            def capture_precise_batch(
                batch_data: CharacterData,
                split: str,
                **kwargs: object,
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
                precise = STAGE_21.run_precise_validation(
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

            self.assertEqual(wide.checkpoint_path.read_bytes(), source_bytes)
            self.assertEqual(treatment.checkpoint_path.read_bytes(), treatment_bytes)
            self.assertEqual([item.name for item in precise.results], ["p=0", "p=0.05"])
            self.assertEqual(precise.bands, report.bands)
            self.assertEqual(len(precise_batches), 6)
            for control_batch, treatment_batch in zip(
                precise_batches[:3],
                precise_batches[3:],
                strict=True,
            ):
                torch.testing.assert_close(
                    control_batch[0], treatment_batch[0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    control_batch[1], treatment_batch[1], rtol=0, atol=0
                )
            self.assertTrue(
                all(
                    len(losses.bands) == 4
                    for result in precise.results
                    for losses in result.batch_losses
                )
            )
            expected_differences = tuple(
                STAGE_21._STAGE_20._STAGE_18._subtract_position_losses(
                    candidate,
                    baseline,
                )
                for candidate, baseline in zip(
                    precise.results[1].batch_losses,
                    precise.results[0].batch_losses,
                    strict=True,
                )
            )
            expected_statistics = STAGE_21._position_statistics(
                expected_differences
            )
            self.assertEqual(precise.delta.candidate, "p=0.05")
            self.assertEqual(precise.delta.baseline, "p=0")
            self.assertEqual(precise.delta.mean_delta, expected_statistics.mean)
            self.assertEqual(
                precise.delta.standard_error,
                expected_statistics.standard_error,
            )

    def _assert_tampering_is_rejected(
        self,
        directory: Path,
        payload: object,
        treatment: STAGE_21.DropoutSpec,
        data: CharacterData,
        config: TrainingConfig,
        schedule: STAGE_21.LearningRateSchedule,
        report: STAGE_21.ExperimentReport,
        source_hash: str,
    ) -> None:
        mutations: list[tuple[str, str, object]] = []

        wrong_provenance = copy.deepcopy(payload)
        wrong_provenance["experiment"]["control_checkpoint_sha256"] = "wrong"
        mutations.append(("provenance", "control_checkpoint_sha256", wrong_provenance))

        wrong_generator = copy.deepcopy(payload)
        wrong_generator["training_generator_state"] = torch.Generator().get_state()
        mutations.append(("generator", "generator state", wrong_generator))

        wrong_train = copy.deepcopy(payload)
        wrong_train["best_training_losses"]["overall"] = float("nan")
        mutations.append(("best_train", "best_training_losses", wrong_train))

        wrong_val = copy.deepcopy(payload)
        wrong_val["best_validation_losses"]["overall"] += 1.0
        mutations.append(("best_val", "best validation loss", wrong_val))

        wrong_mask = copy.deepcopy(payload)
        mask_name = next(
            name
            for name in wrong_mask["model_state_dict"]
            if name.endswith(".tril")
        )
        wrong_mask["model_state_dict"][mask_name].zero_()
        mutations.append(("causal_mask", "causal mask", wrong_mask))

        step = int(payload["step"])
        if step:
            wrong_optimizer = copy.deepcopy(payload)
            first_state = next(iter(wrong_optimizer["optimizer_state_dict"]["state"].values()))
            first_state["step"] = step + 1
            mutations.append(("optimizer", "optimizer step", wrong_optimizer))

        for label, message, corrupted in mutations:
            with self.subTest(tamper=label):
                path = directory / f"tampered_{label}.pt"
                torch.save(corrupted, path)
                with self.assertRaisesRegex(ValueError, message):
                    STAGE_21.validate_stage_21_checkpoint(
                        path,
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
                        control_checkpoint_sha256=source_hash,
                        control_checkpoint_step=report.control.best_step,
                    )

    @unittest.skipUnless(
        STAGE_21.DEFAULT_CONTROL_CHECKPOINT_PATH.is_file(),
        "canonical Stage 20 C=128 checkpoint is not present",
    )
    def test_canonical_stage_20_input_hash_initialization_and_step(self) -> None:
        data = CharacterData.from_file(STAGE_21.DEFAULT_DATA_PATH)
        schedule = STAGE_21.LearningRateSchedule()
        config = TrainingConfig(
            batch_size=32,
            block_size=64,
            n_embd=128,
            learning_rate=schedule.initial_learning_rate,
            max_iters=18_000,
            eval_interval=500,
            eval_iters=100,
            seed=1_337,
        )
        _, initial_hash, parameter_count = STAGE_21.create_shared_initial_state(
            data,
            config,
            4,
            4,
        )
        reference = STAGE_21.validate_stage_20_reference(
            STAGE_21.DEFAULT_CONTROL_CHECKPOINT_PATH,
            data,
            config,
            4,
            4,
            schedule,
            initial_state_sha256=initial_hash,
            training_batch_seed=1_338,
            training_rng_seed=1_340,
            expected_checkpoint_sha256=STAGE_21.DEFAULT_CONTROL_CHECKPOINT_SHA256,
            expected_step=STAGE_21.DEFAULT_CONTROL_BEST_STEP,
        )

        self.assertEqual(
            STAGE_21.checkpoint_sha256(STAGE_21.DEFAULT_CONTROL_CHECKPOINT_PATH),
            STAGE_21.DEFAULT_CONTROL_CHECKPOINT_SHA256,
        )
        self.assertEqual(
            STAGE_21.fingerprint_data(data),
            "fa03e609cfc0769eb1fa73fb4e81239fee0c6e9ab0608c549b0a742a25535493",
        )
        self.assertEqual(initial_hash, STAGE_21.DEFAULT_INITIAL_STATE_SHA256)
        self.assertEqual(parameter_count, 816_705)
        self.assertEqual(reference.initial_state_sha256, initial_hash)
        self.assertEqual(reference.checkpoint_step, 10_500)


if __name__ == "__main__":
    unittest.main()
