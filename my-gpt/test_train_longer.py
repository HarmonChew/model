import importlib.util
import math
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary
from training import EvaluationRecord, LossEstimate


MODULE_PATH = Path(__file__).with_name("11_train_longer.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_11_train_longer",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_11 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_11
SPEC.loader.exec_module(STAGE_11)

BestValidationCheckpoint = STAGE_11.BestValidationCheckpoint
GPTLanguageModel = STAGE_11.GPTLanguageModel


class TrainLongerTests(unittest.TestCase):
    def setUp(self) -> None:
        text = "abcd" * 100
        vocabulary = CharacterVocabulary.from_text(text)
        token_ids = torch.tensor(
            vocabulary.encode(text),
            dtype=torch.long,
        )
        self.data = CharacterData(
            vocabulary=vocabulary,
            train_data=token_ids[:320],
            val_data=token_ids[320:],
            num_characters=len(text),
        )
        self.config = TrainingConfig(
            batch_size=2,
            block_size=4,
            n_embd=8,
            learning_rate=1e-2,
            max_iters=10,
            eval_interval=3,
            eval_iters=1,
            seed=17,
        )
        self.device = torch.device("cpu")

    def make_model(self) -> torch.nn.Module:
        return GPTLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
            n_head=2,
            n_layer=1,
        )

    def run_optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
    ) -> None:
        inputs, targets = self.data.get_batch(
            "train",
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            device=self.device,
            generator=generator,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def assert_nested_equal(self, actual: object, expected: object) -> None:
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return

        if isinstance(expected, Mapping):
            self.assertIsInstance(actual, Mapping)
            assert isinstance(actual, Mapping)
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
            return

        if isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected))
            assert isinstance(actual, (list, tuple))
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(
                actual,
                expected,
                strict=True,
            ):
                self.assert_nested_equal(actual_item, expected_item)
            return

        self.assertEqual(actual, expected)

    def assert_model_states_equal(
        self,
        actual: torch.nn.Module,
        expected: torch.nn.Module,
    ) -> None:
        self.assert_nested_equal(
            actual.state_dict(),
            expected.state_dict(),
        )

    def test_defaults_keep_stage_10_settings_and_target_10_000(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_11.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 64)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.max_iters, 10_000)
        self.assertEqual(args.eval_interval, 500)
        self.assertIsNone(args.resume_from)
        self.assertFalse(args.allow_optimizer_restart)
        self.assertEqual(args.legacy_step, 5_000)
        self.assertEqual(
            args.checkpoint_path,
            STAGE_11.DEFAULT_CHECKPOINT_PATH,
        )

    def test_legacy_weights_require_explicit_optimizer_restart(self) -> None:
        torch.manual_seed(self.config.seed)
        source_model = self.make_model()

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "legacy.pt"
            torch.save(source_model.state_dict(), checkpoint_path)

            torch.manual_seed(self.config.seed + 1)
            loaded_model = self.make_model()
            optimizer = torch.optim.AdamW(
                loaded_model.parameters(),
                lr=self.config.learning_rate,
            )
            training_generator = torch.Generator().manual_seed(
                self.config.seed + 1
            )

            with self.assertRaisesRegex(
                ValueError,
                "allow-optimizer-restart",
            ):
                STAGE_11.load_resume_checkpoint(
                    checkpoint_path,
                    loaded_model,
                    optimizer,
                    training_generator,
                    self.data,
                    self.config,
                    self.device,
                    allow_optimizer_restart=False,
                    legacy_step=5,
                )

            resume = STAGE_11.load_resume_checkpoint(
                checkpoint_path,
                loaded_model,
                optimizer,
                training_generator,
                self.data,
                self.config,
                self.device,
                allow_optimizer_restart=True,
                legacy_step=5,
            )

        self.assert_model_states_equal(loaded_model, source_model)
        self.assertEqual(optimizer.state, {})
        self.assertEqual(resume.mode, "legacy_weights")
        self.assertEqual(resume.start_step, 5)
        self.assertTrue(math.isinf(resume.best_val_loss))
        self.assertIsNone(resume.best_step)
        self.assertFalse(resume.optimizer_restored)
        self.assertFalse(resume.training_generator_restored)
        self.assertIn("restarted AdamW", resume.description)

        expected_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )
        STAGE_11.advance_training_generator(
            expected_generator,
            num_steps=5,
            batch_size=self.config.batch_size,
            num_train_tokens=len(self.data.train_data),
            block_size=self.config.block_size,
        )
        torch.testing.assert_close(
            training_generator.get_state(),
            expected_generator.get_state(),
            rtol=0,
            atol=0,
        )

    def test_full_checkpoint_restores_optimizer_and_generator(self) -> None:
        torch.manual_seed(self.config.seed)
        source_model = self.make_model()
        source_optimizer = torch.optim.AdamW(
            source_model.parameters(),
            lr=self.config.learning_rate,
        )
        source_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )
        self.run_optimizer_step(
            source_model,
            source_optimizer,
            source_generator,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "full.pt"
            torch.save(
                {
                    "step": 1,
                    "model_state_dict": source_model.state_dict(),
                    "optimizer_state_dict": source_optimizer.state_dict(),
                    "best_val_loss": 1.75,
                    "best_step": 1,
                    "training_generator_state": (
                        source_generator.get_state()
                    ),
                },
                checkpoint_path,
            )

            torch.manual_seed(self.config.seed + 2)
            loaded_model = self.make_model()
            loaded_optimizer = torch.optim.AdamW(
                loaded_model.parameters(),
                lr=0.123,
            )
            loaded_generator = torch.Generator().manual_seed(999)
            resume = STAGE_11.load_resume_checkpoint(
                checkpoint_path,
                loaded_model,
                loaded_optimizer,
                loaded_generator,
                self.data,
                self.config,
                self.device,
                allow_optimizer_restart=False,
                legacy_step=5_000,
            )

        self.assert_model_states_equal(loaded_model, source_model)
        self.assert_nested_equal(
            loaded_optimizer.state_dict(),
            source_optimizer.state_dict(),
        )
        torch.testing.assert_close(
            loaded_generator.get_state(),
            source_generator.get_state(),
            rtol=0,
            atol=0,
        )
        self.assertEqual(resume.mode, "full")
        self.assertEqual(resume.start_step, 1)
        self.assertEqual(resume.best_val_loss, 1.75)
        self.assertEqual(resume.best_step, 1)
        self.assertTrue(resume.optimizer_restored)
        self.assertTrue(resume.training_generator_restored)

    def test_strict_improvement_saves_a_full_best_checkpoint(self) -> None:
        torch.manual_seed(self.config.seed)
        model = self.make_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
        training_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )
        self.run_optimizer_step(model, optimizer, training_generator)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "nested" / "best.pt"
            )
            tracker = BestValidationCheckpoint(
                model=model,
                optimizer=optimizer,
                training_generator=training_generator,
                path=checkpoint_path,
                device=self.device,
                config=self.config,
                n_head=2,
                n_layer=1,
            )
            first = EvaluationRecord(
                step=1,
                losses=LossEstimate(train=1.8, val=1.9),
            )
            tied = EvaluationRecord(
                step=2,
                losses=LossEstimate(train=1.7, val=1.9),
            )
            improved = EvaluationRecord(
                step=3,
                losses=LossEstimate(train=1.6, val=1.8),
            )

            self.assertTrue(tracker.consider(first))
            first_payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            required_keys = {
                "step",
                "model_state_dict",
                "optimizer_state_dict",
                "best_val_loss",
                "best_step",
                "training_generator_state",
            }
            self.assertTrue(required_keys.issubset(first_payload))
            self.assertEqual(first_payload["step"], 1)
            self.assertEqual(first_payload["best_val_loss"], 1.9)
            self.assertEqual(first_payload["best_step"], 1)
            self.assert_nested_equal(
                first_payload["model_state_dict"],
                model.state_dict(),
            )
            self.assert_nested_equal(
                first_payload["optimizer_state_dict"],
                optimizer.state_dict(),
            )
            torch.testing.assert_close(
                first_payload["training_generator_state"],
                training_generator.get_state(),
                rtol=0,
                atol=0,
            )

            with torch.no_grad():
                model.token_embedding_table.weight.add_(1.0)

            self.assertFalse(tracker.consider(tied))
            tied_payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assert_nested_equal(tied_payload, first_payload)

            self.assertTrue(tracker.consider(improved))
            improved_payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(improved_payload["step"], 3)
            self.assertEqual(improved_payload["best_val_loss"], 1.8)
            self.assertEqual(improved_payload["best_step"], 3)
            self.assert_nested_equal(
                improved_payload["model_state_dict"],
                model.state_dict(),
            )
            self.assertFalse(
                checkpoint_path.with_name(
                    f".{checkpoint_path.name}.tmp"
                ).exists()
            )

    def test_train_until_uses_absolute_steps_and_remaining_updates(self) -> None:
        torch.manual_seed(self.config.seed)
        model = self.make_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
        training_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )
        evaluations = [
            LossEstimate(train=2.0, val=2.2),
            LossEstimate(train=1.9, val=2.1),
            LossEstimate(train=1.8, val=2.0),
            LossEstimate(train=1.7, val=1.9),
        ]
        callback_records: list[EvaluationRecord] = []

        with (
            mock.patch.object(
                STAGE_11,
                "evaluate_on_fixed_batches",
                side_effect=evaluations,
            ) as evaluate,
            mock.patch.object(
                optimizer,
                "step",
                wraps=optimizer.step,
            ) as optimizer_step,
        ):
            result = STAGE_11.train_until(
                model,
                optimizer,
                self.data,
                self.config,
                self.device,
                training_generator,
                start_step=5,
                on_evaluation=callback_records.append,
            )

        self.assertEqual(optimizer_step.call_count, 5)
        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual(
            [record.step for record in result.history],
            [5, 6, 9, 10],
        )
        self.assertEqual(callback_records, list(result.history))
        self.assertEqual(result.initial, evaluations[0])
        self.assertEqual(result.final, evaluations[-1])

        expected_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )
        for _ in range(5):
            self.data.get_batch(
                "train",
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                device=self.device,
                generator=expected_generator,
            )
        torch.testing.assert_close(
            training_generator.get_state(),
            expected_generator.get_state(),
            rtol=0,
            atol=0,
        )

    def test_train_until_at_target_performs_no_update(self) -> None:
        model = self.make_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
        training_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )
        baseline = LossEstimate(train=1.8, val=2.0)

        with (
            mock.patch.object(
                STAGE_11,
                "evaluate_on_fixed_batches",
                return_value=baseline,
            ) as evaluate,
            mock.patch.object(
                optimizer,
                "step",
                wraps=optimizer.step,
            ) as optimizer_step,
        ):
            result = STAGE_11.train_until(
                model,
                optimizer,
                self.data,
                self.config,
                self.device,
                training_generator,
                start_step=self.config.max_iters,
                on_evaluation=lambda record: None,
            )

        self.assertEqual(optimizer_step.call_count, 0)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(
            [record.step for record in result.history],
            [self.config.max_iters],
        )
        self.assertEqual(result.initial, baseline)
        self.assertEqual(result.final, baseline)

    def test_split_full_resume_matches_uninterrupted_training(self) -> None:
        total_config = TrainingConfig(
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
            learning_rate=self.config.learning_rate,
            max_iters=6,
            eval_interval=3,
            eval_iters=1,
            seed=self.config.seed,
        )
        first_half_config = TrainingConfig(
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
            learning_rate=self.config.learning_rate,
            max_iters=3,
            eval_interval=3,
            eval_iters=1,
            seed=self.config.seed,
        )
        fixed_loss = LossEstimate(train=2.0, val=2.1)

        torch.manual_seed(self.config.seed)
        uninterrupted_model = self.make_model()
        uninterrupted_optimizer = torch.optim.AdamW(
            uninterrupted_model.parameters(),
            lr=self.config.learning_rate,
        )
        uninterrupted_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )
        with mock.patch.object(
            STAGE_11,
            "evaluate_on_fixed_batches",
            return_value=fixed_loss,
        ):
            STAGE_11.train_until(
                uninterrupted_model,
                uninterrupted_optimizer,
                self.data,
                total_config,
                self.device,
                uninterrupted_generator,
                start_step=0,
                on_evaluation=lambda record: None,
            )

        torch.manual_seed(self.config.seed)
        split_model = self.make_model()
        split_optimizer = torch.optim.AdamW(
            split_model.parameters(),
            lr=self.config.learning_rate,
        )
        split_generator = torch.Generator().manual_seed(self.config.seed + 1)
        with mock.patch.object(
            STAGE_11,
            "evaluate_on_fixed_batches",
            return_value=fixed_loss,
        ):
            STAGE_11.train_until(
                split_model,
                split_optimizer,
                self.data,
                first_half_config,
                self.device,
                split_generator,
                start_step=0,
                on_evaluation=lambda record: None,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "split.pt"
            torch.save(
                {
                    "step": 3,
                    "model_state_dict": split_model.state_dict(),
                    "optimizer_state_dict": split_optimizer.state_dict(),
                    "best_val_loss": fixed_loss.val,
                    "best_step": 3,
                    "training_generator_state": split_generator.get_state(),
                    "optimizer_restart_step": None,
                    "optimizer_provenance_known": True,
                },
                checkpoint_path,
            )

            torch.manual_seed(999)
            resumed_model = self.make_model()
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.parameters(),
                lr=self.config.learning_rate,
            )
            resumed_generator = torch.Generator().manual_seed(999)
            resume = STAGE_11.load_resume_checkpoint(
                checkpoint_path,
                resumed_model,
                resumed_optimizer,
                resumed_generator,
                self.data,
                total_config,
                self.device,
                allow_optimizer_restart=False,
                legacy_step=5_000,
                n_head=2,
                n_layer=1,
            )

        with mock.patch.object(
            STAGE_11,
            "evaluate_on_fixed_batches",
            return_value=fixed_loss,
        ):
            STAGE_11.train_until(
                resumed_model,
                resumed_optimizer,
                self.data,
                total_config,
                self.device,
                resumed_generator,
                start_step=resume.start_step,
                on_evaluation=lambda record: None,
            )

        self.assert_model_states_equal(resumed_model, uninterrupted_model)
        self.assert_nested_equal(
            resumed_optimizer.state_dict(),
            uninterrupted_optimizer.state_dict(),
        )
        torch.testing.assert_close(
            resumed_generator.get_state(),
            uninterrupted_generator.get_state(),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(
        hasattr(torch, "xpu") and torch.xpu.is_available(),
        "XPU is unavailable",
    )
    def test_full_checkpoint_restores_device_mapped_xpu_rng(self) -> None:
        device = torch.device("xpu")
        torch.manual_seed(self.config.seed)
        torch.xpu.manual_seed_all(self.config.seed)
        model = self.make_model().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
        training_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )

        def run_xpu_step() -> None:
            inputs, targets = self.data.get_batch(
                "train",
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                device=device,
                generator=training_generator,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "xpu.pt"
            tracker = BestValidationCheckpoint(
                model=model,
                optimizer=optimizer,
                training_generator=training_generator,
                path=checkpoint_path,
                device=device,
                config=self.config,
                n_head=2,
                n_layer=1,
                data_fingerprint=STAGE_11.fingerprint_data(self.data),
            )
            run_xpu_step()
            tracker.consider(
                EvaluationRecord(
                    step=1,
                    losses=LossEstimate(train=1.8, val=1.9),
                )
            )
            run_xpu_step()
            expected_cpu_rng = torch.get_rng_state().clone()
            expected_xpu_rng = [
                state.clone() for state in torch.xpu.get_rng_state_all()
            ]
            tracker.consider(
                EvaluationRecord(
                    step=2,
                    losses=LossEstimate(train=1.7, val=1.8),
                )
            )

            torch.rand(3)
            torch.rand(3, device=device)
            loaded_model = self.make_model().to(device)
            loaded_optimizer = torch.optim.AdamW(
                loaded_model.parameters(),
                lr=self.config.learning_rate,
            )
            loaded_generator = torch.Generator().manual_seed(999)
            STAGE_11.load_resume_checkpoint(
                checkpoint_path,
                loaded_model,
                loaded_optimizer,
                loaded_generator,
                self.data,
                self.config,
                device,
                allow_optimizer_restart=False,
                legacy_step=5_000,
                n_head=2,
                n_layer=1,
            )

        self.assert_model_states_equal(loaded_model, model)
        self.assert_nested_equal(
            loaded_optimizer.state_dict(),
            optimizer.state_dict(),
        )
        torch.testing.assert_close(
            torch.get_rng_state(),
            expected_cpu_rng,
            rtol=0,
            atol=0,
        )
        for actual, expected in zip(
            torch.xpu.get_rng_state_all(),
            expected_xpu_rng,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
