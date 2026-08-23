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

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary
from training import (
    BenchmarkConfig,
    EvaluationRecord,
    LossEstimate,
    TrainingResult,
    benchmark_training,
)


MODULE_PATH = Path(__file__).with_name("10_scale_width.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_10_scale_width",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_10 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_10
SPEC.loader.exec_module(STAGE_10)

BestValidationCheckpoint = STAGE_10.BestValidationCheckpoint
GPTLanguageModel = STAGE_10.GPTLanguageModel


class ScaleWidthTests(unittest.TestCase):
    vocab_size = 65
    block_size = 64
    n_embd = 64
    n_head = 4
    n_layer = 4

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.model = GPTLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
        )

    def test_stage_10_changes_only_the_default_width(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_10.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 64)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.max_iters, 5_000)
        self.assertEqual(args.eval_interval, 500)

    def test_width_dependent_shapes_and_exact_parameter_budget(self) -> None:
        self.assertEqual(
            self.model.token_embedding_table.weight.shape,
            (65, 64),
        )
        self.assertEqual(
            self.model.position_embedding_table.weight.shape,
            (64, 64),
        )
        self.assertEqual(self.model.ln_f.weight.shape, (64,))
        self.assertEqual(self.model.lm_head.weight.shape, (65, 64))

        block_counts = [
            sum(parameter.numel() for parameter in block.parameters())
            for block in self.model.blocks
        ]
        self.assertEqual(block_counts, [49_792] * 4)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            211_777,
        )

        first_block = self.model.blocks[0]
        self.assertEqual(first_block.sa.head_size, 16)
        self.assertEqual(first_block.sa.proj.weight.shape, (64, 64))
        self.assertEqual(first_block.ffwd.net[0].weight.shape, (256, 64))
        self.assertEqual(first_block.ffwd.net[2].weight.shape, (64, 256))

    def test_attention_shape_still_depends_on_context_not_width(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size))
        targets = torch.randint(self.vocab_size, (2, self.block_size))
        logits, loss = self.model(inputs, targets)
        attention = self.model.get_attention_weights(inputs)

        self.assertEqual(logits.shape, (2, 64, 65))
        self.assertEqual(attention.shape, (2, 4, 64, 64))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(math.isfinite(loss.item()))

        loss.backward()
        for block in self.model.blocks:
            gradient = block.sa.heads[0].query.weight.grad
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(torch.isfinite(gradient).all())


class BenchmarkIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        text = "abcd" * 100
        vocabulary = CharacterVocabulary.from_text(text)
        token_ids = torch.tensor(vocabulary.encode(text), dtype=torch.long)
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
            max_iters=3,
            eval_interval=2,
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

    def test_standalone_benchmark_updates_only_disposable_model(self) -> None:
        torch.manual_seed(self.config.seed)
        experiment_model = self.make_model()
        benchmark_model = self.make_model()
        experiment_before = {
            name: tensor.detach().clone()
            for name, tensor in experiment_model.state_dict().items()
        }
        benchmark_before = (
            benchmark_model.token_embedding_table.weight.detach().clone()
        )
        optimizer = torch.optim.AdamW(
            benchmark_model.parameters(),
            lr=self.config.learning_rate,
        )

        stats = benchmark_training(
            benchmark_model,
            optimizer,
            self.data,
            self.config,
            self.device,
            torch.Generator().manual_seed(self.config.seed + 1),
            BenchmarkConfig(num_warmup=1, num_steps=2),
        )

        self.assertGreater(stats.iterations_per_sec, 0.0)
        self.assertAlmostEqual(
            stats.tokens_per_sec,
            stats.iterations_per_sec
            * self.config.batch_size
            * self.config.block_size,
        )
        self.assertFalse(
            torch.equal(
                benchmark_before,
                benchmark_model.token_embedding_table.weight.detach(),
            )
        )
        for name, tensor in experiment_model.state_dict().items():
            torch.testing.assert_close(
                tensor,
                experiment_before[name],
                rtol=0,
                atol=0,
            )

    def test_reseeding_after_benchmark_restores_model_initialization(self) -> None:
        torch.manual_seed(self.config.seed)
        expected_model = self.make_model()

        STAGE_10.run_independent_benchmark(
            self.data,
            self.config,
            n_head=2,
            n_layer=1,
            device=self.device,
            benchmark=BenchmarkConfig(num_warmup=1, num_steps=1),
        )
        STAGE_10.seed_everything(self.config.seed)
        actual_model = self.make_model()

        for expected, actual in zip(
            expected_model.state_dict().values(),
            actual_model.state_dict().values(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class BestValidationCheckpointTests(unittest.TestCase):
    def test_only_strict_improvements_replace_the_checkpoint(self) -> None:
        model = GPTLanguageModel(
            vocab_size=4,
            block_size=4,
            n_embd=8,
            n_head=2,
            n_layer=1,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "nested" / "best.pt"
            )
            tracker = BestValidationCheckpoint(model, checkpoint_path)
            first = EvaluationRecord(
                step=0,
                losses=LossEstimate(train=2.0, val=2.1),
            )
            tied = EvaluationRecord(
                step=500,
                losses=LossEstimate(train=1.9, val=2.1),
            )
            improved = EvaluationRecord(
                step=1_000,
                losses=LossEstimate(train=1.8, val=2.0),
            )

            self.assertTrue(tracker.consider(first))
            first_state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            with torch.no_grad():
                model.token_embedding_table.weight.add_(1.0)

            self.assertFalse(tracker.consider(tied))
            tied_state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(
                tied_state["token_embedding_table.weight"],
                first_state["token_embedding_table.weight"],
            )

            self.assertTrue(tracker.consider(improved))
            final_state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(
                final_state["token_embedding_table.weight"],
                model.token_embedding_table.weight,
            )
            self.assertEqual(tracker.best_step, 1_000)
            self.assertEqual(tracker.best_val_loss, 2.0)

    def test_non_finite_validation_loss_is_rejected(self) -> None:
        model = GPTLanguageModel(
            vocab_size=4,
            block_size=4,
            n_embd=8,
            n_head=2,
            n_layer=1,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pt"
            tracker = BestValidationCheckpoint(model, checkpoint_path)
            record = EvaluationRecord(
                step=500,
                losses=LossEstimate(train=2.0, val=float("nan")),
            )

            with self.assertRaisesRegex(ValueError, "must be finite"):
                tracker.consider(record)

            self.assertIsNone(tracker.best_step)
            self.assertFalse(checkpoint_path.exists())


class ExperimentWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        text = "abcd" * 100
        vocabulary = CharacterVocabulary.from_text(text)
        token_ids = torch.tensor(vocabulary.encode(text), dtype=torch.long)
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
            max_iters=1,
            eval_interval=1,
            eval_iters=1,
            seed=17,
        )
        self.device = torch.device("cpu")

    def test_final_evaluation_can_replace_the_best_checkpoint(self) -> None:
        captured: dict[str, torch.Tensor] = {}

        def fake_train(model, *args, on_evaluation=None, **kwargs):
            initial = LossEstimate(train=2.0, val=2.1)
            initial_record = EvaluationRecord(step=0, losses=initial)
            assert on_evaluation is not None
            on_evaluation(initial_record)

            with torch.no_grad():
                model.token_embedding_table.weight.add_(1.0)
            captured["final_embedding"] = (
                model.token_embedding_table.weight.detach().clone()
            )
            final = LossEstimate(train=1.7, val=1.9)
            return TrainingResult(
                initial=initial,
                final=final,
                history=(initial_record,),
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pt"
            with (
                mock.patch.object(STAGE_10, "train_model", new=fake_train),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = STAGE_10.run_experiment(
                    self.data,
                    self.config,
                    n_head=2,
                    n_layer=1,
                    device=self.device,
                    sample_length=0,
                    benchmark=None,
                    checkpoint_path=checkpoint_path,
                )

            saved = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(report.best_step, self.config.max_iters)
            self.assertEqual(report.best_val_loss, 1.9)
            self.assertAlmostEqual(report.generalization_gap, 0.2)
            torch.testing.assert_close(
                saved["token_embedding_table.weight"],
                captured["final_embedding"],
            )

    def test_generation_reloads_an_earlier_best_checkpoint(self) -> None:
        captured: dict[str, torch.Tensor] = {}

        def fake_train(model, *args, on_evaluation=None, **kwargs):
            initial = LossEstimate(train=2.0, val=1.9)
            initial_record = EvaluationRecord(step=0, losses=initial)
            captured["best_embedding"] = (
                model.token_embedding_table.weight.detach().clone()
            )
            assert on_evaluation is not None
            on_evaluation(initial_record)

            with torch.no_grad():
                model.token_embedding_table.weight.add_(1.0)
            final = LossEstimate(train=1.7, val=2.0)
            return TrainingResult(
                initial=initial,
                final=final,
                history=(initial_record,),
            )

        def capture_generation(model, idx, max_new_tokens):
            captured["generation_embedding"] = (
                model.token_embedding_table.weight.detach().clone()
            )
            return idx

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pt"
            with (
                mock.patch.object(STAGE_10, "train_model", new=fake_train),
                mock.patch.object(
                    GPTLanguageModel,
                    "generate",
                    new=capture_generation,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = STAGE_10.run_experiment(
                    self.data,
                    self.config,
                    n_head=2,
                    n_layer=1,
                    device=self.device,
                    sample_length=0,
                    benchmark=None,
                    checkpoint_path=checkpoint_path,
                )

            self.assertEqual(report.best_step, 0)
            torch.testing.assert_close(
                captured["generation_embedding"],
                captured["best_embedding"],
            )


if __name__ == "__main__":
    unittest.main()
