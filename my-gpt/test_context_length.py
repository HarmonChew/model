import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary
from training import BenchmarkConfig, BenchmarkStats, train_model


MODULE_PATH = Path(__file__).with_name("09_context_length.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_9_context_length",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_9
SPEC.loader.exec_module(STAGE_9)

GPTLanguageModel = STAGE_9.GPTLanguageModel


class ContextLengthTests(unittest.TestCase):
    vocab_size = 65
    block_size = 64
    n_embd = 32
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

    def test_stage_9_changes_only_the_default_context_length(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_9.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 32)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)

    def test_context_dependent_shapes_and_parameter_count(self) -> None:
        self.assertEqual(
            self.model.position_embedding_table.weight.shape,
            (64, 32),
        )

        for block in self.model.blocks:
            for head in block.sa.heads:
                self.assertEqual(head.tril.shape, (64, 64))

        inputs = torch.randint(
            self.vocab_size,
            (2, self.block_size),
        )
        targets = torch.randint(
            self.vocab_size,
            (2, self.block_size),
        )
        logits, loss = self.model(inputs, targets)
        attention = self.model.get_attention_weights(inputs, block_index=0)

        self.assertEqual(logits.shape, (2, 64, 65))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(math.isfinite(loss.item()))
        self.assertEqual(attention.shape, (2, 4, 64, 64))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            56_769,
        )


class TrainingBenchmarkTests(unittest.TestCase):
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

    def test_benchmark_measures_existing_steps_without_training_extra_steps(
        self,
    ) -> None:
        torch.manual_seed(self.config.seed)
        reference_model = self.make_model()
        benchmarked_model = self.make_model()
        benchmarked_model.load_state_dict(reference_model.state_dict())

        reference_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )
        benchmarked_generator = torch.Generator().manual_seed(
            self.config.seed + 1
        )

        reference = train_model(
            reference_model,
            self.data,
            self.config,
            self.device,
            reference_generator,
        )
        benchmarked = train_model(
            benchmarked_model,
            self.data,
            self.config,
            self.device,
            benchmarked_generator,
            benchmark=BenchmarkConfig(num_warmup=1, num_steps=2),
        )

        self.assertIsNone(reference.benchmark)
        self.assertIsInstance(benchmarked.benchmark, BenchmarkStats)
        stats = benchmarked.benchmark
        assert stats is not None

        self.assertGreater(stats.seconds, 0.0)
        self.assertGreater(stats.iterations_per_sec, 0.0)
        self.assertGreater(stats.tokens_per_sec, 0.0)
        self.assertAlmostEqual(
            stats.tokens_per_sec,
            stats.iterations_per_sec
            * self.config.batch_size
            * self.config.block_size,
        )
        self.assertEqual(stats.peak_allocated_mb, 0.0)
        self.assertEqual(stats.peak_reserved_mb, 0.0)

        self.assertEqual(reference.initial, benchmarked.initial)
        self.assertEqual(reference.final, benchmarked.final)
        self.assertEqual(reference.history, benchmarked.history)
        self.assertEqual(
            [record.step for record in benchmarked.history],
            [0, 2],
        )
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


if __name__ == "__main__":
    unittest.main()
