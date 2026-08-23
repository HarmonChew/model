import math
import unittest

import torch
import torch.nn.functional as F

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary
from model import EmbeddingLanguageModel
from train import apply_shared_initialization, capture_shared_initialization
from training import estimate_loss, train_model


class EmbeddingLanguageModelTests(unittest.TestCase):
    vocab_size = 65
    block_size = 8
    n_embd = 32

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.model = EmbeddingLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
        )

    def test_forward_shapes_and_cross_entropy(self) -> None:
        inputs = torch.randint(
            self.vocab_size,
            (32, self.block_size),
        )
        targets = torch.randint(
            self.vocab_size,
            (32, self.block_size),
        )

        logits, loss = self.model(inputs, targets)

        self.assertEqual(logits.shape, (32, 8, 65))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertEqual(loss.ndim, 0)
        expected = F.cross_entropy(
            logits.reshape(32 * 8, 65),
            targets.reshape(32 * 8),
        )
        torch.testing.assert_close(loss, expected)

    def test_embedding_decomposition_shapes(self) -> None:
        inputs = torch.randint(self.vocab_size, (32, self.block_size))
        token_embeddings = self.model.token_embedding_table(inputs)
        positions = torch.arange(self.block_size)
        position_table = self.model.position_embedding_table
        self.assertIsNotNone(position_table)
        assert position_table is not None
        position_embeddings = position_table(positions)
        combined = token_embeddings + position_embeddings
        expected_logits = self.model.lm_head(combined)
        actual_logits, _ = self.model(inputs)

        self.assertEqual(token_embeddings.shape, (32, 8, 32))
        self.assertEqual(positions.shape, (8,))
        self.assertEqual(position_embeddings.shape, (8, 32))
        self.assertEqual(combined.shape, (32, 8, 32))
        torch.testing.assert_close(actual_logits, expected_logits)

    def test_parameter_shapes_and_count(self) -> None:
        parameters = dict(self.model.named_parameters())

        self.assertEqual(
            parameters["token_embedding_table.weight"].shape,
            (65, 32),
        )
        self.assertEqual(
            parameters["position_embedding_table.weight"].shape,
            (8, 32),
        )
        self.assertEqual(parameters["lm_head.weight"].shape, (65, 32))
        self.assertEqual(parameters["lm_head.bias"].shape, (65,))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            4_481,
        )

    def test_backward_reaches_every_learned_component(self) -> None:
        inputs = torch.randint(self.vocab_size, (32, self.block_size))
        targets = torch.randint(self.vocab_size, (32, self.block_size))
        _, loss = self.model(inputs, targets)
        assert loss is not None
        loss.backward()
        parameters = dict(self.model.named_parameters())

        expected_shapes = {
            "token_embedding_table.weight": (65, 32),
            "position_embedding_table.weight": (8, 32),
            "lm_head.weight": (65, 32),
        }
        for name, shape in expected_shapes.items():
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertEqual(tuple(gradient.shape), shape)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_other_tokens_cannot_change_a_positions_logits(self) -> None:
        first = torch.arange(self.block_size) % self.vocab_size
        second = (first + 9) % self.vocab_size
        second[-1] = first[-1]
        inputs = torch.stack((first, second))
        logits, _ = self.model(inputs)

        torch.testing.assert_close(
            logits[0, -1],
            logits[1, -1],
            rtol=0,
            atol=0,
        )

    def test_positions_only_matter_when_enabled(self) -> None:
        no_positions = EmbeddingLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
            use_position_embeddings=False,
        )
        repeated_token = torch.full((1, self.block_size), 7)
        no_position_logits, _ = no_positions(repeated_token)
        position_logits, _ = self.model(repeated_token)

        torch.testing.assert_close(
            no_position_logits[:, 0],
            no_position_logits[:, -1],
            rtol=1e-6,
            atol=1e-6,
        )
        position_difference = (
            position_logits[:, 0] - position_logits[:, -1]
        ).abs().max()
        self.assertGreater(
            position_difference.item(),
            1e-3,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in no_positions.parameters()),
            4_225,
        )

    def test_forward_rejects_context_beyond_block_size(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size + 1))

        with self.assertRaisesRegex(ValueError, "exceeds block_size"):
            self.model(inputs)

    def test_generation_crops_a_long_context(self) -> None:
        context = torch.randint(self.vocab_size, (2, 20))
        generated = self.model.generate(context, max_new_tokens=3)

        self.assertEqual(generated.shape, (2, 23))
        torch.testing.assert_close(generated[:, :20], context)

    def test_initial_loss_is_finite(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        _, loss = self.model(inputs, targets)

        assert loss is not None
        self.assertTrue(math.isfinite(loss.item()))


class DataAndTrainingTests(unittest.TestCase):
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
            batch_size=8,
            block_size=3,
            n_embd=4,
            learning_rate=5e-2,
            max_iters=6,
            eval_interval=3,
            eval_iters=2,
            seed=17,
        )
        self.device = torch.device("cpu")

    def test_batches_are_reproducible_next_token_pairs(self) -> None:
        first_generator = torch.Generator().manual_seed(99)
        second_generator = torch.Generator().manual_seed(99)
        first_inputs, first_targets = self.data.get_batch(
            "train",
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            device=self.device,
            generator=first_generator,
        )
        second_inputs, second_targets = self.data.get_batch(
            "train",
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            device=self.device,
            generator=second_generator,
        )

        torch.testing.assert_close(first_inputs, second_inputs)
        torch.testing.assert_close(first_targets, second_targets)
        torch.testing.assert_close(first_inputs[:, 1:], first_targets[:, :-1])

    def test_evaluation_is_repeatable_with_a_reseeded_generator(self) -> None:
        model = EmbeddingLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
        )
        first = estimate_loss(
            model,
            self.data,
            self.config,
            self.device,
            torch.Generator().manual_seed(123),
        )
        second = estimate_loss(
            model,
            self.data,
            self.config,
            self.device,
            torch.Generator().manual_seed(123),
        )

        self.assertEqual(first, second)

    def test_training_updates_parameters_and_records_expected_steps(self) -> None:
        model = EmbeddingLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
        )
        before = model.token_embedding_table.weight.detach().clone()
        result = train_model(
            model,
            self.data,
            self.config,
            self.device,
            torch.Generator().manual_seed(self.config.seed + 1),
        )

        self.assertEqual([record.step for record in result.history], [0, 3])
        self.assertFalse(
            torch.equal(before, model.token_embedding_table.weight.detach())
        )
        self.assertTrue(math.isfinite(result.initial.train))
        self.assertTrue(math.isfinite(result.final.val))

    def test_ab_models_receive_identical_shared_weights(self) -> None:
        shared = capture_shared_initialization(
            self.data.vocabulary.size,
            self.config,
        )
        model_a = EmbeddingLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
            use_position_embeddings=False,
        )
        model_b = EmbeddingLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
            use_position_embeddings=True,
        )
        apply_shared_initialization(model_a, shared)
        apply_shared_initialization(model_b, shared)

        for name in (
            "token_embedding_table.weight",
            "lm_head.weight",
            "lm_head.bias",
        ):
            torch.testing.assert_close(
                model_a.state_dict()[name],
                model_b.state_dict()[name],
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
