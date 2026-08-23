import math
import unittest

import torch
import torch.nn.functional as F

from causal_average_model import CausalAverageLanguageModel
from model import EmbeddingLanguageModel


class CausalAverageLanguageModelTests(unittest.TestCase):
    vocab_size = 65
    block_size = 8
    n_embd = 32

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.model = CausalAverageLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
        )

    def test_forward_matches_manual_causal_average(self) -> None:
        inputs = torch.randint(self.vocab_size, (3, self.block_size))
        targets = torch.randint(self.vocab_size, (3, self.block_size))
        logits, loss = self.model(inputs, targets)

        token_embeddings = self.model.token_embedding_table(inputs)
        positions = torch.arange(self.block_size)
        position_embeddings = self.model.position_embedding_table(positions)
        x = token_embeddings + position_embeddings
        weights = self.model.tril / self.model.tril.sum(dim=1, keepdim=True)
        expected_logits = self.model.lm_head(weights @ x)
        expected_loss = F.cross_entropy(
            expected_logits.reshape(3 * self.block_size, self.vocab_size),
            targets.reshape(3 * self.block_size),
        )

        self.assertEqual(logits.shape, (3, 8, 65))
        torch.testing.assert_close(logits, expected_logits)
        assert loss is not None
        torch.testing.assert_close(loss, expected_loss)

    def test_weights_are_uniform_normalized_and_causal(self) -> None:
        weights = self.model.tril / self.model.tril.sum(dim=1, keepdim=True)

        torch.testing.assert_close(weights.sum(dim=1), torch.ones(8))
        self.assertTrue(
            torch.equal(weights.triu(diagonal=1), torch.zeros(8, 8))
        )
        torch.testing.assert_close(
            weights[3],
            torch.tensor([0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]),
        )

    def test_tril_is_a_buffer_not_a_parameter(self) -> None:
        parameters = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())

        self.assertNotIn("tril", parameters)
        self.assertIn("tril", buffers)
        self.assertIn("tril", self.model.state_dict())
        self.assertEqual(buffers["tril"].shape, (8, 8))
        self.assertFalse(buffers["tril"].requires_grad)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            4_481,
        )

    def test_prefix_change_has_a_known_effect_on_final_logits(self) -> None:
        # Build a tiny deterministic information path. Token 1 contributes 8
        # in channel 0; averaging it into position 7 contributes exactly 1.
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.token_embedding_table.weight[1, 0] = self.block_size
            self.model.lm_head.weight[0, 0] = 1

        first = torch.zeros(self.block_size, dtype=torch.long)
        second = first.clone()
        second[0] = 1
        logits, _ = self.model(torch.stack((first, second)))

        self.assertEqual(logits[1, -1, 0] - logits[0, -1, 0], 1.0)

    def test_future_tokens_cannot_change_earlier_logits(self) -> None:
        first = torch.arange(self.block_size) % self.vocab_size
        second = first.clone()
        second[4:] = (second[4:] + 11) % self.vocab_size
        logits, _ = self.model(torch.stack((first, second)))

        torch.testing.assert_close(
            logits[0, :4],
            logits[1, :4],
            rtol=0,
            atol=0,
        )

    def test_matched_baseline_remains_token_local(self) -> None:
        torch.manual_seed(2026)
        baseline = EmbeddingLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
        )
        causal = CausalAverageLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
        )
        causal.load_state_dict(baseline.state_dict(), strict=False)

        first = torch.arange(self.block_size) % self.vocab_size
        second = (first + 3) % self.vocab_size
        second[-1] = first[-1]
        inputs = torch.stack((first, second))
        baseline_logits, _ = baseline(inputs)
        causal_logits, _ = causal(inputs)

        torch.testing.assert_close(
            baseline_logits[0, -1],
            baseline_logits[1, -1],
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            (causal_logits[0, -1] - causal_logits[1, -1])
            .abs()
            .max()
            .item(),
            0.0,
        )

    def test_backward_reaches_all_learned_components(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        _, loss = self.model(inputs, targets)
        assert loss is not None
        loss.backward()

        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        self.assertTrue(math.isfinite(loss.item()))

    def test_forward_rejects_context_beyond_block_size(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size + 1))

        with self.assertRaisesRegex(ValueError, "exceeds block_size"):
            self.model(inputs)

    def test_generation_crops_a_long_context(self) -> None:
        context = torch.randint(self.vocab_size, (2, 20))
        generated = self.model.generate(context, max_new_tokens=3)

        self.assertEqual(generated.shape, (2, 23))
        torch.testing.assert_close(generated[:, :20], context)


if __name__ == "__main__":
    unittest.main()
