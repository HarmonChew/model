import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).with_name("04_single_head_attention.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_3_single_head_attention",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_3
SPEC.loader.exec_module(STAGE_3)

Head = STAGE_3.Head
SingleHeadAttentionLanguageModel = STAGE_3.SingleHeadAttentionLanguageModel


class HeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.head = Head(n_embd=32, head_size=32, block_size=8)

    def test_qkv_scores_weights_and_output_shapes(self) -> None:
        x = torch.randn(4, 8, 32)
        q = self.head.query(x)
        k = self.head.key(x)
        v = self.head.value(x)
        scores = q @ k.transpose(-2, -1)
        out, weights = self.head(x, return_weights=True)

        self.assertEqual(q.shape, (4, 8, 32))
        self.assertEqual(k.shape, (4, 8, 32))
        self.assertEqual(v.shape, (4, 8, 32))
        self.assertEqual(scores.shape, (4, 8, 8))
        self.assertEqual(weights.shape, (4, 8, 8))
        self.assertEqual(out.shape, (4, 8, 32))

    def test_attention_is_normalized_and_strictly_causal(self) -> None:
        x = torch.randn(3, 8, 32)
        _, weights = self.head(x, return_weights=True)

        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(3, 8),
        )
        self.assertTrue(
            torch.equal(
                weights.triu(diagonal=1),
                torch.zeros_like(weights),
            )
        )
        torch.testing.assert_close(weights[:, 0, 0], torch.ones(3))

    def test_forward_matches_manual_scaled_attention(self) -> None:
        x = torch.randn(2, 5, 32)
        q = self.head.query(x)
        k = self.head.key(x)
        v = self.head.value(x)
        scores = (q @ k.transpose(-2, -1)) * (32**-0.5)
        scores = scores.masked_fill(
            self.head.tril[:5, :5] == 0,
            float("-inf"),
        )
        expected_weights = F.softmax(scores, dim=-1)
        expected_output = expected_weights @ v

        output, weights = self.head(x, return_weights=True)
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(output, expected_output)

    def test_mask_is_a_buffer_not_a_parameter(self) -> None:
        self.assertNotIn("tril", dict(self.head.named_parameters()))
        self.assertIn("tril", dict(self.head.named_buffers()))
        self.assertFalse(self.head.tril.requires_grad)


class SingleHeadAttentionLanguageModelTests(unittest.TestCase):
    vocab_size = 65
    block_size = 8
    n_embd = 32

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.model = SingleHeadAttentionLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
        )

    def test_forward_shape_loss_and_parameter_count(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        logits, loss = self.model(inputs, targets)

        self.assertEqual(logits.shape, (4, 8, 65))
        assert loss is not None
        self.assertTrue(math.isfinite(loss.item()))
        expected_loss = F.cross_entropy(
            logits.reshape(4 * 8, 65),
            targets.reshape(4 * 8),
        )
        torch.testing.assert_close(loss, expected_loss)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            7_553,
        )

    def test_parameter_shapes(self) -> None:
        parameters = dict(self.model.named_parameters())
        expected = {
            "token_embedding_table.weight": (65, 32),
            "position_embedding_table.weight": (8, 32),
            "sa_head.key.weight": (32, 32),
            "sa_head.query.weight": (32, 32),
            "sa_head.value.weight": (32, 32),
            "lm_head.weight": (65, 32),
            "lm_head.bias": (65,),
        }

        self.assertEqual(
            {name: tuple(parameter.shape) for name, parameter in parameters.items()},
            expected,
        )

    def test_get_attention_weights_matches_head(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size))
        positions = torch.arange(self.block_size)
        x = self.model.token_embedding_table(inputs)
        x = x + self.model.position_embedding_table(positions)
        _, expected = self.model.sa_head(x, return_weights=True)
        actual = self.model.get_attention_weights(inputs)

        torch.testing.assert_close(actual, expected)

    def test_prefix_tokens_can_change_final_logits(self) -> None:
        # With equal scores, attention becomes the Stage 2 causal average.
        # This deterministic setup gives the first source token a clear path
        # through V and the language-model head to the final logits.
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.token_embedding_table.weight[1, 0] = self.block_size
            self.model.sa_head.value.weight[0, 0] = 1
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

    def test_backward_reaches_every_learned_component(self) -> None:
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
