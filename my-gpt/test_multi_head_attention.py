import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_PATH = Path(__file__).with_name("05_multi_head_attention.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_4_multi_head_attention",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_4
SPEC.loader.exec_module(STAGE_4)

MultiHeadAttention = STAGE_4.MultiHeadAttention
MultiHeadAttentionLanguageModel = STAGE_4.MultiHeadAttentionLanguageModel


class MultiHeadAttentionTests(unittest.TestCase):
    n_embd = 32
    n_head = 4
    head_size = 8
    block_size = 8

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.attention = MultiHeadAttention(
            n_embd=self.n_embd,
            n_head=self.n_head,
            block_size=self.block_size,
        )

    def test_heads_are_registered_independent_modules(self) -> None:
        self.assertIsInstance(self.attention.heads, nn.ModuleList)
        self.assertEqual(len(self.attention.heads), self.n_head)
        self.assertEqual(self.attention.head_size, self.head_size)
        self.assertEqual(
            len({id(head.query.weight) for head in self.attention.heads}),
            self.n_head,
        )

        for head in self.attention.heads:
            self.assertEqual(head.query.weight.shape, (8, 32))
            self.assertEqual(head.key.weight.shape, (8, 32))
            self.assertEqual(head.value.weight.shape, (8, 32))

    def test_concatenates_outputs_and_stacks_weights(self) -> None:
        x = torch.randn(3, self.block_size, self.n_embd)
        expected_outputs = []
        expected_weights = []

        for head in self.attention.heads:
            output, weights = head(x, return_weights=True)
            expected_outputs.append(output)
            expected_weights.append(weights)

        output, weights = self.attention(x, return_weights=True)

        self.assertEqual(output.shape, (3, 8, 32))
        self.assertEqual(weights.shape, (3, 4, 8, 8))
        torch.testing.assert_close(
            output,
            torch.cat(expected_outputs, dim=-1),
        )
        torch.testing.assert_close(
            weights,
            torch.stack(expected_weights, dim=1),
        )

    def test_every_head_is_normalized_and_causally_masked(self) -> None:
        x = torch.randn(3, self.block_size, self.n_embd)
        _, weights = self.attention(x, return_weights=True)

        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(3, 4, 8),
        )
        self.assertTrue(
            torch.equal(
                weights.triu(diagonal=1),
                torch.zeros_like(weights),
            )
        )

    def test_each_head_matches_manual_scaled_attention(self) -> None:
        x = torch.randn(2, 5, self.n_embd)

        for head in self.attention.heads:
            q = head.query(x)
            k = head.key(x)
            v = head.value(x)
            scores = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
            scores = scores.masked_fill(
                head.tril[:5, :5] == 0,
                float("-inf"),
            )
            expected_weights = F.softmax(scores, dim=-1)
            expected_output = expected_weights @ v

            output, weights = head(x, return_weights=True)
            torch.testing.assert_close(weights, expected_weights)
            torch.testing.assert_close(output, expected_output)

    def test_qkv_parameter_budget_matches_one_wide_head(self) -> None:
        self.assertEqual(
            sum(parameter.numel() for parameter in self.attention.parameters()),
            3 * self.n_embd * self.n_embd,
        )
        self.assertEqual(len(dict(self.attention.named_buffers())), self.n_head)

    def test_embedding_width_must_split_evenly_across_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            MultiHeadAttention(n_embd=30, n_head=4, block_size=8)


class MultiHeadAttentionLanguageModelTests(unittest.TestCase):
    vocab_size = 65
    block_size = 8
    n_embd = 32
    n_head = 4

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.model = MultiHeadAttentionLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
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
            "lm_head.weight": (65, 32),
            "lm_head.bias": (65,),
        }

        for head_index in range(self.n_head):
            for projection in ("key", "query", "value"):
                expected[f"sa.heads.{head_index}.{projection}.weight"] = (8, 32)

        self.assertEqual(
            {name: tuple(parameter.shape) for name, parameter in parameters.items()},
            expected,
        )

    def test_get_attention_weights_has_batch_head_time_time_shape(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size))
        positions = torch.arange(self.block_size)
        x = self.model.token_embedding_table(inputs)
        x = x + self.model.position_embedding_table(positions)
        _, expected = self.model.sa(x, return_weights=True)
        actual = self.model.get_attention_weights(inputs)

        self.assertEqual(actual.shape, (2, 4, 8, 8))
        torch.testing.assert_close(actual, expected)

    def test_prefix_tokens_can_change_final_logits(self) -> None:
        # Zero Q/K makes every head a causal average. Build one exact path
        # through value channel 0 of head 0 into vocabulary logit 0.
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.token_embedding_table.weight[1, 0] = self.block_size
            self.model.sa.heads[0].value.weight[0, 0] = 1
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

    def test_backward_reaches_every_head_and_learned_component(self) -> None:
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
