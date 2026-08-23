import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_PATH = Path(__file__).with_name("08_transformer_block.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_8_transformer_block",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_8
SPEC.loader.exec_module(STAGE_8)

Block = STAGE_8.Block
GPTLanguageModel = STAGE_8.GPTLanguageModel
MultiHeadAttention = STAGE_8.MultiHeadAttention


class ProjectedMultiHeadAttentionTests(unittest.TestCase):
    n_embd = 32
    n_head = 4
    block_size = 8

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.attention = MultiHeadAttention(
            n_embd=self.n_embd,
            n_head=self.n_head,
            block_size=self.block_size,
        )

    def test_output_projection_shape_and_parameter_budget(self) -> None:
        self.assertIsInstance(self.attention.proj, nn.Linear)
        self.assertEqual(self.attention.proj.in_features, self.n_embd)
        self.assertEqual(self.attention.proj.out_features, self.n_embd)
        self.assertEqual(self.attention.proj.weight.shape, (32, 32))
        self.assertEqual(self.attention.proj.bias.shape, (32,))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.attention.parameters()),
            3 * self.n_embd * self.n_embd + self.n_embd**2 + self.n_embd,
        )

    def test_projection_is_applied_with_and_without_returned_weights(self) -> None:
        x = torch.randn(3, self.block_size, self.n_embd)
        head_results = [head(x, return_weights=True) for head in self.attention.heads]
        concatenated = torch.cat([result[0] for result in head_results], dim=-1)
        expected_output = self.attention.proj(concatenated)
        expected_weights = torch.stack(
            [result[1] for result in head_results],
            dim=1,
        )

        output = self.attention(x)
        output_with_weights, weights = self.attention(x, return_weights=True)

        assert isinstance(output, torch.Tensor)
        self.assertEqual(output.shape, (3, 8, 32))
        self.assertEqual(weights.shape, (3, 4, 8, 8))
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(output_with_weights, expected_output)
        torch.testing.assert_close(weights, expected_weights)

    def test_projection_can_move_one_heads_output_into_another_slice(self) -> None:
        attention = MultiHeadAttention(n_embd=4, n_head=2, block_size=1)
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.zero_()
            attention.heads[0].value.weight[0, 0] = 1
            attention.proj.weight[3, 0] = 1

        x = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
        output = attention(x)
        assert isinstance(output, torch.Tensor)
        expected = torch.tensor([[[0.0, 0.0, 0.0, 2.0]]])
        torch.testing.assert_close(output, expected)


class TransformerBlockTests(unittest.TestCase):
    n_embd = 32
    n_head = 4
    block_size = 8

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.block = Block(
            n_embd=self.n_embd,
            n_head=self.n_head,
            block_size=self.block_size,
        )

    def test_structure_and_parameter_count(self) -> None:
        self.assertIsInstance(self.block.sa, MultiHeadAttention)
        self.assertIsInstance(self.block.ln1, nn.LayerNorm)
        self.assertIsInstance(self.block.ln2, nn.LayerNorm)
        self.assertIsInstance(self.block.ffwd.net[1], nn.GELU)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.block.parameters()),
            12_608,
        )

    def test_forward_is_exactly_two_pre_norm_residual_updates(self) -> None:
        x = torch.randn(4, self.block_size, self.n_embd)
        attention_update = self.block.sa(self.block.ln1(x))
        assert isinstance(attention_update, torch.Tensor)
        after_attention = x + attention_update
        expected = after_attention + self.block.ffwd(
            self.block.ln2(after_attention)
        )

        actual = self.block(x)

        self.assertEqual(actual.shape, x.shape)
        torch.testing.assert_close(actual, expected)

    def test_zero_updates_make_the_block_an_identity_function(self) -> None:
        with torch.no_grad():
            for parameter in self.block.sa.parameters():
                parameter.zero_()
            for parameter in self.block.ffwd.parameters():
                parameter.zero_()

        x = torch.randn(4, self.block_size, self.n_embd)
        torch.testing.assert_close(self.block(x), x, atol=0, rtol=0)


class GPTLanguageModelTests(unittest.TestCase):
    vocab_size = 65
    block_size = 8
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

    def test_four_independent_blocks_and_final_layer_norm_are_registered(self) -> None:
        self.assertIsInstance(self.model.blocks, nn.ModuleList)
        self.assertEqual(len(self.model.blocks), 4)
        self.assertEqual(len({id(block) for block in self.model.blocks}), 4)
        self.assertEqual(
            len({id(block.sa.proj.weight) for block in self.model.blocks}),
            4,
        )
        self.assertIsInstance(self.model.ln_f, nn.LayerNorm)
        self.assertEqual(self.model.ln_f.normalized_shape, (self.n_embd,))

    def test_forward_matches_the_explicit_stack_final_norm_and_head(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))

        x = self.model._representations(inputs)
        for block in self.model.blocks:
            x = block(x)
        expected_normalized = self.model.ln_f(x)
        expected_logits = self.model.lm_head(expected_normalized)
        expected_loss = F.cross_entropy(
            expected_logits.reshape(4 * 8, 65),
            targets.reshape(4 * 8),
        )

        logits, loss = self.model(inputs, targets)

        self.assertEqual(logits.shape, (4, 8, 65))
        torch.testing.assert_close(logits, expected_logits)
        assert loss is not None
        self.assertTrue(math.isfinite(loss.item()))
        torch.testing.assert_close(loss, expected_loss)

    def test_every_block_output_preserves_b_t_c(self) -> None:
        inputs = torch.randint(self.vocab_size, (32, self.block_size))
        x = self.model._representations(inputs)
        self.assertEqual(x.shape, (32, 8, 32))

        for block in self.model.blocks:
            x = block(x)
            self.assertEqual(x.shape, (32, 8, 32))

        self.assertEqual(self.model.ln_f(x).shape, (32, 8, 32))

    def test_final_layer_norm_has_the_expected_initial_scale(self) -> None:
        inputs = torch.randint(self.vocab_size, (32, self.block_size))
        x = self.model._representations(inputs)
        for block in self.model.blocks:
            x = block(x)
        normalized = self.model.ln_f(x)

        torch.testing.assert_close(
            normalized.mean(dim=-1),
            torch.zeros(32, 8),
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            normalized.var(dim=-1, unbiased=False),
            torch.ones(32, 8),
            atol=2e-5,
            rtol=0,
        )
        self.assertAlmostEqual(
            normalized.norm().item(),
            math.sqrt(32 * 8 * 32),
            places=3,
        )

    def test_exact_default_parameter_count(self) -> None:
        block_counts = [
            sum(parameter.numel() for parameter in block.parameters())
            for block in self.model.blocks
        ]
        self.assertEqual(block_counts, [12_608] * 4)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            54_977,
        )

        parameters = dict(self.model.named_parameters())
        for block_index in range(4):
            self.assertEqual(
                parameters[f"blocks.{block_index}.sa.proj.weight"].shape,
                (32, 32),
            )
            self.assertEqual(
                parameters[f"blocks.{block_index}.sa.proj.bias"].shape,
                (32,),
            )
        self.assertEqual(parameters["ln_f.weight"].shape, (32,))
        self.assertEqual(parameters["ln_f.bias"].shape, (32,))

    def test_attention_helper_propagates_through_preceding_blocks(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size))
        x = self.model._representations(inputs)
        for block in self.model.blocks[:2]:
            x = block(x)
        selected = self.model.blocks[2]
        result = selected.sa(selected.ln1(x), return_weights=True)
        assert isinstance(result, tuple)
        _, expected_weights = result

        actual_weights = self.model.get_attention_weights(inputs, block_index=2)

        self.assertEqual(actual_weights.shape, (2, 4, 8, 8))
        torch.testing.assert_close(actual_weights, expected_weights)
        torch.testing.assert_close(
            actual_weights.sum(dim=-1),
            torch.ones(2, 4, 8),
        )
        self.assertTrue(
            torch.equal(
                actual_weights.triu(diagonal=1),
                torch.zeros_like(actual_weights),
            )
        )

    def test_future_tokens_cannot_change_earlier_logits_through_four_blocks(self) -> None:
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

    def test_backward_reaches_a_query_in_every_block(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        _, loss = self.model(inputs, targets)
        assert loss is not None
        loss.backward()

        for block_index, block in enumerate(self.model.blocks):
            gradient = block.sa.heads[0].query.weight.grad
            with self.subTest(block=block_index):
                self.assertIsNotNone(gradient)
                assert gradient is not None
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_generation_crops_a_long_context(self) -> None:
        context = torch.randint(self.vocab_size, (2, 20))
        generated = self.model.generate(context, max_new_tokens=3)

        self.assertEqual(generated.shape, (2, 23))
        torch.testing.assert_close(generated[:, :20], context)

    def test_invalid_depth_head_split_and_context_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_layer must be positive"):
            GPTLanguageModel(
                vocab_size=self.vocab_size,
                block_size=self.block_size,
                n_embd=self.n_embd,
                n_head=self.n_head,
                n_layer=0,
            )

        with self.assertRaisesRegex(ValueError, "must be divisible"):
            GPTLanguageModel(
                vocab_size=self.vocab_size,
                block_size=self.block_size,
                n_embd=30,
                n_head=self.n_head,
                n_layer=self.n_layer,
            )

        with self.assertRaisesRegex(ValueError, "exceeds block_size"):
            self.model(torch.randint(self.vocab_size, (2, self.block_size + 1)))

        with self.assertRaisesRegex(IndexError, "block_index"):
            self.model.get_attention_weights(
                torch.randint(self.vocab_size, (2, self.block_size)),
                block_index=4,
            )


if __name__ == "__main__":
    unittest.main()
