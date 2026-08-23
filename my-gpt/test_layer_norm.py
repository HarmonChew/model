import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_PATH = Path(__file__).with_name("07_layer_norm.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_7_layer_norm",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_7 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_7
SPEC.loader.exec_module(STAGE_7)

FeedForward = STAGE_7.FeedForward
FeedForwardLanguageModel = STAGE_7.FeedForwardLanguageModel


class FeedForwardTests(unittest.TestCase):
    n_embd = 32

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.ffwd = FeedForward(self.n_embd)

    def test_structure_intermediate_shapes_and_parameter_count(self) -> None:
        self.assertIsInstance(self.ffwd.net, nn.Sequential)
        self.assertIsInstance(self.ffwd.net[0], nn.Linear)
        self.assertIsInstance(self.ffwd.net[1], nn.GELU)
        self.assertIsInstance(self.ffwd.net[2], nn.Linear)

        x = torch.randn(4, 8, self.n_embd)
        hidden = self.ffwd.net[0](x)
        activated = self.ffwd.net[1](hidden)
        output = self.ffwd.net[2](activated)

        self.assertEqual(hidden.shape, (4, 8, 128))
        self.assertEqual(activated.shape, (4, 8, 128))
        self.assertEqual(output.shape, (4, 8, 32))
        torch.testing.assert_close(self.ffwd(x), output)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.ffwd.parameters()),
            8_352,
        )

    def test_changing_one_position_cannot_change_other_positions(self) -> None:
        with torch.no_grad():
            for parameter in self.ffwd.parameters():
                parameter.zero_()
            self.ffwd.net[0].weight[0, 0] = 1
            self.ffwd.net[2].weight[0, 0] = 1

        first = torch.zeros(1, 8, self.n_embd)
        second = first.clone()
        second[:, 0, 0] = 1
        first_output = self.ffwd(first)
        second_output = self.ffwd(second)
        difference = (first_output - second_output).abs()

        torch.testing.assert_close(
            difference[:, 0, 0],
            F.gelu(torch.tensor(1.0)).reshape(1),
        )
        self.assertTrue(
            torch.equal(
                difference[:, 1:, :],
                torch.zeros_like(difference[:, 1:, :]),
            )
        )

    def test_embedding_width_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            FeedForward(0)


class LayerNormLanguageModelTests(unittest.TestCase):
    vocab_size = 65
    block_size = 8
    n_embd = 32
    n_head = 4

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.model = FeedForwardLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
        )

    def test_forward_uses_both_pre_norm_residual_updates(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        logits, loss = self.model(inputs, targets)

        x0 = self.model._representations(inputs)
        norm1 = self.model.ln1(x0)
        attention_update = self.model.sa(norm1)
        self.assertIsInstance(attention_update, torch.Tensor)
        assert isinstance(attention_update, torch.Tensor)
        x1 = x0 + attention_update
        norm2 = self.model.ln2(x1)
        ff_update = self.model.ffwd(norm2)
        x2 = x1 + ff_update
        expected_logits = self.model.lm_head(x2)
        expected_loss = F.cross_entropy(
            expected_logits.reshape(4 * 8, 65),
            targets.reshape(4 * 8),
        )

        self.assertEqual(logits.shape, (4, 8, 65))
        torch.testing.assert_close(logits, expected_logits)
        assert loss is not None
        self.assertTrue(math.isfinite(loss.item()))
        torch.testing.assert_close(loss, expected_loss)

    def test_pre_norm_flow_has_controlled_exact_values(self) -> None:
        model = FeedForwardLanguageModel(
            vocab_size=2,
            block_size=1,
            n_embd=4,
            n_head=1,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.token_embedding_table.weight[0, 0] = 2
            model.ln1.bias[0] = 5
            model.sa.heads[0].value.weight[0, 0] = 3
            model.ln2.bias[0] = 7
            model.ffwd.net[0].weight[0, 0] = 2
            model.ffwd.net[2].weight[0, 0] = 3
            model.lm_head.weight[0, 0] = 1

        inputs = torch.tensor([[0]])
        x0 = model._representations(inputs)
        norm1 = model.ln1(x0)
        attention_update = model.sa(norm1)
        assert isinstance(attention_update, torch.Tensor)
        x1 = x0 + attention_update
        norm2 = model.ln2(x1)
        hidden = model.ffwd.net[0](norm2)
        activated = model.ffwd.net[1](hidden)
        ff_update = model.ffwd.net[2](activated)
        x2 = x1 + ff_update
        logits, _ = model(inputs)

        gelu_14 = F.gelu(torch.tensor(14.0)).item()
        expected_first_channels = (
            2.0,
            5.0,
            15.0,
            17.0,
            7.0,
            14.0,
            gelu_14,
            3 * gelu_14,
            17 + 3 * gelu_14,
        )
        actual_vectors = (
            x0,
            norm1,
            attention_update,
            x1,
            norm2,
            hidden,
            activated,
            ff_update,
            x2,
        )
        widths = (4, 4, 4, 4, 4, 16, 16, 4, 4)

        for actual, expected_first_channel, width in zip(
            actual_vectors,
            expected_first_channels,
            widths,
            strict=True,
        ):
            expected = torch.zeros(1, 1, width)
            expected[0, 0, 0] = expected_first_channel
            torch.testing.assert_close(actual, expected)

        expected_logits = torch.tensor([[[17 + 3 * gelu_14, 0.0]]])
        torch.testing.assert_close(logits, expected_logits)

    def test_every_pre_norm_checkpoint_preserves_b_t_c(self) -> None:
        inputs = torch.randint(self.vocab_size, (32, self.block_size))
        x0 = self.model._representations(inputs)
        norm1 = self.model.ln1(x0)
        attention_update = self.model.sa(norm1)
        assert isinstance(attention_update, torch.Tensor)
        x1 = x0 + attention_update
        norm2 = self.model.ln2(x1)
        hidden = self.model.ffwd.net[0](norm2)
        activated = self.model.ffwd.net[1](hidden)
        ff_update = self.model.ffwd.net[2](activated)
        x2 = x1 + ff_update

        expected_shape = (32, 8, 32)
        for tensor in (
            x0,
            norm1,
            attention_update,
            x1,
            norm2,
            ff_update,
            x2,
        ):
            self.assertEqual(tensor.shape, expected_shape)
        self.assertEqual(hidden.shape, (32, 8, 128))
        self.assertEqual(activated.shape, (32, 8, 128))

    def test_layer_norm_shapes_and_initial_affine_parameters(self) -> None:
        for layer_norm in (self.model.ln1, self.model.ln2):
            self.assertIsInstance(layer_norm, nn.LayerNorm)
            self.assertEqual(layer_norm.normalized_shape, (self.n_embd,))
            torch.testing.assert_close(
                layer_norm.weight,
                torch.ones(self.n_embd),
            )
            torch.testing.assert_close(
                layer_norm.bias,
                torch.zeros(self.n_embd),
            )

    def test_layer_norm_normalizes_each_token_across_features(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        x = torch.randn(3, 5, self.n_embd, generator=generator) * 10 + 5
        normalized = self.model.ln1(x)

        torch.testing.assert_close(
            normalized.mean(dim=-1),
            torch.zeros(3, 5),
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            normalized.var(dim=-1, unbiased=False),
            torch.ones(3, 5),
            atol=2e-6,
            rtol=0,
        )

    def test_layer_norm_does_not_mix_tokens_or_batch_examples(self) -> None:
        generator = torch.Generator().manual_seed(2027)
        first = torch.randn(2, 4, self.n_embd, generator=generator)
        second = first.clone()
        second[1, 3] += torch.linspace(0, 10, self.n_embd)

        first_output = self.model.ln1(first)
        second_output = self.model.ln1(second)

        torch.testing.assert_close(
            first_output[0],
            second_output[0],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            first_output[1, :3],
            second_output[1, :3],
            atol=0,
            rtol=0,
        )
        self.assertGreater(
            (first_output[1, 3] - second_output[1, 3]).abs().max().item(),
            0,
        )

    def test_layer_norm_affine_transform_matches_the_formula(self) -> None:
        x = torch.tensor([[[1.0, 2.0, 4.0, 8.0] + [3.0] * 28]])
        with torch.no_grad():
            self.model.ln1.weight.copy_(torch.linspace(0.5, 1.5, self.n_embd))
            self.model.ln1.bias.copy_(torch.linspace(-1.0, 1.0, self.n_embd))

        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        standardized = (x - mean) / torch.sqrt(variance + self.model.ln1.eps)
        expected = standardized * self.model.ln1.weight + self.model.ln1.bias

        torch.testing.assert_close(self.model.ln1(x), expected)

    def test_zero_updates_leave_the_residual_stream_untouched(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))

        with torch.no_grad():
            for parameter in self.model.sa.parameters():
                parameter.zero_()
            for parameter in self.model.ffwd.parameters():
                parameter.zero_()

        x0 = self.model._representations(inputs)
        logits, _ = self.model(inputs)
        expected_logits = self.model.lm_head(x0)
        torch.testing.assert_close(logits, expected_logits)

    def test_parameter_shapes_and_count(self) -> None:
        parameters = dict(self.model.named_parameters())
        expected = {
            "token_embedding_table.weight": (65, 32),
            "position_embedding_table.weight": (8, 32),
            "ffwd.net.0.weight": (128, 32),
            "ffwd.net.0.bias": (128,),
            "ffwd.net.2.weight": (32, 128),
            "ffwd.net.2.bias": (32,),
            "ln1.weight": (32,),
            "ln1.bias": (32,),
            "ln2.weight": (32,),
            "ln2.bias": (32,),
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
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            16_033,
        )

    def test_attention_remains_normalized_and_causally_masked(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size))
        weights = self.model.get_attention_weights(inputs)

        self.assertEqual(weights.shape, (2, 4, 8, 8))
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(2, 4, 8),
        )
        self.assertTrue(
            torch.equal(
                weights.triu(diagonal=1),
                torch.zeros_like(weights),
            )
        )

    def test_attention_weight_helper_uses_the_normalized_input(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size))
        x0 = self.model._representations(inputs)
        result = self.model.sa(self.model.ln1(x0), return_weights=True)
        assert isinstance(result, tuple)
        _, expected_weights = result

        torch.testing.assert_close(
            self.model.get_attention_weights(inputs),
            expected_weights,
        )

    def test_prefix_tokens_can_still_change_final_logits(self) -> None:
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.ln1.weight.fill_(1)
            self.model.token_embedding_table.weight[1, 0] = self.block_size
            self.model.sa.heads[0].value.weight[0, 0] = 1
            self.model.lm_head.weight[0, 0] = 1

        first = torch.zeros(self.block_size, dtype=torch.long)
        second = first.clone()
        second[0] = 1
        logits, _ = self.model(torch.stack((first, second)))

        self.assertGreater(logits[1, -1, 0] - logits[0, -1, 0], 0)

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

    def test_backward_reaches_every_trainable_component(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        _, loss = self.model(inputs, targets)
        assert loss is not None
        loss.backward()

        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                assert parameter.grad is not None
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_direct_residual_route_carries_gradient_when_updates_are_zero(self) -> None:
        with torch.no_grad():
            for parameter in self.model.sa.parameters():
                parameter.zero_()
            for parameter in self.model.ffwd.parameters():
                parameter.zero_()

        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        _, loss = self.model(inputs, targets)
        assert loss is not None
        loss.backward()

        for name in (
            "token_embedding_table.weight",
            "position_embedding_table.weight",
            "lm_head.weight",
        ):
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_forward_rejects_context_beyond_block_size(self) -> None:
        inputs = torch.randint(self.vocab_size, (2, self.block_size + 1))

        with self.assertRaisesRegex(ValueError, "exceeds block_size"):
            self.model(inputs)

    def test_generation_crops_a_long_context(self) -> None:
        context = torch.randint(self.vocab_size, (2, 20))
        generated = self.model.generate(context, max_new_tokens=3)

        self.assertEqual(generated.shape, (2, 23))
        torch.testing.assert_close(generated[:, :20], context)

    def test_embedding_width_must_split_evenly_across_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            FeedForwardLanguageModel(
                vocab_size=self.vocab_size,
                block_size=self.block_size,
                n_embd=30,
                n_head=4,
            )


if __name__ == "__main__":
    unittest.main()
