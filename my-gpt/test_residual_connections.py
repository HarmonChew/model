import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_PATH = Path(__file__).with_name("06_residual_connections.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_6_residual_connections",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_6
SPEC.loader.exec_module(STAGE_6)

FeedForward = STAGE_6.FeedForward
FeedForwardLanguageModel = STAGE_6.FeedForwardLanguageModel


class FeedForwardTests(unittest.TestCase):
    n_embd = 32

    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.ffwd = FeedForward(self.n_embd)

    def test_structure_intermediate_shapes_and_parameter_count(self) -> None:
        self.assertIsInstance(self.ffwd.net, nn.Sequential)
        self.assertIsInstance(self.ffwd.net[0], nn.Linear)
        self.assertIsInstance(self.ffwd.net[1], nn.ReLU)
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
        # Construct one deterministic positive path through both linear layers.
        with torch.no_grad():
            for parameter in self.ffwd.parameters():
                parameter.zero_()
            self.ffwd.net[0].weight[0, 0] = 1
            self.ffwd.net[2].weight[0, 0] = 1

        first = torch.zeros(1, 8, self.n_embd)
        second = first.clone()
        second[:, 0, 0] = 10
        first_output = self.ffwd(first)
        second_output = self.ffwd(second)
        difference = (first_output - second_output).abs()

        self.assertEqual(difference[:, 0, :].max().item(), 10.0)
        self.assertTrue(
            torch.equal(
                difference[:, 1:, :],
                torch.zeros_like(difference[:, 1:, :]),
            )
        )

    def test_embedding_width_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            FeedForward(0)


class ResidualConnectionLanguageModelTests(unittest.TestCase):
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

    def test_forward_adds_attention_and_ffn_updates_before_lm_head(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        logits, loss = self.model(inputs, targets)

        x0 = self.model._representations(inputs)
        attention_update = self.model.sa(x0)
        self.assertIsInstance(attention_update, torch.Tensor)
        assert isinstance(attention_update, torch.Tensor)
        x1 = x0 + attention_update
        ff_update = self.model.ffwd(x1)
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
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            15_905,
        )

    def test_residual_flow_has_expected_exact_values(self) -> None:
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
            model.sa.heads[0].value.weight[0, 0] = 3
            model.ffwd.net[0].weight[0, 0] = 2
            model.ffwd.net[2].weight[0, 0] = 3
            model.lm_head.weight[0, 0] = 1

        inputs = torch.tensor([[0]])
        x0 = model._representations(inputs)
        attention_update = model.sa(x0)
        assert isinstance(attention_update, torch.Tensor)
        x1 = x0 + attention_update
        ff_update = model.ffwd(x1)
        x2 = x1 + ff_update
        logits, _ = model(inputs)

        expected_vectors = (2.0, 6.0, 8.0, 48.0, 56.0)
        actual_vectors = (x0, attention_update, x1, ff_update, x2)
        for actual, expected_first_channel in zip(
            actual_vectors,
            expected_vectors,
            strict=True,
        ):
            expected = torch.tensor([[[expected_first_channel, 0, 0, 0]]])
            torch.testing.assert_close(actual, expected)

        torch.testing.assert_close(logits, torch.tensor([[[56.0, 0.0]]]))

    def test_every_residual_checkpoint_preserves_b_t_c(self) -> None:
        inputs = torch.randint(self.vocab_size, (32, self.block_size))
        x0 = self.model._representations(inputs)
        attention_update = self.model.sa(x0)
        assert isinstance(attention_update, torch.Tensor)
        x1 = x0 + attention_update
        ff_update = self.model.ffwd(x1)
        x2 = x1 + ff_update

        expected_shape = (32, 8, 32)
        self.assertEqual(x0.shape, expected_shape)
        self.assertEqual(attention_update.shape, expected_shape)
        self.assertEqual(x1.shape, expected_shape)
        self.assertEqual(ff_update.shape, expected_shape)
        self.assertEqual(x2.shape, expected_shape)

    def test_zero_updates_make_both_sublayers_identity_paths(self) -> None:
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

    def test_parameter_shapes(self) -> None:
        parameters = dict(self.model.named_parameters())
        expected = {
            "token_embedding_table.weight": (65, 32),
            "position_embedding_table.weight": (8, 32),
            "ffwd.net.0.weight": (128, 32),
            "ffwd.net.0.bias": (128,),
            "ffwd.net.2.weight": (32, 128),
            "ffwd.net.2.bias": (32,),
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

    def test_prefix_tokens_can_still_change_final_logits(self) -> None:
        # Build one exact positive path through attention, the FFN, and lm_head.
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.token_embedding_table.weight[1, 0] = self.block_size
            self.model.sa.heads[0].value.weight[0, 0] = 1
            self.model.ffwd.net[0].weight[0, 0] = 1
            self.model.ffwd.net[2].weight[0, 0] = 1
            self.model.lm_head.weight[0, 0] = 1

        first = torch.zeros(self.block_size, dtype=torch.long)
        second = first.clone()
        second[0] = 1
        logits, _ = self.model(torch.stack((first, second)))

        self.assertEqual(logits[1, -1, 0] - logits[0, -1, 0], 2.0)

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

    def test_backward_reaches_attention_ffn_embeddings_and_head(self) -> None:
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
