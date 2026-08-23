import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_PATH = Path(__file__).with_name("05_feed_forward.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_5_feed_forward",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_5
SPEC.loader.exec_module(STAGE_5)

FeedForward = STAGE_5.FeedForward
FeedForwardLanguageModel = STAGE_5.FeedForwardLanguageModel


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


class FeedForwardLanguageModelTests(unittest.TestCase):
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

    def test_forward_is_attention_then_ffn_then_lm_head(self) -> None:
        inputs = torch.randint(self.vocab_size, (4, self.block_size))
        targets = torch.randint(self.vocab_size, (4, self.block_size))
        logits, loss = self.model(inputs, targets)

        representations = self.model._representations(inputs)
        attention_output = self.model.sa(representations)
        expected_logits = self.model.lm_head(self.model.ffwd(attention_output))
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
