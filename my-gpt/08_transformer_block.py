from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import EvaluationRecord, TrainingResult, seed_everything, train_model


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
    ) -> None:
        super().__init__()

        if n_embd <= 0 or head_size <= 0 or block_size <= 0:
            raise ValueError(
                "n_embd, head_size, and block_size must all be positive"
            )

        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (B, T, C), got {tuple(x.shape)}"
            )

        _, sequence_length, channels = x.shape

        if sequence_length == 0:
            raise ValueError("x must contain at least one token")

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size "
                f"{self.block_size}"
            )

        if channels != self.n_embd:
            raise ValueError(
                f"x has {channels} channels, but this head expects "
                f"{self.n_embd}"
            )

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1)
        weights = weights * (self.head_size**-0.5)
        weights = weights.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf"),
        )
        weights = F.softmax(weights, dim=-1)
        out = weights @ v

        if return_weights:
            return out, weights

        return out


class MultiHeadAttention(nn.Module):
    """Concatenate causal heads, then mix them with an output projection."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
    ) -> None:
        super().__init__()

        if n_embd <= 0 or n_head <= 0 or block_size <= 0:
            raise ValueError("n_embd, n_head, and block_size must be positive")

        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [
                Head(
                    n_embd=n_embd,
                    head_size=self.head_size,
                    block_size=block_size,
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_weights:
            outputs = []
            weights = []

            for head in self.heads:
                result = head(x, return_weights=True)
                assert isinstance(result, tuple)
                head_output, head_weights = result
                outputs.append(head_output)
                weights.append(head_weights)

            concatenated = torch.cat(outputs, dim=-1)
            return self.proj(concatenated), torch.stack(weights, dim=1)

        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(concatenated)


class FeedForward(nn.Module):
    """Process each position independently across its feature channels."""

    def __init__(self, n_embd: int) -> None:
        super().__init__()

        if n_embd <= 0:
            raise ValueError("n_embd must be positive")

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One pre-norm Transformer block with two residual updates."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
    ) -> None:
        super().__init__()

        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
        )
        self.ffwd = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Communication between positions.
        attention_update = self.sa(self.ln1(x))
        assert isinstance(attention_update, torch.Tensor)
        x = x + attention_update

        # Computation within each position.
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """A small GPT-shaped, decoder-only character language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
    ) -> None:
        super().__init__()

        positive_values = {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
            )

        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _representations(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(
                f"idx must have shape (B, T), got {tuple(idx.shape)}"
            )

        _, sequence_length = idx.shape

        if sequence_length == 0:
            raise ValueError("idx must contain at least one token")

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size "
                f"{self.block_size}"
            )

        token_embeddings = self.token_embedding_table(idx)
        positions = torch.arange(sequence_length, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)
        return token_embeddings + position_embeddings

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if targets is not None and targets.shape != idx.shape:
            raise ValueError(
                "targets must have the same (B, T) shape as idx; "
                f"got {tuple(targets.shape)} and {tuple(idx.shape)}"
            )

        x = self._representations(idx)
        batch_size, sequence_length, _ = x.shape

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            _, _, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, vocab_size),
                targets.reshape(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def get_attention_weights(
        self,
        idx: torch.Tensor,
        block_index: int = 0,
    ) -> torch.Tensor:
        if not 0 <= block_index < len(self.blocks):
            raise IndexError(
                f"block_index must be in [0, {len(self.blocks) - 1}], "
                f"got {block_index}"
            )

        x = self._representations(idx)
        for block in self.blocks[:block_index]:
            x = block(x)

        selected_block = self.blocks[block_index]
        result = selected_block.sa(
            selected_block.ln1(x),
            return_weights=True,
        )
        assert isinstance(result, tuple)
        _, weights = result
        return weights

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("idx must have non-empty shape (B, T)")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            final_logits = logits[:, -1, :]
            probabilities = F.softmax(final_logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


@dataclass(frozen=True, slots=True)
class ShapeAndNormReport:
    embedding_shape: tuple[int, ...]
    block_shapes: tuple[tuple[int, ...], ...]
    final_layer_norm_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    embedding_norm: float
    block_norms: tuple[float, ...]
    final_layer_norm_norm: float


@dataclass(frozen=True, slots=True)
class GradientReport:
    query_gradient_norms: tuple[float, ...]
    all_finite: bool
    all_nonzero: bool


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    parameter_count: int
    shapes_and_norms: ShapeAndNormReport
    gradients: GradientReport
    training: TrainingResult
    sample: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a four-block GPT-shaped character model."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def print_parameter_report(model: GPTLanguageModel) -> int:
    parameter_count = _parameter_count(model)
    token_count = _parameter_count(model.token_embedding_table)
    position_count = _parameter_count(model.position_embedding_table)
    final_norm_count = _parameter_count(model.ln_f)
    lm_head_count = _parameter_count(model.lm_head)

    print(f"\nTrainable parameters: {parameter_count:,}")
    print(f"  token embedding:       {token_count:6,d}")
    print(f"  position embedding:    {position_count:6,d}")

    for index, block in enumerate(model.blocks):
        print(f"  block {index}:              {_parameter_count(block):6,d}")

    print(f"  final LayerNorm:        {final_norm_count:6,d}")
    print(f"  LM head:                {lm_head_count:6,d}")

    first_block = model.blocks[0]
    qkv_count = sum(
        _parameter_count(head) for head in first_block.sa.heads
    )
    projection_count = _parameter_count(first_block.sa.proj)
    ffn_count = _parameter_count(first_block.ffwd)
    layer_norm_count = (
        _parameter_count(first_block.ln1)
        + _parameter_count(first_block.ln2)
    )
    print("\nPer-block decomposition")
    print(f"  attention QKV:          {qkv_count:6,d}")
    print(f"  attention projection:   {projection_count:6,d}")
    print(f"  feed-forward network:   {ffn_count:6,d}")
    print(f"  two LayerNorms:         {layer_norm_count:6,d}")

    return parameter_count


@torch.no_grad()
def print_shape_and_norm_report(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
) -> ShapeAndNormReport:
    generator = torch.Generator().manual_seed(config.seed + 101)
    inputs, _ = data.get_batch(
        "train",
        batch_size=config.batch_size,
        block_size=config.block_size,
        device=device,
        generator=generator,
    )

    x = model._representations(inputs)
    embedding_shape = tuple(x.shape)
    embedding_norm = x.norm().item()
    block_shapes = []
    block_norms = []

    print("\nResidual-stream shape and norm checkpoint")
    print(f"  embedding:       shape={tuple(x.shape)} norm={embedding_norm:.6f}")

    for index, block in enumerate(model.blocks):
        x = block(x)
        block_shapes.append(tuple(x.shape))
        block_norms.append(x.norm().item())
        print(
            f"  after block {index}:   shape={tuple(x.shape)} "
            f"norm={block_norms[-1]:.6f}"
        )

    x = model.ln_f(x)
    final_layer_norm_shape = tuple(x.shape)
    final_layer_norm_norm = x.norm().item()
    logits = model.lm_head(x)
    expected_norm = math.sqrt(x.numel())

    print(
        f"  after final LN: shape={tuple(x.shape)} "
        f"norm={final_layer_norm_norm:.6f}"
    )
    print(f"  expected LN norm near sqrt({x.numel()}): {expected_norm:.6f}")
    print(f"  logits:          shape={tuple(logits.shape)}")

    attention_weights = model.get_attention_weights(inputs, block_index=0)
    row_sums = attention_weights.sum(dim=-1)
    future_weights = attention_weights.triu(diagonal=1)
    print("\nFirst-block attention checkpoint")
    print(f"  weights:                   {tuple(attention_weights.shape)}")
    print(
        "  attention rows sum to 1:  "
        f"{bool(torch.allclose(row_sums, torch.ones_like(row_sums)))}"
    )
    print(
        "  future weights exactly 0: "
        f"{bool(torch.all(future_weights == 0).item())}"
    )

    return ShapeAndNormReport(
        embedding_shape=embedding_shape,
        block_shapes=tuple(block_shapes),
        final_layer_norm_shape=final_layer_norm_shape,
        logits_shape=tuple(logits.shape),
        embedding_norm=embedding_norm,
        block_norms=tuple(block_norms),
        final_layer_norm_norm=final_layer_norm_norm,
    )


def print_gradient_report(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
) -> GradientReport:
    generator = torch.Generator().manual_seed(config.seed + 102)
    inputs, targets = data.get_batch(
        "train",
        batch_size=config.batch_size,
        block_size=config.block_size,
        device=device,
        generator=generator,
    )
    _, loss = model(inputs, targets)
    assert loss is not None
    model.zero_grad(set_to_none=True)
    loss.backward()

    gradient_norms = []
    all_finite = True
    all_nonzero = True

    print("\nGradient checkpoint")
    print(f"  loss: {loss.item():.6f}")
    for index, block in enumerate(model.blocks):
        gradient = block.sa.heads[0].query.weight.grad

        if gradient is None:
            norm = 0.0
            finite = False
            nonzero = False
        else:
            norm = gradient.norm().item()
            finite = bool(torch.isfinite(gradient).all().item())
            nonzero = bool(torch.any(gradient != 0).item())

        gradient_norms.append(norm)
        all_finite = all_finite and finite
        all_nonzero = all_nonzero and nonzero
        print(
            f"  block {index} head 0 query: norm={norm:.8f} "
            f"finite={finite} nonzero={nonzero}"
        )

    model.zero_grad(set_to_none=True)
    return GradientReport(
        query_gradient_norms=tuple(gradient_norms),
        all_finite=all_finite,
        all_nonzero=all_nonzero,
    )


def print_evaluation(record: EvaluationRecord) -> None:
    print(
        f"  step {record.step:4d} | "
        f"train {record.losses.train:.4f} | "
        f"val {record.losses.val:.4f}"
    )


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
) -> ExperimentReport:
    seed_everything(config.seed)
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)

    parameter_count = print_parameter_report(model)
    shapes_and_norms = print_shape_and_norm_report(
        model,
        data,
        config,
        device,
    )
    gradients = print_gradient_report(model, data, config, device)

    print("\nTraining")
    training_generator = torch.Generator().manual_seed(config.seed + 1)
    result = train_model(
        model,
        data,
        config,
        device,
        training_generator,
        on_evaluation=print_evaluation,
    )

    model.eval()
    seed_everything(config.seed + 2)
    start_id = data.vocabulary.stoi.get("\n", 0)
    context = torch.tensor([[start_id]], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=sample_length)
    sample = data.vocabulary.decode(generated[0].cpu().tolist())

    print("\nFinal checkpoint")
    print(
        f"  initial train/val: {result.initial.train:.4f} / "
        f"{result.initial.val:.4f}"
    )
    print(
        f"  final train/val:   {result.final.train:.4f} / "
        f"{result.final.val:.4f}"
    )
    print("\nGenerated text")
    print(sample)

    return ExperimentReport(
        parameter_count=parameter_count,
        shapes_and_norms=shapes_and_norms,
        gradients=gradients,
        training=result,
        sample=sample,
    )


def main() -> None:
    args = parse_args()

    if args.sample_length < 0:
        raise ValueError("sample_length must be non-negative")

    config = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)

    if args.n_head <= 0 or config.n_embd % args.n_head != 0:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({args.n_head})"
        )

    if args.n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {args.n_layer}")

    print("Stage 8: stacked Transformer blocks")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={args.n_head}, "
        f"D={config.n_embd // args.n_head}, FF={4 * config.n_embd}, "
        f"L={args.n_layer}"
    )

    run_experiment(
        data,
        config,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
    )


if __name__ == "__main__":
    main()
