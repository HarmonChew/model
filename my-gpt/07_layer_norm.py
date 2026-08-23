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
    """Run independent attention heads and concatenate their outputs."""

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
                out, head_weights = result
                outputs.append(out)
                weights.append(head_weights)

            return torch.cat(outputs, dim=-1), torch.stack(weights, dim=1)

        return torch.cat([head(x) for head in self.heads], dim=-1)


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


class FeedForwardLanguageModel(nn.Module):
    """One pre-norm attention/FFN block followed by a language-model head."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
    ) -> None:
        super().__init__()

        if vocab_size <= 0 or block_size <= 0 or n_embd <= 0 or n_head <= 0:
            raise ValueError(
                "vocab_size, block_size, n_embd, and n_head must be positive"
            )

        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
        )
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
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

        # Normalize only the sublayer input; keep the residual stream direct.
        attention_update = self.sa(self.ln1(x))
        assert isinstance(attention_update, torch.Tensor)
        x = x + attention_update

        # The FFN gets its own normalized view of the updated residual stream.
        x = x + self.ffwd(self.ln2(x))
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
    def get_attention_weights(self, idx: torch.Tensor) -> torch.Tensor:
        x = self._representations(idx)
        result = self.sa(self.ln1(x), return_weights=True)
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
class LayerNormStatistics:
    before_mean: float
    before_variance: float
    after_mean: float
    after_variance: float


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    parameter_count: int
    layer_norm: LayerNormStatistics
    training: TrainingResult
    ffn_position_differences: tuple[float, ...]
    sample: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one pre-norm attention/FFN block with GELU."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
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


@torch.no_grad()
def print_standalone_layer_norm_demo(
    n_embd: int,
    device: torch.device,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed + 100)
    x = (torch.randn(2, 4, n_embd, generator=generator) * 10 + 5).to(device)
    layer_norm = nn.LayerNorm(n_embd).to(device)
    y = layer_norm(x)

    print("\nStandalone LayerNorm experiment")
    print(f"  shape:           {x.shape}")
    print(f"  before mean:     {x[0, 0].mean().item():.6f}")
    print(
        "  before variance: "
        f"{x[0, 0].var(unbiased=False).item():.6f}"
    )
    print(f"  after mean:      {y[0, 0].mean().item():.6f}")
    print(
        "  after variance:  "
        f"{y[0, 0].var(unbiased=False).item():.6f}"
    )


def print_parameter_report(model: FeedForwardLanguageModel) -> int:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"\nTrainable parameters: {parameter_count}")

    for name, parameter in model.named_parameters():
        print(
            f"  parameter {name:34s} "
            f"{str(tuple(parameter.shape)):14s} {parameter.numel()}"
        )

    print("\nNon-trainable buffers")
    for name, buffer in model.named_buffers():
        print(f"  buffer    {name:34s} {str(tuple(buffer.shape)):14s}")

    return parameter_count


@torch.no_grad()
def print_layer_norm_parameters(model: FeedForwardLanguageModel) -> None:
    print("\nInitial LayerNorm parameters")
    for name, parameter in model.named_parameters():
        if name.startswith("ln"):
            print(f"  {name} {parameter.shape}")
            print(f"    {parameter.detach().cpu()}")


@torch.no_grad()
def print_shape_report(
    model: FeedForwardLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
) -> LayerNormStatistics:
    generator = torch.Generator().manual_seed(config.seed + 101)
    inputs, _ = data.get_batch(
        "train",
        batch_size=config.batch_size,
        block_size=config.block_size,
        device=device,
        generator=generator,
    )
    x0 = model._representations(inputs)
    norm1 = model.ln1(x0)
    attention_result = model.sa(norm1, return_weights=True)
    assert isinstance(attention_result, tuple)
    attention_update, attention_weights = attention_result
    x1 = x0 + attention_update
    norm2 = model.ln2(x1)
    hidden = model.ffwd.net[0](norm2)
    activated = model.ffwd.net[1](hidden)
    ff_update = model.ffwd.net[2](activated)
    x2 = x1 + ff_update

    row_sums = attention_weights.sum(dim=-1)
    future_weights = attention_weights.triu(diagonal=1)
    token_position = min(7, config.block_size - 1)
    before_token = x0[0, token_position]
    after_token = norm1[0, token_position]
    statistics = LayerNormStatistics(
        before_mean=before_token.mean().item(),
        before_variance=before_token.var(unbiased=False).item(),
        after_mean=after_token.mean().item(),
        after_variance=after_token.var(unbiased=False).item(),
    )

    print("\nShape checkpoint")
    print(f"  x0:                    {x0.shape}")
    print(f"  ln1(x0):               {norm1.shape}")
    print(f"  attention update:      {attention_update.shape}")
    print(f"  x1:                    {x1.shape}")
    print(f"  ln2(x1):               {norm2.shape}")
    print(f"  FF hidden:             {hidden.shape}")
    print(f"  after GELU:            {activated.shape}")
    print(f"  FFN update:            {ff_update.shape}")
    print(f"  x2:                    {x2.shape}")
    print(f"  attention weights:     {attention_weights.shape}")
    print(
        "  attention rows sum to 1:    "
        f"{bool(torch.allclose(row_sums, torch.ones_like(row_sums)))}"
    )
    print(
        "  future weights exactly 0:   "
        f"{bool(torch.all(future_weights == 0).item())}"
    )

    print(f"\nLayerNorm token diagnostic (b=0, t={token_position})")
    print("  BEFORE LN1")
    print(f"    mean:     {statistics.before_mean:.10f}")
    print(f"    variance: {statistics.before_variance:.10f}")
    print("  AFTER LN1")
    print(f"    mean:     {statistics.after_mean:.10f}")
    print(f"    variance: {statistics.after_variance:.10f}")

    print("\nResidual magnitudes")
    print(f"  embedding norm:        {x0.norm().item():.6f}")
    print(f"  ln1 output norm:       {norm1.norm().item():.6f}")
    print(f"  attention update norm: {attention_update.norm().item():.6f}")
    print(f"  after attention norm:  {x1.norm().item():.6f}")
    print(f"  ln2 output norm:       {norm2.norm().item():.6f}")
    print(f"  FFN update norm:       {ff_update.norm().item():.6f}")
    print(f"  final norm:            {x2.norm().item():.6f}")

    return statistics


@torch.no_grad()
def measure_ffn_position_independence(
    model: FeedForwardLanguageModel,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[float, ...]:
    generator = torch.Generator().manual_seed(config.seed + 202)
    first = torch.randn(
        1,
        config.block_size,
        config.n_embd,
        generator=generator,
    ).to(device)
    second = first.clone()
    second[:, 0, :] += torch.linspace(
        0,
        10,
        config.n_embd,
        device=device,
    )

    first_output = model.ffwd(model.ln2(first))
    second_output = model.ffwd(model.ln2(second))
    differences = (first_output - second_output).abs().amax(dim=(0, 2))
    result = tuple(float(value.item()) for value in differences)

    print("\nLayerNorm + FFN position-independence test")
    print("  changed only input position 0")
    for position, difference in enumerate(result):
        print(f"  position {position}: max output change {difference:.10f}")
    print(
        "  all unchanged positions exactly 0: "
        f"{all(difference == 0.0 for difference in result[1:])}"
    )

    return result


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
    device: torch.device,
    sample_length: int,
) -> ExperimentReport:
    seed_everything(config.seed)
    print_standalone_layer_norm_demo(config.n_embd, device, config.seed)
    model = FeedForwardLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
    ).to(device)

    parameter_count = print_parameter_report(model)
    print_layer_norm_parameters(model)
    layer_norm_statistics = print_shape_report(model, data, config, device)
    ffn_position_differences = measure_ffn_position_independence(
        model,
        config,
        device,
    )

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
        layer_norm=layer_norm_statistics,
        training=result,
        ffn_position_differences=ffn_position_differences,
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

    print("Stage 7: pre-norm LayerNorm and GELU")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={args.n_head}, "
        f"D={config.n_embd // args.n_head}, FF={4 * config.n_embd}"
    )

    run_experiment(
        data,
        config,
        args.n_head,
        device,
        args.sample_length,
    )


if __name__ == "__main__":
    main()
