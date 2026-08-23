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

        # The mask follows the module between devices and is saved in the
        # state dict, but it is fixed and is not an optimized parameter.
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

        # Each projection maps (B, T, C) -> (B, T, D).
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # (B, T, D) @ (B, D, T) -> (B, T, T). These scores are
        # dynamically recreated for every input; they are not parameters.
        weights = q @ k.transpose(-2, -1)
        weights = weights * (self.head_size**-0.5)

        # A forbidden score must be -inf, not zero: softmax(-inf) is exactly
        # zero, so no information or gradient can travel back from the future.
        weights = weights.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float("-inf"),
        )
        weights = F.softmax(weights, dim=-1)

        # (B, T, T) @ (B, T, D) -> (B, T, D).
        out = weights @ v

        if return_weights:
            return out, weights

        return out


class SingleHeadAttentionLanguageModel(nn.Module):
    """A character language model with one causal self-attention head."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
    ) -> None:
        super().__init__()

        if vocab_size <= 0 or block_size <= 0 or n_embd <= 0:
            raise ValueError(
                "vocab_size, block_size, and n_embd must all be positive"
            )

        self.block_size = block_size
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(
            n_embd=n_embd,
            head_size=n_embd,
            block_size=block_size,
        )
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

        # (B, T) -> (B, T, C).
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

        # (B, T, C) -> (B, T, C), because this first head uses D = C.
        x = self.sa_head(x)
        assert isinstance(x, torch.Tensor)

        # (B, T, C) -> (B, T, V).
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
        result = self.sa_head(x, return_weights=True)
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
class ExperimentReport:
    parameter_count: int
    training: TrainingResult
    initial_prefix_difference: float
    final_prefix_difference: float
    sample: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one causal self-attention head from scratch."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--attention-text", default="To be or")
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def print_parameter_report(model: SingleHeadAttentionLanguageModel) -> int:
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
def print_shape_report(
    model: SingleHeadAttentionLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    generator = torch.Generator().manual_seed(config.seed + 101)
    inputs, _ = data.get_batch(
        "train",
        batch_size=config.batch_size,
        block_size=config.block_size,
        device=device,
        generator=generator,
    )
    _, sequence_length = inputs.shape
    token_embeddings = model.token_embedding_table(inputs)
    positions = torch.arange(sequence_length, device=device)
    position_embeddings = model.position_embedding_table(positions)
    x = token_embeddings + position_embeddings
    q = model.sa_head.query(x)
    k = model.sa_head.key(x)
    v = model.sa_head.value(x)
    scores = q @ k.transpose(-2, -1)
    output = model.sa_head(x)
    assert isinstance(output, torch.Tensor)

    print("\nShape checkpoint")
    print(f"  idx:                  {inputs.shape}")
    print(f"  token embeddings:     {token_embeddings.shape}")
    print(f"  position embeddings:  {position_embeddings.shape}")
    print(f"  x:                    {x.shape}")
    print(f"  q:                    {q.shape}")
    print(f"  k:                    {k.shape}")
    print(f"  v:                    {v.shape}")
    print(f"  q @ k^T:              {scores.shape}")
    print(f"  attention output:     {output.shape}")


@torch.no_grad()
def measure_prefix_change(
    model: SingleHeadAttentionLanguageModel,
    vocab_size: int,
    device: torch.device,
) -> float:
    first = torch.arange(model.block_size, device=device) % vocab_size
    second = (first + 1) % vocab_size
    second[-1] = first[-1]
    logits, _ = model(torch.stack((first, second)))
    difference = (logits[0, -1] - logits[1, -1]).abs().max()
    return float(difference.item())


@torch.no_grad()
def inspect_attention(
    model: SingleHeadAttentionLanguageModel,
    data: CharacterData,
    text: str,
    device: torch.device,
    *,
    label: str,
    show_readable_rows: bool,
) -> torch.Tensor:
    if not text:
        raise ValueError("attention_text must contain at least one character")

    if len(text) > model.block_size:
        raise ValueError(
            f"attention_text has {len(text)} characters, but block_size is "
            f"{model.block_size}"
        )

    try:
        token_ids = data.vocabulary.encode(text)
    except KeyError as error:
        raise ValueError(
            f"attention_text contains out-of-vocabulary character {error.args[0]!r}"
        ) from error

    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    weights = model.get_attention_weights(ids)[0].cpu()
    row_sums = weights.sum(dim=-1)
    future_weights = weights.triu(diagonal=1)
    future_is_zero = bool(torch.equal(future_weights, torch.zeros_like(weights)))

    print(f"\n{label}: {text!r}")
    print(f"  weights shape:             {tuple(weights.shape)}")
    print(f"  row sums:                  {row_sums}")
    print(f"  rows sum to 1:             {bool(torch.allclose(row_sums, torch.ones_like(row_sums)))}")
    print(f"  future weights exactly 0:  {future_is_zero}")
    print("  matrix:")
    print(weights)

    if show_readable_rows:
        for target, query_token in enumerate(text):
            print(f"\n  Query position {target}: {query_token!r}")
            for source in range(target + 1):
                print(
                    f"    attends to {text[source]!r:6s} at {source}: "
                    f"{weights[target, source]:.3f}"
                )

    return weights


def print_evaluation(record: EvaluationRecord) -> None:
    print(
        f"  step {record.step:4d} | "
        f"train {record.losses.train:.4f} | "
        f"val {record.losses.val:.4f}"
    )


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    sample_length: int,
    attention_text: str,
) -> ExperimentReport:
    seed_everything(config.seed)
    model = SingleHeadAttentionLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
    ).to(device)

    parameter_count = print_parameter_report(model)
    print_shape_report(model, data, config, device)
    initial_prefix_difference = measure_prefix_change(
        model,
        data.vocabulary.size,
        device,
    )
    print(
        "\nPrefix test before training\n"
        "  max final-logit change after replacing every prefix token: "
        f"{initial_prefix_difference:.10f}"
    )
    inspect_attention(
        model,
        data,
        attention_text,
        device,
        label="Attention before training",
        show_readable_rows=False,
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
    final_prefix_difference = measure_prefix_change(
        model,
        data.vocabulary.size,
        device,
    )
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
    print(
        "  initial/final prefix-change delta: "
        f"{initial_prefix_difference:.10f} / {final_prefix_difference:.10f}"
    )
    print("\nGenerated text")
    print(sample)

    inspect_attention(
        model,
        data,
        attention_text,
        device,
        label="Learned attention",
        show_readable_rows=True,
    )

    return ExperimentReport(
        parameter_count=parameter_count,
        training=result,
        initial_prefix_difference=initial_prefix_difference,
        final_prefix_difference=final_prefix_difference,
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

    print("Stage 3: one causal self-attention head")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, D={config.n_embd}"
    )

    run_experiment(
        data,
        config,
        device,
        args.sample_length,
        args.attention_text,
    )


if __name__ == "__main__":
    main()
