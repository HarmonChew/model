from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from model import EmbeddingLanguageModel
from training import EvaluationRecord, TrainingResult, seed_everything, train_model


SHARED_PARAMETER_NAMES = (
    "token_embedding_table.weight",
    "lm_head.weight",
    "lm_head.bias",
)


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    label: str
    parameter_count: int
    training: TrainingResult
    embedding_change: float
    independence_difference: float
    sample: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Stage 1 character embedding language model."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-iters", type=int, default=3_000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--compare-positions",
        action="store_true",
        help="Train matched models without and with position embeddings.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "xpu" and not (
        hasattr(torch, "xpu") and torch.xpu.is_available()
    ):
        raise RuntimeError("XPU was requested but is not available")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if requested == "mps" and not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")

    return torch.device(requested)


def capture_shared_initialization(
    vocab_size: int,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Create common token/head weights for a fair A/B comparison."""
    seed_everything(config.seed)
    reference = EmbeddingLanguageModel(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        use_position_embeddings=False,
    )
    state = reference.state_dict()
    return {
        name: state[name].detach().clone()
        for name in SHARED_PARAMETER_NAMES
    }


def apply_shared_initialization(
    model: EmbeddingLanguageModel,
    shared_state: dict[str, torch.Tensor],
) -> None:
    state = model.state_dict()

    with torch.no_grad():
        for name, initial_value in shared_state.items():
            state[name].copy_(initial_value)


def print_shape_report(
    model: EmbeddingLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    if model.position_embedding_table is None:
        raise ValueError("The shape report requires position embeddings")

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
    logits = model.lm_head(x)

    print("\nShape checkpoint")
    print(f"  input IDs:             {inputs.shape}")
    print(f"  token embeddings:      {token_embeddings.shape}")
    print(f"  positions:             {positions.shape}")
    print(f"  position embeddings:   {position_embeddings.shape}")
    print(f"  combined x:            {x.shape}")
    print(f"  logits:                {logits.shape}")


def print_parameter_report(model: EmbeddingLanguageModel) -> int:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"\nParameters: {parameter_count}")

    for name, parameter in model.named_parameters():
        print(
            f"  {name:36s} "
            f"{str(tuple(parameter.shape)):14s} "
            f"{parameter.numel()}"
        )

    return parameter_count


def print_gradient_report(
    model: EmbeddingLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
) -> None:
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

    names = [
        "token_embedding_table.weight",
        "position_embedding_table.weight",
        "lm_head.weight",
    ]
    parameters = dict(model.named_parameters())

    print("\nGradient checkpoint")
    for name in names:
        parameter = parameters.get(name)
        gradient = None if parameter is None else parameter.grad
        if gradient is None:
            print(f"  {name:36s} missing")
        else:
            print(
                f"  {name:36s} {str(tuple(gradient.shape)):14s} "
                f"norm={torch.linalg.vector_norm(gradient).item():.6f} "
                f"finite={bool(torch.isfinite(gradient).all().item())}"
            )

    used_token_id = int(inputs[0, 0].item())
    used_character = data.vocabulary.itos[used_token_id]
    token_gradient = parameters["token_embedding_table.weight"].grad
    assert token_gradient is not None
    used_token_norm = torch.linalg.vector_norm(
        token_gradient[used_token_id]
    ).item()
    print(
        f"  used token row {used_character!r:20s} "
        f"{str((config.n_embd,)):14s} norm={used_token_norm:.6f}"
    )

    model.zero_grad(set_to_none=True)


@torch.no_grad()
def measure_cross_position_independence(
    model: EmbeddingLanguageModel,
    vocab_size: int,
    device: torch.device,
) -> float:
    sequence_a = torch.arange(model.block_size) % vocab_size
    sequence_b = (sequence_a + 1) % vocab_size
    sequence_b[-1] = sequence_a[-1]
    inputs = torch.stack((sequence_a, sequence_b)).to(device)
    logits, _ = model(inputs)
    difference = (logits[0, -1] - logits[1, -1]).abs().max()
    return float(difference.item())


def print_evaluation(record: EvaluationRecord) -> None:
    print(
        f"  step {record.step:4d} | "
        f"train {record.losses.train:.4f} | "
        f"val {record.losses.val:.4f}"
    )


def train_experiment(
    *,
    label: str,
    use_position_embeddings: bool,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    sample_length: int,
    shared_state: dict[str, torch.Tensor],
    detailed_diagnostics: bool,
) -> ExperimentReport:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    seed_everything(config.seed)
    model = EmbeddingLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        use_position_embeddings=use_position_embeddings,
    )
    apply_shared_initialization(model, shared_state)
    model = model.to(device)

    if detailed_diagnostics:
        print_shape_report(model, data, config, device)

    parameter_count = print_parameter_report(model)

    if detailed_diagnostics:
        print_gradient_report(model, data, config, device)

    tracked_character = (
        "q" if "q" in data.vocabulary.stoi else data.vocabulary.chars[0]
    )
    tracked_token = data.vocabulary.stoi[tracked_character]
    embedding_before = (
        model.token_embedding_table.weight[tracked_token].detach().clone()
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
    embedding_after = model.token_embedding_table.weight[tracked_token].detach()
    embedding_change = float(
        torch.linalg.vector_norm(embedding_after - embedding_before).item()
    )
    independence_difference = measure_cross_position_independence(
        model,
        data.vocabulary.size,
        device,
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
    print(
        f"  {tracked_character!r} embedding L2 change: "
        f"{embedding_change:.6f}"
    )
    print(
        "  max final-logit change after replacing every prefix token: "
        f"{independence_difference:.10f}"
    )
    print("\nGenerated text")
    print(sample)

    return ExperimentReport(
        label=label,
        parameter_count=parameter_count,
        training=result,
        embedding_change=embedding_change,
        independence_difference=independence_difference,
        sample=sample,
    )


def print_comparison(reports: list[ExperimentReport]) -> None:
    print(f"\n{'=' * 72}\nPosition embedding comparison\n{'=' * 72}")
    print(
        f"{'model':30s} {'params':>7s} "
        f"{'initial train/val':>21s} {'final train/val':>19s}"
    )

    for report in reports:
        initial = report.training.initial
        final = report.training.final
        print(
            f"{report.label:30s} {report.parameter_count:7d} "
            f"{initial.train:8.4f}/{initial.val:<8.4f} "
            f"{final.train:8.4f}/{final.val:<8.4f}"
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

    print("Stage 1: embeddings")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}"
    )

    shared_state = capture_shared_initialization(
        data.vocabulary.size,
        config,
    )
    reports = []

    if args.compare_positions:
        reports.append(
            train_experiment(
                label="A: token embeddings only",
                use_position_embeddings=False,
                data=data,
                config=config,
                device=device,
                sample_length=args.sample_length,
                shared_state=shared_state,
                detailed_diagnostics=False,
            )
        )

    reports.append(
        train_experiment(
            label="B: token + position embeddings",
            use_position_embeddings=True,
            data=data,
            config=config,
            device=device,
            sample_length=args.sample_length,
            shared_state=shared_state,
            detailed_diagnostics=True,
        )
    )

    if len(reports) > 1:
        print_comparison(reports)


if __name__ == "__main__":
    main()
