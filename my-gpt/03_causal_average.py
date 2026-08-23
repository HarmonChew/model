from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch

from causal_average_model import CausalAverageLanguageModel
from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from model import EmbeddingLanguageModel
from train import resolve_device
from training import EvaluationRecord, TrainingResult, seed_everything, train_model


LanguageModel: TypeAlias = EmbeddingLanguageModel | CausalAverageLanguageModel


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    label: str
    parameter_count: int
    training: TrainingResult
    prefix_difference: float
    sample: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed causal-average character model."
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
        "--compare-embedding",
        action="store_true",
        help=(
            "Also train the previous token-local embedding model with the "
            "same initialization and sampled batches."
        ),
    )
    return parser.parse_args()


def capture_initial_parameters(
    vocab_size: int,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Create one learned initialization shared by both model variants."""
    seed_everything(config.seed)
    reference = EmbeddingLanguageModel(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
    )
    return {
        name: parameter.detach().clone()
        for name, parameter in reference.named_parameters()
    }


def apply_initial_parameters(
    model: LanguageModel,
    initial_parameters: dict[str, torch.Tensor],
) -> None:
    parameters = dict(model.named_parameters())

    if parameters.keys() != initial_parameters.keys():
        raise ValueError("Models do not expose the same learned parameters")

    with torch.no_grad():
        for name, initial_value in initial_parameters.items():
            parameters[name].copy_(initial_value)


def print_parameter_report(model: LanguageModel) -> int:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"\nTrainable parameters: {parameter_count}")

    for name, parameter in model.named_parameters():
        print(
            f"  parameter {name:34s} "
            f"{str(tuple(parameter.shape)):14s} {parameter.numel()}"
        )

    buffers = list(model.named_buffers())
    if buffers:
        print("\nNon-trainable buffers")
        for name, buffer in buffers:
            print(f"  buffer    {name:34s} {str(tuple(buffer.shape)):14s}")

    return parameter_count


def print_causal_weights(model: CausalAverageLanguageModel) -> None:
    weights = model.tril / model.tril.sum(dim=1, keepdim=True)
    print("\nFixed causal averaging weights")
    print(weights.detach().cpu())
    print("Row sums:", weights.sum(dim=1).detach().cpu())


@torch.no_grad()
def measure_prefix_change(
    model: LanguageModel,
    vocab_size: int,
    device: torch.device,
) -> float:
    """Change every prefix token while keeping the final token unchanged."""
    first = torch.arange(model.block_size, device=device) % vocab_size
    second = (first + 1) % vocab_size
    second[-1] = first[-1]
    inputs = torch.stack((first, second))
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
    model: LanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    sample_length: int,
    initial_parameters: dict[str, torch.Tensor],
    show_causal_weights: bool,
) -> ExperimentReport:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    apply_initial_parameters(model, initial_parameters)
    model = model.to(device)

    parameter_count = print_parameter_report(model)
    if show_causal_weights:
        assert isinstance(model, CausalAverageLanguageModel)
        print_causal_weights(model)

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
    final_prefix_difference = measure_prefix_change(
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
        "  max final-logit change after replacing every prefix token: "
        f"{final_prefix_difference:.10f}"
    )
    print("\nGenerated text")
    print(sample)

    return ExperimentReport(
        label=label,
        parameter_count=parameter_count,
        training=result,
        prefix_difference=final_prefix_difference,
        sample=sample,
    )


def print_comparison(reports: list[ExperimentReport]) -> None:
    print(f"\n{'=' * 92}\nContext communication comparison\n{'=' * 92}")
    print(
        f"{'model':29s} {'params':>7s} "
        f"{'initial train/val':>21s} {'final train/val':>19s} "
        f"{'prefix delta':>14s}"
    )

    for report in reports:
        initial = report.training.initial
        final = report.training.final
        print(
            f"{report.label:29s} {report.parameter_count:7d} "
            f"{initial.train:8.4f}/{initial.val:<8.4f} "
            f"{final.train:8.4f}/{final.val:<8.4f} "
            f"{report.prefix_difference:14.10f}"
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
    initial_parameters = capture_initial_parameters(
        data.vocabulary.size,
        config,
    )

    print("Stage 2: fixed causal averaging")
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

    reports = []
    if args.compare_embedding:
        reports.append(
            train_experiment(
                label="Embedding (token-local)",
                model=EmbeddingLanguageModel(
                    vocab_size=data.vocabulary.size,
                    block_size=config.block_size,
                    n_embd=config.n_embd,
                ),
                data=data,
                config=config,
                device=device,
                sample_length=args.sample_length,
                initial_parameters=initial_parameters,
                show_causal_weights=False,
            )
        )

    reports.append(
        train_experiment(
            label="Fixed causal average",
            model=CausalAverageLanguageModel(
                vocab_size=data.vocabulary.size,
                block_size=config.block_size,
                n_embd=config.n_embd,
            ),
            data=data,
            config=config,
            device=device,
            sample_length=args.sample_length,
            initial_parameters=initial_parameters,
            show_causal_weights=True,
        )
    )

    if len(reports) > 1:
        print_comparison(reports)


if __name__ == "__main__":
    main()
