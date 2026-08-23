from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import (
    BenchmarkConfig,
    BenchmarkStats,
    EvaluationRecord,
    TrainingResult,
    seed_everything,
    train_model,
)


# Stage 9 deliberately reuses the exact Stage 8 architecture. Loading the
# numbered checkpoint this way keeps the historical lesson runnable while
# making block_size the only model-axis change in this experiment.
_STAGE_8_PATH = Path(__file__).with_name("08_transformer_block.py")
_STAGE_8_MODULE_NAME = "stage_8_transformer_block_for_stage_9"
_STAGE_8_SPEC = importlib.util.spec_from_file_location(
    _STAGE_8_MODULE_NAME,
    _STAGE_8_PATH,
)
assert _STAGE_8_SPEC is not None and _STAGE_8_SPEC.loader is not None
_STAGE_8 = importlib.util.module_from_spec(_STAGE_8_SPEC)
sys.modules[_STAGE_8_MODULE_NAME] = _STAGE_8
_STAGE_8_SPEC.loader.exec_module(_STAGE_8)

GPTLanguageModel = _STAGE_8.GPTLanguageModel
ShapeAndNormReport = _STAGE_8.ShapeAndNormReport
GradientReport = _STAGE_8.GradientReport
ExperimentReport = _STAGE_8.ExperimentReport
print_parameter_report = _STAGE_8.print_parameter_report
print_shape_and_norm_report = _STAGE_8.print_shape_and_norm_report
print_gradient_report = _STAGE_8.print_gradient_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale the Stage 8 model's context from 8 to 64 tokens."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def print_evaluation(record: EvaluationRecord) -> None:
    print(
        f"  step {record.step:4d} | "
        f"train {record.losses.train:.4f} | "
        f"val {record.losses.val:.4f}"
    )


def print_benchmark(stats: BenchmarkStats) -> None:
    print("\nTraining benchmark")
    values = {
        "seconds": stats.seconds,
        "iterations_per_sec": stats.iterations_per_sec,
        "tokens_per_sec": stats.tokens_per_sec,
        "peak_allocated_mb": stats.peak_allocated_mb,
        "peak_reserved_mb": stats.peak_reserved_mb,
    }

    for key, value in values.items():
        print(f"  {key:22s}: {value:.3f}")


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    benchmark: BenchmarkConfig | None,
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
    result: TrainingResult = train_model(
        model,
        data,
        config,
        device,
        training_generator,
        on_evaluation=print_evaluation,
        benchmark=benchmark,
    )

    if result.benchmark is not None:
        print_benchmark(result.benchmark)

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

    if args.benchmark_warmup < 0:
        raise ValueError("benchmark-warmup must be non-negative")

    if args.benchmark_steps < 0:
        raise ValueError("benchmark-steps must be non-negative")

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

    benchmark = None
    if args.benchmark_steps > 0:
        benchmark = BenchmarkConfig(
            num_warmup=args.benchmark_warmup,
            num_steps=args.benchmark_steps,
        )

    print("Stage 9: context length 8 -> 64")
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
        benchmark,
    )


if __name__ == "__main__":
    main()
