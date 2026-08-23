from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass, field
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
    benchmark_training,
    seed_everything,
    train_model,
)


# Stage 10 keeps the Stage 9 context and architecture, then changes only the
# residual-stream width from 32 to 64 channels.
_STAGE_9_PATH = Path(__file__).with_name("09_context_length.py")
_STAGE_9_MODULE_NAME = "stage_9_context_length_for_stage_10"
_STAGE_9_SPEC = importlib.util.spec_from_file_location(
    _STAGE_9_MODULE_NAME,
    _STAGE_9_PATH,
)
assert _STAGE_9_SPEC is not None and _STAGE_9_SPEC.loader is not None
_STAGE_9 = importlib.util.module_from_spec(_STAGE_9_SPEC)
sys.modules[_STAGE_9_MODULE_NAME] = _STAGE_9
_STAGE_9_SPEC.loader.exec_module(_STAGE_9)

GPTLanguageModel = _STAGE_9.GPTLanguageModel
ShapeAndNormReport = _STAGE_9.ShapeAndNormReport
GradientReport = _STAGE_9.GradientReport
print_parameter_report = _STAGE_9.print_parameter_report
print_shape_and_norm_report = _STAGE_9.print_shape_and_norm_report
print_gradient_report = _STAGE_9.print_gradient_report

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "stage_10_best_model.pt"
)


@dataclass(slots=True)
class BestValidationCheckpoint:
    """Track and persist the model with the lowest sampled validation loss."""

    model: GPTLanguageModel
    path: Path
    best_val_loss: float = field(default=float("inf"), init=False)
    best_step: int | None = field(default=None, init=False)

    def consider(self, record: EvaluationRecord) -> bool:
        if not math.isfinite(record.losses.val):
            raise ValueError(
                "validation loss must be finite at step "
                f"{record.step}, got {record.losses.val}"
            )

        if record.losses.val >= self.best_val_loss:
            return False

        self.best_val_loss = record.losses.val
        self.best_step = record.step
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.path)
        return True


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    parameter_count: int
    shapes_and_norms: ShapeAndNormReport
    gradients: GradientReport
    training: TrainingResult
    benchmark: BenchmarkStats | None
    generalization_gap: float
    best_val_loss: float
    best_step: int
    checkpoint_path: Path
    sample: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale the Stage 9 model's width from 32 to 64 channels."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
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
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def print_evaluation(record: EvaluationRecord, *, is_best: bool) -> None:
    marker = "  * best" if is_best else ""
    print(
        f"  step {record.step:4d} | "
        f"train {record.losses.train:.4f} | "
        f"val {record.losses.val:.4f}{marker}"
    )


def print_benchmark(stats: BenchmarkStats) -> None:
    print("\nIndependent training benchmark")
    values = {
        "seconds": stats.seconds,
        "iterations_per_sec": stats.iterations_per_sec,
        "tokens_per_sec": stats.tokens_per_sec,
        "peak_allocated_mb": stats.peak_allocated_mb,
        "peak_reserved_mb": stats.peak_reserved_mb,
    }

    for key, value in values.items():
        print(f"  {key:22s}: {value:.3f}")


def clear_accelerator_cache(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def run_independent_benchmark(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    benchmark: BenchmarkConfig,
) -> BenchmarkStats:
    seed_everything(config.seed)
    bench_model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)
    bench_optimizer = torch.optim.AdamW(
        bench_model.parameters(),
        lr=config.learning_rate,
    )
    benchmark_generator = torch.Generator().manual_seed(config.seed + 1)
    stats = benchmark_training(
        bench_model,
        bench_optimizer,
        data,
        config,
        device,
        benchmark_generator,
        benchmark,
    )

    del bench_optimizer
    del bench_model
    clear_accelerator_cache(device)
    return stats


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    benchmark: BenchmarkConfig | None,
    checkpoint_path: Path,
) -> ExperimentReport:
    benchmark_stats = None
    if benchmark is not None:
        benchmark_stats = run_independent_benchmark(
            data,
            config,
            n_head,
            n_layer,
            device,
            benchmark,
        )
        print_benchmark(benchmark_stats)

    # Benchmarking consumes RNG state and optimizer updates only on the
    # disposable model above. Reseeding here makes the actual run independent.
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
    best = BestValidationCheckpoint(model=model, path=checkpoint_path)

    def record_evaluation(record: EvaluationRecord) -> None:
        print_evaluation(record, is_best=best.consider(record))

    print("\nTraining")
    training_generator = torch.Generator().manual_seed(config.seed + 1)
    result: TrainingResult = train_model(
        model,
        data,
        config,
        device,
        training_generator,
        on_evaluation=record_evaluation,
    )

    # train_model evaluates after the final update separately from its periodic
    # callback, so include that result explicitly in best-checkpoint tracking.
    if config.max_iters > 0:
        final_record = EvaluationRecord(
            step=config.max_iters,
            losses=result.final,
        )
        print_evaluation(final_record, is_best=best.consider(final_record))

    assert best.best_step is not None
    generalization_gap = result.final.val - result.final.train

    best_state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)
    del best_state
    model.eval()
    seed_everything(config.seed + 2)
    start_id = data.vocabulary.stoi.get("\n", 0)
    context = torch.tensor([[start_id]], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=sample_length)
    sample = data.vocabulary.decode(generated[0].cpu().tolist())

    print("\nStage 10 experimental summary")
    print(
        f"  B / T / C / H / D / L: "
        f"{config.batch_size} / {config.block_size} / {config.n_embd} / "
        f"{n_head} / {config.n_embd // n_head} / {n_layer}"
    )
    print(f"  parameter count:           {parameter_count:,}")
    print(f"  final train loss:          {result.final.train:.4f}")
    print(f"  final validation loss:     {result.final.val:.4f}")
    print(f"  generalization gap:        {generalization_gap:.4f}")
    print(
        f"  best validation loss:      {best.best_val_loss:.4f} "
        f"(step {best.best_step})"
    )
    if benchmark_stats is None:
        print("  benchmark:                 disabled")
    else:
        print(
            f"  iterations/sec:            "
            f"{benchmark_stats.iterations_per_sec:.3f}"
        )
        print(
            f"  tokens/sec:                "
            f"{benchmark_stats.tokens_per_sec:.3f}"
        )
        print(
            f"  peak allocated memory:     "
            f"{benchmark_stats.peak_allocated_mb:.3f} MB"
        )
        print(
            f"  peak reserved memory:      "
            f"{benchmark_stats.peak_reserved_mb:.3f} MB"
        )
    print(f"  best checkpoint:           {checkpoint_path}")
    print("\nGenerated text")
    print(sample)

    return ExperimentReport(
        parameter_count=parameter_count,
        shapes_and_norms=shapes_and_norms,
        gradients=gradients,
        training=result,
        benchmark=benchmark_stats,
        generalization_gap=generalization_gap,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        checkpoint_path=checkpoint_path,
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

    print("Stage 10: scale width 32 -> 64")
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
        args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
