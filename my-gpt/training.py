from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from config import TrainingConfig
from data_utils import CharacterData
from model import EmbeddingLanguageModel


@dataclass(frozen=True, slots=True)
class LossEstimate:
    train: float
    val: float


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    step: int
    losses: LossEstimate


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configure warmup and timed optimizer steps for a benchmark."""

    num_warmup: int = 20
    num_steps: int = 100

    def __post_init__(self) -> None:
        if self.num_warmup < 0:
            raise ValueError(
                f"num_warmup must be non-negative, got {self.num_warmup}"
            )

        if self.num_steps <= 0:
            raise ValueError(
                f"num_steps must be positive, got {self.num_steps}"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    """Wall-clock throughput and peak allocator usage for training steps."""

    seconds: float
    iterations_per_sec: float
    tokens_per_sec: float
    peak_allocated_mb: float
    peak_reserved_mb: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    initial: LossEstimate
    final: LossEstimate
    history: tuple[EvaluationRecord, ...]
    benchmark: BenchmarkStats | None = None


def sync_device(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before reading the clock."""

    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def reset_peak_memory_stats(device: torch.device) -> None:
    """Reset allocator peaks on accelerators that expose these counters."""

    if device.type == "xpu":
        torch.xpu.memory.reset_peak_memory_stats(device)
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_stats(device: torch.device) -> tuple[int, int]:
    """Return peak allocated and reserved bytes, or zeros if unsupported."""

    if device.type == "xpu":
        return (
            torch.xpu.memory.max_memory_allocated(device),
            torch.xpu.memory.max_memory_reserved(device),
        )

    if device.type == "cuda":
        return (
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        )

    return 0, 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(
    model: EmbeddingLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    generator: torch.Generator,
) -> LossEstimate:
    was_training = model.training
    model.eval()
    estimates: dict[str, float] = {}

    for split in ("train", "val"):
        losses = []

        for _ in range(config.eval_iters):
            inputs, targets = data.get_batch(
                split,
                batch_size=config.batch_size,
                block_size=config.block_size,
                device=device,
                generator=generator,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            losses.append(loss.item())

        estimates[split] = sum(losses) / len(losses)

    model.train(was_training)
    return LossEstimate(train=estimates["train"], val=estimates["val"])


def benchmark_training(
    model: EmbeddingLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    benchmark: BenchmarkConfig,
) -> BenchmarkStats:
    """Time optimizer steps on a caller-owned, disposable model.

    Keeping the model and optimizer outside this helper makes it explicit that
    benchmark updates do not have to touch the model used by an experiment.
    """

    model.train()
    benchmark_started_at: float | None = None
    total_steps = benchmark.num_warmup + benchmark.num_steps

    for step in range(total_steps):
        if step == benchmark.num_warmup:
            sync_device(device)
            reset_peak_memory_stats(device)
            benchmark_started_at = time.perf_counter()

        inputs, targets = data.get_batch(
            "train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=training_generator,
        )
        _, loss = model(inputs, targets)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    sync_device(device)
    assert benchmark_started_at is not None
    elapsed = time.perf_counter() - benchmark_started_at
    iterations_per_sec = benchmark.num_steps / elapsed
    tokens_per_iteration = config.batch_size * config.block_size

    peak_allocated, peak_reserved = get_peak_memory_stats(device)

    return BenchmarkStats(
        seconds=elapsed,
        iterations_per_sec=iterations_per_sec,
        tokens_per_sec=iterations_per_sec * tokens_per_iteration,
        peak_allocated_mb=peak_allocated / 1024**2,
        peak_reserved_mb=peak_reserved / 1024**2,
    )


def train_model(
    model: EmbeddingLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    *,
    on_evaluation: Callable[[EvaluationRecord], None] | None = None,
    benchmark: BenchmarkConfig | None = None,
) -> TrainingResult:
    if benchmark is not None:
        required_steps = benchmark.num_warmup + benchmark.num_steps
        if config.max_iters < required_steps:
            raise ValueError(
                "benchmark requires at least "
                f"{required_steps} training steps, but max_iters is "
                f"{config.max_iters}"
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    history: list[EvaluationRecord] = []
    benchmark_started_at: float | None = None
    benchmark_stats: BenchmarkStats | None = None

    def evaluate_on_fixed_batches() -> LossEstimate:
        # Recreate this generator for every evaluation so that initial,
        # intermediate, and final losses use exactly the same sampled blocks.
        evaluation_generator = torch.Generator().manual_seed(config.seed + 2)
        return estimate_loss(
            model,
            data,
            config,
            device,
            evaluation_generator,
        )

    if config.max_iters == 0:
        initial = evaluate_on_fixed_batches()
        record = EvaluationRecord(step=0, losses=initial)
        history.append(record)

        if on_evaluation is not None:
            on_evaluation(record)

        return TrainingResult(
            initial=initial,
            final=initial,
            history=tuple(history),
        )

    for step in range(config.max_iters):
        if step % config.eval_interval == 0:
            losses = evaluate_on_fixed_batches()
            record = EvaluationRecord(step=step, losses=losses)
            history.append(record)

            if on_evaluation is not None:
                on_evaluation(record)

        if benchmark is not None and step == benchmark.num_warmup:
            sync_device(device)
            reset_peak_memory_stats(device)
            benchmark_started_at = time.perf_counter()

        inputs, targets = data.get_batch(
            "train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=training_generator,
        )
        _, loss = model(inputs, targets)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (
            benchmark is not None
            and step + 1 == benchmark.num_warmup + benchmark.num_steps
        ):
            sync_device(device)
            assert benchmark_started_at is not None
            elapsed = time.perf_counter() - benchmark_started_at
            iterations_per_sec = benchmark.num_steps / elapsed
            tokens_per_iteration = config.batch_size * config.block_size

            peak_allocated, peak_reserved = get_peak_memory_stats(device)

            benchmark_stats = BenchmarkStats(
                seconds=elapsed,
                iterations_per_sec=iterations_per_sec,
                tokens_per_sec=(
                    iterations_per_sec * tokens_per_iteration
                ),
                peak_allocated_mb=peak_allocated / 1024**2,
                peak_reserved_mb=peak_reserved / 1024**2,
            )

    final = evaluate_on_fixed_batches()
    return TrainingResult(
        initial=history[0].losses,
        final=final,
        history=tuple(history),
        benchmark=benchmark_stats,
    )
