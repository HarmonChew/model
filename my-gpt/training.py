from __future__ import annotations

import random
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
class TrainingResult:
    initial: LossEstimate
    final: LossEstimate
    history: tuple[EvaluationRecord, ...]


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


def train_model(
    model: EmbeddingLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    *,
    on_evaluation: Callable[[EvaluationRecord], None] | None = None,
) -> TrainingResult:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    history: list[EvaluationRecord] = []

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

    final = evaluate_on_fixed_batches()
    return TrainingResult(
        initial=history[0].losses,
        final=final,
        history=tuple(history),
    )
