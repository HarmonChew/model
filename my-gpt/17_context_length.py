from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import (
    BenchmarkConfig,
    BenchmarkStats,
    get_peak_memory_stats,
    reset_peak_memory_stats,
    seed_everything,
    sync_device,
)


# Stage 17 keeps the Stage 15/16 p=0 architecture and proven three-phase
# schedule. Its treatment is context length, and its batching protocol is new.
_STAGE_15_PATH = Path(__file__).with_name("15_dropout_from_initialization.py")
_STAGE_15_MODULE_NAME = "stage_15_dropout_from_initialization_for_stage_17"
_STAGE_15_SPEC = importlib.util.spec_from_file_location(
    _STAGE_15_MODULE_NAME,
    _STAGE_15_PATH,
)
assert _STAGE_15_SPEC is not None and _STAGE_15_SPEC.loader is not None
_STAGE_15 = importlib.util.module_from_spec(_STAGE_15_SPEC)
sys.modules[_STAGE_15_MODULE_NAME] = _STAGE_15
_STAGE_15_SPEC.loader.exec_module(_STAGE_15)

GPTLanguageModel = _STAGE_15.GPTLanguageModel
LearningRateSchedule = _STAGE_15.LearningRateSchedule
BestValidationCheckpoint = _STAGE_15.BestValidationCheckpoint

checkpoint_sha256 = _STAGE_15.checkpoint_sha256
clear_accelerator_cache = _STAGE_15.clear_accelerator_cache
fingerprint_data = _STAGE_15.fingerprint_data
generate_from_final_model = _STAGE_15.generate_from_final_model
set_optimizer_learning_rate = _STAGE_15.set_optimizer_learning_rate
tensor_mapping_sha256 = _STAGE_15.tensor_mapping_sha256
tensor_sha256 = _STAGE_15.tensor_sha256

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_17_control_best_checkpoint.pt"
)
DEFAULT_TREATMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_17_context_128_best_checkpoint.pt"
)

DEFAULT_CONTROL_BATCH_SIZE = 32
DEFAULT_CONTROL_BLOCK_SIZE = 64
DEFAULT_TREATMENT_BATCH_SIZE = 16
DEFAULT_TREATMENT_BLOCK_SIZE = 128
DEFAULT_MAX_ITERS = _STAGE_15.DEFAULT_MAX_ITERS
DEFAULT_INITIAL_LEARNING_RATE = _STAGE_15.DEFAULT_INITIAL_LEARNING_RATE
DEFAULT_FIRST_DECAY_STEP = _STAGE_15.DEFAULT_FIRST_DECAY_STEP
DEFAULT_MIDDLE_LEARNING_RATE = _STAGE_15.DEFAULT_MIDDLE_LEARNING_RATE
DEFAULT_SECOND_DECAY_STEP = _STAGE_15.DEFAULT_SECOND_DECAY_STEP
DEFAULT_FINAL_LEARNING_RATE = _STAGE_15.DEFAULT_FINAL_LEARNING_RATE
DEFAULT_PRECISE_EVAL_ITERS = 500
DEFAULT_DROPOUT = 0.0
DROPOUT_PLACEMENT = _STAGE_15.DROPOUT_PLACEMENT


@dataclass(frozen=True, slots=True)
class ContextSpec:
    name: str
    batch_size: int
    block_size: int
    checkpoint_path: Path

    @property
    def tokens_per_update(self) -> int:
        return self.batch_size * self.block_size


@dataclass(frozen=True, slots=True)
class PairedContextBatch:
    """One sampled long segment panel represented at both context lengths."""

    starts: torch.Tensor
    control_inputs: torch.Tensor
    control_targets: torch.Tensor
    treatment_inputs: torch.Tensor
    treatment_targets: torch.Tensor

    def for_spec(
        self,
        spec: ContextSpec,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            spec.batch_size == self.control_inputs.shape[0]
            and spec.block_size == self.control_inputs.shape[1]
        ):
            inputs, targets = self.control_inputs, self.control_targets
        elif (
            spec.batch_size == self.treatment_inputs.shape[0]
            and spec.block_size == self.treatment_inputs.shape[1]
        ):
            inputs, targets = self.treatment_inputs, self.treatment_targets
        else:
            raise ValueError(
                f"spec {spec.name!r} with B={spec.batch_size}, "
                f"T={spec.block_size} does not match this paired batch"
            )
        if device is None:
            return inputs, targets
        return inputs.to(device), targets.to(device)


@dataclass(frozen=True, slots=True)
class SplitLosses:
    overall: float
    first_half: float
    second_half: float


@dataclass(frozen=True, slots=True)
class ContextLossEstimate:
    train: SplitLosses
    val: SplitLosses


@dataclass(frozen=True, slots=True)
class ContextEvaluationRecord:
    step: int
    losses: ContextLossEstimate


@dataclass(frozen=True, slots=True)
class ContextTrainingResult:
    initial: ContextLossEstimate
    final: ContextLossEstimate
    history: tuple[ContextEvaluationRecord, ...]
    benchmark: BenchmarkStats | None


@dataclass(frozen=True, slots=True)
class PairedInitialStates:
    control: dict[str, torch.Tensor]
    treatment: dict[str, torch.Tensor]
    common_state_sha256: str
    control_state_sha256: str
    treatment_state_sha256: str
    extra_position_rows_sha256: str
    control_parameter_count: int
    treatment_parameter_count: int


@dataclass(frozen=True, slots=True)
class BranchReport:
    spec: ContextSpec
    parameter_count: int
    initial_state_sha256: str
    training: ContextTrainingResult
    best_val_loss: float
    best_step: int
    best_validation_losses: SplitLosses
    final_batch_generator_sha256: str
    attention_shape: tuple[int, ...]
    sample: str

    @property
    def generalization_gap(self) -> float:
        return self.training.final.val.overall - self.training.final.train.overall


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    initialization: PairedInitialStates
    initialization_seed: int
    training_batch_seed: int
    training_rng_seed: int
    schedule: LearningRateSchedule
    targets_per_update: int
    control: BranchReport
    treatment: BranchReport
    final_val_delta: float
    best_val_delta: float

    @property
    def final_winner(self) -> str:
        if self.final_val_delta < 0:
            return self.treatment.spec.name
        if self.final_val_delta > 0:
            return self.control.spec.name
        return "tie"

    @property
    def best_winner(self) -> str:
        if self.best_val_delta < 0:
            return self.treatment.spec.name
        if self.best_val_delta > 0:
            return self.control.spec.name
        return "tie"


@dataclass(frozen=True, slots=True)
class SplitStatistics:
    mean: SplitLosses
    standard_error: SplitLosses


@dataclass(frozen=True, slots=True)
class PreciseBranchResult:
    name: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_step: int
    fixed_panel_loss: float
    statistics: SplitStatistics
    batch_losses: tuple[SplitLosses, ...]


@dataclass(frozen=True, slots=True)
class PreciseDelta:
    candidate: str
    baseline: str
    mean_delta: SplitLosses
    standard_error: SplitLosses
    confidence_low: SplitLosses
    confidence_high: SplitLosses


@dataclass(frozen=True, slots=True)
class PreciseValidationReport:
    eval_iters: int
    seed: int
    results: tuple[PreciseBranchResult, ...]
    delta: PreciseDelta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare T=64 with T=128 while pairing every target character "
            "and holding B*T constant at each optimizer update."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--control-batch-size", type=int, default=DEFAULT_CONTROL_BATCH_SIZE
    )
    parser.add_argument(
        "--control-block-size", type=int, default=DEFAULT_CONTROL_BLOCK_SIZE
    )
    parser.add_argument(
        "--treatment-batch-size", type=int, default=DEFAULT_TREATMENT_BATCH_SIZE
    )
    parser.add_argument(
        "--treatment-block-size", type=int, default=DEFAULT_TREATMENT_BLOCK_SIZE
    )
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        default=DEFAULT_INITIAL_LEARNING_RATE,
    )
    parser.add_argument(
        "--first-decay-step", type=int, default=DEFAULT_FIRST_DECAY_STEP
    )
    parser.add_argument(
        "--middle-learning-rate",
        type=float,
        default=DEFAULT_MIDDLE_LEARNING_RATE,
    )
    parser.add_argument(
        "--second-decay-step", type=int, default=DEFAULT_SECOND_DECAY_STEP
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=DEFAULT_FINAL_LEARNING_RATE,
    )
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument(
        "--training-batch-seed",
        type=int,
        default=None,
        help="Defaults to seed + 1.",
    )
    parser.add_argument(
        "--training-rng-seed",
        type=int,
        default=None,
        help="Defaults to seed + 3 and is reset before each branch.",
    )
    parser.add_argument(
        "--precise-eval-iters",
        type=int,
        default=DEFAULT_PRECISE_EVAL_ITERS,
        help="Fresh paired validation segments after training; 0 skips.",
    )
    parser.add_argument(
        "--precise-eval-seed",
        type=int,
        default=None,
        help="Defaults to seed + 7.",
    )
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument(
        "--control-checkpoint-path",
        type=Path,
        default=DEFAULT_CONTROL_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--treatment-checkpoint-path",
        type=Path,
        default=DEFAULT_TREATMENT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _validate_non_negative_integer(value: int, *, name: str) -> int:
    return _STAGE_15._validate_non_negative_integer(value, name=name)


def _validate_context_geometry(
    control: ContextSpec,
    treatment: ContextSpec,
) -> int:
    for spec in (control, treatment):
        if spec.batch_size <= 0:
            raise ValueError(
                f"{spec.name} batch size must be positive, got {spec.batch_size}"
            )
        if spec.block_size <= 0:
            raise ValueError(
                f"{spec.name} block size must be positive, got {spec.block_size}"
            )

    if treatment.block_size != 2 * control.block_size:
        raise ValueError(
            "Stage 17 requires treatment block size to be exactly twice "
            "the control block size"
        )
    if control.batch_size != 2 * treatment.batch_size:
        raise ValueError(
            "Stage 17 requires control batch size to be exactly twice "
            "the treatment batch size"
        )
    if control.tokens_per_update != treatment.tokens_per_update:
        raise ValueError("control and treatment must have equal B*T")
    if control.name == treatment.name:
        raise ValueError("control and treatment names must differ")
    return control.tokens_per_update


def validate_context_protocol(
    control: ContextSpec,
    treatment: ContextSpec,
) -> int:
    tokens_per_update = _validate_context_geometry(control, treatment)
    paths = {
        "control checkpoint": control.checkpoint_path.resolve(),
        "control temporary checkpoint": control.checkpoint_path.with_name(
            f".{control.checkpoint_path.name}.tmp"
        ).resolve(),
        "treatment checkpoint": treatment.checkpoint_path.resolve(),
        "treatment temporary checkpoint": treatment.checkpoint_path.with_name(
            f".{treatment.checkpoint_path.name}.tmp"
        ).resolve(),
    }
    labels = tuple(paths)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            first_path = paths[first_label]
            second_path = paths[second_label]
            if first_path == second_path:
                raise ValueError(
                    f"{first_label} and {second_label} must use distinct paths"
                )
            if (
                first_path.exists()
                and second_path.exists()
                and first_path.samefile(second_path)
            ):
                raise ValueError(
                    f"{first_label} and {second_label} must not be the same file"
                )
    return tokens_per_update


def _split_source(data: CharacterData, split: str) -> torch.Tensor:
    if split == "train":
        return data.train_data
    if split == "val":
        return data.val_data
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")


def build_paired_context_batch(
    source: torch.Tensor,
    starts: torch.Tensor,
    *,
    control_block_size: int,
    treatment_block_size: int,
) -> PairedContextBatch:
    """Build short windows and long windows with identical flattened targets.

    Short windows are interleaved by source segment. Consequently,
    control_targets.reshape(-1) is exactly treatment_targets.reshape(-1),
    not merely an equal multiset.
    """

    if source.ndim != 1:
        raise ValueError(f"source must be one-dimensional, got {source.shape}")
    if control_block_size <= 0:
        raise ValueError("control block size must be positive")
    if treatment_block_size <= 0:
        raise ValueError("treatment block size must be positive")
    if treatment_block_size != 2 * control_block_size:
        raise ValueError("treatment block size must equal 2 * control block size")
    if len(source) <= treatment_block_size:
        raise ValueError(
            f"source has {len(source)} tokens, but treatment block size is "
            f"{treatment_block_size}"
        )

    starts_cpu = torch.as_tensor(starts, dtype=torch.long, device="cpu")
    if starts_cpu.ndim != 1 or starts_cpu.numel() == 0:
        raise ValueError("starts must be a non-empty one-dimensional tensor")
    maximum_start = len(source) - treatment_block_size - 1
    if int(starts_cpu.min()) < 0 or int(starts_cpu.max()) > maximum_start:
        raise ValueError(
            f"starts must be in [0, {maximum_start}], got "
            f"[{int(starts_cpu.min())}, {int(starts_cpu.max())}]"
        )

    offsets = torch.arange(treatment_block_size)
    indices = starts_cpu[:, None] + offsets[None, :]
    treatment_inputs_cpu = source[indices]
    treatment_targets_cpu = source[indices + 1]

    segment_count = starts_cpu.numel()
    control_inputs_cpu = treatment_inputs_cpu.reshape(
        segment_count * 2,
        control_block_size,
    )
    control_targets_cpu = treatment_targets_cpu.reshape(
        segment_count * 2,
        control_block_size,
    )

    if not torch.equal(
        control_targets_cpu.reshape(-1),
        treatment_targets_cpu.reshape(-1),
    ):
        raise RuntimeError("paired batching failed to preserve target order")

    return PairedContextBatch(
        starts=starts_cpu.clone(),
        control_inputs=control_inputs_cpu,
        control_targets=control_targets_cpu,
        treatment_inputs=treatment_inputs_cpu,
        treatment_targets=treatment_targets_cpu,
    )


def get_paired_context_batch(
    data: CharacterData,
    split: str,
    control: ContextSpec,
    treatment: ContextSpec,
    generator: torch.Generator,
) -> PairedContextBatch:
    _validate_context_geometry(control, treatment)
    source = _split_source(data, split)
    if len(source) <= treatment.block_size:
        raise ValueError(
            f"The {split} split has {len(source)} tokens, but treatment "
            f"block_size is {treatment.block_size}"
        )
    starts = torch.randint(
        0,
        len(source) - treatment.block_size,
        (treatment.batch_size,),
        generator=generator,
    )
    batch = build_paired_context_batch(
        source,
        starts,
        control_block_size=control.block_size,
        treatment_block_size=treatment.block_size,
    )
    if batch.control_inputs.shape != (control.batch_size, control.block_size):
        raise RuntimeError("constructed control batch has the wrong shape")
    return batch


def split_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    spec: ContextSpec,
    *,
    control: ContextSpec,
    treatment: ContextSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return overall, first-half, and second-half mean cross entropy."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (B,T,V), got {logits.shape}")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match the first two logits dimensions")
    if tuple(targets.shape) != (spec.batch_size, spec.block_size):
        raise ValueError(
            f"targets do not match {spec.name} spec: {tuple(targets.shape)}"
        )

    vocab_size = logits.shape[-1]
    overall = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )
    spec_geometry = (spec.batch_size, spec.block_size)
    treatment_geometry = (treatment.batch_size, treatment.block_size)
    control_geometry = (control.batch_size, control.block_size)
    if spec_geometry == treatment_geometry:
        first_logits = logits[:, : control.block_size]
        first_targets = targets[:, : control.block_size]
        second_logits = logits[:, control.block_size :]
        second_targets = targets[:, control.block_size :]
    elif spec_geometry == control_geometry:
        segment_count = treatment.batch_size
        reshaped_logits = logits.reshape(
            segment_count,
            2,
            control.block_size,
            vocab_size,
        )
        reshaped_targets = targets.reshape(
            segment_count,
            2,
            control.block_size,
        )
        first_logits = reshaped_logits[:, 0]
        first_targets = reshaped_targets[:, 0]
        second_logits = reshaped_logits[:, 1]
        second_targets = reshaped_targets[:, 1]
    else:
        raise ValueError(f"unknown context spec: {spec}")

    first = F.cross_entropy(
        first_logits.reshape(-1, vocab_size),
        first_targets.reshape(-1),
    )
    second = F.cross_entropy(
        second_logits.reshape(-1, vocab_size),
        second_targets.reshape(-1),
    )
    return overall, first, second


def _as_split_losses(
    losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> SplitLosses:
    return SplitLosses(*(loss.detach().item() for loss in losses))


def _mean_split_losses(values: Sequence[SplitLosses]) -> SplitLosses:
    if not values:
        raise ValueError("at least one split-loss value is required")
    count = len(values)
    return SplitLosses(
        overall=sum(value.overall for value in values) / count,
        first_half=sum(value.first_half for value in values) / count,
        second_half=sum(value.second_half for value in values) / count,
    )


@torch.no_grad()
def estimate_split_loss(
    model: GPTLanguageModel,
    data: CharacterData,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
    device: torch.device,
    *,
    split: str,
    eval_iters: int,
    generator: torch.Generator,
) -> SplitLosses:
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    was_training = model.training
    model.eval()
    values: list[SplitLosses] = []
    try:
        for _ in range(eval_iters):
            paired = get_paired_context_batch(
                data,
                split,
                control,
                treatment,
                generator,
            )
            inputs, targets = paired.for_spec(spec, device)
            logits, _ = model(inputs)
            values.append(
                _as_split_losses(
                    split_cross_entropy(
                        logits,
                        targets,
                        spec,
                        control=control,
                        treatment=treatment,
                    )
                )
            )
    finally:
        model.train(was_training)
    return _mean_split_losses(values)


def evaluate_on_fixed_paired_batches(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
    device: torch.device,
) -> ContextLossEstimate:
    generator = torch.Generator().manual_seed(config.seed + 2)
    train = estimate_split_loss(
        model,
        data,
        spec,
        control,
        treatment,
        device,
        split="train",
        eval_iters=config.eval_iters,
        generator=generator,
    )
    val = estimate_split_loss(
        model,
        data,
        spec,
        control,
        treatment,
        device,
        split="val",
        eval_iters=config.eval_iters,
        generator=generator,
    )
    return ContextLossEstimate(train=train, val=val)


def _build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=DEFAULT_DROPOUT,
    )
    if device is not None:
        model = model.to(device)
    return model


def _clone_state(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _project_treatment_state_to_control(
    treatment_state: Mapping[str, torch.Tensor],
    control_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Project the longer state onto every tensor represented by T=64."""

    if treatment_state.keys() != control_state.keys():
        raise RuntimeError("context models have different state-dict keys")
    projected: dict[str, torch.Tensor] = {}
    for name, control_tensor in control_state.items():
        treatment_tensor = treatment_state[name]
        if treatment_tensor.shape == control_tensor.shape:
            projected[name] = treatment_tensor.detach().cpu().clone()
        elif name == "position_embedding_table.weight":
            if (
                treatment_tensor.ndim != 2
                or treatment_tensor.shape[1] != control_tensor.shape[1]
                or treatment_tensor.shape[0] < control_tensor.shape[0]
            ):
                raise RuntimeError("unexpected positional-embedding shapes")
            projected[name] = (
                treatment_tensor[: control_tensor.shape[0]].detach().cpu().clone()
            )
        elif name.endswith(".tril"):
            if treatment_tensor.ndim != 2 or control_tensor.ndim != 2:
                raise RuntimeError("causal mask buffers must be matrices")
            projected[name] = (
                treatment_tensor[
                    : control_tensor.shape[0],
                    : control_tensor.shape[1],
                ]
                .detach()
                .cpu()
                .clone()
            )
        else:
            raise RuntimeError(
                f"unexpected context-dependent state tensor {name!r}: "
                f"{tuple(control_tensor.shape)} vs "
                f"{tuple(treatment_tensor.shape)}"
            )
    return projected


def create_paired_initial_states(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    n_head: int,
    n_layer: int,
) -> PairedInitialStates:
    """Create overlap-matched T=64 and T=128 initial model states.

    The control is constructed first from the canonical seed. Treatment-only
    positional rows are drawn from the continuation of that RNG stream. A
    correctly sized treatment shell is then filled with every shared control
    tensor, the shared positional prefix, and those new rows.
    """

    if control_config.seed != treatment_config.seed:
        raise ValueError("control and treatment initialization seeds must match")
    if control_config.n_embd != treatment_config.n_embd:
        raise ValueError("control and treatment embedding widths must match")
    if treatment_config.block_size != 2 * control_config.block_size:
        raise ValueError("treatment context must be exactly twice the control")

    seed_everything(control_config.seed)
    control_model = _build_model(data, control_config, n_head, n_layer)
    control_state = _clone_state(control_model)
    control_parameter_count = sum(
        parameter.numel()
        for parameter in control_model.parameters()
        if parameter.requires_grad
    )

    # Draw the treatment-only positional rows from the continuation of the
    # canonical control RNG stream. This avoids reusing random values that
    # already initialized shared control weights.
    extra_row_count = treatment_config.block_size - control_config.block_size
    extra_position_rows = torch.empty(extra_row_count, control_config.n_embd)
    torch.nn.init.normal_(extra_position_rows)
    extra_position_rows_sha256 = tensor_sha256(extra_position_rows)

    # This model supplies correctly sized buffers and a strict-load target.
    # Every trainable tensor is overwritten below, so its placeholder draws do
    # not enter either branch's initialization.
    treatment_model = _build_model(data, treatment_config, n_head, n_layer)
    treatment_state = _clone_state(treatment_model)
    treatment_parameter_count = sum(
        parameter.numel()
        for parameter in treatment_model.parameters()
        if parameter.requires_grad
    )

    for name, control_tensor in control_state.items():
        treatment_tensor = treatment_state[name]
        if treatment_tensor.shape == control_tensor.shape:
            treatment_state[name] = control_tensor.clone()
        elif name == "position_embedding_table.weight":
            treatment_state[name][: control_config.block_size].copy_(control_tensor)
            treatment_state[name][control_config.block_size :].copy_(
                extra_position_rows
            )
        elif name.endswith(".tril"):
            overlap = treatment_tensor[
                : control_tensor.shape[0],
                : control_tensor.shape[1],
            ]
            if not torch.equal(overlap, control_tensor):
                raise RuntimeError(f"causal-mask overlap differs for {name}")
        else:
            raise RuntimeError(f"unhandled context-dependent tensor: {name}")

    projected = _project_treatment_state_to_control(
        treatment_state,
        control_state,
    )
    common_hash = tensor_mapping_sha256(control_state)
    if tensor_mapping_sha256(projected) != common_hash:
        raise RuntimeError("T=128 initialization does not match the T=64 overlap")

    # Strict-load both independently before returning portable CPU snapshots.
    control_model.load_state_dict(control_state, strict=True)
    treatment_model.load_state_dict(treatment_state, strict=True)
    del control_model
    del treatment_model
    return PairedInitialStates(
        control=control_state,
        treatment=treatment_state,
        common_state_sha256=common_hash,
        control_state_sha256=tensor_mapping_sha256(control_state),
        treatment_state_sha256=tensor_mapping_sha256(treatment_state),
        extra_position_rows_sha256=extra_position_rows_sha256,
        control_parameter_count=control_parameter_count,
        treatment_parameter_count=treatment_parameter_count,
    )


def verify_initial_first_half_equivalence(
    control_model: GPTLanguageModel,
    treatment_model: GPTLanguageModel,
    data: CharacterData,
    control: ContextSpec,
    treatment: ContextSpec,
    device: torch.device,
    *,
    seed: int,
) -> float:
    """Check that added context is the only initial first-half difference."""

    generator = torch.Generator().manual_seed(seed)
    paired = get_paired_context_batch(
        data,
        "val",
        control,
        treatment,
        generator,
    )
    control_was_training = control_model.training
    treatment_was_training = treatment_model.training
    control_model.eval()
    treatment_model.eval()
    try:
        with torch.no_grad():
            control_inputs, _ = paired.for_spec(control, device)
            treatment_inputs, _ = paired.for_spec(treatment, device)
            control_logits, _ = control_model(
                control_inputs.reshape(
                    treatment.batch_size,
                    2,
                    control.block_size,
                )[:, 0]
            )
            treatment_logits, _ = treatment_model(treatment_inputs)
            difference = (
                (control_logits - treatment_logits[:, : control.block_size])
                .abs()
                .max()
                .item()
            )
    finally:
        control_model.train(control_was_training)
        treatment_model.train(treatment_was_training)
    if difference > 1e-6:
        raise RuntimeError(
            "initial first-half logits differ despite overlap-matched state: "
            f"max absolute difference {difference}"
        )
    return difference


@dataclass(slots=True)
class ContextValidationCheckpoint(BestValidationCheckpoint):
    """Persist a Stage 17 best checkpoint and the complete pairing contract."""

    branch_spec: ContextSpec | None = None
    control_spec: ContextSpec | None = None
    treatment_spec: ContextSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initialization_seed: int = 0
    branch_initial_state_sha256: str = ""
    common_state_sha256: str = ""
    extra_position_rows_sha256: str = ""
    training_batch_seed: int = 0
    training_rng_seed: int = 0
    attention_shape: tuple[int, ...] = ()
    best_validation_losses: SplitLosses | None = None

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(ContextValidationCheckpoint, self)._payload(step)
        if (
            self.branch_spec is None
            or self.control_spec is None
            or self.treatment_spec is None
            or self.schedule is None
            or self.best_validation_losses is None
        ):
            raise RuntimeError("incomplete Stage 17 checkpoint metadata")

        optimizer_rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(optimizer_rates) != 1:
            raise RuntimeError("all optimizer groups must use one LR")
        optimizer_rate = optimizer_rates.pop()
        expected_rate = self.schedule.for_update(max(0, step - 1))
        if optimizer_rate != expected_rate:
            raise RuntimeError(
                f"optimizer LR at step {step} must be {expected_rate:g}, "
                f"got {optimizer_rate:g}"
            )

        schedule_metadata = self.schedule.as_metadata(max_iters=self.target_step)
        payload["checkpoint_kind"] = "best"
        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "residual_dropout": DEFAULT_DROPOUT,
                "dropout_placement": DROPOUT_PLACEMENT,
            }
        )
        training_config = payload["training_config"]
        assert isinstance(training_config, dict)
        training_config.update(
            {
                "learning_rate": optimizer_rate,
                "max_iters": self.target_step,
                "learning_rate_schedule": schedule_metadata,
                "residual_dropout": DEFAULT_DROPOUT,
                "training_batch_seed": self.training_batch_seed,
                "training_rng_seed": self.training_rng_seed,
                "paired_segment_batch_size": self.treatment_spec.batch_size,
                "targets_per_update": self.control_spec.tokens_per_update,
                "total_target_characters": (
                    self.control_spec.tokens_per_update * self.target_step
                ),
            }
        )
        payload["initialization"] = {
            "kind": "overlap_matched_context_initialization",
            "seed": self.initialization_seed,
            "branch_state_sha256": self.branch_initial_state_sha256,
            "common_t64_projection_sha256": self.common_state_sha256,
            "shared_position_rows": self.control_spec.block_size,
            "extra_position_rows": (
                self.treatment_spec.block_size - self.control_spec.block_size
            ),
            "extra_position_initialization": ("normal_from_post_control_rng_stream"),
            "extra_position_rows_sha256": self.extra_position_rows_sha256,
        }
        payload["experiment"] = {
            "stage": 17,
            "branch": self.branch_spec.name,
            "from_scratch": True,
            "comparison_variable": "context_length",
            "branch_batch_size": self.branch_spec.batch_size,
            "branch_block_size": self.branch_spec.block_size,
            "control_batch_size": self.control_spec.batch_size,
            "control_block_size": self.control_spec.block_size,
            "treatment_batch_size": self.treatment_spec.batch_size,
            "treatment_block_size": self.treatment_spec.block_size,
            "paired_target_characters": True,
            "identical_targets_per_update": True,
            "targets_per_update": self.control_spec.tokens_per_update,
            "total_target_characters": (
                self.control_spec.tokens_per_update * self.target_step
            ),
            "overlap_matched_initialization": True,
            "identical_full_initialization": False,
            "common_t64_projection_sha256": self.common_state_sha256,
            "initialization_seed": self.initialization_seed,
            "training_batch_seed": self.training_batch_seed,
            "training_rng_seed": self.training_rng_seed,
            "residual_dropout": DEFAULT_DROPOUT,
            "learning_rate_schedule": schedule_metadata,
            "attention_shape": self.attention_shape,
        }
        payload["best_validation_losses"] = {
            "overall": self.best_validation_losses.overall,
            "first_half": self.best_validation_losses.first_half,
            "second_half": self.best_validation_losses.second_half,
        }
        return payload

    def consider(self, record: ContextEvaluationRecord) -> bool:
        val_loss = record.losses.val.overall
        if not math.isfinite(val_loss):
            raise ValueError(f"validation loss at step {record.step} must be finite")
        if val_loss >= self.best_val_loss:
            return False
        self.best_val_loss = val_loss
        self.best_step = record.step
        self.best_validation_losses = record.losses.val
        self.save(record.step)
        return True


def inspect_attention_shape(
    model: GPTLanguageModel,
    spec: ContextSpec,
    n_head: int,
    device: torch.device,
) -> tuple[int, ...]:
    was_training = model.training
    model.eval()
    try:
        inputs = torch.zeros(
            (spec.batch_size, spec.block_size),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            weights = model.get_attention_weights(inputs, block_index=0)
        shape = tuple(weights.shape)
        expected = (
            spec.batch_size,
            n_head,
            spec.block_size,
            spec.block_size,
        )
        if shape != expected:
            raise RuntimeError(
                f"{spec.name} attention shape must be {expected}, got {shape}"
            )
        return shape
    finally:
        model.train(was_training)


def train_with_paired_schedule(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
    device: torch.device,
    training_generator: torch.Generator,
    schedule: LearningRateSchedule,
    *,
    on_evaluation: Callable[[ContextEvaluationRecord], None],
    benchmark: BenchmarkConfig | None = None,
    on_update: Callable[[int, float, PairedContextBatch], None] | None = None,
) -> ContextTrainingResult:
    schedule.validate_target_step(config.max_iters)
    if benchmark is not None:
        required = benchmark.num_warmup + benchmark.num_steps
        if required > config.max_iters:
            raise ValueError(
                f"benchmark requires {required} steps, but max_iters is "
                f"{config.max_iters}"
            )

    history: list[ContextEvaluationRecord] = []
    benchmark_started_at: float | None = None
    benchmark_elapsed = 0.0
    benchmark_peak_allocated = 0
    benchmark_peak_reserved = 0
    benchmark_stats: BenchmarkStats | None = None

    def record(step: int) -> ContextEvaluationRecord:
        evaluation = ContextEvaluationRecord(
            step=step,
            losses=evaluate_on_fixed_paired_batches(
                model,
                data,
                config,
                spec,
                control,
                treatment,
                device,
            ),
        )
        history.append(evaluation)
        on_evaluation(evaluation)
        return evaluation

    model.train()
    initial_record = record(0)
    for update_index in range(config.max_iters):
        learning_rate = schedule.for_update(update_index)
        set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)

        if benchmark is not None and update_index == benchmark.num_warmup:
            sync_device(device)
            reset_peak_memory_stats(device)
            benchmark_started_at = time.perf_counter()

        paired = get_paired_context_batch(
            data,
            "train",
            control,
            treatment,
            training_generator,
        )
        if on_update is not None:
            on_update(update_index, learning_rate, paired)
        inputs, targets = paired.for_spec(spec, device)
        logits, overall = model(inputs, targets)
        assert overall is not None

        overall.backward()
        optimizer.step()
        del logits
        del overall
        del inputs
        del targets
        del paired

        completed_step = update_index + 1
        if (
            benchmark is not None
            and completed_step == benchmark.num_warmup + benchmark.num_steps
        ):
            sync_device(device)
            assert benchmark_started_at is not None
            elapsed = benchmark_elapsed + time.perf_counter() - benchmark_started_at
            iterations_per_sec = benchmark.num_steps / elapsed
            peak_allocated, peak_reserved = get_peak_memory_stats(device)
            benchmark_peak_allocated = max(
                benchmark_peak_allocated,
                peak_allocated,
            )
            benchmark_peak_reserved = max(
                benchmark_peak_reserved,
                peak_reserved,
            )
            benchmark_stats = BenchmarkStats(
                seconds=elapsed,
                iterations_per_sec=iterations_per_sec,
                tokens_per_sec=(iterations_per_sec * spec.tokens_per_update),
                peak_allocated_mb=benchmark_peak_allocated / 1024**2,
                peak_reserved_mb=benchmark_peak_reserved / 1024**2,
            )

        should_evaluate = (
            completed_step % config.eval_interval == 0
            or completed_step == schedule.first_decay_step
            or completed_step == schedule.second_decay_step
            or completed_step == config.max_iters
        )
        benchmark_end_step = (
            benchmark.num_warmup + benchmark.num_steps
            if benchmark is not None
            else None
        )
        timing_needs_pause = (
            should_evaluate
            and benchmark_started_at is not None
            and benchmark_stats is None
            and benchmark_end_step is not None
            and completed_step < benchmark_end_step
        )
        if timing_needs_pause:
            sync_device(device)
            benchmark_elapsed += time.perf_counter() - benchmark_started_at
            peak_allocated, peak_reserved = get_peak_memory_stats(device)
            benchmark_peak_allocated = max(
                benchmark_peak_allocated,
                peak_allocated,
            )
            benchmark_peak_reserved = max(
                benchmark_peak_reserved,
                peak_reserved,
            )
        if should_evaluate:
            record(completed_step)
        if timing_needs_pause:
            clear_accelerator_cache(device)
            reset_peak_memory_stats(device)
            sync_device(device)
            benchmark_started_at = time.perf_counter()

    return ContextTrainingResult(
        initial=initial_record.losses,
        final=history[-1].losses,
        history=tuple(history),
        benchmark=benchmark_stats,
    )


def _validate_branch_config(
    config: TrainingConfig,
    spec: ContextSpec,
    schedule: LearningRateSchedule,
) -> None:
    if config.batch_size != spec.batch_size:
        raise ValueError(
            f"{spec.name} config batch size {config.batch_size} does not "
            f"match spec {spec.batch_size}"
        )
    if config.block_size != spec.block_size:
        raise ValueError(
            f"{spec.name} config block size {config.block_size} does not "
            f"match spec {spec.block_size}"
        )
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError(
            f"{spec.name} config learning rate must equal schedule initial LR"
        )
    schedule.validate_target_step(config.max_iters)


def make_evaluation_callback(
    best: ContextValidationCheckpoint,
    schedule: LearningRateSchedule,
) -> Callable[[ContextEvaluationRecord], None]:
    def record_evaluation(record: ContextEvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        update_index = max(0, record.step - 1)
        learning_rate = schedule.for_update(update_index)
        train = record.losses.train
        val = record.losses.val
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr {learning_rate:g} | "
            f"train {train.overall:.4f} | val {val.overall:.4f} | "
            f"val first {val.first_half:.4f} | "
            f"val second {val.second_half:.4f} | "
            f"gap {val.overall - train.overall:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )

    return record_evaluation


def _print_benchmark(name: str, stats: BenchmarkStats) -> None:
    print(f"\n{name} training benchmark")
    print(f"  seconds:              {stats.seconds:.3f}")
    print(f"  iterations/sec:       {stats.iterations_per_sec:.3f}")
    print(f"  target characters/sec:{stats.tokens_per_sec:12.3f}")
    print(f"  peak allocated MiB:   {stats.peak_allocated_mb:.3f}")
    print(f"  peak reserved MiB:    {stats.peak_reserved_mb:.3f}")


def benchmark_context_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> BenchmarkStats:
    """Benchmark a disposable branch on the real paired-batch update path."""

    clear_accelerator_cache(device)
    model = _build_model(data, config, n_head, n_layer, device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    generator = torch.Generator().manual_seed(training_batch_seed)
    seed_everything(training_rng_seed)
    model.train()

    started_at: float | None = None
    total_steps = benchmark.num_warmup + benchmark.num_steps
    for update_index in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        if update_index == benchmark.num_warmup:
            sync_device(device)
            reset_peak_memory_stats(device)
            started_at = time.perf_counter()

        paired = get_paired_context_batch(
            data,
            "train",
            control,
            treatment,
            generator,
        )
        inputs, targets = paired.for_spec(spec, device)
        logits, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        del logits
        del loss
        del inputs
        del targets
        del paired

    sync_device(device)
    assert started_at is not None
    elapsed = time.perf_counter() - started_at
    peak_allocated, peak_reserved = get_peak_memory_stats(device)
    iterations_per_sec = benchmark.num_steps / elapsed
    stats = BenchmarkStats(
        seconds=elapsed,
        iterations_per_sec=iterations_per_sec,
        tokens_per_sec=iterations_per_sec * spec.tokens_per_update,
        peak_allocated_mb=peak_allocated / 1024**2,
        peak_reserved_mb=peak_reserved / 1024**2,
    )

    del optimizer
    del model
    clear_accelerator_cache(device)
    return stats


def aggregate_benchmark_stats(
    runs: Sequence[BenchmarkStats],
    *,
    steps_per_run: int,
) -> BenchmarkStats:
    if not runs:
        raise ValueError("at least one benchmark run is required")
    if steps_per_run <= 0:
        raise ValueError("steps_per_run must be positive")
    elapsed = sum(run.seconds for run in runs)
    iterations_per_sec = len(runs) * steps_per_run / elapsed
    tokens_per_iteration = runs[0].tokens_per_sec / runs[0].iterations_per_sec
    return BenchmarkStats(
        seconds=elapsed,
        iterations_per_sec=iterations_per_sec,
        tokens_per_sec=iterations_per_sec * tokens_per_iteration,
        peak_allocated_mb=max(run.peak_allocated_mb for run in runs),
        peak_reserved_mb=max(run.peak_reserved_mb for run in runs),
    )


def run_counterbalanced_benchmarks(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    control: ContextSpec,
    treatment: ContextSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    initialization: PairedInitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> tuple[BenchmarkStats, BenchmarkStats]:
    """Run adjacent C-T-T-C disposable benchmarks and aggregate by branch."""

    control_runs: list[BenchmarkStats] = []
    treatment_runs: list[BenchmarkStats] = []
    ordered_runs = (
        (
            control_config,
            control,
            initialization.control,
            control_runs,
        ),
        (
            treatment_config,
            treatment,
            initialization.treatment,
            treatment_runs,
        ),
        (
            treatment_config,
            treatment,
            initialization.treatment,
            treatment_runs,
        ),
        (
            control_config,
            control,
            initialization.control,
            control_runs,
        ),
    )
    for config, spec, initial_state, destination in ordered_runs:
        destination.append(
            benchmark_context_branch(
                data,
                config,
                spec,
                control,
                treatment,
                n_head,
                n_layer,
                device,
                initial_state,
                training_batch_seed=training_batch_seed,
                training_rng_seed=training_rng_seed,
                benchmark=benchmark,
            )
        )

    control_stats = aggregate_benchmark_stats(
        control_runs,
        steps_per_run=benchmark.num_steps,
    )
    treatment_stats = aggregate_benchmark_stats(
        treatment_runs,
        steps_per_run=benchmark.num_steps,
    )
    return control_stats, treatment_stats


def _expected_attention_shape(
    spec: ContextSpec,
    n_head: int,
) -> tuple[int, ...]:
    return (spec.batch_size, n_head, spec.block_size, spec.block_size)


def run_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    common_state_sha256: str,
    extra_position_rows_sha256: str,
    *,
    initialization_seed: int,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig | None,
) -> BranchReport:
    _validate_branch_config(config, spec, schedule)
    clear_accelerator_cache(device)
    model = _build_model(data, config, n_head, n_layer, device)
    model.load_state_dict(initial_state, strict=True)
    loaded_hash = tensor_mapping_sha256(model.state_dict())
    if loaded_hash != initial_state_sha256:
        raise RuntimeError(f"{spec.name} did not load its exact initial state")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.for_update(0),
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    expected_attention_shape = _expected_attention_shape(spec, n_head)
    best = ContextValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=spec.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        branch_spec=spec,
        control_spec=control,
        treatment_spec=treatment,
        schedule=schedule,
        target_step=config.max_iters,
        initialization_seed=initialization_seed,
        branch_initial_state_sha256=initial_state_sha256,
        common_state_sha256=common_state_sha256,
        extra_position_rows_sha256=extra_position_rows_sha256,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        attention_shape=expected_attention_shape,
    )
    record_evaluation = make_evaluation_callback(best, schedule)

    print(f"\n{spec.name} branch (B={spec.batch_size}, T={spec.block_size}, dropout=0)")
    print(f"  branch initialization: {initial_state_sha256}")
    print(f"  common T=64 projection:{common_state_sha256}")
    print(f"  training-batch seed:   {training_batch_seed}")
    print(f"  training RNG seed:     {training_rng_seed}")

    seed_everything(training_rng_seed)
    result = train_with_paired_schedule(
        model,
        optimizer,
        data,
        config,
        spec,
        control,
        treatment,
        device,
        training_generator,
        schedule,
        on_evaluation=record_evaluation,
        benchmark=benchmark,
    )
    if result.benchmark is not None:
        _print_benchmark(spec.name, result.benchmark)

    # Inspect after the timed region so this diagnostic cannot inflate the
    # benchmark allocator peaks.
    attention_shape = inspect_attention_shape(model, spec, n_head, device)
    if attention_shape != expected_attention_shape:
        raise RuntimeError("attention shape changed during training")

    assert best.best_step is not None
    assert best.best_validation_losses is not None
    final_batch_generator_sha256 = tensor_sha256(training_generator.get_state())
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=config.seed + 5,
    )
    report = BranchReport(
        spec=spec,
        parameter_count=parameter_count,
        initial_state_sha256=initial_state_sha256,
        training=result,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        best_validation_losses=best.best_validation_losses,
        final_batch_generator_sha256=final_batch_generator_sha256,
        attention_shape=attention_shape,
        sample=sample,
    )

    del record_evaluation
    del best
    del optimizer
    del model
    clear_accelerator_cache(device)
    return report


def run_experiment(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    control: ContextSpec,
    treatment: ContextSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig | None,
) -> ExperimentReport:
    targets_per_update = validate_context_protocol(control, treatment)
    _validate_branch_config(control_config, control, schedule)
    _validate_branch_config(treatment_config, treatment, schedule)
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if control_config.n_embd != treatment_config.n_embd:
        raise ValueError("branch embedding widths must match")
    if (
        control_config.max_iters != treatment_config.max_iters
        or control_config.eval_interval != treatment_config.eval_interval
        or control_config.eval_iters != treatment_config.eval_iters
        or control_config.seed != treatment_config.seed
    ):
        raise ValueError("branch training/evaluation schedules must match")
    if n_head <= 0 or control_config.n_embd % n_head != 0:
        raise ValueError(
            f"n_embd ({control_config.n_embd}) must be divisible by a "
            f"positive n_head ({n_head})"
        )
    if n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {n_layer}")
    if benchmark is not None:
        benchmark_steps = benchmark.num_warmup + benchmark.num_steps
        if benchmark_steps > control_config.max_iters:
            raise ValueError(
                f"benchmark requires {benchmark_steps} steps, but max_iters "
                f"is {control_config.max_iters}"
            )
    for split in ("train", "val"):
        source = _split_source(data, split)
        if len(source) <= treatment.block_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but treatment "
                f"block_size is {treatment.block_size}"
            )
    _validate_non_negative_integer(training_batch_seed, name="training batch seed")
    _validate_non_negative_integer(training_rng_seed, name="training RNG seed")

    initialization = create_paired_initial_states(
        data,
        control_config,
        treatment_config,
        n_head,
        n_layer,
    )
    print("\nControlled from-initialization protocol")
    print("  Both branches train from step zero on paired target characters.")
    print(
        "  Every common parameter and the first 64 positional rows are "
        "initialized identically."
    )
    print(
        f"  Targets per update: {targets_per_update:,}; total per branch: "
        f"{targets_per_update * control_config.max_iters:,}."
    )
    print(f"  common T=64 projection SHA-256: {initialization.common_state_sha256}")
    print(
        "  extra T=128 position rows SHA-256: "
        f"{initialization.extra_position_rows_sha256}"
    )

    # This audit uses CPU models and runs before either branch's accelerator
    # benchmark. It catches accidental initialization confounds directly.
    control_audit_model = _build_model(
        data,
        control_config,
        n_head,
        n_layer,
        torch.device("cpu"),
    )
    treatment_audit_model = _build_model(
        data,
        treatment_config,
        n_head,
        n_layer,
        torch.device("cpu"),
    )
    control_audit_model.load_state_dict(initialization.control, strict=True)
    treatment_audit_model.load_state_dict(initialization.treatment, strict=True)
    initial_first_half_max_difference = verify_initial_first_half_equivalence(
        control_audit_model,
        treatment_audit_model,
        data,
        control,
        treatment,
        torch.device("cpu"),
        seed=control_config.seed + 6,
    )
    del control_audit_model
    del treatment_audit_model
    print(
        "  Initial first-half max logit difference: "
        f"{initial_first_half_max_difference:.3e}"
    )

    control_benchmark: BenchmarkStats | None = None
    treatment_benchmark: BenchmarkStats | None = None
    if benchmark is not None:
        print("\nCounterbalanced disposable training benchmark")
        print("  Order: T=64, T=128, T=128, T=64")
        print(
            f"  Each run: {benchmark.num_warmup} warmup + "
            f"{benchmark.num_steps} timed updates"
        )
        control_benchmark, treatment_benchmark = run_counterbalanced_benchmarks(
            data,
            control_config,
            treatment_config,
            control,
            treatment,
            n_head,
            n_layer,
            device,
            initialization,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            benchmark=benchmark,
        )
        _print_benchmark(control.name, control_benchmark)
        _print_benchmark(treatment.name, treatment_benchmark)

    control_report = run_branch(
        data,
        control_config,
        control,
        control,
        treatment,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        initialization.control,
        initialization.control_state_sha256,
        initialization.common_state_sha256,
        initialization.extra_position_rows_sha256,
        initialization_seed=control_config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        benchmark=None,
    )
    treatment_report = run_branch(
        data,
        treatment_config,
        treatment,
        control,
        treatment,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        initialization.treatment,
        initialization.treatment_state_sha256,
        initialization.common_state_sha256,
        initialization.extra_position_rows_sha256,
        initialization_seed=treatment_config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        benchmark=None,
    )

    if control_benchmark is not None and treatment_benchmark is not None:
        control_report = replace(
            control_report,
            training=replace(
                control_report.training,
                benchmark=control_benchmark,
            ),
        )
        treatment_report = replace(
            treatment_report,
            training=replace(
                treatment_report.training,
                benchmark=treatment_benchmark,
            ),
        )

    if control_report.parameter_count != initialization.control_parameter_count:
        raise RuntimeError("control parameter count changed")
    if treatment_report.parameter_count != (initialization.treatment_parameter_count):
        raise RuntimeError("treatment parameter count changed")
    expected_parameter_delta = (
        treatment.block_size - control.block_size
    ) * control_config.n_embd
    if (
        treatment_report.parameter_count - control_report.parameter_count
        != expected_parameter_delta
    ):
        raise RuntimeError("parameter delta is not solely positional embeddings")
    if control_report.final_batch_generator_sha256 != (
        treatment_report.final_batch_generator_sha256
    ):
        raise RuntimeError("branches did not consume the same sampled starts")
    if tuple(record.step for record in control_report.training.history) != tuple(
        record.step for record in treatment_report.training.history
    ):
        raise RuntimeError("branches were not evaluated at matching steps")

    final_val_delta = (
        treatment_report.training.final.val.overall
        - control_report.training.final.val.overall
    )
    best_val_delta = treatment_report.best_val_loss - control_report.best_val_loss

    print("\nStage 17 experimental summary")
    print(
        f"  control B/T, treatment B/T:  "
        f"{control.batch_size}/{control.block_size}, "
        f"{treatment.batch_size}/{treatment.block_size}"
    )
    print(f"  target characters/update:    {targets_per_update:,}")
    print(
        f"  parameter counts:            "
        f"{control_report.parameter_count:,} / "
        f"{treatment_report.parameter_count:,}"
    )
    print(
        f"  attention shapes:            {control_report.attention_shape} / "
        f"{treatment_report.attention_shape}"
    )
    print(
        f"  attention elements/block:    "
        f"{math.prod(control_report.attention_shape):,} / "
        f"{math.prod(treatment_report.attention_shape):,}"
    )
    print(
        f"  control final train / val:   "
        f"{control_report.training.final.train.overall:.4f} / "
        f"{control_report.training.final.val.overall:.4f}"
    )
    print(
        f"  treatment final train / val: "
        f"{treatment_report.training.final.train.overall:.4f} / "
        f"{treatment_report.training.final.val.overall:.4f}"
    )
    print(
        f"  final val delta (128-64):    {final_val_delta:+.4f} (negative favors T=128)"
    )
    print(
        f"  control best val:            {control_report.best_val_loss:.4f} "
        f"@ {control_report.best_step}"
    )
    print(
        f"  treatment best val:          {treatment_report.best_val_loss:.4f} "
        f"@ {treatment_report.best_step}"
    )
    print(
        f"  best val delta (128-64):     {best_val_delta:+.4f} (negative favors T=128)"
    )
    print(f"  control checkpoint:          {control.checkpoint_path}")
    print(f"  treatment checkpoint:        {treatment.checkpoint_path}")
    print("\nT=64 final-step generated text")
    print(control_report.sample)
    print("\nT=128 final-step generated text")
    print(treatment_report.sample)

    return ExperimentReport(
        initialization=initialization,
        initialization_seed=control_config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        targets_per_update=targets_per_update,
        control=control_report,
        treatment=treatment_report,
        final_val_delta=final_val_delta,
        best_val_delta=best_val_delta,
    )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def expected_training_generator_state(
    data: CharacterData,
    treatment: ContextSpec,
    *,
    training_batch_seed: int,
    completed_steps: int,
) -> torch.Tensor:
    """Reconstruct the long-segment start sampler through a saved step."""

    _validate_non_negative_integer(training_batch_seed, name="training batch seed")
    _validate_non_negative_integer(completed_steps, name="completed steps")
    maximum_start = len(data.train_data) - treatment.block_size
    if maximum_start <= 0:
        raise ValueError(
            f"training split has {len(data.train_data)} tokens, but treatment "
            f"block size is {treatment.block_size}"
        )
    generator = torch.Generator().manual_seed(training_batch_seed)
    for _ in range(completed_steps):
        torch.randint(
            0,
            maximum_start,
            (treatment.batch_size,),
            generator=generator,
        )
    return generator.get_state().clone()


def validate_stage_17_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    *,
    initialization_seed: int,
    branch_initial_state_sha256: str,
    common_state_sha256: str,
    extra_position_rows_sha256: str,
    training_batch_seed: int,
    training_rng_seed: int,
) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 17 checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = _require_mapping(checkpoint, name=str(path))
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("Stage 17 checkpoint_version must be 1")
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("Stage 17 checkpoint_kind must be 'best'")
    step = checkpoint.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= config.max_iters
    ):
        raise ValueError(
            f"checkpoint step must be in [0, {config.max_iters}], got {step!r}"
        )
    if checkpoint.get("best_step") != step:
        raise ValueError("checkpoint best_step must equal its saved step")
    best_val_loss = checkpoint.get("best_val_loss")
    if (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise ValueError("checkpoint best_val_loss must be finite")

    architecture = _require_mapping(
        checkpoint.get("architecture"),
        name="architecture",
    )
    expected_architecture = {
        "block_size": spec.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "residual_dropout": DEFAULT_DROPOUT,
        "dropout_placement": DROPOUT_PLACEMENT,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"checkpoint architecture.{key} must be {expected!r}, "
                f"got {architecture.get(key)!r}"
            )
    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("checkpoint data fingerprint does not match")
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 17 checkpoint must have uninterrupted AdamW state")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 17 optimizer provenance must be known")

    schedule_metadata = schedule.as_metadata(max_iters=config.max_iters)
    initialization = _require_mapping(
        checkpoint.get("initialization"),
        name="initialization",
    )
    expected_initialization = {
        "kind": "overlap_matched_context_initialization",
        "seed": initialization_seed,
        "branch_state_sha256": branch_initial_state_sha256,
        "common_t64_projection_sha256": common_state_sha256,
        "shared_position_rows": control.block_size,
        "extra_position_rows": treatment.block_size - control.block_size,
        "extra_position_initialization": "normal_from_post_control_rng_stream",
        "extra_position_rows_sha256": extra_position_rows_sha256,
    }
    for key, expected in expected_initialization.items():
        if initialization.get(key) != expected:
            raise ValueError(
                f"checkpoint initialization.{key} must be {expected!r}, "
                f"got {initialization.get(key)!r}"
            )

    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    expected_experiment = {
        "stage": 17,
        "branch": spec.name,
        "from_scratch": True,
        "comparison_variable": "context_length",
        "branch_batch_size": spec.batch_size,
        "branch_block_size": spec.block_size,
        "control_batch_size": control.batch_size,
        "control_block_size": control.block_size,
        "treatment_batch_size": treatment.batch_size,
        "treatment_block_size": treatment.block_size,
        "paired_target_characters": True,
        "identical_targets_per_update": True,
        "targets_per_update": control.tokens_per_update,
        "total_target_characters": control.tokens_per_update * config.max_iters,
        "overlap_matched_initialization": True,
        "identical_full_initialization": False,
        "common_t64_projection_sha256": common_state_sha256,
        "initialization_seed": initialization_seed,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "residual_dropout": DEFAULT_DROPOUT,
        "learning_rate_schedule": schedule_metadata,
        "attention_shape": _expected_attention_shape(spec, n_head),
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"checkpoint experiment.{key} must be {expected!r}, "
                f"got {experiment.get(key)!r}"
            )

    training_config = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    expected_training = {
        "batch_size": spec.batch_size,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": config.max_iters,
        "learning_rate_schedule": schedule_metadata,
        "residual_dropout": DEFAULT_DROPOUT,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "paired_segment_batch_size": treatment.batch_size,
        "targets_per_update": control.tokens_per_update,
        "total_target_characters": control.tokens_per_update * config.max_iters,
    }
    for key, expected in expected_training.items():
        if training_config.get(key) != expected:
            raise ValueError(
                f"checkpoint training_config.{key} must be {expected!r}, "
                f"got {training_config.get(key)!r}"
            )
    expected_learning_rate = schedule.for_update(max(0, step - 1))
    if training_config.get("learning_rate") != expected_learning_rate:
        raise ValueError("checkpoint training learning rate is inconsistent")

    split_losses = _require_mapping(
        checkpoint.get("best_validation_losses"),
        name="best_validation_losses",
    )
    for key in ("overall", "first_half", "second_half"):
        value = split_losses.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"checkpoint best_validation_losses.{key} must be finite")
    if float(split_losses["overall"]) != float(best_val_loss):
        raise ValueError("overall split loss must equal best_val_loss")

    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"),
        name="optimizer_state_dict",
    )
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("checkpoint optimizer must contain parameter groups")
    if not all(isinstance(group, Mapping) and "lr" in group for group in param_groups):
        raise ValueError("every optimizer parameter group must contain an LR")
    optimizer_rates = {
        float(group["lr"]) for group in param_groups if isinstance(group, Mapping)
    }
    if len(optimizer_rates) != 1 or optimizer_rates != {expected_learning_rate}:
        raise ValueError("checkpoint optimizer LR is inconsistent")
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("checkpoint training generator state must be a tensor")
    expected_generator_state = expected_training_generator_state(
        data,
        treatment,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator_state):
        raise ValueError(
            "checkpoint training generator state does not match the paired "
            f"long-start stream at step {step}"
        )
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    with torch.random.fork_rng(devices=[]):
        compatibility_model = _build_model(data, config, n_head, n_layer)
        compatibility_model.load_state_dict(model_state, strict=True)
        del compatibility_model
    return checkpoint


@torch.no_grad()
def estimate_validation_precise(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
    device: torch.device,
    *,
    eval_iters: int,
    seed: int,
) -> tuple[SplitLosses, ...]:
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    _validate_non_negative_integer(seed, name="precise evaluation seed")
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    values: list[SplitLosses] = []
    try:
        for _ in range(eval_iters):
            paired = get_paired_context_batch(
                data,
                "val",
                control,
                treatment,
                generator,
            )
            inputs, targets = paired.for_spec(spec, device)
            logits, _ = model(inputs)
            values.append(
                _as_split_losses(
                    split_cross_entropy(
                        logits,
                        targets,
                        spec,
                        control=control,
                        treatment=treatment,
                    )
                )
            )
    finally:
        model.train(was_training)
    return tuple(values)


def _mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    mean = tensor.mean().item()
    if len(values) == 1:
        return mean, 0.0
    standard_error = (tensor.std(unbiased=True) / math.sqrt(len(values))).item()
    return mean, standard_error


def _split_statistics(values: Sequence[SplitLosses]) -> SplitStatistics:
    overall = _mean_and_standard_error([value.overall for value in values])
    first = _mean_and_standard_error([value.first_half for value in values])
    second = _mean_and_standard_error([value.second_half for value in values])
    return SplitStatistics(
        mean=SplitLosses(overall[0], first[0], second[0]),
        standard_error=SplitLosses(overall[1], first[1], second[1]),
    )


def _paired_delta(
    candidate: PreciseBranchResult,
    baseline: PreciseBranchResult,
) -> PreciseDelta:
    if len(candidate.batch_losses) != len(baseline.batch_losses):
        raise ValueError("paired validation results must have equal lengths")
    differences = tuple(
        SplitLosses(
            candidate_value.overall - baseline_value.overall,
            candidate_value.first_half - baseline_value.first_half,
            candidate_value.second_half - baseline_value.second_half,
        )
        for candidate_value, baseline_value in zip(
            candidate.batch_losses,
            baseline.batch_losses,
            strict=True,
        )
    )
    statistics = _split_statistics(differences)
    low = SplitLosses(
        statistics.mean.overall - 1.96 * statistics.standard_error.overall,
        statistics.mean.first_half - 1.96 * statistics.standard_error.first_half,
        statistics.mean.second_half - 1.96 * statistics.standard_error.second_half,
    )
    high = SplitLosses(
        statistics.mean.overall + 1.96 * statistics.standard_error.overall,
        statistics.mean.first_half + 1.96 * statistics.standard_error.first_half,
        statistics.mean.second_half + 1.96 * statistics.standard_error.second_half,
    )
    return PreciseDelta(
        candidate=candidate.name,
        baseline=baseline.name,
        mean_delta=statistics.mean,
        standard_error=statistics.standard_error,
        confidence_low=low,
        confidence_high=high,
    )


def run_precise_validation(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    control: ContextSpec,
    treatment: ContextSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    initialization: PairedInitialStates,
    *,
    initialization_seed: int,
    training_batch_seed: int,
    training_rng_seed: int,
    eval_iters: int,
    seed: int,
) -> PreciseValidationReport:
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    results: list[PreciseBranchResult] = []
    branches = (
        (
            control,
            control_config,
            initialization.control_state_sha256,
        ),
        (
            treatment,
            treatment_config,
            initialization.treatment_state_sha256,
        ),
    )
    for spec, config, branch_initial_hash in branches:
        before_hash = checkpoint_sha256(spec.checkpoint_path)
        checkpoint = validate_stage_17_checkpoint(
            spec.checkpoint_path,
            data,
            config,
            spec,
            control,
            treatment,
            n_head,
            n_layer,
            schedule,
            initialization_seed=initialization_seed,
            branch_initial_state_sha256=branch_initial_hash,
            common_state_sha256=initialization.common_state_sha256,
            extra_position_rows_sha256=(initialization.extra_position_rows_sha256),
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
        )
        model = _build_model(data, config, n_head, n_layer, device)
        model_state = _require_mapping(
            checkpoint.get("model_state_dict"),
            name="model_state_dict",
        )
        model.load_state_dict(model_state, strict=True)
        batch_losses = estimate_validation_precise(
            model,
            data,
            config,
            spec,
            control,
            treatment,
            device,
            eval_iters=eval_iters,
            seed=seed,
        )
        if checkpoint_sha256(spec.checkpoint_path) != before_hash:
            raise RuntimeError(
                f"checkpoint changed during evaluation: {spec.checkpoint_path}"
            )
        results.append(
            PreciseBranchResult(
                name=spec.name,
                checkpoint_path=spec.checkpoint_path,
                checkpoint_sha256=before_hash,
                checkpoint_step=int(checkpoint["step"]),
                fixed_panel_loss=float(checkpoint["best_val_loss"]),
                statistics=_split_statistics(batch_losses),
                batch_losses=batch_losses,
            )
        )
        del model
        clear_accelerator_cache(device)

    delta = _paired_delta(results[1], results[0])
    return PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        results=tuple(results),
        delta=delta,
    )


def print_precise_validation(report: PreciseValidationReport) -> None:
    print("\nFresh paired validation of Stage 17 best checkpoints")
    print(
        f"  {report.eval_iters} paired validation batches per checkpoint, "
        f"seed={report.seed}"
    )
    for result in report.results:
        mean = result.statistics.mean
        se = result.statistics.standard_error
        print(
            f"  {result.name:16s} | step {result.checkpoint_step:5d} | "
            f"fixed {result.fixed_panel_loss:.4f} | overall "
            f"{mean.overall:.6f} +/- {se.overall:.6f} SE | first "
            f"{mean.first_half:.6f} +/- {se.first_half:.6f} | second "
            f"{mean.second_half:.6f} +/- {se.second_half:.6f}"
        )
        print(f"    checkpoint SHA-256: {result.checkpoint_sha256}")
    delta = report.delta
    for label, mean, se, low, high in (
        (
            "overall",
            delta.mean_delta.overall,
            delta.standard_error.overall,
            delta.confidence_low.overall,
            delta.confidence_high.overall,
        ),
        (
            "first half/window",
            delta.mean_delta.first_half,
            delta.standard_error.first_half,
            delta.confidence_low.first_half,
            delta.confidence_high.first_half,
        ),
        (
            "second half/window",
            delta.mean_delta.second_half,
            delta.standard_error.second_half,
            delta.confidence_low.second_half,
            delta.confidence_high.second_half,
        ),
    ):
        print(
            f"  delta {delta.candidate} - {delta.baseline}, {label}: "
            f"{mean:+.6f} +/- {se:.6f} SE "
            f"(95% CI [{low:+.6f}, {high:+.6f}]); "
            "negative favors T=128"
        )


def main() -> None:
    args = parse_args()
    if args.sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if args.precise_eval_iters < 0:
        raise ValueError("precise-eval-iters must be non-negative")
    if args.benchmark_warmup < 0:
        raise ValueError("benchmark-warmup must be non-negative")
    if args.benchmark_steps < 0:
        raise ValueError("benchmark-steps must be non-negative")
    if args.n_head <= 0 or args.n_embd % args.n_head != 0:
        raise ValueError(
            f"n_embd ({args.n_embd}) must be divisible by a positive "
            f"n_head ({args.n_head})"
        )
    if args.n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {args.n_layer}")

    schedule = LearningRateSchedule(
        initial_learning_rate=args.initial_learning_rate,
        first_decay_step=args.first_decay_step,
        middle_learning_rate=args.middle_learning_rate,
        second_decay_step=args.second_decay_step,
        final_learning_rate=args.final_learning_rate,
    )
    control = ContextSpec(
        name="T=64 control",
        batch_size=args.control_batch_size,
        block_size=args.control_block_size,
        checkpoint_path=args.control_checkpoint_path,
    )
    treatment = ContextSpec(
        name="T=128 treatment",
        batch_size=args.treatment_batch_size,
        block_size=args.treatment_block_size,
        checkpoint_path=args.treatment_checkpoint_path,
    )
    validate_context_protocol(control, treatment)
    common_config = {
        "n_embd": args.n_embd,
        "learning_rate": schedule.initial_learning_rate,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "seed": args.seed,
    }
    control_config = TrainingConfig(
        batch_size=control.batch_size,
        block_size=control.block_size,
        **common_config,
    )
    treatment_config = TrainingConfig(
        batch_size=treatment.batch_size,
        block_size=treatment.block_size,
        **common_config,
    )
    schedule.validate_target_step(args.max_iters)
    training_batch_seed = (
        args.training_batch_seed
        if args.training_batch_seed is not None
        else args.seed + 1
    )
    training_rng_seed = (
        args.training_rng_seed if args.training_rng_seed is not None else args.seed + 3
    )
    precise_eval_seed = (
        args.precise_eval_seed if args.precise_eval_seed is not None else args.seed + 7
    )
    for name, value in (
        ("training batch seed", training_batch_seed),
        ("training RNG seed", training_rng_seed),
        ("precise evaluation seed", precise_eval_seed),
    ):
        _validate_non_negative_integer(value, name=name)
    benchmark = None
    if args.benchmark_steps > 0:
        benchmark = BenchmarkConfig(
            num_warmup=args.benchmark_warmup,
            num_steps=args.benchmark_steps,
        )

    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)
    print("Stage 17: controlled context scaling T=64 -> 128")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Control: B={control.batch_size}, T={control.block_size}; "
        f"treatment: B={treatment.batch_size}, T={treatment.block_size}; "
        f"C={args.n_embd}, H={args.n_head}, "
        f"D={args.n_embd // args.n_head}, FF={4 * args.n_embd}, "
        f"L={args.n_layer}, dropout=0"
    )
    print(
        "Schedule by zero-based update: "
        f"[0, {schedule.first_decay_step:,}) "
        f"lr={schedule.initial_learning_rate:g}; "
        f"[{schedule.first_decay_step:,}, "
        f"{schedule.second_decay_step:,}) "
        f"lr={schedule.middle_learning_rate:g}; "
        f"[{schedule.second_decay_step:,}, {args.max_iters:,}) "
        f"lr={schedule.final_learning_rate:g}"
    )

    report = run_experiment(
        data,
        control_config,
        treatment_config,
        control,
        treatment,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        schedule,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        benchmark=benchmark,
    )

    if args.precise_eval_iters > 0:
        precise_report = run_precise_validation(
            data,
            control_config,
            treatment_config,
            control,
            treatment,
            args.n_head,
            args.n_layer,
            device,
            schedule,
            report.initialization,
            initialization_seed=report.initialization_seed,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            eval_iters=args.precise_eval_iters,
            seed=precise_eval_seed,
        )
        print_precise_validation(precise_report)


if __name__ == "__main__":
    main()
