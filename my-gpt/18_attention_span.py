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


# Stage 18 retains Stage 17's winning p=0 architecture and training schedule,
# but fixes both branches at B=16, T=128. The sole treatment is whether each
# query may attend to all earlier positions or only its latest 64 positions.
# The experiment identity and geometry policy are module settings so Stage 19
# can reuse the paired-training machinery without duplicating this large file.
_STAGE_17_PATH = Path(__file__).with_name("17_context_length.py")
_STAGE_17_MODULE_NAME = "stage_17_context_length_for_stage_18"
_STAGE_17_SPEC = importlib.util.spec_from_file_location(
    _STAGE_17_MODULE_NAME,
    _STAGE_17_PATH,
)
assert _STAGE_17_SPEC is not None and _STAGE_17_SPEC.loader is not None
_STAGE_17 = importlib.util.module_from_spec(_STAGE_17_SPEC)
sys.modules[_STAGE_17_MODULE_NAME] = _STAGE_17
_STAGE_17_SPEC.loader.exec_module(_STAGE_17)

GPTLanguageModel = _STAGE_17.GPTLanguageModel
LearningRateSchedule = _STAGE_17.LearningRateSchedule
BestValidationCheckpoint = _STAGE_17.BestValidationCheckpoint

aggregate_benchmark_stats = _STAGE_17.aggregate_benchmark_stats
checkpoint_sha256 = _STAGE_17.checkpoint_sha256
clear_accelerator_cache = _STAGE_17.clear_accelerator_cache
fingerprint_data = _STAGE_17.fingerprint_data
generate_from_final_model = _STAGE_17.generate_from_final_model
set_optimizer_learning_rate = _STAGE_17.set_optimizer_learning_rate
tensor_mapping_sha256 = _STAGE_17.tensor_mapping_sha256
tensor_sha256 = _STAGE_17.tensor_sha256

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_LOCAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_18_local_64_best_checkpoint.pt"
)
DEFAULT_FULL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_18_full_128_best_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = 16
DEFAULT_BLOCK_SIZE = 128
DEFAULT_LOCAL_CONTEXT = 64
DEFAULT_MAX_ITERS = _STAGE_17.DEFAULT_MAX_ITERS
DEFAULT_INITIAL_LEARNING_RATE = _STAGE_17.DEFAULT_INITIAL_LEARNING_RATE
DEFAULT_FIRST_DECAY_STEP = _STAGE_17.DEFAULT_FIRST_DECAY_STEP
DEFAULT_MIDDLE_LEARNING_RATE = _STAGE_17.DEFAULT_MIDDLE_LEARNING_RATE
DEFAULT_SECOND_DECAY_STEP = _STAGE_17.DEFAULT_SECOND_DECAY_STEP
DEFAULT_FINAL_LEARNING_RATE = _STAGE_17.DEFAULT_FINAL_LEARNING_RATE
DEFAULT_PRECISE_EVAL_ITERS = _STAGE_17.DEFAULT_PRECISE_EVAL_ITERS
DEFAULT_DROPOUT = 0.0
DROPOUT_PLACEMENT = _STAGE_17.DROPOUT_PLACEMENT

EXPERIMENT_STAGE = 18
EXPERIMENT_TITLE = "isolated attention span"
COMPARISON_VARIABLE = "maximum_attention_span"
PROTOCOL_GEOMETRY = "half_direct_attention_span"
PROTOCOL_NAME = "attention-span"
INITIAL_EQUIVALENCE_LABEL = "first-half"
SUMMARY_SPAN_LABEL = "attention span"
PRINT_RECEPTIVE_FIELD_DIAGNOSTIC = False


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """One attention-span branch; None means unrestricted causal context."""

    name: str
    max_context: int | None
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch name must not be empty")
        if self.max_context is not None and (
            isinstance(self.max_context, bool)
            or not isinstance(self.max_context, int)
            or self.max_context <= 0
        ):
            raise ValueError(
                "max_context must be a positive integer or None, got "
                f"{self.max_context!r}"
            )

    @property
    def context_label(self) -> str:
        if self.max_context is None:
            return "full"
        return str(self.max_context)


@dataclass(frozen=True, slots=True)
class PositionLosses:
    overall: float
    first_half: float
    second_half: float
    bands: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PositionLossEstimate:
    train: PositionLosses
    val: PositionLosses


@dataclass(frozen=True, slots=True)
class AttentionEvaluationRecord:
    step: int
    losses: PositionLossEstimate


@dataclass(frozen=True, slots=True)
class AttentionTrainingResult:
    initial: PositionLossEstimate
    final: PositionLossEstimate
    history: tuple[AttentionEvaluationRecord, ...]
    benchmark: BenchmarkStats | None = None


@dataclass(frozen=True, slots=True)
class PairedInitialStates:
    local: dict[str, torch.Tensor]
    full: dict[str, torch.Tensor]
    parameter_state_sha256: str
    local_state_sha256: str
    full_state_sha256: str
    local_mask_sha256: str
    full_mask_sha256: str
    parameter_count: int
    differing_state_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BranchReport:
    spec: AttentionSpec
    parameter_count: int
    initial_state_sha256: str
    parameter_state_sha256: str
    training: AttentionTrainingResult
    best_val_loss: float
    best_step: int
    best_validation_losses: PositionLosses
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
    bands: tuple[tuple[int, int], ...]
    local: BranchReport
    full: BranchReport
    initial_first_half_max_difference: float
    final_val_delta: float
    best_val_delta: float

    @property
    def initial_shared_prefix_length(self) -> int:
        """Return the query-prefix length whose branch masks are identical."""

        assert self.local.spec.max_context is not None
        return self.local.spec.max_context

    @property
    def initial_shared_prefix_max_difference(self) -> float:
        """Return the initialization audit under its general Stage 19 name."""

        return self.initial_first_half_max_difference

    @property
    def final_winner(self) -> str:
        if self.final_val_delta < 0:
            return self.full.spec.name
        if self.final_val_delta > 0:
            return self.local.spec.name
        return "tie"

    @property
    def best_winner(self) -> str:
        if self.best_val_delta < 0:
            return self.full.spec.name
        if self.best_val_delta > 0:
            return self.local.spec.name
        return "tie"


@dataclass(frozen=True, slots=True)
class PositionStatistics:
    mean: PositionLosses
    standard_error: PositionLosses


@dataclass(frozen=True, slots=True)
class PreciseBranchResult:
    name: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_step: int
    fixed_panel_loss: float
    statistics: PositionStatistics
    batch_losses: tuple[PositionLosses, ...]


@dataclass(frozen=True, slots=True)
class PreciseDelta:
    candidate: str
    baseline: str
    mean_delta: PositionLosses
    standard_error: PositionLosses
    confidence_low: PositionLosses
    confidence_high: PositionLosses
    critical_value: float


@dataclass(frozen=True, slots=True)
class PreciseValidationReport:
    eval_iters: int
    seed: int
    bands: tuple[tuple[int, int], ...]
    results: tuple[PreciseBranchResult, ...]
    delta: PreciseDelta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Stage {EXPERIMENT_STAGE}: compare local-{DEFAULT_LOCAL_CONTEXT} "
            f"with full-{DEFAULT_BLOCK_SIZE} causal attention while keeping "
            "batch geometry, parameters, initialization, and batches fixed."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--local-context",
        "--local-window",
        dest="local_context",
        type=int,
        default=DEFAULT_LOCAL_CONTEXT,
        help="Per-layer causal attention window for the local branch.",
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
        help="Defaults to seed + 1 and is recreated for each branch.",
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
        help=(
            "Fresh paired validation batches after training; 0 skips and "
            "positive values must be at least 2."
        ),
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
        "--local-checkpoint-path",
        type=Path,
        default=DEFAULT_LOCAL_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--full-checkpoint-path",
        type=Path,
        default=DEFAULT_FULL_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _validate_non_negative_integer(value: int, *, name: str) -> int:
    return _STAGE_17._validate_non_negative_integer(value, name=name)


def make_position_bands(block_size: int) -> tuple[tuple[int, int], ...]:
    """Return the four equal position bands used by the experiment report."""

    if block_size <= 0 or block_size % 4 != 0:
        raise ValueError(
            f"block_size must be positive and divisible by four, got {block_size}"
        )
    width = block_size // 4
    return tuple((start, start + width) for start in range(0, block_size, width))


def build_attention_mask(
    block_size: int,
    max_context: int | None,
) -> torch.Tensor:
    """Build a causal mask, optionally limited to the latest N positions."""

    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if max_context is not None:
        if isinstance(max_context, bool) or not isinstance(max_context, int):
            raise ValueError(
                f"max_context must be a positive integer or None, got {max_context!r}"
            )
        if not 0 < max_context <= block_size:
            raise ValueError(
                f"max_context must be in [1, {block_size}], got {max_context}"
            )

    positions = torch.arange(block_size)
    query_pos = positions[:, None]
    key_pos = positions[None, :]
    mask = key_pos <= query_pos
    if max_context is not None:
        mask &= key_pos >= query_pos - max_context + 1
    return mask


def theoretical_receptive_field_width(
    n_layer: int,
    direct_attention_span: int,
) -> int:
    """Return the uncapped receptive-field width after stacked local layers."""

    for value, name in (
        (n_layer, "n_layer"),
        (direct_attention_span, "direct_attention_span"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return 1 + n_layer * (direct_attention_span - 1)


def receptive_field_widths_by_layer(
    n_layer: int,
    direct_attention_span: int,
) -> tuple[int, ...]:
    """Return the uncapped local receptive-field width after each block."""

    theoretical_receptive_field_width(n_layer, direct_attention_span)
    return tuple(
        theoretical_receptive_field_width(layer, direct_attention_span)
        for layer in range(1, n_layer + 1)
    )


def print_receptive_field_diagnostic(
    n_layer: int,
    direct_attention_span: int,
) -> None:
    """Print the local branch's theoretical receptive field by block."""

    for layer, receptive_field in enumerate(
        receptive_field_widths_by_layer(n_layer, direct_attention_span),
        start=1,
    ):
        print(f"after block {layer}: max receptive field = {receptive_field}")


def local_span_for_half_receptive_field(
    block_size: int,
    n_layer: int,
) -> int:
    """Return the smallest window whose stacked RF reaches half the sequence."""

    for value, name in ((block_size, "block_size"), (n_layer, "n_layer")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    target_width = math.ceil(block_size / 2)
    return 1 + math.ceil((target_width - 1) / n_layer)


def attention_mask_kind(spec: AttentionSpec) -> str:
    if spec.max_context is None:
        return "full_causal"
    return "causal_sliding_window"


def effective_receptive_field_width(
    block_size: int,
    n_layer: int,
    spec: AttentionSpec,
) -> int:
    for value, name in ((block_size, "block_size"), (n_layer, "n_layer")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if spec.max_context is None:
        return block_size
    return min(
        block_size,
        theoretical_receptive_field_width(n_layer, spec.max_context),
    )


def _resolved_output_paths(spec: AttentionSpec) -> tuple[Path, Path]:
    path = spec.checkpoint_path.resolve()
    temporary = spec.checkpoint_path.with_name(
        f".{spec.checkpoint_path.name}.tmp"
    ).resolve()
    return path, temporary


def validate_attention_protocol(
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    *,
    n_layer: int | None = None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    if local.name == full.name:
        raise ValueError("local and full branch names must differ")
    if local.max_context is None:
        raise ValueError("the local branch must set max_context")
    if full.max_context is not None:
        raise ValueError("the full branch must use max_context=None")
    if local.max_context > config.block_size:
        raise ValueError("local max_context must not exceed block_size")
    if PROTOCOL_GEOMETRY == "half_direct_attention_span":
        if config.block_size != 2 * local.max_context:
            raise ValueError(
                f"Stage {EXPERIMENT_STAGE} requires block_size to equal "
                "twice local max_context"
            )
    elif PROTOCOL_GEOMETRY == "half_effective_receptive_field":
        if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer <= 0:
            raise ValueError(
                "n_layer must be a positive integer for the effective "
                f"receptive-field protocol, got {n_layer!r}"
            )
        expected_span = local_span_for_half_receptive_field(
            config.block_size,
            n_layer,
        )
        if local.max_context != expected_span:
            raise ValueError(
                f"Stage {EXPERIMENT_STAGE} requires local max_context="
                f"{expected_span} for block_size={config.block_size} and "
                f"n_layer={n_layer}, got {local.max_context}"
            )
        local_receptive_field = theoretical_receptive_field_width(
            n_layer,
            local.max_context,
        )
        if local_receptive_field >= config.block_size:
            raise ValueError(
                "the local receptive field must remain smaller than "
                "block_size for this protocol"
            )
    else:
        raise RuntimeError(f"unknown protocol geometry {PROTOCOL_GEOMETRY!r}")

    labelled_paths = {
        "local checkpoint": _resolved_output_paths(local)[0],
        "local temporary checkpoint": _resolved_output_paths(local)[1],
        "full checkpoint": _resolved_output_paths(full)[0],
        "full temporary checkpoint": _resolved_output_paths(full)[1],
    }
    labels = tuple(labelled_paths)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            first_path = labelled_paths[first_label]
            second_path = labelled_paths[second_label]
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

    return config.batch_size * config.block_size, make_position_bands(config.block_size)


def _apply_attention_mask(
    model: GPTLanguageModel,
    max_context: int | None,
) -> None:
    mask = build_attention_mask(model.block_size, max_context)
    for block in model.blocks:
        for head in block.sa.heads:
            head.tril.copy_(mask.to(device=head.tril.device, dtype=head.tril.dtype))


def _build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    spec: AttentionSpec,
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
    _apply_attention_mask(model, spec.max_context)
    if device is not None:
        model = model.to(device)
    return model


def _clone_state(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _clone_parameters(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _mask_state_names(state: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    names = tuple(sorted(name for name in state if name.endswith(".tril")))
    if not names:
        raise RuntimeError("model state contains no attention-mask buffers")
    return names


def create_paired_initial_states(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    local: AttentionSpec,
    full: AttentionSpec,
) -> PairedInitialStates:
    """Create states whose learned tensors are identical and masks differ."""

    validate_attention_protocol(config, local, full, n_layer=n_layer)
    seed_everything(config.seed)
    full_model = _build_model(data, config, n_head, n_layer, full)
    full_state = _clone_state(full_model)
    parameter_state = _clone_parameters(full_model)
    parameter_count = sum(
        parameter.numel()
        for parameter in full_model.parameters()
        if parameter.requires_grad
    )

    local_state = {name: tensor.clone() for name, tensor in full_state.items()}
    mask_names = _mask_state_names(full_state)
    local_mask = build_attention_mask(config.block_size, local.max_context)
    full_mask = build_attention_mask(config.block_size, full.max_context)
    for name in mask_names:
        local_state[name] = local_mask.to(dtype=local_state[name].dtype)
        full_state[name] = full_mask.to(dtype=full_state[name].dtype)

    differing_names = tuple(
        sorted(
            name
            for name in full_state
            if not torch.equal(local_state[name], full_state[name])
        )
    )
    if differing_names != mask_names:
        raise RuntimeError(
            "initial states must differ only in attention-mask buffers; got "
            f"{differing_names}"
        )

    local_model = _build_model(data, config, n_head, n_layer, local)
    local_model.load_state_dict(local_state, strict=True)
    full_model.load_state_dict(full_state, strict=True)
    local_parameters = _clone_parameters(local_model)
    if local_parameters.keys() != parameter_state.keys() or any(
        not torch.equal(local_parameters[name], parameter_state[name])
        for name in parameter_state
    ):
        raise RuntimeError("branches do not have identical learned initialization")

    parameter_hash = tensor_mapping_sha256(parameter_state)
    result = PairedInitialStates(
        local=local_state,
        full=full_state,
        parameter_state_sha256=parameter_hash,
        local_state_sha256=tensor_mapping_sha256(local_state),
        full_state_sha256=tensor_mapping_sha256(full_state),
        local_mask_sha256=tensor_sha256(local_mask),
        full_mask_sha256=tensor_sha256(full_mask),
        parameter_count=parameter_count,
        differing_state_names=differing_names,
    )
    del local_model
    del full_model
    return result


def verify_model_mask(
    model: GPTLanguageModel,
    spec: AttentionSpec,
) -> None:
    expected_bool = build_attention_mask(model.block_size, spec.max_context)
    observed_count = 0
    for block in model.blocks:
        for head in block.sa.heads:
            observed_count += 1
            observed = head.tril.detach().cpu()
            expected = expected_bool.to(dtype=observed.dtype)
            if not torch.equal(observed, expected):
                raise RuntimeError(f"{spec.name} contains an incorrect mask buffer")
    if observed_count == 0:
        raise RuntimeError("model contains no attention heads")


@torch.no_grad()
def verify_initial_shared_prefix_equivalence(
    local_model: GPTLanguageModel,
    full_model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    seed: int,
    prefix_length: int,
) -> float:
    if (
        isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not 0 < prefix_length <= config.block_size
    ):
        raise ValueError(
            "prefix_length must be an integer in "
            f"[1, {config.block_size}], got {prefix_length!r}"
        )
    generator = torch.Generator().manual_seed(seed)
    inputs, _ = data.get_batch(
        "val",
        batch_size=config.batch_size,
        block_size=config.block_size,
        device=device,
        generator=generator,
    )
    local_was_training = local_model.training
    full_was_training = full_model.training
    local_model.eval()
    full_model.eval()
    try:
        local_logits, _ = local_model(inputs)
        full_logits, _ = full_model(inputs)
        difference = (
            (local_logits[:, :prefix_length] - full_logits[:, :prefix_length])
            .abs()
            .max()
            .item()
        )
    finally:
        local_model.train(local_was_training)
        full_model.train(full_was_training)
    if difference > 1e-6:
        raise RuntimeError(
            "initial shared-prefix logits differ despite identical parameters "
            f"and masks there: max absolute difference {difference}"
        )
    return difference


def verify_initial_first_half_equivalence(
    local_model: GPTLanguageModel,
    full_model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    seed: int,
) -> float:
    """Retain the Stage 18 audit API for callers and saved documentation."""

    return verify_initial_shared_prefix_equivalence(
        local_model,
        full_model,
        data,
        config,
        device,
        seed=seed,
        prefix_length=config.block_size // 2,
    )


def position_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bands: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, ...]:
    """Return overall, half, and position-band mean cross entropies."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (B,T,V), got {logits.shape}")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match the first two logits dimensions")
    _, sequence_length, vocab_size = logits.shape
    if sequence_length % 2 != 0:
        raise ValueError("sequence length must be even")
    if not bands:
        raise ValueError("at least one position band is required")
    expected_start = 0
    for start, end in bands:
        if start != expected_start or not start < end <= sequence_length:
            raise ValueError("bands must exactly and contiguously cover the sequence")
        expected_start = end
    if expected_start != sequence_length:
        raise ValueError("bands must exactly and contiguously cover the sequence")

    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    midpoint = sequence_length // 2
    return (
        token_losses.mean(),
        token_losses[:, :midpoint].mean(),
        token_losses[:, midpoint:].mean(),
        *(token_losses[:, start:end].mean() for start, end in bands),
    )


def _as_position_losses(losses: Sequence[torch.Tensor]) -> PositionLosses:
    if len(losses) < 4:
        raise ValueError("overall, two halves, and at least one band are required")
    values = tuple(loss.detach().item() for loss in losses)
    return PositionLosses(
        overall=values[0],
        first_half=values[1],
        second_half=values[2],
        bands=values[3:],
    )


def _mean_position_losses(values: Sequence[PositionLosses]) -> PositionLosses:
    if not values:
        raise ValueError("at least one position-loss value is required")
    band_count = len(values[0].bands)
    if any(len(value.bands) != band_count for value in values):
        raise ValueError("all position losses must have the same bands")
    count = len(values)
    return PositionLosses(
        overall=sum(value.overall for value in values) / count,
        first_half=sum(value.first_half for value in values) / count,
        second_half=sum(value.second_half for value in values) / count,
        bands=tuple(
            sum(value.bands[index] for value in values) / count
            for index in range(band_count)
        ),
    )


@torch.no_grad()
def estimate_position_loss(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    bands: Sequence[tuple[int, int]],
    *,
    split: str,
    eval_iters: int,
    generator: torch.Generator,
) -> PositionLosses:
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    was_training = model.training
    model.eval()
    values: list[PositionLosses] = []
    try:
        for _ in range(eval_iters):
            inputs, targets = data.get_batch(
                split,
                batch_size=config.batch_size,
                block_size=config.block_size,
                device=device,
                generator=generator,
            )
            logits, _ = model(inputs)
            values.append(
                _as_position_losses(position_cross_entropy(logits, targets, bands))
            )
    finally:
        model.train(was_training)
    return _mean_position_losses(values)


def evaluate_on_fixed_batches(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    bands: Sequence[tuple[int, int]],
) -> PositionLossEstimate:
    # Recreate one generator for every evaluation and branch. This fixes the
    # complete train/validation panel and makes branch comparisons paired.
    generator = torch.Generator().manual_seed(config.seed + 2)
    train = estimate_position_loss(
        model,
        data,
        config,
        device,
        bands,
        split="train",
        eval_iters=config.eval_iters,
        generator=generator,
    )
    val = estimate_position_loss(
        model,
        data,
        config,
        device,
        bands,
        split="val",
        eval_iters=config.eval_iters,
        generator=generator,
    )
    return PositionLossEstimate(train=train, val=val)


@dataclass(slots=True)
class AttentionValidationCheckpoint(BestValidationCheckpoint):
    """Persist a best checkpoint and its experimental isolation contract."""

    branch_spec: AttentionSpec | None = None
    local_spec: AttentionSpec | None = None
    full_spec: AttentionSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initialization_seed: int = 0
    branch_initial_state_sha256: str = ""
    parameter_state_sha256: str = ""
    local_mask_sha256: str = ""
    full_mask_sha256: str = ""
    training_batch_seed: int = 0
    training_rng_seed: int = 0
    attention_shape: tuple[int, ...] = ()
    bands: tuple[tuple[int, int], ...] = ()
    best_validation_losses: PositionLosses | None = None

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(AttentionValidationCheckpoint, self)._payload(step)
        if (
            self.branch_spec is None
            or self.local_spec is None
            or self.full_spec is None
            or self.schedule is None
            or self.best_validation_losses is None
            or not self.bands
        ):
            raise RuntimeError(
                f"incomplete Stage {EXPERIMENT_STAGE} checkpoint metadata"
            )

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
                "attention_mask": attention_mask_kind(self.branch_spec),
                "attention_mask_scope": "per_layer",
                "max_context": self.branch_spec.max_context,
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
                "targets_per_update": (self.config.batch_size * self.config.block_size),
                "total_target_characters": (
                    self.config.batch_size * self.config.block_size * self.target_step
                ),
            }
        )
        payload["initialization"] = {
            "kind": "identical_learned_attention_initialization",
            "seed": self.initialization_seed,
            "branch_state_sha256": self.branch_initial_state_sha256,
            "parameter_state_sha256": self.parameter_state_sha256,
            "identical_learned_tensors": True,
            "state_differs_only_in_attention_masks": True,
            "local_mask_sha256": self.local_mask_sha256,
            "full_mask_sha256": self.full_mask_sha256,
        }
        payload["experiment"] = {
            "stage": EXPERIMENT_STAGE,
            "branch": self.branch_spec.name,
            "from_scratch": True,
            "comparison_variable": COMPARISON_VARIABLE,
            "branch_max_context": self.branch_spec.max_context,
            "branch_direct_attention_span": (
                self.branch_spec.max_context or self.config.block_size
            ),
            "uncapped_local_receptive_field_width": (
                theoretical_receptive_field_width(
                    self.n_layer,
                    self.branch_spec.max_context,
                )
                if self.branch_spec.max_context is not None
                else None
            ),
            "effective_receptive_field_width": effective_receptive_field_width(
                self.config.block_size,
                self.n_layer,
                self.branch_spec,
            ),
            "local_max_context": self.local_spec.max_context,
            "full_max_context": self.full_spec.max_context,
            "batch_size": self.config.batch_size,
            "block_size": self.config.block_size,
            "identical_parameter_count": True,
            "identical_learned_initialization": True,
            "identical_training_batches": True,
            "identical_position_embeddings": True,
            "identical_attention_shapes": True,
            "targets_per_update": self.config.batch_size * self.config.block_size,
            "total_target_characters": (
                self.config.batch_size * self.config.block_size * self.target_step
            ),
            "initialization_seed": self.initialization_seed,
            "training_batch_seed": self.training_batch_seed,
            "training_rng_seed": self.training_rng_seed,
            "residual_dropout": DEFAULT_DROPOUT,
            "learning_rate_schedule": schedule_metadata,
            "attention_shape": self.attention_shape,
            "position_bands": self.bands,
        }
        losses = self.best_validation_losses
        payload["best_validation_losses"] = {
            "overall": losses.overall,
            "first_half": losses.first_half,
            "second_half": losses.second_half,
            "bands": [
                {"start": start, "end": end, "loss": loss}
                for (start, end), loss in zip(self.bands, losses.bands, strict=True)
            ],
        }
        return payload

    def consider(self, record: AttentionEvaluationRecord) -> bool:
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
    config: TrainingConfig,
    spec: AttentionSpec,
    n_head: int,
    device: torch.device,
) -> tuple[int, ...]:
    verify_model_mask(model, spec)
    was_training = model.training
    model.eval()
    try:
        inputs = torch.zeros(
            (config.batch_size, config.block_size),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            weights = model.get_attention_weights(inputs, block_index=0)
        shape = tuple(weights.shape)
        expected = (
            config.batch_size,
            n_head,
            config.block_size,
            config.block_size,
        )
        if shape != expected:
            raise RuntimeError(
                f"{spec.name} attention shape must be {expected}, got {shape}"
            )
        allowed = build_attention_mask(config.block_size, spec.max_context).to(device)
        forbidden = weights.masked_select(~allowed[None, None])
        if forbidden.numel() and torch.count_nonzero(forbidden).item() != 0:
            raise RuntimeError(f"{spec.name} assigns weight to masked positions")
        return shape
    finally:
        model.train(was_training)


def train_with_schedule(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    schedule: LearningRateSchedule,
    bands: Sequence[tuple[int, int]],
    *,
    on_evaluation: Callable[[AttentionEvaluationRecord], None],
    on_update: Callable[[int, float, torch.Tensor, torch.Tensor], None] | None = None,
) -> AttentionTrainingResult:
    schedule.validate_target_step(config.max_iters)
    history: list[AttentionEvaluationRecord] = []

    def record(step: int) -> AttentionEvaluationRecord:
        evaluation = AttentionEvaluationRecord(
            step=step,
            losses=evaluate_on_fixed_batches(
                model,
                data,
                config,
                device,
                bands,
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
        inputs, targets = data.get_batch(
            "train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=training_generator,
        )
        if on_update is not None:
            on_update(update_index, learning_rate, inputs, targets)
        logits, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        del logits
        del loss
        del inputs
        del targets

        completed_step = update_index + 1
        should_evaluate = (
            completed_step % config.eval_interval == 0
            or completed_step == schedule.first_decay_step
            or completed_step == schedule.second_decay_step
            or completed_step == config.max_iters
        )
        if should_evaluate:
            record(completed_step)

    return AttentionTrainingResult(
        initial=initial_record.losses,
        final=history[-1].losses,
        history=tuple(history),
    )


def make_evaluation_callback(
    best: AttentionValidationCheckpoint,
    schedule: LearningRateSchedule,
) -> Callable[[AttentionEvaluationRecord], None]:
    def record_evaluation(record: AttentionEvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        learning_rate = schedule.for_update(max(0, record.step - 1))
        train = record.losses.train
        val = record.losses.val
        band_text = " | ".join(
            f"val {start:03d}-{end - 1:03d} {loss:.4f}"
            for (start, end), loss in zip(best.bands, val.bands, strict=True)
        )
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr {learning_rate:g} | "
            f"train {train.overall:.4f} | val {val.overall:.4f} | "
            f"val first {val.first_half:.4f} | "
            f"val second {val.second_half:.4f} | {band_text} | "
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


def benchmark_attention_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> BenchmarkStats:
    """Benchmark a disposable branch on the exact experiment update path."""

    clear_accelerator_cache(device)
    model = _build_model(data, config, n_head, n_layer, spec, device)
    model.load_state_dict(initial_state, strict=True)
    verify_model_mask(model, spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
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
        inputs, targets = data.get_batch(
            "train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=generator,
        )
        logits, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        del logits
        del loss
        del inputs
        del targets

    sync_device(device)
    assert started_at is not None
    elapsed = time.perf_counter() - started_at
    peak_allocated, peak_reserved = get_peak_memory_stats(device)
    iterations_per_sec = benchmark.num_steps / elapsed
    stats = BenchmarkStats(
        seconds=elapsed,
        iterations_per_sec=iterations_per_sec,
        tokens_per_sec=(iterations_per_sec * config.batch_size * config.block_size),
        peak_allocated_mb=peak_allocated / 1024**2,
        peak_reserved_mb=peak_reserved / 1024**2,
    )
    del optimizer
    del model
    clear_accelerator_cache(device)
    return stats


def run_counterbalanced_benchmarks(
    data: CharacterData,
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    initialization: PairedInitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> tuple[BenchmarkStats, BenchmarkStats]:
    local_runs: list[BenchmarkStats] = []
    full_runs: list[BenchmarkStats] = []
    ordered_runs = (
        (local, initialization.local, local_runs),
        (full, initialization.full, full_runs),
        (full, initialization.full, full_runs),
        (local, initialization.local, local_runs),
    )
    for spec, initial_state, destination in ordered_runs:
        destination.append(
            benchmark_attention_branch(
                data,
                config,
                spec,
                n_head,
                n_layer,
                device,
                initial_state,
                training_batch_seed=training_batch_seed,
                training_rng_seed=training_rng_seed,
                benchmark=benchmark,
            )
        )
    return (
        aggregate_benchmark_stats(local_runs, steps_per_run=benchmark.num_steps),
        aggregate_benchmark_stats(full_runs, steps_per_run=benchmark.num_steps),
    )


def _expected_attention_shape(
    config: TrainingConfig,
    n_head: int,
) -> tuple[int, ...]:
    return (
        config.batch_size,
        n_head,
        config.block_size,
        config.block_size,
    )


def _validate_config(
    config: TrainingConfig,
    schedule: LearningRateSchedule,
) -> None:
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError("config learning rate must equal schedule initial LR")
    schedule.validate_target_step(config.max_iters)


def run_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: AttentionSpec,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    parameter_state_sha256: str,
    local_mask_sha256: str,
    full_mask_sha256: str,
    *,
    initialization_seed: int,
    training_batch_seed: int,
    training_rng_seed: int,
) -> BranchReport:
    _validate_config(config, schedule)
    clear_accelerator_cache(device)
    model = _build_model(data, config, n_head, n_layer, spec, device)
    model.load_state_dict(initial_state, strict=True)
    verify_model_mask(model, spec)
    if tensor_mapping_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError(f"{spec.name} did not load its exact initial state")
    if tensor_mapping_sha256(_clone_parameters(model)) != parameter_state_sha256:
        raise RuntimeError(f"{spec.name} learned initialization is not shared")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.for_update(0),
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    expected_shape = _expected_attention_shape(config, n_head)
    best = AttentionValidationCheckpoint(
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
        local_spec=local,
        full_spec=full,
        schedule=schedule,
        target_step=config.max_iters,
        initialization_seed=initialization_seed,
        branch_initial_state_sha256=initial_state_sha256,
        parameter_state_sha256=parameter_state_sha256,
        local_mask_sha256=local_mask_sha256,
        full_mask_sha256=full_mask_sha256,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        attention_shape=expected_shape,
        bands=bands,
    )
    record_evaluation = make_evaluation_callback(best, schedule)

    print(
        f"\n{spec.name} branch (B={config.batch_size}, T={config.block_size}, "
        f"max_context={spec.context_label}, dropout=0)"
    )
    print(f"  branch state:         {initial_state_sha256}")
    print(f"  shared parameters:    {parameter_state_sha256}")
    print(f"  training-batch seed:  {training_batch_seed}")
    print(f"  training RNG seed:    {training_rng_seed}")

    seed_everything(training_rng_seed)
    result = train_with_schedule(
        model,
        optimizer,
        data,
        config,
        device,
        training_generator,
        schedule,
        bands,
        on_evaluation=record_evaluation,
    )
    attention_shape = inspect_attention_shape(
        model,
        config,
        spec,
        n_head,
        device,
    )
    assert best.best_step is not None
    assert best.best_validation_losses is not None
    final_generator_hash = tensor_sha256(training_generator.get_state())
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
        parameter_state_sha256=parameter_state_sha256,
        training=result,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        best_validation_losses=best.best_validation_losses,
        final_batch_generator_sha256=final_generator_hash,
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
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
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
    _validate_config(config, schedule)
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if n_head <= 0 or config.n_embd % n_head != 0:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({n_head})"
        )
    if n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {n_layer}")
    targets_per_update, bands = validate_attention_protocol(
        config,
        local,
        full,
        n_layer=n_layer,
    )
    if benchmark is not None:
        required = benchmark.num_warmup + benchmark.num_steps
        if required > config.max_iters:
            raise ValueError(
                f"benchmark requires {required} steps, but max_iters is "
                f"{config.max_iters}"
            )
    for split, source in (("train", data.train_data), ("val", data.val_data)):
        if len(source) <= config.block_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but block_size "
                f"is {config.block_size}"
            )
    _validate_non_negative_integer(training_batch_seed, name="training batch seed")
    _validate_non_negative_integer(training_rng_seed, name="training RNG seed")

    initialization = create_paired_initial_states(
        data,
        config,
        n_head,
        n_layer,
        local,
        full,
    )
    print(f"\nControlled from-initialization {PROTOCOL_NAME} protocol")
    print(
        "  Both branches train from step zero on identical "
        f"{config.block_size}-token batches."
    )
    print(
        "  Every learned tensor, including all "
        f"{config.block_size} position rows, is identical."
    )
    print("  Initial states differ only in the attention-mask buffers.")
    print(
        f"  Targets per update: {targets_per_update:,}; total per branch: "
        f"{targets_per_update * config.max_iters:,}."
    )
    print(f"  shared parameter SHA-256:   {initialization.parameter_state_sha256}")
    print(f"  local mask SHA-256:         {initialization.local_mask_sha256}")
    print(f"  full mask SHA-256:          {initialization.full_mask_sha256}")

    local_audit_model = _build_model(
        data, config, n_head, n_layer, local, torch.device("cpu")
    )
    full_audit_model = _build_model(
        data, config, n_head, n_layer, full, torch.device("cpu")
    )
    local_audit_model.load_state_dict(initialization.local, strict=True)
    full_audit_model.load_state_dict(initialization.full, strict=True)
    initial_shared_prefix_length = local.max_context
    initial_shared_prefix_max_difference = verify_initial_shared_prefix_equivalence(
        local_audit_model,
        full_audit_model,
        data,
        config,
        torch.device("cpu"),
        seed=config.seed + 6,
        prefix_length=initial_shared_prefix_length,
    )
    del local_audit_model
    del full_audit_model
    if INITIAL_EQUIVALENCE_LABEL == "first-half":
        print(
            "  Initial first-half max logit difference: "
            f"{initial_shared_prefix_max_difference:.3e}"
        )
    else:
        print(
            f"  Initial {INITIAL_EQUIVALENCE_LABEL} max logit difference "
            f"(positions 0-{initial_shared_prefix_length - 1}): "
            f"{initial_shared_prefix_max_difference:.3e}"
        )

    local_benchmark: BenchmarkStats | None = None
    full_benchmark: BenchmarkStats | None = None
    if benchmark is not None:
        print("\nCounterbalanced disposable training benchmark")
        print(f"  Order: {local.name}, {full.name}, {full.name}, {local.name}")
        print(
            f"  Each run: {benchmark.num_warmup} warmup + "
            f"{benchmark.num_steps} timed updates"
        )
        local_benchmark, full_benchmark = run_counterbalanced_benchmarks(
            data,
            config,
            local,
            full,
            n_head,
            n_layer,
            device,
            initialization,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            benchmark=benchmark,
        )
        _print_benchmark(local.name, local_benchmark)
        _print_benchmark(full.name, full_benchmark)

    local_report = run_branch(
        data,
        config,
        local,
        local,
        full,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        bands,
        initialization.local,
        initialization.local_state_sha256,
        initialization.parameter_state_sha256,
        initialization.local_mask_sha256,
        initialization.full_mask_sha256,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )
    full_report = run_branch(
        data,
        config,
        full,
        local,
        full,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        bands,
        initialization.full,
        initialization.full_state_sha256,
        initialization.parameter_state_sha256,
        initialization.local_mask_sha256,
        initialization.full_mask_sha256,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )

    if local_benchmark is not None and full_benchmark is not None:
        local_report = replace(
            local_report,
            training=replace(local_report.training, benchmark=local_benchmark),
        )
        full_report = replace(
            full_report,
            training=replace(full_report.training, benchmark=full_benchmark),
        )
    if local_report.parameter_count != initialization.parameter_count:
        raise RuntimeError("local parameter count changed")
    if full_report.parameter_count != initialization.parameter_count:
        raise RuntimeError("full parameter count changed")
    if local_report.parameter_count != full_report.parameter_count:
        raise RuntimeError("branches do not have identical parameter counts")
    if (
        local_report.final_batch_generator_sha256
        != full_report.final_batch_generator_sha256
    ):
        raise RuntimeError("branches did not consume identical training batches")
    if tuple(record.step for record in local_report.training.history) != tuple(
        record.step for record in full_report.training.history
    ):
        raise RuntimeError("branches were not evaluated at matching steps")

    final_val_delta = (
        full_report.training.final.val.overall - local_report.training.final.val.overall
    )
    best_val_delta = full_report.best_val_loss - local_report.best_val_loss
    print(f"\nStage {EXPERIMENT_STAGE} experimental summary")
    print(f"  both branches B/T:            {config.batch_size}/{config.block_size}")
    print(
        f"  local/full {SUMMARY_SPAN_LABEL}:    {local.max_context}/{config.block_size}"
    )
    print(
        "  local uncapped receptive field:"
        f" {theoretical_receptive_field_width(n_layer, local.max_context):,} "
        "positions across layers"
    )
    print(f"  target characters/update:     {targets_per_update:,}")
    print(f"  parameter count (both):       {local_report.parameter_count:,}")
    print(f"  attention shape (both):       {local_report.attention_shape}")
    print(
        f"  attention elements/block:     {math.prod(local_report.attention_shape):,}"
    )
    print(
        f"  local final train / val:      "
        f"{local_report.training.final.train.overall:.4f} / "
        f"{local_report.training.final.val.overall:.4f}"
    )
    print(
        f"  full final train / val:       "
        f"{full_report.training.final.train.overall:.4f} / "
        f"{full_report.training.final.val.overall:.4f}"
    )
    print(
        f"  final val delta (full-local): {final_val_delta:+.4f} "
        "(negative favors full context)"
    )
    print(
        f"  local best val:               {local_report.best_val_loss:.4f} "
        f"@ {local_report.best_step}"
    )
    print(
        f"  full best val:                {full_report.best_val_loss:.4f} "
        f"@ {full_report.best_step}"
    )
    print(
        f"  best val delta (full-local):  {best_val_delta:+.4f} "
        "(negative favors full context)"
    )
    for index, (start, end) in enumerate(bands):
        delta = (
            full_report.training.final.val.bands[index]
            - local_report.training.final.val.bands[index]
        )
        print(f"  final val delta {start:03d}-{end - 1:03d}:      {delta:+.4f}")
    print(f"  local checkpoint:             {local.checkpoint_path}")
    print(f"  full checkpoint:              {full.checkpoint_path}")
    print(f"\n{local.name} final-step generated text")
    print(local_report.sample)
    print(f"\n{full.name} final-step generated text")
    print(full_report.sample)

    return ExperimentReport(
        initialization=initialization,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        targets_per_update=targets_per_update,
        bands=bands,
        local=local_report,
        full=full_report,
        initial_first_half_max_difference=initial_shared_prefix_max_difference,
        final_val_delta=final_val_delta,
        best_val_delta=best_val_delta,
    )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def expected_training_generator_state(
    data: CharacterData,
    config: TrainingConfig,
    *,
    training_batch_seed: int,
    completed_steps: int,
) -> torch.Tensor:
    _validate_non_negative_integer(completed_steps, name="completed steps")
    generator = torch.Generator().manual_seed(training_batch_seed)
    maximum = len(data.train_data) - config.block_size
    if maximum <= 0:
        raise ValueError("training data is too short for the configured block size")
    for _ in range(completed_steps):
        torch.randint(
            0,
            maximum,
            (config.batch_size,),
            generator=generator,
        )
    return generator.get_state().clone()


def validate_experiment_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    spec: AttentionSpec,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    *,
    initialization_seed: int,
    branch_initial_state_sha256: str,
    parameter_state_sha256: str,
    local_mask_sha256: str,
    full_mask_sha256: str,
    training_batch_seed: int,
    training_rng_seed: int,
) -> Mapping[str, object]:
    """Validate the scientific and resumability metadata before evaluation."""

    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="root",
    )
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("checkpoint version must be 1")
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("checkpoint kind must be 'best'")
    step = checkpoint.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= config.max_iters
    ):
        raise ValueError("checkpoint step is invalid")
    if checkpoint.get("best_step") != step:
        raise ValueError("checkpoint best_step must equal step")
    best_val_loss = checkpoint.get("best_val_loss")
    if (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise ValueError("checkpoint best_val_loss must be finite")
    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("checkpoint data fingerprint does not match")
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError(
            f"Stage {EXPERIMENT_STAGE} checkpoints must not restart the optimizer"
        )
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError(f"Stage {EXPERIMENT_STAGE} optimizer provenance must be known")

    architecture = _require_mapping(checkpoint.get("architecture"), name="architecture")
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "residual_dropout": DEFAULT_DROPOUT,
        "dropout_placement": DROPOUT_PLACEMENT,
        "attention_mask": attention_mask_kind(spec),
        "attention_mask_scope": "per_layer",
        "max_context": spec.max_context,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"checkpoint architecture.{key} must be {expected!r}, "
                f"got {architecture.get(key)!r}"
            )

    initialization = _require_mapping(
        checkpoint.get("initialization"), name="initialization"
    )
    expected_initialization = {
        "kind": "identical_learned_attention_initialization",
        "seed": initialization_seed,
        "branch_state_sha256": branch_initial_state_sha256,
        "parameter_state_sha256": parameter_state_sha256,
        "identical_learned_tensors": True,
        "state_differs_only_in_attention_masks": True,
        "local_mask_sha256": local_mask_sha256,
        "full_mask_sha256": full_mask_sha256,
    }
    for key, expected in expected_initialization.items():
        if initialization.get(key) != expected:
            raise ValueError(
                f"checkpoint initialization.{key} must be {expected!r}, "
                f"got {initialization.get(key)!r}"
            )

    schedule_metadata = schedule.as_metadata(max_iters=config.max_iters)
    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    expected_experiment = {
        "stage": EXPERIMENT_STAGE,
        "branch": spec.name,
        "from_scratch": True,
        "comparison_variable": COMPARISON_VARIABLE,
        "branch_max_context": spec.max_context,
        "branch_direct_attention_span": spec.max_context or config.block_size,
        "uncapped_local_receptive_field_width": (
            theoretical_receptive_field_width(n_layer, spec.max_context)
            if spec.max_context is not None
            else None
        ),
        "effective_receptive_field_width": effective_receptive_field_width(
            config.block_size,
            n_layer,
            spec,
        ),
        "local_max_context": local.max_context,
        "full_max_context": full.max_context,
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "identical_parameter_count": True,
        "identical_learned_initialization": True,
        "identical_training_batches": True,
        "identical_position_embeddings": True,
        "identical_attention_shapes": True,
        "targets_per_update": config.batch_size * config.block_size,
        "total_target_characters": (
            config.batch_size * config.block_size * config.max_iters
        ),
        "initialization_seed": initialization_seed,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "residual_dropout": DEFAULT_DROPOUT,
        "learning_rate_schedule": schedule_metadata,
        "attention_shape": _expected_attention_shape(config, n_head),
        "position_bands": bands,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"checkpoint experiment.{key} must be {expected!r}, "
                f"got {experiment.get(key)!r}"
            )

    training_config = _require_mapping(
        checkpoint.get("training_config"), name="training_config"
    )
    expected_training = {
        "batch_size": config.batch_size,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": config.max_iters,
        "learning_rate_schedule": schedule_metadata,
        "residual_dropout": DEFAULT_DROPOUT,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
        "total_target_characters": (
            config.batch_size * config.block_size * config.max_iters
        ),
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

    saved_losses = _require_mapping(
        checkpoint.get("best_validation_losses"),
        name="best_validation_losses",
    )
    for key in ("overall", "first_half", "second_half"):
        value = saved_losses.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"checkpoint best_validation_losses.{key} must be finite")
    if float(saved_losses["overall"]) != float(best_val_loss):
        raise ValueError("overall position loss must equal best_val_loss")
    saved_bands = saved_losses.get("bands")
    if not isinstance(saved_bands, list) or len(saved_bands) != len(bands):
        raise ValueError("checkpoint position-band losses are incomplete")
    for saved, (start, end) in zip(saved_bands, bands, strict=True):
        mapping = _require_mapping(saved, name="position band")
        if mapping.get("start") != start or mapping.get("end") != end:
            raise ValueError("checkpoint position-band boundaries are inconsistent")
        loss = mapping.get("loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ValueError("checkpoint position-band loss must be finite")

    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"), name="optimizer_state_dict"
    )
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("checkpoint optimizer must contain parameter groups")
    if not all(isinstance(group, Mapping) and "lr" in group for group in param_groups):
        raise ValueError("every optimizer parameter group must contain an LR")
    optimizer_rates = {
        float(group["lr"]) for group in param_groups if isinstance(group, Mapping)
    }
    if optimizer_rates != {expected_learning_rate}:
        raise ValueError("checkpoint optimizer LR is inconsistent")

    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("checkpoint training generator state must be a tensor")
    expected_generator = expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator):
        raise ValueError(
            "checkpoint training generator state does not match the batch "
            f"stream at step {step}"
        )

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    with torch.random.fork_rng(devices=[]):
        compatibility_model = _build_model(
            data,
            config,
            n_head,
            n_layer,
            spec,
        )
        compatibility_model.load_state_dict(model_state, strict=True)
        verify_model_mask(compatibility_model, spec)
        del compatibility_model
    return checkpoint


# Retain the public Stage 18 name for existing callers and artifacts.
validate_stage_18_checkpoint = validate_experiment_checkpoint


@torch.no_grad()
def estimate_validation_precise(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    bands: Sequence[tuple[int, int]],
    *,
    eval_iters: int,
    seed: int,
) -> tuple[PositionLosses, ...]:
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    _validate_non_negative_integer(seed, name="precise evaluation seed")
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    values: list[PositionLosses] = []
    try:
        for _ in range(eval_iters):
            inputs, targets = data.get_batch(
                "val",
                batch_size=config.batch_size,
                block_size=config.block_size,
                device=device,
                generator=generator,
            )
            logits, _ = model(inputs)
            values.append(
                _as_position_losses(position_cross_entropy(logits, targets, bands))
            )
    finally:
        model.train(was_training)
    return tuple(values)


def _mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("at least two values are required for an SE")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    mean = tensor.mean().item()
    standard_error = (tensor.std(unbiased=True) / math.sqrt(len(values))).item()
    return mean, standard_error


_STUDENT_T_975 = (
    float("nan"),
    12.706204736,
    4.302652730,
    3.182446305,
    2.776445105,
    2.570581836,
    2.446911851,
    2.364624252,
    2.306004135,
    2.262157163,
    2.228138852,
    2.200985160,
    2.178812830,
    2.160368656,
    2.144786688,
    2.131449546,
    2.119905299,
    2.109815578,
    2.100922040,
    2.093024054,
    2.085963447,
    2.079613845,
    2.073873068,
    2.068657610,
    2.063898562,
    2.059538553,
    2.055529439,
    2.051830516,
    2.048407142,
    2.045229642,
    2.042272456,
)


def _paired_t_95_critical_value(sample_count: int) -> float:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        raise ValueError(
            f"sample_count must be an integer of at least 2, got {sample_count!r}"
        )
    degrees_of_freedom = sample_count - 1
    if degrees_of_freedom < len(_STUDENT_T_975):
        return _STUDENT_T_975[degrees_of_freedom]

    # Cornish-Fisher expansion for the 0.975 Student-t quantile. At df > 30
    # this closely tracks the exact quantile and converges to the normal 1.96.
    z = 1.959963984540054
    inverse_df = 1.0 / degrees_of_freedom
    return (
        z
        + (z**3 + z) * inverse_df / 4
        + (5 * z**5 + 16 * z**3 + 3 * z) * inverse_df**2 / 96
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) * inverse_df**3 / 384
    )


def _position_statistics(values: Sequence[PositionLosses]) -> PositionStatistics:
    if not values:
        raise ValueError("at least one position-loss value is required")
    band_count = len(values[0].bands)
    if any(len(value.bands) != band_count for value in values):
        raise ValueError("all position losses must have the same bands")
    overall = _mean_and_standard_error([value.overall for value in values])
    first = _mean_and_standard_error([value.first_half for value in values])
    second = _mean_and_standard_error([value.second_half for value in values])
    band_stats = tuple(
        _mean_and_standard_error([value.bands[index] for value in values])
        for index in range(band_count)
    )
    return PositionStatistics(
        mean=PositionLosses(
            overall=overall[0],
            first_half=first[0],
            second_half=second[0],
            bands=tuple(stat[0] for stat in band_stats),
        ),
        standard_error=PositionLosses(
            overall=overall[1],
            first_half=first[1],
            second_half=second[1],
            bands=tuple(stat[1] for stat in band_stats),
        ),
    )


def _subtract_position_losses(
    candidate: PositionLosses,
    baseline: PositionLosses,
) -> PositionLosses:
    if len(candidate.bands) != len(baseline.bands):
        raise ValueError("position losses must have matching bands")
    return PositionLosses(
        overall=candidate.overall - baseline.overall,
        first_half=candidate.first_half - baseline.first_half,
        second_half=candidate.second_half - baseline.second_half,
        bands=tuple(
            candidate_value - baseline_value
            for candidate_value, baseline_value in zip(
                candidate.bands,
                baseline.bands,
                strict=True,
            )
        ),
    )


def _confidence_bound(
    mean: PositionLosses,
    standard_error: PositionLosses,
    multiplier: float,
) -> PositionLosses:
    return PositionLosses(
        overall=mean.overall + multiplier * standard_error.overall,
        first_half=mean.first_half + multiplier * standard_error.first_half,
        second_half=mean.second_half + multiplier * standard_error.second_half,
        bands=tuple(
            mean_value + multiplier * se_value
            for mean_value, se_value in zip(
                mean.bands,
                standard_error.bands,
                strict=True,
            )
        ),
    )


def _paired_delta(
    candidate: PreciseBranchResult,
    baseline: PreciseBranchResult,
) -> PreciseDelta:
    if len(candidate.batch_losses) != len(baseline.batch_losses):
        raise ValueError("paired validation results must have equal lengths")
    differences = tuple(
        _subtract_position_losses(candidate_value, baseline_value)
        for candidate_value, baseline_value in zip(
            candidate.batch_losses,
            baseline.batch_losses,
            strict=True,
        )
    )
    statistics = _position_statistics(differences)
    critical_value = _paired_t_95_critical_value(len(differences))
    return PreciseDelta(
        candidate=candidate.name,
        baseline=baseline.name,
        mean_delta=statistics.mean,
        standard_error=statistics.standard_error,
        confidence_low=_confidence_bound(
            statistics.mean, statistics.standard_error, -critical_value
        ),
        confidence_high=_confidence_bound(
            statistics.mean, statistics.standard_error, critical_value
        ),
        critical_value=critical_value,
    )


def run_precise_validation(
    data: CharacterData,
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    initialization: PairedInitialStates,
    bands: tuple[tuple[int, int], ...],
    *,
    initialization_seed: int,
    training_batch_seed: int,
    training_rng_seed: int,
    eval_iters: int,
    seed: int,
) -> PreciseValidationReport:
    if eval_iters < 2:
        raise ValueError(
            f"eval_iters must be at least 2 for paired inference, got {eval_iters}"
        )
    results: list[PreciseBranchResult] = []
    for spec, initial_state_hash in (
        (local, initialization.local_state_sha256),
        (full, initialization.full_state_sha256),
    ):
        before_hash = checkpoint_sha256(spec.checkpoint_path)
        checkpoint = validate_experiment_checkpoint(
            spec.checkpoint_path,
            data,
            config,
            spec,
            local,
            full,
            n_head,
            n_layer,
            schedule,
            bands,
            initialization_seed=initialization_seed,
            branch_initial_state_sha256=initial_state_hash,
            parameter_state_sha256=initialization.parameter_state_sha256,
            local_mask_sha256=initialization.local_mask_sha256,
            full_mask_sha256=initialization.full_mask_sha256,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
        )
        model = _build_model(data, config, n_head, n_layer, spec, device)
        model_state = _require_mapping(
            checkpoint.get("model_state_dict"), name="model_state_dict"
        )
        model.load_state_dict(model_state, strict=True)
        verify_model_mask(model, spec)
        batch_losses = estimate_validation_precise(
            model,
            data,
            config,
            device,
            bands,
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
                statistics=_position_statistics(batch_losses),
                batch_losses=batch_losses,
            )
        )
        del model
        clear_accelerator_cache(device)
    return PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        bands=bands,
        results=tuple(results),
        delta=_paired_delta(results[1], results[0]),
    )


def print_precise_validation(report: PreciseValidationReport) -> None:
    print(f"\nFresh paired validation of Stage {EXPERIMENT_STAGE} best checkpoints")
    print(
        f"  {report.eval_iters} paired validation batches per checkpoint, "
        f"seed={report.seed}"
    )
    for result in report.results:
        mean = result.statistics.mean
        se = result.statistics.standard_error
        print(
            f"  {result.name:18s} | step {result.checkpoint_step:5d} | "
            f"fixed {result.fixed_panel_loss:.4f} | overall "
            f"{mean.overall:.6f} +/- {se.overall:.6f} SE | first "
            f"{mean.first_half:.6f} +/- {se.first_half:.6f} | second "
            f"{mean.second_half:.6f} +/- {se.second_half:.6f}"
        )
        for (start, end), band_mean, band_se in zip(
            report.bands,
            mean.bands,
            se.bands,
            strict=True,
        ):
            print(
                f"    positions {start:03d}-{end - 1:03d}: "
                f"{band_mean:.6f} +/- {band_se:.6f} SE"
            )
        print(f"    checkpoint SHA-256: {result.checkpoint_sha256}")

    delta = report.delta
    print(
        "  Paired Student-t 95% intervals use critical value "
        f"{delta.critical_value:.6f}."
    )
    rows = [
        (
            "overall",
            delta.mean_delta.overall,
            delta.standard_error.overall,
            delta.confidence_low.overall,
            delta.confidence_high.overall,
        ),
        (
            "first half",
            delta.mean_delta.first_half,
            delta.standard_error.first_half,
            delta.confidence_low.first_half,
            delta.confidence_high.first_half,
        ),
        (
            "second half",
            delta.mean_delta.second_half,
            delta.standard_error.second_half,
            delta.confidence_low.second_half,
            delta.confidence_high.second_half,
        ),
    ]
    rows.extend(
        (
            f"positions {start:03d}-{end - 1:03d}",
            delta.mean_delta.bands[index],
            delta.standard_error.bands[index],
            delta.confidence_low.bands[index],
            delta.confidence_high.bands[index],
        )
        for index, (start, end) in enumerate(report.bands)
    )
    for label, mean, se, low, high in rows:
        print(
            f"  delta {delta.candidate} - {delta.baseline}, {label}: "
            f"{mean:+.6f} +/- {se:.6f} SE "
            f"(paired-t 95% CI [{low:+.6f}, {high:+.6f}]); "
            "negative favors full context"
        )


def main() -> None:
    args = parse_args()
    if args.sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if args.precise_eval_iters < 0 or args.precise_eval_iters == 1:
        raise ValueError("precise-eval-iters must be 0 or at least 2")
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
    config = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        learning_rate=schedule.initial_learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    local = AttentionSpec(
        name=f"local-{args.local_context} control",
        max_context=args.local_context,
        checkpoint_path=args.local_checkpoint_path,
    )
    full = AttentionSpec(
        name=f"full-{args.block_size} treatment",
        max_context=None,
        checkpoint_path=args.full_checkpoint_path,
    )
    _, bands = validate_attention_protocol(
        config,
        local,
        full,
        n_layer=args.n_layer,
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
    print(
        f"Stage {EXPERIMENT_STAGE}: {EXPERIMENT_TITLE} "
        f"local-{args.local_context} vs full-{args.block_size}"
    )
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Both: B={config.batch_size}, T={config.block_size}, "
        f"C={args.n_embd}, H={args.n_head}, "
        f"D={args.n_embd // args.n_head}, FF={4 * args.n_embd}, "
        f"L={args.n_layer}, dropout=0"
    )
    print(
        f"Masks: local max_context={local.max_context}; full causal context; "
        f"bands={bands}"
    )
    if PRINT_RECEPTIVE_FIELD_DIAGNOSTIC:
        print("Local theoretical receptive field by block:")
        print_receptive_field_diagnostic(args.n_layer, args.local_context)
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
        config,
        local,
        full,
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
            config,
            local,
            full,
            args.n_head,
            args.n_layer,
            device,
            schedule,
            report.initialization,
            report.bands,
            initialization_seed=report.initialization_seed,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            eval_iters=args.precise_eval_iters,
            seed=precise_eval_seed,
        )
        print_precise_validation(precise_report)


if __name__ == "__main__":
    main()
