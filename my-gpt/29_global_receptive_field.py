from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch

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


# Stage 29 reuses the audited training, statistics, and Stage 26 checkpoint
# primitives from Stage 27. The experiment itself is implemented here: its
# continuous local mask has a six-layer theoretical receptive field that
# covers all 128 positions, while every individual layer remains local.
_STAGE_27_PATH = Path(__file__).with_name("27_context_isolation.py")
_STAGE_27_MODULE_NAME = "stage_27_context_isolation_for_stage_29"
_STAGE_27_SPEC = importlib.util.spec_from_file_location(
    _STAGE_27_MODULE_NAME,
    _STAGE_27_PATH,
)
assert _STAGE_27_SPEC is not None and _STAGE_27_SPEC.loader is not None
_STAGE_27 = importlib.util.module_from_spec(_STAGE_27_SPEC)
sys.modules[_STAGE_27_MODULE_NAME] = _STAGE_27
_STAGE_27_SPEC.loader.exec_module(_STAGE_27)

_STAGE_18 = _STAGE_27._STAGE_18
_STAGE_26 = _STAGE_27._STAGE_26

GPTLanguageModel = _STAGE_27.GPTLanguageModel
LearningRateSchedule = _STAGE_27.LearningRateSchedule
BestValidationCheckpoint = _STAGE_27.BestValidationCheckpoint
PositionLosses = _STAGE_27.PositionLosses
AttentionEvaluationRecord = _STAGE_27.AttentionEvaluationRecord
AttentionTrainingResult = _STAGE_27.AttentionTrainingResult
PreciseBranchResult = _STAGE_27.PreciseBranchResult
PreciseDelta = _STAGE_27.PreciseDelta

checkpoint_sha256 = _STAGE_27.checkpoint_sha256
clear_accelerator_cache = _STAGE_27.clear_accelerator_cache
fingerprint_data = _STAGE_27.fingerprint_data
generate_from_final_model = _STAGE_27.generate_from_final_model
tensor_mapping_sha256 = _STAGE_27.tensor_mapping_sha256
tensor_sha256 = _STAGE_27.tensor_sha256


EXPERIMENT_STAGE = 29
EXPERIMENT_TITLE = "global receptive field through local routing"
COMPARISON_VARIABLE = "direct_global_attention_vs_multihop_global_reachability"
PROTOCOL_NAME = "global_receptive_field_local_routing"

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_LOCAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_29_window_23_at_128_best_checkpoint.pt"
)
DEFAULT_LOCAL_ANCHOR_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_29_window_23_at_128_step_13500_checkpoint.pt"
)
DEFAULT_FULL_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_26_context_128_best_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = 16
DEFAULT_BLOCK_SIZE = 128
DEFAULT_LOCAL_CONTEXT = 23
DEFAULT_N_EMBD = 128
DEFAULT_N_HEAD = 4
DEFAULT_N_LAYER = 6
DEFAULT_DROPOUT = 0.10
DEFAULT_MAX_ITERS = 18_000
DEFAULT_INITIAL_LEARNING_RATE = 1e-3
DEFAULT_FIRST_DECAY_STEP = 10_000
DEFAULT_MIDDLE_LEARNING_RATE = 3e-4
DEFAULT_SECOND_DECAY_STEP = 13_000
DEFAULT_FINAL_LEARNING_RATE = 1e-4
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_EVAL_ITERS = 100
DEFAULT_PRECISE_EVAL_ITERS = 500
DEFAULT_SEED = 1_337
DEFAULT_TRAINING_BATCH_SEED = 1_338
DEFAULT_FIXED_PANEL_SEED = 1_339
DEFAULT_TRAINING_RNG_SEED = 1_340
DEFAULT_SAMPLE_SEED = 1_342
DEFAULT_INITIAL_AUDIT_SEED = 1_343
# Reuse Stage 28's held-out panels so the W=12 and W=23 deltas share examples.
DEFAULT_PRECISE_EVAL_SEED = 1_346

DROPOUT_PLACEMENT = _STAGE_26.DROPOUT_PLACEMENT
DEFAULT_DATA_FINGERPRINT = _STAGE_26.DEFAULT_DATA_FINGERPRINT
DEFAULT_PARAMETER_COUNT = _STAGE_26.DEFAULT_TREATMENT_PARAMETER_COUNT
DEFAULT_SOURCE_FULL_INITIAL_STATE_SHA256 = (
    _STAGE_26.DEFAULT_TREATMENT_INITIAL_STATE_SHA256
)
DEFAULT_SOURCE_COMMON_T64_STATE_SHA256 = _STAGE_26.DEFAULT_CONTROL_INITIAL_STATE_SHA256
DEFAULT_SOURCE_EXTRA_POSITION_ROWS_SHA256 = _STAGE_26.DEFAULT_EXTRA_POSITION_ROWS_SHA256
DEFAULT_SOURCE_SHARED_PARAMETER_SHA256 = (
    _STAGE_26.DEFAULT_SHARED_PARAMETER_INITIALIZATION_SHA256
)
DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256 = (
    _STAGE_27.DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256
)
DEFAULT_FULL_REFERENCE_STEP = 13_500

DEFAULT_PARAMETER_INITIALIZATION_SHA256 = (
    _STAGE_27.DEFAULT_PARAMETER_INITIALIZATION_SHA256
)
DEFAULT_FULL_INITIAL_STATE_SHA256 = _STAGE_27.DEFAULT_FULL_INITIAL_STATE_SHA256
DEFAULT_LOCAL_INITIAL_STATE_SHA256 = (
    "4feb26dae434a6fad7a866895b3e53d7efcec79d3dc31f6af5a412fb758cd4c1"
)
DEFAULT_FULL_MASK_SHA256 = _STAGE_27.DEFAULT_FULL_MASK_SHA256
DEFAULT_FULL_FLOAT_MASK_SHA256 = _STAGE_27.DEFAULT_FULL_FLOAT_MASK_SHA256
DEFAULT_LOCAL_MASK_SHA256 = (
    "bab7c894395eb65e7ac2d5b7d2eaa0f8687e991914182ec2ad36858d0f16544b"
)
DEFAULT_LOCAL_FLOAT_MASK_SHA256 = (
    "d0ec90278c3d187be126d359bdb43d62e763eb478dc5c3d0fa07ca30f8603089"
)
DEFAULT_THEORETICAL_RECEPTIVE_FIELD = 133
DEFAULT_FIRST_MULTI_HOP_QUERY = 23
DEFAULT_LOCAL_ALLOWED_EDGES_PER_HEAD = 2_691
DEFAULT_REMOVED_EDGES_PER_HEAD = 5_565


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """One attention policy over the common dense T=128 geometry."""

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
    def mask_kind(self) -> str:
        if self.max_context is None:
            return "full_causal"
        return "causal_sliding_window"


@dataclass(frozen=True, slots=True)
class ReachabilityAudit:
    direct_attention_span: int
    n_layer: int
    theoretical_receptive_field_width: int
    effective_receptive_field_width: int
    widths_by_layer: tuple[int, ...]
    final_width_by_query: tuple[int, ...]
    final_earliest_source_by_query: tuple[int, ...]
    first_query_with_unavailable_source: int | None
    no_context_reset: bool
    causal: bool
    contiguous: bool

    @property
    def receptive_field_width(self) -> int:
        """Compatibility shorthand for the effective end-to-end width."""

        return self.effective_receptive_field_width

    @property
    def earliest_reachable_key_at_last_query(self) -> int:
        return self.final_earliest_source_by_query[-1]

    @property
    def has_hard_reset(self) -> bool:
        return not self.no_context_reset

    @property
    def all_causal_sources_reachable(self) -> bool:
        return (
            self.first_query_with_unavailable_source is None
            and all(source == 0 for source in self.final_earliest_source_by_query)
            and self.final_width_by_query
            == tuple(range(1, len(self.final_width_by_query) + 1))
        )


@dataclass(frozen=True, slots=True)
class PairedInitialStates:
    local: dict[str, torch.Tensor]
    full: dict[str, torch.Tensor]
    parameter_state_sha256: str
    local_state_sha256: str
    full_state_sha256: str
    local_mask_sha256: str
    full_mask_sha256: str
    local_float_mask_sha256: str
    full_float_mask_sha256: str
    parameter_count: int
    differing_state_names: tuple[str, ...]
    source_common_t64_state_sha256: str
    source_extra_position_rows_sha256: str
    source_shared_parameter_sha256: str


@dataclass(frozen=True, slots=True)
class FullReference:
    spec: AttentionSpec
    checkpoint_sha256: str
    checkpoint_step: int
    best_training_losses: PositionLosses
    best_validation_losses: PositionLosses
    checkpoint: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LocalTrainingReport:
    spec: AttentionSpec
    parameter_count: int
    initial_state_sha256: str
    parameter_state_sha256: str
    training: AttentionTrainingResult
    best_training_losses: PositionLosses
    best_validation_losses: PositionLosses
    best_val_loss: float
    best_step: int
    anchor_training_losses: PositionLosses
    anchor_validation_losses: PositionLosses
    anchor_step: int
    final_batch_generator_sha256: str
    attention_shape: tuple[int, ...]
    sample: str
    benchmark: BenchmarkStats | None = None

    @property
    def best_gap(self) -> float:
        return self.best_validation_losses.overall - self.best_training_losses.overall


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    initialization: PairedInitialStates
    initialization_seed: int
    training_batch_seed: int
    training_rng_seed: int
    schedule: LearningRateSchedule
    bands: tuple[tuple[int, int], ...]
    reachability: ReachabilityAudit
    local: LocalTrainingReport
    full: FullReference
    initial_shared_prefix_length: int
    initial_shared_prefix_max_difference: float
    local_benchmark: BenchmarkStats | None
    full_benchmark: BenchmarkStats | None
    local_checkpoint_sha256: str
    local_anchor_checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class RangeDelta:
    """Paired validation delta over one predeclared contiguous position range."""

    candidate: str
    baseline: str
    start: int
    end: int
    mean_delta: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    critical_value: float


@dataclass(frozen=True, slots=True)
class EndpointValidationReport:
    name: str
    full_natural: PreciseBranchResult
    local_natural: PreciseBranchResult
    natural_delta: PreciseDelta
    multi_hop_delta: RangeDelta


@dataclass(frozen=True, slots=True)
class Stage29PreciseValidationReport:
    eval_iters: int
    seed: int
    bands: tuple[tuple[int, int], ...]
    anchor: EndpointValidationReport
    selected_best: EndpointValidationReport


def build_attention_mask(
    block_size: int,
    max_context: int | None,
) -> torch.Tensor:
    """Build full causal attention or a continuous latest-N window."""

    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
    if max_context is not None:
        if (
            isinstance(max_context, bool)
            or not isinstance(max_context, int)
            or not 0 < max_context <= block_size
        ):
            raise ValueError(
                f"max_context must be in [1, {block_size}] or None, got {max_context!r}"
            )

    positions = torch.arange(block_size)
    query_position = positions[:, None]
    key_position = positions[None, :]
    allowed = key_position <= query_position
    if max_context is not None:
        allowed &= key_position >= query_position - max_context + 1
    return allowed


def theoretical_receptive_field_width(
    n_layer: int,
    direct_attention_span: int,
) -> int:
    for value, name in (
        (n_layer, "n_layer"),
        (direct_attention_span, "direct_attention_span"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return 1 + n_layer * (direct_attention_span - 1)


def minimum_local_context_for_global_receptive_field(
    block_size: int,
    n_layer: int,
) -> int:
    """Return the smallest uniform window whose stacked RF covers block_size."""

    for value, name in ((block_size, "block_size"), (n_layer, "n_layer")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return 1 + math.ceil((block_size - 1) / n_layer)


def receptive_field_widths_by_layer(
    n_layer: int,
    direct_attention_span: int,
) -> tuple[int, ...]:
    theoretical_receptive_field_width(n_layer, direct_attention_span)
    return tuple(
        theoretical_receptive_field_width(layer, direct_attention_span)
        for layer in range(1, n_layer + 1)
    )


def effective_receptive_field_width(
    block_size: int,
    n_layer: int,
    spec: AttentionSpec,
) -> int:
    if spec.max_context is None:
        return block_size
    return min(
        block_size,
        theoretical_receptive_field_width(n_layer, spec.max_context),
    )


def audit_bounded_reachability(
    mask: torch.Tensor,
    n_layer: int,
) -> ReachabilityAudit:
    """Prove continuous local access and exact stacked source reachability."""

    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be a square bool tensor")
    if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer <= 0:
        raise ValueError(f"n_layer must be a positive integer, got {n_layer!r}")
    block_size = int(mask.shape[0])
    if not torch.all(mask.diagonal()).item():
        raise RuntimeError("every query must retain its own position")
    if torch.count_nonzero(torch.triu(mask, diagonal=1)).item() != 0:
        raise RuntimeError("attention mask is not causal")

    row_counts = tuple(int(value) for value in mask.sum(dim=1).tolist())
    direct_span = max(row_counts)
    if direct_span <= 0:
        raise RuntimeError("attention mask contains an empty query row")
    expected_mask = build_attention_mask(block_size, direct_span)
    contiguous = torch.equal(mask, expected_mask)
    if not contiguous:
        raise RuntimeError("local attention rows are not continuous windows")
    expected_counts = tuple(min(query + 1, direct_span) for query in range(block_size))
    no_reset = row_counts == expected_counts and all(
        bool(mask[query, query - 1]) for query in range(1, block_size)
    )
    if not no_reset:
        raise RuntimeError("local attention contains a context reset")

    # reach[q, s] means representation q can depend on source token s.  The
    # diagonal is present, so multiplying by the one-layer relation includes
    # residual/self paths as well as attention relay.
    reach = torch.eye(block_size, dtype=torch.bool)
    observed_layer_widths: list[int] = []
    for layer in range(1, n_layer + 1):
        reach = (mask.float() @ reach.float()) > 0
        expected_width = theoretical_receptive_field_width(layer, direct_span)
        widths = tuple(int(value) for value in reach.sum(dim=1).tolist())
        expected_by_query = tuple(
            min(query + 1, expected_width) for query in range(block_size)
        )
        if widths != expected_by_query:
            raise RuntimeError(
                f"layer {layer} reachability is not the expected contiguous "
                f"width {expected_width}"
            )
        for query in range(block_size):
            earliest = max(0, query - expected_width + 1)
            expected_row = torch.zeros(block_size, dtype=torch.bool)
            expected_row[earliest : query + 1] = True
            if not torch.equal(reach[query], expected_row):
                raise RuntimeError(
                    f"layer {layer}, query {query} has non-contiguous reachability"
                )
        observed_layer_widths.append(expected_width)

    final_widths = tuple(int(value) for value in reach.sum(dim=1).tolist())
    earliest_sources = tuple(
        int(torch.nonzero(reach[query], as_tuple=False)[0].item())
        for query in range(block_size)
    )
    first_unavailable = next(
        (query for query, earliest in enumerate(earliest_sources) if earliest > 0),
        None,
    )
    theoretical = theoretical_receptive_field_width(n_layer, direct_span)
    return ReachabilityAudit(
        direct_attention_span=direct_span,
        n_layer=n_layer,
        theoretical_receptive_field_width=theoretical,
        effective_receptive_field_width=min(block_size, theoretical),
        widths_by_layer=tuple(observed_layer_widths),
        final_width_by_query=final_widths,
        final_earliest_source_by_query=earliest_sources,
        first_query_with_unavailable_source=first_unavailable,
        no_context_reset=no_reset,
        causal=True,
        contiguous=contiguous,
    )


def verify_global_reachability(
    audit: ReachabilityAudit,
    block_size: int,
) -> None:
    """Require final-layer reachability to cover every causal source origin."""

    if audit.effective_receptive_field_width != block_size:
        raise RuntimeError("local receptive field does not cover the full block")
    if not audit.all_causal_sources_reachable:
        raise RuntimeError("not every causal source is reachable at the final layer")


def make_position_bands(
    block_size: int,
    local_context: int = DEFAULT_LOCAL_CONTEXT,
    n_layer: int = DEFAULT_N_LAYER,
) -> tuple[tuple[int, int], ...]:
    """Partition queries by the local hop depth needed to reach source 0."""

    for value, name in (
        (block_size, "block_size"),
        (local_context, "local_context"),
        (n_layer, "n_layer"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if local_context > block_size:
        raise ValueError("local_context must not exceed block_size")
    cut_points = {0, block_size}
    for width in receptive_field_widths_by_layer(n_layer, local_context):
        cut_points.add(min(block_size, width))
    ordered_cut_points = sorted(cut_points)
    return tuple(
        (start, end)
        for start, end in zip(ordered_cut_points, ordered_cut_points[1:])
        if start < end
    )


def local_hops_for_oldest_source(query: int, local_context: int) -> int:
    """Return a 1-indexed local-routing bucket for reaching the oldest source.

    Query zero is grouped into the first self/direct bucket so contiguous
    position bands retain one depth label.
    """

    for value, name in ((query, "query"), (local_context, "local_context")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    if query < 0:
        raise ValueError("query must be non-negative")
    if local_context <= 1:
        if query == 0 and local_context == 1:
            return 1
        raise ValueError("local_context must exceed 1 for nonzero queries")
    return max(1, math.ceil(query / (local_context - 1)))


def hop_depths_for_bands(
    bands: Sequence[tuple[int, int]],
    local_context: int,
) -> tuple[int, ...]:
    """Return the constant 1-indexed routing bucket represented by each band."""

    depths: list[int] = []
    for start, end in bands:
        if not 0 <= start < end:
            raise ValueError(f"invalid position band {(start, end)!r}")
        start_depth = local_hops_for_oldest_source(start, local_context)
        end_depth = local_hops_for_oldest_source(end - 1, local_context)
        if start_depth != end_depth:
            raise ValueError(f"position band {(start, end)!r} crosses hop depths")
        depths.append(start_depth)
    return tuple(depths)


def _mask_state_names(state: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    names = tuple(sorted(name for name in state if name.endswith(".tril")))
    if not names:
        raise RuntimeError("model state contains no attention-mask buffers")
    return names


def _clone_state(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def clone_parameters(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def apply_attention_mask(model: GPTLanguageModel, mask: torch.Tensor) -> None:
    if mask.dtype != torch.bool:
        raise ValueError("attention mask must have bool dtype")
    expected_shape = (model.block_size, model.block_size)
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"attention mask must have shape {expected_shape}, got {tuple(mask.shape)}"
        )
    for block in model.blocks:
        for head in block.sa.heads:
            head.tril.copy_(mask.to(device=head.tril.device, dtype=head.tril.dtype))


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    spec: AttentionSpec,
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
    apply_attention_mask(
        model,
        build_attention_mask(config.block_size, spec.max_context),
    )
    if device is not None:
        model = model.to(device)
    return model


def verify_model_mask(model: GPTLanguageModel, spec: AttentionSpec) -> None:
    expected_bool = build_attention_mask(model.block_size, spec.max_context)
    observed_count = 0
    for block in model.blocks:
        for head in block.sa.heads:
            observed_count += 1
            observed = head.tril.detach().cpu()
            if not torch.equal(observed, expected_bool.to(dtype=observed.dtype)):
                raise RuntimeError(
                    f"{spec.name} contains an incorrect attention-mask buffer"
                )
    if observed_count == 0:
        raise RuntimeError("model contains no attention heads")


def _stage_26_configs(
    config: TrainingConfig,
) -> tuple[TrainingConfig, TrainingConfig]:
    if config.block_size % 2:
        raise ValueError("Stage 26 source initialization requires an even block size")
    common = {
        "n_embd": config.n_embd,
        "learning_rate": config.learning_rate,
        "max_iters": config.max_iters,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
    }
    control = TrainingConfig(
        batch_size=config.batch_size * 2,
        block_size=config.block_size // 2,
        **common,
    )
    treatment = TrainingConfig(
        batch_size=config.batch_size,
        block_size=config.block_size,
        **common,
    )
    return control, treatment


def _resolved_paths(path: Path) -> tuple[Path, Path]:
    resolved = path.resolve()
    temporary = path.with_name(f".{path.name}.tmp").resolve()
    return resolved, temporary


def validate_attention_protocol(
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int = DEFAULT_N_HEAD,
    n_layer: int = DEFAULT_N_LAYER,
    *,
    anchor_checkpoint_path: Path | None = None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    if local.name == full.name:
        raise ValueError("local and full branch names must differ")
    if local.max_context is None:
        raise ValueError("the local branch must set max_context")
    if full.max_context is not None:
        raise ValueError("the full reference must use max_context=None")
    if local.max_context >= config.block_size:
        raise ValueError("local max_context must remain below block_size")
    if n_head <= 0 or config.n_embd % n_head:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by positive n_head ({n_head})"
        )
    if n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {n_layer}")
    receptive_field = theoretical_receptive_field_width(
        n_layer,
        local.max_context,
    )
    minimum_context = minimum_local_context_for_global_receptive_field(
        config.block_size,
        n_layer,
    )
    if local.max_context != minimum_context:
        raise ValueError(
            "local max_context must be the minimum uniform global-RF window "
            f"({minimum_context}), got {local.max_context}"
        )
    if receptive_field < config.block_size:
        raise ValueError(
            "local end-to-end receptive field must cover the full block"
        )

    labelled_paths = {
        "local checkpoint": _resolved_paths(local.checkpoint_path)[0],
        "local temporary checkpoint": _resolved_paths(local.checkpoint_path)[1],
        "full reference": _resolved_paths(full.checkpoint_path)[0],
        "full temporary path": _resolved_paths(full.checkpoint_path)[1],
    }
    if anchor_checkpoint_path is not None:
        labelled_paths.update(
            {
                "local anchor checkpoint": _resolved_paths(anchor_checkpoint_path)[0],
                "local anchor temporary checkpoint": _resolved_paths(
                    anchor_checkpoint_path
                )[1],
            }
        )
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

    bands = make_position_bands(
        config.block_size,
        local.max_context,
        n_layer,
    )
    if bands[0] != (0, local.max_context):
        raise RuntimeError("the first band must be the exact direct-mask prefix")
    if hop_depths_for_bands(bands, local.max_context) != tuple(
        range(1, len(bands) + 1)
    ):
        raise RuntimeError("position bands must advance one local hop at a time")
    return config.batch_size * config.block_size, bands


def _is_canonical_protocol(
    data: CharacterData,
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
) -> bool:
    return (
        fingerprint_data(data) == DEFAULT_DATA_FINGERPRINT
        and data.vocabulary.size == 65
        and config.batch_size == DEFAULT_BATCH_SIZE
        and config.block_size == DEFAULT_BLOCK_SIZE
        and config.n_embd == DEFAULT_N_EMBD
        and config.learning_rate == DEFAULT_INITIAL_LEARNING_RATE
        and config.max_iters == DEFAULT_MAX_ITERS
        and config.eval_interval == DEFAULT_EVAL_INTERVAL
        and config.eval_iters == DEFAULT_EVAL_ITERS
        and config.seed == DEFAULT_SEED
        and local.max_context == DEFAULT_LOCAL_CONTEXT
        and full.max_context is None
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
    )


def create_paired_initial_states(
    data: CharacterData,
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
) -> PairedInitialStates:
    """Regenerate Stage 26 T=128 and change only all 24 mask buffers."""

    validate_attention_protocol(config, local, full, n_head, n_layer)
    control_config, treatment_config = _stage_26_configs(config)
    source = _STAGE_26.create_paired_initial_states(
        data,
        control_config,
        treatment_config,
        n_head,
        n_layer,
    )
    full_state = {
        name: tensor.detach().cpu().clone() for name, tensor in source.treatment.items()
    }
    local_state = {name: tensor.clone() for name, tensor in full_state.items()}
    local_mask = build_attention_mask(config.block_size, local.max_context)
    full_mask = build_attention_mask(config.block_size, full.max_context)
    mask_names = _mask_state_names(full_state)
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
            "Stage 29 initial states must differ only in attention masks; "
            f"got {differing_names}"
        )

    local_model = build_model(data, config, local, n_head, n_layer)
    full_model = build_model(data, config, full, n_head, n_layer)
    local_model.load_state_dict(local_state, strict=True)
    full_model.load_state_dict(full_state, strict=True)
    local_parameters = clone_parameters(local_model)
    full_parameters = clone_parameters(full_model)
    if local_parameters.keys() != full_parameters.keys() or any(
        not torch.equal(local_parameters[name], full_parameters[name])
        for name in local_parameters
    ):
        raise RuntimeError("Stage 29 learned initializations are not identical")
    if any(
        local_parameters[name].data_ptr() == full_parameters[name].data_ptr()
        for name in local_parameters
    ):
        raise RuntimeError("Stage 29 parameter snapshots share storage")

    parameter_count = sum(
        parameter.numel()
        for parameter in local_model.parameters()
        if parameter.requires_grad
    )
    result = PairedInitialStates(
        local=local_state,
        full=full_state,
        parameter_state_sha256=tensor_mapping_sha256(local_parameters),
        local_state_sha256=tensor_mapping_sha256(local_state),
        full_state_sha256=tensor_mapping_sha256(full_state),
        local_mask_sha256=tensor_sha256(local_mask),
        full_mask_sha256=tensor_sha256(full_mask),
        local_float_mask_sha256=tensor_sha256(local_mask.float()),
        full_float_mask_sha256=tensor_sha256(full_mask.float()),
        parameter_count=parameter_count,
        differing_state_names=differing_names,
        source_common_t64_state_sha256=source.common_state_sha256,
        source_extra_position_rows_sha256=source.extra_position_rows_sha256,
        source_shared_parameter_sha256=(source.shared_parameter_initialization_sha256),
    )
    del local_model
    del full_model

    if _is_canonical_protocol(data, config, local, full, n_head, n_layer):
        reachability = audit_bounded_reachability(local_mask, n_layer)
        verify_global_reachability(reachability, config.block_size)
        local_allowed_edges = int(local_mask.sum().item())
        removed_edges = int(full_mask.sum().item()) - local_allowed_edges
        observed = {
            "parameter count": result.parameter_count,
            "parameter initialization": result.parameter_state_sha256,
            "local state": result.local_state_sha256,
            "full state": result.full_state_sha256,
            "source full state": result.full_state_sha256,
            "source common T64 state": result.source_common_t64_state_sha256,
            "source extra position rows": (result.source_extra_position_rows_sha256),
            "source shared parameters": (result.source_shared_parameter_sha256),
            "local mask": result.local_mask_sha256,
            "full mask": result.full_mask_sha256,
            "local float mask": result.local_float_mask_sha256,
            "full float mask": result.full_float_mask_sha256,
            "mask count": len(result.differing_state_names),
            "theoretical receptive field": (
                reachability.theoretical_receptive_field_width
            ),
            "first unavailable-source query": (
                reachability.first_query_with_unavailable_source
            ),
            "first multi-hop query": local.max_context,
            "local edges per head": local_allowed_edges,
            "removed edges per head": removed_edges,
        }
        expected = {
            "parameter count": DEFAULT_PARAMETER_COUNT,
            "parameter initialization": DEFAULT_PARAMETER_INITIALIZATION_SHA256,
            "local state": DEFAULT_LOCAL_INITIAL_STATE_SHA256,
            "full state": DEFAULT_FULL_INITIAL_STATE_SHA256,
            "source full state": DEFAULT_SOURCE_FULL_INITIAL_STATE_SHA256,
            "source common T64 state": DEFAULT_SOURCE_COMMON_T64_STATE_SHA256,
            "source extra position rows": (DEFAULT_SOURCE_EXTRA_POSITION_ROWS_SHA256),
            "source shared parameters": DEFAULT_SOURCE_SHARED_PARAMETER_SHA256,
            "local mask": DEFAULT_LOCAL_MASK_SHA256,
            "full mask": DEFAULT_FULL_MASK_SHA256,
            "local float mask": DEFAULT_LOCAL_FLOAT_MASK_SHA256,
            "full float mask": DEFAULT_FULL_FLOAT_MASK_SHA256,
            "mask count": DEFAULT_N_HEAD * DEFAULT_N_LAYER,
            "theoretical receptive field": (DEFAULT_THEORETICAL_RECEPTIVE_FIELD),
            "first unavailable-source query": None,
            "first multi-hop query": DEFAULT_FIRST_MULTI_HOP_QUERY,
            "local edges per head": DEFAULT_LOCAL_ALLOWED_EDGES_PER_HEAD,
            "removed edges per head": DEFAULT_REMOVED_EDGES_PER_HEAD,
        }
        if observed != expected:
            raise RuntimeError(
                "canonical Stage 29 initialization changed: "
                f"expected {expected}, got {observed}"
            )
    return result


def verify_initial_shared_prefix_equivalence(
    local_model: GPTLanguageModel,
    full_model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    seed: int,
    prefix_length: int | None = None,
) -> float:
    """Require exact initialization logits over identical direct-mask rows."""

    if prefix_length is None:
        prefix_length = DEFAULT_LOCAL_CONTEXT
    return _STAGE_18.verify_initial_shared_prefix_equivalence(
        local_model,
        full_model,
        data,
        config,
        device,
        seed=seed,
        prefix_length=prefix_length,
    )


def _serialize_losses(
    losses: PositionLosses,
    bands: Sequence[tuple[int, int]],
) -> dict[str, object]:
    return _STAGE_27._serialize_losses(losses, bands)


def _deserialize_losses(
    value: object,
    bands: Sequence[tuple[int, int]],
    *,
    name: str,
) -> PositionLosses:
    return _STAGE_27._deserialize_losses(value, bands, name=name)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    return _STAGE_27._require_mapping(value, name=name)


def _position_losses_from_stage_26(
    value: object,
    *,
    name: str,
) -> PositionLosses:
    return _STAGE_27._position_losses_from_stage_26(value, name=name)


def validate_stage_26_full_reference(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    full_spec: AttentionSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    initialization: PairedInitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_checkpoint_sha256: str,
    expected_step: int,
) -> FullReference:
    """Validate and rewrap the immutable Stage 26 T=128 checkpoint."""

    if full_spec.max_context is not None:
        raise ValueError("the Stage 26 reference must use full causal attention")
    adapter = _STAGE_27.CrossHalfSpec(
        name=full_spec.name,
        allow_cross_half=True,
        boundary=config.block_size // 2,
        checkpoint_path=full_spec.checkpoint_path,
    )
    reference = _STAGE_27.validate_stage_26_full_reference(
        path,
        data,
        config,
        adapter,
        n_head,
        n_layer,
        schedule,
        initialization,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_step=expected_step,
    )
    return FullReference(
        spec=full_spec,
        checkpoint_sha256=reference.checkpoint_sha256,
        checkpoint_step=reference.checkpoint_step,
        best_training_losses=reference.best_training_losses,
        best_validation_losses=reference.best_validation_losses,
        checkpoint=reference.checkpoint,
    )


def assert_matched_training_stream_state(
    full_checkpoint: Mapping[str, object],
    local_checkpoint: Mapping[str, object],
) -> None:
    _STAGE_27.assert_matched_training_stream_state(
        full_checkpoint,
        local_checkpoint,
    )


@dataclass(slots=True)
class LocalValidationCheckpoint(BestValidationCheckpoint):
    """Save selected-best and unconditional matched-step local artifacts."""

    local_spec: AttentionSpec | None = None
    full_spec: AttentionSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initialization_seed: int = 0
    branch_initial_state_sha256: str = ""
    full_initial_state_sha256: str = ""
    parameter_state_sha256: str = ""
    local_mask_sha256: str = ""
    full_mask_sha256: str = ""
    source_common_t64_state_sha256: str = ""
    source_extra_position_rows_sha256: str = ""
    source_shared_parameter_sha256: str = ""
    full_reference_sha256: str = ""
    full_reference_step: int = 0
    training_batch_seed: int = 0
    training_rng_seed: int = 0
    attention_shape: tuple[int, ...] = ()
    bands: tuple[tuple[int, int], ...] = ()
    reachability: ReachabilityAudit | None = None
    allowed_edges_per_head: int = 0
    removed_edges_per_head: int = 0
    anchor_path: Path | None = None
    anchor_step: int = 0
    best_training_losses: PositionLosses | None = None
    best_validation_losses: PositionLosses | None = None
    anchor_training_losses: PositionLosses | None = None
    anchor_validation_losses: PositionLosses | None = None
    anchor_saved: bool = False

    def _decorate_payload(
        self,
        payload: dict[str, object],
        *,
        step: int,
        checkpoint_kind: str,
    ) -> dict[str, object]:
        if (
            self.local_spec is None
            or self.full_spec is None
            or self.schedule is None
            or self.reachability is None
            or self.best_training_losses is None
            or self.best_validation_losses is None
            or not self.bands
        ):
            raise RuntimeError("incomplete Stage 29 checkpoint metadata")
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
        payload["checkpoint_kind"] = checkpoint_kind
        architecture = payload["architecture"]
        training = payload["training_config"]
        assert isinstance(architecture, dict)
        assert isinstance(training, dict)
        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        architecture.update(
            {
                "head_size": self.config.n_embd // self.n_head,
                "feed_forward_width": 4 * self.config.n_embd,
                "residual_dropout": DEFAULT_DROPOUT,
                "dropout_placement": DROPOUT_PLACEMENT,
                "attention_mask": self.local_spec.mask_kind,
                "attention_mask_scope": "every_head_in_every_layer",
                "max_context_per_layer": self.local_spec.max_context,
                "per_layer_attention_window": self.local_spec.max_context,
                "minimum_global_receptive_field_window": (
                    minimum_local_context_for_global_receptive_field(
                        self.config.block_size,
                        self.reachability.n_layer,
                    )
                ),
                "direct_attention_span": self.reachability.direct_attention_span,
                "theoretical_receptive_field_width": (
                    self.reachability.theoretical_receptive_field_width
                ),
                "effective_receptive_field_width": (
                    self.reachability.effective_receptive_field_width
                ),
                "parameter_count": parameter_count,
            }
        )
        optimizer_group = self.optimizer.param_groups[0]
        training.update(
            {
                "learning_rate": optimizer_rate,
                "max_iters": self.target_step,
                "learning_rate_schedule": schedule_metadata,
                "residual_dropout": DEFAULT_DROPOUT,
                "training_batch_seed": self.training_batch_seed,
                "training_rng_seed": self.training_rng_seed,
                "targets_per_update": self.config.batch_size * self.config.block_size,
                "total_target_characters": (
                    self.config.batch_size * self.config.block_size * self.target_step
                ),
                "optimizer": "AdamW",
                "optimizer_betas": tuple(optimizer_group["betas"]),
                "optimizer_eps": float(optimizer_group["eps"]),
                "optimizer_weight_decay": float(optimizer_group["weight_decay"]),
            }
        )
        payload["initialization"] = {
            "kind": "stage_26_t128_mask_matched",
            "seed": self.initialization_seed,
            "branch_state_sha256": self.branch_initial_state_sha256,
            "full_state_sha256": self.full_initial_state_sha256,
            "parameter_state_sha256": self.parameter_state_sha256,
            "identical_learned_tensors": True,
            "state_differs_only_in_attention_masks": True,
            "local_mask_sha256": self.local_mask_sha256,
            "full_mask_sha256": self.full_mask_sha256,
            "source_stage": 26,
            "source_common_t64_state_sha256": (self.source_common_t64_state_sha256),
            "source_extra_position_rows_sha256": (
                self.source_extra_position_rows_sha256
            ),
            "source_shared_parameter_sha256": (self.source_shared_parameter_sha256),
        }
        payload["experiment"] = {
            "stage": EXPERIMENT_STAGE,
            "protocol": PROTOCOL_NAME,
            "branch": self.local_spec.name,
            "from_scratch": True,
            "comparison_variable": COMPARISON_VARIABLE,
            "local_mask_policy": self.local_spec.mask_kind,
            "full_mask_policy": self.full_spec.mask_kind,
            "max_context_per_layer": self.local_spec.max_context,
            "per_layer_attention_window": self.local_spec.max_context,
            "minimum_global_receptive_field_window": (
                minimum_local_context_for_global_receptive_field(
                    self.config.block_size,
                    self.reachability.n_layer,
                )
            ),
            "direct_attention_span": self.reachability.direct_attention_span,
            "uncapped_receptive_field_width": (
                self.reachability.theoretical_receptive_field_width
            ),
            "effective_receptive_field_width": (
                self.reachability.effective_receptive_field_width
            ),
            "receptive_field_widths_by_layer": self.reachability.widths_by_layer,
            "first_query_with_unavailable_source": (
                self.reachability.first_query_with_unavailable_source
            ),
            "all_causal_source_origins_reachable_at_final_layer": (
                self.reachability.all_causal_sources_reachable
            ),
            "global_end_to_end_receptive_field": (
                self.reachability.effective_receptive_field_width
                == self.config.block_size
            ),
            "direct_global_attention": (
                self.local_spec.max_context >= self.config.block_size
            ),
            "first_query_with_shorter_full_path": self.local_spec.max_context,
            "maximum_oldest_source_path_hops": local_hops_for_oldest_source(
                self.config.block_size - 1,
                self.local_spec.max_context,
            ),
            "hop_depths_by_position_band": hop_depths_for_bands(
                self.bands,
                self.local_spec.max_context,
            ),
            "context_reset": False,
            "continuous_local_history": True,
            "continuous_context_without_hard_reset": True,
            "multilayer_relay_accounted_for": True,
            "intervention_changes_reachable_source_origins_at_final_layer": (
                not self.reachability.all_causal_sources_reachable
            ),
            "intervention_changes_direct_fan_in": True,
            "intervention_changes_information_path_length": True,
            "allowed_attention_edges_per_head": self.allowed_edges_per_head,
            "removed_attention_edges_per_head": self.removed_edges_per_head,
            "dense_attention_computation": True,
            "batch_size": self.config.batch_size,
            "block_size": self.config.block_size,
            "identical_parameter_count": True,
            "identical_learned_initialization": True,
            "identical_position_embeddings": True,
            "identical_training_batch_stream": True,
            "matched_training_rng_seed": True,
            "identical_dropout_probability": True,
            "matched_dropout_rng_stream": True,
            "identical_dropout_draws_directly_audited": False,
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
            "shorter_path_position_range": (
                self.local_spec.max_context,
                self.config.block_size,
            ),
            "full_reference_source_stage": 26,
            "full_reference_checkpoint_sha256": self.full_reference_sha256,
            "full_reference_step": self.full_reference_step,
            "full_reference_reused_read_only": True,
            "only_local_branch_newly_trained": True,
        }
        payload["evaluation_protocol"] = {
            "checkpoint_selection_objective": "overall_validation_loss",
            "primary_endpoint": "matched_anchor_natural_overall_validation_loss",
            "predeclared_mechanistic_diagnostic": (
                "positions_"
                f"{self.local_spec.max_context}_through_"
                f"{self.config.block_size - 1}_multi_hop_loss"
            ),
            "mechanistic_diagnostic_position_range": (
                self.local_spec.max_context,
                self.config.block_size,
            ),
            "first_query_with_shorter_full_path": self.local_spec.max_context,
            "position_band_local_hops_to_oldest_source": hop_depths_for_bands(
                self.bands,
                self.local_spec.max_context,
            ),
            "natural_comparison": (
                "full_weights_full_mask_minus_local_weights_local_mask"
            ),
            "selected_best_endpoint": "secondary",
            "same_weight_mask_ablations": False,
            "crossed_mask_variants": False,
            "held_out_validation_natural_masks_only": True,
            "initial_exact_prefix": (0, self.local_spec.max_context),
            "fixed_panel_seed": self.config.seed + 2,
            "split_order": ("train", "val"),
            "batches_per_split": self.config.eval_iters,
            "position_bands": self.bands,
            "confidence_interval_scope": "validation_batch_sampling_only",
            "multiplicity_adjustment": "none",
            "secondary_intervals": "exploratory_pointwise_unadjusted",
        }
        payload["best_training_losses"] = _serialize_losses(
            self.best_training_losses,
            self.bands,
        )
        payload["best_validation_losses"] = _serialize_losses(
            self.best_validation_losses,
            self.bands,
        )
        payload["anchor_step"] = self.anchor_step
        if checkpoint_kind == "anchor":
            if (
                self.anchor_training_losses is None
                or self.anchor_validation_losses is None
            ):
                raise RuntimeError("anchor losses are unavailable")
            payload["anchor_training_losses"] = _serialize_losses(
                self.anchor_training_losses,
                self.bands,
            )
            payload["anchor_validation_losses"] = _serialize_losses(
                self.anchor_validation_losses,
                self.bands,
            )
        return payload

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(LocalValidationCheckpoint, self)._payload(step)
        return self._decorate_payload(payload, step=step, checkpoint_kind="best")

    def _save_anchor(self, record: AttentionEvaluationRecord) -> None:
        if self.anchor_path is None:
            raise RuntimeError("anchor checkpoint path is unavailable")
        self.anchor_training_losses = record.losses.train
        self.anchor_validation_losses = record.losses.val
        payload = super(LocalValidationCheckpoint, self)._payload(record.step)
        payload = self._decorate_payload(
            payload,
            step=record.step,
            checkpoint_kind="anchor",
        )
        self.anchor_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.anchor_path.with_name(f".{self.anchor_path.name}.tmp")
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(self.anchor_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self.anchor_saved = True

    def consider(self, record: AttentionEvaluationRecord) -> bool:
        val_loss = record.losses.val.overall
        if not math.isfinite(val_loss):
            raise ValueError(f"validation loss at step {record.step} must be finite")
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_step = record.step
            self.best_training_losses = record.losses.train
            self.best_validation_losses = record.losses.val
            self.save(record.step)
        # This is deliberately independent of checkpoint selection.
        if record.step == self.anchor_step:
            self._save_anchor(record)
        return is_best


def validate_stage_29_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    full_reference: FullReference,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    checkpoint_kind: str,
) -> Mapping[str, object]:
    """Strictly validate a selected-best or matched-step Stage 29 artifact."""

    if checkpoint_kind not in {"best", "anchor"}:
        raise ValueError("checkpoint_kind must be 'best' or 'anchor'")
    if not path.is_file():
        raise FileNotFoundError(f"Stage 29 checkpoint not found: {path}")
    _, expected_bands = validate_attention_protocol(
        config,
        local,
        full,
        n_head,
        n_layer,
    )
    if bands != expected_bands:
        raise ValueError("checkpoint bands differ from the Stage 29 protocol")
    local_mask = build_attention_mask(config.block_size, local.max_context)
    full_mask = build_attention_mask(config.block_size, full.max_context)
    reachability = audit_bounded_reachability(local_mask, n_layer)
    verify_global_reachability(reachability, config.block_size)
    allowed_edges = int(local_mask.sum().item())
    removed_edges = int(full_mask.sum().item() - allowed_edges)

    before_hash = checkpoint_sha256(path)
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="root",
    )
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("checkpoint version must be 1")
    if checkpoint.get("checkpoint_kind") != checkpoint_kind:
        raise ValueError(
            f"checkpoint kind must be {checkpoint_kind!r}, "
            f"got {checkpoint.get('checkpoint_kind')!r}"
        )
    step = checkpoint.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= config.max_iters
    ):
        raise ValueError("checkpoint step is invalid")
    best_step = checkpoint.get("best_step")
    if (
        isinstance(best_step, bool)
        or not isinstance(best_step, int)
        or not 0 <= best_step <= step
    ):
        raise ValueError("checkpoint best_step is invalid")
    if checkpoint_kind == "best" and best_step != step:
        raise ValueError("best checkpoint step must equal best_step")
    if checkpoint.get("anchor_step") != full_reference.checkpoint_step:
        raise ValueError("anchor_step does not match the full reference")
    if checkpoint_kind == "anchor" and step != full_reference.checkpoint_step:
        raise ValueError("anchor checkpoint is not at the full-reference step")
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
        raise ValueError("Stage 29 must not restart the optimizer")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 29 optimizer provenance must be known")
    _STAGE_27._validate_rng_state(checkpoint)

    architecture = _require_mapping(
        checkpoint.get("architecture"),
        name="architecture",
    )
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "residual_dropout": DEFAULT_DROPOUT,
        "dropout_placement": DROPOUT_PLACEMENT,
        "attention_mask": local.mask_kind,
        "attention_mask_scope": "every_head_in_every_layer",
        "max_context_per_layer": local.max_context,
        "per_layer_attention_window": local.max_context,
        "minimum_global_receptive_field_window": (
            minimum_local_context_for_global_receptive_field(
                config.block_size,
                n_layer,
            )
        ),
        "direct_attention_span": reachability.direct_attention_span,
        "theoretical_receptive_field_width": (
            reachability.theoretical_receptive_field_width
        ),
        "effective_receptive_field_width": (
            reachability.effective_receptive_field_width
        ),
        "parameter_count": initialization.parameter_count,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"checkpoint architecture.{key} must be {expected!r}, "
                f"got {architecture.get(key)!r}"
            )

    expected_learning_rate = schedule.for_update(max(0, step - 1))
    training = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    expected_training = {
        "batch_size": config.batch_size,
        "learning_rate": expected_learning_rate,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": config.max_iters,
        "learning_rate_schedule": schedule.as_metadata(max_iters=config.max_iters),
        "residual_dropout": DEFAULT_DROPOUT,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
        "total_target_characters": (
            config.batch_size * config.block_size * config.max_iters
        ),
        "optimizer": "AdamW",
        "optimizer_betas": (0.9, 0.999),
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0.01,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"checkpoint training_config.{key} must be {expected!r}, "
                f"got {training.get(key)!r}"
            )

    initialization_metadata = _require_mapping(
        checkpoint.get("initialization"),
        name="initialization",
    )
    expected_initialization = {
        "kind": "stage_26_t128_mask_matched",
        "seed": config.seed,
        "branch_state_sha256": initialization.local_state_sha256,
        "full_state_sha256": initialization.full_state_sha256,
        "parameter_state_sha256": initialization.parameter_state_sha256,
        "identical_learned_tensors": True,
        "state_differs_only_in_attention_masks": True,
        "local_mask_sha256": initialization.local_mask_sha256,
        "full_mask_sha256": initialization.full_mask_sha256,
        "source_stage": 26,
        "source_common_t64_state_sha256": (
            initialization.source_common_t64_state_sha256
        ),
        "source_extra_position_rows_sha256": (
            initialization.source_extra_position_rows_sha256
        ),
        "source_shared_parameter_sha256": (
            initialization.source_shared_parameter_sha256
        ),
    }
    for key, expected in expected_initialization.items():
        if initialization_metadata.get(key) != expected:
            raise ValueError(f"checkpoint initialization.{key} must be {expected!r}")

    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    expected_experiment = {
        "stage": EXPERIMENT_STAGE,
        "protocol": PROTOCOL_NAME,
        "branch": local.name,
        "from_scratch": True,
        "comparison_variable": COMPARISON_VARIABLE,
        "local_mask_policy": local.mask_kind,
        "full_mask_policy": full.mask_kind,
        "max_context_per_layer": local.max_context,
        "per_layer_attention_window": local.max_context,
        "minimum_global_receptive_field_window": (
            minimum_local_context_for_global_receptive_field(
                config.block_size,
                n_layer,
            )
        ),
        "direct_attention_span": reachability.direct_attention_span,
        "uncapped_receptive_field_width": (
            reachability.theoretical_receptive_field_width
        ),
        "effective_receptive_field_width": (
            reachability.effective_receptive_field_width
        ),
        "receptive_field_widths_by_layer": reachability.widths_by_layer,
        "first_query_with_unavailable_source": (
            reachability.first_query_with_unavailable_source
        ),
        "all_causal_source_origins_reachable_at_final_layer": (
            reachability.all_causal_sources_reachable
        ),
        "global_end_to_end_receptive_field": (
            reachability.effective_receptive_field_width == config.block_size
        ),
        "direct_global_attention": local.max_context >= config.block_size,
        "first_query_with_shorter_full_path": local.max_context,
        "maximum_oldest_source_path_hops": local_hops_for_oldest_source(
            config.block_size - 1,
            local.max_context,
        ),
        "hop_depths_by_position_band": hop_depths_for_bands(
            bands,
            local.max_context,
        ),
        "context_reset": False,
        "continuous_local_history": True,
        "continuous_context_without_hard_reset": True,
        "multilayer_relay_accounted_for": True,
        "intervention_changes_reachable_source_origins_at_final_layer": (
            not reachability.all_causal_sources_reachable
        ),
        "intervention_changes_direct_fan_in": True,
        "intervention_changes_information_path_length": True,
        "allowed_attention_edges_per_head": allowed_edges,
        "removed_attention_edges_per_head": removed_edges,
        "dense_attention_computation": True,
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "identical_parameter_count": True,
        "identical_learned_initialization": True,
        "identical_position_embeddings": True,
        "identical_training_batch_stream": True,
        "matched_training_rng_seed": True,
        "identical_dropout_probability": True,
        "matched_dropout_rng_stream": True,
        "identical_dropout_draws_directly_audited": False,
        "identical_attention_shapes": True,
        "targets_per_update": config.batch_size * config.block_size,
        "total_target_characters": (
            config.batch_size * config.block_size * config.max_iters
        ),
        "initialization_seed": config.seed,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "residual_dropout": DEFAULT_DROPOUT,
        "learning_rate_schedule": schedule.as_metadata(max_iters=config.max_iters),
        "attention_shape": (
            config.batch_size,
            n_head,
            config.block_size,
            config.block_size,
        ),
        "position_bands": bands,
        "shorter_path_position_range": (local.max_context, config.block_size),
        "full_reference_source_stage": 26,
        "full_reference_checkpoint_sha256": full_reference.checkpoint_sha256,
        "full_reference_step": full_reference.checkpoint_step,
        "full_reference_reused_read_only": True,
        "only_local_branch_newly_trained": True,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"checkpoint experiment.{key} must be {expected!r}, "
                f"got {experiment.get(key)!r}"
            )
    if set(experiment) != set(expected_experiment):
        raise ValueError("checkpoint experiment keys differ from the protocol")

    evaluation = _require_mapping(
        checkpoint.get("evaluation_protocol"),
        name="evaluation_protocol",
    )
    expected_evaluation = {
        "checkpoint_selection_objective": "overall_validation_loss",
        "primary_endpoint": "matched_anchor_natural_overall_validation_loss",
        "predeclared_mechanistic_diagnostic": (
            "positions_"
            f"{local.max_context}_through_"
            f"{config.block_size - 1}_multi_hop_loss"
        ),
        "mechanistic_diagnostic_position_range": (
            local.max_context,
            config.block_size,
        ),
        "first_query_with_shorter_full_path": local.max_context,
        "position_band_local_hops_to_oldest_source": hop_depths_for_bands(
            bands,
            local.max_context,
        ),
        "natural_comparison": ("full_weights_full_mask_minus_local_weights_local_mask"),
        "selected_best_endpoint": "secondary",
        "same_weight_mask_ablations": False,
        "crossed_mask_variants": False,
        "held_out_validation_natural_masks_only": True,
        "initial_exact_prefix": (0, local.max_context),
        "fixed_panel_seed": config.seed + 2,
        "split_order": ("train", "val"),
        "batches_per_split": config.eval_iters,
        "position_bands": bands,
        "confidence_interval_scope": "validation_batch_sampling_only",
        "multiplicity_adjustment": "none",
        "secondary_intervals": "exploratory_pointwise_unadjusted",
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(
                f"checkpoint evaluation_protocol.{key} must be {expected!r}"
            )
    if set(evaluation) != set(expected_evaluation):
        raise ValueError(
            "checkpoint evaluation_protocol keys differ from the protocol"
        )

    best_train = _deserialize_losses(
        checkpoint.get("best_training_losses"),
        bands,
        name="best_training_losses",
    )
    best_val = _deserialize_losses(
        checkpoint.get("best_validation_losses"),
        bands,
        name="best_validation_losses",
    )
    if best_val.overall != float(best_val_loss):
        raise ValueError("best validation loss does not match best_val_loss")
    if checkpoint_kind == "anchor":
        _deserialize_losses(
            checkpoint.get("anchor_training_losses"),
            bands,
            name="anchor_training_losses",
        )
        _deserialize_losses(
            checkpoint.get("anchor_validation_losses"),
            bands,
            name="anchor_validation_losses",
        )
    del best_train

    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"),
        name="optimizer_state_dict",
    )
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("checkpoint optimizer must contain parameter groups")
    optimizer_rates = {
        float(group["lr"])
        for group in param_groups
        if isinstance(group, Mapping) and "lr" in group
    }
    if len(optimizer_rates) != 1 or optimizer_rates != {expected_learning_rate}:
        raise ValueError("checkpoint optimizer learning rate is inconsistent")

    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("checkpoint training generator state must be a tensor")
    expected_generator = _STAGE_18.expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator):
        raise ValueError(
            "checkpoint training generator state does not match the batch stream"
        )

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    compatibility_model = build_model(data, config, local, n_head, n_layer)
    compatibility_model.load_state_dict(model_state, strict=True)
    verify_model_mask(compatibility_model, local)
    del compatibility_model
    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError("Stage 29 checkpoint changed during validation")
    return checkpoint


def inspect_attention_shape(
    model: GPTLanguageModel,
    config: TrainingConfig,
    spec: AttentionSpec,
    n_head: int,
    device: torch.device,
) -> tuple[int, ...]:
    """Verify logical masking while retaining dense BxHxTxT tensors."""

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
        allowed = build_attention_mask(
            config.block_size,
            spec.max_context,
        ).to(device)
        forbidden = weights.masked_select(~allowed[None, None])
        if forbidden.numel() and torch.count_nonzero(forbidden).item() != 0:
            raise RuntimeError(f"{spec.name} assigns weight to masked positions")
        return shape
    finally:
        model.train(was_training)


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
    """Benchmark the exact dense Stage 29 training path on disposable state."""

    clear_accelerator_cache(device)
    model = build_model(data, config, spec, n_head, n_layer, device)
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
        tokens_per_sec=iterations_per_sec * config.batch_size * config.block_size,
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
    for spec, state, destination in ordered_runs:
        destination.append(
            benchmark_attention_branch(
                data,
                config,
                spec,
                n_head,
                n_layer,
                device,
                state,
                training_batch_seed=training_batch_seed,
                training_rng_seed=training_rng_seed,
                benchmark=benchmark,
            )
        )
    return (
        _STAGE_18.aggregate_benchmark_stats(
            local_runs,
            steps_per_run=benchmark.num_steps,
        ),
        _STAGE_18.aggregate_benchmark_stats(
            full_runs,
            steps_per_run=benchmark.num_steps,
        ),
    )


def _print_benchmark(name: str, stats: BenchmarkStats) -> None:
    print(f"\n{name} dense training benchmark")
    print(f"  seconds:               {stats.seconds:.3f}")
    print(f"  iterations/sec:        {stats.iterations_per_sec:.3f}")
    print(f"  target characters/sec: {stats.tokens_per_sec:.3f}")
    print(f"  peak allocated MiB:    {stats.peak_allocated_mb:.3f}")
    print(f"  peak reserved MiB:     {stats.peak_reserved_mb:.3f}")


def run_local_branch(
    data: CharacterData,
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    full_reference: FullReference,
    local_anchor_checkpoint_path: Path,
    reachability: ReachabilityAudit,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    on_update=None,
) -> LocalTrainingReport:
    """Train only the global-RF local branch; Full remains pinned read-only."""

    clear_accelerator_cache(device)
    model = build_model(data, config, local, n_head, n_layer, device)
    model.load_state_dict(initialization.local, strict=True)
    verify_model_mask(model, local)
    if tensor_mapping_sha256(model.state_dict()) != initialization.local_state_sha256:
        raise RuntimeError("local branch did not load its exact initial state")
    if (
        tensor_mapping_sha256(clone_parameters(model))
        != initialization.parameter_state_sha256
    ):
        raise RuntimeError("local learned initialization changed")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.for_update(0),
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    expected_shape = (
        config.batch_size,
        n_head,
        config.block_size,
        config.block_size,
    )
    local_mask = build_attention_mask(config.block_size, local.max_context)
    full_mask = build_attention_mask(config.block_size, full.max_context)
    best = LocalValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=local.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        local_spec=local,
        full_spec=full,
        schedule=schedule,
        target_step=config.max_iters,
        initialization_seed=config.seed,
        branch_initial_state_sha256=initialization.local_state_sha256,
        full_initial_state_sha256=initialization.full_state_sha256,
        parameter_state_sha256=initialization.parameter_state_sha256,
        local_mask_sha256=initialization.local_mask_sha256,
        full_mask_sha256=initialization.full_mask_sha256,
        source_common_t64_state_sha256=(initialization.source_common_t64_state_sha256),
        source_extra_position_rows_sha256=(
            initialization.source_extra_position_rows_sha256
        ),
        source_shared_parameter_sha256=(initialization.source_shared_parameter_sha256),
        full_reference_sha256=full_reference.checkpoint_sha256,
        full_reference_step=full_reference.checkpoint_step,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        attention_shape=expected_shape,
        bands=bands,
        reachability=reachability,
        allowed_edges_per_head=int(local_mask.sum().item()),
        removed_edges_per_head=int(full_mask.sum().item() - local_mask.sum().item()),
        anchor_path=local_anchor_checkpoint_path,
        anchor_step=full_reference.checkpoint_step,
    )
    record_evaluation = _STAGE_18.make_evaluation_callback(best, schedule)

    print(
        f"\n{local.name} branch (B={config.batch_size}, T={config.block_size}, "
        f"W={local.max_context}, RF={reachability.effective_receptive_field_width}, "
        f"dropout={DEFAULT_DROPOUT:g})"
    )
    print(f"  branch initialization: {initialization.local_state_sha256}")
    print(f"  learned parameters:    {initialization.parameter_state_sha256}")
    print(f"  training-batch seed:   {training_batch_seed}")
    print(f"  training RNG seed:     {training_rng_seed}")
    print(f"  matched anchor step:   {full_reference.checkpoint_step}")

    seed_everything(training_rng_seed)
    result = _STAGE_18.train_with_schedule(
        model,
        optimizer,
        data,
        config,
        device,
        training_generator,
        schedule,
        bands,
        on_evaluation=record_evaluation,
        on_update=on_update,
    )
    if not best.anchor_saved:
        raise RuntimeError(
            f"training did not save the predeclared step-{best.anchor_step} anchor"
        )
    if (
        best.best_step is None
        or best.best_training_losses is None
        or best.best_validation_losses is None
        or best.anchor_training_losses is None
        or best.anchor_validation_losses is None
    ):
        raise RuntimeError("Stage 29 checkpoint selection is incomplete")

    attention_shape = inspect_attention_shape(
        model,
        config,
        local,
        n_head,
        device,
    )
    final_generator_hash = tensor_sha256(training_generator.get_state())

    selected_checkpoint = _require_mapping(
        torch.load(local.checkpoint_path, map_location="cpu", weights_only=True),
        name="selected local checkpoint",
    )
    selected_state = _require_mapping(
        selected_checkpoint.get("model_state_dict"),
        name="selected local model_state_dict",
    )
    model.load_state_dict(selected_state, strict=True)
    verify_model_mask(model, local)
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=DEFAULT_SAMPLE_SEED,
    )
    report = LocalTrainingReport(
        spec=local,
        parameter_count=parameter_count,
        initial_state_sha256=initialization.local_state_sha256,
        parameter_state_sha256=initialization.parameter_state_sha256,
        training=result,
        best_training_losses=best.best_training_losses,
        best_validation_losses=best.best_validation_losses,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        anchor_training_losses=best.anchor_training_losses,
        anchor_validation_losses=best.anchor_validation_losses,
        anchor_step=best.anchor_step,
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


def _print_experiment_summary(
    config: TrainingConfig,
    report: ExperimentReport,
    local_anchor_checkpoint_path: Path,
    full_sample: str,
) -> None:
    local_report = report.local
    full_reference = report.full
    matched = (
        full_reference.best_validation_losses.overall
        - local_report.anchor_validation_losses.overall
    )
    selected = (
        full_reference.best_validation_losses.overall
        - local_report.best_validation_losses.overall
    )
    print(f"\nStage {EXPERIMENT_STAGE} experimental summary")
    print(f"  both branches B/T:            {config.batch_size}/{config.block_size}")
    print(
        "  local direct / RF theory/effective: "
        f"{report.reachability.direct_attention_span} / "
        f"{report.reachability.theoretical_receptive_field_width} / "
        f"{report.reachability.effective_receptive_field_width}"
    )
    print(
        "  all causal sources reachable: "
        f"{report.reachability.all_causal_sources_reachable}"
    )
    print("  hard context reset:            none")
    print(f"  parameter count (both):       {local_report.parameter_count:,}")
    print(f"  attention shape (both):       {local_report.attention_shape}")
    print(
        "  local best train/val/gap:     "
        f"{local_report.best_training_losses.overall:.4f} / "
        f"{local_report.best_validation_losses.overall:.4f} / "
        f"{local_report.best_gap:.4f} @ {local_report.best_step}"
    )
    print(
        "  full reference train / val:   "
        f"{full_reference.best_training_losses.overall:.4f} / "
        f"{full_reference.best_validation_losses.overall:.4f} "
        f"@ {full_reference.checkpoint_step}"
    )
    print(f"  matched full-local delta:      {matched:+.4f}")
    print(f"  selected full-local delta:     {selected:+.4f}")
    print("  negative deltas favor Full-128")
    print(f"  local best checkpoint:         {local_report.spec.checkpoint_path}")
    print(f"  local best SHA-256:            {report.local_checkpoint_sha256}")
    print(f"  local matched anchor:          {local_anchor_checkpoint_path}")
    print(f"  local anchor SHA-256:          {report.local_anchor_checkpoint_sha256}")
    print(f"  full read-only reference:      {full_reference.spec.checkpoint_path}")
    print(f"\n{local_report.spec.name} selected-best generated text")
    print(local_report.sample)
    print(f"\n{full_reference.spec.name} selected-best generated text")
    print(full_sample)


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
    local_anchor_checkpoint_path: Path,
    *,
    training_batch_seed: int = DEFAULT_TRAINING_BATCH_SEED,
    training_rng_seed: int = DEFAULT_TRAINING_RNG_SEED,
    full_reference_expected_sha256: str = (DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256),
    full_reference_expected_step: int = DEFAULT_FULL_REFERENCE_STEP,
    benchmark: BenchmarkConfig | None = None,
    on_update=None,
) -> ExperimentReport:
    """Run the controlled one-new-branch Stage 29 protocol."""

    if sample_length < 0:
        raise ValueError("sample_length must be non-negative")
    schedule.validate_target_step(config.max_iters)
    _, bands = validate_attention_protocol(
        config,
        local,
        full,
        n_head,
        n_layer,
        anchor_checkpoint_path=local_anchor_checkpoint_path,
    )
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError("config learning rate must equal schedule initial LR")
    for split, source in (("train", data.train_data), ("val", data.val_data)):
        if len(source) <= config.block_size:
            raise ValueError(
                f"the {split} split has {len(source)} tokens, but block_size "
                f"is {config.block_size}"
            )
    if benchmark is not None:
        required = benchmark.num_warmup + benchmark.num_steps
        if required > config.max_iters:
            raise ValueError(
                f"benchmark requires {required} steps, but max_iters is "
                f"{config.max_iters}"
            )

    initialization = create_paired_initial_states(
        data,
        config,
        local,
        full,
        n_head,
        n_layer,
    )
    local_mask = build_attention_mask(config.block_size, local.max_context)
    full_mask = build_attention_mask(config.block_size, full.max_context)
    reachability = audit_bounded_reachability(local_mask, n_layer)
    verify_global_reachability(reachability, config.block_size)
    removed_edges = int(full_mask.sum().item() - local_mask.sum().item())
    if removed_edges <= 0:
        raise RuntimeError("Stage 29 local mask must remove full-attention edges")

    full_reference = validate_stage_26_full_reference(
        full.checkpoint_path,
        data,
        config,
        full,
        n_head,
        n_layer,
        schedule,
        initialization,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_checkpoint_sha256=full_reference_expected_sha256,
        expected_step=full_reference_expected_step,
    )
    reference_hash = full_reference.checkpoint_sha256
    if not 0 <= full_reference.checkpoint_step <= config.max_iters:
        raise ValueError("full reference step is outside the Stage 29 schedule")
    if (
        full_reference.checkpoint_step % config.eval_interval != 0
        and full_reference.checkpoint_step != schedule.first_decay_step
        and full_reference.checkpoint_step != schedule.second_decay_step
        and full_reference.checkpoint_step != config.max_iters
    ):
        raise ValueError("full reference step is not a Stage 29 evaluation step")

    print("\nControlled global-RF local-routing protocol")
    print("  The Full-128 Stage 26 trajectory is pinned and reused read-only.")
    print(
        "  Only the continuous global-RF local "
        f"W={local.max_context} branch is trained from step zero."
    )
    print("  Initial states differ only in every head's mask buffer.")
    print(f"  learned parameter SHA-256: {initialization.parameter_state_sha256}")
    print(f"  local initial SHA-256:     {initialization.local_state_sha256}")
    print(f"  full initial SHA-256:      {initialization.full_state_sha256}")
    print(f"  local mask SHA-256:        {initialization.local_mask_sha256}")
    print(f"  full mask SHA-256:         {initialization.full_mask_sha256}")
    print(f"  full checkpoint SHA-256:   {reference_hash}")
    print(f"  full checkpoint step:      {full_reference.checkpoint_step}")
    print(
        f"  edges/head: local={int(local_mask.sum().item()):,}; "
        f"full={int(full_mask.sum().item()):,}; removed={removed_edges:,}"
    )
    print(
        "  RF widths after layers:    "
        + ", ".join(str(width) for width in reachability.widths_by_layer)
    )
    print(
        f"  all causal sources reachable at layer {n_layer}: "
        f"{reachability.all_causal_sources_reachable}"
    )
    print(
        "  first query requiring multiple local hops: "
        f"{local.max_context}"
    )
    print("  continuous predecessor access: proven; no reset")

    local_audit_model = build_model(
        data,
        config,
        local,
        n_head,
        n_layer,
        torch.device("cpu"),
    )
    full_audit_model = build_model(
        data,
        config,
        full,
        n_head,
        n_layer,
        torch.device("cpu"),
    )
    local_audit_model.load_state_dict(initialization.local, strict=True)
    full_audit_model.load_state_dict(initialization.full, strict=True)
    initial_difference = verify_initial_shared_prefix_equivalence(
        local_audit_model,
        full_audit_model,
        data,
        config,
        torch.device("cpu"),
        seed=DEFAULT_INITIAL_AUDIT_SEED,
        prefix_length=local.max_context,
    )
    del local_audit_model
    del full_audit_model
    print(
        f"  Initial [0,{local.max_context}) max logit difference: "
        f"{initial_difference:.3e}"
    )

    local_benchmark: BenchmarkStats | None = None
    full_benchmark: BenchmarkStats | None = None
    if benchmark is not None:
        print("\nCounterbalanced disposable dense training benchmark")
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
        if checkpoint_sha256(full.checkpoint_path) != reference_hash:
            raise RuntimeError("full reference changed during benchmark")

    local_report = run_local_branch(
        data,
        config,
        local,
        full,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        bands,
        initialization,
        full_reference,
        local_anchor_checkpoint_path,
        reachability,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        on_update=on_update,
    )
    local_report = replace(local_report, benchmark=local_benchmark)
    if checkpoint_sha256(full.checkpoint_path) != reference_hash:
        raise RuntimeError("full reference changed while training local branch")

    validate_stage_29_checkpoint(
        local.checkpoint_path,
        data,
        config,
        local,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        initialization,
        full_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        checkpoint_kind="best",
    )
    anchor_checkpoint = validate_stage_29_checkpoint(
        local_anchor_checkpoint_path,
        data,
        config,
        local,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        initialization,
        full_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        checkpoint_kind="anchor",
    )
    assert_matched_training_stream_state(
        full_reference.checkpoint,
        anchor_checkpoint,
    )
    print("  Matched anchor batch and RNG states: identical")
    local_checkpoint_hash = checkpoint_sha256(local.checkpoint_path)
    local_anchor_checkpoint_hash = checkpoint_sha256(local_anchor_checkpoint_path)

    full_model = build_model(data, config, full, n_head, n_layer, device)
    full_state = _require_mapping(
        full_reference.checkpoint.get("model_state_dict"),
        name="full reference model_state_dict",
    )
    full_model.load_state_dict(full_state, strict=True)
    verify_model_mask(full_model, full)
    full_sample = generate_from_final_model(
        full_model,
        data,
        device,
        sample_length=sample_length,
        seed=DEFAULT_SAMPLE_SEED,
    )
    del full_model
    clear_accelerator_cache(device)
    if checkpoint_sha256(full.checkpoint_path) != reference_hash:
        raise RuntimeError("full reference changed during generation")

    report = ExperimentReport(
        initialization=initialization,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        bands=bands,
        reachability=reachability,
        local=local_report,
        full=full_reference,
        initial_shared_prefix_length=local.max_context,
        initial_shared_prefix_max_difference=initial_difference,
        local_benchmark=local_benchmark,
        full_benchmark=full_benchmark,
        local_checkpoint_sha256=local_checkpoint_hash,
        local_anchor_checkpoint_sha256=local_anchor_checkpoint_hash,
    )
    _print_experiment_summary(
        config,
        report,
        local_anchor_checkpoint_path,
        full_sample,
    )
    return report


ValidationPanel = tuple[torch.Tensor, torch.Tensor]


def make_validation_panels(
    data: CharacterData,
    config: TrainingConfig,
    *,
    eval_iters: int,
    seed: int,
) -> tuple[ValidationPanel, ...]:
    return _STAGE_27.make_validation_panels(
        data,
        config,
        eval_iters=eval_iters,
        seed=seed,
    )


@torch.no_grad()
def _evaluate_natural_on_panels(
    model: GPTLanguageModel,
    panels: Sequence[ValidationPanel],
    device: torch.device,
    bands: Sequence[tuple[int, int]],
) -> tuple[PositionLosses, ...]:
    if not panels:
        raise ValueError("at least one validation panel is required")
    was_training = model.training
    model.eval()
    values: list[PositionLosses] = []
    try:
        for cpu_inputs, cpu_targets in panels:
            inputs = cpu_inputs.to(device)
            targets = cpu_targets.to(device)
            logits, _ = model(inputs)
            values.append(
                _STAGE_18._as_position_losses(
                    _STAGE_18.position_cross_entropy(logits, targets, bands)
                )
            )
            del inputs
            del targets
            del logits
    finally:
        model.train(was_training)
    return tuple(values)


def _precise_result(
    *,
    name: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    checkpoint_step: int,
    fixed_panel_loss: float,
    batch_losses: tuple[PositionLosses, ...],
) -> PreciseBranchResult:
    return _STAGE_27._precise_result(
        name=name,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        checkpoint_step=checkpoint_step,
        fixed_panel_loss=fixed_panel_loss,
        batch_losses=batch_losses,
    )


def _evaluate_full_natural(
    data: CharacterData,
    config: TrainingConfig,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    panels: Sequence[ValidationPanel],
    reference: FullReference,
) -> PreciseBranchResult:
    model = build_model(data, config, full, n_head, n_layer, device)
    state = _require_mapping(
        reference.checkpoint.get("model_state_dict"),
        name="full reference model_state_dict",
    )
    model.load_state_dict(state, strict=True)
    verify_model_mask(model, full)
    losses = _evaluate_natural_on_panels(model, panels, device, bands)
    verify_model_mask(model, full)
    del model
    clear_accelerator_cache(device)
    return _precise_result(
        name="full-trained/full-mask",
        checkpoint_path=full.checkpoint_path,
        checkpoint_hash=reference.checkpoint_sha256,
        checkpoint_step=reference.checkpoint_step,
        fixed_panel_loss=reference.best_validation_losses.overall,
        batch_losses=losses,
    )


def _evaluate_local_natural(
    path: Path,
    checkpoint: Mapping[str, object],
    checkpoint_hash: str,
    fixed_panel_loss: float,
    data: CharacterData,
    config: TrainingConfig,
    local: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    panels: Sequence[ValidationPanel],
    *,
    endpoint_label: str,
) -> PreciseBranchResult:
    model = build_model(data, config, local, n_head, n_layer, device)
    state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name=f"{endpoint_label} local model_state_dict",
    )
    model.load_state_dict(state, strict=True)
    verify_model_mask(model, local)
    losses = _evaluate_natural_on_panels(model, panels, device, bands)
    verify_model_mask(model, local)
    del model
    clear_accelerator_cache(device)
    step = checkpoint.get("step")
    assert isinstance(step, int) and not isinstance(step, bool)
    return _precise_result(
        name=f"local-trained/local-mask ({endpoint_label})",
        checkpoint_path=path,
        checkpoint_hash=checkpoint_hash,
        checkpoint_step=step,
        fixed_panel_loss=fixed_panel_loss,
        batch_losses=losses,
    )


def position_range_mean(
    losses: PositionLosses,
    bands: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> float:
    """Combine complete partition bands into one token-weighted range mean."""

    if len(losses.bands) != len(bands):
        raise ValueError("position losses and position bands must have equal lengths")
    selected = tuple(
        (band_start, band_end, value)
        for (band_start, band_end), value in zip(bands, losses.bands, strict=True)
        if band_start >= start and band_end <= end
    )
    if (
        not selected
        or selected[0][0] != start
        or selected[-1][1] != end
        or any(
            first[1] != second[0]
            for first, second in zip(selected, selected[1:])
        )
    ):
        raise ValueError("requested range must be exactly covered by complete bands")
    width = end - start
    if width <= 0:
        raise ValueError("position range must be non-empty")
    return sum(
        (band_end - band_start) * value
        for band_start, band_end, value in selected
    ) / width


def paired_position_range_delta(
    candidate: PreciseBranchResult,
    baseline: PreciseBranchResult,
    bands: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> RangeDelta:
    """Compute a paired Student-t interval from per-panel range aggregates."""

    if len(candidate.batch_losses) != len(baseline.batch_losses):
        raise ValueError("paired validation results must have equal lengths")
    differences = tuple(
        position_range_mean(candidate_loss, bands, start, end)
        - position_range_mean(baseline_loss, bands, start, end)
        for candidate_loss, baseline_loss in zip(
            candidate.batch_losses,
            baseline.batch_losses,
            strict=True,
        )
    )
    mean_delta, standard_error = _STAGE_18._mean_and_standard_error(differences)
    critical_value = _STAGE_18._paired_t_95_critical_value(len(differences))
    return RangeDelta(
        candidate=candidate.name,
        baseline=baseline.name,
        start=start,
        end=end,
        mean_delta=mean_delta,
        standard_error=standard_error,
        confidence_low=mean_delta - critical_value * standard_error,
        confidence_high=mean_delta + critical_value * standard_error,
        critical_value=critical_value,
    )


def _make_endpoint_report(
    name: str,
    full_natural: PreciseBranchResult,
    local_natural: PreciseBranchResult,
    bands: tuple[tuple[int, int], ...],
    multi_hop_start: int,
) -> EndpointValidationReport:
    return EndpointValidationReport(
        name=name,
        full_natural=full_natural,
        local_natural=local_natural,
        natural_delta=_STAGE_18._paired_delta(full_natural, local_natural),
        multi_hop_delta=paired_position_range_delta(
            full_natural,
            local_natural,
            bands,
            multi_hop_start,
            bands[-1][1],
        ),
    )


def run_paired_validation(
    data: CharacterData,
    config: TrainingConfig,
    local: AttentionSpec,
    full: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    experiment: ExperimentReport,
    local_anchor_checkpoint_path: Path,
    *,
    eval_iters: int,
    seed: int,
) -> Stage29PreciseValidationReport:
    """Compare only naturally trained/naturally masked models on one panel."""

    _, expected_bands = validate_attention_protocol(
        config,
        local,
        full,
        n_head,
        n_layer,
        anchor_checkpoint_path=local_anchor_checkpoint_path,
    )
    if experiment.bands != expected_bands:
        raise ValueError("held-out validation bands differ from the training protocol")
    panels = make_validation_panels(
        data,
        config,
        eval_iters=eval_iters,
        seed=seed,
    )
    full_hash_before = checkpoint_sha256(full.checkpoint_path)
    if full_hash_before != experiment.full.checkpoint_sha256:
        raise RuntimeError("full reference changed before held-out validation")
    anchor_hash = checkpoint_sha256(local_anchor_checkpoint_path)
    if anchor_hash != experiment.local_anchor_checkpoint_sha256:
        raise RuntimeError(
            "local anchor differs from the artifact produced by Stage 29 training"
        )
    best_hash = checkpoint_sha256(local.checkpoint_path)
    if best_hash != experiment.local_checkpoint_sha256:
        raise RuntimeError(
            "local best differs from the artifact produced by Stage 29 training"
        )

    full_natural = _evaluate_full_natural(
        data,
        config,
        full,
        n_head,
        n_layer,
        device,
        experiment.bands,
        panels,
        experiment.full,
    )
    anchor_checkpoint = validate_stage_29_checkpoint(
        local_anchor_checkpoint_path,
        data,
        config,
        local,
        full,
        n_head,
        n_layer,
        schedule,
        experiment.bands,
        experiment.initialization,
        experiment.full,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        checkpoint_kind="anchor",
    )
    anchor_losses = _deserialize_losses(
        anchor_checkpoint.get("anchor_validation_losses"),
        experiment.bands,
        name="anchor_validation_losses",
    )
    anchor_local = _evaluate_local_natural(
        local_anchor_checkpoint_path,
        anchor_checkpoint,
        anchor_hash,
        anchor_losses.overall,
        data,
        config,
        local,
        n_head,
        n_layer,
        device,
        experiment.bands,
        panels,
        endpoint_label="matched-anchor",
    )

    best_checkpoint = validate_stage_29_checkpoint(
        local.checkpoint_path,
        data,
        config,
        local,
        full,
        n_head,
        n_layer,
        schedule,
        experiment.bands,
        experiment.initialization,
        experiment.full,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        checkpoint_kind="best",
    )
    best_local = _evaluate_local_natural(
        local.checkpoint_path,
        best_checkpoint,
        best_hash,
        float(best_checkpoint["best_val_loss"]),
        data,
        config,
        local,
        n_head,
        n_layer,
        device,
        experiment.bands,
        panels,
        endpoint_label="selected-best",
    )

    if checkpoint_sha256(full.checkpoint_path) != full_hash_before:
        raise RuntimeError("full reference changed during held-out validation")
    if checkpoint_sha256(local_anchor_checkpoint_path) != anchor_hash:
        raise RuntimeError("local anchor changed during held-out validation")
    if checkpoint_sha256(local.checkpoint_path) != best_hash:
        raise RuntimeError("local best checkpoint changed during held-out validation")

    return Stage29PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        bands=experiment.bands,
        anchor=_make_endpoint_report(
            "matched step",
            full_natural,
            anchor_local,
            experiment.bands,
            local.max_context or config.block_size,
        ),
        selected_best=_make_endpoint_report(
            "selected best",
            full_natural,
            best_local,
            experiment.bands,
            local.max_context or config.block_size,
        ),
    )


def _print_variant(result: PreciseBranchResult) -> None:
    _STAGE_27._print_variant(result)


def _print_delta(
    label: str,
    delta: PreciseDelta,
    bands: Sequence[tuple[int, int]],
) -> None:
    _STAGE_27._print_delta(label, delta, bands)


def print_paired_validation(report: Stage29PreciseValidationReport) -> None:
    print(f"\nHeld-out natural-mask paired validation of Stage {EXPERIMENT_STAGE}")
    print(
        f"  {report.eval_iters} shared validation panels, seed={report.seed}; "
        "no crossed masks"
    )
    for endpoint in (report.anchor, report.selected_best):
        print(f"\n{endpoint.name} endpoint")
        for result in (endpoint.full_natural, endpoint.local_natural):
            _print_variant(result)
            for (start, end), band_mean, band_se in zip(
                report.bands,
                result.statistics.mean.bands,
                result.statistics.standard_error.bands,
                strict=True,
            ):
                print(
                    f"    positions {start:03d}-{end - 1:03d}: "
                    f"{band_mean:.6f} +/- {band_se:.6f} SE"
                )
        _print_delta(
            "natural full-minus-local",
            endpoint.natural_delta,
            report.bands,
        )
        diagnostic = endpoint.multi_hop_delta
        print(
            "  predeclared multi-hop aggregate "
            f"[{diagnostic.start},{diagnostic.end}): "
            f"{diagnostic.mean_delta:+.6f} +/- "
            f"{diagnostic.standard_error:.6f} SE; 95% CI "
            f"[{diagnostic.confidence_low:+.6f}, "
            f"{diagnostic.confidence_high:+.6f}]"
        )
        print("  Negative deltas favor Full-128.")
    print(
        "\n  The matched-step natural overall endpoint is primary. Its "
        "predeclared "
        f"[{report.anchor.multi_hop_delta.start},"
        f"{report.anchor.multi_hop_delta.end}) multi-hop aggregate is the mechanistic "
        "path-length/direct-fan-in diagnostic."
    )
    print(
        "  Selected-best and remaining position-band intervals are secondary "
        "or exploratory, pointwise, and unadjusted for multiplicity."
    )
    print(
        "  Both branches can theoretically reach every causal source by the "
        "final layer. A Full win can reflect direct fan-in, shorter paths, "
        "or reduced relay bottlenecks; this mask does not separate them."
    )


# A familiar alias for callers that use the Stage 27 print name.
print_precise_validation = print_paired_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 29: train a continuous W=23/L=6 global-RF local branch and "
            "compare it with the pinned Stage 26 Full-128 checkpoint."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument(
        "--precise-eval-iters",
        type=int,
        default=DEFAULT_PRECISE_EVAL_ITERS,
        help="Held-out paired panels after training; 0 skips, otherwise at least 2.",
    )
    parser.add_argument(
        "--precise-eval-seed",
        type=int,
        default=DEFAULT_PRECISE_EVAL_SEED,
    )
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument(
        "--local-checkpoint-path",
        type=Path,
        default=DEFAULT_LOCAL_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--local-anchor-checkpoint-path",
        type=Path,
        default=DEFAULT_LOCAL_ANCHOR_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--full-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_FULL_REFERENCE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if args.precise_eval_iters < 0 or args.precise_eval_iters == 1:
        raise ValueError("precise-eval-iters must be 0 or at least 2")
    if args.precise_eval_seed < 0:
        raise ValueError("precise-eval-seed must be non-negative")
    if args.precise_eval_seed == DEFAULT_FIXED_PANEL_SEED:
        raise ValueError("precise-eval-seed must differ from fixed-panel seed")
    if args.benchmark_warmup < 0:
        raise ValueError("benchmark-warmup must be non-negative")
    if args.benchmark_steps < 0:
        raise ValueError("benchmark-steps must be non-negative")

    schedule = LearningRateSchedule(
        initial_learning_rate=DEFAULT_INITIAL_LEARNING_RATE,
        first_decay_step=DEFAULT_FIRST_DECAY_STEP,
        middle_learning_rate=DEFAULT_MIDDLE_LEARNING_RATE,
        second_decay_step=DEFAULT_SECOND_DECAY_STEP,
        final_learning_rate=DEFAULT_FINAL_LEARNING_RATE,
    )
    config = TrainingConfig(
        batch_size=DEFAULT_BATCH_SIZE,
        block_size=DEFAULT_BLOCK_SIZE,
        n_embd=DEFAULT_N_EMBD,
        learning_rate=schedule.initial_learning_rate,
        max_iters=DEFAULT_MAX_ITERS,
        eval_interval=DEFAULT_EVAL_INTERVAL,
        eval_iters=DEFAULT_EVAL_ITERS,
        seed=DEFAULT_SEED,
    )
    local = AttentionSpec(
        name="continuous global-RF local window (W=23, RF=133)",
        max_context=DEFAULT_LOCAL_CONTEXT,
        checkpoint_path=args.local_checkpoint_path,
    )
    full = AttentionSpec(
        name="full-128 Stage 26 reference",
        max_context=None,
        checkpoint_path=args.full_reference_checkpoint_path,
    )
    validate_attention_protocol(
        config,
        local,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        anchor_checkpoint_path=args.local_anchor_checkpoint_path,
    )
    schedule.validate_target_step(config.max_iters)

    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)
    if not _is_canonical_protocol(
        data,
        config,
        local,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
    ):
        raise RuntimeError("the Stage 29 CLI is reserved for the canonical protocol")
    print(f"Stage {EXPERIMENT_STAGE}: {EXPERIMENT_TITLE}")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    elif device.type == "cuda":
        print(f"Accelerator: {torch.cuda.get_device_name(device)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Both: B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={DEFAULT_N_HEAD}, "
        f"D={config.n_embd // DEFAULT_N_HEAD}, FF={4 * config.n_embd}, "
        f"L={DEFAULT_N_LAYER}, dropout={DEFAULT_DROPOUT:g}"
    )
    print(
        f"Masks: local = continuous causal W={DEFAULT_LOCAL_CONTEXT} in every "
        f"layer (RF={DEFAULT_THEORETICAL_RECEPTIVE_FIELD}); reference = "
        "ordinary Full-128 causal attention"
    )
    print(
        "Schedule by zero-based update: "
        f"[0, {schedule.first_decay_step:,}) "
        f"lr={schedule.initial_learning_rate:g}; "
        f"[{schedule.first_decay_step:,}, {schedule.second_decay_step:,}) "
        f"lr={schedule.middle_learning_rate:g}; "
        f"[{schedule.second_decay_step:,}, {config.max_iters:,}) "
        f"lr={schedule.final_learning_rate:g}"
    )
    print(
        "Seeds: initialization=1337, batches=1338, fixed=1339, "
        f"training RNG=1340, held-out={args.precise_eval_seed}"
    )

    benchmark = None
    if args.benchmark_steps > 0:
        benchmark = BenchmarkConfig(
            num_warmup=args.benchmark_warmup,
            num_steps=args.benchmark_steps,
        )
    experiment = run_experiment(
        data,
        config,
        local,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        device,
        args.sample_length,
        schedule,
        args.local_anchor_checkpoint_path,
        training_batch_seed=DEFAULT_TRAINING_BATCH_SEED,
        training_rng_seed=DEFAULT_TRAINING_RNG_SEED,
        full_reference_expected_sha256=(DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256),
        full_reference_expected_step=DEFAULT_FULL_REFERENCE_STEP,
        benchmark=benchmark,
    )
    if args.precise_eval_iters > 0:
        precise = run_paired_validation(
            data,
            config,
            local,
            full,
            DEFAULT_N_HEAD,
            DEFAULT_N_LAYER,
            device,
            schedule,
            experiment,
            args.local_anchor_checkpoint_path,
            eval_iters=args.precise_eval_iters,
            seed=args.precise_eval_seed,
        )
        print_paired_validation(precise)
    if (
        checkpoint_sha256(full.checkpoint_path)
        != DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Stage 26 full reference changed during Stage 29")


if __name__ == "__main__":
    main()
