from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
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


# Stage 27 uses Stage 18 only for its well-tested position-loss, training-loop,
# and paired-statistics primitives. Its sliding-window mask is deliberately not
# reused: a six-layer local window can relay information across the midpoint.
_STAGE_18_PATH = Path(__file__).with_name("18_attention_span.py")
_STAGE_18_MODULE_NAME = "stage_18_attention_span_for_stage_27"
_STAGE_18_SPEC = importlib.util.spec_from_file_location(
    _STAGE_18_MODULE_NAME,
    _STAGE_18_PATH,
)
assert _STAGE_18_SPEC is not None and _STAGE_18_SPEC.loader is not None
_STAGE_18 = importlib.util.module_from_spec(_STAGE_18_SPEC)
sys.modules[_STAGE_18_MODULE_NAME] = _STAGE_18
_STAGE_18_SPEC.loader.exec_module(_STAGE_18)


# Stage 26 owns the exact T=128 initialization and the validator for the
# immutable full-context reference checkpoint.
_STAGE_26_PATH = Path(__file__).with_name("26_context_length.py")
_STAGE_26_MODULE_NAME = "stage_26_context_length_for_stage_27"
_STAGE_26_SPEC = importlib.util.spec_from_file_location(
    _STAGE_26_MODULE_NAME,
    _STAGE_26_PATH,
)
assert _STAGE_26_SPEC is not None and _STAGE_26_SPEC.loader is not None
_STAGE_26 = importlib.util.module_from_spec(_STAGE_26_SPEC)
sys.modules[_STAGE_26_MODULE_NAME] = _STAGE_26
_STAGE_26_SPEC.loader.exec_module(_STAGE_26)


GPTLanguageModel = _STAGE_26.GPTLanguageModel
LearningRateSchedule = _STAGE_26.LearningRateSchedule
BestValidationCheckpoint = _STAGE_18.BestValidationCheckpoint
PositionLosses = _STAGE_18.PositionLosses
PositionLossEstimate = _STAGE_18.PositionLossEstimate
AttentionEvaluationRecord = _STAGE_18.AttentionEvaluationRecord
AttentionTrainingResult = _STAGE_18.AttentionTrainingResult
PositionStatistics = _STAGE_18.PositionStatistics
PreciseBranchResult = _STAGE_18.PreciseBranchResult
PreciseDelta = _STAGE_18.PreciseDelta

checkpoint_sha256 = _STAGE_26.checkpoint_sha256
clear_accelerator_cache = _STAGE_26.clear_accelerator_cache
fingerprint_data = _STAGE_26.fingerprint_data
generate_from_final_model = _STAGE_26.generate_from_final_model
tensor_mapping_sha256 = _STAGE_26.tensor_mapping_sha256
tensor_sha256 = _STAGE_26.tensor_sha256


EXPERIMENT_STAGE = 27
EXPERIMENT_TITLE = "hard cross-half context isolation"
COMPARISON_VARIABLE = "cross_half_information_access"

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_BLOCK_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_27_block_64_at_128_best_checkpoint.pt"
)
DEFAULT_BLOCK_ANCHOR_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_27_block_64_at_128_step_13500_checkpoint.pt"
)
DEFAULT_FULL_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_26_context_128_best_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = 16
DEFAULT_BLOCK_SIZE = 128
DEFAULT_BOUNDARY = 64
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
# This panel is new relative to Stage 26's seed-1,344 fresh comparison.
DEFAULT_PRECISE_EVAL_SEED = 1_345
DEFAULT_ACCELERATOR_NAME = "Intel(R) Arc(TM) 140T GPU (16GB)"

DROPOUT_PLACEMENT = _STAGE_26.DROPOUT_PLACEMENT
DEFAULT_DATA_FINGERPRINT = _STAGE_26.DEFAULT_DATA_FINGERPRINT
DEFAULT_PARAMETER_COUNT = _STAGE_26.DEFAULT_TREATMENT_PARAMETER_COUNT
DEFAULT_SOURCE_FULL_INITIAL_STATE_SHA256 = (
    _STAGE_26.DEFAULT_TREATMENT_INITIAL_STATE_SHA256
)
DEFAULT_SOURCE_COMMON_T64_STATE_SHA256 = (
    _STAGE_26.DEFAULT_CONTROL_INITIAL_STATE_SHA256
)
DEFAULT_SOURCE_EXTRA_POSITION_ROWS_SHA256 = (
    _STAGE_26.DEFAULT_EXTRA_POSITION_ROWS_SHA256
)
DEFAULT_SOURCE_SHARED_PARAMETER_SHA256 = (
    _STAGE_26.DEFAULT_SHARED_PARAMETER_INITIALIZATION_SHA256
)
DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256 = (
    "3f2ba55ab544569351e4e4b5f01432c9967f5d7629598ff6483db55c17b00be6"
)
DEFAULT_FULL_REFERENCE_STEP = 13_500

DEFAULT_FULL_MASK_SHA256 = (
    "91fc1df848a40db0decca26c27935e4ae74bc4ec8a09f3bf8ee6dda243dceabd"
)
DEFAULT_BLOCK_MASK_SHA256 = (
    "0da03313111bb460fa08e2bb07657accda4630d1baf59dcda25d7127fca141b8"
)
DEFAULT_FULL_FLOAT_MASK_SHA256 = (
    "6e0ea88e6932f1ff6b1d3eea15e7b360115e87e91a363587c8be24bca9944c12"
)
DEFAULT_BLOCK_FLOAT_MASK_SHA256 = (
    "ab46ba5c6548262515aba0dd213f127909d1b8d4d5070f90c859e1b3ab32119d"
)
DEFAULT_PARAMETER_INITIALIZATION_SHA256 = (
    "c29ae7a57c7db105f572b58c17578aea86564c7c9ae8dfb203e3179e61600477"
)
DEFAULT_FULL_INITIAL_STATE_SHA256 = (
    "c27f642dfb3ba186430d1f553b14a30fe3ce90c5062e9b2fe0ff448e88b6c8e8"
)
DEFAULT_BLOCK_INITIAL_STATE_SHA256 = (
    "f2cd4762e4bc96f0086df2f2cbff0e364ddf06f66b67aac4b5a24e9e530f686d"
)


@dataclass(frozen=True, slots=True)
class CrossHalfSpec:
    """One Stage 27 mask policy over a common T=128 model geometry."""

    name: str
    allow_cross_half: bool
    boundary: int
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch name must not be empty")
        if not isinstance(self.allow_cross_half, bool):
            raise ValueError("allow_cross_half must be a bool")
        if (
            isinstance(self.boundary, bool)
            or not isinstance(self.boundary, int)
            or self.boundary <= 0
        ):
            raise ValueError(
                f"boundary must be a positive integer, got {self.boundary!r}"
            )

    @property
    def mask_kind(self) -> str:
        if self.allow_cross_half:
            return "full_causal"
        return "causal_block_diagonal"


@dataclass(frozen=True, slots=True)
class Stage27InitialStates:
    block: dict[str, torch.Tensor]
    full: dict[str, torch.Tensor]
    parameter_state_sha256: str
    block_state_sha256: str
    full_state_sha256: str
    block_mask_sha256: str
    full_mask_sha256: str
    block_float_mask_sha256: str
    full_float_mask_sha256: str
    parameter_count: int
    differing_state_names: tuple[str, ...]
    source_common_t64_state_sha256: str
    source_extra_position_rows_sha256: str
    source_shared_parameter_sha256: str


@dataclass(frozen=True, slots=True)
class FullReference:
    spec: CrossHalfSpec
    checkpoint_sha256: str
    checkpoint_step: int
    best_training_losses: PositionLosses
    best_validation_losses: PositionLosses
    checkpoint: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BlockTrainingReport:
    spec: CrossHalfSpec
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
        return (
            self.best_validation_losses.overall
            - self.best_training_losses.overall
        )


@dataclass(frozen=True, slots=True)
class Stage27ExperimentReport:
    initialization: Stage27InitialStates
    initialization_seed: int
    training_batch_seed: int
    training_rng_seed: int
    schedule: LearningRateSchedule
    bands: tuple[tuple[int, int], ...]
    block: BlockTrainingReport
    full: FullReference
    initial_first_half_max_difference: float
    block_benchmark: BenchmarkStats | None
    full_benchmark: BenchmarkStats | None
    block_checkpoint_sha256: str
    block_anchor_checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class EndpointValidationReport:
    name: str
    full_natural: PreciseBranchResult
    full_blocked: PreciseBranchResult
    block_natural: PreciseBranchResult
    block_opened: PreciseBranchResult
    natural_delta: PreciseDelta
    full_weight_mask_delta: PreciseDelta
    block_weight_mask_delta: PreciseDelta
    full_weight_first_half_exact: bool
    block_weight_first_half_exact: bool


@dataclass(frozen=True, slots=True)
class Stage27PreciseValidationReport:
    eval_iters: int
    seed: int
    bands: tuple[tuple[int, int], ...]
    anchor: EndpointValidationReport
    selected_best: EndpointValidationReport


def build_cross_half_attention_mask(
    block_size: int,
    boundary: int,
    *,
    allow_cross_half: bool,
) -> torch.Tensor:
    """Return full causal attention or two causally isolated halves."""

    for value, name in ((block_size, "block_size"), (boundary, "boundary")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if block_size != 2 * boundary:
        raise ValueError(
            f"block_size must equal twice boundary, got {block_size} and {boundary}"
        )
    if not isinstance(allow_cross_half, bool):
        raise ValueError("allow_cross_half must be a bool")

    positions = torch.arange(block_size)
    query_position = positions[:, None]
    key_position = positions[None, :]
    allowed = key_position <= query_position
    if not allow_cross_half:
        allowed &= (query_position // boundary) == (key_position // boundary)
    return allowed


def make_position_bands(
    block_size: int,
    boundary: int,
) -> tuple[tuple[int, int], ...]:
    """Use two first-half bands and four finer context-sensitive bands."""

    if block_size != 2 * boundary or boundary % 2 or boundary % 4:
        raise ValueError(
            "Stage 27 requires block_size=2*boundary and boundary divisible by four"
        )
    first_width = boundary // 2
    second_width = boundary // 4
    return (
        (0, first_width),
        (first_width, boundary),
        (boundary, boundary + second_width),
        (boundary + second_width, boundary + 2 * second_width),
        (boundary + 2 * second_width, boundary + 3 * second_width),
        (boundary + 3 * second_width, block_size),
    )


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


def apply_attention_mask(
    model: GPTLanguageModel,
    mask: torch.Tensor,
) -> None:
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
    spec: CrossHalfSpec,
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
    mask = build_cross_half_attention_mask(
        config.block_size,
        spec.boundary,
        allow_cross_half=spec.allow_cross_half,
    )
    apply_attention_mask(model, mask)
    if device is not None:
        model = model.to(device)
    return model


def verify_model_mask(model: GPTLanguageModel, spec: CrossHalfSpec) -> None:
    expected_bool = build_cross_half_attention_mask(
        model.block_size,
        spec.boundary,
        allow_cross_half=spec.allow_cross_half,
    )
    observed_count = 0
    for block in model.blocks:
        for head in block.sa.heads:
            observed_count += 1
            observed = head.tril.detach().cpu()
            expected = expected_bool.to(dtype=observed.dtype)
            if not torch.equal(observed, expected):
                raise RuntimeError(
                    f"{spec.name} contains an incorrect attention-mask buffer"
                )
    if observed_count == 0:
        raise RuntimeError("model contains no attention heads")


@contextmanager
def temporary_attention_mask(
    model: GPTLanguageModel,
    mask: torch.Tensor,
):
    """Apply an inference-only mask and restore every buffer on exit."""

    originals = [
        head.tril.detach().clone()
        for block in model.blocks
        for head in block.sa.heads
    ]
    try:
        apply_attention_mask(model, mask)
        yield model
    finally:
        index = 0
        for block in model.blocks:
            for head in block.sa.heads:
                head.tril.copy_(originals[index])
                index += 1


def _stage_26_configs(
    config: TrainingConfig,
    boundary: int,
) -> tuple[TrainingConfig, TrainingConfig]:
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
        block_size=boundary,
        **common,
    )
    treatment = TrainingConfig(
        batch_size=config.batch_size,
        block_size=config.block_size,
        **common,
    )
    return control, treatment


def create_stage_27_initial_states(
    data: CharacterData,
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
) -> Stage27InitialStates:
    """Regenerate Stage 26 T=128 and change only the 24 mask buffers."""

    validate_cross_half_protocol(config, block, full, n_head, n_layer)
    control_config, treatment_config = _stage_26_configs(config, block.boundary)
    source = _STAGE_26.create_paired_initial_states(
        data,
        control_config,
        treatment_config,
        n_head,
        n_layer,
    )
    full_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in source.treatment.items()
    }
    block_state = {name: tensor.clone() for name, tensor in full_state.items()}
    full_mask = build_cross_half_attention_mask(
        config.block_size,
        block.boundary,
        allow_cross_half=True,
    )
    block_mask = build_cross_half_attention_mask(
        config.block_size,
        block.boundary,
        allow_cross_half=False,
    )
    mask_names = _mask_state_names(full_state)
    for name in mask_names:
        full_state[name] = full_mask.to(dtype=full_state[name].dtype)
        block_state[name] = block_mask.to(dtype=block_state[name].dtype)

    differing_names = tuple(
        sorted(
            name
            for name in full_state
            if not torch.equal(block_state[name], full_state[name])
        )
    )
    if differing_names != mask_names:
        raise RuntimeError(
            "Stage 27 initial states must differ only in attention masks; "
            f"got {differing_names}"
        )

    full_model = build_model(data, config, full, n_head, n_layer)
    block_model = build_model(data, config, block, n_head, n_layer)
    full_model.load_state_dict(full_state, strict=True)
    block_model.load_state_dict(block_state, strict=True)
    full_parameters = clone_parameters(full_model)
    block_parameters = clone_parameters(block_model)
    if full_parameters.keys() != block_parameters.keys() or any(
        not torch.equal(full_parameters[name], block_parameters[name])
        for name in full_parameters
    ):
        raise RuntimeError("Stage 27 learned initializations are not identical")
    if any(
        full_parameters[name].data_ptr() == block_parameters[name].data_ptr()
        for name in full_parameters
    ):
        raise RuntimeError("Stage 27 branch parameter snapshots share storage")

    parameter_count = sum(
        parameter.numel()
        for parameter in full_model.parameters()
        if parameter.requires_grad
    )
    result = Stage27InitialStates(
        block=block_state,
        full=full_state,
        parameter_state_sha256=tensor_mapping_sha256(full_parameters),
        block_state_sha256=tensor_mapping_sha256(block_state),
        full_state_sha256=tensor_mapping_sha256(full_state),
        block_mask_sha256=tensor_sha256(block_mask),
        full_mask_sha256=tensor_sha256(full_mask),
        block_float_mask_sha256=tensor_sha256(block_mask.float()),
        full_float_mask_sha256=tensor_sha256(full_mask.float()),
        parameter_count=parameter_count,
        differing_state_names=differing_names,
        source_common_t64_state_sha256=source.common_state_sha256,
        source_extra_position_rows_sha256=source.extra_position_rows_sha256,
        source_shared_parameter_sha256=(
            source.shared_parameter_initialization_sha256
        ),
    )
    del block_model
    del full_model

    if _is_canonical_protocol(data, config, block, full, n_head, n_layer):
        observed = {
            "parameter count": result.parameter_count,
            "parameter initialization": result.parameter_state_sha256,
            "block state": result.block_state_sha256,
            "full state": result.full_state_sha256,
            "block mask": result.block_mask_sha256,
            "full mask": result.full_mask_sha256,
            "block float mask": result.block_float_mask_sha256,
            "full float mask": result.full_float_mask_sha256,
            "mask count": len(result.differing_state_names),
        }
        expected = {
            "parameter count": DEFAULT_PARAMETER_COUNT,
            "parameter initialization": DEFAULT_PARAMETER_INITIALIZATION_SHA256,
            "block state": DEFAULT_BLOCK_INITIAL_STATE_SHA256,
            "full state": DEFAULT_FULL_INITIAL_STATE_SHA256,
            "block mask": DEFAULT_BLOCK_MASK_SHA256,
            "full mask": DEFAULT_FULL_MASK_SHA256,
            "block float mask": DEFAULT_BLOCK_FLOAT_MASK_SHA256,
            "full float mask": DEFAULT_FULL_FLOAT_MASK_SHA256,
            "mask count": DEFAULT_N_HEAD * DEFAULT_N_LAYER,
        }
        if observed != expected:
            raise RuntimeError(
                "canonical Stage 27 initialization changed: "
                f"expected {expected}, got {observed}"
            )
    return result


def _resolved_paths(path: Path) -> tuple[Path, Path]:
    resolved = path.resolve()
    temporary = path.with_name(f".{path.name}.tmp").resolve()
    return resolved, temporary


def validate_cross_half_protocol(
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    *,
    anchor_checkpoint_path: Path | None = None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    if block.name == full.name:
        raise ValueError("block and full branch names must differ")
    if block.allow_cross_half:
        raise ValueError("the block control must forbid cross-half attention")
    if not full.allow_cross_half:
        raise ValueError("the full reference must allow cross-half attention")
    if block.boundary != full.boundary:
        raise ValueError("block and full branches must use the same boundary")
    if config.block_size != 2 * block.boundary:
        raise ValueError("block_size must equal twice the cross-half boundary")
    if n_head <= 0 or config.n_embd % n_head != 0:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({n_head})"
        )
    if n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {n_layer}")

    labelled_paths = {
        "block checkpoint": _resolved_paths(block.checkpoint_path)[0],
        "block temporary checkpoint": _resolved_paths(block.checkpoint_path)[1],
        "full reference": _resolved_paths(full.checkpoint_path)[0],
        "full temporary path": _resolved_paths(full.checkpoint_path)[1],
    }
    if anchor_checkpoint_path is not None:
        labelled_paths.update(
            {
                "block anchor checkpoint": _resolved_paths(
                    anchor_checkpoint_path
                )[0],
                "block anchor temporary checkpoint": _resolved_paths(
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

    bands = make_position_bands(config.block_size, block.boundary)
    return config.batch_size * config.block_size, bands


def _is_canonical_protocol(
    data: CharacterData,
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
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
        and block.boundary == DEFAULT_BOUNDARY
        and full.boundary == DEFAULT_BOUNDARY
        and not block.allow_cross_half
        and full.allow_cross_half
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
    )


def assert_hard_cross_half_barrier(mask: torch.Tensor, boundary: int) -> None:
    """Prove the mask has no cross-half path, including multilayer relay."""

    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be a square bool tensor")
    block_size = mask.shape[0]
    if block_size != 2 * boundary:
        raise ValueError("mask size must equal twice boundary")
    if torch.count_nonzero(mask[boundary:, :boundary]).item() != 0:
        raise RuntimeError("block mask contains a cross-half attention edge")
    expected_half = torch.tril(torch.ones(boundary, boundary, dtype=torch.bool))
    if not torch.equal(mask[:boundary, :boundary], expected_half):
        raise RuntimeError("first diagonal block is not causal")
    if not torch.equal(mask[boundary:, boundary:], expected_half):
        raise RuntimeError("second diagonal block is not causal")
    if torch.count_nonzero(mask[:boundary, boundary:]).item() != 0:
        raise RuntimeError("block mask contains a future-half edge")

    # Boolean transitive closure is still block diagonal, so stacking any
    # number of these attention layers cannot relay information across halves.
    reachability = mask.clone()
    for _ in range(block_size):
        composed = (reachability.float() @ reachability.float()) > 0
        updated = reachability | composed
        if torch.equal(updated, reachability):
            break
        reachability = updated
    if torch.count_nonzero(reachability[boundary:, :boundary]).item() != 0:
        raise RuntimeError("cross-half information can relay across layers")


def _serialize_losses(
    losses: PositionLosses,
    bands: Sequence[tuple[int, int]],
) -> dict[str, object]:
    if len(losses.bands) != len(bands):
        raise ValueError("loss bands do not match the Stage 27 bands")
    return {
        "overall": losses.overall,
        "first_half": losses.first_half,
        "second_half": losses.second_half,
        "bands": [
            {"start": start, "end": end, "loss": loss}
            for (start, end), loss in zip(bands, losses.bands, strict=True)
        ],
    }


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def _deserialize_losses(
    value: object,
    bands: Sequence[tuple[int, int]],
    *,
    name: str,
) -> PositionLosses:
    mapping = _require_mapping(value, name=name)
    scalars: list[float] = []
    for key in ("overall", "first_half", "second_half"):
        item = mapping.get(key)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"checkpoint {name}.{key} must be finite")
        scalars.append(float(item))
    raw_bands = mapping.get("bands")
    if not isinstance(raw_bands, list) or len(raw_bands) != len(bands):
        raise ValueError(f"checkpoint {name}.bands are incomplete")
    band_losses: list[float] = []
    for raw, (start, end) in zip(raw_bands, bands, strict=True):
        band = _require_mapping(raw, name=f"{name} band")
        if band.get("start") != start or band.get("end") != end:
            raise ValueError(f"checkpoint {name} band boundaries are inconsistent")
        loss = band.get("loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ValueError(f"checkpoint {name} band loss must be finite")
        band_losses.append(float(loss))
    return PositionLosses(*scalars, bands=tuple(band_losses))


@dataclass(slots=True)
class BlockValidationCheckpoint(BestValidationCheckpoint):
    """Save the Stage 27 best model and the predeclared matched-step anchor."""

    block_spec: CrossHalfSpec | None = None
    full_spec: CrossHalfSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initialization_seed: int = 0
    branch_initial_state_sha256: str = ""
    full_initial_state_sha256: str = ""
    parameter_state_sha256: str = ""
    block_mask_sha256: str = ""
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
            self.block_spec is None
            or self.full_spec is None
            or self.schedule is None
            or self.best_training_losses is None
            or self.best_validation_losses is None
            or not self.bands
        ):
            raise RuntimeError("incomplete Stage 27 checkpoint metadata")
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
                "attention_mask": self.block_spec.mask_kind,
                "attention_mask_scope": "every_head_in_every_layer",
                "cross_half_boundary": self.block_spec.boundary,
                "cross_half_attention_allowed": False,
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
                    self.config.batch_size
                    * self.config.block_size
                    * self.target_step
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
            "block_mask_sha256": self.block_mask_sha256,
            "full_mask_sha256": self.full_mask_sha256,
            "source_stage": 26,
            "source_common_t64_state_sha256": (
                self.source_common_t64_state_sha256
            ),
            "source_extra_position_rows_sha256": (
                self.source_extra_position_rows_sha256
            ),
            "source_shared_parameter_sha256": (
                self.source_shared_parameter_sha256
            ),
        }
        payload["experiment"] = {
            "stage": EXPERIMENT_STAGE,
            "branch": self.block_spec.name,
            "from_scratch": True,
            "comparison_variable": COMPARISON_VARIABLE,
            "hard_cross_half_information_barrier": True,
            "cross_half_boundary": self.block_spec.boundary,
            "block_mask_policy": self.block_spec.mask_kind,
            "full_mask_policy": self.full_spec.mask_kind,
            "blocked_cross_half_edges_per_head": self.block_spec.boundary**2,
            "multilayer_cross_half_relay_possible": False,
            "batch_size": self.config.batch_size,
            "block_size": self.config.block_size,
            "identical_parameter_count": True,
            "identical_learned_initialization": True,
            "identical_position_embeddings": True,
            "identical_training_batch_stream": True,
            "matched_training_rng_seed": True,
            "identical_dropout_probability": True,
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
            "full_reference_source_stage": 26,
            "full_reference_checkpoint_sha256": self.full_reference_sha256,
            "full_reference_step": self.full_reference_step,
            "full_reference_reused_read_only": True,
            "only_block_branch_newly_trained": True,
        }
        payload["evaluation_protocol"] = {
            "checkpoint_selection_objective": "overall_validation_loss",
            "primary_trained_model_endpoint": (
                "matched_anchor_natural_overall_validation_loss"
            ),
            "primary_mechanistic_endpoint": "fresh_second_half_loss",
            "natural_comparison": (
                "full_weights_full_mask_minus_block_weights_block_mask"
            ),
            "same_weight_mask_ablations": True,
            "crossed_mask_variants_persisted": False,
            "first_half_same_weight_identity_required": True,
            "first_half_same_weight_logit_identity_required": True,
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
        payload = super()._payload(step)
        return self._decorate_payload(payload, step=step, checkpoint_kind="best")

    def _save_anchor(self, record: AttentionEvaluationRecord) -> None:
        if self.anchor_path is None:
            raise RuntimeError("anchor checkpoint path is unavailable")
        self.anchor_training_losses = record.losses.train
        self.anchor_validation_losses = record.losses.val
        payload = super()._payload(record.step)
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
            raise ValueError(
                f"validation loss at step {record.step} must be finite"
            )
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_step = record.step
            self.best_training_losses = record.losses.train
            self.best_validation_losses = record.losses.val
            self.save(record.step)
        if record.step == self.anchor_step:
            self._save_anchor(record)
        return is_best


def _position_losses_from_stage_26(value: object, *, name: str) -> PositionLosses:
    mapping = _require_mapping(value, name=name)
    values: list[float] = []
    for key in ("overall", "first_half", "second_half"):
        item = mapping.get(key)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"checkpoint {name}.{key} must be finite")
        values.append(float(item))
    return PositionLosses(*values, bands=())


def validate_stage_26_full_reference(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    full_spec: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    initialization: Stage27InitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_checkpoint_sha256: str,
    expected_step: int,
) -> FullReference:
    """Validate Stage 26 under its own contract and never rewrite it."""

    if not full_spec.allow_cross_half:
        raise ValueError("the Stage 26 reference must use full causal attention")
    if not path.is_file():
        raise FileNotFoundError(f"Stage 26 full checkpoint not found: {path}")
    before_hash = checkpoint_sha256(path)
    if before_hash != expected_checkpoint_sha256:
        raise ValueError(
            "Stage 26 full checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, got {before_hash}"
        )

    control_config, treatment_config = _stage_26_configs(config, full_spec.boundary)
    control = _STAGE_26.ContextSpec(
        name=f"T={control_config.block_size} control",
        batch_size=control_config.batch_size,
        block_size=control_config.block_size,
        checkpoint_path=_STAGE_26.DEFAULT_CONTROL_CHECKPOINT_PATH,
    )
    treatment = _STAGE_26.ContextSpec(
        name=f"T={treatment_config.block_size} treatment",
        batch_size=treatment_config.batch_size,
        block_size=treatment_config.block_size,
        checkpoint_path=path,
    )
    checkpoint = _STAGE_26.validate_stage_26_checkpoint(
        path,
        data,
        treatment_config,
        treatment,
        control,
        treatment,
        n_head,
        n_layer,
        schedule,
        initialization_seed=config.seed,
        branch_initial_state_sha256=initialization.full_state_sha256,
        common_state_sha256=initialization.source_common_t64_state_sha256,
        extra_position_rows_sha256=(
            initialization.source_extra_position_rows_sha256
        ),
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        shared_parameter_initialization_sha256=(
            initialization.source_shared_parameter_sha256
        ),
    )
    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError("Stage 26 full checkpoint changed during validation")
    if checkpoint.get("step") != expected_step:
        raise ValueError(
            f"Stage 26 full checkpoint must be at step {expected_step}, "
            f"got {checkpoint.get('step')!r}"
        )

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="Stage 26 model_state_dict",
    )
    compatibility_model = build_model(
        data,
        config,
        full_spec,
        n_head,
        n_layer,
    )
    compatibility_model.load_state_dict(model_state, strict=True)
    verify_model_mask(compatibility_model, full_spec)
    del compatibility_model
    return FullReference(
        spec=full_spec,
        checkpoint_sha256=before_hash,
        checkpoint_step=expected_step,
        best_training_losses=_position_losses_from_stage_26(
            checkpoint.get("best_training_losses"),
            name="best_training_losses",
        ),
        best_validation_losses=_position_losses_from_stage_26(
            checkpoint.get("best_validation_losses"),
            name="best_validation_losses",
        ),
        checkpoint=checkpoint,
    )


def _validate_rng_state(checkpoint: Mapping[str, object]) -> None:
    rng_state = _require_mapping(checkpoint.get("rng_state"), name="rng_state")
    if not isinstance(rng_state.get("cpu"), torch.Tensor):
        raise ValueError("checkpoint rng_state.cpu must be a tensor")
    for accelerator in ("xpu", "cuda"):
        states = rng_state.get(accelerator)
        if states is not None and (
            not isinstance(states, list)
            or not all(isinstance(value, torch.Tensor) for value in states)
        ):
            raise ValueError(
                f"checkpoint rng_state.{accelerator} must be a list of tensors"
            )


def assert_matched_training_stream_state(
    full_checkpoint: Mapping[str, object],
    block_checkpoint: Mapping[str, object],
) -> None:
    """Require identical batch-generator and stochastic states at one step."""

    full_generator = full_checkpoint.get("training_generator_state")
    block_generator = block_checkpoint.get("training_generator_state")
    if not isinstance(full_generator, torch.Tensor) or not isinstance(
        block_generator,
        torch.Tensor,
    ):
        raise ValueError("both checkpoints must contain training generator tensors")
    if not torch.equal(full_generator.cpu(), block_generator.cpu()):
        raise RuntimeError("matched checkpoints consumed different training batches")

    full_rng = _require_mapping(full_checkpoint.get("rng_state"), name="full rng_state")
    block_rng = _require_mapping(
        block_checkpoint.get("rng_state"),
        name="block rng_state",
    )
    if full_rng.keys() != block_rng.keys():
        raise RuntimeError(
            "matched checkpoints were captured on different backend classes"
        )
    for backend in full_rng:
        full_state = full_rng[backend]
        block_state = block_rng[backend]
        if isinstance(full_state, torch.Tensor) and isinstance(
            block_state,
            torch.Tensor,
        ):
            equal = torch.equal(full_state.cpu(), block_state.cpu())
        elif isinstance(full_state, list) and isinstance(block_state, list):
            equal = len(full_state) == len(block_state) and all(
                isinstance(full_item, torch.Tensor)
                and isinstance(block_item, torch.Tensor)
                and torch.equal(full_item.cpu(), block_item.cpu())
                for full_item, block_item in zip(
                    full_state,
                    block_state,
                    strict=True,
                )
            )
        else:
            equal = False
        if not equal:
            raise RuntimeError(
                f"matched checkpoints consumed different {backend} RNG streams"
            )


def validate_stage_27_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: Stage27InitialStates,
    full_reference: FullReference,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    checkpoint_kind: str,
) -> Mapping[str, object]:
    """Strictly validate either the selected block model or its anchor."""

    if checkpoint_kind not in {"best", "anchor"}:
        raise ValueError("checkpoint_kind must be 'best' or 'anchor'")
    if not path.is_file():
        raise FileNotFoundError(f"Stage 27 checkpoint not found: {path}")
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
        raise ValueError("checkpoint anchor_step does not match the full reference")
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
        raise ValueError("Stage 27 must not restart the optimizer")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 27 optimizer provenance must be known")
    _validate_rng_state(checkpoint)

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
        "attention_mask": block.mask_kind,
        "attention_mask_scope": "every_head_in_every_layer",
        "cross_half_boundary": block.boundary,
        "cross_half_attention_allowed": False,
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
        "branch_state_sha256": initialization.block_state_sha256,
        "full_state_sha256": initialization.full_state_sha256,
        "parameter_state_sha256": initialization.parameter_state_sha256,
        "identical_learned_tensors": True,
        "state_differs_only_in_attention_masks": True,
        "block_mask_sha256": initialization.block_mask_sha256,
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
            raise ValueError(
                f"checkpoint initialization.{key} must be {expected!r}"
            )

    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    expected_experiment = {
        "stage": EXPERIMENT_STAGE,
        "branch": block.name,
        "from_scratch": True,
        "comparison_variable": COMPARISON_VARIABLE,
        "hard_cross_half_information_barrier": True,
        "cross_half_boundary": block.boundary,
        "block_mask_policy": block.mask_kind,
        "full_mask_policy": full.mask_kind,
        "blocked_cross_half_edges_per_head": block.boundary**2,
        "multilayer_cross_half_relay_possible": False,
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "identical_parameter_count": True,
        "identical_learned_initialization": True,
        "identical_position_embeddings": True,
        "identical_training_batch_stream": True,
        "matched_training_rng_seed": True,
        "identical_dropout_probability": True,
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
        "full_reference_source_stage": 26,
        "full_reference_checkpoint_sha256": (
            full_reference.checkpoint_sha256
        ),
        "full_reference_step": full_reference.checkpoint_step,
        "full_reference_reused_read_only": True,
        "only_block_branch_newly_trained": True,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"checkpoint experiment.{key} must be {expected!r}, "
                f"got {experiment.get(key)!r}"
            )

    evaluation = _require_mapping(
        checkpoint.get("evaluation_protocol"),
        name="evaluation_protocol",
    )
    expected_evaluation = {
        "checkpoint_selection_objective": "overall_validation_loss",
        "primary_trained_model_endpoint": (
            "matched_anchor_natural_overall_validation_loss"
        ),
        "primary_mechanistic_endpoint": "fresh_second_half_loss",
        "natural_comparison": "full_weights_full_mask_minus_block_weights_block_mask",
        "same_weight_mask_ablations": True,
        "crossed_mask_variants_persisted": False,
        "first_half_same_weight_identity_required": True,
        "first_half_same_weight_logit_identity_required": True,
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
    compatibility_model = build_model(data, config, block, n_head, n_layer)
    compatibility_model.load_state_dict(model_state, strict=True)
    verify_model_mask(compatibility_model, block)
    del compatibility_model
    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError("Stage 27 checkpoint changed during validation")
    return checkpoint


def inspect_attention_shape(
    model: GPTLanguageModel,
    config: TrainingConfig,
    spec: CrossHalfSpec,
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
        allowed = build_cross_half_attention_mask(
            config.block_size,
            spec.boundary,
            allow_cross_half=spec.allow_cross_half,
        ).to(device)
        forbidden = weights.masked_select(~allowed[None, None])
        if forbidden.numel() and torch.count_nonzero(forbidden).item() != 0:
            raise RuntimeError(f"{spec.name} assigns weight to masked positions")
        return shape
    finally:
        model.train(was_training)


def benchmark_cross_half_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> BenchmarkStats:
    """Benchmark a disposable branch on the exact Stage 27 update path."""

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
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    initialization: Stage27InitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> tuple[BenchmarkStats, BenchmarkStats]:
    block_runs: list[BenchmarkStats] = []
    full_runs: list[BenchmarkStats] = []
    ordered_runs = (
        (block, initialization.block, block_runs),
        (full, initialization.full, full_runs),
        (full, initialization.full, full_runs),
        (block, initialization.block, block_runs),
    )
    for spec, state, destination in ordered_runs:
        destination.append(
            benchmark_cross_half_branch(
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
            block_runs,
            steps_per_run=benchmark.num_steps,
        ),
        _STAGE_18.aggregate_benchmark_stats(
            full_runs,
            steps_per_run=benchmark.num_steps,
        ),
    )


def _print_benchmark(name: str, stats: BenchmarkStats) -> None:
    print(f"\n{name} training benchmark")
    print(f"  seconds:              {stats.seconds:.3f}")
    print(f"  iterations/sec:       {stats.iterations_per_sec:.3f}")
    print(f"  target characters/sec:{stats.tokens_per_sec:12.3f}")
    print(f"  peak allocated MiB:   {stats.peak_allocated_mb:.3f}")
    print(f"  peak reserved MiB:    {stats.peak_reserved_mb:.3f}")


def run_block_branch(
    data: CharacterData,
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: Stage27InitialStates,
    full_reference: FullReference,
    anchor_checkpoint_path: Path,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    on_update=None,
) -> BlockTrainingReport:
    """Train only the block branch; the full trajectory is reused read-only."""

    clear_accelerator_cache(device)
    model = build_model(data, config, block, n_head, n_layer, device)
    model.load_state_dict(initialization.block, strict=True)
    verify_model_mask(model, block)
    if tensor_mapping_sha256(model.state_dict()) != initialization.block_state_sha256:
        raise RuntimeError("block branch did not load its exact initial state")
    if (
        tensor_mapping_sha256(clone_parameters(model))
        != initialization.parameter_state_sha256
    ):
        raise RuntimeError("block branch learned initialization changed")

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
    best = BlockValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=block.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        block_spec=block,
        full_spec=full,
        schedule=schedule,
        target_step=config.max_iters,
        initialization_seed=config.seed,
        branch_initial_state_sha256=initialization.block_state_sha256,
        full_initial_state_sha256=initialization.full_state_sha256,
        parameter_state_sha256=initialization.parameter_state_sha256,
        block_mask_sha256=initialization.block_mask_sha256,
        full_mask_sha256=initialization.full_mask_sha256,
        source_common_t64_state_sha256=(
            initialization.source_common_t64_state_sha256
        ),
        source_extra_position_rows_sha256=(
            initialization.source_extra_position_rows_sha256
        ),
        source_shared_parameter_sha256=(
            initialization.source_shared_parameter_sha256
        ),
        full_reference_sha256=full_reference.checkpoint_sha256,
        full_reference_step=full_reference.checkpoint_step,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        attention_shape=expected_shape,
        bands=bands,
        anchor_path=anchor_checkpoint_path,
        anchor_step=full_reference.checkpoint_step,
    )
    record_evaluation = _STAGE_18.make_evaluation_callback(best, schedule)

    print(
        f"\n{block.name} branch (B={config.batch_size}, T={config.block_size}, "
        f"boundary={block.boundary}, dropout={DEFAULT_DROPOUT:g})"
    )
    print(f"  branch initialization: {initialization.block_state_sha256}")
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
        raise RuntimeError("Stage 27 checkpoint selection is incomplete")

    attention_shape = inspect_attention_shape(
        model,
        config,
        block,
        n_head,
        device,
    )
    final_generator_hash = tensor_sha256(training_generator.get_state())

    # Generate from the selected checkpoint, not the later final weights.
    selected_checkpoint = _require_mapping(
        torch.load(block.checkpoint_path, map_location="cpu", weights_only=True),
        name="selected block checkpoint",
    )
    selected_state = _require_mapping(
        selected_checkpoint.get("model_state_dict"),
        name="selected block model_state_dict",
    )
    model.load_state_dict(selected_state, strict=True)
    verify_model_mask(model, block)
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=config.seed + 5,
    )
    report = BlockTrainingReport(
        spec=block,
        parameter_count=parameter_count,
        initial_state_sha256=initialization.block_state_sha256,
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


def verify_initial_first_half_equivalence(
    block_model: GPTLanguageModel,
    full_model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    seed: int,
) -> float:
    return _STAGE_18.verify_initial_shared_prefix_equivalence(
        block_model,
        full_model,
        data,
        config,
        device,
        seed=seed,
        prefix_length=config.block_size // 2,
    )


def _print_stage_27_summary(
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    report: Stage27ExperimentReport,
    block_anchor_checkpoint_path: Path,
    full_sample: str,
) -> None:
    block_report = report.block
    full_reference = report.full
    matched = (
        full_reference.best_validation_losses.overall
        - block_report.anchor_validation_losses.overall
    )
    matched_first = (
        full_reference.best_validation_losses.first_half
        - block_report.anchor_validation_losses.first_half
    )
    matched_second = (
        full_reference.best_validation_losses.second_half
        - block_report.anchor_validation_losses.second_half
    )
    selected = (
        full_reference.best_validation_losses.overall
        - block_report.best_validation_losses.overall
    )

    print(f"\nStage {EXPERIMENT_STAGE} experimental summary")
    print(f"  both branches B/T:             {config.batch_size}/{config.block_size}")
    print(f"  hard boundary:                 {block.boundary}")
    print(f"  target characters/update:      {config.batch_size * config.block_size:,}")
    print(f"  parameter count (both):        {block_report.parameter_count:,}")
    print(f"  attention shape (both):        {block_report.attention_shape}")
    print(
        "  attention elements/block:      "
        f"{math.prod(block_report.attention_shape):,}"
    )
    print(
        "  block best train / val / gap:  "
        f"{block_report.best_training_losses.overall:.4f} / "
        f"{block_report.best_validation_losses.overall:.4f} / "
        f"{block_report.best_gap:.4f} @ {block_report.best_step}"
    )
    print(
        "  full reference train / val:    "
        f"{full_reference.best_training_losses.overall:.4f} / "
        f"{full_reference.best_validation_losses.overall:.4f} "
        f"@ {full_reference.checkpoint_step}"
    )
    print(
        "  matched full-block delta:       "
        f"{matched:+.4f} @ {full_reference.checkpoint_step}"
    )
    print(f"  matched first-half delta:       {matched_first:+.4f}")
    print(f"  matched second-half delta:      {matched_second:+.4f}")
    print(f"  selected full-block delta:      {selected:+.4f}")
    print("  negative deltas favor cross-half information access")
    print(f"  block best checkpoint:          {block.checkpoint_path}")
    print(f"  block best SHA-256:             {report.block_checkpoint_sha256}")
    print(f"  block matched anchor:           {block_anchor_checkpoint_path}")
    print(
        "  block anchor SHA-256:           "
        f"{report.block_anchor_checkpoint_sha256}"
    )
    print(f"  full read-only reference:       {full.checkpoint_path}")
    print(f"\n{block.name} selected-best generated text")
    print(block_report.sample)
    print(f"\n{full.name} selected-best generated text")
    print(full_sample)


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    block_anchor_checkpoint_path: Path,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    full_reference_expected_sha256: str,
    full_reference_expected_step: int,
    benchmark: BenchmarkConfig | None,
    on_update=None,
) -> Stage27ExperimentReport:
    if sample_length < 0:
        raise ValueError("sample_length must be non-negative")
    schedule.validate_target_step(config.max_iters)
    targets_per_update, bands = validate_cross_half_protocol(
        config,
        block,
        full,
        n_head,
        n_layer,
        anchor_checkpoint_path=block_anchor_checkpoint_path,
    )
    del targets_per_update
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

    initialization = create_stage_27_initial_states(
        data,
        config,
        block,
        full,
        n_head,
        n_layer,
    )
    block_mask = build_cross_half_attention_mask(
        config.block_size,
        block.boundary,
        allow_cross_half=False,
    )
    full_mask = build_cross_half_attention_mask(
        config.block_size,
        full.boundary,
        allow_cross_half=True,
    )
    assert_hard_cross_half_barrier(block_mask, block.boundary)
    removed_edges = int(full_mask.sum().item() - block_mask.sum().item())
    if removed_edges != block.boundary**2:
        raise RuntimeError("Stage 27 did not remove exactly boundary-squared edges")

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
        raise ValueError("full reference step is outside the Stage 27 schedule")
    if (
        full_reference.checkpoint_step % config.eval_interval != 0
        and full_reference.checkpoint_step != schedule.first_decay_step
        and full_reference.checkpoint_step != schedule.second_decay_step
        and full_reference.checkpoint_step != config.max_iters
    ):
        raise ValueError("full reference step is not a Stage 27 evaluation step")

    print("\nControlled from-initialization hard-barrier protocol")
    print("  The full Stage 26 trajectory is reused as a pinned read-only branch.")
    print("  Only the block-diagonal branch is newly trained from step zero.")
    print("  Both trajectories use identical T=128 learned initialization.")
    print("  Initial states differ only in every head's attention-mask buffer.")
    print(f"  learned parameter SHA-256: {initialization.parameter_state_sha256}")
    print(f"  block initial state SHA-256:{initialization.block_state_sha256}")
    print(f"  full initial state SHA-256: {initialization.full_state_sha256}")
    print(f"  block mask SHA-256:          {initialization.block_mask_sha256}")
    print(f"  full mask SHA-256:           {initialization.full_mask_sha256}")
    print(f"  full checkpoint SHA-256:     {reference_hash}")
    print(f"  full checkpoint step:        {full_reference.checkpoint_step}")
    print(
        f"  allowed edges/head:           block={int(block_mask.sum().item()):,}; "
        f"full={int(full_mask.sum().item()):,}; removed={removed_edges:,}"
    )

    block_audit_model = build_model(
        data,
        config,
        block,
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
    block_audit_model.load_state_dict(initialization.block, strict=True)
    full_audit_model.load_state_dict(initialization.full, strict=True)
    initial_first_half_max_difference = verify_initial_first_half_equivalence(
        block_audit_model,
        full_audit_model,
        data,
        config,
        torch.device("cpu"),
        seed=config.seed + 6,
    )
    del block_audit_model
    del full_audit_model
    print(
        "  Initial first-half max logit difference: "
        f"{initial_first_half_max_difference:.3e}"
    )

    block_benchmark: BenchmarkStats | None = None
    full_benchmark: BenchmarkStats | None = None
    if benchmark is not None:
        print("\nCounterbalanced disposable training benchmark")
        print(f"  Order: {block.name}, {full.name}, {full.name}, {block.name}")
        print(
            f"  Each run: {benchmark.num_warmup} warmup + "
            f"{benchmark.num_steps} timed updates"
        )
        block_benchmark, full_benchmark = run_counterbalanced_benchmarks(
            data,
            config,
            block,
            full,
            n_head,
            n_layer,
            device,
            initialization,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            benchmark=benchmark,
        )
        _print_benchmark(block.name, block_benchmark)
        _print_benchmark(full.name, full_benchmark)
        if checkpoint_sha256(full.checkpoint_path) != reference_hash:
            raise RuntimeError("full reference changed during disposable benchmark")

    block_report = run_block_branch(
        data,
        config,
        block,
        full,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        bands,
        initialization,
        full_reference,
        block_anchor_checkpoint_path,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        on_update=on_update,
    )
    block_report = replace(block_report, benchmark=block_benchmark)
    if checkpoint_sha256(full.checkpoint_path) != reference_hash:
        raise RuntimeError("full reference changed while training the block branch")

    validate_stage_27_checkpoint(
        block.checkpoint_path,
        data,
        config,
        block,
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
    anchor_checkpoint = validate_stage_27_checkpoint(
        block_anchor_checkpoint_path,
        data,
        config,
        block,
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
    block_checkpoint_hash = checkpoint_sha256(block.checkpoint_path)
    block_anchor_checkpoint_hash = checkpoint_sha256(
        block_anchor_checkpoint_path
    )

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
        seed=config.seed + 5,
    )
    del full_model
    clear_accelerator_cache(device)
    if checkpoint_sha256(full.checkpoint_path) != reference_hash:
        raise RuntimeError("full reference changed during generation")

    report = Stage27ExperimentReport(
        initialization=initialization,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        bands=bands,
        block=block_report,
        full=full_reference,
        initial_first_half_max_difference=initial_first_half_max_difference,
        block_benchmark=block_benchmark,
        full_benchmark=full_benchmark,
        block_checkpoint_sha256=block_checkpoint_hash,
        block_anchor_checkpoint_sha256=block_anchor_checkpoint_hash,
    )
    _print_stage_27_summary(
        config,
        block,
        full,
        report,
        block_anchor_checkpoint_path,
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
    if eval_iters < 2:
        raise ValueError(
            f"eval_iters must be at least 2 for paired inference, got {eval_iters}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if seed == config.seed + 2:
        raise ValueError("fresh validation seed must differ from fixed-panel seed")
    generator = torch.Generator().manual_seed(seed)
    panels: list[ValidationPanel] = []
    for _ in range(eval_iters):
        inputs, targets = data.get_batch(
            "val",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=torch.device("cpu"),
            generator=generator,
        )
        panels.append((inputs.clone(), targets.clone()))
    return tuple(panels)


@torch.no_grad()
def _evaluate_on_panels(
    model: GPTLanguageModel,
    panels: Sequence[ValidationPanel],
    device: torch.device,
    bands: Sequence[tuple[int, int]],
    mask: torch.Tensor,
) -> tuple[PositionLosses, ...]:
    if not panels:
        raise ValueError("at least one validation panel is required")
    was_training = model.training
    model.eval()
    values: list[PositionLosses] = []
    try:
        with temporary_attention_mask(model, mask):
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


@torch.no_grad()
def _evaluate_mask_pair_on_panels(
    model: GPTLanguageModel,
    panels: Sequence[ValidationPanel],
    device: torch.device,
    bands: Sequence[tuple[int, int]],
    first_mask: torch.Tensor,
    second_mask: torch.Tensor,
    *,
    boundary: int,
) -> tuple[tuple[PositionLosses, ...], tuple[PositionLosses, ...]]:
    """Evaluate two masks and prove their shared-prefix logits are identical."""

    if not panels:
        raise ValueError("at least one validation panel is required")
    if boundary <= 0 or boundary >= model.block_size:
        raise ValueError("boundary must lie strictly inside the model block")
    was_training = model.training
    model.eval()
    first_values: list[PositionLosses] = []
    second_values: list[PositionLosses] = []
    try:
        for panel_index, (cpu_inputs, cpu_targets) in enumerate(panels):
            inputs = cpu_inputs.to(device)
            targets = cpu_targets.to(device)
            with temporary_attention_mask(model, first_mask):
                first_logits, _ = model(inputs)
            with temporary_attention_mask(model, second_mask):
                second_logits, _ = model(inputs)
            first_prefix = first_logits[:, :boundary]
            second_prefix = second_logits[:, :boundary]
            if not torch.equal(first_prefix, second_prefix):
                maximum_difference = float(
                    (first_prefix - second_prefix).abs().max().item()
                )
                raise RuntimeError(
                    "same-weight masks changed first-half logits on validation "
                    f"panel {panel_index}: max difference {maximum_difference}"
                )
            first_values.append(
                _STAGE_18._as_position_losses(
                    _STAGE_18.position_cross_entropy(
                        first_logits,
                        targets,
                        bands,
                    )
                )
            )
            second_values.append(
                _STAGE_18._as_position_losses(
                    _STAGE_18.position_cross_entropy(
                        second_logits,
                        targets,
                        bands,
                    )
                )
            )
            del inputs
            del targets
            del first_logits
            del second_logits
    finally:
        model.train(was_training)
    return tuple(first_values), tuple(second_values)


def _precise_result(
    *,
    name: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    checkpoint_step: int,
    fixed_panel_loss: float,
    batch_losses: tuple[PositionLosses, ...],
) -> PreciseBranchResult:
    return PreciseBranchResult(
        name=name,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_step=checkpoint_step,
        fixed_panel_loss=fixed_panel_loss,
        statistics=_STAGE_18._position_statistics(batch_losses),
        batch_losses=batch_losses,
    )


def _first_half_batchwise_exact(
    first: PreciseBranchResult,
    second: PreciseBranchResult,
) -> bool:
    if len(first.batch_losses) != len(second.batch_losses):
        return False
    if not first.batch_losses:
        return False
    first_half_band_count = 2
    return all(
        first_loss.first_half == second_loss.first_half
        and first_loss.bands[:first_half_band_count]
        == second_loss.bands[:first_half_band_count]
        for first_loss, second_loss in zip(
            first.batch_losses,
            second.batch_losses,
            strict=True,
        )
    )


def _evaluate_full_weight_pair(
    data: CharacterData,
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    panels: Sequence[ValidationPanel],
    full_reference: FullReference,
) -> tuple[PreciseBranchResult, PreciseBranchResult]:
    model = build_model(data, config, full, n_head, n_layer, device)
    state = _require_mapping(
        full_reference.checkpoint.get("model_state_dict"),
        name="full reference model_state_dict",
    )
    model.load_state_dict(state, strict=True)
    verify_model_mask(model, full)
    full_mask = build_cross_half_attention_mask(
        config.block_size,
        full.boundary,
        allow_cross_half=True,
    )
    block_mask = build_cross_half_attention_mask(
        config.block_size,
        block.boundary,
        allow_cross_half=False,
    )
    full_losses, blocked_losses = _evaluate_mask_pair_on_panels(
        model,
        panels,
        device,
        bands,
        full_mask,
        block_mask,
        boundary=full.boundary,
    )
    verify_model_mask(model, full)
    del model
    clear_accelerator_cache(device)
    natural = _precise_result(
        name="full-trained/full-mask",
        checkpoint_path=full.checkpoint_path,
        checkpoint_hash=full_reference.checkpoint_sha256,
        checkpoint_step=full_reference.checkpoint_step,
        fixed_panel_loss=full_reference.best_validation_losses.overall,
        batch_losses=full_losses,
    )
    blocked = _precise_result(
        name="full-trained/block-mask (ephemeral)",
        checkpoint_path=full.checkpoint_path,
        checkpoint_hash=full_reference.checkpoint_sha256,
        checkpoint_step=full_reference.checkpoint_step,
        fixed_panel_loss=float("nan"),
        batch_losses=blocked_losses,
    )
    if not _first_half_batchwise_exact(natural, blocked):
        raise RuntimeError(
            "same full-trained weights changed first-half loss under mask ablation"
        )
    return natural, blocked


def _evaluate_block_weight_pair(
    path: Path,
    checkpoint: Mapping[str, object],
    checkpoint_hash: str,
    fixed_panel_loss: float,
    data: CharacterData,
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    panels: Sequence[ValidationPanel],
    *,
    endpoint_label: str,
) -> tuple[PreciseBranchResult, PreciseBranchResult]:
    model = build_model(data, config, block, n_head, n_layer, device)
    state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name=f"{endpoint_label} block model_state_dict",
    )
    model.load_state_dict(state, strict=True)
    verify_model_mask(model, block)
    block_mask = build_cross_half_attention_mask(
        config.block_size,
        block.boundary,
        allow_cross_half=False,
    )
    full_mask = build_cross_half_attention_mask(
        config.block_size,
        full.boundary,
        allow_cross_half=True,
    )
    blocked_losses, opened_losses = _evaluate_mask_pair_on_panels(
        model,
        panels,
        device,
        bands,
        block_mask,
        full_mask,
        boundary=block.boundary,
    )
    verify_model_mask(model, block)
    del model
    clear_accelerator_cache(device)
    step = checkpoint.get("step")
    assert isinstance(step, int) and not isinstance(step, bool)
    natural = _precise_result(
        name=f"block-trained/block-mask ({endpoint_label})",
        checkpoint_path=path,
        checkpoint_hash=checkpoint_hash,
        checkpoint_step=step,
        fixed_panel_loss=fixed_panel_loss,
        batch_losses=blocked_losses,
    )
    opened = _precise_result(
        name=f"block-trained/full-mask ({endpoint_label}, ephemeral)",
        checkpoint_path=path,
        checkpoint_hash=checkpoint_hash,
        checkpoint_step=step,
        fixed_panel_loss=float("nan"),
        batch_losses=opened_losses,
    )
    if not _first_half_batchwise_exact(natural, opened):
        raise RuntimeError(
            "same block-trained weights changed first-half loss when opening mask"
        )
    return natural, opened


def _make_endpoint_report(
    name: str,
    full_natural: PreciseBranchResult,
    full_blocked: PreciseBranchResult,
    block_natural: PreciseBranchResult,
    block_opened: PreciseBranchResult,
) -> EndpointValidationReport:
    return EndpointValidationReport(
        name=name,
        full_natural=full_natural,
        full_blocked=full_blocked,
        block_natural=block_natural,
        block_opened=block_opened,
        natural_delta=_STAGE_18._paired_delta(full_natural, block_natural),
        full_weight_mask_delta=_STAGE_18._paired_delta(
            full_natural,
            full_blocked,
        ),
        block_weight_mask_delta=_STAGE_18._paired_delta(
            block_opened,
            block_natural,
        ),
        full_weight_first_half_exact=_first_half_batchwise_exact(
            full_natural,
            full_blocked,
        ),
        block_weight_first_half_exact=_first_half_batchwise_exact(
            block_natural,
            block_opened,
        ),
    )


def run_four_artifact_validation(
    data: CharacterData,
    config: TrainingConfig,
    block: CrossHalfSpec,
    full: CrossHalfSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    experiment: Stage27ExperimentReport,
    block_anchor_checkpoint_path: Path,
    *,
    eval_iters: int,
    seed: int,
) -> Stage27PreciseValidationReport:
    """Evaluate two persisted checkpoints under both masks on one panel."""

    if experiment.bands != make_position_bands(config.block_size, block.boundary):
        raise ValueError("fresh-validation bands differ from the training protocol")
    panels = make_validation_panels(
        data,
        config,
        eval_iters=eval_iters,
        seed=seed,
    )
    full_hash_before = checkpoint_sha256(full.checkpoint_path)
    if full_hash_before != experiment.full.checkpoint_sha256:
        raise RuntimeError("full reference changed before fresh validation")
    anchor_hash = checkpoint_sha256(block_anchor_checkpoint_path)
    if anchor_hash != experiment.block_anchor_checkpoint_sha256:
        raise RuntimeError(
            "block anchor differs from the artifact produced by Stage 27 training"
        )
    best_hash = checkpoint_sha256(block.checkpoint_path)
    if best_hash != experiment.block_checkpoint_sha256:
        raise RuntimeError(
            "block best differs from the artifact produced by Stage 27 training"
        )

    full_natural, full_blocked = _evaluate_full_weight_pair(
        data,
        config,
        block,
        full,
        n_head,
        n_layer,
        device,
        experiment.bands,
        panels,
        experiment.full,
    )

    anchor_checkpoint = validate_stage_27_checkpoint(
        block_anchor_checkpoint_path,
        data,
        config,
        block,
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
    anchor_block, anchor_opened = _evaluate_block_weight_pair(
        block_anchor_checkpoint_path,
        anchor_checkpoint,
        anchor_hash,
        anchor_losses.overall,
        data,
        config,
        block,
        full,
        n_head,
        n_layer,
        device,
        experiment.bands,
        panels,
        endpoint_label="matched-anchor",
    )

    best_checkpoint = validate_stage_27_checkpoint(
        block.checkpoint_path,
        data,
        config,
        block,
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
    best_block, best_opened = _evaluate_block_weight_pair(
        block.checkpoint_path,
        best_checkpoint,
        best_hash,
        float(best_checkpoint["best_val_loss"]),
        data,
        config,
        block,
        full,
        n_head,
        n_layer,
        device,
        experiment.bands,
        panels,
        endpoint_label="selected-best",
    )

    if checkpoint_sha256(full.checkpoint_path) != full_hash_before:
        raise RuntimeError("full reference changed during fresh validation")
    if checkpoint_sha256(block_anchor_checkpoint_path) != anchor_hash:
        raise RuntimeError("block anchor changed during fresh validation")
    if checkpoint_sha256(block.checkpoint_path) != best_hash:
        raise RuntimeError("block best checkpoint changed during fresh validation")

    return Stage27PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        bands=experiment.bands,
        anchor=_make_endpoint_report(
            "matched step",
            full_natural,
            full_blocked,
            anchor_block,
            anchor_opened,
        ),
        selected_best=_make_endpoint_report(
            "selected best",
            full_natural,
            full_blocked,
            best_block,
            best_opened,
        ),
    )


def _print_variant(result: PreciseBranchResult) -> None:
    mean = result.statistics.mean
    se = result.statistics.standard_error
    print(
        f"  {result.name} | step {result.checkpoint_step:5d} | "
        f"overall {mean.overall:.6f} +/- {se.overall:.6f} SE | "
        f"first {mean.first_half:.6f} +/- {se.first_half:.6f} | "
        f"second {mean.second_half:.6f} +/- {se.second_half:.6f}"
    )


def _print_delta(
    label: str,
    delta: PreciseDelta,
    bands: Sequence[tuple[int, int]],
) -> None:
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
        for index, (start, end) in enumerate(bands)
    )
    print(
        f"  {label}: {delta.candidate} - {delta.baseline}; "
        f"paired-t critical={delta.critical_value:.6f}"
    )
    for row_label, mean, se, low, high in rows:
        print(
            f"    {row_label:23s} {mean:+.6f} +/- {se:.6f} SE "
            f"(95% CI [{low:+.6f}, {high:+.6f}])"
        )


def print_precise_validation(report: Stage27PreciseValidationReport) -> None:
    print(f"\nFresh four-condition validation of Stage {EXPERIMENT_STAGE}")
    print(
        f"  {report.eval_iters} shared validation panels, seed={report.seed}; "
        "crossed masks are ephemeral"
    )
    for endpoint in (report.anchor, report.selected_best):
        print(f"\n{endpoint.name} endpoint")
        for result in (
            endpoint.full_natural,
            endpoint.full_blocked,
            endpoint.block_natural,
            endpoint.block_opened,
        ):
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
        print(
            "  same-weight first-half logits exact: "
            f"full={endpoint.full_weight_first_half_exact}, "
            f"block={endpoint.block_weight_first_half_exact}"
        )
        _print_delta("natural full-minus-block", endpoint.natural_delta, report.bands)
        _print_delta(
            "full-weight mask effect",
            endpoint.full_weight_mask_delta,
            report.bands,
        )
        _print_delta(
            "block-weight mask effect (exploratory)",
            endpoint.block_weight_mask_delta,
            report.bands,
        )
        print("  Negative deltas favor cross-half information access.")
    print(
        "\n  The matched-step natural overall endpoint and its predeclared "
        "second-half diagnostic carry primary evidential weight."
    )
    print(
        "  Selected-best, crossed-mask, and position-band intervals are "
        "exploratory, pointwise, and unadjusted for multiplicity."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 27: train a hard block-diagonal 64+64 control and compare "
            "it with the pinned Stage 26 full-context checkpoint."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument(
        "--precise-eval-iters",
        type=int,
        default=DEFAULT_PRECISE_EVAL_ITERS,
        help="Fresh paired panels after training; 0 skips, otherwise at least 2.",
    )
    parser.add_argument(
        "--precise-eval-seed",
        type=int,
        default=DEFAULT_PRECISE_EVAL_SEED,
    )
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument(
        "--block-checkpoint-path",
        type=Path,
        default=DEFAULT_BLOCK_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--block-anchor-checkpoint-path",
        type=Path,
        default=DEFAULT_BLOCK_ANCHOR_CHECKPOINT_PATH,
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
    block = CrossHalfSpec(
        name="block-64-at-128 control",
        allow_cross_half=False,
        boundary=DEFAULT_BOUNDARY,
        checkpoint_path=args.block_checkpoint_path,
    )
    full = CrossHalfSpec(
        name="full-128 Stage 26 reference",
        allow_cross_half=True,
        boundary=DEFAULT_BOUNDARY,
        checkpoint_path=args.full_reference_checkpoint_path,
    )
    _, bands = validate_cross_half_protocol(
        config,
        block,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        anchor_checkpoint_path=args.block_anchor_checkpoint_path,
    )
    del bands
    schedule.validate_target_step(config.max_iters)

    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)
    if not _is_canonical_protocol(
        data,
        config,
        block,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
    ):
        raise RuntimeError("the Stage 27 CLI is reserved for the canonical protocol")
    if device.type != "xpu":
        raise RuntimeError(
            "canonical Stage 27 must use the same XPU backend class as Stage 26; "
            "use the programmatic API for CPU plumbing tests"
        )
    accelerator_name = torch.xpu.get_device_name(0)
    if accelerator_name != DEFAULT_ACCELERATOR_NAME:
        raise RuntimeError(
            "canonical Stage 27 must use the Stage 26 accelerator model "
            f"{DEFAULT_ACCELERATOR_NAME!r}, got {accelerator_name!r}"
        )

    print(f"Stage {EXPERIMENT_STAGE}: {EXPERIMENT_TITLE}")
    print(f"Device: {device}")
    print(f"Accelerator: {accelerator_name}")
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
        "Masks: block control = two independent causal 64-token islands; "
        "full reference = ordinary T=128 causal attention"
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
        f"training RNG=1340, fresh={args.precise_eval_seed}"
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
        block,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        device,
        args.sample_length,
        schedule,
        args.block_anchor_checkpoint_path,
        training_batch_seed=DEFAULT_TRAINING_BATCH_SEED,
        training_rng_seed=DEFAULT_TRAINING_RNG_SEED,
        full_reference_expected_sha256=(
            DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256
        ),
        full_reference_expected_step=DEFAULT_FULL_REFERENCE_STEP,
        benchmark=benchmark,
    )
    if args.precise_eval_iters > 0:
        precise = run_four_artifact_validation(
            data,
            config,
            block,
            full,
            DEFAULT_N_HEAD,
            DEFAULT_N_LAYER,
            device,
            schedule,
            experiment,
            args.block_anchor_checkpoint_path,
            eval_iters=args.precise_eval_iters,
            seed=args.precise_eval_seed,
        )
        print_precise_validation(precise)
    if (
        checkpoint_sha256(full.checkpoint_path)
        != DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Stage 26 full reference changed during Stage 27")


if __name__ == "__main__":
    main()
