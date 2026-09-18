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


# Stage 30 keeps the exact Stage 29 geometry, training loop, statistics, and
# reference contracts.  Its intervention is deliberately smaller: one of the
# six W=23 masks is restored to ordinary full causal attention.
_STAGE_29_PATH = Path(__file__).with_name("29_global_receptive_field.py")
_STAGE_29_MODULE_NAME = "stage_29_global_receptive_field_for_stage_30"
_STAGE_29_SPEC = importlib.util.spec_from_file_location(
    _STAGE_29_MODULE_NAME,
    _STAGE_29_PATH,
)
assert _STAGE_29_SPEC is not None and _STAGE_29_SPEC.loader is not None
_STAGE_29 = importlib.util.module_from_spec(_STAGE_29_SPEC)
sys.modules[_STAGE_29_MODULE_NAME] = _STAGE_29
_STAGE_29_SPEC.loader.exec_module(_STAGE_29)

_STAGE_27 = _STAGE_29._STAGE_27
_STAGE_18 = _STAGE_29._STAGE_18
_STAGE_26 = _STAGE_29._STAGE_26

GPTLanguageModel = _STAGE_29.GPTLanguageModel
LearningRateSchedule = _STAGE_29.LearningRateSchedule
BestValidationCheckpoint = _STAGE_29.BestValidationCheckpoint
PositionLosses = _STAGE_29.PositionLosses
AttentionEvaluationRecord = _STAGE_29.AttentionEvaluationRecord
AttentionTrainingResult = _STAGE_29.AttentionTrainingResult
PreciseBranchResult = _STAGE_29.PreciseBranchResult
PreciseDelta = _STAGE_29.PreciseDelta
RangeDelta = _STAGE_29.RangeDelta
UniformAttentionSpec = _STAGE_29.AttentionSpec
FullReference = _STAGE_29.FullReference

checkpoint_sha256 = _STAGE_29.checkpoint_sha256
clear_accelerator_cache = _STAGE_29.clear_accelerator_cache
fingerprint_data = _STAGE_29.fingerprint_data
generate_from_final_model = _STAGE_29.generate_from_final_model
tensor_mapping_sha256 = _STAGE_29.tensor_mapping_sha256
tensor_sha256 = _STAGE_29.tensor_sha256
build_attention_mask = _STAGE_29.build_attention_mask
make_validation_panels = _STAGE_29.make_validation_panels
position_range_mean = _STAGE_29.position_range_mean
paired_position_range_delta = _STAGE_29.paired_position_range_delta


EXPERIMENT_STAGE = 30
EXPERIMENT_TITLE = "sparse global shortcut"
COMPARISON_VARIABLE = "one_middle_global_layer_vs_all_local_and_all_global"
PROTOCOL_NAME = "sparse_global_shortcut_in_depth"

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_HYBRID_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_30_hybrid_layer_3_at_128_best_checkpoint.pt"
)
DEFAULT_HYBRID_ANCHOR_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_30_hybrid_layer_3_at_128_step_13500_checkpoint.pt"
)
DEFAULT_WINDOW_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_29_window_23_at_128_best_checkpoint.pt"
)
DEFAULT_WINDOW_ANCHOR_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_29_window_23_at_128_step_13500_checkpoint.pt"
)
DEFAULT_FULL_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_26_context_128_best_checkpoint.pt"
)

# Backwards-friendly names for callers that use candidate/local terminology.
DEFAULT_LOCAL_CHECKPOINT_PATH = DEFAULT_HYBRID_CHECKPOINT_PATH
DEFAULT_LOCAL_ANCHOR_CHECKPOINT_PATH = DEFAULT_HYBRID_ANCHOR_CHECKPOINT_PATH

DEFAULT_BATCH_SIZE = 16
DEFAULT_BLOCK_SIZE = 128
DEFAULT_LOCAL_CONTEXT = 23
DEFAULT_LAYER_CONTEXTS = (23, 23, None, 23, 23, 23)
DEFAULT_GLOBAL_LAYER_INDEX = 2
DEFAULT_GLOBAL_LAYER_NUMBER = DEFAULT_GLOBAL_LAYER_INDEX + 1
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
# Unlike Stage 29, which deliberately reused Stage 28's panels, this is a new
# post-Stage-29 held-out research panel.
DEFAULT_PRECISE_EVAL_SEED = 1_347

DROPOUT_PLACEMENT = _STAGE_29.DROPOUT_PLACEMENT
DEFAULT_DATA_FINGERPRINT = _STAGE_29.DEFAULT_DATA_FINGERPRINT
DEFAULT_PARAMETER_COUNT = _STAGE_29.DEFAULT_PARAMETER_COUNT
DEFAULT_PARAMETER_INITIALIZATION_SHA256 = (
    _STAGE_29.DEFAULT_PARAMETER_INITIALIZATION_SHA256
)
DEFAULT_FULL_INITIAL_STATE_SHA256 = _STAGE_29.DEFAULT_FULL_INITIAL_STATE_SHA256
DEFAULT_WINDOW_INITIAL_STATE_SHA256 = _STAGE_29.DEFAULT_LOCAL_INITIAL_STATE_SHA256
DEFAULT_HYBRID_INITIAL_STATE_SHA256 = (
    "f96825b691f3b77542222838f2e0fb308b0ce39318e0f51abfbaa2fc014304d5"
)
DEFAULT_HYBRID_MASK_STACK_SHA256 = (
    "5aab8cbfff98eed7c3146df3499fcffe5fce5cb3fe4bb183213d8299888d37ef"
)
DEFAULT_HYBRID_FLOAT_MASK_STACK_SHA256 = (
    "2005a635e15075b06c1a0ef6eebaab3bed5678bfb68de3b6b0192ad80916308e"
)
DEFAULT_HYBRID_MASK_BUFFER_MAPPING_SHA256 = (
    "5af692a38c69f823f5f9d835a84f1845a87388020f21dbd7863c5a6fa38b6486"
)
DEFAULT_FULL_MASK_STACK_SHA256 = (
    "814e1d940075711c12926bcc8ededed356db590187ddf2ca7bce64d609732c81"
)
DEFAULT_FULL_FLOAT_MASK_STACK_SHA256 = (
    "f8226054bdcd197a561d9414a288421093b868244e891337121cdd3d8727a0fa"
)
DEFAULT_WINDOW_MASK_SHA256 = _STAGE_29.DEFAULT_LOCAL_MASK_SHA256
DEFAULT_FULL_MASK_SHA256 = _STAGE_29.DEFAULT_FULL_MASK_SHA256
DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256 = (
    _STAGE_29.DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256
)
DEFAULT_FULL_REFERENCE_STEP = 13_500
DEFAULT_WINDOW_REFERENCE_CHECKPOINT_SHA256 = (
    "8e89725f25a9320bd5a30d3956b7a32aa732a902eab9de7c76dd67021f17c4f8"
)
DEFAULT_WINDOW_ANCHOR_REFERENCE_CHECKPOINT_SHA256 = (
    "d2169e263d76bc730e831a108f32b2aeae358a40e61895cd55c50a14e1fca1da"
)
DEFAULT_WINDOW_REFERENCE_STEP = 13_500
DEFAULT_WINDOW_ANCHOR_REFERENCE_STEP = 13_500
DEFAULT_MATCHED_STEP = 13_500
DEFAULT_DIAGNOSTIC_START = 23
DEFAULT_ALLOWED_EDGES_BY_LAYER_PER_HEAD = (2_691, 2_691, 8_256, 2_691, 2_691, 2_691)
DEFAULT_ALLOWED_EDGES_PER_HEAD_ACROSS_DEPTH = 21_711
DEFAULT_FULL_EDGES_PER_HEAD_ACROSS_DEPTH = 49_536
DEFAULT_REMOVED_EDGES_PER_HEAD_ACROSS_DEPTH = 27_825
DEFAULT_DIFFERING_MASK_BUFFER_COUNT = 20


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """A causal attention policy with an explicit context for every layer.

    ``None`` denotes full causal attention.  Stage 30 requires exactly one
    such layer and a common bounded window in all remaining layers, but its
    position is deliberately part of the spec so tiny CPU tests can move it.
    """

    name: str
    layer_contexts: tuple[int | None, ...]
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch name must not be empty")
        contexts = tuple(self.layer_contexts)
        object.__setattr__(self, "layer_contexts", contexts)
        if not contexts:
            raise ValueError("layer_contexts must not be empty")
        for context in contexts:
            if context is not None and (
                isinstance(context, bool)
                or not isinstance(context, int)
                or context <= 0
            ):
                raise ValueError(
                    "each layer context must be a positive integer or None, "
                    f"got {context!r}"
                )
        if contexts.count(None) != 1:
            raise ValueError("a sparse-global shortcut must contain exactly one full layer")
        bounded = tuple(context for context in contexts if context is not None)
        if not bounded:
            raise ValueError("a sparse-global shortcut must contain a local layer")
        if len(set(bounded)) != 1:
            raise ValueError("all bounded layers must use one common local context")

    @property
    def n_layer(self) -> int:
        return len(self.layer_contexts)

    @property
    def global_layer_index(self) -> int:
        return self.layer_contexts.index(None)

    @property
    def global_layer_indices(self) -> tuple[int, ...]:
        return (self.global_layer_index,)

    @property
    def local_context(self) -> int:
        return next(context for context in self.layer_contexts if context is not None)

    @property
    def per_layer_contexts(self) -> tuple[int | None, ...]:
        return self.layer_contexts

    @property
    def mask_kind(self) -> str:
        return "layerwise_causal_sparse_global_shortcut"


LayerwiseAttentionSpec = AttentionSpec
HybridAttentionSpec = AttentionSpec


@dataclass(frozen=True, slots=True)
class LayerwiseReachabilityAudit:
    layer_contexts: tuple[int | None, ...]
    direct_attention_spans: tuple[int, ...]
    widths_by_layer: tuple[int, ...]
    earliest_source_by_layer: tuple[tuple[int, ...], ...]
    final_width_by_query: tuple[int, ...]
    final_earliest_source_by_query: tuple[int, ...]
    first_query_with_unavailable_source: int | None
    first_global_reachability_layer: int | None
    all_causal_sources_reachable_by_layer: tuple[bool, ...]
    no_context_reset: bool
    causal: bool
    contiguous: bool

    @property
    def n_layer(self) -> int:
        return len(self.layer_contexts)

    @property
    def global_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, context in enumerate(self.layer_contexts) if context is None
        )

    @property
    def effective_receptive_field_width(self) -> int:
        return max(self.final_width_by_query)

    @property
    def theoretical_receptive_field_width(self) -> int:
        # With an explicit full layer, the exact effective width is the useful
        # theoretical result; there is no meaningful uncapped sliding-window sum.
        return self.effective_receptive_field_width

    @property
    def receptive_field_width(self) -> int:
        return self.effective_receptive_field_width

    @property
    def direct_attention_span(self) -> int:
        return max(self.direct_attention_spans)

    @property
    def all_causal_sources_reachable(self) -> bool:
        return (
            self.first_query_with_unavailable_source is None
            and self.all_causal_sources_reachable_by_layer[-1]
        )

    @property
    def has_hard_reset(self) -> bool:
        return not self.no_context_reset


@dataclass(frozen=True, slots=True)
class PairedInitialStates:
    hybrid: dict[str, torch.Tensor]
    full: dict[str, torch.Tensor]
    window: dict[str, torch.Tensor]
    parameter_state_sha256: str
    hybrid_state_sha256: str
    full_state_sha256: str
    window_state_sha256: str
    hybrid_mask_stack_sha256: str
    hybrid_float_mask_stack_sha256: str
    hybrid_mask_buffer_mapping_sha256: str
    full_mask_stack_sha256: str
    full_float_mask_stack_sha256: str
    window_mask_sha256: str
    full_mask_sha256: str
    parameter_count: int
    differing_state_names: tuple[str, ...]
    source_common_t64_state_sha256: str
    source_extra_position_rows_sha256: str
    source_shared_parameter_sha256: str
    window_initialization: object

    # Duck-typed names used by the audited Stage 26 reference validator.
    @property
    def local(self) -> dict[str, torch.Tensor]:
        return self.hybrid

    @property
    def local_state_sha256(self) -> str:
        return self.hybrid_state_sha256


@dataclass(frozen=True, slots=True)
class WindowReference:
    spec: UniformAttentionSpec
    anchor_path: Path
    best_checkpoint_sha256: str
    anchor_checkpoint_sha256: str
    best_step: int
    anchor_step: int
    best_training_losses: PositionLosses
    best_validation_losses: PositionLosses
    anchor_training_losses: PositionLosses
    anchor_validation_losses: PositionLosses
    best_checkpoint: Mapping[str, object]
    anchor_checkpoint: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class HybridTrainingReport:
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
    reachability: LayerwiseReachabilityAudit
    hybrid: HybridTrainingReport
    full: FullReference
    window: WindowReference
    initial_shared_prefix_length: int
    initial_shared_prefix_max_difference: float
    hybrid_benchmark: BenchmarkStats | None
    full_benchmark: BenchmarkStats | None
    hybrid_checkpoint_sha256: str
    hybrid_anchor_checkpoint_sha256: str

    @property
    def local(self) -> HybridTrainingReport:
        return self.hybrid

    @property
    def local_checkpoint_sha256(self) -> str:
        return self.hybrid_checkpoint_sha256

    @property
    def local_anchor_checkpoint_sha256(self) -> str:
        return self.hybrid_anchor_checkpoint_sha256


@dataclass(frozen=True, slots=True)
class EndpointValidationReport:
    name: str
    full_natural: PreciseBranchResult
    hybrid_natural: PreciseBranchResult
    window_natural: PreciseBranchResult
    full_minus_hybrid: PreciseDelta
    hybrid_minus_window: PreciseDelta
    full_minus_window: PreciseDelta
    full_minus_hybrid_multi_hop: RangeDelta
    hybrid_minus_window_multi_hop: RangeDelta
    full_minus_window_multi_hop: RangeDelta
    recovery_fraction: float | None
    multi_hop_recovery_fraction: float | None

    @property
    def natural_delta(self) -> PreciseDelta:
        return self.full_minus_hybrid

    @property
    def multi_hop_delta(self) -> RangeDelta:
        return self.full_minus_hybrid_multi_hop


@dataclass(frozen=True, slots=True)
class Stage30PreciseValidationReport:
    eval_iters: int
    seed: int
    bands: tuple[tuple[int, int], ...]
    anchor: EndpointValidationReport
    selected_best: EndpointValidationReport


def build_layer_attention_masks(
    block_size: int,
    spec: AttentionSpec,
) -> tuple[torch.Tensor, ...]:
    masks = tuple(
        build_attention_mask(block_size, context) for context in spec.layer_contexts
    )
    if any(context is not None and context > block_size for context in spec.layer_contexts):
        raise ValueError(f"bounded layer contexts must not exceed block_size {block_size}")
    return masks


def _infer_context(mask: torch.Tensor) -> int | None:
    block_size = int(mask.shape[0])
    full = build_attention_mask(block_size, None)
    if torch.equal(mask, full):
        return None
    span = max(int(value) for value in mask.sum(dim=1).tolist())
    if not torch.equal(mask, build_attention_mask(block_size, span)):
        raise RuntimeError("attention rows are not one contiguous causal policy")
    return span


def audit_layerwise_reachability(
    masks: Sequence[torch.Tensor],
) -> LayerwiseReachabilityAudit:
    if not masks:
        raise ValueError("at least one layer mask is required")
    first = masks[0]
    if first.dtype != torch.bool or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("every mask must be a square bool tensor")
    block_size = int(first.shape[0])
    contexts: list[int | None] = []
    spans: list[int] = []
    for mask in masks:
        if mask.dtype != torch.bool or tuple(mask.shape) != (block_size, block_size):
            raise ValueError("all layer masks must be same-size square bool tensors")
        if not torch.all(mask.diagonal()).item():
            raise RuntimeError("every query must retain its own position")
        if torch.count_nonzero(torch.triu(mask, diagonal=1)).item() != 0:
            raise RuntimeError("attention mask is not causal")
        context = _infer_context(mask)
        contexts.append(context)
        spans.append(block_size if context is None else context)
        if not torch.all(mask.diagonal(offset=-1)).item():
            raise RuntimeError("attention mask contains a context reset")

    reach = torch.eye(block_size, dtype=torch.bool)
    widths_by_layer: list[int] = []
    earliest_by_layer: list[tuple[int, ...]] = []
    global_by_layer: list[bool] = []
    first_global: int | None = None
    full = build_attention_mask(block_size, None)
    for layer_number, mask in enumerate(masks, start=1):
        reach = (mask.to(torch.int32) @ reach.to(torch.int32)) > 0
        if torch.count_nonzero(reach & ~full).item() != 0:
            raise RuntimeError("composed reachability became non-causal")
        earliest = tuple(
            int(torch.nonzero(reach[query], as_tuple=False)[0].item())
            for query in range(block_size)
        )
        widths = tuple(int(value) for value in reach.sum(dim=1).tolist())
        for query, (source, width) in enumerate(zip(earliest, widths, strict=True)):
            expected = torch.zeros(block_size, dtype=torch.bool)
            expected[source : query + 1] = True
            if width != query - source + 1 or not torch.equal(reach[query], expected):
                raise RuntimeError("layerwise reachability is not contiguous")
        global_now = torch.equal(reach, full)
        if global_now and first_global is None:
            first_global = layer_number
        widths_by_layer.append(widths[-1])
        earliest_by_layer.append(earliest)
        global_by_layer.append(global_now)

    final_widths = tuple(int(value) for value in reach.sum(dim=1).tolist())
    final_earliest = earliest_by_layer[-1]
    first_unavailable = next(
        (query for query, earliest in enumerate(final_earliest) if earliest > 0),
        None,
    )
    return LayerwiseReachabilityAudit(
        layer_contexts=tuple(contexts),
        direct_attention_spans=tuple(spans),
        widths_by_layer=tuple(widths_by_layer),
        earliest_source_by_layer=tuple(earliest_by_layer),
        final_width_by_query=final_widths,
        final_earliest_source_by_query=final_earliest,
        first_query_with_unavailable_source=first_unavailable,
        first_global_reachability_layer=first_global,
        all_causal_sources_reachable_by_layer=tuple(global_by_layer),
        no_context_reset=True,
        causal=True,
        contiguous=True,
    )


def verify_global_reachability(
    audit: LayerwiseReachabilityAudit,
    block_size: int,
) -> None:
    if audit.effective_receptive_field_width != block_size:
        raise RuntimeError("hybrid receptive field does not cover the full block")
    if not audit.all_causal_sources_reachable:
        raise RuntimeError("not every causal source is reachable at the final layer")


def apply_attention_masks(
    model: GPTLanguageModel,
    masks: Sequence[torch.Tensor],
) -> None:
    masks = tuple(masks)
    if len(masks) != len(model.blocks):
        raise ValueError(
            f"expected {len(model.blocks)} layer masks, got {len(masks)}"
        )
    expected_shape = (model.block_size, model.block_size)
    for block, mask in zip(model.blocks, masks, strict=True):
        if mask.dtype != torch.bool or tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"each attention mask must be bool with shape {expected_shape}"
            )
        for head in block.sa.heads:
            head.tril.copy_(mask.to(device=head.tril.device, dtype=head.tril.dtype))


def apply_attention_mask(model: GPTLanguageModel, masks: Sequence[torch.Tensor]) -> None:
    """Compatibility alias; Stage 30 always applies a sequence of masks."""

    apply_attention_masks(model, masks)


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    spec: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    if spec.n_layer != n_layer:
        raise ValueError(
            f"spec has {spec.n_layer} layers but n_layer is {n_layer}"
        )
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=DEFAULT_DROPOUT,
    )
    apply_attention_masks(model, build_layer_attention_masks(config.block_size, spec))
    if device is not None:
        model = model.to(device)
    return model


def verify_model_masks(model: GPTLanguageModel, spec: AttentionSpec) -> None:
    expected = build_layer_attention_masks(model.block_size, spec)
    if len(model.blocks) != len(expected):
        raise RuntimeError("model depth differs from the layerwise attention spec")
    observed = 0
    for layer, (block, mask) in enumerate(zip(model.blocks, expected, strict=True)):
        for head in block.sa.heads:
            observed += 1
            actual = head.tril.detach().cpu()
            if not torch.equal(actual, mask.to(dtype=actual.dtype)):
                raise RuntimeError(
                    f"{spec.name} contains an incorrect attention-mask buffer "
                    f"in layer {layer}"
                )
    if observed == 0:
        raise RuntimeError("model contains no attention heads")


verify_model_mask = verify_model_masks


def _clone_state(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }


def clone_parameters(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _mask_layer_index(name: str) -> int:
    pieces = name.split(".")
    if len(pieces) < 2 or pieces[0] != "blocks" or not pieces[1].isdigit():
        raise ValueError(f"cannot infer layer index from mask name {name!r}")
    return int(pieces[1])


def make_position_bands(
    block_size: int,
    local_context: int,
    n_layer: int,
) -> tuple[tuple[int, int], ...]:
    # Preserve Stage 29's position partition so all three branches can be
    # compared band-for-band.  The labels describe the W=23 reference routes,
    # not the hybrid route after its global exchange.
    return _STAGE_29.make_position_bands(block_size, local_context, n_layer)


def validate_attention_protocol(
    config: TrainingConfig,
    hybrid: AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int = DEFAULT_N_HEAD,
    n_layer: int = DEFAULT_N_LAYER,
    *,
    hybrid_anchor_checkpoint_path: Path | None = None,
    window_anchor_checkpoint_path: Path | None = None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    if len({hybrid.name, window.name, full.name}) != 3:
        raise ValueError("hybrid, window, and full branch names must differ")
    if hybrid.n_layer != n_layer:
        raise ValueError("hybrid layer_contexts length must equal n_layer")
    if window.max_context is None:
        raise ValueError("the Stage 29 reference must use bounded attention")
    if full.max_context is not None:
        raise ValueError("the Stage 26 reference must use full attention")
    if hybrid.local_context != window.max_context:
        raise ValueError("hybrid local layers must match the Stage 29 window")
    if hybrid.local_context >= config.block_size:
        raise ValueError("hybrid local context must remain below block_size")
    if any(
        context is not None and context > config.block_size
        for context in hybrid.layer_contexts
    ):
        raise ValueError("hybrid layer context exceeds block_size")
    if n_head <= 0 or config.n_embd % n_head:
        raise ValueError("n_embd must be divisible by a positive n_head")
    if n_layer <= 0:
        raise ValueError("n_layer must be positive")
    expected_window = _STAGE_29.minimum_local_context_for_global_receptive_field(
        config.block_size,
        n_layer,
    )
    if window.max_context != expected_window:
        raise ValueError(
            "Stage 30 requires the Stage 29 minimum global-RF window "
            f"({expected_window}), got {window.max_context}"
        )

    labelled_paths = {
        "hybrid best": hybrid.checkpoint_path,
        "window best": window.checkpoint_path,
        "full reference": full.checkpoint_path,
    }
    if hybrid_anchor_checkpoint_path is not None:
        labelled_paths["hybrid anchor"] = hybrid_anchor_checkpoint_path
    if window_anchor_checkpoint_path is not None:
        labelled_paths["window anchor"] = window_anchor_checkpoint_path
    resolved: dict[str, Path] = {}
    for label, path in labelled_paths.items():
        resolved[label] = path.resolve()
        resolved[f"{label} temporary"] = path.with_name(f".{path.name}.tmp").resolve()
    labels = tuple(resolved)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            if resolved[first_label] == resolved[second_label]:
                raise ValueError(f"{first_label} and {second_label} must be distinct")
            if (
                resolved[first_label].exists()
                and resolved[second_label].exists()
                and resolved[first_label].samefile(resolved[second_label])
            ):
                raise ValueError(
                    f"{first_label} and {second_label} must not be the same file"
                )

    masks = build_layer_attention_masks(config.block_size, hybrid)
    audit = audit_layerwise_reachability(masks)
    verify_global_reachability(audit, config.block_size)
    if audit.first_global_reachability_layer != hybrid.global_layer_index + 1:
        raise RuntimeError("the full layer must be the first globally reachable layer")
    bands = make_position_bands(config.block_size, hybrid.local_context, n_layer)
    if bands[0] != (0, hybrid.local_context):
        raise RuntimeError("the first band must be the exact shared-mask prefix")
    return config.batch_size * config.block_size, bands


def _is_canonical_protocol(
    data: CharacterData,
    config: TrainingConfig,
    hybrid: AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
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
        and hybrid.layer_contexts == DEFAULT_LAYER_CONTEXTS
        and window.max_context == DEFAULT_LOCAL_CONTEXT
        and full.max_context is None
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
    )


def create_paired_initial_states(
    data: CharacterData,
    config: TrainingConfig,
    hybrid: AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
) -> PairedInitialStates:
    """Regenerate Stage 26 initialization and change only 20 canonical masks."""

    validate_attention_protocol(config, hybrid, window, full, n_head, n_layer)
    window_initialization = _STAGE_29.create_paired_initial_states(
        data,
        config,
        window,
        full,
        n_head,
        n_layer,
    )
    full_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in window_initialization.full.items()
    }
    window_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in window_initialization.local.items()
    }
    hybrid_state = {name: tensor.clone() for name, tensor in full_state.items()}
    masks = build_layer_attention_masks(config.block_size, hybrid)
    mask_names = tuple(sorted(name for name in hybrid_state if name.endswith(".tril")))
    if len(mask_names) != n_head * n_layer:
        raise RuntimeError("unexpected number of attention-mask buffers")
    for name in mask_names:
        layer = _mask_layer_index(name)
        hybrid_state[name] = masks[layer].to(dtype=hybrid_state[name].dtype)

    differing = tuple(
        sorted(
            name
            for name in full_state
            if not torch.equal(hybrid_state[name], full_state[name])
        )
    )
    expected_differing = tuple(
        name
        for name in mask_names
        if _mask_layer_index(name) != hybrid.global_layer_index
    )
    if differing != expected_differing:
        raise RuntimeError(
            "hybrid and Full initial states must differ only in bounded-layer masks"
        )

    hybrid_model = build_model(data, config, hybrid, n_head, n_layer)
    hybrid_model.load_state_dict(hybrid_state, strict=True)
    full_model = _STAGE_29.build_model(data, config, full, n_head, n_layer)
    full_model.load_state_dict(full_state, strict=True)
    hybrid_parameters = clone_parameters(hybrid_model)
    full_parameters = clone_parameters(full_model)
    if hybrid_parameters.keys() != full_parameters.keys() or any(
        not torch.equal(hybrid_parameters[name], full_parameters[name])
        for name in hybrid_parameters
    ):
        raise RuntimeError("Stage 30 learned initializations are not identical")
    parameter_count = sum(
        parameter.numel()
        for parameter in hybrid_model.parameters()
        if parameter.requires_grad
    )
    del hybrid_model
    del full_model

    mask_stack = torch.stack(masks)
    full_stack = torch.stack(
        (build_attention_mask(config.block_size, None),) * n_layer
    )
    result = PairedInitialStates(
        hybrid=hybrid_state,
        full=full_state,
        window=window_state,
        parameter_state_sha256=tensor_mapping_sha256(hybrid_parameters),
        hybrid_state_sha256=tensor_mapping_sha256(hybrid_state),
        full_state_sha256=tensor_mapping_sha256(full_state),
        window_state_sha256=tensor_mapping_sha256(window_state),
        hybrid_mask_stack_sha256=tensor_sha256(mask_stack),
        hybrid_float_mask_stack_sha256=tensor_sha256(mask_stack.float()),
        hybrid_mask_buffer_mapping_sha256=tensor_mapping_sha256(
            {name: hybrid_state[name] for name in mask_names}
        ),
        full_mask_stack_sha256=tensor_sha256(full_stack),
        full_float_mask_stack_sha256=tensor_sha256(full_stack.float()),
        window_mask_sha256=tensor_sha256(
            build_attention_mask(config.block_size, window.max_context)
        ),
        full_mask_sha256=tensor_sha256(build_attention_mask(config.block_size, None)),
        parameter_count=parameter_count,
        differing_state_names=differing,
        source_common_t64_state_sha256=(
            window_initialization.source_common_t64_state_sha256
        ),
        source_extra_position_rows_sha256=(
            window_initialization.source_extra_position_rows_sha256
        ),
        source_shared_parameter_sha256=(
            window_initialization.source_shared_parameter_sha256
        ),
        window_initialization=window_initialization,
    )

    if _is_canonical_protocol(data, config, hybrid, window, full, n_head, n_layer):
        observed = {
            "parameter_count": result.parameter_count,
            "parameter_state": result.parameter_state_sha256,
            "hybrid_state": result.hybrid_state_sha256,
            "full_state": result.full_state_sha256,
            "window_state": result.window_state_sha256,
            "hybrid_mask_stack": result.hybrid_mask_stack_sha256,
            "hybrid_float_mask_stack": result.hybrid_float_mask_stack_sha256,
            "hybrid_mask_buffer_mapping": (
                result.hybrid_mask_buffer_mapping_sha256
            ),
            "full_mask_stack": result.full_mask_stack_sha256,
            "full_float_mask_stack": result.full_float_mask_stack_sha256,
            "differing_masks": len(result.differing_state_names),
        }
        expected = {
            "parameter_count": DEFAULT_PARAMETER_COUNT,
            "parameter_state": DEFAULT_PARAMETER_INITIALIZATION_SHA256,
            "hybrid_state": DEFAULT_HYBRID_INITIAL_STATE_SHA256,
            "full_state": DEFAULT_FULL_INITIAL_STATE_SHA256,
            "window_state": DEFAULT_WINDOW_INITIAL_STATE_SHA256,
            "hybrid_mask_stack": DEFAULT_HYBRID_MASK_STACK_SHA256,
            "hybrid_float_mask_stack": DEFAULT_HYBRID_FLOAT_MASK_STACK_SHA256,
            "hybrid_mask_buffer_mapping": (
                DEFAULT_HYBRID_MASK_BUFFER_MAPPING_SHA256
            ),
            "full_mask_stack": DEFAULT_FULL_MASK_STACK_SHA256,
            "full_float_mask_stack": DEFAULT_FULL_FLOAT_MASK_STACK_SHA256,
            "differing_masks": DEFAULT_DIFFERING_MASK_BUFFER_COUNT,
        }
        if observed != expected:
            raise RuntimeError(
                f"canonical Stage 30 initialization changed: expected {expected}, "
                f"got {observed}"
            )
    return result


def verify_initial_shared_prefix_equivalence(
    hybrid_model: GPTLanguageModel,
    full_model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    seed: int,
    prefix_length: int | None = None,
) -> float:
    if prefix_length is None:
        prefix_length = DEFAULT_LOCAL_CONTEXT
    return _STAGE_18.verify_initial_shared_prefix_equivalence(
        hybrid_model,
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
    return _STAGE_29._serialize_losses(losses, bands)


def _deserialize_losses(
    value: object,
    bands: Sequence[tuple[int, int]],
    *,
    name: str,
) -> PositionLosses:
    return _STAGE_29._deserialize_losses(value, bands, name=name)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    return _STAGE_29._require_mapping(value, name=name)


def validate_stage_26_full_reference(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    full_spec: UniformAttentionSpec,
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
    return _STAGE_29.validate_stage_26_full_reference(
        path,
        data,
        config,
        full_spec,
        n_head,
        n_layer,
        schedule,
        initialization.window_initialization,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_step=expected_step,
    )


def assert_matched_training_stream_state(
    first_checkpoint: Mapping[str, object],
    second_checkpoint: Mapping[str, object],
) -> None:
    _STAGE_29.assert_matched_training_stream_state(first_checkpoint, second_checkpoint)


def validate_stage_29_references(
    data: CharacterData,
    config: TrainingConfig,
    window: UniformAttentionSpec,
    window_anchor_path: Path,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    full_reference: FullReference,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_best_sha256: str,
    expected_anchor_sha256: str,
    expected_best_step: int,
    expected_anchor_step: int,
) -> WindowReference:
    """Strictly validate both immutable Stage 29 W=23 artifacts."""

    expected = (
        (window.checkpoint_path, expected_best_sha256, "best"),
        (window_anchor_path, expected_anchor_sha256, "anchor"),
    )
    hashes_before: dict[Path, str] = {}
    checkpoints: dict[str, Mapping[str, object]] = {}
    for path, expected_hash, kind in expected:
        if not path.is_file():
            raise FileNotFoundError(f"Stage 29 {kind} reference not found: {path}")
        observed_hash = checkpoint_sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Stage 29 {kind} checkpoint SHA-256 mismatch: expected "
                f"{expected_hash}, got {observed_hash}"
            )
        hashes_before[path] = observed_hash
        checkpoints[kind] = _STAGE_29.validate_stage_29_checkpoint(
            path,
            data,
            config,
            window,
            full,
            n_head,
            n_layer,
            schedule,
            bands,
            initialization.window_initialization,
            full_reference,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            checkpoint_kind=kind,
        )

    best = checkpoints["best"]
    anchor = checkpoints["anchor"]
    if best.get("step") != expected_best_step:
        raise ValueError(
            f"Stage 29 best reference must be at step {expected_best_step}, "
            f"got {best.get('step')!r}"
        )
    if anchor.get("step") != expected_anchor_step:
        raise ValueError(
            f"Stage 29 anchor reference must be at step {expected_anchor_step}, "
            f"got {anchor.get('step')!r}"
        )
    if anchor.get("best_step") != expected_best_step:
        raise ValueError("Stage 29 anchor does not record the pinned selected-best step")
    if expected_anchor_step != full_reference.checkpoint_step:
        raise ValueError("Stage 29 and Full matched reference steps must agree")
    assert_matched_training_stream_state(full_reference.checkpoint, anchor)

    best_training = _deserialize_losses(
        best.get("best_training_losses"),
        bands,
        name="Stage 29 best_training_losses",
    )
    best_validation = _deserialize_losses(
        best.get("best_validation_losses"),
        bands,
        name="Stage 29 best_validation_losses",
    )
    anchor_training = _deserialize_losses(
        anchor.get("anchor_training_losses"),
        bands,
        name="Stage 29 anchor_training_losses",
    )
    anchor_validation = _deserialize_losses(
        anchor.get("anchor_validation_losses"),
        bands,
        name="Stage 29 anchor_validation_losses",
    )
    for path, before_hash in hashes_before.items():
        if checkpoint_sha256(path) != before_hash:
            raise RuntimeError("Stage 29 reference changed during validation")
    return WindowReference(
        spec=window,
        anchor_path=window_anchor_path,
        best_checkpoint_sha256=hashes_before[window.checkpoint_path],
        anchor_checkpoint_sha256=hashes_before[window_anchor_path],
        best_step=expected_best_step,
        anchor_step=expected_anchor_step,
        best_training_losses=best_training,
        best_validation_losses=best_validation,
        anchor_training_losses=anchor_training,
        anchor_validation_losses=anchor_validation,
        best_checkpoint=best,
        anchor_checkpoint=anchor,
    )


def _allowed_edges_by_layer(
    config: TrainingConfig,
    hybrid: AttentionSpec,
) -> tuple[int, ...]:
    return tuple(
        int(mask.sum().item())
        for mask in build_layer_attention_masks(config.block_size, hybrid)
    )


def _architecture_metadata(
    config: TrainingConfig,
    hybrid: AttentionSpec,
    n_head: int,
    n_layer: int,
    initialization: PairedInitialStates,
    reachability: LayerwiseReachabilityAudit,
) -> dict[str, object]:
    return {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "residual_dropout": DEFAULT_DROPOUT,
        "dropout_placement": DROPOUT_PLACEMENT,
        "attention_mask": hybrid.mask_kind,
        "attention_mask_scope": "explicit_policy_per_layer_every_head",
        "layer_contexts": hybrid.layer_contexts,
        "per_layer_attention_windows": hybrid.layer_contexts,
        "global_layer_index": hybrid.global_layer_index,
        "global_layer_indices": hybrid.global_layer_indices,
        "global_layer_number": hybrid.global_layer_index + 1,
        "local_context": hybrid.local_context,
        "direct_attention_spans_by_layer": reachability.direct_attention_spans,
        "effective_receptive_field_widths_by_layer": reachability.widths_by_layer,
        "parameter_count": initialization.parameter_count,
    }


def _training_metadata(
    config: TrainingConfig,
    schedule: LearningRateSchedule,
    *,
    step: int,
    training_batch_seed: int,
    training_rng_seed: int,
) -> dict[str, object]:
    return {
        "batch_size": config.batch_size,
        "learning_rate": schedule.for_update(max(0, step - 1)),
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


def _initialization_metadata(
    config: TrainingConfig,
    initialization: PairedInitialStates,
) -> dict[str, object]:
    return {
        "kind": "stage_26_t128_layerwise_mask_matched",
        "seed": config.seed,
        "hybrid_state_sha256": initialization.hybrid_state_sha256,
        "window_state_sha256": initialization.window_state_sha256,
        "full_state_sha256": initialization.full_state_sha256,
        "parameter_state_sha256": initialization.parameter_state_sha256,
        "identical_learned_tensors": True,
        "state_differs_only_in_attention_masks": True,
        "differing_mask_buffer_names": initialization.differing_state_names,
        "differing_mask_buffer_count": len(initialization.differing_state_names),
        "hybrid_mask_stack_sha256": initialization.hybrid_mask_stack_sha256,
        "hybrid_float_mask_stack_sha256": (
            initialization.hybrid_float_mask_stack_sha256
        ),
        "hybrid_mask_buffer_mapping_sha256": (
            initialization.hybrid_mask_buffer_mapping_sha256
        ),
        "full_mask_stack_sha256": initialization.full_mask_stack_sha256,
        "full_float_mask_stack_sha256": initialization.full_float_mask_stack_sha256,
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


def _experiment_metadata(
    config: TrainingConfig,
    hybrid: AttentionSpec,
    n_head: int,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    reachability: LayerwiseReachabilityAudit,
    full_reference: FullReference,
    window_reference: WindowReference,
    schedule: LearningRateSchedule,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
) -> dict[str, object]:
    allowed_by_layer = _allowed_edges_by_layer(config, hybrid)
    full_edges = config.block_size * (config.block_size + 1) // 2
    return {
        "stage": EXPERIMENT_STAGE,
        "protocol": PROTOCOL_NAME,
        "branch": hybrid.name,
        "from_scratch": True,
        "comparison_variable": COMPARISON_VARIABLE,
        "layer_contexts": hybrid.layer_contexts,
        "per_layer_attention_windows": hybrid.layer_contexts,
        "global_layer_index": hybrid.global_layer_index,
        "global_layer_indices": hybrid.global_layer_indices,
        "global_layer_number": hybrid.global_layer_index + 1,
        "global_attention_layers": 1,
        "local_attention_layers": hybrid.n_layer - 1,
        "local_context": hybrid.local_context,
        "first_query_with_topology_difference": hybrid.local_context,
        "receptive_field_widths_by_layer": reachability.widths_by_layer,
        "first_global_reachability_layer": reachability.first_global_reachability_layer,
        "first_query_with_unavailable_source": (
            reachability.first_query_with_unavailable_source
        ),
        "all_causal_source_origins_reachable_at_final_layer": (
            reachability.all_causal_sources_reachable
        ),
        "context_reset": False,
        "continuous_local_history": True,
        "intervention_changes_direct_fan_in": True,
        "intervention_changes_information_path_length": True,
        "intervention_changes_reachable_source_origins_at_final_layer": False,
        "allowed_attention_edges_by_layer_per_head": allowed_by_layer,
        "allowed_attention_edges_per_head_across_depth": sum(allowed_by_layer),
        "removed_attention_edges_per_head_across_depth": (
            hybrid.n_layer * full_edges - sum(allowed_by_layer)
        ),
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
        "topology_divergent_position_range": (
            hybrid.local_context,
            config.block_size,
        ),
        "full_reference_source_stage": 26,
        "full_reference_checkpoint_sha256": full_reference.checkpoint_sha256,
        "full_reference_step": full_reference.checkpoint_step,
        "window_reference_source_stage": 29,
        "window_best_reference_checkpoint_sha256": (
            window_reference.best_checkpoint_sha256
        ),
        "window_best_reference_step": window_reference.best_step,
        "window_anchor_reference_checkpoint_sha256": (
            window_reference.anchor_checkpoint_sha256
        ),
        "window_anchor_reference_step": window_reference.anchor_step,
        "all_references_reused_read_only": True,
        "only_hybrid_branch_newly_trained": True,
        "initial_parameter_count": initialization.parameter_count,
        "initial_mask_buffer_difference_count": len(
            initialization.differing_state_names
        ),
    }


def _evaluation_metadata(
    config: TrainingConfig,
    hybrid: AttentionSpec,
    bands: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    return {
        "checkpoint_selection_objective": "overall_validation_loss",
        "primary_endpoint": "matched_step_full_minus_hybrid_natural_overall",
        "primary_comparison": "full_minus_hybrid",
        "secondary_comparisons": ("hybrid_minus_window", "full_minus_window"),
        "selected_best_endpoint": "secondary",
        "predeclared_mechanistic_diagnostic": (
            f"positions_{hybrid.local_context}_through_"
            f"{config.block_size - 1}_aggregate_loss"
        ),
        "mechanistic_diagnostic_position_range": (
            hybrid.local_context,
            config.block_size,
        ),
        "descriptive_recovery_fraction": (
            "(window_loss-hybrid_loss)/(window_loss-full_loss)"
        ),
        "recovery_fraction_inferential_status": "descriptive_only",
        "same_weight_mask_ablations": False,
        "crossed_mask_variants": False,
        "held_out_validation_natural_masks_only": True,
        "initial_exact_prefix": (0, hybrid.local_context),
        "fixed_panel_seed": config.seed + 2,
        "default_held_out_panel_seed": DEFAULT_PRECISE_EVAL_SEED,
        "split_order": ("train", "val"),
        "batches_per_split": config.eval_iters,
        "position_bands": bands,
        "confidence_interval_scope": "validation_batch_sampling_only",
        "multiplicity_adjustment": "none",
        "secondary_intervals": "exploratory_pointwise_unadjusted",
    }


@dataclass(slots=True)
class HybridValidationCheckpoint(BestValidationCheckpoint):
    hybrid_spec: AttentionSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initialization: PairedInitialStates | None = None
    reachability: LayerwiseReachabilityAudit | None = None
    full_reference: FullReference | None = None
    window_reference: WindowReference | None = None
    training_batch_seed: int = 0
    training_rng_seed: int = 0
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
            self.hybrid_spec is None
            or self.schedule is None
            or self.initialization is None
            or self.reachability is None
            or self.full_reference is None
            or self.window_reference is None
            or self.best_training_losses is None
            or self.best_validation_losses is None
            or not self.bands
        ):
            raise RuntimeError("incomplete Stage 30 checkpoint metadata")
        rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        expected_rate = self.schedule.for_update(max(0, step - 1))
        if rates != {expected_rate}:
            raise RuntimeError("checkpoint optimizer learning rate is inconsistent")
        payload["checkpoint_kind"] = checkpoint_kind
        payload["architecture"] = _architecture_metadata(
            self.config,
            self.hybrid_spec,
            self.n_head,
            self.n_layer,
            self.initialization,
            self.reachability,
        )
        payload["training_config"] = _training_metadata(
            self.config,
            self.schedule,
            step=step,
            training_batch_seed=self.training_batch_seed,
            training_rng_seed=self.training_rng_seed,
        )
        payload["initialization"] = _initialization_metadata(
            self.config,
            self.initialization,
        )
        payload["experiment"] = _experiment_metadata(
            self.config,
            self.hybrid_spec,
            self.n_head,
            self.bands,
            self.initialization,
            self.reachability,
            self.full_reference,
            self.window_reference,
            self.schedule,
            training_batch_seed=self.training_batch_seed,
            training_rng_seed=self.training_rng_seed,
        )
        payload["evaluation_protocol"] = _evaluation_metadata(
            self.config,
            self.hybrid_spec,
            self.bands,
        )
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
        return self._decorate_payload(
            super(HybridValidationCheckpoint, self)._payload(step),
            step=step,
            checkpoint_kind="best",
        )

    def _save_anchor(self, record: AttentionEvaluationRecord) -> None:
        if self.anchor_path is None:
            raise RuntimeError("anchor checkpoint path is unavailable")
        self.anchor_training_losses = record.losses.train
        self.anchor_validation_losses = record.losses.val
        payload = self._decorate_payload(
            super(HybridValidationCheckpoint, self)._payload(record.step),
            step=record.step,
            checkpoint_kind="anchor",
        )
        self.anchor_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.anchor_path.with_name(f".{self.anchor_path.name}.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(self.anchor_path)
        finally:
            if temporary.exists():
                temporary.unlink()
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
        if record.step == self.anchor_step:
            self._save_anchor(record)
        return is_best


def _require_exact_metadata(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    name: str,
) -> None:
    if set(observed) != set(expected):
        missing = tuple(sorted(set(expected) - set(observed)))
        extra = tuple(sorted(set(observed) - set(expected)))
        raise ValueError(
            f"checkpoint {name} keys differ from the protocol; "
            f"missing={missing}, extra={extra}"
        )
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            raise ValueError(
                f"checkpoint {name}.{key} must be {expected_value!r}, "
                f"got {observed[key]!r}"
            )


def validate_stage_30_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    hybrid: AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    full_reference: FullReference,
    window_reference: WindowReference,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    checkpoint_kind: str,
) -> Mapping[str, object]:
    """Strictly validate a selected-best or matched-step Stage 30 artifact."""

    if checkpoint_kind not in {"best", "anchor"}:
        raise ValueError("checkpoint_kind must be 'best' or 'anchor'")
    if not path.is_file():
        raise FileNotFoundError(f"Stage 30 checkpoint not found: {path}")
    _, expected_bands = validate_attention_protocol(
        config,
        hybrid,
        window,
        full,
        n_head,
        n_layer,
    )
    if bands != expected_bands:
        raise ValueError("checkpoint bands differ from the Stage 30 protocol")
    masks = build_layer_attention_masks(config.block_size, hybrid)
    reachability = audit_layerwise_reachability(masks)
    verify_global_reachability(reachability, config.block_size)

    reference_hashes = {
        full.checkpoint_path: full_reference.checkpoint_sha256,
        window.checkpoint_path: window_reference.best_checkpoint_sha256,
        window_reference.anchor_path: window_reference.anchor_checkpoint_sha256,
    }
    if any(path.resolve() == reference_path.resolve() for reference_path in reference_hashes):
        raise ValueError("Stage 30 checkpoint path must not alias a read-only reference")
    for reference_path, expected_hash in reference_hashes.items():
        if checkpoint_sha256(reference_path) != expected_hash:
            raise RuntimeError(f"read-only reference changed: {reference_path}")

    before_hash = checkpoint_sha256(path)
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="root",
    )
    expected_top_level = {
        "checkpoint_version",
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "best_val_loss",
        "best_step",
        "training_generator_state",
        "rng_state",
        "architecture",
        "training_config",
        "data_fingerprint",
        "optimizer_restart_step",
        "optimizer_provenance_known",
        "checkpoint_kind",
        "initialization",
        "experiment",
        "evaluation_protocol",
        "best_training_losses",
        "best_validation_losses",
        "anchor_step",
    }
    if checkpoint_kind == "anchor":
        expected_top_level.update(
            {"anchor_training_losses", "anchor_validation_losses"}
        )
    if set(checkpoint) != expected_top_level:
        missing = tuple(sorted(expected_top_level - set(checkpoint)))
        extra = tuple(sorted(set(checkpoint) - expected_top_level))
        raise ValueError(
            "checkpoint root keys differ from the Stage 30 contract; "
            f"missing={missing}, extra={extra}"
        )
    if checkpoint["checkpoint_version"] != 1:
        raise ValueError("checkpoint version must be 1")
    if checkpoint["checkpoint_kind"] != checkpoint_kind:
        raise ValueError("checkpoint kind is incorrect")
    step = checkpoint["step"]
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= config.max_iters
    ):
        raise ValueError("checkpoint step is invalid")
    best_step = checkpoint["best_step"]
    if (
        isinstance(best_step, bool)
        or not isinstance(best_step, int)
        or not 0 <= best_step <= step
    ):
        raise ValueError("checkpoint best_step is invalid")
    if checkpoint_kind == "best" and step != best_step:
        raise ValueError("best checkpoint step must equal best_step")
    if checkpoint["anchor_step"] != full_reference.checkpoint_step:
        raise ValueError("anchor_step does not match the Full reference")
    if checkpoint["anchor_step"] != window_reference.anchor_step:
        raise ValueError("anchor_step does not match the Stage 29 reference")
    if checkpoint_kind == "anchor" and step != window_reference.anchor_step:
        raise ValueError("anchor checkpoint is not at the matched step")
    best_val_loss = checkpoint["best_val_loss"]
    if (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise ValueError("checkpoint best_val_loss must be finite")
    if checkpoint["data_fingerprint"] != fingerprint_data(data):
        raise ValueError("checkpoint data fingerprint does not match")
    if checkpoint["optimizer_restart_step"] is not None:
        raise ValueError("Stage 30 must not restart the optimizer")
    if checkpoint["optimizer_provenance_known"] is not True:
        raise ValueError("Stage 30 optimizer provenance must be known")
    _STAGE_27._validate_rng_state(checkpoint)

    architecture = _require_mapping(checkpoint["architecture"], name="architecture")
    _require_exact_metadata(
        architecture,
        _architecture_metadata(
            config,
            hybrid,
            n_head,
            n_layer,
            initialization,
            reachability,
        ),
        name="architecture",
    )
    training = _require_mapping(
        checkpoint["training_config"],
        name="training_config",
    )
    _require_exact_metadata(
        training,
        _training_metadata(
            config,
            schedule,
            step=step,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
        ),
        name="training_config",
    )
    initialization_metadata = _require_mapping(
        checkpoint["initialization"],
        name="initialization",
    )
    _require_exact_metadata(
        initialization_metadata,
        _initialization_metadata(config, initialization),
        name="initialization",
    )
    experiment_metadata = _require_mapping(
        checkpoint["experiment"],
        name="experiment",
    )
    _require_exact_metadata(
        experiment_metadata,
        _experiment_metadata(
            config,
            hybrid,
            n_head,
            bands,
            initialization,
            reachability,
            full_reference,
            window_reference,
            schedule,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
        ),
        name="experiment",
    )
    evaluation_metadata = _require_mapping(
        checkpoint["evaluation_protocol"],
        name="evaluation_protocol",
    )
    _require_exact_metadata(
        evaluation_metadata,
        _evaluation_metadata(config, hybrid, bands),
        name="evaluation_protocol",
    )

    best_training = _deserialize_losses(
        checkpoint["best_training_losses"],
        bands,
        name="best_training_losses",
    )
    best_validation = _deserialize_losses(
        checkpoint["best_validation_losses"],
        bands,
        name="best_validation_losses",
    )
    if best_validation.overall != float(best_val_loss):
        raise ValueError("best validation loss does not match best_val_loss")
    del best_training
    if checkpoint_kind == "anchor":
        _deserialize_losses(
            checkpoint["anchor_training_losses"],
            bands,
            name="anchor_training_losses",
        )
        _deserialize_losses(
            checkpoint["anchor_validation_losses"],
            bands,
            name="anchor_validation_losses",
        )

    optimizer_state = _require_mapping(
        checkpoint["optimizer_state_dict"],
        name="optimizer_state_dict",
    )
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("checkpoint optimizer must contain parameter groups")
    expected_rate = schedule.for_update(max(0, step - 1))
    rates = {
        float(group["lr"])
        for group in param_groups
        if isinstance(group, Mapping) and "lr" in group
    }
    if rates != {expected_rate}:
        raise ValueError("checkpoint optimizer learning rate is inconsistent")

    generator_state = checkpoint["training_generator_state"]
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("checkpoint training generator state must be a tensor")
    expected_generator = _STAGE_18.expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator):
        raise ValueError("checkpoint batch-generator state is inconsistent")

    model_state = _require_mapping(checkpoint["model_state_dict"], name="model_state_dict")
    mask_state = {
        name: tensor
        for name, tensor in model_state.items()
        if name.endswith(".tril") and isinstance(tensor, torch.Tensor)
    }
    if tensor_mapping_sha256(mask_state) != initialization.hybrid_mask_buffer_mapping_sha256:
        raise ValueError("checkpoint attention-mask buffer fingerprint is incorrect")
    compatibility_model = build_model(data, config, hybrid, n_head, n_layer)
    compatibility_model.load_state_dict(model_state, strict=True)
    verify_model_masks(compatibility_model, hybrid)
    del compatibility_model

    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError("Stage 30 checkpoint changed during validation")
    for reference_path, expected_hash in reference_hashes.items():
        if checkpoint_sha256(reference_path) != expected_hash:
            raise RuntimeError("a read-only reference changed during validation")
    return checkpoint


# Shorter alias used by some experiment harnesses.
validate_checkpoint = validate_stage_30_checkpoint


def inspect_attention_shape(
    model: GPTLanguageModel,
    config: TrainingConfig,
    spec: AttentionSpec,
    n_head: int,
    device: torch.device,
) -> tuple[int, ...]:
    """Verify all logical layer masks while retaining dense BxHxTxT scores."""

    verify_model_masks(model, spec)
    expected_shape = (
        config.batch_size,
        n_head,
        config.block_size,
        config.block_size,
    )
    masks = build_layer_attention_masks(config.block_size, spec)
    was_training = model.training
    model.eval()
    try:
        inputs = torch.zeros(
            (config.batch_size, config.block_size),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            for layer, allowed in enumerate(masks):
                weights = model.get_attention_weights(inputs, block_index=layer)
                if tuple(weights.shape) != expected_shape:
                    raise RuntimeError(
                        f"layer {layer} attention shape must be {expected_shape}, "
                        f"got {tuple(weights.shape)}"
                    )
                forbidden = weights.masked_select(~allowed.to(device)[None, None])
                if forbidden.numel() and torch.count_nonzero(forbidden).item() != 0:
                    raise RuntimeError(
                        f"{spec.name} layer {layer} assigns weight to masked positions"
                    )
        return expected_shape
    finally:
        model.train(was_training)


def _benchmark_branch(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    *,
    hybrid_spec: AttentionSpec | None,
    uniform_spec: UniformAttentionSpec | None,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> BenchmarkStats:
    if (hybrid_spec is None) == (uniform_spec is None):
        raise ValueError("provide exactly one attention spec to benchmark")
    clear_accelerator_cache(device)
    if hybrid_spec is not None:
        model = build_model(data, config, hybrid_spec, n_head, n_layer, device)
        verify_model_masks(model, hybrid_spec)
    else:
        assert uniform_spec is not None
        model = _STAGE_29.build_model(
            data,
            config,
            uniform_spec,
            n_head,
            n_layer,
            device,
        )
        _STAGE_29.verify_model_mask(model, uniform_spec)
    model.load_state_dict(initial_state, strict=True)
    if hybrid_spec is not None:
        verify_model_masks(model, hybrid_spec)
    else:
        assert uniform_spec is not None
        _STAGE_29.verify_model_mask(model, uniform_spec)
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
        del logits, loss, inputs, targets
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
    del optimizer, model
    clear_accelerator_cache(device)
    return stats


def run_counterbalanced_benchmarks(
    data: CharacterData,
    config: TrainingConfig,
    hybrid: AttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    initialization: PairedInitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> tuple[BenchmarkStats, BenchmarkStats]:
    hybrid_runs: list[BenchmarkStats] = []
    full_runs: list[BenchmarkStats] = []
    order = ("hybrid", "full", "full", "hybrid")
    for branch in order:
        if branch == "hybrid":
            destination = hybrid_runs
            state = initialization.hybrid
            hybrid_spec, uniform_spec = hybrid, None
        else:
            destination = full_runs
            state = initialization.full
            hybrid_spec, uniform_spec = None, full
        destination.append(
            _benchmark_branch(
                data,
                config,
                n_head,
                n_layer,
                device,
                state,
                hybrid_spec=hybrid_spec,
                uniform_spec=uniform_spec,
                training_batch_seed=training_batch_seed,
                training_rng_seed=training_rng_seed,
                benchmark=benchmark,
            )
        )
    return (
        _STAGE_18.aggregate_benchmark_stats(
            hybrid_runs,
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


def run_hybrid_branch(
    data: CharacterData,
    config: TrainingConfig,
    hybrid: AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    full_reference: FullReference,
    window_reference: WindowReference,
    hybrid_anchor_checkpoint_path: Path,
    reachability: LayerwiseReachabilityAudit,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    on_update: Callable[[int, float, torch.Tensor, torch.Tensor], None] | None = None,
) -> HybridTrainingReport:
    """Train only the hybrid branch; all Stage 26/29 artifacts stay read-only."""

    del window, full  # Their validated identities are carried by the references.
    clear_accelerator_cache(device)
    model = build_model(data, config, hybrid, n_head, n_layer, device)
    model.load_state_dict(initialization.hybrid, strict=True)
    verify_model_masks(model, hybrid)
    if tensor_mapping_sha256(model.state_dict()) != initialization.hybrid_state_sha256:
        raise RuntimeError("hybrid branch did not load its exact initial state")
    if tensor_mapping_sha256(clone_parameters(model)) != initialization.parameter_state_sha256:
        raise RuntimeError("hybrid learned initialization changed")
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.for_update(0),
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    best = HybridValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=hybrid.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        hybrid_spec=hybrid,
        schedule=schedule,
        target_step=config.max_iters,
        initialization=initialization,
        reachability=reachability,
        full_reference=full_reference,
        window_reference=window_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        bands=bands,
        anchor_path=hybrid_anchor_checkpoint_path,
        anchor_step=full_reference.checkpoint_step,
    )
    record_evaluation = _STAGE_18.make_evaluation_callback(best, schedule)

    print(
        f"\n{hybrid.name} branch (B={config.batch_size}, T={config.block_size}, "
        f"contexts={hybrid.layer_contexts}, dropout={DEFAULT_DROPOUT:g})"
    )
    print(f"  branch initialization: {initialization.hybrid_state_sha256}")
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
        raise RuntimeError("Stage 30 checkpoint selection is incomplete")
    attention_shape = inspect_attention_shape(model, config, hybrid, n_head, device)
    final_generator_hash = tensor_sha256(training_generator.get_state())

    selected = _require_mapping(
        torch.load(hybrid.checkpoint_path, map_location="cpu", weights_only=True),
        name="selected hybrid checkpoint",
    )
    selected_state = _require_mapping(
        selected.get("model_state_dict"),
        name="selected hybrid model_state_dict",
    )
    model.load_state_dict(selected_state, strict=True)
    verify_model_masks(model, hybrid)
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=DEFAULT_SAMPLE_SEED,
    )
    report = HybridTrainingReport(
        spec=hybrid,
        parameter_count=parameter_count,
        initial_state_sha256=initialization.hybrid_state_sha256,
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
    del record_evaluation, best, optimizer, model
    clear_accelerator_cache(device)
    return report


def _reference_hashes(
    full_reference: FullReference,
    window_reference: WindowReference,
) -> dict[Path, str]:
    return {
        full_reference.spec.checkpoint_path: full_reference.checkpoint_sha256,
        window_reference.spec.checkpoint_path: (
            window_reference.best_checkpoint_sha256
        ),
        window_reference.anchor_path: window_reference.anchor_checkpoint_sha256,
    }


def _verify_reference_hashes(
    references: Mapping[Path, str],
    *,
    phase: str,
) -> None:
    for path, expected in references.items():
        observed = checkpoint_sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"read-only reference changed {phase}: {path}; "
                f"expected {expected}, got {observed}"
            )


def _print_experiment_summary(
    config: TrainingConfig,
    report: ExperimentReport,
    hybrid_anchor_path: Path,
) -> None:
    hybrid = report.hybrid
    full = report.full
    window = report.window
    matched_full_hybrid = (
        full.best_validation_losses.overall - hybrid.anchor_validation_losses.overall
    )
    matched_hybrid_window = (
        hybrid.anchor_validation_losses.overall
        - window.anchor_validation_losses.overall
    )
    selected_full_hybrid = (
        full.best_validation_losses.overall - hybrid.best_validation_losses.overall
    )
    print(f"\nStage {EXPERIMENT_STAGE} experimental summary")
    print(f"  all branches B/T:             {config.batch_size}/{config.block_size}")
    print(f"  hybrid contexts:              {hybrid.spec.layer_contexts}")
    print(
        "  effective RF widths/layer:    "
        + ", ".join(str(value) for value in report.reachability.widths_by_layer)
    )
    print(
        "  first global reachability:    layer "
        f"{report.reachability.first_global_reachability_layer}"
    )
    print(f"  parameter count (all):        {hybrid.parameter_count:,}")
    print(f"  attention shape (all):        {hybrid.attention_shape}")
    print(
        "  hybrid best train/val/gap:    "
        f"{hybrid.best_training_losses.overall:.4f} / "
        f"{hybrid.best_validation_losses.overall:.4f} / "
        f"{hybrid.best_gap:.4f} @ {hybrid.best_step}"
    )
    print(
        "  W23 best train/val:           "
        f"{window.best_training_losses.overall:.4f} / "
        f"{window.best_validation_losses.overall:.4f} @ {window.best_step}"
    )
    print(
        "  Full reference train/val:     "
        f"{full.best_training_losses.overall:.4f} / "
        f"{full.best_validation_losses.overall:.4f} @ {full.checkpoint_step}"
    )
    print(f"  matched Full-Hybrid delta:     {matched_full_hybrid:+.4f}")
    print(f"  matched Hybrid-W23 delta:      {matched_hybrid_window:+.4f}")
    print(f"  selected Full-Hybrid delta:    {selected_full_hybrid:+.4f}")
    print("  Negative candidate-baseline deltas favor the named candidate.")
    print(f"  hybrid best checkpoint:        {hybrid.spec.checkpoint_path}")
    print(f"  hybrid best SHA-256:           {report.hybrid_checkpoint_sha256}")
    print(f"  hybrid matched anchor:         {hybrid_anchor_path}")
    print(f"  hybrid anchor SHA-256:         {report.hybrid_anchor_checkpoint_sha256}")
    print(f"  W23 references:                {window.spec.checkpoint_path}")
    print(f"  Full reference:                {full.spec.checkpoint_path}")
    print(f"\n{hybrid.spec.name} selected-best generated text")
    print(hybrid.sample)


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    hybrid: AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    hybrid_anchor_checkpoint_path: Path,
    window_anchor_checkpoint_path: Path,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    full_reference_expected_sha256: str,
    full_reference_expected_step: int,
    window_best_expected_sha256: str,
    window_anchor_expected_sha256: str,
    window_best_expected_step: int,
    window_anchor_expected_step: int,
    benchmark: BenchmarkConfig | None = None,
    on_update: Callable[[int, float, torch.Tensor, torch.Tensor], None] | None = None,
) -> ExperimentReport:
    """Run Stage 30 while treating all Stage 26/29 artifacts as immutable."""

    _, bands = validate_attention_protocol(
        config,
        hybrid,
        window,
        full,
        n_head,
        n_layer,
        hybrid_anchor_checkpoint_path=hybrid_anchor_checkpoint_path,
        window_anchor_checkpoint_path=window_anchor_checkpoint_path,
    )
    if sample_length < 0:
        raise ValueError("sample_length must be non-negative")
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError("config learning rate must equal schedule initial LR")
    schedule.validate_target_step(config.max_iters)
    for split, source in (("train", data.train_data), ("val", data.val_data)):
        if len(source) <= config.block_size:
            raise ValueError(
                f"the {split} split has {len(source)} tokens, but block_size is "
                f"{config.block_size}"
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
        hybrid,
        window,
        full,
        n_head,
        n_layer,
    )
    reachability = audit_layerwise_reachability(
        build_layer_attention_masks(config.block_size, hybrid)
    )
    verify_global_reachability(reachability, config.block_size)
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
    window_reference = validate_stage_29_references(
        data,
        config,
        window,
        window_anchor_checkpoint_path,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        initialization,
        full_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_best_sha256=window_best_expected_sha256,
        expected_anchor_sha256=window_anchor_expected_sha256,
        expected_best_step=window_best_expected_step,
        expected_anchor_step=window_anchor_expected_step,
    )
    if full_reference.checkpoint_step != window_reference.anchor_step:
        raise ValueError("Full and W23 references must share one matched step")
    references = _reference_hashes(full_reference, window_reference)
    _verify_reference_hashes(references, phase="before Stage 30 training")

    hybrid_audit_model = build_model(
        data,
        config,
        hybrid,
        n_head,
        n_layer,
        torch.device("cpu"),
    )
    full_audit_model = _STAGE_29.build_model(
        data,
        config,
        full,
        n_head,
        n_layer,
        torch.device("cpu"),
    )
    hybrid_audit_model.load_state_dict(initialization.hybrid, strict=True)
    full_audit_model.load_state_dict(initialization.full, strict=True)
    initial_difference = verify_initial_shared_prefix_equivalence(
        hybrid_audit_model,
        full_audit_model,
        data,
        config,
        torch.device("cpu"),
        seed=DEFAULT_INITIAL_AUDIT_SEED,
        prefix_length=hybrid.local_context,
    )
    del hybrid_audit_model, full_audit_model

    print("\nControlled sparse-global-shortcut protocol")
    print("  Stage 26 Full and both Stage 29 W23 artifacts are pinned read-only.")
    print("  Only the one-global-layer hybrid is trained from step zero.")
    print(f"  layer contexts:              {hybrid.layer_contexts}")
    print(f"  learned parameter SHA-256:  {initialization.parameter_state_sha256}")
    print(f"  hybrid initial SHA-256:     {initialization.hybrid_state_sha256}")
    print(f"  mask-buffer differences:    {len(initialization.differing_state_names)}")
    print(f"  RF widths after layers:     {reachability.widths_by_layer}")
    print(
        f"  Initial [0,{hybrid.local_context}) max logit difference: "
        f"{initial_difference:.3e}"
    )

    hybrid_benchmark: BenchmarkStats | None = None
    full_benchmark: BenchmarkStats | None = None
    if benchmark is not None:
        print("\nCounterbalanced disposable dense training benchmark")
        hybrid_benchmark, full_benchmark = run_counterbalanced_benchmarks(
            data,
            config,
            hybrid,
            full,
            n_head,
            n_layer,
            device,
            initialization,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            benchmark=benchmark,
        )
        _print_benchmark(hybrid.name, hybrid_benchmark)
        _print_benchmark(full.name, full_benchmark)
        _verify_reference_hashes(references, phase="during benchmarking")

    hybrid_report = run_hybrid_branch(
        data,
        config,
        hybrid,
        window,
        full,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        bands,
        initialization,
        full_reference,
        window_reference,
        hybrid_anchor_checkpoint_path,
        reachability,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        on_update=on_update,
    )
    hybrid_report = replace(hybrid_report, benchmark=hybrid_benchmark)
    _verify_reference_hashes(references, phase="during hybrid training")

    validate_stage_30_checkpoint(
        hybrid.checkpoint_path,
        data,
        config,
        hybrid,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        initialization,
        full_reference,
        window_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        checkpoint_kind="best",
    )
    anchor_checkpoint = validate_stage_30_checkpoint(
        hybrid_anchor_checkpoint_path,
        data,
        config,
        hybrid,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        initialization,
        full_reference,
        window_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        checkpoint_kind="anchor",
    )
    assert_matched_training_stream_state(full_reference.checkpoint, anchor_checkpoint)
    assert_matched_training_stream_state(
        window_reference.anchor_checkpoint,
        anchor_checkpoint,
    )
    print("  Full/W23/Hybrid matched batch and RNG states: identical")
    hybrid_hash = checkpoint_sha256(hybrid.checkpoint_path)
    hybrid_anchor_hash = checkpoint_sha256(hybrid_anchor_checkpoint_path)
    _verify_reference_hashes(references, phase="during checkpoint validation")

    report = ExperimentReport(
        initialization=initialization,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        bands=bands,
        reachability=reachability,
        hybrid=hybrid_report,
        full=full_reference,
        window=window_reference,
        initial_shared_prefix_length=hybrid.local_context,
        initial_shared_prefix_max_difference=initial_difference,
        hybrid_benchmark=hybrid_benchmark,
        full_benchmark=full_benchmark,
        hybrid_checkpoint_sha256=hybrid_hash,
        hybrid_anchor_checkpoint_sha256=hybrid_anchor_hash,
    )
    _print_experiment_summary(config, report, hybrid_anchor_checkpoint_path)
    return report


ValidationPanel = tuple[torch.Tensor, torch.Tensor]


@torch.no_grad()
def _evaluate_natural_on_panels(
    model: GPTLanguageModel,
    panels: Sequence[ValidationPanel],
    device: torch.device,
    bands: Sequence[tuple[int, int]],
) -> tuple[PositionLosses, ...]:
    return _STAGE_29._evaluate_natural_on_panels(model, panels, device, bands)


def _precise_result(
    *,
    name: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    checkpoint_step: int,
    fixed_panel_loss: float,
    batch_losses: tuple[PositionLosses, ...],
) -> PreciseBranchResult:
    return _STAGE_29._precise_result(
        name=name,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        checkpoint_step=checkpoint_step,
        fixed_panel_loss=fixed_panel_loss,
        batch_losses=batch_losses,
    )


def _evaluate_uniform_checkpoint(
    *,
    name: str,
    path: Path,
    checkpoint: Mapping[str, object],
    checkpoint_hash: str,
    fixed_panel_loss: float,
    data: CharacterData,
    config: TrainingConfig,
    spec: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    panels: Sequence[ValidationPanel],
) -> PreciseBranchResult:
    model = _STAGE_29.build_model(data, config, spec, n_head, n_layer, device)
    state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name=f"{name} model_state_dict",
    )
    model.load_state_dict(state, strict=True)
    _STAGE_29.verify_model_mask(model, spec)
    losses = _evaluate_natural_on_panels(model, panels, device, bands)
    _STAGE_29.verify_model_mask(model, spec)
    del model
    clear_accelerator_cache(device)
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError(f"{name} checkpoint step is invalid")
    return _precise_result(
        name=name,
        checkpoint_path=path,
        checkpoint_hash=checkpoint_hash,
        checkpoint_step=step,
        fixed_panel_loss=fixed_panel_loss,
        batch_losses=losses,
    )


def _evaluate_hybrid_checkpoint(
    *,
    name: str,
    path: Path,
    checkpoint: Mapping[str, object],
    checkpoint_hash: str,
    fixed_panel_loss: float,
    data: CharacterData,
    config: TrainingConfig,
    spec: AttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    panels: Sequence[ValidationPanel],
) -> PreciseBranchResult:
    model = build_model(data, config, spec, n_head, n_layer, device)
    state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name=f"{name} model_state_dict",
    )
    model.load_state_dict(state, strict=True)
    verify_model_masks(model, spec)
    losses = _evaluate_natural_on_panels(model, panels, device, bands)
    verify_model_masks(model, spec)
    del model
    clear_accelerator_cache(device)
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError(f"{name} checkpoint step is invalid")
    return _precise_result(
        name=name,
        checkpoint_path=path,
        checkpoint_hash=checkpoint_hash,
        checkpoint_step=step,
        fixed_panel_loss=fixed_panel_loss,
        batch_losses=losses,
    )


def descriptive_recovery_fraction(
    full_minus_window: float,
    hybrid_minus_window: float,
) -> float | None:
    """Return the fraction of W23's deficit to Full recovered by Hybrid.

    Both arguments use candidate-minus-W23 orientation.  The ratio is purely
    descriptive; Stage 30 does not attach a confidence interval to it.
    """

    if not math.isfinite(full_minus_window) or not math.isfinite(
        hybrid_minus_window
    ):
        raise ValueError("recovery deltas must be finite")
    if full_minus_window == 0.0:
        return None
    return hybrid_minus_window / full_minus_window


def _make_endpoint_report(
    name: str,
    full_natural: PreciseBranchResult,
    hybrid_natural: PreciseBranchResult,
    window_natural: PreciseBranchResult,
    bands: tuple[tuple[int, int], ...],
    multi_hop_start: int,
) -> EndpointValidationReport:
    full_hybrid = _STAGE_18._paired_delta(full_natural, hybrid_natural)
    hybrid_window = _STAGE_18._paired_delta(hybrid_natural, window_natural)
    full_window = _STAGE_18._paired_delta(full_natural, window_natural)
    full_hybrid_range = paired_position_range_delta(
        full_natural,
        hybrid_natural,
        bands,
        multi_hop_start,
        bands[-1][1],
    )
    hybrid_window_range = paired_position_range_delta(
        hybrid_natural,
        window_natural,
        bands,
        multi_hop_start,
        bands[-1][1],
    )
    full_window_range = paired_position_range_delta(
        full_natural,
        window_natural,
        bands,
        multi_hop_start,
        bands[-1][1],
    )
    return EndpointValidationReport(
        name=name,
        full_natural=full_natural,
        hybrid_natural=hybrid_natural,
        window_natural=window_natural,
        full_minus_hybrid=full_hybrid,
        hybrid_minus_window=hybrid_window,
        full_minus_window=full_window,
        full_minus_hybrid_multi_hop=full_hybrid_range,
        hybrid_minus_window_multi_hop=hybrid_window_range,
        full_minus_window_multi_hop=full_window_range,
        recovery_fraction=descriptive_recovery_fraction(
            full_window.mean_delta.overall,
            hybrid_window.mean_delta.overall,
        ),
        multi_hop_recovery_fraction=descriptive_recovery_fraction(
            full_window_range.mean_delta,
            hybrid_window_range.mean_delta,
        ),
    )


def run_paired_validation(
    data: CharacterData,
    config: TrainingConfig,
    hybrid: AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    experiment: ExperimentReport,
    hybrid_anchor_checkpoint_path: Path,
    window_anchor_checkpoint_path: Path,
    *,
    eval_iters: int,
    seed: int,
) -> Stage30PreciseValidationReport:
    """Evaluate Full, Hybrid, and W23 natural checkpoints on shared panels."""

    _, expected_bands = validate_attention_protocol(
        config,
        hybrid,
        window,
        full,
        n_head,
        n_layer,
        hybrid_anchor_checkpoint_path=hybrid_anchor_checkpoint_path,
        window_anchor_checkpoint_path=window_anchor_checkpoint_path,
    )
    if experiment.bands != expected_bands:
        raise ValueError("held-out validation bands differ from training")
    if eval_iters < 2:
        raise ValueError("paired validation requires at least two panels")
    panels = make_validation_panels(
        data,
        config,
        eval_iters=eval_iters,
        seed=seed,
    )
    references = _reference_hashes(experiment.full, experiment.window)
    generated = {
        hybrid.checkpoint_path: experiment.hybrid_checkpoint_sha256,
        hybrid_anchor_checkpoint_path: experiment.hybrid_anchor_checkpoint_sha256,
    }
    _verify_reference_hashes(references, phase="before held-out validation")
    for path, expected_hash in generated.items():
        if checkpoint_sha256(path) != expected_hash:
            raise RuntimeError(f"Stage 30 artifact changed before validation: {path}")

    # Re-run strict contracts immediately before consuming any persisted state.
    window_reference = validate_stage_29_references(
        data,
        config,
        window,
        window_anchor_checkpoint_path,
        full,
        n_head,
        n_layer,
        schedule,
        experiment.bands,
        experiment.initialization,
        experiment.full,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        expected_best_sha256=experiment.window.best_checkpoint_sha256,
        expected_anchor_sha256=experiment.window.anchor_checkpoint_sha256,
        expected_best_step=experiment.window.best_step,
        expected_anchor_step=experiment.window.anchor_step,
    )
    hybrid_anchor_checkpoint = validate_stage_30_checkpoint(
        hybrid_anchor_checkpoint_path,
        data,
        config,
        hybrid,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        experiment.bands,
        experiment.initialization,
        experiment.full,
        window_reference,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        checkpoint_kind="anchor",
    )
    hybrid_best_checkpoint = validate_stage_30_checkpoint(
        hybrid.checkpoint_path,
        data,
        config,
        hybrid,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        experiment.bands,
        experiment.initialization,
        experiment.full,
        window_reference,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        checkpoint_kind="best",
    )

    full_result = _evaluate_uniform_checkpoint(
        name="full-trained/full-mask",
        path=full.checkpoint_path,
        checkpoint=experiment.full.checkpoint,
        checkpoint_hash=experiment.full.checkpoint_sha256,
        fixed_panel_loss=experiment.full.best_validation_losses.overall,
        data=data,
        config=config,
        spec=full,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )
    window_anchor_result = _evaluate_uniform_checkpoint(
        name="W23-trained/W23-mask (matched-anchor)",
        path=window_anchor_checkpoint_path,
        checkpoint=window_reference.anchor_checkpoint,
        checkpoint_hash=window_reference.anchor_checkpoint_sha256,
        fixed_panel_loss=window_reference.anchor_validation_losses.overall,
        data=data,
        config=config,
        spec=window,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )
    window_best_result = _evaluate_uniform_checkpoint(
        name="W23-trained/W23-mask (selected-best)",
        path=window.checkpoint_path,
        checkpoint=window_reference.best_checkpoint,
        checkpoint_hash=window_reference.best_checkpoint_sha256,
        fixed_panel_loss=window_reference.best_validation_losses.overall,
        data=data,
        config=config,
        spec=window,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )
    hybrid_anchor_losses = _deserialize_losses(
        hybrid_anchor_checkpoint.get("anchor_validation_losses"),
        experiment.bands,
        name="hybrid anchor_validation_losses",
    )
    hybrid_anchor_result = _evaluate_hybrid_checkpoint(
        name="hybrid-trained/hybrid-mask (matched-anchor)",
        path=hybrid_anchor_checkpoint_path,
        checkpoint=hybrid_anchor_checkpoint,
        checkpoint_hash=experiment.hybrid_anchor_checkpoint_sha256,
        fixed_panel_loss=hybrid_anchor_losses.overall,
        data=data,
        config=config,
        spec=hybrid,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )
    hybrid_best_result = _evaluate_hybrid_checkpoint(
        name="hybrid-trained/hybrid-mask (selected-best)",
        path=hybrid.checkpoint_path,
        checkpoint=hybrid_best_checkpoint,
        checkpoint_hash=experiment.hybrid_checkpoint_sha256,
        fixed_panel_loss=float(hybrid_best_checkpoint["best_val_loss"]),
        data=data,
        config=config,
        spec=hybrid,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )

    _verify_reference_hashes(references, phase="during held-out validation")
    for path, expected_hash in generated.items():
        if checkpoint_sha256(path) != expected_hash:
            raise RuntimeError(f"Stage 30 artifact changed during validation: {path}")
    return Stage30PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        bands=experiment.bands,
        anchor=_make_endpoint_report(
            "matched step",
            full_result,
            hybrid_anchor_result,
            window_anchor_result,
            experiment.bands,
            hybrid.local_context,
        ),
        selected_best=_make_endpoint_report(
            "selected best",
            full_result,
            hybrid_best_result,
            window_best_result,
            experiment.bands,
            hybrid.local_context,
        ),
    )


def _print_variant(result: PreciseBranchResult) -> None:
    _STAGE_29._print_variant(result)


def _print_delta(
    label: str,
    delta: PreciseDelta,
    bands: Sequence[tuple[int, int]],
) -> None:
    _STAGE_29._print_delta(label, delta, bands)


def _print_range_delta(label: str, delta: RangeDelta) -> None:
    print(
        f"  {label} [{delta.start},{delta.end}): "
        f"{delta.mean_delta:+.6f} +/- {delta.standard_error:.6f} SE; "
        f"95% CI [{delta.confidence_low:+.6f}, "
        f"{delta.confidence_high:+.6f}]"
    )


def print_paired_validation(report: Stage30PreciseValidationReport) -> None:
    print(f"\nHeld-out three-branch paired validation of Stage {EXPERIMENT_STAGE}")
    print(
        f"  {report.eval_iters} shared validation panels, seed={report.seed}; "
        "natural trained masks only"
    )
    for endpoint in (report.anchor, report.selected_best):
        print(f"\n{endpoint.name} endpoint")
        full_hybrid_scope = "primary" if endpoint is report.anchor else "secondary"
        for result in (
            endpoint.full_natural,
            endpoint.hybrid_natural,
            endpoint.window_natural,
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
        _print_delta(
            f"{full_hybrid_scope} natural Full-minus-Hybrid",
            endpoint.full_minus_hybrid,
            report.bands,
        )
        _print_delta(
            "secondary natural Hybrid-minus-W23",
            endpoint.hybrid_minus_window,
            report.bands,
        )
        _print_delta(
            "secondary natural Full-minus-W23",
            endpoint.full_minus_window,
            report.bands,
        )
        _print_range_delta(
            "predeclared Full-minus-Hybrid topology-divergent aggregate",
            endpoint.full_minus_hybrid_multi_hop,
        )
        _print_range_delta(
            "secondary Hybrid-minus-W23 topology-divergent aggregate",
            endpoint.hybrid_minus_window_multi_hop,
        )
        _print_range_delta(
            "secondary Full-minus-W23 topology-divergent aggregate",
            endpoint.full_minus_window_multi_hop,
        )
        if endpoint.recovery_fraction is None:
            print("  descriptive overall recovery fraction: undefined (zero deficit)")
        else:
            print(
                "  descriptive overall recovery fraction: "
                f"{endpoint.recovery_fraction:.1%}"
            )
        if endpoint.multi_hop_recovery_fraction is None:
            print(
                "  descriptive topology-range recovery fraction: "
                "undefined (zero deficit)"
            )
        else:
            print(
                "  descriptive topology-range recovery fraction: "
                f"{endpoint.multi_hop_recovery_fraction:.1%}"
            )
    print(
        "\n  The matched-step Full-minus-Hybrid natural overall endpoint is "
        "primary. Hybrid-minus-W23, Full-minus-W23, selected-best results, "
        "position bands, and recovery fractions are secondary/descriptive."
    )
    print(
        "  Recovery fractions are ratios of paired point estimates and do not "
        "have inferential intervals."
    )


print_precise_validation = print_paired_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 30: train a W23 hybrid with one Full layer at depth 3 and "
            "compare pinned Full-128 and W23 references."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument(
        "--precise-eval-iters",
        type=int,
        default=DEFAULT_PRECISE_EVAL_ITERS,
    )
    parser.add_argument(
        "--precise-eval-seed",
        type=int,
        default=DEFAULT_PRECISE_EVAL_SEED,
    )
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument(
        "--hybrid-checkpoint-path",
        type=Path,
        default=DEFAULT_HYBRID_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--hybrid-anchor-checkpoint-path",
        type=Path,
        default=DEFAULT_HYBRID_ANCHOR_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--window-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_WINDOW_REFERENCE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--window-anchor-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_WINDOW_ANCHOR_REFERENCE_CHECKPOINT_PATH,
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
    if args.precise_eval_seed != DEFAULT_PRECISE_EVAL_SEED:
        raise ValueError(
            "canonical Stage 30 requires the predeclared newly held-out "
            f"precise-eval-seed {DEFAULT_PRECISE_EVAL_SEED}"
        )
    if args.benchmark_warmup < 0 or args.benchmark_steps < 0:
        raise ValueError("benchmark counts must be non-negative")

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
    hybrid = AttentionSpec(
        name="W23 with one Full layer at depth 3",
        layer_contexts=DEFAULT_LAYER_CONTEXTS,
        checkpoint_path=args.hybrid_checkpoint_path,
    )
    window = UniformAttentionSpec(
        name="continuous global-RF local window (W=23, RF=133)",
        max_context=DEFAULT_LOCAL_CONTEXT,
        checkpoint_path=args.window_reference_checkpoint_path,
    )
    full = UniformAttentionSpec(
        name="full-128 Stage 26 reference",
        max_context=None,
        checkpoint_path=args.full_reference_checkpoint_path,
    )
    validate_attention_protocol(
        config,
        hybrid,
        window,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        hybrid_anchor_checkpoint_path=args.hybrid_anchor_checkpoint_path,
        window_anchor_checkpoint_path=args.window_anchor_reference_checkpoint_path,
    )
    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)
    if not _is_canonical_protocol(
        data,
        config,
        hybrid,
        window,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
    ):
        raise RuntimeError("the Stage 30 CLI is reserved for the canonical protocol")
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
        f"All: B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={DEFAULT_N_HEAD}, "
        f"D={config.n_embd // DEFAULT_N_HEAD}, FF={4 * config.n_embd}, "
        f"L={DEFAULT_N_LAYER}, dropout={DEFAULT_DROPOUT:g}"
    )
    print(f"Hybrid layer contexts: {DEFAULT_LAYER_CONTEXTS}")
    print(
        "Schedule by zero-based update: "
        f"[0, {schedule.first_decay_step:,}) lr={schedule.initial_learning_rate:g}; "
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
        hybrid,
        window,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        device,
        args.sample_length,
        schedule,
        args.hybrid_anchor_checkpoint_path,
        args.window_anchor_reference_checkpoint_path,
        training_batch_seed=DEFAULT_TRAINING_BATCH_SEED,
        training_rng_seed=DEFAULT_TRAINING_RNG_SEED,
        full_reference_expected_sha256=DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256,
        full_reference_expected_step=DEFAULT_FULL_REFERENCE_STEP,
        window_best_expected_sha256=DEFAULT_WINDOW_REFERENCE_CHECKPOINT_SHA256,
        window_anchor_expected_sha256=(
            DEFAULT_WINDOW_ANCHOR_REFERENCE_CHECKPOINT_SHA256
        ),
        window_best_expected_step=DEFAULT_WINDOW_REFERENCE_STEP,
        window_anchor_expected_step=DEFAULT_WINDOW_ANCHOR_REFERENCE_STEP,
        benchmark=benchmark,
    )
    if args.precise_eval_iters > 0:
        precise = run_paired_validation(
            data,
            config,
            hybrid,
            window,
            full,
            DEFAULT_N_HEAD,
            DEFAULT_N_LAYER,
            device,
            schedule,
            experiment,
            args.hybrid_anchor_checkpoint_path,
            args.window_anchor_reference_checkpoint_path,
            eval_iters=args.precise_eval_iters,
            seed=args.precise_eval_seed,
        )
        print_paired_validation(precise)
    _verify_reference_hashes(
        _reference_hashes(experiment.full, experiment.window),
        phase="during the complete Stage 30 CLI",
    )


if __name__ == "__main__":
    main()
