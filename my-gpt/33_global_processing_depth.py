from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import BenchmarkConfig, BenchmarkStats, seed_everything


# Stage 33 adds a third final Full-attention layer to Stage 32's Last2Full
# topology. Stage 32 remains the audited source for every earlier reference,
# while Stage 30 still supplies the generic layerwise-mask/training primitives.
_STAGE_32_PATH = Path(__file__).with_name("32_iterative_global_refinement.py")
_STAGE_32_MODULE_NAME = "stage_32_iterative_global_refinement_for_stage_33"
_STAGE_32_SPEC = importlib.util.spec_from_file_location(
    _STAGE_32_MODULE_NAME,
    _STAGE_32_PATH,
)
assert _STAGE_32_SPEC is not None and _STAGE_32_SPEC.loader is not None
_STAGE_32 = importlib.util.module_from_spec(_STAGE_32_SPEC)
sys.modules[_STAGE_32_MODULE_NAME] = _STAGE_32
_STAGE_32_SPEC.loader.exec_module(_STAGE_32)

_STAGE_31 = _STAGE_32._STAGE_31
_STAGE_30 = _STAGE_32._STAGE_30

_STAGE_29 = _STAGE_30._STAGE_29
_STAGE_27 = _STAGE_30._STAGE_27
_STAGE_18 = _STAGE_30._STAGE_18
_STAGE_26 = _STAGE_30._STAGE_26

GPTLanguageModel = _STAGE_30.GPTLanguageModel
LearningRateSchedule = _STAGE_30.LearningRateSchedule
BestValidationCheckpoint = _STAGE_30.BestValidationCheckpoint
PositionLosses = _STAGE_30.PositionLosses
AttentionEvaluationRecord = _STAGE_30.AttentionEvaluationRecord
AttentionTrainingResult = _STAGE_30.AttentionTrainingResult
PreciseBranchResult = _STAGE_30.PreciseBranchResult
PreciseDelta = _STAGE_30.PreciseDelta
RangeDelta = _STAGE_30.RangeDelta
UniformAttentionSpec = _STAGE_30.UniformAttentionSpec
FullReference = _STAGE_30.FullReference
PairedInitialStates = _STAGE_30.PairedInitialStates
WindowReference = _STAGE_30.WindowReference
LayerwiseReachabilityAudit = _STAGE_30.LayerwiseReachabilityAudit
HybridTrainingReport = _STAGE_30.HybridTrainingReport
LastThreeGlobalTrainingReport = HybridTrainingReport

checkpoint_sha256 = _STAGE_30.checkpoint_sha256
clear_accelerator_cache = _STAGE_30.clear_accelerator_cache
fingerprint_data = _STAGE_30.fingerprint_data
generate_from_final_model = _STAGE_30.generate_from_final_model
tensor_mapping_sha256 = _STAGE_30.tensor_mapping_sha256
tensor_sha256 = _STAGE_30.tensor_sha256
build_attention_mask = _STAGE_30.build_attention_mask
make_validation_panels = _STAGE_30.make_validation_panels
paired_position_range_delta = _STAGE_30.paired_position_range_delta


EXPERIMENT_STAGE = 33
EXPERIMENT_TITLE = "global processing depth"
COMPARISON_VARIABLE = (
    "three_final_global_layers_vs_two_final_global_layers_all_local_and_all_global"
)
PROTOCOL_NAME = "final_global_attention_depth_scaling"

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_LAST_THREE_GLOBAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_33_last_three_global_layers_4_5_6_at_128_best_checkpoint.pt"
)
DEFAULT_LAST_THREE_GLOBAL_ANCHOR_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_33_last_three_global_layers_4_5_6_at_128_step_13500_checkpoint.pt"
)
DEFAULT_LAST_TWO_GLOBAL_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_32_last_two_global_layers_5_6_at_128_best_checkpoint.pt"
)
DEFAULT_LAST_TWO_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_32_last_two_global_layers_5_6_at_128_step_13500_checkpoint.pt"
)
DEFAULT_LAST_GLOBAL_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_31_last_global_layer_6_at_128_best_checkpoint.pt"
)
DEFAULT_LAST_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_31_last_global_layer_6_at_128_step_13500_checkpoint.pt"
)
# Stages 31 and 30 are transitive dependencies of Stage 32's validator.
DEFAULT_MIDDLE_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_30_hybrid_layer_3_at_128_best_checkpoint.pt"
)
DEFAULT_MIDDLE_ANCHOR_REFERENCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_30_hybrid_layer_3_at_128_step_13500_checkpoint.pt"
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

# Compatibility names used by the earlier candidate/hybrid harnesses.
DEFAULT_HYBRID_CHECKPOINT_PATH = DEFAULT_LAST_THREE_GLOBAL_CHECKPOINT_PATH
DEFAULT_HYBRID_ANCHOR_CHECKPOINT_PATH = DEFAULT_LAST_THREE_GLOBAL_ANCHOR_CHECKPOINT_PATH
DEFAULT_LOCAL_CHECKPOINT_PATH = DEFAULT_LAST_THREE_GLOBAL_CHECKPOINT_PATH
DEFAULT_LOCAL_ANCHOR_CHECKPOINT_PATH = DEFAULT_LAST_THREE_GLOBAL_ANCHOR_CHECKPOINT_PATH

DEFAULT_BATCH_SIZE = 16
DEFAULT_BLOCK_SIZE = 128
DEFAULT_LOCAL_CONTEXT = 23
DEFAULT_LAYER_CONTEXTS = (23, 23, 23, None, None, None)
DEFAULT_LAST_THREE_GLOBAL_LAYER_CONTEXTS = DEFAULT_LAYER_CONTEXTS
DEFAULT_LAST_TWO_GLOBAL_LAYER_CONTEXTS = (23, 23, 23, 23, None, None)
DEFAULT_LAST_GLOBAL_LAYER_CONTEXTS = (23, 23, 23, 23, 23, None)
DEFAULT_MIDDLE_LAYER_CONTEXTS = (23, 23, None, 23, 23, 23)
DEFAULT_GLOBAL_LAYER_INDICES = (3, 4, 5)
DEFAULT_GLOBAL_LAYER_NUMBERS = (4, 5, 6)
DEFAULT_FIRST_GLOBAL_LAYER_INDEX = 3
DEFAULT_FIRST_GLOBAL_LAYER_NUMBER = 4
DEFAULT_LAST_GLOBAL_LAYER_INDEX = 5
DEFAULT_LAST_GLOBAL_LAYER_NUMBER = 6
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
# Stage 33 was selected after inspecting Stage 32's seed-1349 panels.
DEFAULT_PRECISE_EVAL_SEED = 1_350
DEFAULT_ACCELERATOR_NAME = "Intel(R) Arc(TM) 140T GPU (16GB)"

DROPOUT_PLACEMENT = _STAGE_30.DROPOUT_PLACEMENT
DEFAULT_DATA_FINGERPRINT = _STAGE_30.DEFAULT_DATA_FINGERPRINT
DEFAULT_PARAMETER_COUNT = _STAGE_30.DEFAULT_PARAMETER_COUNT
DEFAULT_PARAMETER_INITIALIZATION_SHA256 = (
    _STAGE_30.DEFAULT_PARAMETER_INITIALIZATION_SHA256
)
DEFAULT_FULL_INITIAL_STATE_SHA256 = _STAGE_30.DEFAULT_FULL_INITIAL_STATE_SHA256
DEFAULT_WINDOW_INITIAL_STATE_SHA256 = _STAGE_30.DEFAULT_WINDOW_INITIAL_STATE_SHA256
DEFAULT_LAST_THREE_GLOBAL_INITIAL_STATE_SHA256 = (
    "909a474abffc8459f0616f27ec95c3a1d47fa547622f620eaec324ac49e73a9d"
)
DEFAULT_HYBRID_INITIAL_STATE_SHA256 = DEFAULT_LAST_THREE_GLOBAL_INITIAL_STATE_SHA256
DEFAULT_LAST_THREE_GLOBAL_MASK_STACK_SHA256 = (
    "7d4a87eab60a08228dbe6b1b8f0f999d9e8a74af08b4d3d295512e192194d35b"
)
DEFAULT_HYBRID_MASK_STACK_SHA256 = DEFAULT_LAST_THREE_GLOBAL_MASK_STACK_SHA256
DEFAULT_LAST_THREE_GLOBAL_FLOAT_MASK_STACK_SHA256 = (
    "4291fe296032a817d8b14ac2f2322105d60e90c18fa105e3634226e16b2a0736"
)
DEFAULT_HYBRID_FLOAT_MASK_STACK_SHA256 = (
    DEFAULT_LAST_THREE_GLOBAL_FLOAT_MASK_STACK_SHA256
)
DEFAULT_LAST_THREE_GLOBAL_MASK_BUFFER_MAPPING_SHA256 = (
    "10d2d6aa5633701371e31616368914b1de8159a93840f8dce6e668cf9aa8775d"
)
DEFAULT_HYBRID_MASK_BUFFER_MAPPING_SHA256 = (
    DEFAULT_LAST_THREE_GLOBAL_MASK_BUFFER_MAPPING_SHA256
)
DEFAULT_FULL_MASK_STACK_SHA256 = _STAGE_30.DEFAULT_FULL_MASK_STACK_SHA256
DEFAULT_FULL_FLOAT_MASK_STACK_SHA256 = _STAGE_30.DEFAULT_FULL_FLOAT_MASK_STACK_SHA256
DEFAULT_WINDOW_MASK_SHA256 = _STAGE_30.DEFAULT_WINDOW_MASK_SHA256
DEFAULT_FULL_MASK_SHA256 = _STAGE_30.DEFAULT_FULL_MASK_SHA256
DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256 = (
    _STAGE_30.DEFAULT_FULL_REFERENCE_CHECKPOINT_SHA256
)
DEFAULT_FULL_REFERENCE_STEP = 13_500
DEFAULT_WINDOW_REFERENCE_CHECKPOINT_SHA256 = (
    _STAGE_30.DEFAULT_WINDOW_REFERENCE_CHECKPOINT_SHA256
)
DEFAULT_WINDOW_ANCHOR_REFERENCE_CHECKPOINT_SHA256 = (
    _STAGE_30.DEFAULT_WINDOW_ANCHOR_REFERENCE_CHECKPOINT_SHA256
)
DEFAULT_WINDOW_REFERENCE_STEP = 10_500
DEFAULT_WINDOW_ANCHOR_REFERENCE_STEP = 13_500
DEFAULT_LAST_TWO_GLOBAL_REFERENCE_CHECKPOINT_SHA256 = (
    "bbf72631da378d58d977d6e30daad6b5944d8a2edcc0b5318981d5b495d9e632"
)
DEFAULT_LAST_TWO_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_SHA256 = (
    "9a332f83436c510ba8bdf908e42e10a51f34fc3c02c4ac31c74efa13b6553db9"
)
DEFAULT_LAST_TWO_GLOBAL_REFERENCE_STEP = 16_500
DEFAULT_LAST_TWO_GLOBAL_ANCHOR_REFERENCE_STEP = 13_500
DEFAULT_LAST_GLOBAL_REFERENCE_CHECKPOINT_SHA256 = (
    _STAGE_32.DEFAULT_LAST_GLOBAL_REFERENCE_CHECKPOINT_SHA256
)
DEFAULT_LAST_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_SHA256 = (
    _STAGE_32.DEFAULT_LAST_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_SHA256
)
DEFAULT_LAST_GLOBAL_REFERENCE_STEP = _STAGE_32.DEFAULT_LAST_GLOBAL_REFERENCE_STEP
DEFAULT_LAST_GLOBAL_ANCHOR_REFERENCE_STEP = (
    _STAGE_32.DEFAULT_LAST_GLOBAL_ANCHOR_REFERENCE_STEP
)
DEFAULT_MIDDLE_REFERENCE_CHECKPOINT_SHA256 = (
    "dadb65bd5b8af294a3df8368e923afb02e90faceece88b074ecfa52bbd147800"
)
DEFAULT_MIDDLE_ANCHOR_REFERENCE_CHECKPOINT_SHA256 = (
    "90485453900376a5788958e53936aebdb0c418e474d0f1d895ffbf7617245d00"
)
DEFAULT_MIDDLE_REFERENCE_STEP = 10_500
DEFAULT_MIDDLE_ANCHOR_REFERENCE_STEP = 13_500
DEFAULT_MATCHED_STEP = 13_500
DEFAULT_DIAGNOSTIC_START = 23
DEFAULT_ALLOWED_EDGES_BY_LAYER_PER_HEAD = (
    2_691,
    2_691,
    2_691,
    8_256,
    8_256,
    8_256,
)
DEFAULT_ALLOWED_EDGES_PER_HEAD_ACROSS_DEPTH = 32_841
DEFAULT_FULL_EDGES_PER_HEAD_ACROSS_DEPTH = 49_536
DEFAULT_REMOVED_EDGES_PER_HEAD_ACROSS_DEPTH = 16_695
DEFAULT_DIFFERING_MASK_BUFFER_COUNT = 12
DEFAULT_DIFFERING_FROM_WINDOW_MASK_BUFFER_COUNT = 12
DEFAULT_DIFFERING_FROM_LAST_TWO_GLOBAL_MASK_BUFFER_COUNT = 4
DEFAULT_DIFFERING_FROM_LAST_GLOBAL_MASK_BUFFER_COUNT = 8
DEFAULT_DIFFERING_FROM_MIDDLE_MASK_BUFFER_COUNT = 16


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """Three consecutive final Full layers after uniform local layers."""

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
        if contexts.count(None) != 3:
            raise ValueError(
                "a three-global-layer branch must contain exactly three Full layers"
            )
        if contexts[-3:] != (None, None, None):
            raise ValueError("the three Full layers must be consecutive and final")
        bounded = tuple(context for context in contexts if context is not None)
        if not bounded or len(set(bounded)) != 1:
            raise ValueError("all remaining layers must use one common local context")

    @property
    def n_layer(self) -> int:
        return len(self.layer_contexts)

    @property
    def global_layer_index(self) -> int:
        """Compatibility alias for the first global layer index."""

        return self.global_layer_indices[0]

    @property
    def global_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, context in enumerate(self.layer_contexts)
            if context is None
        )

    @property
    def local_context(self) -> int:
        return next(context for context in self.layer_contexts if context is not None)

    @property
    def per_layer_contexts(self) -> tuple[int | None, ...]:
        return self.layer_contexts

    @property
    def mask_kind(self) -> str:
        # This value intentionally matches Stages 30-32 and their generic
        # model/mask implementation.
        return "layerwise_causal_sparse_global_shortcut"


LayerwiseAttentionSpec = AttentionSpec
HybridAttentionSpec = AttentionSpec
LastThreeGlobalAttentionSpec = AttentionSpec
LastTwoGlobalAttentionSpec = _STAGE_32.AttentionSpec


@dataclass(frozen=True, slots=True)
class LastTwoGlobalReference:
    spec: _STAGE_32.AttentionSpec
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
    initialization: PairedInitialStates
    last_global_reference: _STAGE_32.LastGlobalReference


LastReference = LastTwoGlobalReference


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    initialization: PairedInitialStates
    initialization_seed: int
    training_batch_seed: int
    training_rng_seed: int
    schedule: LearningRateSchedule
    bands: tuple[tuple[int, int], ...]
    reachability: LayerwiseReachabilityAudit
    last_three_global: LastThreeGlobalTrainingReport
    last_two_global: LastTwoGlobalReference
    full: FullReference
    window: WindowReference
    initial_shared_prefix_length: int
    initial_shared_prefix_max_difference: float
    last_three_global_benchmark: BenchmarkStats | None
    full_benchmark: BenchmarkStats | None
    last_three_global_checkpoint_sha256: str
    last_three_global_anchor_checkpoint_sha256: str

    @property
    def hybrid(self) -> LastThreeGlobalTrainingReport:
        return self.last_three_global

    @property
    def last(self) -> LastTwoGlobalReference:
        return self.last_two_global

    @property
    def last_two_global_reference(self) -> LastTwoGlobalReference:
        return self.last_two_global

    @property
    def local(self) -> LastThreeGlobalTrainingReport:
        return self.last_three_global

    @property
    def hybrid_benchmark(self) -> BenchmarkStats | None:
        return self.last_three_global_benchmark

    @property
    def hybrid_checkpoint_sha256(self) -> str:
        return self.last_three_global_checkpoint_sha256

    @property
    def hybrid_anchor_checkpoint_sha256(self) -> str:
        return self.last_three_global_anchor_checkpoint_sha256

    @property
    def local_checkpoint_sha256(self) -> str:
        return self.last_three_global_checkpoint_sha256

    @property
    def local_anchor_checkpoint_sha256(self) -> str:
        return self.last_three_global_anchor_checkpoint_sha256


@dataclass(frozen=True, slots=True)
class EndpointValidationReport:
    name: str
    full_natural: PreciseBranchResult
    last_three_global_natural: PreciseBranchResult
    last_two_global_natural: PreciseBranchResult
    window_natural: PreciseBranchResult
    last_three_global_minus_last_two_global: PreciseDelta
    full_minus_last_three_global: PreciseDelta
    last_three_global_minus_window: PreciseDelta
    full_minus_last_two_global: PreciseDelta
    last_two_global_minus_window: PreciseDelta
    full_minus_window: PreciseDelta
    last_three_global_minus_last_two_global_range: RangeDelta
    full_minus_last_three_global_range: RangeDelta
    last_three_global_minus_window_range: RangeDelta
    full_minus_last_two_global_range: RangeDelta
    last_two_global_minus_window_range: RangeDelta
    full_minus_window_range: RangeDelta
    last_three_global_recovery_fraction: float | None
    last_two_global_recovery_fraction: float | None

    @property
    def hybrid_natural(self) -> PreciseBranchResult:
        return self.last_three_global_natural

    @property
    def full_minus_hybrid(self) -> PreciseDelta:
        return self.full_minus_last_three_global

    @property
    def hybrid_minus_window(self) -> PreciseDelta:
        return self.last_three_global_minus_window

    @property
    def natural_delta(self) -> PreciseDelta:
        return self.last_three_global_minus_last_two_global

    @property
    def multi_hop_delta(self) -> RangeDelta:
        return self.last_three_global_minus_last_two_global_range


@dataclass(frozen=True, slots=True)
class Stage33PreciseValidationReport:
    eval_iters: int
    seed: int
    bands: tuple[tuple[int, int], ...]
    anchor: EndpointValidationReport
    selected_best: EndpointValidationReport


build_layer_attention_masks = _STAGE_30.build_layer_attention_masks
audit_layerwise_reachability = _STAGE_30.audit_layerwise_reachability
verify_global_reachability = _STAGE_30.verify_global_reachability
apply_attention_masks = _STAGE_30.apply_attention_masks
apply_attention_mask = _STAGE_30.apply_attention_mask
build_model = _STAGE_30.build_model
verify_model_masks = _STAGE_30.verify_model_masks
verify_model_mask = verify_model_masks
clone_parameters = _STAGE_30.clone_parameters
make_position_bands = _STAGE_30.make_position_bands
verify_initial_shared_prefix_equivalence = (
    _STAGE_30.verify_initial_shared_prefix_equivalence
)
inspect_attention_shape = _STAGE_30.inspect_attention_shape
run_counterbalanced_benchmarks = _STAGE_30.run_counterbalanced_benchmarks
_print_benchmark = _STAGE_30._print_benchmark
_serialize_losses = _STAGE_30._serialize_losses
_deserialize_losses = _STAGE_30._deserialize_losses
_require_mapping = _STAGE_30._require_mapping
_require_exact_metadata = _STAGE_30._require_exact_metadata
_evaluate_natural_on_panels = _STAGE_30._evaluate_natural_on_panels
_precise_result = _STAGE_30._precise_result
_evaluate_uniform_checkpoint = _STAGE_30._evaluate_uniform_checkpoint
_evaluate_hybrid_checkpoint = _STAGE_30._evaluate_hybrid_checkpoint
descriptive_recovery_fraction = _STAGE_30.descriptive_recovery_fraction
assert_matched_training_stream_state = _STAGE_30.assert_matched_training_stream_state


def validate_attention_protocol(
    config: TrainingConfig,
    last_three_global: AttentionSpec,
    last_two_global: _STAGE_32.AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int = DEFAULT_N_HEAD,
    n_layer: int = DEFAULT_N_LAYER,
    *,
    last_three_global_anchor_checkpoint_path: Path | None = None,
    last_two_global_anchor_checkpoint_path: Path | None = None,
    window_anchor_checkpoint_path: Path | None = None,
    last_global: _STAGE_31.AttentionSpec | None = None,
    last_global_anchor_checkpoint_path: Path | None = None,
    middle_global: _STAGE_31.AttentionSpec | None = None,
    middle_global_anchor_checkpoint_path: Path | None = None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Pin the four reported branches and isolate global-computation depth."""

    if len({last_three_global.name, last_two_global.name, window.name, full.name}) != 4:
        raise ValueError("Last3Full, Last2Full, window, and full names must differ")
    for name, spec in (
        ("Last3Full", last_three_global),
        ("Last2Full", last_two_global),
    ):
        if spec.n_layer != n_layer:
            raise ValueError(f"{name} layer_contexts length must equal n_layer")
    expected_candidate_indices = (n_layer - 3, n_layer - 2, n_layer - 1)
    if last_three_global.global_layer_indices != expected_candidate_indices:
        raise ValueError(
            "Stage 33 requires three consecutive final Full layers at indices "
            f"{expected_candidate_indices}"
        )
    expected_reference_indices = (n_layer - 2, n_layer - 1)
    if last_two_global.global_layer_indices != expected_reference_indices:
        raise ValueError(
            "the Stage 32 reference must have two consecutive final Full "
            f"layers at indices {expected_reference_indices}"
        )
    if last_three_global.local_context != last_two_global.local_context:
        raise ValueError("Last3Full and Last2Full must share one local window")
    if (last_global is None) != (last_global_anchor_checkpoint_path is None):
        raise ValueError(
            "the transitive LastFull spec and anchor must be provided together"
        )
    if (middle_global is None) != (middle_global_anchor_checkpoint_path is None):
        raise ValueError(
            "the transitive MidFull spec and anchor must be provided together"
        )
    if last_global is not None and middle_global is not None:
        _STAGE_32.validate_attention_protocol(
            config,
            last_two_global,
            last_global,
            window,
            full,
            n_head,
            n_layer,
            last_two_global_anchor_checkpoint_path=(
                last_two_global_anchor_checkpoint_path
            ),
            last_global_anchor_checkpoint_path=(last_global_anchor_checkpoint_path),
            window_anchor_checkpoint_path=window_anchor_checkpoint_path,
            middle_global=middle_global,
            middle_global_anchor_checkpoint_path=(middle_global_anchor_checkpoint_path),
        )
    elif last_global is not None or middle_global is not None:
        raise ValueError(
            "LastFull and MidFull transitive references must be supplied together"
        )
    if last_global is not None and last_global.name == last_three_global.name:
        raise ValueError("Last3Full and transitive LastFull names must differ")
    if middle_global is not None:
        if middle_global.n_layer != n_layer:
            raise ValueError("MidFull layer_contexts length must equal n_layer")
        expected_middle_index = (n_layer - 1) // 2
        if middle_global.global_layer_indices != (expected_middle_index,):
            raise ValueError(
                "the Stage 30 reference must place one Full layer at middle index "
                f"{expected_middle_index}"
            )
        if middle_global.local_context != last_three_global.local_context:
            raise ValueError("MidFull must share the same local window")
        if middle_global.name in {
            last_three_global.name,
            last_two_global.name,
            last_global.name,
            window.name,
            full.name,
        }:
            raise ValueError("all candidate and reference names must differ")
    if window.max_context is None or full.max_context is not None:
        raise ValueError("references must be bounded W23 and Full attention")
    if last_three_global.local_context != window.max_context:
        raise ValueError("all local layers must match the Stage 29 window")
    if last_three_global.local_context >= config.block_size:
        raise ValueError("local context must remain below block_size")

    expected_window = _STAGE_29.minimum_local_context_for_global_receptive_field(
        config.block_size,
        n_layer,
    )
    if window.max_context != expected_window:
        raise ValueError(
            "Stage 33 requires the Stage 29 minimum global-RF window "
            f"({expected_window}), got {window.max_context}"
        )
    if n_head <= 0 or config.n_embd % n_head:
        raise ValueError("n_embd must be divisible by a positive n_head")
    if n_layer <= 0:
        raise ValueError("n_layer must be positive")

    labelled_paths = {
        "Last3Full best": last_three_global.checkpoint_path,
        "Last2Full best": last_two_global.checkpoint_path,
        "window best": window.checkpoint_path,
        "full reference": full.checkpoint_path,
    }
    optional_paths = (
        ("Last3Full anchor", last_three_global_anchor_checkpoint_path),
        ("Last2Full anchor", last_two_global_anchor_checkpoint_path),
        (
            "LastFull best",
            last_global.checkpoint_path if last_global is not None else None,
        ),
        ("LastFull anchor", last_global_anchor_checkpoint_path),
        ("window anchor", window_anchor_checkpoint_path),
        (
            "MidFull best",
            middle_global.checkpoint_path if middle_global is not None else None,
        ),
        ("MidFull anchor", middle_global_anchor_checkpoint_path),
    )
    for label, path in optional_paths:
        if path is not None:
            labelled_paths[label] = path
    resolved: dict[str, Path] = {}
    for label, path in labelled_paths.items():
        resolved[label] = path.resolve()
        resolved[f"{label} temporary"] = path.with_name(f".{path.name}.tmp").resolve()
    labels = tuple(resolved)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            first = resolved[first_label]
            second = resolved[second_label]
            if first == second:
                raise ValueError(f"{first_label} and {second_label} must be distinct")
            if first.exists() and second.exists() and first.samefile(second):
                raise ValueError(
                    f"{first_label} and {second_label} must not be the same file"
                )

    candidate_masks = build_layer_attention_masks(config.block_size, last_three_global)
    candidate_audit = audit_layerwise_reachability(candidate_masks)
    verify_global_reachability(candidate_audit, config.block_size)
    reference_masks = build_layer_attention_masks(config.block_size, last_two_global)
    reference_audit = audit_layerwise_reachability(reference_masks)
    verify_global_reachability(reference_audit, config.block_size)
    expected_candidate_first_global_layer = (
        last_three_global.global_layer_indices[0] + 1
    )
    if (
        candidate_audit.first_global_reachability_layer
        != expected_candidate_first_global_layer
    ):
        raise RuntimeError(
            "the first Last3Full Full layer must create global exchange at layer "
            f"{expected_candidate_first_global_layer}"
        )
    expected_reference_first_global_layer = last_two_global.global_layer_indices[0] + 1
    if (
        reference_audit.first_global_reachability_layer
        != expected_reference_first_global_layer
    ):
        raise RuntimeError(
            "the first Last2Full Full layer must create global exchange at layer "
            f"{expected_reference_first_global_layer}"
        )
    bands = make_position_bands(
        config.block_size, last_three_global.local_context, n_layer
    )
    if bands[0] != (0, last_three_global.local_context):
        raise RuntimeError("the first band must be the exact shared-mask prefix")
    return config.batch_size * config.block_size, bands


def _is_canonical_protocol(
    data: CharacterData,
    config: TrainingConfig,
    last_three_global: AttentionSpec,
    last_two_global: _STAGE_32.AttentionSpec,
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
        and last_three_global.layer_contexts == DEFAULT_LAYER_CONTEXTS
        and last_two_global.layer_contexts == DEFAULT_LAST_TWO_GLOBAL_LAYER_CONTEXTS
        and window.max_context == DEFAULT_LOCAL_CONTEXT
        and full.max_context is None
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
    )


def create_paired_initial_states(
    data: CharacterData,
    config: TrainingConfig,
    last_three_global: AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
) -> PairedInitialStates:
    """Regenerate Stage 26 initialization with Full masks in layers 4, 5, and 6."""

    if last_three_global.n_layer != n_layer:
        raise ValueError("Last3Full layer_contexts length must equal n_layer")
    if last_three_global.global_layer_indices != (
        n_layer - 3,
        n_layer - 2,
        n_layer - 1,
    ):
        raise ValueError("Last3Full must place three consecutive Full masks last")
    if window.max_context is None or full.max_context is not None:
        raise ValueError("references must be bounded W23 and Full attention")
    if last_three_global.local_context != window.max_context:
        raise ValueError("Last3Full local layers must match the W23 reference")
    if last_three_global.local_context > config.block_size:
        raise ValueError("Last3Full local context exceeds block_size")
    if last_three_global.checkpoint_path.resolve() in {
        window.checkpoint_path.resolve(),
        full.checkpoint_path.resolve(),
    }:
        raise ValueError("Last3Full checkpoint must not alias a reference")

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
    last_state = {name: tensor.clone() for name, tensor in full_state.items()}
    masks = build_layer_attention_masks(config.block_size, last_three_global)
    mask_names = tuple(sorted(name for name in last_state if name.endswith(".tril")))
    if len(mask_names) != n_head * n_layer:
        raise RuntimeError("unexpected number of attention-mask buffers")
    for name in mask_names:
        layer = _STAGE_30._mask_layer_index(name)
        last_state[name] = masks[layer].to(dtype=last_state[name].dtype)

    differing = tuple(
        sorted(
            name
            for name in full_state
            if not torch.equal(last_state[name], full_state[name])
        )
    )
    expected_differing = tuple(
        name
        for name in mask_names
        if _STAGE_30._mask_layer_index(name)
        not in last_three_global.global_layer_indices
    )
    if differing != expected_differing:
        raise RuntimeError(
            "Last3Full and Full initial states must differ only in local-layer masks"
        )

    last_model = build_model(data, config, last_three_global, n_head, n_layer)
    last_model.load_state_dict(last_state, strict=True)
    full_model = _STAGE_29.build_model(data, config, full, n_head, n_layer)
    full_model.load_state_dict(full_state, strict=True)
    last_parameters = clone_parameters(last_model)
    full_parameters = clone_parameters(full_model)
    if last_parameters.keys() != full_parameters.keys() or any(
        not torch.equal(last_parameters[name], full_parameters[name])
        for name in last_parameters
    ):
        raise RuntimeError("Stage 33 learned initializations are not identical")
    parameter_count = sum(
        parameter.numel()
        for parameter in last_model.parameters()
        if parameter.requires_grad
    )
    del last_model, full_model

    mask_stack = torch.stack(masks)
    full_stack = torch.stack((build_attention_mask(config.block_size, None),) * n_layer)
    result = PairedInitialStates(
        hybrid=last_state,
        full=full_state,
        window=window_state,
        parameter_state_sha256=tensor_mapping_sha256(last_parameters),
        hybrid_state_sha256=tensor_mapping_sha256(last_state),
        full_state_sha256=tensor_mapping_sha256(full_state),
        window_state_sha256=tensor_mapping_sha256(window_state),
        hybrid_mask_stack_sha256=tensor_sha256(mask_stack),
        hybrid_float_mask_stack_sha256=tensor_sha256(mask_stack.float()),
        hybrid_mask_buffer_mapping_sha256=tensor_mapping_sha256(
            {name: last_state[name] for name in mask_names}
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
    is_candidate_canonical = (
        fingerprint_data(data) == DEFAULT_DATA_FINGERPRINT
        and config.batch_size == DEFAULT_BATCH_SIZE
        and config.block_size == DEFAULT_BLOCK_SIZE
        and config.n_embd == DEFAULT_N_EMBD
        and config.learning_rate == DEFAULT_INITIAL_LEARNING_RATE
        and config.max_iters == DEFAULT_MAX_ITERS
        and config.eval_interval == DEFAULT_EVAL_INTERVAL
        and config.eval_iters == DEFAULT_EVAL_ITERS
        and config.seed == DEFAULT_SEED
        and last_three_global.layer_contexts == DEFAULT_LAYER_CONTEXTS
        and window.max_context == DEFAULT_LOCAL_CONTEXT
        and full.max_context is None
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
    )
    if is_candidate_canonical:
        observed = {
            "parameter_count": result.parameter_count,
            "parameter_state": result.parameter_state_sha256,
            "last_three_global_state": result.hybrid_state_sha256,
            "full_state": result.full_state_sha256,
            "window_state": result.window_state_sha256,
            "last_three_global_mask_stack": result.hybrid_mask_stack_sha256,
            "last_three_global_float_mask_stack": result.hybrid_float_mask_stack_sha256,
            "last_three_global_mask_mapping": result.hybrid_mask_buffer_mapping_sha256,
            "differing_masks": len(result.differing_state_names),
        }
        expected = {
            "parameter_count": DEFAULT_PARAMETER_COUNT,
            "parameter_state": DEFAULT_PARAMETER_INITIALIZATION_SHA256,
            "last_three_global_state": DEFAULT_LAST_THREE_GLOBAL_INITIAL_STATE_SHA256,
            "full_state": DEFAULT_FULL_INITIAL_STATE_SHA256,
            "window_state": DEFAULT_WINDOW_INITIAL_STATE_SHA256,
            "last_three_global_mask_stack": DEFAULT_LAST_THREE_GLOBAL_MASK_STACK_SHA256,
            "last_three_global_float_mask_stack": (
                DEFAULT_LAST_THREE_GLOBAL_FLOAT_MASK_STACK_SHA256
            ),
            "last_three_global_mask_mapping": (
                DEFAULT_LAST_THREE_GLOBAL_MASK_BUFFER_MAPPING_SHA256
            ),
            "differing_masks": DEFAULT_DIFFERING_MASK_BUFFER_COUNT,
        }
        if observed != expected:
            raise RuntimeError(
                f"canonical Stage 33 initialization changed: expected {expected}, "
                f"got {observed}"
            )
    return result


def validate_stage_26_full_reference(*args: object, **kwargs: object) -> FullReference:
    return _STAGE_30.validate_stage_26_full_reference(*args, **kwargs)


def validate_stage_29_references(*args: object, **kwargs: object) -> WindowReference:
    return _STAGE_30.validate_stage_29_references(*args, **kwargs)


def validate_stage_32_references(
    data: CharacterData,
    config: TrainingConfig,
    last_two_global: _STAGE_32.AttentionSpec,
    last_two_global_anchor_path: Path,
    last_global: _STAGE_31.AttentionSpec,
    last_global_anchor_path: Path,
    middle_global: _STAGE_31.AttentionSpec,
    middle_global_anchor_path: Path,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    full_reference: FullReference,
    window_reference: WindowReference,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_best_sha256: str,
    expected_anchor_sha256: str,
    expected_best_step: int,
    expected_anchor_step: int,
    expected_last_global_best_sha256: str,
    expected_last_global_anchor_sha256: str,
    expected_last_global_best_step: int,
    expected_last_global_anchor_step: int,
    expected_middle_best_sha256: str,
    expected_middle_anchor_sha256: str,
    expected_middle_best_step: int,
    expected_middle_anchor_step: int,
) -> LastTwoGlobalReference:
    """Strictly validate Stage 32 Last2Full and all transitive provenance."""

    # Stage 32's validator must receive its own Last2Full initialization, not
    # Stage 33's Last3Full initialization. The learned tensors are shared, but
    # their immutable mask buffers intentionally differ at layer 4.
    last_two_initialization = _STAGE_32.create_paired_initial_states(
        data,
        config,
        last_two_global,
        window,
        full,
        n_head,
        n_layer,
    )
    last_global_reference = _STAGE_32.validate_stage_31_references(
        data,
        config,
        last_global,
        last_global_anchor_path,
        middle_global,
        middle_global_anchor_path,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        full_reference,
        window_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_best_sha256=expected_last_global_best_sha256,
        expected_anchor_sha256=expected_last_global_anchor_sha256,
        expected_best_step=expected_last_global_best_step,
        expected_anchor_step=expected_last_global_anchor_step,
        expected_middle_best_sha256=expected_middle_best_sha256,
        expected_middle_anchor_sha256=expected_middle_anchor_sha256,
        expected_middle_best_step=expected_middle_best_step,
        expected_middle_anchor_step=expected_middle_anchor_step,
    )
    expected = (
        (last_two_global.checkpoint_path, expected_best_sha256, "best"),
        (last_two_global_anchor_path, expected_anchor_sha256, "anchor"),
    )
    hashes_before: dict[Path, str] = {}
    checkpoints: dict[str, Mapping[str, object]] = {}
    for path, expected_hash, kind in expected:
        if not path.is_file():
            raise FileNotFoundError(f"Stage 32 {kind} reference not found: {path}")
        observed_hash = checkpoint_sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Stage 32 {kind} checkpoint SHA-256 mismatch: expected "
                f"{expected_hash}, got {observed_hash}"
            )
        hashes_before[path] = observed_hash
        checkpoints[kind] = _STAGE_32.validate_stage_32_checkpoint(
            path,
            data,
            config,
            last_two_global,
            last_global,
            window,
            full,
            n_head,
            n_layer,
            schedule,
            bands,
            last_two_initialization,
            full_reference,
            window_reference,
            last_global_reference,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            checkpoint_kind=kind,
        )

    best = checkpoints["best"]
    anchor = checkpoints["anchor"]
    if best.get("step") != expected_best_step:
        raise ValueError(
            f"Stage 32 best reference must be at step {expected_best_step}, "
            f"got {best.get('step')!r}"
        )
    if anchor.get("step") != expected_anchor_step:
        raise ValueError(
            f"Stage 32 anchor reference must be at step {expected_anchor_step}, "
            f"got {anchor.get('step')!r}"
        )
    # The matched-step artifact was saved at step 13,500, when its incumbent
    # best was step 11,500; the later selected best is step 16,500. The pinned
    # anchor SHA-256 and Stage 32's native validator authenticate that historical
    # selection state, so it must not be equated with the eventual best step.
    if expected_anchor_step != full_reference.checkpoint_step:
        raise ValueError("Stage 32 and Full matched reference steps must agree")
    if expected_anchor_step != window_reference.anchor_step:
        raise ValueError("Stage 32 and W23 matched reference steps must agree")
    if expected_anchor_step != last_global_reference.anchor_step:
        raise ValueError("Stage 32 and LastFull matched reference steps must agree")
    assert_matched_training_stream_state(full_reference.checkpoint, anchor)
    assert_matched_training_stream_state(window_reference.anchor_checkpoint, anchor)
    assert_matched_training_stream_state(
        last_global_reference.anchor_checkpoint,
        anchor,
    )

    best_training = _deserialize_losses(
        best.get("best_training_losses"), bands, name="Stage 32 best_training_losses"
    )
    best_validation = _deserialize_losses(
        best.get("best_validation_losses"),
        bands,
        name="Stage 32 best_validation_losses",
    )
    anchor_training = _deserialize_losses(
        anchor.get("anchor_training_losses"),
        bands,
        name="Stage 32 anchor_training_losses",
    )
    anchor_validation = _deserialize_losses(
        anchor.get("anchor_validation_losses"),
        bands,
        name="Stage 32 anchor_validation_losses",
    )
    for path, before_hash in hashes_before.items():
        if checkpoint_sha256(path) != before_hash:
            raise RuntimeError("Stage 32 reference changed during validation")
    return LastTwoGlobalReference(
        spec=last_two_global,
        anchor_path=last_two_global_anchor_path,
        best_checkpoint_sha256=hashes_before[last_two_global.checkpoint_path],
        anchor_checkpoint_sha256=hashes_before[last_two_global_anchor_path],
        best_step=expected_best_step,
        anchor_step=expected_anchor_step,
        best_training_losses=best_training,
        best_validation_losses=best_validation,
        anchor_training_losses=anchor_training,
        anchor_validation_losses=anchor_validation,
        best_checkpoint=best,
        anchor_checkpoint=anchor,
        initialization=last_two_initialization,
        last_global_reference=last_global_reference,
    )


def _allowed_edges_by_layer(
    config: TrainingConfig,
    last_three_global: AttentionSpec,
) -> tuple[int, ...]:
    return tuple(
        int(mask.sum().item())
        for mask in build_layer_attention_masks(config.block_size, last_three_global)
    )


def _architecture_metadata(
    config: TrainingConfig,
    last_three_global: AttentionSpec,
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
        "attention_mask": last_three_global.mask_kind,
        "attention_mask_scope": "explicit_policy_per_layer_every_head",
        "layer_contexts": last_three_global.layer_contexts,
        "per_layer_attention_windows": last_three_global.layer_contexts,
        "first_global_layer_index": last_three_global.global_layer_indices[0],
        "global_layer_indices": last_three_global.global_layer_indices,
        "global_layer_numbers": tuple(
            index + 1 for index in last_three_global.global_layer_indices
        ),
        "global_layer_placement": "consecutive_final_suffix",
        "global_attention_layers": len(last_three_global.global_layer_indices),
        "local_context": last_three_global.local_context,
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
    return _STAGE_30._training_metadata(
        config,
        schedule,
        step=step,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )


def _initialization_metadata(
    config: TrainingConfig,
    initialization: PairedInitialStates,
) -> dict[str, object]:
    return {
        "kind": "stage_26_t128_last_three_global_masks_matched",
        "seed": config.seed,
        "last_three_global_state_sha256": initialization.hybrid_state_sha256,
        "window_state_sha256": initialization.window_state_sha256,
        "full_state_sha256": initialization.full_state_sha256,
        "parameter_state_sha256": initialization.parameter_state_sha256,
        "identical_learned_tensors": True,
        "state_differs_only_in_attention_masks": True,
        "differing_mask_buffer_names": initialization.differing_state_names,
        "differing_mask_buffer_count": len(initialization.differing_state_names),
        "last_three_global_mask_stack_sha256": initialization.hybrid_mask_stack_sha256,
        "last_three_global_float_mask_stack_sha256": (
            initialization.hybrid_float_mask_stack_sha256
        ),
        "last_three_global_mask_buffer_mapping_sha256": (
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
    last_three_global: AttentionSpec,
    last_two_global: _STAGE_32.AttentionSpec,
    n_head: int,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    reachability: LayerwiseReachabilityAudit,
    full_reference: FullReference,
    window_reference: WindowReference,
    middle_reference: LastTwoGlobalReference,
    schedule: LearningRateSchedule,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
) -> dict[str, object]:
    allowed_by_layer = _allowed_edges_by_layer(config, last_three_global)
    full_edges = config.block_size * (config.block_size + 1) // 2
    reference_differences = tuple(
        sorted(
            name
            for name in initialization.hybrid
            if not torch.equal(
                initialization.hybrid[name],
                middle_reference.initialization.hybrid[name],
            )
        )
    )
    if any(not name.endswith(".tril") for name in reference_differences):
        raise RuntimeError("Last3Full and Last2Full differ in learned state")
    return {
        "stage": EXPERIMENT_STAGE,
        "protocol": PROTOCOL_NAME,
        "branch": last_three_global.name,
        "from_scratch": True,
        "comparison_variable": COMPARISON_VARIABLE,
        "layer_contexts": last_three_global.layer_contexts,
        "per_layer_attention_windows": last_three_global.layer_contexts,
        "first_global_layer_index": last_three_global.global_layer_indices[0],
        "global_layer_indices": last_three_global.global_layer_indices,
        "global_layer_numbers": tuple(
            index + 1 for index in last_three_global.global_layer_indices
        ),
        "global_layer_placement": "consecutive_final_suffix",
        "global_attention_layers": 3,
        "local_attention_layers": last_three_global.n_layer - 3,
        "local_layers_before_global": last_three_global.n_layer - 3,
        "local_layers_after_global": 0,
        "local_context": last_three_global.local_context,
        "last_two_global_reference_layer_contexts": last_two_global.layer_contexts,
        "last_two_global_reference_global_layer_indices": (
            last_two_global.global_layer_indices
        ),
        "last_two_global_reference_global_layer_numbers": tuple(
            index + 1 for index in last_two_global.global_layer_indices
        ),
        "last_two_global_reference_local_layers_after_global": (0),
        "last_three_vs_last_two_differing_mask_buffer_names": (reference_differences),
        "last_three_vs_last_two_differing_mask_buffer_count": (
            len(reference_differences)
        ),
        "first_query_with_topology_difference": last_three_global.local_context,
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
        "intervention_changes_global_layer_placement": False,
        "intervention_changes_number_of_global_layers": True,
        "intervention_adds_iterative_global_computation": True,
        "intervention_adds_third_global_processing_round": True,
        "intervention_changes_reachable_source_origins_at_final_layer": False,
        "no_post_global_local_processing": True,
        "allowed_attention_edges_by_layer_per_head": allowed_by_layer,
        "allowed_attention_edges_per_head_across_depth": sum(allowed_by_layer),
        "removed_attention_edges_per_head_across_depth": (
            last_three_global.n_layer * full_edges - sum(allowed_by_layer)
        ),
        "same_edge_count_as_last_two_global_reference": False,
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
            last_three_global.local_context,
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
        "last_two_global_reference_source_stage": 32,
        "last_two_global_best_reference_checkpoint_sha256": (
            middle_reference.best_checkpoint_sha256
        ),
        "last_two_global_best_reference_step": middle_reference.best_step,
        "last_two_global_anchor_reference_checkpoint_sha256": (
            middle_reference.anchor_checkpoint_sha256
        ),
        "last_two_global_anchor_reference_step": middle_reference.anchor_step,
        "last_global_reference_source_stage": 31,
        "last_global_best_reference_checkpoint_sha256": (
            middle_reference.last_global_reference.best_checkpoint_sha256
        ),
        "last_global_best_reference_step": (
            middle_reference.last_global_reference.best_step
        ),
        "last_global_anchor_reference_checkpoint_sha256": (
            middle_reference.last_global_reference.anchor_checkpoint_sha256
        ),
        "last_global_anchor_reference_step": (
            middle_reference.last_global_reference.anchor_step
        ),
        "middle_reference_source_stage": 30,
        "middle_best_reference_checkpoint_sha256": (
            middle_reference.last_global_reference.middle_reference.best_checkpoint_sha256
        ),
        "middle_best_reference_step": (
            middle_reference.last_global_reference.middle_reference.best_step
        ),
        "middle_anchor_reference_checkpoint_sha256": (
            middle_reference.last_global_reference.middle_reference.anchor_checkpoint_sha256
        ),
        "middle_anchor_reference_step": (
            middle_reference.last_global_reference.middle_reference.anchor_step
        ),
        "all_references_reused_read_only": True,
        "only_last_three_global_branch_newly_trained": True,
        "initial_parameter_count": initialization.parameter_count,
        "initial_mask_buffer_difference_count_vs_full": len(
            initialization.differing_state_names
        ),
    }


def _evaluation_metadata(
    config: TrainingConfig,
    last_three_global: AttentionSpec,
    bands: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    return {
        "checkpoint_selection_objective": "overall_validation_loss",
        "primary_endpoint": (
            "matched_step_last_three_global_minus_last_two_global_natural_overall"
        ),
        "primary_comparison": "last_three_global_minus_last_two_global",
        "secondary_comparisons": (
            "full_minus_last_three_global",
            "last_three_global_minus_window",
            "full_minus_last_two_global",
            "last_two_global_minus_window",
            "full_minus_window",
        ),
        "selected_best_endpoint": "secondary",
        "predeclared_mechanistic_diagnostic": (
            f"positions_{last_three_global.local_context}_through_"
            f"{config.block_size - 1}_aggregate_loss"
        ),
        "mechanistic_diagnostic_position_range": (
            last_three_global.local_context,
            config.block_size,
        ),
        "descriptive_last_three_global_recovery_fraction": (
            "(window_loss-last_three_global_loss)/(window_loss-full_loss)"
        ),
        "descriptive_last_two_global_recovery_fraction": (
            "(window_loss-last_two_global_loss)/(window_loss-full_loss)"
        ),
        "recovery_fraction_inferential_status": "descriptive_only",
        "same_weight_mask_ablations": False,
        "crossed_mask_variants": False,
        "held_out_validation_natural_masks_only": True,
        "initial_exact_prefix": (0, last_three_global.local_context),
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
class LastThreeGlobalValidationCheckpoint(BestValidationCheckpoint):
    last_three_global_spec: AttentionSpec | None = None
    last_two_global_spec: _STAGE_32.AttentionSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initialization: PairedInitialStates | None = None
    reachability: LayerwiseReachabilityAudit | None = None
    full_reference: FullReference | None = None
    window_reference: WindowReference | None = None
    middle_reference: LastTwoGlobalReference | None = None
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
            self.last_three_global_spec is None
            or self.last_two_global_spec is None
            or self.schedule is None
            or self.initialization is None
            or self.reachability is None
            or self.full_reference is None
            or self.window_reference is None
            or self.middle_reference is None
            or self.best_training_losses is None
            or self.best_validation_losses is None
            or not self.bands
        ):
            raise RuntimeError("incomplete Stage 33 checkpoint metadata")
        rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        expected_rate = self.schedule.for_update(max(0, step - 1))
        if rates != {expected_rate}:
            raise RuntimeError("checkpoint optimizer learning rate is inconsistent")
        payload["checkpoint_kind"] = checkpoint_kind
        payload["architecture"] = _architecture_metadata(
            self.config,
            self.last_three_global_spec,
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
            self.last_three_global_spec,
            self.last_two_global_spec,
            self.n_head,
            self.bands,
            self.initialization,
            self.reachability,
            self.full_reference,
            self.window_reference,
            self.middle_reference,
            self.schedule,
            training_batch_seed=self.training_batch_seed,
            training_rng_seed=self.training_rng_seed,
        )
        payload["evaluation_protocol"] = _evaluation_metadata(
            self.config,
            self.last_three_global_spec,
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
            super()._payload(step),
            step=step,
            checkpoint_kind="best",
        )

    def _save_anchor(self, record: AttentionEvaluationRecord) -> None:
        if self.anchor_path is None:
            raise RuntimeError("anchor checkpoint path is unavailable")
        self.anchor_training_losses = record.losses.train
        self.anchor_validation_losses = record.losses.val
        payload = self._decorate_payload(
            super()._payload(record.step),
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


HybridValidationCheckpoint = LastThreeGlobalValidationCheckpoint


def _reference_hashes(
    full_reference: FullReference,
    window_reference: WindowReference,
    middle_reference: LastTwoGlobalReference,
) -> dict[Path, str]:
    last_global = middle_reference.last_global_reference
    middle = last_global.middle_reference
    return {
        full_reference.spec.checkpoint_path: full_reference.checkpoint_sha256,
        window_reference.spec.checkpoint_path: (
            window_reference.best_checkpoint_sha256
        ),
        window_reference.anchor_path: window_reference.anchor_checkpoint_sha256,
        middle_reference.spec.checkpoint_path: (
            middle_reference.best_checkpoint_sha256
        ),
        middle_reference.anchor_path: middle_reference.anchor_checkpoint_sha256,
        last_global.spec.checkpoint_path: last_global.best_checkpoint_sha256,
        last_global.anchor_path: last_global.anchor_checkpoint_sha256,
        middle.spec.checkpoint_path: middle.best_checkpoint_sha256,
        middle.anchor_path: middle.anchor_checkpoint_sha256,
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


def validate_stage_33_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    last_three_global: AttentionSpec,
    last_two_global: _STAGE_32.AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    full_reference: FullReference,
    window_reference: WindowReference,
    middle_reference: LastTwoGlobalReference,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    checkpoint_kind: str,
) -> Mapping[str, object]:
    """Strictly validate a selected-best or matched-step Stage 33 artifact."""

    if checkpoint_kind not in {"best", "anchor"}:
        raise ValueError("checkpoint_kind must be 'best' or 'anchor'")
    if not path.is_file():
        raise FileNotFoundError(f"Stage 33 checkpoint not found: {path}")
    _, expected_bands = validate_attention_protocol(
        config,
        last_three_global,
        last_two_global,
        window,
        full,
        n_head,
        n_layer,
    )
    if bands != expected_bands:
        raise ValueError("checkpoint bands differ from the Stage 33 protocol")
    reachability = audit_layerwise_reachability(
        build_layer_attention_masks(config.block_size, last_three_global)
    )
    verify_global_reachability(reachability, config.block_size)

    reference_hashes = _reference_hashes(
        full_reference,
        window_reference,
        middle_reference,
    )
    if any(
        path.resolve() == reference_path.resolve()
        for reference_path in reference_hashes
    ):
        raise ValueError(
            "Stage 33 checkpoint path must not alias a read-only reference"
        )
    _verify_reference_hashes(reference_hashes, phase="before checkpoint validation")

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
            "checkpoint root keys differ from the Stage 33 contract; "
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
        raise ValueError("anchor_step does not match the W23 reference")
    if checkpoint["anchor_step"] != middle_reference.anchor_step:
        raise ValueError("anchor_step does not match the Last2Full reference")
    if checkpoint_kind == "anchor" and step != middle_reference.anchor_step:
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
        raise ValueError("Stage 33 must not restart the optimizer")
    if checkpoint["optimizer_provenance_known"] is not True:
        raise ValueError("Stage 33 optimizer provenance must be known")
    _STAGE_27._validate_rng_state(checkpoint)

    architecture = _require_mapping(checkpoint["architecture"], name="architecture")
    _require_exact_metadata(
        architecture,
        _architecture_metadata(
            config,
            last_three_global,
            n_head,
            n_layer,
            initialization,
            reachability,
        ),
        name="architecture",
    )
    training = _require_mapping(checkpoint["training_config"], name="training_config")
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
            last_three_global,
            last_two_global,
            n_head,
            bands,
            initialization,
            reachability,
            full_reference,
            window_reference,
            middle_reference,
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
        _evaluation_metadata(config, last_three_global, bands),
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
    del best_training
    if best_validation.overall != float(best_val_loss):
        raise ValueError("best validation loss does not match best_val_loss")
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

    model_state = _require_mapping(
        checkpoint["model_state_dict"],
        name="model_state_dict",
    )
    mask_state = {
        name: tensor
        for name, tensor in model_state.items()
        if name.endswith(".tril") and isinstance(tensor, torch.Tensor)
    }
    if (
        tensor_mapping_sha256(mask_state)
        != initialization.hybrid_mask_buffer_mapping_sha256
    ):
        raise ValueError("checkpoint attention-mask buffer fingerprint is incorrect")
    compatibility_model = build_model(
        data,
        config,
        last_three_global,
        n_head,
        n_layer,
    )
    compatibility_model.load_state_dict(model_state, strict=True)
    verify_model_masks(compatibility_model, last_three_global)
    del compatibility_model

    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError("Stage 33 checkpoint changed during validation")
    _verify_reference_hashes(reference_hashes, phase="during checkpoint validation")
    return checkpoint


validate_checkpoint = validate_stage_33_checkpoint


def run_last_three_global_branch(
    data: CharacterData,
    config: TrainingConfig,
    last_three_global: AttentionSpec,
    last_two_global: _STAGE_32.AttentionSpec,
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
    middle_reference: LastTwoGlobalReference,
    last_three_global_anchor_checkpoint_path: Path,
    reachability: LayerwiseReachabilityAudit,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    on_update: Callable[[int, float, torch.Tensor, torch.Tensor], None] | None = None,
) -> LastThreeGlobalTrainingReport:
    """Train Last3Full from step zero; keep Stages 26/29/30/31/32 read-only."""

    del last_two_global, window, full
    clear_accelerator_cache(device)
    model = build_model(data, config, last_three_global, n_head, n_layer, device)
    model.load_state_dict(initialization.hybrid, strict=True)
    verify_model_masks(model, last_three_global)
    if tensor_mapping_sha256(model.state_dict()) != initialization.hybrid_state_sha256:
        raise RuntimeError("Last3Full did not load its exact initial state")
    if (
        tensor_mapping_sha256(clone_parameters(model))
        != initialization.parameter_state_sha256
    ):
        raise RuntimeError("Last3Full learned initialization changed")
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.for_update(0))
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    best = LastThreeGlobalValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=last_three_global.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        last_three_global_spec=last_three_global,
        last_two_global_spec=middle_reference.spec,
        schedule=schedule,
        target_step=config.max_iters,
        initialization=initialization,
        reachability=reachability,
        full_reference=full_reference,
        window_reference=window_reference,
        middle_reference=middle_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        bands=bands,
        anchor_path=last_three_global_anchor_checkpoint_path,
        anchor_step=full_reference.checkpoint_step,
    )
    record_evaluation = _STAGE_18.make_evaluation_callback(best, schedule)

    print(
        f"\n{last_three_global.name} branch (B={config.batch_size}, "
        f"T={config.block_size}, contexts={last_three_global.layer_contexts}, "
        f"dropout={DEFAULT_DROPOUT:g})"
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
        raise RuntimeError("Stage 33 checkpoint selection is incomplete")
    attention_shape = inspect_attention_shape(
        model,
        config,
        last_three_global,
        n_head,
        device,
    )
    final_generator_hash = tensor_sha256(training_generator.get_state())

    selected = _require_mapping(
        torch.load(
            last_three_global.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        ),
        name="selected Last3Full checkpoint",
    )
    selected_state = _require_mapping(
        selected.get("model_state_dict"),
        name="selected Last3Full model_state_dict",
    )
    model.load_state_dict(selected_state, strict=True)
    verify_model_masks(model, last_three_global)
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=DEFAULT_SAMPLE_SEED,
    )
    report = LastThreeGlobalTrainingReport(
        spec=last_three_global,
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


run_hybrid_branch = run_last_three_global_branch


def _print_experiment_summary(
    config: TrainingConfig,
    report: ExperimentReport,
    last_three_global_anchor_path: Path,
) -> None:
    last = report.last_three_global
    middle = report.last_two_global
    full = report.full
    window = report.window
    matched_last_middle = (
        last.anchor_validation_losses.overall - middle.anchor_validation_losses.overall
    )
    matched_full_last = (
        full.best_validation_losses.overall - last.anchor_validation_losses.overall
    )
    matched_last_window = (
        last.anchor_validation_losses.overall - window.anchor_validation_losses.overall
    )
    print(f"\nStage {EXPERIMENT_STAGE} experimental summary")
    print(f"  all branches B/T:               {config.batch_size}/{config.block_size}")
    print(f"  Last3Full contexts:              {last.spec.layer_contexts}")
    print(f"  Last2Full contexts:               {middle.spec.layer_contexts}")
    print(
        "  Last3Full RF widths/layer:      "
        + ", ".join(str(value) for value in report.reachability.widths_by_layer)
    )
    print(
        "  first global reachability:      layer "
        f"{report.reachability.first_global_reachability_layer}"
    )
    print(f"  parameter count (all):          {last.parameter_count:,}")
    print(f"  attention shape (all):          {last.attention_shape}")
    print(
        "  Last3Full best train/val/gap:    "
        f"{last.best_training_losses.overall:.4f} / "
        f"{last.best_validation_losses.overall:.4f} / "
        f"{last.best_gap:.4f} @ {last.best_step}"
    )
    print(
        "  Last2Full best train/val:         "
        f"{middle.best_training_losses.overall:.4f} / "
        f"{middle.best_validation_losses.overall:.4f} @ {middle.best_step}"
    )
    print(
        "  W23 best train/val:             "
        f"{window.best_training_losses.overall:.4f} / "
        f"{window.best_validation_losses.overall:.4f} @ {window.best_step}"
    )
    print(
        "  Full reference train/val:       "
        f"{full.best_training_losses.overall:.4f} / "
        f"{full.best_validation_losses.overall:.4f} @ {full.checkpoint_step}"
    )
    print(f"  matched Last3Full-Last2Full delta:  {matched_last_middle:+.4f}")
    print(f"  matched Full-Last3Full delta:     {matched_full_last:+.4f}")
    print(f"  matched Last3Full-W23 delta:      {matched_last_window:+.4f}")
    print("  Negative candidate-baseline deltas favor the named candidate.")
    print(f"  Last3Full best checkpoint:        {last.spec.checkpoint_path}")
    print(
        f"  Last3Full best SHA-256:           {report.last_three_global_checkpoint_sha256}"
    )
    print(f"  Last3Full matched anchor:         {last_three_global_anchor_path}")
    print(
        "  Last3Full anchor SHA-256:         "
        f"{report.last_three_global_anchor_checkpoint_sha256}"
    )
    print(f"  Last2Full references:              {middle.spec.checkpoint_path}")
    print(f"  W23 references:                  {window.spec.checkpoint_path}")
    print(f"  Full reference:                  {full.spec.checkpoint_path}")
    print(f"\n{last.spec.name} selected-best generated text")
    print(last.sample)


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    last_three_global: AttentionSpec,
    last_two_global: _STAGE_32.AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    last_three_global_anchor_checkpoint_path: Path,
    last_two_global_anchor_checkpoint_path: Path,
    window_anchor_checkpoint_path: Path,
    *,
    last_global: _STAGE_31.AttentionSpec,
    last_global_anchor_checkpoint_path: Path,
    middle_global: _STAGE_31.AttentionSpec,
    middle_global_anchor_checkpoint_path: Path,
    training_batch_seed: int,
    training_rng_seed: int,
    full_reference_expected_sha256: str,
    full_reference_expected_step: int,
    window_best_expected_sha256: str,
    window_anchor_expected_sha256: str,
    window_best_expected_step: int,
    window_anchor_expected_step: int,
    last_two_global_best_expected_sha256: str,
    last_two_global_anchor_expected_sha256: str,
    last_two_global_best_expected_step: int,
    last_two_global_anchor_expected_step: int,
    last_global_best_expected_sha256: str,
    last_global_anchor_expected_sha256: str,
    last_global_best_expected_step: int,
    last_global_anchor_expected_step: int,
    middle_best_expected_sha256: str,
    middle_anchor_expected_sha256: str,
    middle_best_expected_step: int,
    middle_anchor_expected_step: int,
    benchmark: BenchmarkConfig | None = None,
    on_update: Callable[[int, float, torch.Tensor, torch.Tensor], None] | None = None,
) -> ExperimentReport:
    """Train Last3Full while authenticating every earlier artifact read-only."""

    _, bands = validate_attention_protocol(
        config,
        last_three_global,
        last_two_global,
        window,
        full,
        n_head,
        n_layer,
        last_three_global_anchor_checkpoint_path=last_three_global_anchor_checkpoint_path,
        last_two_global_anchor_checkpoint_path=last_two_global_anchor_checkpoint_path,
        window_anchor_checkpoint_path=window_anchor_checkpoint_path,
        last_global=last_global,
        last_global_anchor_checkpoint_path=last_global_anchor_checkpoint_path,
        middle_global=middle_global,
        middle_global_anchor_checkpoint_path=middle_global_anchor_checkpoint_path,
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
        last_three_global,
        window,
        full,
        n_head,
        n_layer,
    )
    reachability = audit_layerwise_reachability(
        build_layer_attention_masks(config.block_size, last_three_global)
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
    middle_reference = validate_stage_32_references(
        data,
        config,
        last_two_global,
        last_two_global_anchor_checkpoint_path,
        last_global,
        last_global_anchor_checkpoint_path,
        middle_global,
        middle_global_anchor_checkpoint_path,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        full_reference,
        window_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_best_sha256=last_two_global_best_expected_sha256,
        expected_anchor_sha256=last_two_global_anchor_expected_sha256,
        expected_best_step=last_two_global_best_expected_step,
        expected_anchor_step=last_two_global_anchor_expected_step,
        expected_last_global_best_sha256=last_global_best_expected_sha256,
        expected_last_global_anchor_sha256=last_global_anchor_expected_sha256,
        expected_last_global_best_step=last_global_best_expected_step,
        expected_last_global_anchor_step=last_global_anchor_expected_step,
        expected_middle_best_sha256=middle_best_expected_sha256,
        expected_middle_anchor_sha256=middle_anchor_expected_sha256,
        expected_middle_best_step=middle_best_expected_step,
        expected_middle_anchor_step=middle_anchor_expected_step,
    )
    if (
        len(
            tuple(
                name
                for name in initialization.hybrid
                if not torch.equal(
                    initialization.hybrid[name],
                    middle_reference.initialization.hybrid[name],
                )
            )
        )
        != n_head
    ):
        raise RuntimeError("Last3Full and Last2Full must differ in exactly one layer")
    references = _reference_hashes(
        full_reference,
        window_reference,
        middle_reference,
    )
    _verify_reference_hashes(references, phase="before Stage 33 training")

    last_audit_model = build_model(
        data,
        config,
        last_three_global,
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
    last_audit_model.load_state_dict(initialization.hybrid, strict=True)
    full_audit_model.load_state_dict(initialization.full, strict=True)
    initial_difference = verify_initial_shared_prefix_equivalence(
        last_audit_model,
        full_audit_model,
        data,
        config,
        torch.device("cpu"),
        seed=DEFAULT_INITIAL_AUDIT_SEED,
        prefix_length=last_three_global.local_context,
    )
    del last_audit_model, full_audit_model

    print("\nControlled final-global-depth protocol")
    print(
        "  Stages 26 Full, 29 W23, 30 MidFull, 31 LastFull, and 32 "
        "Last2Full are read-only."
    )
    print("  Only Last3Full (Full attention at layers 4-6) is trained from step zero.")
    print(f"  Last3Full contexts:           {last_three_global.layer_contexts}")
    print(f"  Last2Full contexts:            {last_two_global.layer_contexts}")
    print(f"  learned parameter SHA-256:  {initialization.parameter_state_sha256}")
    print(f"  Last3Full initial SHA-256:   {initialization.hybrid_state_sha256}")
    print(f"  mask differences vs Full:   {len(initialization.differing_state_names)}")
    print(f"  RF widths after layers:     {reachability.widths_by_layer}")
    print(
        f"  Initial [0,{last_three_global.local_context}) max logit difference: "
        f"{initial_difference:.3e}"
    )

    last_benchmark: BenchmarkStats | None = None
    full_benchmark: BenchmarkStats | None = None
    if benchmark is not None:
        print("\nCounterbalanced disposable dense training benchmark")
        last_benchmark, full_benchmark = run_counterbalanced_benchmarks(
            data,
            config,
            last_three_global,
            full,
            n_head,
            n_layer,
            device,
            initialization,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            benchmark=benchmark,
        )
        _print_benchmark(last_three_global.name, last_benchmark)
        _print_benchmark(full.name, full_benchmark)
        _verify_reference_hashes(references, phase="during benchmarking")

    last_report = run_last_three_global_branch(
        data,
        config,
        last_three_global,
        last_two_global,
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
        middle_reference,
        last_three_global_anchor_checkpoint_path,
        reachability,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        on_update=on_update,
    )
    last_report = replace(last_report, benchmark=last_benchmark)
    _verify_reference_hashes(references, phase="during Last3Full training")

    validate_stage_33_checkpoint(
        last_three_global.checkpoint_path,
        data,
        config,
        last_three_global,
        last_two_global,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        initialization,
        full_reference,
        window_reference,
        middle_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        checkpoint_kind="best",
    )
    anchor_checkpoint = validate_stage_33_checkpoint(
        last_three_global_anchor_checkpoint_path,
        data,
        config,
        last_three_global,
        last_two_global,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        bands,
        initialization,
        full_reference,
        window_reference,
        middle_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        checkpoint_kind="anchor",
    )
    for reference_checkpoint in (
        full_reference.checkpoint,
        window_reference.anchor_checkpoint,
        middle_reference.last_global_reference.middle_reference.anchor_checkpoint,
        middle_reference.last_global_reference.anchor_checkpoint,
        middle_reference.anchor_checkpoint,
    ):
        assert_matched_training_stream_state(reference_checkpoint, anchor_checkpoint)
    print(
        "  Full/W23/MidFull/LastFull/Last2Full/Last3Full streams: identical at anchor"
    )
    last_hash = checkpoint_sha256(last_three_global.checkpoint_path)
    last_anchor_hash = checkpoint_sha256(last_three_global_anchor_checkpoint_path)
    _verify_reference_hashes(references, phase="during checkpoint validation")

    report = ExperimentReport(
        initialization=initialization,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        bands=bands,
        reachability=reachability,
        last_three_global=last_report,
        last_two_global=middle_reference,
        full=full_reference,
        window=window_reference,
        initial_shared_prefix_length=last_three_global.local_context,
        initial_shared_prefix_max_difference=initial_difference,
        last_three_global_benchmark=last_benchmark,
        full_benchmark=full_benchmark,
        last_three_global_checkpoint_sha256=last_hash,
        last_three_global_anchor_checkpoint_sha256=last_anchor_hash,
    )
    _print_experiment_summary(config, report, last_three_global_anchor_checkpoint_path)
    return report


ValidationPanel = tuple[torch.Tensor, torch.Tensor]


def _make_endpoint_report(
    name: str,
    full_natural: PreciseBranchResult,
    last_three_global_natural: PreciseBranchResult,
    last_two_global_natural: PreciseBranchResult,
    window_natural: PreciseBranchResult,
    bands: tuple[tuple[int, int], ...],
    diagnostic_start: int,
) -> EndpointValidationReport:
    pairs = {
        "last_middle": (last_three_global_natural, last_two_global_natural),
        "full_last": (full_natural, last_three_global_natural),
        "last_window": (last_three_global_natural, window_natural),
        "full_middle": (full_natural, last_two_global_natural),
        "middle_window": (last_two_global_natural, window_natural),
        "full_window": (full_natural, window_natural),
    }
    deltas = {
        key: _STAGE_18._paired_delta(candidate, baseline)
        for key, (candidate, baseline) in pairs.items()
    }
    ranges = {
        key: paired_position_range_delta(
            candidate,
            baseline,
            bands,
            diagnostic_start,
            bands[-1][1],
        )
        for key, (candidate, baseline) in pairs.items()
    }
    return EndpointValidationReport(
        name=name,
        full_natural=full_natural,
        last_three_global_natural=last_three_global_natural,
        last_two_global_natural=last_two_global_natural,
        window_natural=window_natural,
        last_three_global_minus_last_two_global=deltas["last_middle"],
        full_minus_last_three_global=deltas["full_last"],
        last_three_global_minus_window=deltas["last_window"],
        full_minus_last_two_global=deltas["full_middle"],
        last_two_global_minus_window=deltas["middle_window"],
        full_minus_window=deltas["full_window"],
        last_three_global_minus_last_two_global_range=ranges["last_middle"],
        full_minus_last_three_global_range=ranges["full_last"],
        last_three_global_minus_window_range=ranges["last_window"],
        full_minus_last_two_global_range=ranges["full_middle"],
        last_two_global_minus_window_range=ranges["middle_window"],
        full_minus_window_range=ranges["full_window"],
        last_three_global_recovery_fraction=descriptive_recovery_fraction(
            deltas["full_window"].mean_delta.overall,
            deltas["last_window"].mean_delta.overall,
        ),
        last_two_global_recovery_fraction=descriptive_recovery_fraction(
            deltas["full_window"].mean_delta.overall,
            deltas["middle_window"].mean_delta.overall,
        ),
    )


def run_paired_validation(
    data: CharacterData,
    config: TrainingConfig,
    last_three_global: AttentionSpec,
    last_two_global: _STAGE_32.AttentionSpec,
    window: UniformAttentionSpec,
    full: UniformAttentionSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    experiment: ExperimentReport,
    last_three_global_anchor_checkpoint_path: Path,
    last_two_global_anchor_checkpoint_path: Path,
    window_anchor_checkpoint_path: Path,
    *,
    eval_iters: int,
    seed: int,
) -> Stage33PreciseValidationReport:
    """Evaluate all four natural checkpoints on exactly shared panels."""

    _, expected_bands = validate_attention_protocol(
        config,
        last_three_global,
        last_two_global,
        window,
        full,
        n_head,
        n_layer,
        last_three_global_anchor_checkpoint_path=last_three_global_anchor_checkpoint_path,
        last_two_global_anchor_checkpoint_path=last_two_global_anchor_checkpoint_path,
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
    references = _reference_hashes(
        experiment.full,
        experiment.window,
        experiment.last_two_global,
    )
    generated = {
        last_three_global.checkpoint_path: experiment.last_three_global_checkpoint_sha256,
        last_three_global_anchor_checkpoint_path: (
            experiment.last_three_global_anchor_checkpoint_sha256
        ),
    }
    _verify_reference_hashes(references, phase="before held-out validation")
    for path, expected_hash in generated.items():
        if checkpoint_sha256(path) != expected_hash:
            raise RuntimeError(f"Stage 33 artifact changed before validation: {path}")

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
    middle_reference = experiment.last_two_global
    for reference_path, reference_kind in (
        (last_two_global.checkpoint_path, "best"),
        (last_two_global_anchor_checkpoint_path, "anchor"),
    ):
        _STAGE_32.validate_stage_32_checkpoint(
            reference_path,
            data,
            config,
            last_two_global,
            middle_reference.last_global_reference.spec,
            window,
            full,
            n_head,
            n_layer,
            schedule,
            experiment.bands,
            middle_reference.initialization,
            experiment.full,
            window_reference,
            middle_reference.last_global_reference,
            training_batch_seed=experiment.training_batch_seed,
            training_rng_seed=experiment.training_rng_seed,
            checkpoint_kind=reference_kind,
        )
    last_anchor_checkpoint = validate_stage_33_checkpoint(
        last_three_global_anchor_checkpoint_path,
        data,
        config,
        last_three_global,
        last_two_global,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        experiment.bands,
        experiment.initialization,
        experiment.full,
        window_reference,
        middle_reference,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        checkpoint_kind="anchor",
    )
    last_best_checkpoint = validate_stage_33_checkpoint(
        last_three_global.checkpoint_path,
        data,
        config,
        last_three_global,
        last_two_global,
        window,
        full,
        n_head,
        n_layer,
        schedule,
        experiment.bands,
        experiment.initialization,
        experiment.full,
        window_reference,
        middle_reference,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        checkpoint_kind="best",
    )

    full_result = _evaluate_uniform_checkpoint(
        name="Full-trained/Full-mask",
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
    middle_anchor_result = _evaluate_hybrid_checkpoint(
        name="Last2Full-trained/Last2Full-mask (matched-anchor)",
        path=last_two_global_anchor_checkpoint_path,
        checkpoint=middle_reference.anchor_checkpoint,
        checkpoint_hash=middle_reference.anchor_checkpoint_sha256,
        fixed_panel_loss=middle_reference.anchor_validation_losses.overall,
        data=data,
        config=config,
        spec=last_two_global,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )
    middle_best_result = _evaluate_hybrid_checkpoint(
        name="Last2Full-trained/Last2Full-mask (selected-best)",
        path=last_two_global.checkpoint_path,
        checkpoint=middle_reference.best_checkpoint,
        checkpoint_hash=middle_reference.best_checkpoint_sha256,
        fixed_panel_loss=middle_reference.best_validation_losses.overall,
        data=data,
        config=config,
        spec=last_two_global,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )
    last_anchor_losses = _deserialize_losses(
        last_anchor_checkpoint.get("anchor_validation_losses"),
        experiment.bands,
        name="Last3Full anchor_validation_losses",
    )
    last_anchor_result = _evaluate_hybrid_checkpoint(
        name="Last3Full-trained/Last3Full-mask (matched-anchor)",
        path=last_three_global_anchor_checkpoint_path,
        checkpoint=last_anchor_checkpoint,
        checkpoint_hash=experiment.last_three_global_anchor_checkpoint_sha256,
        fixed_panel_loss=last_anchor_losses.overall,
        data=data,
        config=config,
        spec=last_three_global,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )
    last_best_result = _evaluate_hybrid_checkpoint(
        name="Last3Full-trained/Last3Full-mask (selected-best)",
        path=last_three_global.checkpoint_path,
        checkpoint=last_best_checkpoint,
        checkpoint_hash=experiment.last_three_global_checkpoint_sha256,
        fixed_panel_loss=float(last_best_checkpoint["best_val_loss"]),
        data=data,
        config=config,
        spec=last_three_global,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=experiment.bands,
        panels=panels,
    )

    _verify_reference_hashes(references, phase="during held-out validation")
    for path, expected_hash in generated.items():
        if checkpoint_sha256(path) != expected_hash:
            raise RuntimeError(f"Stage 33 artifact changed during validation: {path}")
    return Stage33PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        bands=experiment.bands,
        anchor=_make_endpoint_report(
            "matched step",
            full_result,
            last_anchor_result,
            middle_anchor_result,
            window_anchor_result,
            experiment.bands,
            last_three_global.local_context,
        ),
        selected_best=_make_endpoint_report(
            "selected best",
            full_result,
            last_best_result,
            middle_best_result,
            window_best_result,
            experiment.bands,
            last_three_global.local_context,
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


def print_paired_validation(report: Stage33PreciseValidationReport) -> None:
    print(f"\nHeld-out four-branch paired validation of Stage {EXPERIMENT_STAGE}")
    print(
        f"  {report.eval_iters} shared validation panels, seed={report.seed}; "
        "natural trained masks only"
    )
    for endpoint in (report.anchor, report.selected_best):
        print(f"\n{endpoint.name} endpoint")
        placement_scope = "primary" if endpoint is report.anchor else "secondary"
        for result in (
            endpoint.full_natural,
            endpoint.last_three_global_natural,
            endpoint.last_two_global_natural,
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
        delta_rows = (
            (
                f"{placement_scope} natural Last3Full-minus-Last2Full",
                endpoint.last_three_global_minus_last_two_global,
            ),
            (
                "secondary natural Full-minus-Last3Full",
                endpoint.full_minus_last_three_global,
            ),
            (
                "secondary natural Last3Full-minus-W23",
                endpoint.last_three_global_minus_window,
            ),
            ("audit natural Full-minus-Last2Full", endpoint.full_minus_last_two_global),
            (
                "audit natural Last2Full-minus-W23",
                endpoint.last_two_global_minus_window,
            ),
            ("audit natural Full-minus-W23", endpoint.full_minus_window),
        )
        for label, delta in delta_rows:
            _print_delta(label, delta, report.bands)
        range_rows = (
            (
                "predeclared Last3Full-minus-Last2Full topology aggregate",
                endpoint.last_three_global_minus_last_two_global_range,
            ),
            (
                "secondary Full-minus-Last3Full topology aggregate",
                endpoint.full_minus_last_three_global_range,
            ),
            (
                "secondary Last3Full-minus-W23 topology aggregate",
                endpoint.last_three_global_minus_window_range,
            ),
            (
                "audit Full-minus-Last2Full topology aggregate",
                endpoint.full_minus_last_two_global_range,
            ),
            (
                "audit Last2Full-minus-W23 topology aggregate",
                endpoint.last_two_global_minus_window_range,
            ),
            (
                "audit Full-minus-W23 topology aggregate",
                endpoint.full_minus_window_range,
            ),
        )
        for label, delta in range_rows:
            _print_range_delta(label, delta)
        for label, value in (
            ("Last3Full", endpoint.last_three_global_recovery_fraction),
            ("Last2Full", endpoint.last_two_global_recovery_fraction),
        ):
            if value is None:
                print(f"  descriptive {label} recovery fraction: undefined")
            else:
                print(f"  descriptive {label} recovery fraction: {value:.1%}")
    print(
        "\n  Matched-step Last3Full-minus-Last2Full natural overall is primary. "
        "Negative values favor adding a third consecutive final Full layer."
    )
    print(
        "  Full-minus-Last3Full and Last3Full-minus-W23 test sufficiency and "
        "benefit; selected-best, reference deltas, bands, and ratios are secondary."
    )


print_precise_validation = print_paired_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 33: train W23 with three final Full layers and compare pinned "
            "Last2Full, W23, and Full references."
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
        "--last-three-global-checkpoint-path",
        type=Path,
        default=DEFAULT_LAST_THREE_GLOBAL_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--last-three-global-anchor-checkpoint-path",
        type=Path,
        default=DEFAULT_LAST_THREE_GLOBAL_ANCHOR_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--last-two-global-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_LAST_TWO_GLOBAL_REFERENCE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--last-two-global-anchor-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_LAST_TWO_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--last-global-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_LAST_GLOBAL_REFERENCE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--last-global-anchor-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_LAST_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--middle-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_MIDDLE_REFERENCE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--middle-anchor-reference-checkpoint-path",
        type=Path,
        default=DEFAULT_MIDDLE_ANCHOR_REFERENCE_CHECKPOINT_PATH,
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
            "canonical Stage 33 requires the predeclared newly held-out "
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
    last_three_global = AttentionSpec(
        name="Last3Full: W23 layers 1-3, Full layers 4-6",
        layer_contexts=DEFAULT_LAYER_CONTEXTS,
        checkpoint_path=args.last_three_global_checkpoint_path,
    )
    last_two_global = _STAGE_32.AttentionSpec(
        name="Last2Full: W23 layers 1-4, Full layers 5-6",
        layer_contexts=DEFAULT_LAST_TWO_GLOBAL_LAYER_CONTEXTS,
        checkpoint_path=args.last_two_global_reference_checkpoint_path,
    )
    last_global = _STAGE_31.AttentionSpec(
        name="LastFull: W23 layers 1-5, Full layer 6",
        layer_contexts=DEFAULT_LAST_GLOBAL_LAYER_CONTEXTS,
        checkpoint_path=args.last_global_reference_checkpoint_path,
    )
    middle_global = _STAGE_31.AttentionSpec(
        name="W23 with one Full layer at depth 3",
        layer_contexts=DEFAULT_MIDDLE_LAYER_CONTEXTS,
        checkpoint_path=args.middle_reference_checkpoint_path,
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
        last_three_global,
        last_two_global,
        window,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        last_three_global_anchor_checkpoint_path=(
            args.last_three_global_anchor_checkpoint_path
        ),
        last_two_global_anchor_checkpoint_path=(
            args.last_two_global_anchor_reference_checkpoint_path
        ),
        window_anchor_checkpoint_path=args.window_anchor_reference_checkpoint_path,
        last_global=last_global,
        last_global_anchor_checkpoint_path=(
            args.last_global_anchor_reference_checkpoint_path
        ),
        middle_global=middle_global,
        middle_global_anchor_checkpoint_path=(
            args.middle_anchor_reference_checkpoint_path
        ),
    )
    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)
    if not _is_canonical_protocol(
        data,
        config,
        last_three_global,
        last_two_global,
        window,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
    ):
        raise RuntimeError("the Stage 33 CLI is reserved for the canonical protocol")
    if device.type != "xpu":
        raise RuntimeError(
            "canonical Stage 33 must use the same XPU backend class as Stages "
            "26, 29, 30, 31, and 32; use the programmatic API for CPU plumbing "
            "tests"
        )
    accelerator_name = torch.xpu.get_device_name(0)
    if accelerator_name != DEFAULT_ACCELERATOR_NAME:
        raise RuntimeError(
            "canonical Stage 33 requires accelerator "
            f"{DEFAULT_ACCELERATOR_NAME!r}, got {accelerator_name!r}"
        )

    print(f"Stage {EXPERIMENT_STAGE}: {EXPERIMENT_TITLE}")
    print(f"Device: {device}")
    print(f"Accelerator: {accelerator_name}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"All: B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={DEFAULT_N_HEAD}, "
        f"D={config.n_embd // DEFAULT_N_HEAD}, FF={4 * config.n_embd}, "
        f"L={DEFAULT_N_LAYER}, dropout={DEFAULT_DROPOUT:g}"
    )
    print(f"Last3Full layer contexts: {DEFAULT_LAYER_CONTEXTS}")
    print(f"Last2Full layer contexts:  {DEFAULT_LAST_TWO_GLOBAL_LAYER_CONTEXTS}")
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
        last_three_global,
        last_two_global,
        window,
        full,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        device,
        args.sample_length,
        schedule,
        args.last_three_global_anchor_checkpoint_path,
        args.last_two_global_anchor_reference_checkpoint_path,
        args.window_anchor_reference_checkpoint_path,
        last_global=last_global,
        last_global_anchor_checkpoint_path=(
            args.last_global_anchor_reference_checkpoint_path
        ),
        middle_global=middle_global,
        middle_global_anchor_checkpoint_path=(
            args.middle_anchor_reference_checkpoint_path
        ),
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
        last_two_global_best_expected_sha256=(
            DEFAULT_LAST_TWO_GLOBAL_REFERENCE_CHECKPOINT_SHA256
        ),
        last_two_global_anchor_expected_sha256=(
            DEFAULT_LAST_TWO_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_SHA256
        ),
        last_two_global_best_expected_step=DEFAULT_LAST_TWO_GLOBAL_REFERENCE_STEP,
        last_two_global_anchor_expected_step=(
            DEFAULT_LAST_TWO_GLOBAL_ANCHOR_REFERENCE_STEP
        ),
        last_global_best_expected_sha256=(
            DEFAULT_LAST_GLOBAL_REFERENCE_CHECKPOINT_SHA256
        ),
        last_global_anchor_expected_sha256=(
            DEFAULT_LAST_GLOBAL_ANCHOR_REFERENCE_CHECKPOINT_SHA256
        ),
        last_global_best_expected_step=DEFAULT_LAST_GLOBAL_REFERENCE_STEP,
        last_global_anchor_expected_step=(DEFAULT_LAST_GLOBAL_ANCHOR_REFERENCE_STEP),
        middle_best_expected_sha256=DEFAULT_MIDDLE_REFERENCE_CHECKPOINT_SHA256,
        middle_anchor_expected_sha256=(
            DEFAULT_MIDDLE_ANCHOR_REFERENCE_CHECKPOINT_SHA256
        ),
        middle_best_expected_step=DEFAULT_MIDDLE_REFERENCE_STEP,
        middle_anchor_expected_step=DEFAULT_MIDDLE_ANCHOR_REFERENCE_STEP,
        benchmark=benchmark,
    )
    if args.precise_eval_iters > 0:
        precise = run_paired_validation(
            data,
            config,
            last_three_global,
            last_two_global,
            window,
            full,
            DEFAULT_N_HEAD,
            DEFAULT_N_LAYER,
            device,
            schedule,
            experiment,
            args.last_three_global_anchor_checkpoint_path,
            args.last_two_global_anchor_reference_checkpoint_path,
            args.window_anchor_reference_checkpoint_path,
            eval_iters=args.precise_eval_iters,
            seed=args.precise_eval_seed,
        )
        print_paired_validation(precise)
    _verify_reference_hashes(
        _reference_hashes(
            experiment.full,
            experiment.window,
            experiment.last_two_global,
        ),
        phase="during the complete Stage 33 CLI",
    )


if __name__ == "__main__":
    main()
