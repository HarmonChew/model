from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from training import BenchmarkConfig


# Stage 35 combines two previously audited pieces of experiment machinery:
# Stage 26's paired long-segment context comparison and Stage 32's LLLLGG
# layerwise attention topology.  Both are loaded privately so neither
# historical stage is modified when this module is imported.
_STAGE_26_PATH = Path(__file__).with_name("26_context_length.py")
_STAGE_26_MODULE_NAME = "stage_26_context_length_for_stage_35"
_STAGE_26_SPEC = importlib.util.spec_from_file_location(
    _STAGE_26_MODULE_NAME,
    _STAGE_26_PATH,
)
assert _STAGE_26_SPEC is not None and _STAGE_26_SPEC.loader is not None
_STAGE_26 = importlib.util.module_from_spec(_STAGE_26_SPEC)
sys.modules[_STAGE_26_MODULE_NAME] = _STAGE_26
_STAGE_26_SPEC.loader.exec_module(_STAGE_26)

_STAGE_32_PATH = Path(__file__).with_name("32_iterative_global_refinement.py")
_STAGE_32_MODULE_NAME = "stage_32_iterative_global_refinement_for_stage_35"
_STAGE_32_SPEC = importlib.util.spec_from_file_location(
    _STAGE_32_MODULE_NAME,
    _STAGE_32_PATH,
)
assert _STAGE_32_SPEC is not None and _STAGE_32_SPEC.loader is not None
_STAGE_32 = importlib.util.module_from_spec(_STAGE_32_SPEC)
sys.modules[_STAGE_32_MODULE_NAME] = _STAGE_32
_STAGE_32_SPEC.loader.exec_module(_STAGE_32)

_CORE = _STAGE_26._STAGE_17


EXPERIMENT_STAGE = 35
EXPERIMENT_TITLE = "context scaling 128 to 256"
COMPARISON_VARIABLE = "context_length_with_fixed_last_two_global_topology"
PROTOCOL_NAME = "paired_llllgg_context_scaling"

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_35_last_two_global_context_128_best_checkpoint.pt"
)
DEFAULT_TREATMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_35_last_two_global_context_256_best_checkpoint.pt"
)

DEFAULT_CONTROL_BATCH_SIZE = 16
DEFAULT_CONTROL_BLOCK_SIZE = 128
DEFAULT_TREATMENT_BATCH_SIZE = 8
DEFAULT_TREATMENT_BLOCK_SIZE = 256
DEFAULT_LOCAL_CONTEXT = 23
DEFAULT_LAYER_CONTEXTS = (23, 23, 23, 23, None, None)
DEFAULT_GLOBAL_LAYER_INDICES = (4, 5)
DEFAULT_GLOBAL_LAYER_NUMBERS = (5, 6)
DEFAULT_N_EMBD = 128
DEFAULT_N_HEAD = 4
DEFAULT_HEAD_SIZE = 32
DEFAULT_FEED_FORWARD_WIDTH = 512
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
# Stage 35 was chosen after inspecting Stage 34's seed-1,351 panel.
DEFAULT_PRECISE_EVAL_SEED = 1_352

DEFAULT_TARGETS_PER_UPDATE = 2_048
DEFAULT_PRE_GLOBAL_EFFECTIVE_RECEPTIVE_FIELD = 89
DEFAULT_POSITION_BANDS = (
    (0, 128),
    (128, 256),
    (128, 160),
    (160, 192),
    (192, 224),
    (224, 256),
)

DROPOUT_PLACEMENT = _STAGE_26.DROPOUT_PLACEMENT
DEFAULT_DATA_FINGERPRINT = _STAGE_26.DEFAULT_DATA_FINGERPRINT

# Filled with the canonical fingerprints below.  Keeping every initialization
# boundary explicit makes accidental RNG or topology drift fail before a long
# canonical training run begins.
DEFAULT_CONTROL_INITIAL_STATE_SHA256 = (
    "e6e8d393ca8b854f6af04222d2e99c244934f8c8b3c61fab5a0ea4592ae18034"
)
DEFAULT_TREATMENT_INITIAL_STATE_SHA256 = (
    "a05fad1a3ce76c66b7aebf5233aa4276ac91dcd3f6126e69399ef1358cd3f1bb"
)
DEFAULT_SHARED_PARAMETER_INITIALIZATION_SHA256 = (
    "41e65d2cf9951f493c4b4a120dd06bd7c76cde17864d0c77c6d6a83f6261673b"
)
DEFAULT_EXTRA_POSITION_ROWS_SHA256 = (
    "48330f2d163c177a885f2c78599092ecac8e6fa2c9a29ff33c99b3cb584a4a6b"
)
DEFAULT_CONTROL_PARAMETER_COUNT = 1_220_673
DEFAULT_TREATMENT_PARAMETER_COUNT = 1_237_057


GPTLanguageModel = _CORE.GPTLanguageModel
LearningRateSchedule = _CORE.LearningRateSchedule
ContextSpec = _CORE.ContextSpec
PairedContextBatch = _CORE.PairedContextBatch
SplitLosses = _CORE.SplitLosses
ContextLossEstimate = _CORE.ContextLossEstimate
ContextEvaluationRecord = _CORE.ContextEvaluationRecord
ContextTrainingResult = _CORE.ContextTrainingResult
PairedInitialStates = _CORE.PairedInitialStates
BranchReport = _CORE.BranchReport
ExperimentReport = _CORE.ExperimentReport
SplitStatistics = _CORE.SplitStatistics
PreciseBranchResult = _CORE.PreciseBranchResult
PreciseDelta = _CORE.PreciseDelta
PreciseValidationReport = _CORE.PreciseValidationReport
ContextValidationCheckpoint = _CORE.ContextValidationCheckpoint
AttentionSpec = _STAGE_32.AttentionSpec
LayerwiseReachabilityAudit = _STAGE_32.LayerwiseReachabilityAudit

checkpoint_sha256 = _CORE.checkpoint_sha256
clear_accelerator_cache = _CORE.clear_accelerator_cache
fingerprint_data = _CORE.fingerprint_data
seed_everything = _CORE.seed_everything
tensor_mapping_sha256 = _CORE.tensor_mapping_sha256
tensor_sha256 = _CORE.tensor_sha256
build_paired_context_batch = _CORE.build_paired_context_batch
get_paired_context_batch = _CORE.get_paired_context_batch
split_cross_entropy = _CORE.split_cross_entropy
estimate_split_loss = _CORE.estimate_split_loss
evaluate_on_fixed_paired_batches = _CORE.evaluate_on_fixed_paired_batches
train_with_paired_schedule = _CORE.train_with_paired_schedule
benchmark_context_branch = _CORE.benchmark_context_branch
aggregate_benchmark_stats = _CORE.aggregate_benchmark_stats
run_counterbalanced_benchmarks = _CORE.run_counterbalanced_benchmarks
run_branch = _CORE.run_branch
expected_training_generator_state = _CORE.expected_training_generator_state
build_attention_mask = _STAGE_32._STAGE_30.build_attention_mask
audit_layerwise_reachability = _STAGE_32.audit_layerwise_reachability
verify_global_reachability = _STAGE_32.verify_global_reachability


def make_attention_spec(block_size: int) -> AttentionSpec:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError(f"block_size must be an integer, got {block_size!r}")
    if block_size < DEFAULT_LOCAL_CONTEXT:
        raise ValueError(
            f"block_size must be at least W{DEFAULT_LOCAL_CONTEXT}, got {block_size}"
        )
    return AttentionSpec(
        name=f"Last2Full LLLLGG at T={block_size}",
        layer_contexts=DEFAULT_LAYER_CONTEXTS,
        checkpoint_path=(
            CHECKPOINT_DIRECTORY / f".stage_35_architecture_t{block_size}.pt"
        ),
    )


def build_layer_attention_masks(block_size: int) -> tuple[torch.Tensor, ...]:
    return _STAGE_32.build_layer_attention_masks(
        block_size,
        make_attention_spec(block_size),
    )


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    if n_layer != DEFAULT_N_LAYER:
        raise ValueError(
            f"Stage 35 requires L={DEFAULT_N_LAYER}, got {n_layer}"
        )
    return _STAGE_32.build_model(
        data,
        config,
        make_attention_spec(config.block_size),
        n_head,
        n_layer,
        device,
    )


def verify_model_masks(model: GPTLanguageModel) -> None:
    _STAGE_32.verify_model_masks(model, make_attention_spec(model.block_size))


def inspect_attention_shape(
    model: GPTLanguageModel,
    spec: ContextSpec,
    n_head: int,
    device: torch.device,
) -> tuple[int, ...]:
    """Audit every layer's mask and dense attention tensor shape."""

    verify_model_masks(model)
    expected = (
        spec.batch_size,
        n_head,
        spec.block_size,
        spec.block_size,
    )
    was_training = model.training
    model.eval()
    try:
        inputs = torch.zeros(
            (spec.batch_size, spec.block_size),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            for layer in range(DEFAULT_N_LAYER):
                observed = tuple(
                    model.get_attention_weights(inputs, block_index=layer).shape
                )
                if observed != expected:
                    raise RuntimeError(
                        f"{spec.name} layer {layer + 1} attention shape must be "
                        f"{expected}, got {observed}"
                    )
        return expected
    finally:
        model.train(was_training)


def make_position_bands(
    control: ContextSpec,
    treatment: ContextSpec,
) -> tuple[tuple[int, int], ...]:
    _CORE._validate_context_geometry(control, treatment)
    if control.block_size % 4:
        raise ValueError(
            "the control block size must be divisible by four for second-half "
            "32-position-style diagnostics"
        )
    start = control.block_size
    width = control.block_size // 4
    return (
        (0, control.block_size),
        (control.block_size, treatment.block_size),
        *tuple((start + i * width, start + (i + 1) * width) for i in range(4)),
    )


_CORE.EXPERIMENT_STAGE = EXPERIMENT_STAGE
_CORE.RECORD_SHARED_PARAMETER_INITIALIZATION_HASH = True
_CORE.RECORD_BEST_TRAINING_LOSSES = True
_CORE.GENERATE_FROM_SELECTED_CHECKPOINT = True
_CORE.DEFAULT_CONTROL_CHECKPOINT_PATH = DEFAULT_CONTROL_CHECKPOINT_PATH
_CORE.DEFAULT_TREATMENT_CHECKPOINT_PATH = DEFAULT_TREATMENT_CHECKPOINT_PATH
_CORE.DEFAULT_CONTROL_BATCH_SIZE = DEFAULT_CONTROL_BATCH_SIZE
_CORE.DEFAULT_CONTROL_BLOCK_SIZE = DEFAULT_CONTROL_BLOCK_SIZE
_CORE.DEFAULT_TREATMENT_BATCH_SIZE = DEFAULT_TREATMENT_BATCH_SIZE
_CORE.DEFAULT_TREATMENT_BLOCK_SIZE = DEFAULT_TREATMENT_BLOCK_SIZE
_CORE.DEFAULT_N_EMBD = DEFAULT_N_EMBD
_CORE.DEFAULT_N_HEAD = DEFAULT_N_HEAD
_CORE.DEFAULT_N_LAYER = DEFAULT_N_LAYER
_CORE.DEFAULT_DROPOUT = DEFAULT_DROPOUT
_CORE.DEFAULT_MAX_ITERS = DEFAULT_MAX_ITERS
_CORE.DEFAULT_INITIAL_LEARNING_RATE = DEFAULT_INITIAL_LEARNING_RATE
_CORE.DEFAULT_FIRST_DECAY_STEP = DEFAULT_FIRST_DECAY_STEP
_CORE.DEFAULT_MIDDLE_LEARNING_RATE = DEFAULT_MIDDLE_LEARNING_RATE
_CORE.DEFAULT_SECOND_DECAY_STEP = DEFAULT_SECOND_DECAY_STEP
_CORE.DEFAULT_FINAL_LEARNING_RATE = DEFAULT_FINAL_LEARNING_RATE
_CORE.DEFAULT_PRECISE_EVAL_ITERS = DEFAULT_PRECISE_EVAL_ITERS
_CORE._build_model = build_model
_CORE.inspect_attention_shape = inspect_attention_shape


_CREATE_PAIRED_INITIAL_STATES_BASE = _STAGE_26._create_paired_initial_states_base


def _is_canonical_initialization(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    n_head: int,
    n_layer: int,
) -> bool:
    return (
        data.vocabulary.size == 65
        and fingerprint_data(data) == DEFAULT_DATA_FINGERPRINT
        and control_config.seed == DEFAULT_SEED
        and treatment_config.seed == DEFAULT_SEED
        and control_config.batch_size == DEFAULT_CONTROL_BATCH_SIZE
        and control_config.block_size == DEFAULT_CONTROL_BLOCK_SIZE
        and treatment_config.batch_size == DEFAULT_TREATMENT_BATCH_SIZE
        and treatment_config.block_size == DEFAULT_TREATMENT_BLOCK_SIZE
        and control_config.n_embd == DEFAULT_N_EMBD
        and treatment_config.n_embd == DEFAULT_N_EMBD
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
    )


def create_paired_initial_states(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    n_head: int,
    n_layer: int,
) -> PairedInitialStates:
    initialization = _CREATE_PAIRED_INITIAL_STATES_BASE(
        data,
        control_config,
        treatment_config,
        n_head,
        n_layer,
    )
    if _is_canonical_initialization(
        data,
        control_config,
        treatment_config,
        n_head,
        n_layer,
    ) and DEFAULT_CONTROL_INITIAL_STATE_SHA256:
        observed = {
            "control state": initialization.control_state_sha256,
            "treatment state": initialization.treatment_state_sha256,
            "shared parameters": (
                initialization.shared_parameter_initialization_sha256
            ),
            "new position rows": initialization.extra_position_rows_sha256,
            "control parameter count": initialization.control_parameter_count,
            "treatment parameter count": initialization.treatment_parameter_count,
        }
        expected = {
            "control state": DEFAULT_CONTROL_INITIAL_STATE_SHA256,
            "treatment state": DEFAULT_TREATMENT_INITIAL_STATE_SHA256,
            "shared parameters": DEFAULT_SHARED_PARAMETER_INITIALIZATION_SHA256,
            "new position rows": DEFAULT_EXTRA_POSITION_ROWS_SHA256,
            "control parameter count": DEFAULT_CONTROL_PARAMETER_COUNT,
            "treatment parameter count": DEFAULT_TREATMENT_PARAMETER_COUNT,
        }
        if observed != expected:
            raise RuntimeError(
                "canonical Stage 35 initialization changed: "
                f"expected {expected}, got {observed}"
            )
    return initialization


_CORE.create_paired_initial_states = create_paired_initial_states


def verify_initial_first_128_equivalence(
    control_model: GPTLanguageModel,
    treatment_model: GPTLanguageModel,
    data: CharacterData,
    control: ContextSpec,
    treatment: ContextSpec,
    device: torch.device,
    *,
    seed: int,
) -> float:
    return _CORE.verify_initial_first_half_equivalence(
        control_model,
        treatment_model,
        data,
        control,
        treatment,
        device,
        seed=seed,
    )


# Compatibility with the generic paired-context harness terminology.
verify_initial_first_half_equivalence = verify_initial_first_128_equivalence


def _logical_edges_by_layer(block_size: int) -> tuple[int, ...]:
    return tuple(int(mask.sum().item()) for mask in build_layer_attention_masks(block_size))


def _extend_stage_35_checkpoint_payload(
    checkpoint: object,
    payload: dict[str, object],
) -> None:
    config = getattr(checkpoint, "config")
    model = getattr(checkpoint, "model")
    optimizer = getattr(checkpoint, "optimizer")
    control = getattr(checkpoint, "control_spec")
    treatment = getattr(checkpoint, "treatment_spec")
    architecture = payload["architecture"]
    training = payload["training_config"]
    initialization = payload["initialization"]
    experiment = payload["experiment"]
    assert isinstance(architecture, dict)
    assert isinstance(training, dict)
    assert isinstance(initialization, dict)
    assert isinstance(experiment, dict)

    masks = build_layer_attention_masks(config.block_size)
    reachability = audit_layerwise_reachability(masks)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture.update(
        {
            "head_size": config.n_embd // getattr(checkpoint, "n_head"),
            "feed_forward_width": 4 * config.n_embd,
            "attention_mask": "W23,W23,W23,W23,Full,Full",
            "attention_mask_scope": "per_layer",
            "topology": "LLLLGG",
            "layer_contexts": DEFAULT_LAYER_CONTEXTS,
            "local_context": DEFAULT_LOCAL_CONTEXT,
            "global_layer_indices": DEFAULT_GLOBAL_LAYER_INDICES,
            "global_layer_numbers": DEFAULT_GLOBAL_LAYER_NUMBERS,
            "pre_global_effective_receptive_field": (
                reachability.widths_by_layer[3]
            ),
            "first_global_reachability_layer": (
                reachability.first_global_reachability_layer
            ),
            "logical_edges_per_head_by_layer": tuple(
                int(mask.sum().item()) for mask in masks
            ),
            "logical_edges_per_head_across_depth": sum(
                int(mask.sum().item()) for mask in masks
            ),
            "parameter_count": parameter_count,
        }
    )
    group = optimizer.param_groups[0]
    training.update(
        {
            "optimizer": "AdamW",
            "optimizer_betas": tuple(group["betas"]),
            "optimizer_eps": float(group["eps"]),
            "optimizer_weight_decay": float(group["weight_decay"]),
        }
    )
    initialization.update(
        {
            "shared_parameter_hash_scope": "all_named_parameters_in_control_model",
            "shared_parameters_byte_identical": True,
            "shared_position_row_range": (0, control.block_size - 1),
            "shared_position_rows_byte_identical": True,
            "extra_position_row_range": (
                control.block_size,
                treatment.block_size - 1,
            ),
            "extra_position_initialization_architecture_native": True,
            "extra_position_initialization_primitive": (
                "torch.nn.Embedding.reset_parameters.normal_"
            ),
            "new_position_rows_isolated_from_shared_parameter_rng": True,
            "initial_first_128_logits_verified": True,
            "common_control_projection_sha256": getattr(
                checkpoint,
                "common_state_sha256",
            ),
            "common_control_projection_block_size": control.block_size,
            # ContextValidationCheckpoint predates Stage 35 and emits the
            # hard-coded key common_t64_projection_sha256. Retain it for its
            # base validator, but identify it unambiguously as a legacy alias.
            "common_t64_projection_sha256_is_legacy_alias": True,
        }
    )
    experiment.update(
        {
            "protocol_name": PROTOCOL_NAME,
            "fixed_topology": "LLLLGG",
            "fixed_layer_contexts": DEFAULT_LAYER_CONTEXTS,
            "paired_long_segments_per_update": treatment.batch_size,
            "control_windows_per_long_segment": 2,
            "same_underlying_target_characters": True,
            "identical_flattened_target_order": True,
            "fresh_control_training_run": True,
            "fresh_treatment_training_run": True,
            "reused_stage_32_trained_checkpoint": False,
            "same_optimizer_configuration": True,
            "same_update_count": True,
            "same_learning_rate_schedule": True,
            "same_training_batch_seed": True,
            "same_training_rng_seed": True,
            "identical_dropout_probability": True,
            "identical_dropout_masks_guaranteed": False,
            "common_control_projection_sha256": getattr(
                checkpoint,
                "common_state_sha256",
            ),
            "common_control_projection_block_size": control.block_size,
            "common_t64_projection_sha256_is_legacy_alias": True,
        }
    )
    payload["evaluation_protocol"] = {
        "paired_long_context_panels": True,
        "panel_batch_size": treatment.batch_size,
        "panel_block_size": treatment.block_size,
        "position_bands": make_position_bands(control, treatment),
        "primary_halves": (
            (0, control.block_size),
            (control.block_size, treatment.block_size),
        ),
        "second_half_quarter_count": 4,
        "control_model_positions_for_each_half": (0, control.block_size - 1),
        "checkpoint_selection_objective": "overall_validation_loss",
        "position_band_losses_are_diagnostics": True,
        "fixed_panel_seed": getattr(checkpoint, "config").seed + 2,
        "model_mode": "eval",
        "dropout_enabled": False,
        "gradients_enabled": False,
    }


_CORE._extend_context_checkpoint_payload = _extend_stage_35_checkpoint_payload
_VALIDATE_BASE_CHECKPOINT = _STAGE_26._validate_stage_17_checkpoint_base


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    return _CORE._require_mapping(value, name=name)


def validate_stage_35_checkpoint(
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
    shared_parameter_initialization_sha256: str = "",
) -> Mapping[str, object]:
    checkpoint = _VALIDATE_BASE_CHECKPOINT(
        path,
        data,
        config,
        spec,
        control,
        treatment,
        n_head,
        n_layer,
        schedule,
        initialization_seed=initialization_seed,
        branch_initial_state_sha256=branch_initial_state_sha256,
        common_state_sha256=common_state_sha256,
        extra_position_rows_sha256=extra_position_rows_sha256,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        shared_parameter_initialization_sha256=(
            shared_parameter_initialization_sha256
        ),
    )

    architecture = _require_mapping(
        checkpoint.get("architecture"), name="architecture"
    )
    expected_reachability = audit_layerwise_reachability(
        build_layer_attention_masks(config.block_size)
    )
    expected_architecture: dict[str, object] = {
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "attention_mask": "W23,W23,W23,W23,Full,Full",
        "attention_mask_scope": "per_layer",
        "topology": "LLLLGG",
        "layer_contexts": DEFAULT_LAYER_CONTEXTS,
        "local_context": DEFAULT_LOCAL_CONTEXT,
        "global_layer_indices": DEFAULT_GLOBAL_LAYER_INDICES,
        "global_layer_numbers": DEFAULT_GLOBAL_LAYER_NUMBERS,
        "pre_global_effective_receptive_field": min(
            DEFAULT_PRE_GLOBAL_EFFECTIVE_RECEPTIVE_FIELD,
            config.block_size,
        ),
        "first_global_reachability_layer": (
            expected_reachability.first_global_reachability_layer
        ),
        "logical_edges_per_head_by_layer": _logical_edges_by_layer(
            config.block_size
        ),
        "logical_edges_per_head_across_depth": sum(
            _logical_edges_by_layer(config.block_size)
        ),
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"checkpoint architecture.{key} must be {expected!r}, "
                f"got {architecture.get(key)!r}"
            )

    training = _require_mapping(
        checkpoint.get("training_config"), name="training_config"
    )
    expected_training = {
        "optimizer": "AdamW",
        "optimizer_betas": (0.9, 0.999),
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0.01,
        "targets_per_update": control.tokens_per_update,
        "paired_segment_batch_size": treatment.batch_size,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"checkpoint training_config.{key} must be {expected!r}, "
                f"got {training.get(key)!r}"
            )

    initialization = _require_mapping(
        checkpoint.get("initialization"), name="initialization"
    )
    expected_initialization = {
        "shared_parameters_byte_identical": True,
        "shared_position_row_range": (0, control.block_size - 1),
        "shared_position_rows_byte_identical": True,
        "extra_position_row_range": (
            control.block_size,
            treatment.block_size - 1,
        ),
        "extra_position_initialization_architecture_native": True,
        "new_position_rows_isolated_from_shared_parameter_rng": True,
        "initial_first_128_logits_verified": True,
        "common_control_projection_sha256": common_state_sha256,
        "common_control_projection_block_size": control.block_size,
        "common_t64_projection_sha256_is_legacy_alias": True,
    }
    for key, expected in expected_initialization.items():
        if initialization.get(key) != expected:
            raise ValueError(
                f"checkpoint initialization.{key} must be {expected!r}, "
                f"got {initialization.get(key)!r}"
            )

    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    expected_experiment = {
        "stage": EXPERIMENT_STAGE,
        "protocol_name": PROTOCOL_NAME,
        "fixed_topology": "LLLLGG",
        "fixed_layer_contexts": DEFAULT_LAYER_CONTEXTS,
        "fresh_control_training_run": True,
        "fresh_treatment_training_run": True,
        "reused_stage_32_trained_checkpoint": False,
        "same_underlying_target_characters": True,
        "identical_flattened_target_order": True,
        "common_control_projection_sha256": common_state_sha256,
        "common_control_projection_block_size": control.block_size,
        "common_t64_projection_sha256_is_legacy_alias": True,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"checkpoint experiment.{key} must be {expected!r}, "
                f"got {experiment.get(key)!r}"
            )

    evaluation = _require_mapping(
        checkpoint.get("evaluation_protocol"), name="evaluation_protocol"
    )
    if evaluation.get("position_bands") != make_position_bands(control, treatment):
        raise ValueError("checkpoint evaluation_protocol.position_bands changed")

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    with torch.random.fork_rng(devices=[]):
        compatibility_model = build_model(data, config, n_head, n_layer)
        compatibility_model.load_state_dict(model_state, strict=True)
        verify_model_masks(compatibility_model)
        parameter_count = sum(
            parameter.numel()
            for parameter in compatibility_model.parameters()
            if parameter.requires_grad
        )
        del compatibility_model
    if architecture.get("parameter_count") != parameter_count:
        raise ValueError(
            "checkpoint architecture.parameter_count does not match the model"
        )
    return checkpoint


_CORE.validate_stage_17_checkpoint = validate_stage_35_checkpoint


@dataclass(frozen=True, slots=True)
class PositionBandLosses:
    bands: tuple[tuple[int, int], ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.bands) != len(self.values) or not self.bands:
            raise ValueError("position bands and values must be non-empty and aligned")

    def for_band(self, start: int, end: int) -> float:
        try:
            return self.values[self.bands.index((start, end))]
        except ValueError as error:
            raise KeyError((start, end)) from error


@dataclass(frozen=True, slots=True)
class PositionBandStatistics:
    mean: PositionBandLosses
    standard_error: PositionBandLosses


@dataclass(frozen=True, slots=True)
class PositionBandBranchResult:
    name: str
    statistics: PositionBandStatistics
    batch_losses: tuple[PositionBandLosses, ...]


@dataclass(frozen=True, slots=True)
class PositionBandDelta:
    candidate: str
    baseline: str
    mean_delta: PositionBandLosses
    standard_error: PositionBandLosses
    confidence_low: PositionBandLosses
    confidence_high: PositionBandLosses
    critical_value: float


@dataclass(frozen=True, slots=True)
class Stage35PreciseValidationReport:
    context: PreciseValidationReport
    position_bands: tuple[tuple[int, int], ...]
    band_results: tuple[PositionBandBranchResult, ...]
    band_delta: PositionBandDelta

    @property
    def eval_iters(self) -> int:
        return self.context.eval_iters

    @property
    def seed(self) -> int:
        return self.context.seed

    @property
    def results(self) -> tuple[PreciseBranchResult, ...]:
        return self.context.results

    @property
    def delta(self) -> PreciseDelta:
        return self.context.delta


def _align_to_long_panel(
    logits: torch.Tensor,
    targets: torch.Tensor,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(logits.shape[:2]) != (spec.batch_size, spec.block_size):
        raise ValueError("logits do not match the branch context spec")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match the first two logits dimensions")
    if (spec.batch_size, spec.block_size) == (
        treatment.batch_size,
        treatment.block_size,
    ):
        return logits, targets
    if (spec.batch_size, spec.block_size) != (
        control.batch_size,
        control.block_size,
    ):
        raise ValueError(f"unknown context spec {spec}")
    vocabulary_size = logits.shape[-1]
    return (
        logits.reshape(
            treatment.batch_size,
            2,
            control.block_size,
            vocabulary_size,
        ).reshape(treatment.batch_size, treatment.block_size, vocabulary_size),
        targets.reshape(
            treatment.batch_size,
            2,
            control.block_size,
        ).reshape(treatment.batch_size, treatment.block_size),
    )


def position_band_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    spec: ContextSpec,
    control: ContextSpec,
    treatment: ContextSpec,
    *,
    bands: Sequence[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, ...]:
    selected_bands = tuple(bands or make_position_bands(control, treatment))
    long_logits, long_targets = _align_to_long_panel(
        logits,
        targets,
        spec,
        control,
        treatment,
    )
    vocabulary_size = logits.shape[-1]
    losses: list[torch.Tensor] = []
    for start, end in selected_bands:
        if not 0 <= start < end <= treatment.block_size:
            raise ValueError(
                f"position band {(start, end)} is outside [0, "
                f"{treatment.block_size})"
            )
        losses.append(
            F.cross_entropy(
                long_logits[:, start:end].reshape(-1, vocabulary_size),
                long_targets[:, start:end].reshape(-1),
            )
        )
    return tuple(losses)


@torch.no_grad()
def estimate_validation_with_position_bands(
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
    bands: Sequence[tuple[int, int]] | None = None,
) -> tuple[tuple[SplitLosses, ...], tuple[PositionBandLosses, ...]]:
    del config  # Geometry is pinned by the branch spec.
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    selected_bands = tuple(bands or make_position_bands(control, treatment))
    generator = torch.Generator().manual_seed(seed)
    split_values: list[SplitLosses] = []
    band_values: list[PositionBandLosses] = []
    was_training = model.training
    model.eval()
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
            split_values.append(
                SplitLosses(
                    *(
                        value.detach().item()
                        for value in split_cross_entropy(
                            logits,
                            targets,
                            spec,
                            control=control,
                            treatment=treatment,
                        )
                    )
                )
            )
            band_values.append(
                PositionBandLosses(
                    selected_bands,
                    tuple(
                        value.detach().item()
                        for value in position_band_cross_entropy(
                            logits,
                            targets,
                            spec,
                            control,
                            treatment,
                            bands=selected_bands,
                        )
                    ),
                )
            )
    finally:
        model.train(was_training)
    return tuple(split_values), tuple(band_values)


def _mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    mean = tensor.mean().item()
    if len(values) == 1:
        return mean, 0.0
    return mean, (tensor.std(unbiased=True) / math.sqrt(len(values))).item()


def _band_statistics(
    values: Sequence[PositionBandLosses],
) -> PositionBandStatistics:
    if not values:
        raise ValueError("at least one position-band result is required")
    bands = values[0].bands
    if any(value.bands != bands for value in values):
        raise ValueError("position-band layouts must match")
    statistics = tuple(
        _mean_and_standard_error([value.values[index] for value in values])
        for index in range(len(bands))
    )
    return PositionBandStatistics(
        mean=PositionBandLosses(bands, tuple(value[0] for value in statistics)),
        standard_error=PositionBandLosses(
            bands,
            tuple(value[1] for value in statistics),
        ),
    )


def _position_band_delta(
    candidate: PositionBandBranchResult,
    baseline: PositionBandBranchResult,
) -> PositionBandDelta:
    if len(candidate.batch_losses) != len(baseline.batch_losses):
        raise ValueError("paired position-band results must have equal lengths")
    if len(candidate.batch_losses) < 2:
        raise ValueError("at least two paired validation batches are required")
    bands = candidate.batch_losses[0].bands
    differences = tuple(
        PositionBandLosses(
            bands,
            tuple(
                candidate_value - baseline_value
                for candidate_value, baseline_value in zip(
                    candidate_batch.values,
                    baseline_batch.values,
                    strict=True,
                )
            ),
        )
        for candidate_batch, baseline_batch in zip(
            candidate.batch_losses,
            baseline.batch_losses,
            strict=True,
        )
    )
    stats = _band_statistics(differences)
    critical = _STAGE_26._paired_t_95_critical_value(len(differences))
    low = tuple(
        mean - critical * se
        for mean, se in zip(
            stats.mean.values,
            stats.standard_error.values,
            strict=True,
        )
    )
    high = tuple(
        mean + critical * se
        for mean, se in zip(
            stats.mean.values,
            stats.standard_error.values,
            strict=True,
        )
    )
    return PositionBandDelta(
        candidate=candidate.name,
        baseline=baseline.name,
        mean_delta=stats.mean,
        standard_error=stats.standard_error,
        confidence_low=PositionBandLosses(bands, low),
        confidence_high=PositionBandLosses(bands, high),
        critical_value=critical,
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
    expected_control_checkpoint_sha256: str | None = None,
    expected_treatment_checkpoint_sha256: str | None = None,
) -> Stage35PreciseValidationReport:
    if isinstance(eval_iters, bool) or not isinstance(eval_iters, int):
        raise ValueError(f"eval_iters must be an integer, got {eval_iters!r}")
    if eval_iters < 2:
        raise ValueError(
            f"eval_iters must be at least 2 for paired inference, got {eval_iters}"
        )
    if seed == control_config.seed + 2:
        raise ValueError("precise evaluation seed must differ from fixed-panel seed")
    bands = make_position_bands(control, treatment)
    context_results: list[PreciseBranchResult] = []
    band_results: list[PositionBandBranchResult] = []
    branches = (
        (
            control,
            control_config,
            initialization.control_state_sha256,
            expected_control_checkpoint_sha256,
        ),
        (
            treatment,
            treatment_config,
            initialization.treatment_state_sha256,
            expected_treatment_checkpoint_sha256,
        ),
    )
    for spec, config, initial_hash, expected_hash in branches:
        observed_hash = checkpoint_sha256(spec.checkpoint_path)
        if expected_hash is not None and observed_hash != expected_hash:
            raise ValueError(
                f"{spec.name} checkpoint SHA-256 must be {expected_hash}, "
                f"got {observed_hash}"
            )
        checkpoint = validate_stage_35_checkpoint(
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
            branch_initial_state_sha256=initial_hash,
            common_state_sha256=initialization.common_state_sha256,
            extra_position_rows_sha256=initialization.extra_position_rows_sha256,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            shared_parameter_initialization_sha256=(
                initialization.shared_parameter_initialization_sha256
            ),
        )
        with torch.random.fork_rng(devices=[]):
            model = build_model(data, config, n_head, n_layer, device)
            model.load_state_dict(
                _require_mapping(
                    checkpoint.get("model_state_dict"),
                    name="model_state_dict",
                ),
                strict=True,
            )
        verify_model_masks(model)
        split_batches, band_batches = estimate_validation_with_position_bands(
            model,
            data,
            config,
            spec,
            control,
            treatment,
            device,
            eval_iters=eval_iters,
            seed=seed,
            bands=bands,
        )
        if checkpoint_sha256(spec.checkpoint_path) != observed_hash:
            raise RuntimeError(
                f"checkpoint changed during evaluation: {spec.checkpoint_path}"
            )
        context_results.append(
            PreciseBranchResult(
                name=spec.name,
                checkpoint_path=spec.checkpoint_path,
                checkpoint_sha256=observed_hash,
                checkpoint_step=int(checkpoint["step"]),
                fixed_panel_loss=float(checkpoint["best_val_loss"]),
                statistics=_CORE._split_statistics(split_batches),
                batch_losses=split_batches,
            )
        )
        band_results.append(
            PositionBandBranchResult(
                name=spec.name,
                statistics=_band_statistics(band_batches),
                batch_losses=band_batches,
            )
        )
        del model
        clear_accelerator_cache(device)

    context_report = PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        results=tuple(context_results),
        delta=_STAGE_26._paired_delta(context_results[1], context_results[0]),
    )
    return Stage35PreciseValidationReport(
        context=context_report,
        position_bands=bands,
        band_results=tuple(band_results),
        band_delta=_position_band_delta(band_results[1], band_results[0]),
    )


_PRINT_CONTEXT_PRECISE = _STAGE_26.print_precise_validation


def _band_label(band: tuple[int, int]) -> str:
    start, end = band
    width = max(3, len(str(end - 1)))
    return f"{start:0{width}d}-{end - 1:0{width}d}"


def print_precise_validation(report: Stage35PreciseValidationReport) -> None:
    _PRINT_CONTEXT_PRECISE(report.context)
    print("\nStage 35 paired position-band diagnostics")
    for result in report.band_results:
        print(f"  {result.name}")
        for band, mean, se in zip(
            result.statistics.mean.bands,
            result.statistics.mean.values,
            result.statistics.standard_error.values,
            strict=True,
        ):
            print(f"    {_band_label(band)}: {mean:.6f} +/- {se:.6f} SE")
    delta = report.band_delta
    print(f"  delta {delta.candidate} - {delta.baseline}")
    for band, mean, se, low, high in zip(
        delta.mean_delta.bands,
        delta.mean_delta.values,
        delta.standard_error.values,
        delta.confidence_low.values,
        delta.confidence_high.values,
        strict=True,
    ):
        print(
            f"    {_band_label(band)}: {mean:+.6f} +/- {se:.6f} SE "
            f"(95% CI [{low:+.6f}, {high:+.6f}]); negative favors T=256"
        )


_CORE.run_precise_validation = run_precise_validation
_CORE.print_precise_validation = print_precise_validation


def _validate_stage_35_options(
    args: object,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    precise_eval_seed: int,
) -> None:
    del training_batch_seed, training_rng_seed
    precise_eval_iters = getattr(args, "precise_eval_iters")
    if precise_eval_iters == 1:
        raise ValueError("precise-eval-iters must be 0 or at least 2")
    if precise_eval_iters > 0 and precise_eval_seed == getattr(args, "seed") + 2:
        raise ValueError("precise evaluation seed must differ from fixed-panel seed")


_CORE._validate_context_stage_options = _validate_stage_35_options
_RUN_EXPERIMENT_BASE = _STAGE_26.run_experiment


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
    if n_layer != DEFAULT_N_LAYER:
        raise ValueError(f"Stage 35 requires L={DEFAULT_N_LAYER}")
    if n_head <= 0 or control_config.n_embd % n_head:
        raise ValueError(
            f"n_embd ({control_config.n_embd}) must be divisible by a "
            f"positive n_head ({n_head})"
        )
    control_masks = build_layer_attention_masks(control.block_size)
    treatment_masks = build_layer_attention_masks(treatment.block_size)
    control_audit = audit_layerwise_reachability(control_masks)
    treatment_audit = audit_layerwise_reachability(treatment_masks)
    verify_global_reachability(control_audit, control.block_size)
    verify_global_reachability(treatment_audit, treatment.block_size)
    print("\nFixed Stage 32 topology for both context branches")
    print(f"  per-layer attention: {DEFAULT_LAYER_CONTEXTS} (LLLLGG)")
    print(
        "  local composition RF after layer 4: "
        f"{control_audit.widths_by_layer[3]} / "
        f"{treatment_audit.widths_by_layer[3]} positions"
    )
    print("  layers 5 and 6 globally refine each branch's complete context")
    return _RUN_EXPERIMENT_BASE(
        data,
        control_config,
        treatment_config,
        control,
        treatment,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        benchmark=benchmark,
    )


_CORE.run_experiment = run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 35: compare LLLLGG T=128 with T=256 from matched fresh "
            "initialization while pairing every target character and holding "
            "B*T at 2,048."
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
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
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
        "--final-learning-rate", type=float, default=DEFAULT_FINAL_LEARNING_RATE
    )
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_EVAL_ITERS)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--training-batch-seed", type=int, default=DEFAULT_TRAINING_BATCH_SEED
    )
    parser.add_argument(
        "--training-rng-seed", type=int, default=DEFAULT_TRAINING_RNG_SEED
    )
    parser.add_argument(
        "--precise-eval-iters", type=int, default=DEFAULT_PRECISE_EVAL_ITERS
    )
    parser.add_argument(
        "--precise-eval-seed", type=int, default=DEFAULT_PRECISE_EVAL_SEED
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


_CORE.parse_args = parse_args


def main() -> None:
    _CORE.main()


if __name__ == "__main__":
    main()
