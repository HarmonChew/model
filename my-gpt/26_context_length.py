from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData


# Stage 26 deliberately reuses Stage 17's extensively tested paired-context
# machinery. Loading it under a private module name gives this stage an
# independent set of protocol globals and preserves Stage 17 exactly.
_STAGE_17_PATH = Path(__file__).with_name("17_context_length.py")
_STAGE_17_MODULE_NAME = "stage_17_context_length_for_stage_26"
_STAGE_17_SPEC = importlib.util.spec_from_file_location(
    _STAGE_17_MODULE_NAME,
    _STAGE_17_PATH,
)
assert _STAGE_17_SPEC is not None and _STAGE_17_SPEC.loader is not None
_STAGE_17 = importlib.util.module_from_spec(_STAGE_17_SPEC)
sys.modules[_STAGE_17_MODULE_NAME] = _STAGE_17
_STAGE_17_SPEC.loader.exec_module(_STAGE_17)


CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_26_context_64_best_checkpoint.pt"
)
DEFAULT_TREATMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_26_context_128_best_checkpoint.pt"
)

DEFAULT_CONTROL_BATCH_SIZE = 32
DEFAULT_CONTROL_BLOCK_SIZE = 64
DEFAULT_TREATMENT_BATCH_SIZE = 16
DEFAULT_TREATMENT_BLOCK_SIZE = 128
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
DEFAULT_PRECISE_EVAL_ITERS = 500
DROPOUT_PLACEMENT = _STAGE_17.DROPOUT_PLACEMENT

# These pins make the canonical initialization an executable provenance
# boundary. The T=64 state hash is the exact Stage 24 L=6 initialization hash.
DEFAULT_CONTROL_INITIAL_STATE_SHA256 = (
    "2a4639d1eff064c51d7cd103e11ddd32b782314e7d0f98b560b860224a937ab0"
)
DEFAULT_TREATMENT_INITIAL_STATE_SHA256 = (
    "c27f642dfb3ba186430d1f553b14a30fe3ce90c5062e9b2fe0ff448e88b6c8e8"
)
DEFAULT_SHARED_PARAMETER_INITIALIZATION_SHA256 = (
    "9f4a599cf43ab3b5e38dc697767afd6f4e8eb908639fec6c5fc139c97ba1d1ee"
)
DEFAULT_EXTRA_POSITION_ROWS_SHA256 = (
    "e2776f1eedcb60b39b36e2ce4bbf210f7fbbdc221178871b39645d2546158da5"
)
DEFAULT_CONTROL_PARAMETER_COUNT = 1_212_481
DEFAULT_TREATMENT_PARAMETER_COUNT = 1_220_673
DEFAULT_DATA_FINGERPRINT = (
    "fa03e609cfc0769eb1fa73fb4e81239fee0c6e9ab0608c549b0a742a25535493"
)


# Configure the private Stage 17 instance before exposing any of its entry
# points. Its functions resolve these values from their defining module at run
# time, so checkpoint creation and validation both carry Stage 26 provenance.
_STAGE_17.EXPERIMENT_STAGE = 26
_STAGE_17.RECORD_SHARED_PARAMETER_INITIALIZATION_HASH = True
_STAGE_17.RECORD_BEST_TRAINING_LOSSES = True
_STAGE_17.GENERATE_FROM_SELECTED_CHECKPOINT = True
_STAGE_17.DEFAULT_CONTROL_CHECKPOINT_PATH = DEFAULT_CONTROL_CHECKPOINT_PATH
_STAGE_17.DEFAULT_TREATMENT_CHECKPOINT_PATH = DEFAULT_TREATMENT_CHECKPOINT_PATH
_STAGE_17.DEFAULT_CONTROL_BATCH_SIZE = DEFAULT_CONTROL_BATCH_SIZE
_STAGE_17.DEFAULT_CONTROL_BLOCK_SIZE = DEFAULT_CONTROL_BLOCK_SIZE
_STAGE_17.DEFAULT_TREATMENT_BATCH_SIZE = DEFAULT_TREATMENT_BATCH_SIZE
_STAGE_17.DEFAULT_TREATMENT_BLOCK_SIZE = DEFAULT_TREATMENT_BLOCK_SIZE
_STAGE_17.DEFAULT_N_EMBD = DEFAULT_N_EMBD
_STAGE_17.DEFAULT_N_HEAD = DEFAULT_N_HEAD
_STAGE_17.DEFAULT_N_LAYER = DEFAULT_N_LAYER
_STAGE_17.DEFAULT_DROPOUT = DEFAULT_DROPOUT
_STAGE_17.DEFAULT_MAX_ITERS = DEFAULT_MAX_ITERS
_STAGE_17.DEFAULT_INITIAL_LEARNING_RATE = DEFAULT_INITIAL_LEARNING_RATE
_STAGE_17.DEFAULT_FIRST_DECAY_STEP = DEFAULT_FIRST_DECAY_STEP
_STAGE_17.DEFAULT_MIDDLE_LEARNING_RATE = DEFAULT_MIDDLE_LEARNING_RATE
_STAGE_17.DEFAULT_SECOND_DECAY_STEP = DEFAULT_SECOND_DECAY_STEP
_STAGE_17.DEFAULT_FINAL_LEARNING_RATE = DEFAULT_FINAL_LEARNING_RATE
_STAGE_17.DEFAULT_PRECISE_EVAL_ITERS = DEFAULT_PRECISE_EVAL_ITERS


GPTLanguageModel = _STAGE_17.GPTLanguageModel
LearningRateSchedule = _STAGE_17.LearningRateSchedule
BestValidationCheckpoint = _STAGE_17.BestValidationCheckpoint
ContextSpec = _STAGE_17.ContextSpec
PairedContextBatch = _STAGE_17.PairedContextBatch
SplitLosses = _STAGE_17.SplitLosses
ContextLossEstimate = _STAGE_17.ContextLossEstimate
ContextEvaluationRecord = _STAGE_17.ContextEvaluationRecord
ContextTrainingResult = _STAGE_17.ContextTrainingResult
PairedInitialStates = _STAGE_17.PairedInitialStates
BranchReport = _STAGE_17.BranchReport
ExperimentReport = _STAGE_17.ExperimentReport
SplitStatistics = _STAGE_17.SplitStatistics
PreciseBranchResult = _STAGE_17.PreciseBranchResult
PreciseDelta = _STAGE_17.PreciseDelta
PreciseValidationReport = _STAGE_17.PreciseValidationReport
ContextValidationCheckpoint = _STAGE_17.ContextValidationCheckpoint

checkpoint_sha256 = _STAGE_17.checkpoint_sha256
clear_accelerator_cache = _STAGE_17.clear_accelerator_cache
fingerprint_data = _STAGE_17.fingerprint_data
generate_from_final_model = _STAGE_17.generate_from_final_model
seed_everything = _STAGE_17.seed_everything
tensor_mapping_sha256 = _STAGE_17.tensor_mapping_sha256
tensor_sha256 = _STAGE_17.tensor_sha256

parse_args = _STAGE_17.parse_args
validate_context_protocol = _STAGE_17.validate_context_protocol
build_paired_context_batch = _STAGE_17.build_paired_context_batch
get_paired_context_batch = _STAGE_17.get_paired_context_batch
split_cross_entropy = _STAGE_17.split_cross_entropy
estimate_split_loss = _STAGE_17.estimate_split_loss
evaluate_on_fixed_paired_batches = _STAGE_17.evaluate_on_fixed_paired_batches
verify_initial_first_half_equivalence = _STAGE_17.verify_initial_first_half_equivalence
inspect_attention_shape = _STAGE_17.inspect_attention_shape
train_with_paired_schedule = _STAGE_17.train_with_paired_schedule
benchmark_context_branch = _STAGE_17.benchmark_context_branch
aggregate_benchmark_stats = _STAGE_17.aggregate_benchmark_stats
run_counterbalanced_benchmarks = _STAGE_17.run_counterbalanced_benchmarks
run_branch = _STAGE_17.run_branch
expected_training_generator_state = _STAGE_17.expected_training_generator_state
estimate_validation_precise = _STAGE_17.estimate_validation_precise
print_precise_validation = _STAGE_17.print_precise_validation

_build_model = _STAGE_17._build_model
_clone_state = _STAGE_17._clone_state
_project_treatment_state_to_control = _STAGE_17._project_treatment_state_to_control
_split_statistics = _STAGE_17._split_statistics
_require_mapping = _STAGE_17._require_mapping


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
    z = 1.959963984540054
    inverse_df = 1.0 / degrees_of_freedom
    return (
        z
        + (z**3 + z) * inverse_df / 4
        + (5 * z**5 + 16 * z**3 + 3 * z) * inverse_df**2 / 96
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) * inverse_df**3 / 384
    )


def _paired_delta(
    candidate: PreciseBranchResult,
    baseline: PreciseBranchResult,
) -> PreciseDelta:
    if len(candidate.batch_losses) != len(baseline.batch_losses):
        raise ValueError("paired validation results must have equal lengths")
    if len(candidate.batch_losses) < 2:
        raise ValueError("at least two paired validation batches are required")
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
    critical_value = _paired_t_95_critical_value(len(differences))
    low = SplitLosses(
        statistics.mean.overall - critical_value * statistics.standard_error.overall,
        statistics.mean.first_half
        - critical_value * statistics.standard_error.first_half,
        statistics.mean.second_half
        - critical_value * statistics.standard_error.second_half,
    )
    high = SplitLosses(
        statistics.mean.overall + critical_value * statistics.standard_error.overall,
        statistics.mean.first_half
        + critical_value * statistics.standard_error.first_half,
        statistics.mean.second_half
        + critical_value * statistics.standard_error.second_half,
    )
    return PreciseDelta(
        candidate=candidate.name,
        baseline=baseline.name,
        mean_delta=statistics.mean,
        standard_error=statistics.standard_error,
        confidence_low=low,
        confidence_high=high,
        critical_value=critical_value,
    )


_STAGE_17._paired_delta = _paired_delta


def _extend_stage_26_checkpoint_payload(
    checkpoint: object,
    payload: dict[str, object],
) -> None:
    control = getattr(checkpoint, "control_spec")
    treatment = getattr(checkpoint, "treatment_spec")
    common_state_sha256 = getattr(checkpoint, "common_state_sha256")
    shared_parameter_sha256 = getattr(
        checkpoint,
        "shared_parameter_initialization_sha256",
    )
    initialization = payload["initialization"]
    experiment = payload["experiment"]
    assert isinstance(initialization, dict)
    assert isinstance(experiment, dict)
    architecture = payload["architecture"]
    training = payload["training_config"]
    assert isinstance(architecture, dict)
    assert isinstance(training, dict)
    model = getattr(checkpoint, "model")
    optimizer = getattr(checkpoint, "optimizer")
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture.update(
        {
            "head_size": getattr(checkpoint, "config").n_embd
            // getattr(checkpoint, "n_head"),
            "feed_forward_width": 4 * getattr(checkpoint, "config").n_embd,
            "attention_mask": "full_causal",
            "attention_mask_scope": "per_layer",
            "max_context": None,
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
            "new_position_rows_isolated_from_shared_parameter_rng": True,
        }
    )
    if (
        common_state_sha256 == DEFAULT_CONTROL_INITIAL_STATE_SHA256
        and getattr(checkpoint, "data_fingerprint") == DEFAULT_DATA_FINGERPRINT
    ):
        initialization.update(
            {
                "t64_initialization_source_stage": 24,
                "source_initial_state_sha256": (DEFAULT_CONTROL_INITIAL_STATE_SHA256),
                "byte_identical_to_source_initialization": True,
            }
        )
    experiment.update(
        {
            "fixed_transformer_depth": getattr(checkpoint, "n_layer"),
            "fixed_residual_dropout": DEFAULT_DROPOUT,
            "paired_long_segments_per_update": treatment.batch_size,
            "control_windows_per_long_segment": (
                treatment.block_size // control.block_size
            ),
            "same_training_rng_seed": True,
            "shared_parameter_initialization_sha256": shared_parameter_sha256,
            "new_position_initialization_sha256": getattr(
                checkpoint,
                "extra_position_rows_sha256",
            ),
            "new_control_training_run": True,
            "reused_stage_24_trained_checkpoint": False,
            "identical_batch_geometry": False,
            "identical_training_targets": True,
            "identical_fixed_training_panels": True,
            "identical_fixed_validation_panels": True,
            "identical_dropout_probability": True,
            "identical_dropout_masks_guaranteed": False,
            "same_optimizer_configuration": True,
            "same_update_count": True,
            "same_learning_rate_schedule": True,
            "fresh_paired_validation_supported": True,
        }
    )
    payload["evaluation_protocol"] = {
        "paired_long_context_panels": True,
        "paired_128_character_panels": treatment.block_size == 128,
        "panel_batch_size": treatment.batch_size,
        "panel_block_size": treatment.block_size,
        "first_half_panel_positions": (0, control.block_size - 1),
        "second_half_panel_positions": (
            control.block_size,
            treatment.block_size - 1,
        ),
        "control_model_positions_for_each_half": (
            0,
            control.block_size - 1,
        ),
        "checkpoint_selection_objective": "overall_validation_loss",
        "half_losses_are_diagnostics": True,
        "fixed_panel_seed": getattr(checkpoint, "config").seed + 2,
        "split_order": ("train", "val"),
        "batches_per_split": getattr(checkpoint, "config").eval_iters,
        "model_mode": "eval",
        "dropout_enabled": False,
        "gradients_enabled": False,
    }


_STAGE_17._extend_context_checkpoint_payload = _extend_stage_26_checkpoint_payload


def _validate_stage_26_options(
    args: object,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    precise_eval_seed: int,
) -> None:
    del training_batch_seed, training_rng_seed
    precise_eval_iters = getattr(args, "precise_eval_iters")
    if precise_eval_iters == 1:
        raise ValueError(
            "precise-eval-iters must be 0 or at least 2 for paired inference"
        )
    fixed_panel_seed = getattr(args, "seed") + 2
    if precise_eval_iters > 0 and precise_eval_seed == fixed_panel_seed:
        raise ValueError(
            "precise evaluation seed must differ from the fixed-panel seed"
        )


_STAGE_17._validate_context_stage_options = _validate_stage_26_options


_create_paired_initial_states_base = _STAGE_17.create_paired_initial_states


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
        and control_config.seed == 1_337
        and treatment_config.seed == 1_337
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
    initialization = _create_paired_initial_states_base(
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
    ):
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
            "shared parameters": (DEFAULT_SHARED_PARAMETER_INITIALIZATION_SHA256),
            "new position rows": DEFAULT_EXTRA_POSITION_ROWS_SHA256,
            "control parameter count": DEFAULT_CONTROL_PARAMETER_COUNT,
            "treatment parameter count": DEFAULT_TREATMENT_PARAMETER_COUNT,
        }
        if observed != expected:
            raise RuntimeError(
                "canonical Stage 26 initialization changed: "
                f"expected {expected}, got {observed}"
            )
    return initialization


# Stage 17's orchestrator performs a global lookup, so installing the checked
# initializer makes the canonical pins apply to CLI and programmatic runs.
_STAGE_17.create_paired_initial_states = create_paired_initial_states

run_experiment = _STAGE_17.run_experiment
_validate_stage_17_checkpoint_base = _STAGE_17.validate_stage_17_checkpoint


def validate_stage_26_checkpoint(
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
    if path.resolve() != spec.checkpoint_path.resolve():
        raise ValueError("checkpoint path does not match the branch spec")
    checkpoint = _validate_stage_17_checkpoint_base(
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
        shared_parameter_initialization_sha256=(shared_parameter_initialization_sha256),
    )
    rng_state = _require_mapping(checkpoint.get("rng_state"), name="rng_state")
    if not isinstance(rng_state.get("cpu"), torch.Tensor):
        raise ValueError("checkpoint rng_state.cpu must be a tensor")
    for accelerator in ("xpu", "cuda"):
        accelerator_states = rng_state.get(accelerator)
        if accelerator_states is not None and (
            not isinstance(accelerator_states, list)
            or not all(isinstance(value, torch.Tensor) for value in accelerator_states)
        ):
            raise ValueError(
                f"checkpoint rng_state.{accelerator} must be a list of tensors"
            )
    architecture = _require_mapping(
        checkpoint.get("architecture"),
        name="architecture",
    )
    with torch.random.fork_rng(devices=[]):
        compatibility_model = _build_model(data, config, n_head, n_layer)
        expected_parameter_count = sum(
            parameter.numel()
            for parameter in compatibility_model.parameters()
            if parameter.requires_grad
        )
        del compatibility_model
    expected_architecture = {
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "attention_mask": "full_causal",
        "attention_mask_scope": "per_layer",
        "max_context": None,
        "parameter_count": expected_parameter_count,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"checkpoint architecture.{key} must be {expected!r}, "
                f"got {architecture.get(key)!r}"
            )
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    causal_masks = {
        name: tensor for name, tensor in model_state.items() if name.endswith(".tril")
    }
    if len(causal_masks) != n_layer * n_head:
        raise ValueError(
            f"checkpoint must contain {n_layer * n_head} causal masks, "
            f"got {len(causal_masks)}"
        )
    for name, mask in causal_masks.items():
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (
            spec.block_size,
            spec.block_size,
        ):
            raise ValueError(
                f"checkpoint causal mask {name} must have shape "
                f"({spec.block_size}, {spec.block_size})"
            )
    training = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    expected_training = {
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
    initialization = _require_mapping(
        checkpoint.get("initialization"),
        name="initialization",
    )
    expected_initialization: dict[str, object] = {
        "shared_parameter_hash_scope": "all_named_parameters_in_control_model",
        "shared_parameters_byte_identical": True,
        "shared_position_row_range": (0, control.block_size - 1),
        "shared_position_rows_byte_identical": True,
        "new_position_rows_isolated_from_shared_parameter_rng": True,
    }
    if (
        common_state_sha256 == DEFAULT_CONTROL_INITIAL_STATE_SHA256
        and fingerprint_data(data) == DEFAULT_DATA_FINGERPRINT
    ):
        expected_initialization.update(
            {
                "t64_initialization_source_stage": 24,
                "source_initial_state_sha256": (DEFAULT_CONTROL_INITIAL_STATE_SHA256),
                "byte_identical_to_source_initialization": True,
            }
        )
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
        "fixed_transformer_depth": n_layer,
        "fixed_residual_dropout": DEFAULT_DROPOUT,
        "paired_long_segments_per_update": treatment.batch_size,
        "control_windows_per_long_segment": 2,
        "same_training_rng_seed": True,
        "shared_parameter_initialization_sha256": (
            shared_parameter_initialization_sha256
        ),
        "new_position_initialization_sha256": extra_position_rows_sha256,
        "new_control_training_run": True,
        "reused_stage_24_trained_checkpoint": False,
        "identical_batch_geometry": False,
        "identical_training_targets": True,
        "identical_fixed_training_panels": True,
        "identical_fixed_validation_panels": True,
        "identical_dropout_probability": True,
        "identical_dropout_masks_guaranteed": False,
        "same_optimizer_configuration": True,
        "same_update_count": True,
        "same_learning_rate_schedule": True,
        "fresh_paired_validation_supported": True,
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
        "paired_long_context_panels": True,
        "paired_128_character_panels": treatment.block_size == 128,
        "panel_batch_size": treatment.batch_size,
        "panel_block_size": treatment.block_size,
        "first_half_panel_positions": (0, control.block_size - 1),
        "second_half_panel_positions": (
            control.block_size,
            treatment.block_size - 1,
        ),
        "control_model_positions_for_each_half": (
            0,
            control.block_size - 1,
        ),
        "checkpoint_selection_objective": "overall_validation_loss",
        "half_losses_are_diagnostics": True,
        "fixed_panel_seed": config.seed + 2,
        "split_order": ("train", "val"),
        "batches_per_split": config.eval_iters,
        "model_mode": "eval",
        "dropout_enabled": False,
        "gradients_enabled": False,
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(
                f"checkpoint evaluation_protocol.{key} must be {expected!r}, "
                f"got {evaluation.get(key)!r}"
            )
    return checkpoint


_STAGE_17.validate_stage_17_checkpoint = validate_stage_26_checkpoint
_run_precise_validation_base = _STAGE_17.run_precise_validation


def run_precise_validation(*args: object, **kwargs: object) -> PreciseValidationReport:
    eval_iters = kwargs.get("eval_iters")
    if isinstance(eval_iters, bool) or not isinstance(eval_iters, int):
        raise ValueError(f"eval_iters must be an integer, got {eval_iters!r}")
    if eval_iters < 2:
        raise ValueError(
            f"eval_iters must be at least 2 for paired inference, got {eval_iters}"
        )
    if len(args) < 2 or not isinstance(args[1], TrainingConfig):
        raise TypeError("control_config must be the second positional argument")
    seed = kwargs.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if seed == args[1].seed + 2:
        raise ValueError(
            "precise evaluation seed must differ from the fixed-panel seed"
        )
    return _run_precise_validation_base(*args, **kwargs)


_STAGE_17.run_precise_validation = run_precise_validation


def main() -> None:
    _STAGE_17.main()


if __name__ == "__main__":
    main()
