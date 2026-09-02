from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from config import TrainingConfig
from data_utils import CharacterData
from train import resolve_device
from training import EvaluationRecord, LossEstimate, TrainingResult


# Stage 39 resumes Stage 38's exact terminal state and changes only the amount
# of optimization. The learning rate remains 3e-5 across the 25k boundary.
_STAGE_38_PATH = Path(__file__).with_name("38_extended_training.py")
_STAGE_38_MODULE_NAME = "stage_38_extended_training_for_stage_39"
_STAGE_38_SPEC = importlib.util.spec_from_file_location(
    _STAGE_38_MODULE_NAME,
    _STAGE_38_PATH,
)
assert _STAGE_38_SPEC is not None and _STAGE_38_SPEC.loader is not None
_STAGE_38 = importlib.util.module_from_spec(_STAGE_38_SPEC)
sys.modules[_STAGE_38_MODULE_NAME] = _STAGE_38
_STAGE_38_SPEC.loader.exec_module(_STAGE_38)


GPTLanguageModel = _STAGE_38.GPTLanguageModel
ExtendedLearningRateSchedule = _STAGE_38.ExtendedLearningRateSchedule
BestValidationCheckpoint = _STAGE_38.BestValidationCheckpoint
checkpoint_sha256 = _STAGE_38.checkpoint_sha256
clear_accelerator_cache = _STAGE_38.clear_accelerator_cache
fingerprint_data = _STAGE_38.fingerprint_data
generate_from_final_model = _STAGE_38.generate_from_final_model
seed_everything = _STAGE_38.seed_everything
tensor_mapping_sha256 = _STAGE_38.tensor_mapping_sha256
tensor_sha256 = _STAGE_38.tensor_sha256


EXPERIMENT_STAGE = 39
EXPERIMENT_TITLE = "Complete Shakespeare compute extension"
PROTOCOL_NAME = "stage_38_terminal_full_state_constant_lr_continuation"
COMPARISON_VARIABLE = "additional_optimization_at_unchanged_learning_rate"

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
DEFAULT_COMPLETE_DATA_PATH = _STAGE_38.DEFAULT_COMPLETE_DATA_PATH
CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_SOURCE_CHECKPOINT_PATH = _STAGE_38.DEFAULT_FINAL_CHECKPOINT_PATH
DEFAULT_BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_39_complete_shakespeare_extended_best_checkpoint.pt"
)
DEFAULT_FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_39_complete_shakespeare_step_32000_final_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = _STAGE_38.DEFAULT_BATCH_SIZE
DEFAULT_BLOCK_SIZE = _STAGE_38.DEFAULT_BLOCK_SIZE
DEFAULT_LOCAL_CONTEXT = _STAGE_38.DEFAULT_LOCAL_CONTEXT
DEFAULT_LAYER_CONTEXTS = _STAGE_38.DEFAULT_LAYER_CONTEXTS
DEFAULT_GLOBAL_LAYER_INDICES = _STAGE_38.DEFAULT_GLOBAL_LAYER_INDICES
DEFAULT_GLOBAL_LAYER_NUMBERS = _STAGE_38.DEFAULT_GLOBAL_LAYER_NUMBERS
DEFAULT_N_EMBD = _STAGE_38.DEFAULT_N_EMBD
DEFAULT_N_HEAD = _STAGE_38.DEFAULT_N_HEAD
DEFAULT_HEAD_SIZE = _STAGE_38.DEFAULT_HEAD_SIZE
DEFAULT_FEED_FORWARD_WIDTH = _STAGE_38.DEFAULT_FEED_FORWARD_WIDTH
DEFAULT_N_LAYER = _STAGE_38.DEFAULT_N_LAYER
DEFAULT_DROPOUT = _STAGE_38.DEFAULT_DROPOUT

DEFAULT_CONTINUATION_LEARNING_RATE = _STAGE_38.DEFAULT_EXTENSION_LEARNING_RATE
DEFAULT_SOURCE_STEP = _STAGE_38.DEFAULT_TARGET_STEP
DEFAULT_TARGET_STEP = 32_000
DEFAULT_EVAL_INTERVAL = _STAGE_38.DEFAULT_EVAL_INTERVAL
DEFAULT_EVAL_ITERS = _STAGE_38.DEFAULT_EVAL_ITERS

DEFAULT_SEED = _STAGE_38.DEFAULT_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_38.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_FIXED_PANEL_SEED = _STAGE_38.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_38.DEFAULT_TRAINING_RNG_SEED
DEFAULT_SAMPLE_SEED = _STAGE_38.DEFAULT_SAMPLE_SEED
DEFAULT_TRAIN_FRACTION = _STAGE_38.DEFAULT_TRAIN_FRACTION

DEFAULT_SOURCE_BEST_TRAIN_LOSS = 1.2115179455280305
DEFAULT_SOURCE_BEST_VAL_LOSS = 1.4270337796211243
DEFAULT_SOURCE_CHECKPOINT_SHA256 = (
    "8393cf786920ebd5e7e89b1b121f7d2492381825b656fbd2ed98d022432561ac"
)
DEFAULT_SOURCE_GENERATOR_SHA256 = (
    "81407a2a3cb7c6ab04d82c2b890edbe8bdadce88fd1d50de7154401cd800fb8d"
)
DEFAULT_SOURCE_TORCH_VERSION = _STAGE_38.DEFAULT_SOURCE_TORCH_VERSION

DEFAULT_COMPLETE_DATA_FINGERPRINT = _STAGE_38.DEFAULT_COMPLETE_DATA_FINGERPRINT
DEFAULT_COMPLETE_NUM_CHARACTERS = _STAGE_38.DEFAULT_COMPLETE_NUM_CHARACTERS
DEFAULT_COMPLETE_TRAIN_CHARACTERS = _STAGE_38.DEFAULT_COMPLETE_TRAIN_CHARACTERS
DEFAULT_COMPLETE_VAL_CHARACTERS = _STAGE_38.DEFAULT_COMPLETE_VAL_CHARACTERS
DEFAULT_COMPLETE_VOCAB_SIZE = _STAGE_38.DEFAULT_COMPLETE_VOCAB_SIZE
DEFAULT_PARAMETER_COUNT = _STAGE_38.DEFAULT_PARAMETER_COUNT
DEFAULT_INITIAL_STATE_SHA256 = _STAGE_38.DEFAULT_INITIAL_STATE_SHA256
DEFAULT_TARGETS_PER_UPDATE = _STAGE_38.DEFAULT_TARGETS_PER_UPDATE
DEFAULT_TOTAL_TARGET_CHARACTERS = DEFAULT_TARGET_STEP * DEFAULT_TARGETS_PER_UPDATE
DEFAULT_ADDED_TARGET_CHARACTERS = (
    DEFAULT_TARGET_STEP - DEFAULT_SOURCE_STEP
) * DEFAULT_TARGETS_PER_UPDATE

CheckpointKind = Literal["best", "final"]


@dataclass(frozen=True, slots=True)
class Stage38Source:
    path: Path
    sha256: str
    checkpoint: Mapping[str, object]
    step: int
    best_step: int
    best_train_loss: float
    best_val_loss: float
    snapshot_train_loss: float
    snapshot_val_loss: float
    runtime_device_type: str
    runtime_device_name: str
    torch_version: str
    canonical: bool


@dataclass(frozen=True, slots=True)
class SourceRestoreAudit:
    record: EvaluationRecord
    train_matches_source: bool
    val_matches_source: bool
    runtime_matched: bool

    @property
    def verified(self) -> bool:
        return (
            self.train_matches_source
            and self.val_matches_source
            and self.runtime_matched
        )


@dataclass(frozen=True, slots=True)
class Stage39Report:
    source: Stage38Source
    training: TrainingResult
    source_audit: SourceRestoreAudit
    best_train_loss: float
    best_val_loss: float
    best_step: int
    final_batch_generator_sha256: str
    best_checkpoint_path: Path
    best_checkpoint_sha256: str
    final_checkpoint_path: Path
    final_checkpoint_sha256: str
    sample: str

    @property
    def final_step(self) -> int:
        return self.training.history[-1].step

    @property
    def generalization_gap(self) -> float:
        return self.training.final.val - self.training.final.train

    @property
    def validation_reversal(self) -> float:
        return self.training.final.val - self.best_val_loss


_require_mapping = _STAGE_38._require_mapping
_require_int = _STAGE_38._require_int
_require_finite_float = _STAGE_38._require_finite_float
_sha256_file = _STAGE_38._sha256_file
_runtime_identity = _STAGE_38._runtime_identity
_validate_rng_state = _STAGE_38._validate_rng_state
_restore_rng_state = _STAGE_38._restore_rng_state
_optimizer_rates = _STAGE_38._optimizer_rates
_validate_adamw_state = _STAGE_38._validate_adamw_state
_validate_optimizer_steps = _STAGE_38._validate_optimizer_steps
_validate_optimizer_hyperparameters = _STAGE_38._validate_optimizer_hyperparameters
_expected_training_generator_state = _STAGE_38._expected_training_generator_state
_set_optimizer_learning_rate = _STAGE_38._set_optimizer_learning_rate
_atomic_save_payload = _STAGE_38._atomic_save_payload


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    return _STAGE_38.build_model(data, config, n_head, n_layer, device)


def verify_model_masks(model: GPTLanguageModel) -> None:
    _STAGE_38.verify_model_masks(model)


def get_stage_38_batch(
    data: CharacterData,
    split: str,
    config: TrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _STAGE_38.get_stage_37_batch(
        data,
        split,
        config,
        generator,
        device,
    )


def evaluate_on_fixed_batches(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    fixed_panel_seed: int,
) -> LossEstimate:
    return _STAGE_38.evaluate_on_fixed_batches(
        model,
        data,
        config,
        device,
        fixed_panel_seed=fixed_panel_seed,
    )


def is_canonical_protocol(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: ExtendedLearningRateSchedule,
    *,
    training_batch_seed: int = DEFAULT_TRAINING_BATCH_SEED,
    fixed_panel_seed: int = DEFAULT_FIXED_PANEL_SEED,
    training_rng_seed: int = DEFAULT_TRAINING_RNG_SEED,
) -> bool:
    return (
        fingerprint_data(data) == DEFAULT_COMPLETE_DATA_FINGERPRINT
        and config.batch_size == DEFAULT_BATCH_SIZE
        and config.block_size == DEFAULT_BLOCK_SIZE
        and config.n_embd == DEFAULT_N_EMBD
        and config.learning_rate == DEFAULT_CONTINUATION_LEARNING_RATE
        and config.max_iters == DEFAULT_TARGET_STEP
        and config.eval_interval == DEFAULT_EVAL_INTERVAL
        and config.eval_iters == DEFAULT_EVAL_ITERS
        and config.seed == DEFAULT_SEED
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
        and training_batch_seed == DEFAULT_TRAINING_BATCH_SEED
        and fixed_panel_seed == DEFAULT_FIXED_PANEL_SEED
        and training_rng_seed == DEFAULT_TRAINING_RNG_SEED
        and schedule == ExtendedLearningRateSchedule()
    )


def is_canonical_stage_38_source_contract(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: ExtendedLearningRateSchedule,
    *,
    training_batch_seed: int = DEFAULT_TRAINING_BATCH_SEED,
    fixed_panel_seed: int = DEFAULT_FIXED_PANEL_SEED,
    training_rng_seed: int = DEFAULT_TRAINING_RNG_SEED,
) -> bool:
    """Identify the authenticated 25k source independently of the new target."""

    return (
        fingerprint_data(data) == DEFAULT_COMPLETE_DATA_FINGERPRINT
        and config.batch_size == DEFAULT_BATCH_SIZE
        and config.block_size == DEFAULT_BLOCK_SIZE
        and config.n_embd == DEFAULT_N_EMBD
        and config.eval_interval == DEFAULT_EVAL_INTERVAL
        and config.eval_iters == DEFAULT_EVAL_ITERS
        and config.seed == DEFAULT_SEED
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
        and training_batch_seed == DEFAULT_TRAINING_BATCH_SEED
        and fixed_panel_seed == DEFAULT_FIXED_PANEL_SEED
        and training_rng_seed == DEFAULT_TRAINING_RNG_SEED
        and schedule == ExtendedLearningRateSchedule()
    )


def validate_protocol(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: ExtendedLearningRateSchedule,
    *,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> None:
    if config.max_iters <= DEFAULT_SOURCE_STEP:
        raise ValueError(
            f"target_step must be above {DEFAULT_SOURCE_STEP:,}, "
            f"got {config.max_iters:,}"
        )
    if config.learning_rate != DEFAULT_CONTINUATION_LEARNING_RATE:
        raise ValueError(
            "Stage 39 requires an unchanged 3e-5 continuation learning rate"
        )
    if schedule.extension_learning_rate != DEFAULT_CONTINUATION_LEARNING_RATE:
        raise ValueError("Stage 39 must preserve Stage 38's 3e-5 LR phase")
    if schedule != ExtendedLearningRateSchedule():
        raise ValueError("Stage 39 requires the canonical prior LR schedule")
    if n_layer != DEFAULT_N_LAYER:
        raise ValueError(f"Stage 39 requires L={DEFAULT_N_LAYER}, got {n_layer}")
    if n_head <= 0 or config.n_embd % n_head:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({n_head})"
        )
    if config.block_size < DEFAULT_LOCAL_CONTEXT:
        raise ValueError(
            f"block_size must be at least W{DEFAULT_LOCAL_CONTEXT}, "
            f"got {config.block_size}"
        )
    if config.batch_size % 2:
        raise ValueError(
            "Stage 39 requires an even batch size for the preserved long-panel sampler"
        )
    if DEFAULT_SOURCE_STEP % config.eval_interval:
        raise ValueError("the 25,000 restore boundary must be evaluated")
    if config.max_iters % config.eval_interval:
        raise ValueError("the target step must align with the evaluation interval")
    panel_size = 2 * config.block_size
    for split, source in (("train", data.train_data), ("val", data.val_data)):
        if len(source) <= panel_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but Stage 39 "
                f"sampling requires a 2T panel of {panel_size} tokens"
            )
    for name, value in (
        ("training batch seed", training_batch_seed),
        ("fixed panel seed", fixed_panel_seed),
        ("training RNG seed", training_rng_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def validate_artifact_paths(
    *,
    source_checkpoint_path: Path,
    best_checkpoint_path: Path,
    final_checkpoint_path: Path,
    source_data_path: Path | None,
) -> None:
    paths = {
        "Stage 38 source checkpoint": source_checkpoint_path.resolve(),
        "Stage 39 best checkpoint": best_checkpoint_path.resolve(),
        "Stage 39 final checkpoint": final_checkpoint_path.resolve(),
    }
    if source_data_path is not None:
        paths["Complete Shakespeare corpus"] = source_data_path.resolve()
    labels = tuple(paths)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            first = paths[first_label]
            second = paths[second_label]
            if first == second:
                raise ValueError(
                    f"{first_label} and {second_label} must use distinct paths"
                )
            if first.exists() and second.exists() and first.samefile(second):
                raise ValueError(
                    f"{first_label} and {second_label} must not be the same file"
                )


def _stage_38_source_config(config: TrainingConfig) -> TrainingConfig:
    return TrainingConfig(
        batch_size=config.batch_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        learning_rate=DEFAULT_CONTINUATION_LEARNING_RATE,
        max_iters=DEFAULT_SOURCE_STEP,
        eval_interval=config.eval_interval,
        eval_iters=config.eval_iters,
        seed=config.seed,
    )


def _stage_37_source_from_metadata(
    checkpoint: Mapping[str, object],
) -> object:
    metadata = _require_mapping(
        checkpoint.get("source_checkpoint"),
        name="source_checkpoint",
    )
    path_name = metadata.get("path_name")
    sha256 = metadata.get("sha256")
    runtime_device_type = metadata.get("runtime_device_type")
    runtime_device_name = metadata.get("runtime_device_name")
    torch_version = metadata.get("torch_version")
    for name, value in (
        ("path_name", path_name),
        ("sha256", sha256),
        ("runtime_device_type", runtime_device_type),
        ("runtime_device_name", runtime_device_name),
        ("torch_version", torch_version),
    ):
        if not isinstance(value, str):
            raise ValueError(f"Stage 38 source metadata {name!r} is missing")
    step = _require_int(metadata.get("step"), name="source step")
    best_step = _require_int(metadata.get("best_step"), name="source best_step")
    best_train = _require_finite_float(
        metadata.get("best_training_loss"),
        name="source best_training_loss",
    )
    best_val = _require_finite_float(
        metadata.get("best_validation_loss"),
        name="source best_validation_loss",
    )
    canonical = (
        sha256 == _STAGE_38.DEFAULT_SOURCE_CHECKPOINT_SHA256
        and step == _STAGE_38.DEFAULT_SOURCE_STEP
        and best_step == _STAGE_38.DEFAULT_SOURCE_STEP
    )
    return _STAGE_38.Stage37Source(
        path=Path(path_name),
        sha256=sha256,
        checkpoint={},
        step=step,
        best_step=best_step,
        best_train_loss=best_train,
        best_val_loss=best_val,
        runtime_device_type=runtime_device_type,
        runtime_device_name=runtime_device_name,
        torch_version=torch_version,
        canonical=canonical,
    )


def validate_and_load_stage_38_source(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: ExtendedLearningRateSchedule,
    device: torch.device,
    *,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> Stage38Source:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 38 source checkpoint not found: {path}")
    before_hash = _sha256_file(path)
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="root",
    )
    stage_37_source = _stage_37_source_from_metadata(checkpoint)
    source_config = _stage_38_source_config(config)
    validated = _STAGE_38.validate_stage_38_checkpoint(
        path,
        data,
        source_config,
        n_head,
        n_layer,
        schedule,
        stage_37_source,
        expected_kind="final",
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    if validated.get("experiment", {}).get("stage") != _STAGE_38.EXPERIMENT_STAGE:
        raise ValueError("source checkpoint is not a Stage 38 artifact")
    step = _require_int(validated.get("step"), name="step")
    best_step = _require_int(validated.get("best_step"), name="best_step")
    if step != DEFAULT_SOURCE_STEP:
        raise ValueError(f"Stage 39 requires the step-{DEFAULT_SOURCE_STEP:,} endpoint")
    if best_step != step:
        raise ValueError("Stage 39 requires Stage 38's best-at-endpoint state")

    runtime = _require_mapping(validated.get("runtime"), name="runtime")
    runtime_device_type = runtime.get("device_type")
    runtime_device_name = runtime.get("device_name")
    saved_torch_version = runtime.get("torch_version")
    if not isinstance(runtime_device_type, str):
        raise ValueError("Stage 38 runtime device type is missing")
    if not isinstance(runtime_device_name, str):
        raise ValueError("Stage 38 runtime device name is missing")
    if not isinstance(saved_torch_version, str):
        raise ValueError("Stage 38 torch version is missing")
    current_device_type, current_device_name = _runtime_identity(device)
    if (
        runtime_device_type != current_device_type
        or runtime_device_name != current_device_name
    ):
        raise ValueError(
            "exact Stage 38 continuation requires the recorded device/backend: "
            f"saved {runtime_device_type}/{runtime_device_name!r}, current "
            f"{current_device_type}/{current_device_name!r}"
        )
    if saved_torch_version != str(torch.__version__):
        raise ValueError(
            "exact Stage 38 continuation requires the recorded torch version: "
            f"saved {saved_torch_version!r}, current {str(torch.__version__)!r}"
        )

    best_train_loss = _require_finite_float(
        validated.get("best_training_loss"),
        name="best_training_loss",
    )
    best_val_loss = _require_finite_float(
        validated.get("best_val_loss"),
        name="best_val_loss",
    )
    snapshot = _require_mapping(
        validated.get("snapshot_losses"),
        name="snapshot_losses",
    )
    snapshot_train_loss = _require_finite_float(
        snapshot.get("train"),
        name="snapshot train loss",
    )
    snapshot_val_loss = _require_finite_float(
        snapshot.get("validation"),
        name="snapshot validation loss",
    )
    if snapshot_train_loss != best_train_loss or snapshot_val_loss != best_val_loss:
        raise ValueError("Stage 38 endpoint and selected losses must match")

    canonical = is_canonical_stage_38_source_contract(
        data,
        config,
        n_head,
        n_layer,
        schedule,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    generator_state = validated.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 38 source training-generator state is missing")
    if canonical:
        if before_hash != DEFAULT_SOURCE_CHECKPOINT_SHA256:
            raise ValueError(
                "canonical Stage 38 source hash mismatch: expected "
                f"{DEFAULT_SOURCE_CHECKPOINT_SHA256}, got {before_hash}"
            )
        if best_train_loss != DEFAULT_SOURCE_BEST_TRAIN_LOSS:
            raise ValueError("canonical Stage 38 source train loss changed")
        if best_val_loss != DEFAULT_SOURCE_BEST_VAL_LOSS:
            raise ValueError("canonical Stage 38 source validation loss changed")
        if saved_torch_version != DEFAULT_SOURCE_TORCH_VERSION:
            raise ValueError("canonical Stage 38 source torch version changed")
        if tensor_sha256(generator_state) != DEFAULT_SOURCE_GENERATOR_SHA256:
            raise ValueError("canonical Stage 38 source generator state changed")

    after_hash = _sha256_file(path)
    if after_hash != before_hash:
        raise RuntimeError("Stage 38 source changed during validation")
    return Stage38Source(
        path=path.resolve(),
        sha256=before_hash,
        checkpoint=validated,
        step=step,
        best_step=best_step,
        best_train_loss=best_train_loss,
        best_val_loss=best_val_loss,
        snapshot_train_loss=snapshot_train_loss,
        snapshot_val_loss=snapshot_val_loss,
        runtime_device_type=runtime_device_type,
        runtime_device_name=runtime_device_name,
        torch_version=saved_torch_version,
        canonical=canonical,
    )


def restore_stage_38_source(
    source: Stage38Source,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    device: torch.device,
) -> None:
    model_state = _require_mapping(
        source.checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    optimizer_state = _require_mapping(
        source.checkpoint.get("optimizer_state_dict"),
        name="optimizer_state_dict",
    )
    generator_state = source.checkpoint.get("training_generator_state")
    assert isinstance(generator_state, torch.Tensor)
    rng_state = _validate_rng_state(
        source.checkpoint.get("rng_state"),
        runtime_device_type=source.runtime_device_type,
    )
    model.load_state_dict(model_state, strict=True)
    verify_model_masks(model)
    optimizer.load_state_dict(optimizer_state)
    training_generator.set_state(generator_state.cpu())
    # Restore last: model construction and all source diagnostics must not
    # consume the next dropout draw in the resumed stochastic stream.
    _restore_rng_state(rng_state, device)


@dataclass(slots=True)
class Stage39ValidationCheckpoint(BestValidationCheckpoint):
    """Persist the strict selected state and unconditional terminal state."""

    schedule: ExtendedLearningRateSchedule | None = None
    target_step: int = 0
    source: Stage38Source | None = None
    source_filename: str | None = None
    source_bytes: int | None = None
    source_sha256: str | None = None
    num_characters: int = 0
    train_characters: int = 0
    val_characters: int = 0
    vocabulary_size: int = 0
    train_fraction: float = 0.0
    targets_per_update: int = 0
    training_batch_seed: int = 0
    fixed_panel_seed: int = 0
    training_rng_seed: int = 0
    parameter_count: int = 0
    initial_state_sha256: str = ""
    best_train_loss: float | None = None
    source_audit_verified: bool = False

    def _decorate_payload(
        self,
        payload: dict[str, Any],
        *,
        step: int,
        checkpoint_kind: CheckpointKind,
        snapshot_losses: LossEstimate,
    ) -> dict[str, Any]:
        if (
            self.schedule is None
            or self.source is None
            or self.best_step is None
            or self.best_train_loss is None
        ):
            raise RuntimeError("incomplete Stage 39 checkpoint metadata")
        optimizer_state = _require_mapping(
            payload.get("optimizer_state_dict"),
            name="optimizer_state_dict",
        )
        if _optimizer_rates(optimizer_state) != {DEFAULT_CONTINUATION_LEARNING_RATE}:
            raise RuntimeError(
                f"optimizer LR at step {step} must remain "
                f"{DEFAULT_CONTINUATION_LEARNING_RATE:g}"
            )
        payload["checkpoint_kind"] = checkpoint_kind
        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "head_size": self.config.n_embd // self.n_head,
                "feed_forward_width": 4 * self.config.n_embd,
                "residual_dropout": DEFAULT_DROPOUT,
                "dropout_placement": (_STAGE_38._STAGE_37._STAGE_36.DROPOUT_PLACEMENT),
                "layer_contexts": DEFAULT_LAYER_CONTEXTS,
                "topology": "LLLLGG",
                "vocabulary_size": self.vocabulary_size,
                "parameter_count": self.parameter_count,
                "attention_mask": "W23,W23,W23,W23,Full,Full",
                "attention_mask_scope": "per_layer",
                "local_context": DEFAULT_LOCAL_CONTEXT,
                "global_layer_indices": DEFAULT_GLOBAL_LAYER_INDICES,
                "global_layer_numbers": DEFAULT_GLOBAL_LAYER_NUMBERS,
            }
        )
        training = payload["training_config"]
        assert isinstance(training, dict)
        training.update(
            {
                "learning_rate": DEFAULT_CONTINUATION_LEARNING_RATE,
                "max_iters": self.target_step,
                "target_step": self.target_step,
                "source_checkpoint_step": self.source.step,
                "continuation_learning_rate": (DEFAULT_CONTINUATION_LEARNING_RATE),
                "learning_rate_schedule": self.schedule.as_metadata(
                    target_step=self.target_step
                ),
                "continuation_segment": {
                    "start_update": self.source.step,
                    "end_update_exclusive": self.target_step,
                    "learning_rate": DEFAULT_CONTINUATION_LEARNING_RATE,
                },
                "residual_dropout": DEFAULT_DROPOUT,
                "training_batch_seed": self.training_batch_seed,
                "fixed_panel_seed": self.fixed_panel_seed,
                "training_rng_seed": self.training_rng_seed,
                "targets_per_update": self.targets_per_update,
                "total_target_presentations": (
                    self.targets_per_update * self.target_step
                ),
                "added_target_presentations_after_stage_38": (
                    self.targets_per_update * (self.target_step - self.source.step)
                ),
            }
        )
        payload["source_checkpoint"] = {
            "stage": _STAGE_38.EXPERIMENT_STAGE,
            "path_name": self.source.path.name,
            "sha256": self.source.sha256,
            "checkpoint_kind": "final",
            "step": self.source.step,
            "best_step": self.source.best_step,
            "best_training_loss": self.source.best_train_loss,
            "best_validation_loss": self.source.best_val_loss,
            "snapshot_training_loss": self.source.snapshot_train_loss,
            "snapshot_validation_loss": self.source.snapshot_val_loss,
            "model_state_restored": True,
            "optimizer_state_restored": True,
            "training_generator_restored": True,
            "cpu_rng_restored": True,
            "accelerator_rng_restored": self.device.type in ("xpu", "cuda"),
            "optimizer_restart": False,
            "runtime_device_type": self.source.runtime_device_type,
            "runtime_device_name": self.source.runtime_device_name,
            "torch_version": self.source.torch_version,
            "restore_audit_verified_at_snapshot": self.source_audit_verified,
        }
        payload["corpus"] = {
            "name": "Complete Shakespeare",
            "source_filename": self.source_filename,
            "source_bytes": self.source_bytes,
            "source_sha256": self.source_sha256,
            "data_fingerprint": self.data_fingerprint,
            "decoded_characters": self.num_characters,
            "train_characters": self.train_characters,
            "validation_characters": self.val_characters,
            "train_fraction": self.train_fraction,
            "vocabulary_size": self.vocabulary_size,
        }
        payload["initialization"] = {
            "origin_stage": 37,
            "kind": "restored_stage_38_terminal_complete_shakespeare_state",
            "fresh_initial_state_sha256": self.initial_state_sha256,
            "architecture_changed": False,
            "corpus_changed": False,
        }
        payload["experiment"] = {
            "stage": EXPERIMENT_STAGE,
            "title": EXPERIMENT_TITLE,
            "protocol": PROTOCOL_NAME,
            "comparison_variable": COMPARISON_VARIABLE,
            "from_scratch": False,
            "same_model_as_stage_38": True,
            "same_corpus_as_stage_38": True,
            "same_split_as_stage_38": True,
            "same_sampler_as_stage_38": True,
            "same_fixed_evaluation_panel_as_stage_38": True,
            "same_learning_rate_as_stage_38_endpoint": True,
            "optimizer_state_preserved": True,
            "optimizer_restarted": False,
            "rng_state_preserved": True,
            "architecture_frozen": True,
            "trained_branches": 1,
        }
        runtime_type, runtime_name = _runtime_identity(self.device)
        payload["runtime"] = {
            "device_type": runtime_type,
            "device_name": runtime_name,
            "torch_version": str(torch.__version__),
            "matches_stage_38_source": (
                runtime_type == self.source.runtime_device_type
                and runtime_name == self.source.runtime_device_name
                and str(torch.__version__) == self.source.torch_version
            ),
        }
        payload["evaluation_protocol"] = {
            "selection_metric": "overall_fixed_panel_validation_loss",
            "fixed_panel_seed": self.fixed_panel_seed,
            "eval_interval": self.config.eval_interval,
            "eval_iters": self.config.eval_iters,
            "strict_improvement": True,
            "dropout_enabled": False,
            "absolute_step_numbering": True,
        }
        payload["snapshot_losses"] = {
            "train": snapshot_losses.train,
            "validation": snapshot_losses.val,
        }
        payload["best_training_loss"] = self.best_train_loss
        return payload

    def _payload(self, step: int) -> dict[str, Any]:
        if self.best_train_loss is None:
            raise RuntimeError("best training loss is unavailable")
        return self._decorate_payload(
            super(Stage39ValidationCheckpoint, self)._payload(step),
            step=step,
            checkpoint_kind="best",
            snapshot_losses=LossEstimate(
                train=self.best_train_loss,
                val=self.best_val_loss,
            ),
        )

    def save_final(self, path: Path, record: EvaluationRecord) -> None:
        payload = self._decorate_payload(
            super(Stage39ValidationCheckpoint, self)._payload(record.step),
            step=record.step,
            checkpoint_kind="final",
            snapshot_losses=record.losses,
        )
        _atomic_save_payload(path, payload)

    def consider(self, record: EvaluationRecord) -> bool:
        if not math.isfinite(record.losses.val):
            raise ValueError(f"validation loss at step {record.step} must be finite")
        if record.losses.val >= self.best_val_loss:
            return False
        self.best_val_loss = record.losses.val
        self.best_step = record.step
        self.best_train_loss = record.losses.train
        self.save(record.step)
        return True


def train_continuation(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    schedule: ExtendedLearningRateSchedule,
    *,
    start_step: int,
    fixed_panel_seed: int,
    on_evaluation: Callable[[EvaluationRecord], None],
) -> TrainingResult:
    if start_step != DEFAULT_SOURCE_STEP:
        raise ValueError(
            f"Stage 39 must start at step {DEFAULT_SOURCE_STEP:,}, got {start_step:,}"
        )
    if start_step >= config.max_iters:
        raise ValueError("start_step must precede the target step")
    if start_step % config.eval_interval:
        raise ValueError("start_step must align with the evaluation interval")
    history: list[EvaluationRecord] = []

    def record(step: int) -> EvaluationRecord:
        evaluation = EvaluationRecord(
            step=step,
            losses=evaluate_on_fixed_batches(
                model,
                data,
                config,
                device,
                fixed_panel_seed=fixed_panel_seed,
            ),
        )
        history.append(evaluation)
        on_evaluation(evaluation)
        return evaluation

    model.train()
    initial_record = record(start_step)
    for update_index in range(start_step, config.max_iters):
        learning_rate = schedule.for_update(update_index)
        if learning_rate != DEFAULT_CONTINUATION_LEARNING_RATE:
            raise RuntimeError(
                f"Stage 39 update {update_index} would change LR to {learning_rate:g}"
            )
        _set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_stage_38_batch(
            data,
            "train",
            config,
            training_generator,
            device,
        )
        logits, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        del logits
        del loss
        del inputs
        del targets

        completed_step = update_index + 1
        if (
            completed_step % config.eval_interval == 0
            or completed_step == config.max_iters
        ):
            record(completed_step)

    return TrainingResult(
        initial=initial_record.losses,
        final=history[-1].losses,
        history=tuple(history),
        benchmark=None,
    )


def _reported_value_matches(observed: float, source: float) -> bool:
    return f"{observed:.4f}" == f"{source:.4f}"


def make_evaluation_callback(
    best: Stage39ValidationCheckpoint,
    source: Stage38Source,
    *,
    final_checkpoint_path: Path,
) -> tuple[Callable[[EvaluationRecord], None], list[SourceRestoreAudit]]:
    source_audits: list[SourceRestoreAudit] = []

    def record_evaluation(record: EvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr "
            f"{DEFAULT_CONTINUATION_LEARNING_RATE:g} | "
            f"train {record.losses.train:.4f} | "
            f"val {record.losses.val:.4f} | "
            f"gap {record.losses.val - record.losses.train:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )

        if record.step == source.step:
            audit = SourceRestoreAudit(
                record=record,
                train_matches_source=_reported_value_matches(
                    record.losses.train,
                    source.snapshot_train_loss,
                ),
                val_matches_source=_reported_value_matches(
                    record.losses.val,
                    source.snapshot_val_loss,
                ),
                runtime_matched=True,
            )
            if not audit.verified:
                raise RuntimeError(
                    "the restored step-25,000 fixed-panel loss does not match "
                    "the Stage 38 source at four decimals: expected "
                    f"{source.snapshot_train_loss:.4f} / "
                    f"{source.snapshot_val_loss:.4f}, got "
                    f"{record.losses.train:.4f} / {record.losses.val:.4f}"
                )
            best.source_audit_verified = audit.verified
            if best.best_step == record.step:
                best.save(record.step)
            source_audits.append(audit)

        if record.step == best.target_step:
            if not best.source_audit_verified:
                raise RuntimeError("training reached its target without source audit")
            best.save_final(final_checkpoint_path, record)

    return record_evaluation, source_audits


def validate_stage_39_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: ExtendedLearningRateSchedule,
    source: Stage38Source,
    *,
    expected_kind: CheckpointKind,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> Mapping[str, object]:
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="root",
    )
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported Stage 39 checkpoint version")
    if checkpoint.get("checkpoint_kind") != expected_kind:
        raise ValueError(
            f"expected a {expected_kind!r} Stage 39 checkpoint, got "
            f"{checkpoint.get('checkpoint_kind')!r}"
        )
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 39 checkpoint must not restart AdamW")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 39 optimizer provenance must remain known")
    step = _require_int(checkpoint.get("step"), name="step")
    best_step = _require_int(checkpoint.get("best_step"), name="best_step")
    best_val_loss = _require_finite_float(
        checkpoint.get("best_val_loss"),
        name="best_val_loss",
    )
    best_train_loss = _require_finite_float(
        checkpoint.get("best_training_loss"),
        name="best_training_loss",
    )
    if expected_kind == "best" and step != best_step:
        raise ValueError("Stage 39 best checkpoint must store its selected step")
    if expected_kind == "final" and step != config.max_iters:
        raise ValueError("Stage 39 final checkpoint is not at the target step")
    if not (source.step <= best_step <= step):
        raise ValueError("Stage 39 best step is outside the saved trajectory")

    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    if experiment.get("stage") != EXPERIMENT_STAGE:
        raise ValueError("checkpoint is not a Stage 39 artifact")
    for key, expected in (
        ("same_model_as_stage_38", True),
        ("same_corpus_as_stage_38", True),
        ("same_split_as_stage_38", True),
        ("same_sampler_as_stage_38", True),
        ("same_fixed_evaluation_panel_as_stage_38", True),
        ("same_learning_rate_as_stage_38_endpoint", True),
        ("optimizer_state_preserved", True),
        ("optimizer_restarted", False),
        ("rng_state_preserved", True),
        ("architecture_frozen", True),
        ("trained_branches", 1),
    ):
        if experiment.get(key) != expected:
            raise ValueError(f"Stage 39 experiment field {key!r} changed")

    source_metadata = _require_mapping(
        checkpoint.get("source_checkpoint"),
        name="source_checkpoint",
    )
    for key, expected in (
        ("stage", _STAGE_38.EXPERIMENT_STAGE),
        ("sha256", source.sha256),
        ("checkpoint_kind", "final"),
        ("step", source.step),
        ("best_step", source.best_step),
        ("model_state_restored", True),
        ("optimizer_state_restored", True),
        ("training_generator_restored", True),
        ("cpu_rng_restored", True),
        ("optimizer_restart", False),
        ("runtime_device_type", source.runtime_device_type),
        ("runtime_device_name", source.runtime_device_name),
        ("torch_version", source.torch_version),
        ("restore_audit_verified_at_snapshot", True),
    ):
        if source_metadata.get(key) != expected:
            raise ValueError(f"Stage 39 source field {key!r} mismatch")

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
        "layer_contexts": DEFAULT_LAYER_CONTEXTS,
        "topology": "LLLLGG",
        "vocabulary_size": data.vocabulary.size,
        "attention_mask": "W23,W23,W23,W23,Full,Full",
    }
    for key, expected in expected_architecture.items():
        observed = architecture.get(key)
        if key == "layer_contexts" and observed is not None:
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(
                f"Stage 39 architecture {key!r} mismatch: expected "
                f"{expected!r}, got {observed!r}"
            )

    training = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    expected_training = {
        "batch_size": config.batch_size,
        "learning_rate": DEFAULT_CONTINUATION_LEARNING_RATE,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": config.max_iters,
        "target_step": config.max_iters,
        "source_checkpoint_step": source.step,
        "continuation_learning_rate": DEFAULT_CONTINUATION_LEARNING_RATE,
        "learning_rate_schedule": schedule.as_metadata(target_step=config.max_iters),
        "continuation_segment": {
            "start_update": source.step,
            "end_update_exclusive": config.max_iters,
            "learning_rate": DEFAULT_CONTINUATION_LEARNING_RATE,
        },
        "training_batch_seed": training_batch_seed,
        "fixed_panel_seed": fixed_panel_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Stage 39 training field {key!r} mismatch")

    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("Stage 39 checkpoint corpus fingerprint mismatch")
    corpus = _require_mapping(checkpoint.get("corpus"), name="corpus")
    for key, expected in (
        ("data_fingerprint", fingerprint_data(data)),
        ("decoded_characters", data.num_characters),
        ("train_characters", len(data.train_data)),
        ("validation_characters", len(data.val_data)),
        ("vocabulary_size", data.vocabulary.size),
    ):
        if corpus.get(key) != expected:
            raise ValueError(f"Stage 39 corpus field {key!r} mismatch")

    runtime = _require_mapping(checkpoint.get("runtime"), name="runtime")
    runtime_type = runtime.get("device_type")
    runtime_name = runtime.get("device_name")
    if runtime_type != source.runtime_device_type:
        raise ValueError("Stage 39 runtime type does not match Stage 38")
    if runtime_name != source.runtime_device_name:
        raise ValueError("Stage 39 runtime device does not match Stage 38")
    if runtime.get("torch_version") != source.torch_version:
        raise ValueError("Stage 39 torch version changed during continuation")
    if runtime.get("matches_stage_38_source") is not True:
        raise ValueError("Stage 39 runtime must match the Stage 38 source")
    assert isinstance(runtime_type, str)
    _validate_rng_state(
        checkpoint.get("rng_state"),
        runtime_device_type=runtime_type,
    )

    optimizer_state = _validate_adamw_state(
        checkpoint.get("optimizer_state_dict"),
        expected_learning_rate=DEFAULT_CONTINUATION_LEARNING_RATE,
    )
    _validate_optimizer_steps(optimizer_state, expected_step=step)
    _validate_optimizer_hyperparameters(optimizer_state)
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 39 training-generator state must be a tensor")
    expected_generator_state = _expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator_state):
        raise ValueError("Stage 39 batch-generator state is inconsistent")

    snapshot_losses = _require_mapping(
        checkpoint.get("snapshot_losses"),
        name="snapshot_losses",
    )
    snapshot_train = _require_finite_float(
        snapshot_losses.get("train"),
        name="snapshot train loss",
    )
    snapshot_val = _require_finite_float(
        snapshot_losses.get("validation"),
        name="snapshot validation loss",
    )
    if expected_kind == "best":
        if snapshot_train != best_train_loss or snapshot_val != best_val_loss:
            raise ValueError("Stage 39 best snapshot losses are inconsistent")

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    with torch.random.fork_rng(devices=[]):
        model = build_model(data, config, n_head, n_layer)
    model.load_state_dict(model_state, strict=True)
    verify_model_masks(model)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if architecture.get("parameter_count") != parameter_count:
        raise ValueError("Stage 39 checkpoint parameter count mismatch")
    return checkpoint


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: ExtendedLearningRateSchedule,
    source_checkpoint_path: Path,
    best_checkpoint_path: Path,
    final_checkpoint_path: Path,
    *,
    source_data_path: Path | None,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> Stage39Report:
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    validate_protocol(
        data,
        config,
        n_head,
        n_layer,
        schedule,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    validate_artifact_paths(
        source_checkpoint_path=source_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
        source_data_path=source_data_path,
    )
    source = validate_and_load_stage_38_source(
        source_checkpoint_path,
        data,
        config,
        n_head,
        n_layer,
        schedule,
        device,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    source_hash_before_training = _sha256_file(source.path)

    clear_accelerator_cache(device)
    seed_everything(config.seed)
    model = build_model(data, config, n_head, n_layer, device)
    verify_model_masks(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_CONTINUATION_LEARNING_RATE,
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    restore_stage_38_source(
        source,
        model,
        optimizer,
        training_generator,
        device,
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if source.canonical and parameter_count != DEFAULT_PARAMETER_COUNT:
        raise RuntimeError("canonical Stage 39 parameter count changed")

    source_corpus = _require_mapping(
        source.checkpoint.get("corpus"),
        name="source corpus",
    )
    source_initialization = _require_mapping(
        source.checkpoint.get("initialization"),
        name="source initialization",
    )
    initial_state_sha256 = source_initialization.get("fresh_initial_state_sha256")
    if not isinstance(initial_state_sha256, str):
        raise ValueError("Stage 38 initialization hash is missing")
    best = Stage39ValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=best_checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        best_val_loss=source.best_val_loss,
        best_step=source.best_step,
        schedule=schedule,
        target_step=config.max_iters,
        source=source,
        source_filename=source_corpus.get("source_filename"),
        source_bytes=source_corpus.get("source_bytes"),
        source_sha256=source_corpus.get("source_sha256"),
        num_characters=data.num_characters,
        train_characters=len(data.train_data),
        val_characters=len(data.val_data),
        vocabulary_size=data.vocabulary.size,
        train_fraction=len(data.train_data) / data.num_characters,
        targets_per_update=config.batch_size * config.block_size,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
        parameter_count=parameter_count,
        initial_state_sha256=initial_state_sha256,
        best_train_loss=source.best_train_loss,
    )
    # The Stage 38 endpoint is also its selected state, so Stage 39 can carry
    # that winner forward before any new update is attempted.
    best.save(source.step)
    callback, source_audits = make_evaluation_callback(
        best,
        source,
        final_checkpoint_path=final_checkpoint_path,
    )

    print("\nComplete Shakespeare continuation (the only Stage 39 branch)")
    print(
        f"  B={config.batch_size}, T={config.block_size}, C={config.n_embd}, "
        f"H={n_head}, D={config.n_embd // n_head}, "
        f"FF={4 * config.n_embd}, L={n_layer}, dropout={DEFAULT_DROPOUT:g}"
    )
    print("  attention: W23, W23, W23, W23, Full, Full")
    print(f"  source:                 {source.path}")
    print(f"  source SHA-256:         {source.sha256}")
    print(f"  restore step:           {source.step:,}")
    print(
        f"  extend to:              {config.max_iters:,} at unchanged lr "
        f"{DEFAULT_CONTINUATION_LEARNING_RATE:g}"
    )
    print("  optimizer lineage:      restored; no restart")
    print("  batch/dropout streams:  restored")

    result = train_continuation(
        model,
        optimizer,
        data,
        config,
        device,
        training_generator,
        schedule,
        start_step=source.step,
        fixed_panel_seed=fixed_panel_seed,
        on_evaluation=callback,
    )
    if len(source_audits) != 1:
        raise RuntimeError("Stage 39 did not perform exactly one source audit")
    source_audit = source_audits[0]
    assert best.best_step is not None
    assert best.best_train_loss is not None

    source_hash_after_training = _sha256_file(source.path)
    if source_hash_after_training != source_hash_before_training:
        raise RuntimeError("Stage 38 source checkpoint changed during Stage 39")
    for artifact_path, kind in (
        (best_checkpoint_path, "best"),
        (final_checkpoint_path, "final"),
    ):
        validate_stage_39_checkpoint(
            artifact_path,
            data,
            config,
            n_head,
            n_layer,
            schedule,
            source,
            expected_kind=kind,
            training_batch_seed=training_batch_seed,
            fixed_panel_seed=fixed_panel_seed,
            training_rng_seed=training_rng_seed,
        )

    selected = _require_mapping(
        torch.load(best_checkpoint_path, map_location="cpu", weights_only=True),
        name="selected Stage 39 checkpoint",
    )
    selected_state = _require_mapping(
        selected.get("model_state_dict"),
        name="selected model_state_dict",
    )
    model.load_state_dict(selected_state, strict=True)
    verify_model_masks(model)
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=DEFAULT_SAMPLE_SEED,
    )
    report = Stage39Report(
        source=source,
        training=result,
        source_audit=source_audit,
        best_train_loss=best.best_train_loss,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        final_batch_generator_sha256=tensor_sha256(training_generator.get_state()),
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_sha256=checkpoint_sha256(best_checkpoint_path),
        final_checkpoint_path=final_checkpoint_path,
        final_checkpoint_sha256=checkpoint_sha256(final_checkpoint_path),
        sample=sample,
    )
    del callback
    del best
    del optimizer
    del model
    clear_accelerator_cache(device)
    return report


def print_final_summary(report: Stage39Report) -> None:
    print("\nStage 39 summary")
    print(
        f"  Stage 38 endpoint:       train "
        f"{report.source.snapshot_train_loss:.4f} / val "
        f"{report.source.snapshot_val_loss:.4f} @ {report.source.step:,}"
    )
    print(f"  source audit verified:   {report.source_audit.verified}")
    print(
        f"  extended selected:       train {report.best_train_loss:.4f} / "
        f"val {report.best_val_loss:.4f} @ {report.best_step:,}"
    )
    print(
        f"  extended final:          train "
        f"{report.training.final.train:.4f} / val "
        f"{report.training.final.val:.4f} @ {report.final_step:,}"
    )
    print(f"  final generalization gap:{report.generalization_gap:9.4f}")
    print(f"  val reversal from best:  {report.validation_reversal:+.4f}")
    print(f"  best checkpoint:         {report.best_checkpoint_path}")
    print(f"  best SHA-256:            {report.best_checkpoint_sha256}")
    print(f"  final checkpoint:        {report.final_checkpoint_path}")
    print(f"  final SHA-256:           {report.final_checkpoint_sha256}")
    print("\nComplete Shakespeare selected-best generated text")
    print(report.sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 39: restore the exact Stage 38 step-25,000 terminal state "
            "and continue the same Complete Shakespeare model at an unchanged "
            "3e-5 learning rate."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_COMPLETE_DATA_PATH)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINT_PATH,
    )
    parser.add_argument("--target-step", type=int, default=DEFAULT_TARGET_STEP)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument(
        "--best-checkpoint-path",
        type=Path,
        default=DEFAULT_BEST_CHECKPOINT_PATH,
    )
    parser.add_argument("--final-checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _default_final_checkpoint_path(target_step: int) -> Path:
    if target_step == DEFAULT_TARGET_STEP:
        return DEFAULT_FINAL_CHECKPOINT_PATH
    return (
        CHECKPOINT_DIRECTORY
        / f"stage_39_complete_shakespeare_step_{target_step}_final_checkpoint.pt"
    )


def main() -> None:
    args = parse_args()
    schedule = ExtendedLearningRateSchedule()
    config = TrainingConfig(
        batch_size=DEFAULT_BATCH_SIZE,
        block_size=DEFAULT_BLOCK_SIZE,
        n_embd=DEFAULT_N_EMBD,
        learning_rate=DEFAULT_CONTINUATION_LEARNING_RATE,
        max_iters=args.target_step,
        eval_interval=DEFAULT_EVAL_INTERVAL,
        eval_iters=DEFAULT_EVAL_ITERS,
        seed=DEFAULT_SEED,
    )
    data = CharacterData.from_file(
        args.data,
        train_fraction=DEFAULT_TRAIN_FRACTION,
    )
    device = resolve_device(args.device)
    final_checkpoint_path = (
        _default_final_checkpoint_path(args.target_step)
        if args.final_checkpoint_path is None
        else args.final_checkpoint_path
    )

    print(f"Stage {EXPERIMENT_STAGE}: {EXPERIMENT_TITLE}")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Source: {args.data.resolve()}")
    print(f"Decoded characters: {data.num_characters:,}")
    print(f"Train / validation: {len(data.train_data):,} / {len(data.val_data):,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Protocol: restore {DEFAULT_SOURCE_STEP:,}, continue to {config.max_iters:,}"
    )
    print(
        f"Continuation LR: [{DEFAULT_SOURCE_STEP:,},{config.max_iters:,}) "
        f"at {DEFAULT_CONTINUATION_LEARNING_RATE:g}"
    )

    report = run_experiment(
        data,
        config,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        device,
        args.sample_length,
        schedule,
        args.source_checkpoint,
        args.best_checkpoint_path,
        final_checkpoint_path,
        source_data_path=args.data,
        training_batch_seed=DEFAULT_TRAINING_BATCH_SEED,
        fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED,
        training_rng_seed=DEFAULT_TRAINING_RNG_SEED,
    )
    print_final_summary(report)


if __name__ == "__main__":
    main()
