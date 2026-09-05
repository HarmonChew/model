from __future__ import annotations

import argparse
import copy
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


# Stage 41 is deliberately a one-branch continuation.  Stage 40 remains the
# authority for the architecture, sampler, checkpoint primitives, and every
# seed.  The only new operation is another 6,000 updates on its winning A-final
# state.
_STAGE_40_PATH = Path(__file__).with_name("40_late_learning_rate_refinement.py")
_STAGE_40_MODULE_NAME = "stage_40_late_learning_rate_refinement_for_stage_41"
_STAGE_40_SPEC = importlib.util.spec_from_file_location(
    _STAGE_40_MODULE_NAME,
    _STAGE_40_PATH,
)
assert _STAGE_40_SPEC is not None and _STAGE_40_SPEC.loader is not None
_STAGE_40 = importlib.util.module_from_spec(_STAGE_40_SPEC)
sys.modules[_STAGE_40_MODULE_NAME] = _STAGE_40
_STAGE_40_SPEC.loader.exec_module(_STAGE_40)


GPTLanguageModel = _STAGE_40.GPTLanguageModel
BestValidationCheckpoint = _STAGE_40.BestValidationCheckpoint
checkpoint_sha256 = _STAGE_40.checkpoint_sha256
clear_accelerator_cache = _STAGE_40.clear_accelerator_cache
fingerprint_data = _STAGE_40.fingerprint_data
generate_from_final_model = _STAGE_40.generate_from_final_model
seed_everything = _STAGE_40.seed_everything
tensor_mapping_sha256 = _STAGE_40.tensor_mapping_sha256
tensor_sha256 = _STAGE_40.tensor_sha256
capture_rng_state = _STAGE_40.capture_rng_state
structured_state_sha256 = _STAGE_40.structured_state_sha256

_require_mapping = _STAGE_40._require_mapping
_require_int = _STAGE_40._require_int
_require_finite_float = _STAGE_40._require_finite_float
_sha256_file = _STAGE_40._sha256_file
_runtime_identity = _STAGE_40._runtime_identity
_validate_rng_state = _STAGE_40._validate_rng_state
_restore_rng_state = _STAGE_40._restore_rng_state
_optimizer_rates = _STAGE_40._optimizer_rates
_validate_adamw_state = _STAGE_40._validate_adamw_state
_validate_optimizer_steps = _STAGE_40._validate_optimizer_steps
_validate_optimizer_hyperparameters = _STAGE_40._validate_optimizer_hyperparameters
_expected_training_generator_state = _STAGE_40._expected_training_generator_state
_set_optimizer_learning_rate = _STAGE_40._set_optimizer_learning_rate
_atomic_save_payload = _STAGE_40._atomic_save_payload


EXPERIMENT_STAGE = 41
EXPERIMENT_TITLE = "Complete Shakespeare final compute extension"
PROTOCOL_NAME = "exact_stage_40_a_final_full_state_continuation"
COMPARISON_VARIABLE = "additional_compute_at_fixed_3e_5"

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
DEFAULT_COMPLETE_DATA_PATH = _STAGE_40.DEFAULT_COMPLETE_DATA_PATH
CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_SOURCE_CHECKPOINT_PATH = _STAGE_40.DEFAULT_CONTROL_FINAL_CHECKPOINT_PATH
DEFAULT_INHERITED_BEST_CHECKPOINT_PATH = (
    _STAGE_40.DEFAULT_CONTROL_BEST_CHECKPOINT_PATH
)
DEFAULT_BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_41_complete_shakespeare_best_checkpoint.pt"
)
DEFAULT_FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_41_complete_shakespeare_step_44000_final_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = _STAGE_40.DEFAULT_BATCH_SIZE
DEFAULT_BLOCK_SIZE = _STAGE_40.DEFAULT_BLOCK_SIZE
DEFAULT_LOCAL_CONTEXT = _STAGE_40.DEFAULT_LOCAL_CONTEXT
DEFAULT_LAYER_CONTEXTS = _STAGE_40.DEFAULT_LAYER_CONTEXTS
DEFAULT_GLOBAL_LAYER_INDICES = _STAGE_40.DEFAULT_GLOBAL_LAYER_INDICES
DEFAULT_GLOBAL_LAYER_NUMBERS = _STAGE_40.DEFAULT_GLOBAL_LAYER_NUMBERS
DEFAULT_N_EMBD = _STAGE_40.DEFAULT_N_EMBD
DEFAULT_N_HEAD = _STAGE_40.DEFAULT_N_HEAD
DEFAULT_HEAD_SIZE = _STAGE_40.DEFAULT_HEAD_SIZE
DEFAULT_FEED_FORWARD_WIDTH = _STAGE_40.DEFAULT_FEED_FORWARD_WIDTH
DEFAULT_N_LAYER = _STAGE_40.DEFAULT_N_LAYER
DEFAULT_DROPOUT = _STAGE_40.DEFAULT_DROPOUT

DEFAULT_CONTINUATION_LEARNING_RATE = _STAGE_40.DEFAULT_CONTROL_LEARNING_RATE
DEFAULT_SOURCE_STEP = _STAGE_40.DEFAULT_TARGET_STEP
DEFAULT_TARGET_STEP = 44_000
DEFAULT_EVAL_INTERVAL = _STAGE_40.DEFAULT_EVAL_INTERVAL
DEFAULT_EVAL_ITERS = _STAGE_40.DEFAULT_EVAL_ITERS

DEFAULT_SEED = _STAGE_40.DEFAULT_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_40.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_FIXED_PANEL_SEED = _STAGE_40.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_40.DEFAULT_TRAINING_RNG_SEED
DEFAULT_SAMPLE_SEED = _STAGE_40.DEFAULT_SAMPLE_SEED
DEFAULT_TRAIN_FRACTION = _STAGE_40.DEFAULT_TRAIN_FRACTION

DEFAULT_SOURCE_BEST_STEP = 37_500
DEFAULT_SOURCE_BEST_TRAIN_LOSS = 1.2028024125099182
DEFAULT_SOURCE_BEST_VAL_LOSS = 1.4204677259922027
DEFAULT_SOURCE_SNAPSHOT_TRAIN_LOSS = 1.2023926413059234
DEFAULT_SOURCE_SNAPSHOT_VAL_LOSS = 1.421003932952881
DEFAULT_SOURCE_CHECKPOINT_SHA256 = (
    "494125b6df4a2eec7f15918e203df9ddbe5326c0a3cf145489a181a664ad56c1"
)
DEFAULT_INHERITED_BEST_CHECKPOINT_SHA256 = (
    "72ed5b3facbabbc32aa76b7d0cff48a342a0624ed2890cd7bd8b9d7a0bd3cbda"
)
DEFAULT_SOURCE_MODEL_SHA256 = (
    "ad0b123a50129a63b52da94504b7990d520b1d583ced166394f1041922f66802"
)
DEFAULT_SOURCE_OPTIMIZER_SHA256 = (
    "350b450c8788454eee822505070546f2500ad2e040fd198b988169a9ace9f410"
)
DEFAULT_SOURCE_GENERATOR_SHA256 = (
    "3fc8abee810af0ae964fdb3cfeb99549fc293b44edb59f4ebc1d9a4a2dcd2973"
)
DEFAULT_SOURCE_RNG_SHA256 = (
    "21795b1d257f3ad202efe9588ce933f86a7628d29d4fa4fd84cee3267c90bbd7"
)
DEFAULT_SOURCE_TORCH_VERSION = _STAGE_40.DEFAULT_SOURCE_TORCH_VERSION

DEFAULT_COMPLETE_DATA_FINGERPRINT = _STAGE_40.DEFAULT_COMPLETE_DATA_FINGERPRINT
DEFAULT_COMPLETE_NUM_CHARACTERS = _STAGE_40.DEFAULT_COMPLETE_NUM_CHARACTERS
DEFAULT_COMPLETE_TRAIN_CHARACTERS = _STAGE_40.DEFAULT_COMPLETE_TRAIN_CHARACTERS
DEFAULT_COMPLETE_VAL_CHARACTERS = _STAGE_40.DEFAULT_COMPLETE_VAL_CHARACTERS
DEFAULT_COMPLETE_VOCAB_SIZE = _STAGE_40.DEFAULT_COMPLETE_VOCAB_SIZE
DEFAULT_PARAMETER_COUNT = _STAGE_40.DEFAULT_PARAMETER_COUNT
DEFAULT_TARGETS_PER_UPDATE = _STAGE_40.DEFAULT_TARGETS_PER_UPDATE
DEFAULT_TOTAL_TARGET_CHARACTERS = DEFAULT_TARGET_STEP * DEFAULT_TARGETS_PER_UPDATE
DEFAULT_ADDED_TARGET_CHARACTERS = (
    DEFAULT_TARGET_STEP - DEFAULT_SOURCE_STEP
) * DEFAULT_TARGETS_PER_UPDATE

CheckpointKind = Literal["best", "final"]


@dataclass(frozen=True, slots=True)
class FinalComputeExtensionSchedule:
    """Preserve all prior LR boundaries and expose the step-38k extension."""

    continuation_learning_rate: float = DEFAULT_CONTINUATION_LEARNING_RATE

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.continuation_learning_rate)
            or self.continuation_learning_rate <= 0
        ):
            raise ValueError("continuation learning rate must be finite and positive")
        if self.continuation_learning_rate != DEFAULT_CONTINUATION_LEARNING_RATE:
            raise ValueError("Stage 41 locks the continuation rate at 3e-5")

    def for_update(self, update_index: int) -> float:
        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise ValueError(f"update index must be an integer, got {update_index!r}")
        if update_index < 0:
            raise ValueError(f"update index must be non-negative, got {update_index}")
        if update_index < DEFAULT_SOURCE_STEP:
            return _STAGE_40.LateLearningRateSchedule(
                DEFAULT_CONTINUATION_LEARNING_RATE
            ).for_update(update_index)
        return self.continuation_learning_rate

    def as_metadata(self, *, target_step: int) -> list[dict[str, int | float]]:
        if target_step <= DEFAULT_SOURCE_STEP:
            raise ValueError(
                f"target step must exceed {DEFAULT_SOURCE_STEP:,}, got {target_step:,}"
            )
        history = _STAGE_40.LateLearningRateSchedule(
            DEFAULT_CONTINUATION_LEARNING_RATE
        ).as_metadata(target_step=DEFAULT_SOURCE_STEP)
        return [
            *history,
            {
                "start_update": DEFAULT_SOURCE_STEP,
                "end_update_exclusive": target_step,
                "learning_rate": self.continuation_learning_rate,
            },
        ]


@dataclass(frozen=True, slots=True)
class Stage40Source:
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
    inherited_best_path: Path | None = None
    inherited_best_sha256: str | None = None


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
class Stage41Report:
    source: Stage40Source
    training: TrainingResult
    source_audit: SourceRestoreAudit
    parameter_count: int
    best_train_loss: float
    best_val_loss: float
    best_step: int
    selected_checkpoint_path: Path
    selected_checkpoint_sha256: str
    selected_checkpoint_stage: int
    final_checkpoint_path: Path
    final_checkpoint_sha256: str
    fork_model_sha256: str
    fork_optimizer_sha256: str
    fork_batch_generator_sha256: str
    fork_rng_state_sha256: str
    final_batch_generator_sha256: str
    final_rng_state_sha256: str
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


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    return _STAGE_40.build_model(data, config, n_head, n_layer, device)


def verify_model_masks(model: GPTLanguageModel) -> None:
    _STAGE_40.verify_model_masks(model)


def get_stage_41_batch(
    data: CharacterData,
    split: str,
    config: TrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _STAGE_40.get_stage_40_batch(data, split, config, generator, device)


def evaluate_on_fixed_batches(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    fixed_panel_seed: int,
) -> LossEstimate:
    return _STAGE_40.evaluate_on_fixed_batches(
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
    *,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
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
    )


def validate_protocol(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    *,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> None:
    if config.max_iters <= DEFAULT_SOURCE_STEP:
        raise ValueError(
            f"target_step must be above {DEFAULT_SOURCE_STEP:,}, got "
            f"{config.max_iters:,}"
        )
    if config.learning_rate != DEFAULT_CONTINUATION_LEARNING_RATE:
        raise ValueError("Stage 41 learning rate must remain 3e-5")
    if n_layer != DEFAULT_N_LAYER:
        raise ValueError(f"Stage 41 requires L={DEFAULT_N_LAYER}, got {n_layer}")
    if n_head <= 0 or config.n_embd % n_head:
        raise ValueError("n_embd must be divisible by a positive n_head")
    if config.block_size < DEFAULT_LOCAL_CONTEXT:
        raise ValueError(f"block_size must be at least W{DEFAULT_LOCAL_CONTEXT}")
    if config.batch_size % 2:
        raise ValueError("Stage 41 requires an even batch size")
    if DEFAULT_SOURCE_STEP % config.eval_interval:
        raise ValueError("the 38,000 source boundary must be evaluated")
    if config.max_iters % config.eval_interval:
        raise ValueError("the target step must align with the evaluation interval")
    panel_size = 2 * config.block_size
    for split, values in (("train", data.train_data), ("val", data.val_data)):
        if len(values) <= panel_size:
            raise ValueError(
                f"The {split} split has {len(values)} tokens, but Stage 41 "
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
    inherited_best_checkpoint_path: Path | None = (
        DEFAULT_INHERITED_BEST_CHECKPOINT_PATH
    ),
) -> None:
    paths = {
        "Stage 40 source checkpoint": source_checkpoint_path.resolve(),
        "Stage 41 best checkpoint": best_checkpoint_path.resolve(),
        "Stage 41 final checkpoint": final_checkpoint_path.resolve(),
    }
    if source_data_path is not None:
        paths["Complete Shakespeare corpus"] = source_data_path.resolve()
    if inherited_best_checkpoint_path is not None:
        paths["Stage 40 inherited-best checkpoint"] = (
            inherited_best_checkpoint_path.resolve()
        )
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


def _source_config(config: TrainingConfig) -> TrainingConfig:
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


def _validate_inherited_best(path: Path, source: Stage40Source) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 40 inherited-best checkpoint not found: {path}")
    sha256 = _sha256_file(path)
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="Stage 40 inherited best",
    )
    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    snapshot = _require_mapping(checkpoint.get("snapshot_losses"), name="snapshot")
    if experiment.get("stage") != _STAGE_40.EXPERIMENT_STAGE:
        raise ValueError("inherited best is not a Stage 40 checkpoint")
    if experiment.get("branch_key") != "control":
        raise ValueError("inherited best is not Stage 40 branch A/control")
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("inherited selected checkpoint must be a best artifact")
    if checkpoint.get("step") != source.best_step:
        raise ValueError("inherited best step does not match the source metadata")
    if checkpoint.get("best_step") != source.best_step:
        raise ValueError("inherited best selected step changed")
    if checkpoint.get("best_val_loss") != source.best_val_loss:
        raise ValueError("inherited best validation loss changed")
    if checkpoint.get("best_training_loss") != source.best_train_loss:
        raise ValueError("inherited best training loss changed")
    if snapshot.get("train") != source.best_train_loss:
        raise ValueError("inherited best snapshot train loss changed")
    if snapshot.get("validation") != source.best_val_loss:
        raise ValueError("inherited best snapshot validation loss changed")
    if source.canonical and sha256 != DEFAULT_INHERITED_BEST_CHECKPOINT_SHA256:
        raise ValueError("canonical Stage 40 inherited-best hash mismatch")
    return sha256


def validate_and_load_stage_40_source(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    *,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
    inherited_best_path: Path | None = DEFAULT_INHERITED_BEST_CHECKPOINT_PATH,
) -> Stage40Source:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 40 source checkpoint not found: {path}")
    before_hash = _sha256_file(path)
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="root",
    )
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported Stage 40 source checkpoint version")
    if checkpoint.get("checkpoint_kind") != "final":
        raise ValueError("Stage 41 requires Stage 40 A's unconditional final checkpoint")
    step = _require_int(checkpoint.get("step"), name="step")
    best_step = _require_int(checkpoint.get("best_step"), name="best_step")
    if step != DEFAULT_SOURCE_STEP:
        raise ValueError(f"Stage 41 requires the step-{DEFAULT_SOURCE_STEP:,} endpoint")
    if best_step > step:
        raise ValueError("Stage 40 selected step lies after its final state")

    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    if experiment.get("stage") != _STAGE_40.EXPERIMENT_STAGE:
        raise ValueError("source checkpoint is not a Stage 40 artifact")
    if experiment.get("branch_key") != "control":
        raise ValueError("Stage 41 must continue Stage 40 branch A/control")
    for key, expected in (
        ("optimizer_state_preserved", True),
        ("optimizer_restarted", False),
        ("rng_state_preserved", True),
        ("architecture_frozen", True),
    ):
        if experiment.get(key) != expected:
            raise ValueError(f"Stage 40 source experiment field {key!r} changed")

    runtime = _require_mapping(checkpoint.get("runtime"), name="runtime")
    runtime_type = runtime.get("device_type")
    runtime_name = runtime.get("device_name")
    saved_torch_version = runtime.get("torch_version")
    if not all(isinstance(value, str) for value in (runtime_type, runtime_name, saved_torch_version)):
        raise ValueError("Stage 40 source runtime metadata is incomplete")
    current_type, current_name = _runtime_identity(device)
    if runtime_type != current_type or runtime_name != current_name:
        raise ValueError(
            "exact Stage 40 continuation requires the recorded device/backend: "
            f"saved {runtime_type}/{runtime_name!r}, current "
            f"{current_type}/{current_name!r}"
        )
    if saved_torch_version != str(torch.__version__):
        raise ValueError(
            "exact Stage 40 continuation requires the recorded torch version: "
            f"saved {saved_torch_version!r}, current {str(torch.__version__)!r}"
        )

    architecture = _require_mapping(checkpoint.get("architecture"), name="architecture")
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "residual_dropout": DEFAULT_DROPOUT,
        "topology": "LLLLGG",
        "vocabulary_size": data.vocabulary.size,
        "parameter_count": DEFAULT_PARAMETER_COUNT,
        "attention_mask": "W23,W23,W23,W23,Full,Full",
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"Stage 40 source architecture field {key!r} mismatch")
    observed_contexts = architecture.get("layer_contexts")
    if observed_contexts is not None and tuple(observed_contexts) != DEFAULT_LAYER_CONTEXTS:
        raise ValueError("Stage 40 source layer contexts changed")
    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("Stage 40 source corpus fingerprint mismatch")

    training = _require_mapping(checkpoint.get("training_config"), name="training_config")
    expected_training = {
        "batch_size": config.batch_size,
        "learning_rate": DEFAULT_CONTINUATION_LEARNING_RATE,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": DEFAULT_SOURCE_STEP,
        "target_step": DEFAULT_SOURCE_STEP,
        "branch_learning_rate": DEFAULT_CONTINUATION_LEARNING_RATE,
        "training_batch_seed": training_batch_seed,
        "fixed_panel_seed": fixed_panel_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Stage 40 source training field {key!r} mismatch")

    optimizer_state = _validate_adamw_state(
        checkpoint.get("optimizer_state_dict"),
        expected_learning_rate=DEFAULT_CONTINUATION_LEARNING_RATE,
    )
    _validate_optimizer_steps(optimizer_state, expected_step=step)
    _validate_optimizer_hyperparameters(optimizer_state)
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 40 source training-generator state must be a tensor")
    expected_generator = _expected_training_generator_state(
        data,
        _source_config(config),
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator):
        raise ValueError("Stage 40 source batch-generator state is inconsistent")
    _validate_rng_state(checkpoint.get("rng_state"), runtime_device_type=runtime_type)

    best_train = _require_finite_float(
        checkpoint.get("best_training_loss"), name="best_training_loss"
    )
    best_val = _require_finite_float(checkpoint.get("best_val_loss"), name="best_val_loss")
    snapshot = _require_mapping(checkpoint.get("snapshot_losses"), name="snapshot_losses")
    snapshot_train = _require_finite_float(snapshot.get("train"), name="snapshot train loss")
    snapshot_val = _require_finite_float(snapshot.get("validation"), name="snapshot val loss")

    model_state = _require_mapping(checkpoint.get("model_state_dict"), name="model_state_dict")
    with torch.random.fork_rng(devices=[]):
        model = build_model(data, config, n_head, n_layer)
    model.load_state_dict(model_state, strict=True)
    verify_model_masks(model)

    canonical = is_canonical_protocol(
        data,
        config,
        n_head,
        n_layer,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    if canonical:
        checks = (
            (before_hash, DEFAULT_SOURCE_CHECKPOINT_SHA256, "checkpoint"),
            (tensor_mapping_sha256(model_state), DEFAULT_SOURCE_MODEL_SHA256, "model"),
            (structured_state_sha256(optimizer_state), DEFAULT_SOURCE_OPTIMIZER_SHA256, "optimizer"),
            (tensor_sha256(generator_state), DEFAULT_SOURCE_GENERATOR_SHA256, "generator"),
            (structured_state_sha256(checkpoint.get("rng_state")), DEFAULT_SOURCE_RNG_SHA256, "Torch RNG"),
        )
        for observed, expected, name in checks:
            if observed != expected:
                raise ValueError(f"canonical Stage 40 source {name} hash mismatch")
        if best_step != DEFAULT_SOURCE_BEST_STEP:
            raise ValueError("canonical Stage 40 source best step changed")
        if best_train != DEFAULT_SOURCE_BEST_TRAIN_LOSS or best_val != DEFAULT_SOURCE_BEST_VAL_LOSS:
            raise ValueError("canonical Stage 40 source selected losses changed")
        if snapshot_train != DEFAULT_SOURCE_SNAPSHOT_TRAIN_LOSS or snapshot_val != DEFAULT_SOURCE_SNAPSHOT_VAL_LOSS:
            raise ValueError("canonical Stage 40 source terminal losses changed")
        if saved_torch_version != DEFAULT_SOURCE_TORCH_VERSION:
            raise ValueError("canonical Stage 40 source torch version changed")

    source = Stage40Source(
        path=path.resolve(),
        sha256=before_hash,
        checkpoint=checkpoint,
        step=step,
        best_step=best_step,
        best_train_loss=best_train,
        best_val_loss=best_val,
        snapshot_train_loss=snapshot_train,
        snapshot_val_loss=snapshot_val,
        runtime_device_type=runtime_type,
        runtime_device_name=runtime_name,
        torch_version=saved_torch_version,
        canonical=canonical,
        inherited_best_path=(None if inherited_best_path is None else inherited_best_path.resolve()),
    )
    inherited_hash = None
    if inherited_best_path is not None:
        inherited_hash = _validate_inherited_best(inherited_best_path, source)
        source = Stage40Source(
            **{
                field: getattr(source, field)
                for field in source.__dataclass_fields__
                if field != "inherited_best_sha256"
            },
            inherited_best_sha256=inherited_hash,
        )
    if _sha256_file(path) != before_hash:
        raise RuntimeError("Stage 40 source changed during validation")
    return source


def restore_stage_40_source(
    source: Stage40Source,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    device: torch.device,
) -> None:
    checkpoint = source.checkpoint
    model_state = _require_mapping(checkpoint.get("model_state_dict"), name="model_state_dict")
    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"), name="optimizer_state_dict"
    )
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 40 source training-generator state is missing")
    runtime_type = getattr(source, "runtime_device_type", device.type)
    rng_state = _validate_rng_state(
        checkpoint.get("rng_state"), runtime_device_type=runtime_type
    )
    model.load_state_dict(model_state, strict=True)
    verify_model_masks(model)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    training_generator.set_state(generator_state.cpu())
    # Restore last.  Model construction and validation must not consume the
    # first dropout draw of the continued Stage 40 stream.
    _restore_rng_state(rng_state, device)


@dataclass(slots=True)
class Stage41ValidationCheckpoint(BestValidationCheckpoint):
    schedule: FinalComputeExtensionSchedule | None = None
    target_step: int = 0
    source: Stage40Source | None = None
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
    fork_model_sha256: str = ""
    fork_optimizer_sha256: str = ""
    fork_batch_generator_sha256: str = ""
    fork_rng_state_sha256: str = ""
    best_train_loss: float | None = None
    source_audit_verified: bool = False
    best_checkpoint_written: bool = False

    def _decorate_payload(
        self,
        payload: dict[str, Any],
        *,
        step: int,
        checkpoint_kind: CheckpointKind,
        snapshot_losses: LossEstimate,
    ) -> dict[str, Any]:
        if self.schedule is None or self.source is None or self.best_step is None or self.best_train_loss is None:
            raise RuntimeError("incomplete Stage 41 checkpoint metadata")
        optimizer_state = _require_mapping(payload.get("optimizer_state_dict"), name="optimizer_state_dict")
        if _optimizer_rates(optimizer_state) != {DEFAULT_CONTINUATION_LEARNING_RATE}:
            raise RuntimeError("Stage 41 optimizer LR changed")
        payload["checkpoint_kind"] = checkpoint_kind
        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "head_size": self.config.n_embd // self.n_head,
                "feed_forward_width": 4 * self.config.n_embd,
                "residual_dropout": DEFAULT_DROPOUT,
                "layer_contexts": DEFAULT_LAYER_CONTEXTS,
                "topology": "LLLLGG",
                "vocabulary_size": self.vocabulary_size,
                "parameter_count": self.parameter_count,
                "attention_mask": "W23,W23,W23,W23,Full,Full",
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
                "continuation_learning_rate": DEFAULT_CONTINUATION_LEARNING_RATE,
                "learning_rate_schedule": self.schedule.as_metadata(target_step=self.target_step),
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
                "total_target_presentations": self.targets_per_update * self.target_step,
                "added_target_presentations_after_stage_40": self.targets_per_update * (self.target_step - self.source.step),
            }
        )
        payload["source_checkpoint"] = {
            "stage": _STAGE_40.EXPERIMENT_STAGE,
            "path_name": self.source.path.name,
            "sha256": self.source.sha256,
            "checkpoint_kind": "final",
            "branch_key": "control",
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
        payload["fork_state"] = {
            "captured_after_exact_restore": True,
            "model_sha256": self.fork_model_sha256,
            "optimizer_sha256": self.fork_optimizer_sha256,
            "training_generator_sha256": self.fork_batch_generator_sha256,
            "torch_rng_sha256": self.fork_rng_state_sha256,
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
            "kind": "restored_stage_40_a_terminal_complete_shakespeare_state",
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
            "same_model_as_stage_40_a": True,
            "same_corpus_as_stage_40": True,
            "same_split_as_stage_40": True,
            "same_sampler_as_stage_40": True,
            "same_fixed_evaluation_panel_as_stage_40": True,
            "same_learning_rate_as_stage_40_a": True,
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
            "matches_stage_40_source": (
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
        payload["selection_provenance"] = (
            "stage_41_extension" if self.best_checkpoint_written else "stage_40_inherited"
        )
        return payload

    def _payload(self, step: int) -> dict[str, Any]:
        if self.best_train_loss is None:
            raise RuntimeError("best training loss is unavailable")
        return self._decorate_payload(
            super(Stage41ValidationCheckpoint, self)._payload(step),
            step=step,
            checkpoint_kind="best",
            snapshot_losses=LossEstimate(train=self.best_train_loss, val=self.best_val_loss),
        )

    def save(self, step: int) -> None:
        self.best_checkpoint_written = True
        super().save(step)

    def save_final(self, path: Path, record: EvaluationRecord) -> None:
        payload = self._decorate_payload(
            super(Stage41ValidationCheckpoint, self)._payload(record.step),
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
    schedule: FinalComputeExtensionSchedule,
    *,
    start_step: int,
    fixed_panel_seed: int,
    on_evaluation: Callable[[EvaluationRecord], None],
) -> TrainingResult:
    if start_step != DEFAULT_SOURCE_STEP:
        raise ValueError(
            f"Stage 41 must start at step {DEFAULT_SOURCE_STEP:,}, got {start_step:,}"
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
            raise RuntimeError(f"Stage 41 update {update_index} would change LR")
        _set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_stage_41_batch(
            data, "train", config, training_generator, device
        )
        logits, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        del logits, loss, inputs, targets

        completed_step = update_index + 1
        if completed_step % config.eval_interval == 0 or completed_step == config.max_iters:
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
    best: Stage41ValidationCheckpoint,
    source: Stage40Source,
    *,
    final_checkpoint_path: Path,
) -> tuple[Callable[[EvaluationRecord], None], list[SourceRestoreAudit]]:
    source_audits: list[SourceRestoreAudit] = []

    def record_evaluation(record: EvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr {DEFAULT_CONTINUATION_LEARNING_RATE:g} | "
            f"train {record.losses.train:.4f} | val {record.losses.val:.4f} | "
            f"gap {record.losses.val - record.losses.train:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )
        if record.step == source.step:
            checkpoint_device = getattr(best, "device", None)
            source_type = getattr(source, "runtime_device_type", "cpu")
            source_name = getattr(source, "runtime_device_name", source_type)
            if isinstance(checkpoint_device, torch.device):
                runtime_type, runtime_name = _runtime_identity(checkpoint_device)
            else:
                # Lightweight callback tests need no device-owned checkpoint;
                # the production checkpoint always supplies one.
                runtime_type, runtime_name = source_type, source_name
            audit = SourceRestoreAudit(
                record=record,
                train_matches_source=_reported_value_matches(
                    record.losses.train, source.snapshot_train_loss
                ),
                val_matches_source=_reported_value_matches(
                    record.losses.val, source.snapshot_val_loss
                ),
                runtime_matched=(runtime_type == source_type and runtime_name == source_name),
            )
            if not audit.verified:
                raise RuntimeError(
                    "the restored step-38,000 fixed-panel result does not match "
                    f"the Stage 40 source: expected {source.snapshot_train_loss:.4f} / "
                    f"{source.snapshot_val_loss:.4f}, got {record.losses.train:.4f} / "
                    f"{record.losses.val:.4f}"
                )
            best.source_audit_verified = True
            if best.best_step == record.step:
                best.save(record.step)
            source_audits.append(audit)
        if record.step == best.target_step:
            if not best.source_audit_verified:
                raise RuntimeError("training reached its target without source audit")
            best.save_final(final_checkpoint_path, record)

    return record_evaluation, source_audits


def validate_stage_41_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: FinalComputeExtensionSchedule,
    source: Stage40Source,
    *,
    expected_kind: CheckpointKind,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> Mapping[str, object]:
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True), name="root"
    )
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported Stage 41 checkpoint version")
    if checkpoint.get("checkpoint_kind") != expected_kind:
        raise ValueError(f"expected a {expected_kind!r} Stage 41 checkpoint")
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 41 checkpoint must not restart AdamW")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 41 optimizer provenance must remain known")
    step = _require_int(checkpoint.get("step"), name="step")
    best_step = _require_int(checkpoint.get("best_step"), name="best_step")
    if expected_kind == "best" and step != best_step:
        raise ValueError("Stage 41 best checkpoint must store its selected step")
    if expected_kind == "final" and step != config.max_iters:
        raise ValueError("Stage 41 final checkpoint is not at the target step")
    if not (source.best_step <= best_step <= step):
        raise ValueError("Stage 41 best step is outside the full trajectory")
    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    if experiment.get("stage") != EXPERIMENT_STAGE:
        raise ValueError("checkpoint is not a Stage 41 artifact")
    for key, expected in (
        ("same_model_as_stage_40_a", True),
        ("same_corpus_as_stage_40", True),
        ("same_sampler_as_stage_40", True),
        ("same_learning_rate_as_stage_40_a", True),
        ("optimizer_state_preserved", True),
        ("optimizer_restarted", False),
        ("rng_state_preserved", True),
        ("architecture_frozen", True),
        ("trained_branches", 1),
    ):
        if experiment.get(key) != expected:
            raise ValueError(f"Stage 41 experiment field {key!r} changed")
    source_metadata = _require_mapping(checkpoint.get("source_checkpoint"), name="source_checkpoint")
    for key, expected in (
        ("stage", _STAGE_40.EXPERIMENT_STAGE),
        ("sha256", source.sha256),
        ("checkpoint_kind", "final"),
        ("branch_key", "control"),
        ("step", source.step),
        ("model_state_restored", True),
        ("optimizer_state_restored", True),
        ("training_generator_restored", True),
        ("cpu_rng_restored", True),
        ("optimizer_restart", False),
        ("restore_audit_verified_at_snapshot", True),
    ):
        if source_metadata.get(key) != expected:
            raise ValueError(f"Stage 41 source field {key!r} mismatch")
    architecture = _require_mapping(checkpoint.get("architecture"), name="architecture")
    for key, expected in (
        ("block_size", config.block_size),
        ("n_embd", config.n_embd),
        ("n_head", n_head),
        ("n_layer", n_layer),
        ("head_size", config.n_embd // n_head),
        ("feed_forward_width", 4 * config.n_embd),
        ("residual_dropout", DEFAULT_DROPOUT),
        ("topology", "LLLLGG"),
        ("vocabulary_size", data.vocabulary.size),
        ("attention_mask", "W23,W23,W23,W23,Full,Full"),
    ):
        if architecture.get(key) != expected:
            raise ValueError(f"Stage 41 architecture field {key!r} mismatch")
    training = _require_mapping(checkpoint.get("training_config"), name="training_config")
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
        "training_batch_seed": training_batch_seed,
        "fixed_panel_seed": fixed_panel_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Stage 41 training field {key!r} mismatch")
    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("Stage 41 checkpoint corpus fingerprint mismatch")
    runtime = _require_mapping(checkpoint.get("runtime"), name="runtime")
    if runtime.get("device_type") != source.runtime_device_type or runtime.get("device_name") != source.runtime_device_name:
        raise ValueError("Stage 41 runtime does not match Stage 40")
    if runtime.get("torch_version") != source.torch_version:
        raise ValueError("Stage 41 torch version changed")
    if runtime.get("matches_stage_40_source") is not True:
        raise ValueError("Stage 41 runtime match was not recorded")
    _validate_rng_state(checkpoint.get("rng_state"), runtime_device_type=source.runtime_device_type)
    optimizer_state = _validate_adamw_state(
        checkpoint.get("optimizer_state_dict"),
        expected_learning_rate=DEFAULT_CONTINUATION_LEARNING_RATE,
    )
    _validate_optimizer_steps(optimizer_state, expected_step=step)
    _validate_optimizer_hyperparameters(optimizer_state)
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 41 training-generator state must be a tensor")
    expected_generator = _expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator):
        raise ValueError("Stage 41 batch-generator state is inconsistent")
    model_state = _require_mapping(checkpoint.get("model_state_dict"), name="model_state_dict")
    with torch.random.fork_rng(devices=[]):
        model = build_model(data, config, n_head, n_layer)
    model.load_state_dict(model_state, strict=True)
    verify_model_masks(model)
    return checkpoint


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: FinalComputeExtensionSchedule,
    source_checkpoint_path: Path,
    best_checkpoint_path: Path,
    final_checkpoint_path: Path,
    *,
    source_data_path: Path | None,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> Stage41Report:
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    validate_protocol(
        data,
        config,
        n_head,
        n_layer,
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
    source = validate_and_load_stage_40_source(
        source_checkpoint_path,
        data,
        config,
        n_head,
        n_layer,
        device,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    source_hash_before = _sha256_file(source.path)

    clear_accelerator_cache(device)
    seed_everything(config.seed)
    model = build_model(data, config, n_head, n_layer, device)
    verify_model_masks(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DEFAULT_CONTINUATION_LEARNING_RATE
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    restore_stage_40_source(source, model, optimizer, training_generator, device)

    fork_model_sha256 = tensor_mapping_sha256(model.state_dict())
    fork_optimizer_sha256 = structured_state_sha256(optimizer.state_dict())
    fork_batch_generator_sha256 = tensor_sha256(training_generator.get_state())
    fork_rng_state_sha256 = structured_state_sha256(capture_rng_state(device))
    if source.canonical:
        for observed, expected, name in (
            (fork_model_sha256, DEFAULT_SOURCE_MODEL_SHA256, "model"),
            (fork_optimizer_sha256, DEFAULT_SOURCE_OPTIMIZER_SHA256, "optimizer"),
            (fork_batch_generator_sha256, DEFAULT_SOURCE_GENERATOR_SHA256, "generator"),
            (fork_rng_state_sha256, DEFAULT_SOURCE_RNG_SHA256, "Torch RNG"),
        ):
            if observed != expected:
                raise RuntimeError(f"Stage 41 restored {name} lineage changed")
    if _optimizer_rates(optimizer.state_dict()) != {DEFAULT_CONTINUATION_LEARNING_RATE}:
        raise RuntimeError("Stage 41 source optimizer LR changed")

    corpus = _require_mapping(source.checkpoint.get("corpus"), name="source corpus")
    initialization = _require_mapping(
        source.checkpoint.get("initialization"), name="source initialization"
    )
    initial_state_sha256 = initialization.get("fresh_initial_state_sha256")
    if not isinstance(initial_state_sha256, str):
        raise ValueError("Stage 40 initialization hash is missing")
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best = Stage41ValidationCheckpoint(
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
        source_filename=corpus.get("source_filename"),
        source_bytes=corpus.get("source_bytes"),
        source_sha256=corpus.get("source_sha256"),
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
        fork_model_sha256=fork_model_sha256,
        fork_optimizer_sha256=fork_optimizer_sha256,
        fork_batch_generator_sha256=fork_batch_generator_sha256,
        fork_rng_state_sha256=fork_rng_state_sha256,
        best_train_loss=source.best_train_loss,
    )
    callback, source_audits = make_evaluation_callback(
        best, source, final_checkpoint_path=final_checkpoint_path
    )

    print("\nStage 41 continuation")
    print(f"  source:                 {source.path}")
    print(f"  source SHA-256:         {source.sha256}")
    print(f"  restore step:           {source.step:,}")
    print(f"  target step:            {config.max_iters:,}")
    print("  optimizer lineage:      restored exactly; no restart or LR change")
    print("  batch/dropout streams:  restored exactly from Stage 40 A final")
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
        raise RuntimeError("Stage 41 did not perform exactly one source audit")
    assert best.best_step is not None and best.best_train_loss is not None

    final_batch_generator_sha256 = tensor_sha256(training_generator.get_state())
    final_rng_state_sha256 = structured_state_sha256(capture_rng_state(device))
    final_model_sha256 = tensor_mapping_sha256(model.state_dict())
    final_optimizer_sha256 = structured_state_sha256(optimizer.state_dict())
    validate_stage_41_checkpoint(
        final_checkpoint_path,
        data,
        config,
        n_head,
        n_layer,
        schedule,
        source,
        expected_kind="final",
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    final_payload = _require_mapping(
        torch.load(final_checkpoint_path, map_location="cpu", weights_only=True),
        name="final Stage 41 checkpoint",
    )
    final_payload_model = _require_mapping(
        final_payload.get("model_state_dict"), name="final model state"
    )
    final_payload_optimizer = _require_mapping(
        final_payload.get("optimizer_state_dict"), name="final optimizer state"
    )
    final_payload_generator = final_payload.get("training_generator_state")
    if not isinstance(final_payload_generator, torch.Tensor):
        raise RuntimeError("terminal checkpoint omitted the batch-generator state")
    terminal_hashes = (
        (tensor_mapping_sha256(final_payload_model), final_model_sha256, "model"),
        (
            structured_state_sha256(final_payload_optimizer),
            final_optimizer_sha256,
            "optimizer",
        ),
        (
            tensor_sha256(final_payload_generator),
            final_batch_generator_sha256,
            "batch generator",
        ),
        (
            structured_state_sha256(final_payload.get("rng_state")),
            final_rng_state_sha256,
            "Torch RNG",
        ),
    )
    for saved_hash, live_hash, name in terminal_hashes:
        if saved_hash != live_hash:
            raise RuntimeError(
                f"terminal checkpoint captured the wrong live {name} state"
            )
    if best.best_checkpoint_written:
        validate_stage_41_checkpoint(
            best_checkpoint_path,
            data,
            config,
            n_head,
            n_layer,
            schedule,
            source,
            expected_kind="best",
            training_batch_seed=training_batch_seed,
            fixed_panel_seed=fixed_panel_seed,
            training_rng_seed=training_rng_seed,
        )
        selected_path = best_checkpoint_path
        selected_stage = EXPERIMENT_STAGE
    else:
        if source.inherited_best_path is None or source.inherited_best_sha256 is None:
            raise RuntimeError("no Stage 41 improvement and no inherited best artifact")
        selected_path = source.inherited_best_path
        selected_stage = _STAGE_40.EXPERIMENT_STAGE
    selected_sha256 = checkpoint_sha256(selected_path)
    selected = _require_mapping(
        torch.load(selected_path, map_location="cpu", weights_only=True),
        name="selected checkpoint",
    )
    selected_state = _require_mapping(selected.get("model_state_dict"), name="selected model state")
    model.load_state_dict(selected_state, strict=True)
    verify_model_masks(model)
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=DEFAULT_SAMPLE_SEED,
    )
    if _sha256_file(source.path) != source_hash_before:
        raise RuntimeError("Stage 40 source changed during Stage 41")

    report = Stage41Report(
        source=source,
        training=result,
        source_audit=source_audits[0],
        parameter_count=parameter_count,
        best_train_loss=best.best_train_loss,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        selected_checkpoint_path=selected_path,
        selected_checkpoint_sha256=selected_sha256,
        selected_checkpoint_stage=selected_stage,
        final_checkpoint_path=final_checkpoint_path,
        final_checkpoint_sha256=checkpoint_sha256(final_checkpoint_path),
        fork_model_sha256=fork_model_sha256,
        fork_optimizer_sha256=fork_optimizer_sha256,
        fork_batch_generator_sha256=fork_batch_generator_sha256,
        fork_rng_state_sha256=fork_rng_state_sha256,
        final_batch_generator_sha256=final_batch_generator_sha256,
        final_rng_state_sha256=final_rng_state_sha256,
        sample=sample,
    )
    del callback, best, optimizer, model
    clear_accelerator_cache(device)
    return report


def print_final_summary(report: Stage41Report) -> None:
    print("\nStage 41 summary")
    print(
        f"  source:   train {report.source.snapshot_train_loss:.4f} / val "
        f"{report.source.snapshot_val_loss:.4f} @ {report.source.step:,}"
    )
    print(
        f"  selected: train {report.best_train_loss:.4f} / val "
        f"{report.best_val_loss:.4f} @ {report.best_step:,}"
    )
    print(
        f"  final:    train {report.training.final.train:.4f} / val "
        f"{report.training.final.val:.4f} @ {report.final_step:,}"
    )
    print(f"  final reversal from selected best: {report.validation_reversal:+.6f}")
    print(
        f"  selected checkpoint (Stage {report.selected_checkpoint_stage}): "
        f"{report.selected_checkpoint_path}"
    )
    print(f"  selected SHA-256: {report.selected_checkpoint_sha256}")
    print(f"  final checkpoint: {report.final_checkpoint_path}")
    print(f"  final SHA-256:    {report.final_checkpoint_sha256}")
    print("  source audit:     exact fixed-panel endpoint reproduced")
    print("  optimizer:        full AdamW lineage preserved")
    print("  RNG streams:      batch and Torch/dropout lineage preserved")
    print("\nSelected-best generated text")
    print(report.sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 41: restore Stage 40 A-final at step 38,000 and continue "
            "the exact 3e-5 optimizer, batch, and RNG lineage to step 44,000."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_COMPLETE_DATA_PATH)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT_PATH)
    parser.add_argument("--target-step", type=int, default=DEFAULT_TARGET_STEP)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--best-checkpoint-path", type=Path, default=DEFAULT_BEST_CHECKPOINT_PATH)
    parser.add_argument("--final-checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "xpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


def _default_final_checkpoint_path(target_step: int) -> Path:
    if target_step == DEFAULT_TARGET_STEP:
        return DEFAULT_FINAL_CHECKPOINT_PATH
    return CHECKPOINT_DIRECTORY / (
        f"stage_41_complete_shakespeare_step_{target_step}_final_checkpoint.pt"
    )


def main() -> None:
    args = parse_args()
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
    data = CharacterData.from_file(args.data, train_fraction=DEFAULT_TRAIN_FRACTION)
    device = resolve_device(args.device)
    schedule = FinalComputeExtensionSchedule()
    final_path = (
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
    print(f"Protocol: exact restore at {DEFAULT_SOURCE_STEP:,}, target {config.max_iters:,}")
    print(f"Continuation LR: {DEFAULT_CONTINUATION_LEARNING_RATE:g}")

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
        final_path,
        source_data_path=args.data,
        training_batch_seed=DEFAULT_TRAINING_BATCH_SEED,
        fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED,
        training_rng_seed=DEFAULT_TRAINING_RNG_SEED,
    )
    print_final_summary(report)


if __name__ == "__main__":
    main()
