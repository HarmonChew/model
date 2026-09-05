from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch

from config import TrainingConfig
from data_utils import CharacterData
from train import resolve_device
from training import EvaluationRecord, LossEstimate, TrainingResult


# Stage 40 independently restores Stage 39's exact terminal state for each
# branch. The model, AdamW moments, batch generator, and Torch dropout RNG are
# common at the fork; only the post-fork optimizer learning rate is changed.
_STAGE_39_PATH = Path(__file__).with_name("39_extended_training.py")
_STAGE_39_MODULE_NAME = "stage_39_extended_training_for_stage_40"
_STAGE_39_SPEC = importlib.util.spec_from_file_location(
    _STAGE_39_MODULE_NAME,
    _STAGE_39_PATH,
)
assert _STAGE_39_SPEC is not None and _STAGE_39_SPEC.loader is not None
_STAGE_39 = importlib.util.module_from_spec(_STAGE_39_SPEC)
sys.modules[_STAGE_39_MODULE_NAME] = _STAGE_39
_STAGE_39_SPEC.loader.exec_module(_STAGE_39)


GPTLanguageModel = _STAGE_39.GPTLanguageModel
ExtendedLearningRateSchedule = _STAGE_39.ExtendedLearningRateSchedule
BestValidationCheckpoint = _STAGE_39.BestValidationCheckpoint
checkpoint_sha256 = _STAGE_39.checkpoint_sha256
clear_accelerator_cache = _STAGE_39.clear_accelerator_cache
fingerprint_data = _STAGE_39.fingerprint_data
generate_from_final_model = _STAGE_39.generate_from_final_model
seed_everything = _STAGE_39.seed_everything
tensor_mapping_sha256 = _STAGE_39.tensor_mapping_sha256
tensor_sha256 = _STAGE_39.tensor_sha256


EXPERIMENT_STAGE = 40
EXPERIMENT_TITLE = "Complete Shakespeare late learning-rate refinement"
PROTOCOL_NAME = "paired_stage_39_terminal_full_state_learning_rate_fork"
COMPARISON_VARIABLE = "post_32000_learning_rate"

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
DEFAULT_COMPLETE_DATA_PATH = _STAGE_39.DEFAULT_COMPLETE_DATA_PATH
CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_STAGE_38_CHECKPOINT_PATH = _STAGE_39.DEFAULT_SOURCE_CHECKPOINT_PATH
DEFAULT_SOURCE_CHECKPOINT_PATH = _STAGE_39.DEFAULT_FINAL_CHECKPOINT_PATH
DEFAULT_CONTROL_BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_40_control_3e_5_best_checkpoint.pt"
)
DEFAULT_CONTROL_FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_40_control_3e_5_step_38000_final_checkpoint.pt"
)
DEFAULT_TREATMENT_BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_40_treatment_1e_5_best_checkpoint.pt"
)
DEFAULT_TREATMENT_FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_40_treatment_1e_5_step_38000_final_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = _STAGE_39.DEFAULT_BATCH_SIZE
DEFAULT_BLOCK_SIZE = _STAGE_39.DEFAULT_BLOCK_SIZE
DEFAULT_LOCAL_CONTEXT = _STAGE_39.DEFAULT_LOCAL_CONTEXT
DEFAULT_LAYER_CONTEXTS = _STAGE_39.DEFAULT_LAYER_CONTEXTS
DEFAULT_GLOBAL_LAYER_INDICES = _STAGE_39.DEFAULT_GLOBAL_LAYER_INDICES
DEFAULT_GLOBAL_LAYER_NUMBERS = _STAGE_39.DEFAULT_GLOBAL_LAYER_NUMBERS
DEFAULT_N_EMBD = _STAGE_39.DEFAULT_N_EMBD
DEFAULT_N_HEAD = _STAGE_39.DEFAULT_N_HEAD
DEFAULT_HEAD_SIZE = _STAGE_39.DEFAULT_HEAD_SIZE
DEFAULT_FEED_FORWARD_WIDTH = _STAGE_39.DEFAULT_FEED_FORWARD_WIDTH
DEFAULT_N_LAYER = _STAGE_39.DEFAULT_N_LAYER
DEFAULT_DROPOUT = _STAGE_39.DEFAULT_DROPOUT

DEFAULT_SOURCE_LEARNING_RATE = _STAGE_39.DEFAULT_CONTINUATION_LEARNING_RATE
DEFAULT_CONTROL_LEARNING_RATE = 3e-5
DEFAULT_TREATMENT_LEARNING_RATE = 1e-5
DEFAULT_SOURCE_STEP = _STAGE_39.DEFAULT_TARGET_STEP
DEFAULT_TARGET_STEP = 38_000
DEFAULT_EVAL_INTERVAL = _STAGE_39.DEFAULT_EVAL_INTERVAL
DEFAULT_EVAL_ITERS = _STAGE_39.DEFAULT_EVAL_ITERS

DEFAULT_SEED = _STAGE_39.DEFAULT_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_39.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_FIXED_PANEL_SEED = _STAGE_39.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_39.DEFAULT_TRAINING_RNG_SEED
DEFAULT_SAMPLE_SEED = _STAGE_39.DEFAULT_SAMPLE_SEED
DEFAULT_FRESH_VALIDATION_SEED = 1_354
DEFAULT_FRESH_VALIDATION_ITERS = 500
DEFAULT_TRAIN_FRACTION = _STAGE_39.DEFAULT_TRAIN_FRACTION

DEFAULT_SOURCE_BEST_TRAIN_LOSS = 1.2065401589870453
DEFAULT_SOURCE_BEST_VAL_LOSS = 1.4235185253620148
DEFAULT_SOURCE_CHECKPOINT_SHA256 = (
    "d34a368bd8a91622ff500c3f64d12d1bb211e782cb1e6909d05d8b8ba2c83cc4"
)
DEFAULT_SOURCE_GENERATOR_SHA256 = (
    "40ba8d83a5cd1d8443d44e0a49d1d4dea931486b93b31f77cb535a3a3a1ce508"
)
DEFAULT_SOURCE_TORCH_VERSION = _STAGE_39.DEFAULT_SOURCE_TORCH_VERSION

DEFAULT_COMPLETE_DATA_FINGERPRINT = _STAGE_39.DEFAULT_COMPLETE_DATA_FINGERPRINT
DEFAULT_COMPLETE_NUM_CHARACTERS = _STAGE_39.DEFAULT_COMPLETE_NUM_CHARACTERS
DEFAULT_COMPLETE_TRAIN_CHARACTERS = _STAGE_39.DEFAULT_COMPLETE_TRAIN_CHARACTERS
DEFAULT_COMPLETE_VAL_CHARACTERS = _STAGE_39.DEFAULT_COMPLETE_VAL_CHARACTERS
DEFAULT_COMPLETE_VOCAB_SIZE = _STAGE_39.DEFAULT_COMPLETE_VOCAB_SIZE
DEFAULT_PARAMETER_COUNT = _STAGE_39.DEFAULT_PARAMETER_COUNT
DEFAULT_INITIAL_STATE_SHA256 = _STAGE_39.DEFAULT_INITIAL_STATE_SHA256
DEFAULT_TARGETS_PER_UPDATE = _STAGE_39.DEFAULT_TARGETS_PER_UPDATE
DEFAULT_TOTAL_TARGET_CHARACTERS = DEFAULT_TARGET_STEP * DEFAULT_TARGETS_PER_UPDATE
DEFAULT_ADDED_TARGET_CHARACTERS_PER_BRANCH = (
    DEFAULT_TARGET_STEP - DEFAULT_SOURCE_STEP
) * DEFAULT_TARGETS_PER_UPDATE
DEFAULT_EXPERIMENT_ADDED_TARGET_CHARACTERS = (
    2 * DEFAULT_ADDED_TARGET_CHARACTERS_PER_BRANCH
)

CheckpointKind = Literal["best", "final"]


@dataclass(frozen=True, slots=True)
class LateLearningRateSchedule:
    """Preserve the complete history and expose an explicit step-32k fork."""

    branch_learning_rate: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.branch_learning_rate)
            or self.branch_learning_rate <= 0
        ):
            raise ValueError(
                "branch learning rate must be finite and positive, got "
                f"{self.branch_learning_rate}"
            )

    def for_update(self, update_index: int) -> float:
        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise ValueError(f"update index must be an integer, got {update_index!r}")
        if update_index < 0:
            raise ValueError(f"update index must be non-negative, got {update_index}")
        if update_index < DEFAULT_SOURCE_STEP:
            return ExtendedLearningRateSchedule().for_update(update_index)
        return self.branch_learning_rate

    def as_metadata(self, *, target_step: int) -> list[dict[str, int | float]]:
        if target_step <= DEFAULT_SOURCE_STEP:
            raise ValueError(
                f"target step must exceed {DEFAULT_SOURCE_STEP:,}, got "
                f"{target_step:,}"
            )
        history = ExtendedLearningRateSchedule().as_metadata(
            target_step=DEFAULT_SOURCE_STEP
        )
        return [
            *history,
            {
                "start_update": DEFAULT_SOURCE_STEP,
                "end_update_exclusive": target_step,
                "learning_rate": self.branch_learning_rate,
            },
        ]


@dataclass(frozen=True, slots=True)
class BranchSpec:
    key: str
    name: str
    learning_rate: float
    best_checkpoint_path: Path
    final_checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.key or not self.name:
            raise ValueError("branch key and name must not be empty")
        LateLearningRateSchedule(self.learning_rate)

    @property
    def schedule(self) -> LateLearningRateSchedule:
        return LateLearningRateSchedule(self.learning_rate)


@dataclass(frozen=True, slots=True)
class Stage39Source:
    path: Path
    sha256: str
    checkpoint: Mapping[str, object]
    stage_38_source: _STAGE_39.Stage38Source
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
class FreshCheckpointResult:
    name: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_step: int
    fixed_panel_loss: float
    mean_loss: float
    standard_error: float
    batch_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FreshPairedDelta:
    candidate: str
    baseline: str
    mean_delta: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    critical_value: float


@dataclass(frozen=True, slots=True)
class FreshValidationReport:
    eval_iters: int
    seed: int
    stage_38: FreshCheckpointResult
    stage_39: FreshCheckpointResult
    delta_32k_minus_25k: FreshPairedDelta
    final_generator_sha256: str


@dataclass(frozen=True, slots=True)
class BranchReport:
    spec: BranchSpec
    training: TrainingResult
    source_audit: SourceRestoreAudit
    parameter_count: int
    best_train_loss: float
    best_val_loss: float
    best_step: int
    fork_model_sha256: str
    fork_optimizer_sha256: str
    fork_batch_generator_sha256: str
    fork_rng_state_sha256: str
    final_batch_generator_sha256: str
    final_rng_state_sha256: str
    best_checkpoint_sha256: str
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


@dataclass(frozen=True, slots=True)
class Stage40Report:
    source: Stage39Source
    fresh_validation: FreshValidationReport
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


_require_mapping = _STAGE_39._require_mapping
_require_int = _STAGE_39._require_int
_require_finite_float = _STAGE_39._require_finite_float
_sha256_file = _STAGE_39._sha256_file
_runtime_identity = _STAGE_39._runtime_identity
_validate_rng_state = _STAGE_39._validate_rng_state
_restore_rng_state = _STAGE_39._restore_rng_state
_optimizer_rates = _STAGE_39._optimizer_rates
_validate_adamw_state = _STAGE_39._validate_adamw_state
_validate_optimizer_steps = _STAGE_39._validate_optimizer_steps
_validate_optimizer_hyperparameters = _STAGE_39._validate_optimizer_hyperparameters
_expected_training_generator_state = _STAGE_39._expected_training_generator_state
_set_optimizer_learning_rate = _STAGE_39._set_optimizer_learning_rate
_atomic_save_payload = _STAGE_39._atomic_save_payload


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    return _STAGE_39.build_model(data, config, n_head, n_layer, device)


def verify_model_masks(model: GPTLanguageModel) -> None:
    _STAGE_39.verify_model_masks(model)


def get_stage_40_batch(
    data: CharacterData,
    split: str,
    config: TrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _STAGE_39.get_stage_38_batch(
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
    return _STAGE_39.evaluate_on_fixed_batches(
        model,
        data,
        config,
        device,
        fixed_panel_seed=fixed_panel_seed,
    )


def capture_rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {"cpu": torch.get_rng_state().clone()}
    if device.type == "xpu":
        state["xpu"] = [value.cpu().clone() for value in torch.xpu.get_rng_state_all()]
    elif device.type == "cuda":
        state["cuda"] = [
            value.cpu().clone() for value in torch.cuda.get_rng_state_all()
        ]
    return state


def structured_state_sha256(value: object) -> str:
    """Hash nested checkpoint state without depending on pickle encoding."""

    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            for component in (
                b"tensor",
                str(tensor.dtype).encode("ascii"),
                repr(tuple(tensor.shape)).encode("ascii"),
                raw,
            ):
                digest.update(len(component).to_bytes(8, "big"))
                digest.update(component)
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
            return
        encoded = f"{type(item).__name__}:{item!r}".encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    update(value)
    return digest.hexdigest()


def _fresh_summary(report: FreshValidationReport | None) -> object:
    if report is None:
        return None
    delta = report.delta_32k_minus_25k
    return {
        "performed_before_training": True,
        "eval_iters": report.eval_iters,
        "seed": report.seed,
        "sampler": "four_contiguous_T512_panels_reshaped_to_B8_T256",
        "stage_38_step_25000": {
            "checkpoint_sha256": report.stage_38.checkpoint_sha256,
            "fixed_panel_validation_loss": report.stage_38.fixed_panel_loss,
            "fresh_mean_validation_loss": report.stage_38.mean_loss,
            "fresh_standard_error": report.stage_38.standard_error,
        },
        "stage_39_step_32000": {
            "checkpoint_sha256": report.stage_39.checkpoint_sha256,
            "fixed_panel_validation_loss": report.stage_39.fixed_panel_loss,
            "fresh_mean_validation_loss": report.stage_39.mean_loss,
            "fresh_standard_error": report.stage_39.standard_error,
        },
        "paired_delta_32000_minus_25000": {
            "mean": delta.mean_delta,
            "standard_error": delta.standard_error,
            "confidence_low_95": delta.confidence_low,
            "confidence_high_95": delta.confidence_high,
            "critical_value": delta.critical_value,
        },
    }


def is_canonical_stage_39_source_contract(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
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
        and config.eval_interval == DEFAULT_EVAL_INTERVAL
        and config.eval_iters == DEFAULT_EVAL_ITERS
        and config.seed == DEFAULT_SEED
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
        and training_batch_seed == DEFAULT_TRAINING_BATCH_SEED
        and fixed_panel_seed == DEFAULT_FIXED_PANEL_SEED
        and training_rng_seed == DEFAULT_TRAINING_RNG_SEED
    )


def is_canonical_protocol(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    control: BranchSpec,
    treatment: BranchSpec,
    *,
    fresh_eval_iters: int = DEFAULT_FRESH_VALIDATION_ITERS,
    fresh_eval_seed: int = DEFAULT_FRESH_VALIDATION_SEED,
    training_batch_seed: int = DEFAULT_TRAINING_BATCH_SEED,
    fixed_panel_seed: int = DEFAULT_FIXED_PANEL_SEED,
    training_rng_seed: int = DEFAULT_TRAINING_RNG_SEED,
) -> bool:
    return (
        is_canonical_stage_39_source_contract(
            data,
            config,
            n_head,
            n_layer,
            training_batch_seed=training_batch_seed,
            fixed_panel_seed=fixed_panel_seed,
            training_rng_seed=training_rng_seed,
        )
        and config.learning_rate == DEFAULT_SOURCE_LEARNING_RATE
        and config.max_iters == DEFAULT_TARGET_STEP
        and control.learning_rate == DEFAULT_CONTROL_LEARNING_RATE
        and treatment.learning_rate == DEFAULT_TREATMENT_LEARNING_RATE
        and fresh_eval_iters == DEFAULT_FRESH_VALIDATION_ITERS
        and fresh_eval_seed == DEFAULT_FRESH_VALIDATION_SEED
    )


def validate_protocol(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    control: BranchSpec,
    treatment: BranchSpec,
    *,
    fresh_eval_iters: int,
    fresh_eval_seed: int,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> None:
    if config.max_iters <= DEFAULT_SOURCE_STEP:
        raise ValueError(
            f"target_step must be above {DEFAULT_SOURCE_STEP:,}, got "
            f"{config.max_iters:,}"
        )
    if config.learning_rate != DEFAULT_SOURCE_LEARNING_RATE:
        raise ValueError("Stage 40 source learning rate must remain 3e-5")
    if control.learning_rate != DEFAULT_CONTROL_LEARNING_RATE:
        raise ValueError("Stage 40 control learning rate must be 3e-5")
    if treatment.learning_rate != DEFAULT_TREATMENT_LEARNING_RATE:
        raise ValueError("Stage 40 treatment learning rate must be 1e-5")
    if control.key == treatment.key or control.name == treatment.name:
        raise ValueError("Stage 40 branch keys and names must be distinct")
    if n_layer != DEFAULT_N_LAYER:
        raise ValueError(f"Stage 40 requires L={DEFAULT_N_LAYER}, got {n_layer}")
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
            "Stage 40 requires an even batch size for the preserved long-panel sampler"
        )
    if DEFAULT_SOURCE_STEP % config.eval_interval:
        raise ValueError("the 32,000 fork boundary must be evaluated")
    if config.max_iters % config.eval_interval:
        raise ValueError("the target step must align with the evaluation interval")
    panel_size = 2 * config.block_size
    for split, values in (("train", data.train_data), ("val", data.val_data)):
        if len(values) <= panel_size:
            raise ValueError(
                f"The {split} split has {len(values)} tokens, but Stage 40 "
                f"sampling requires a 2T panel of {panel_size} tokens"
            )
    if (
        isinstance(fresh_eval_iters, bool)
        or not isinstance(fresh_eval_iters, int)
        or fresh_eval_iters < 2
    ):
        raise ValueError("fresh_eval_iters must be an integer of at least 2")
    for name, value in (
        ("fresh validation seed", fresh_eval_seed),
        ("training batch seed", training_batch_seed),
        ("fixed panel seed", fixed_panel_seed),
        ("training RNG seed", training_rng_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if fresh_eval_seed == fixed_panel_seed:
        raise ValueError("fresh validation seed must differ from the fixed panel seed")


def validate_artifact_paths(
    *,
    source_checkpoint_path: Path,
    stage_38_checkpoint_path: Path,
    control_best_checkpoint_path: Path,
    control_final_checkpoint_path: Path,
    treatment_best_checkpoint_path: Path,
    treatment_final_checkpoint_path: Path,
    source_data_path: Path | None,
) -> None:
    paths = {
        "Stage 39 source checkpoint": source_checkpoint_path.resolve(),
        "Stage 38 fresh-validation checkpoint": stage_38_checkpoint_path.resolve(),
        "Stage 40 control best checkpoint": control_best_checkpoint_path.resolve(),
        "Stage 40 control final checkpoint": control_final_checkpoint_path.resolve(),
        "Stage 40 treatment best checkpoint": treatment_best_checkpoint_path.resolve(),
        "Stage 40 treatment final checkpoint": treatment_final_checkpoint_path.resolve(),
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


def _stage_39_source_config(config: TrainingConfig) -> TrainingConfig:
    return TrainingConfig(
        batch_size=config.batch_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        learning_rate=DEFAULT_SOURCE_LEARNING_RATE,
        max_iters=DEFAULT_SOURCE_STEP,
        eval_interval=config.eval_interval,
        eval_iters=config.eval_iters,
        seed=config.seed,
    )


def _stage_38_source_from_metadata(
    checkpoint: Mapping[str, object],
) -> _STAGE_39.Stage38Source:
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
            raise ValueError(f"Stage 39 source metadata {name!r} is missing")
    step = _require_int(metadata.get("step"), name="Stage 38 source step")
    best_step = _require_int(
        metadata.get("best_step"), name="Stage 38 source best_step"
    )
    best_train = _require_finite_float(
        metadata.get("best_training_loss"),
        name="Stage 38 source best_training_loss",
    )
    best_val = _require_finite_float(
        metadata.get("best_validation_loss"),
        name="Stage 38 source best_validation_loss",
    )
    snapshot_train = _require_finite_float(
        metadata.get("snapshot_training_loss"),
        name="Stage 38 source snapshot_training_loss",
    )
    snapshot_val = _require_finite_float(
        metadata.get("snapshot_validation_loss"),
        name="Stage 38 source snapshot_validation_loss",
    )
    canonical = (
        sha256 == _STAGE_39.DEFAULT_SOURCE_CHECKPOINT_SHA256
        and step == _STAGE_39.DEFAULT_SOURCE_STEP
        and best_step == step
    )
    return _STAGE_39.Stage38Source(
        path=Path(path_name),
        sha256=sha256,
        checkpoint={},
        step=step,
        best_step=best_step,
        best_train_loss=best_train,
        best_val_loss=best_val,
        snapshot_train_loss=snapshot_train,
        snapshot_val_loss=snapshot_val,
        runtime_device_type=runtime_device_type,
        runtime_device_name=runtime_device_name,
        torch_version=torch_version,
        canonical=canonical,
    )


def validate_and_load_stage_39_source(
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
) -> Stage39Source:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 39 source checkpoint not found: {path}")
    before_hash = _sha256_file(path)
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="root",
    )
    stage_38_source = _stage_38_source_from_metadata(checkpoint)
    source_config = _stage_39_source_config(config)
    schedule = ExtendedLearningRateSchedule()
    validated = _STAGE_39.validate_stage_39_checkpoint(
        path,
        data,
        source_config,
        n_head,
        n_layer,
        schedule,
        stage_38_source,
        expected_kind="final",
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    experiment = _require_mapping(validated.get("experiment"), name="experiment")
    if experiment.get("stage") != _STAGE_39.EXPERIMENT_STAGE:
        raise ValueError("source checkpoint is not a Stage 39 artifact")
    step = _require_int(validated.get("step"), name="step")
    best_step = _require_int(validated.get("best_step"), name="best_step")
    if step != DEFAULT_SOURCE_STEP:
        raise ValueError(f"Stage 40 requires the step-{DEFAULT_SOURCE_STEP:,} endpoint")
    if best_step != step:
        raise ValueError("Stage 40 requires Stage 39's best-at-endpoint state")

    runtime = _require_mapping(validated.get("runtime"), name="runtime")
    runtime_device_type = runtime.get("device_type")
    runtime_device_name = runtime.get("device_name")
    saved_torch_version = runtime.get("torch_version")
    if not isinstance(runtime_device_type, str):
        raise ValueError("Stage 39 runtime device type is missing")
    if not isinstance(runtime_device_name, str):
        raise ValueError("Stage 39 runtime device name is missing")
    if not isinstance(saved_torch_version, str):
        raise ValueError("Stage 39 torch version is missing")
    current_device_type, current_device_name = _runtime_identity(device)
    if (
        runtime_device_type != current_device_type
        or runtime_device_name != current_device_name
    ):
        raise ValueError(
            "exact Stage 39 fork requires the recorded device/backend: "
            f"saved {runtime_device_type}/{runtime_device_name!r}, current "
            f"{current_device_type}/{current_device_name!r}"
        )
    if saved_torch_version != str(torch.__version__):
        raise ValueError(
            "exact Stage 39 fork requires the recorded torch version: "
            f"saved {saved_torch_version!r}, current {str(torch.__version__)!r}"
        )

    best_train_loss = _require_finite_float(
        validated.get("best_training_loss"), name="best_training_loss"
    )
    best_val_loss = _require_finite_float(
        validated.get("best_val_loss"), name="best_val_loss"
    )
    snapshot = _require_mapping(validated.get("snapshot_losses"), name="snapshot_losses")
    snapshot_train_loss = _require_finite_float(
        snapshot.get("train"), name="snapshot train loss"
    )
    snapshot_val_loss = _require_finite_float(
        snapshot.get("validation"), name="snapshot validation loss"
    )
    if snapshot_train_loss != best_train_loss or snapshot_val_loss != best_val_loss:
        raise ValueError("Stage 39 endpoint and selected losses must match")

    canonical = is_canonical_stage_39_source_contract(
        data,
        config,
        n_head,
        n_layer,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    generator_state = validated.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 39 source training-generator state is missing")
    if canonical:
        if before_hash != DEFAULT_SOURCE_CHECKPOINT_SHA256:
            raise ValueError(
                "canonical Stage 39 source hash mismatch: expected "
                f"{DEFAULT_SOURCE_CHECKPOINT_SHA256}, got {before_hash}"
            )
        if best_train_loss != DEFAULT_SOURCE_BEST_TRAIN_LOSS:
            raise ValueError("canonical Stage 39 source train loss changed")
        if best_val_loss != DEFAULT_SOURCE_BEST_VAL_LOSS:
            raise ValueError("canonical Stage 39 source validation loss changed")
        if saved_torch_version != DEFAULT_SOURCE_TORCH_VERSION:
            raise ValueError("canonical Stage 39 source torch version changed")
        if tensor_sha256(generator_state) != DEFAULT_SOURCE_GENERATOR_SHA256:
            raise ValueError("canonical Stage 39 source generator state changed")

    after_hash = _sha256_file(path)
    if after_hash != before_hash:
        raise RuntimeError("Stage 39 source changed during validation")
    return Stage39Source(
        path=path.resolve(),
        sha256=before_hash,
        checkpoint=validated,
        stage_38_source=stage_38_source,
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


def reload_stage_39_source(source: Stage39Source) -> Stage39Source:
    """Give each branch a fresh, independently loaded source payload."""

    before_hash = _sha256_file(source.path)
    if before_hash != source.sha256:
        raise RuntimeError("Stage 39 source hash changed before branch restore")
    checkpoint = _require_mapping(
        torch.load(source.path, map_location="cpu", weights_only=True),
        name="reloaded Stage 39 source",
    )
    if _sha256_file(source.path) != before_hash:
        raise RuntimeError("Stage 39 source changed while reloading a branch")
    if checkpoint.get("step") != source.step:
        raise ValueError("reloaded Stage 39 source step changed")
    return replace(source, checkpoint=checkpoint)


def restore_stage_39_source(
    source: Stage39Source,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    device: torch.device,
) -> None:
    model_state = _require_mapping(
        source.checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    optimizer_state = _require_mapping(
        source.checkpoint.get("optimizer_state_dict"), name="optimizer_state_dict"
    )
    generator_state = source.checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 39 source training-generator state is missing")
    rng_state = _validate_rng_state(
        source.checkpoint.get("rng_state"),
        runtime_device_type=source.runtime_device_type,
    )
    model.load_state_dict(model_state, strict=True)
    verify_model_masks(model)
    # AdamW may retain references to CPU scalar step tensors from the supplied
    # mapping. Clone the nested payload so optimizer.step() cannot mutate the
    # authenticated in-memory source used by post-run validation or the next
    # branch restore.
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    training_generator.set_state(generator_state.cpu())
    # Restore last: model construction and diagnostics must not consume the
    # first dropout draw shared by the two Stage 40 branches.
    _restore_rng_state(rng_state, device)


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
        raise ValueError("sample_count must be an integer of at least 2")
    degrees_of_freedom = sample_count - 1
    if degrees_of_freedom < len(_STUDENT_T_975):
        return _STUDENT_T_975[degrees_of_freedom]
    z = 1.959963984540054
    inverse_df = 1.0 / degrees_of_freedom
    return (
        z
        + (z**3 + z) * inverse_df / 4
        + (5 * z**5 + 16 * z**3 + 3 * z) * inverse_df**2 / 96
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z)
        * inverse_df**3
        / 384
    )


def _mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    mean = tensor.mean().item()
    if len(values) == 1:
        return mean, 0.0
    standard_error = (tensor.std(unbiased=True) / math.sqrt(len(values))).item()
    return mean, standard_error


def _fresh_paired_delta(
    candidate: FreshCheckpointResult,
    baseline: FreshCheckpointResult,
) -> FreshPairedDelta:
    if len(candidate.batch_losses) != len(baseline.batch_losses):
        raise ValueError("paired validation results must have equal lengths")
    if len(candidate.batch_losses) < 2:
        raise ValueError("at least two paired validation batches are required")
    differences = tuple(
        candidate_loss - baseline_loss
        for candidate_loss, baseline_loss in zip(
            candidate.batch_losses,
            baseline.batch_losses,
            strict=True,
        )
    )
    mean, standard_error = _mean_and_standard_error(differences)
    critical = _paired_t_95_critical_value(len(differences))
    margin = critical * standard_error
    return FreshPairedDelta(
        candidate=candidate.name,
        baseline=baseline.name,
        mean_delta=mean,
        standard_error=standard_error,
        confidence_low=mean - margin,
        confidence_high=mean + margin,
        critical_value=critical,
    )


@torch.no_grad()
def evaluate_fresh_paired_models(
    models: Sequence[GPTLanguageModel],
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    eval_iters: int,
    seed: int,
    fixed_panel_seed: int,
) -> tuple[tuple[tuple[float, ...], ...], str]:
    if len(models) < 2:
        raise ValueError("fresh paired evaluation requires at least two models")
    if isinstance(eval_iters, bool) or not isinstance(eval_iters, int) or eval_iters < 2:
        raise ValueError("eval_iters must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("fresh validation seed must be a non-negative integer")
    if seed == fixed_panel_seed:
        raise ValueError("fresh validation seed must differ from the fixed panel seed")

    caller_rng = capture_rng_state(device)
    training_modes = tuple(model.training for model in models)
    losses: list[list[float]] = [[] for _ in models]
    generator = torch.Generator().manual_seed(seed)
    try:
        for model in models:
            model.eval()
        for _ in range(eval_iters):
            inputs, targets = get_stage_40_batch(
                data,
                "val",
                config,
                generator,
                device,
            )
            for index, model in enumerate(models):
                logits, loss = model(inputs, targets)
                assert loss is not None
                losses[index].append(loss.item())
                del logits
                del loss
            del inputs
            del targets
    finally:
        for model, was_training in zip(models, training_modes, strict=True):
            model.train(was_training)
        _restore_rng_state(caller_rng, device)
    return tuple(tuple(values) for values in losses), tensor_sha256(generator.get_state())


def run_fresh_validation(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    stage_38_checkpoint_path: Path,
    source: Stage39Source,
    *,
    eval_iters: int,
    seed: int,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> FreshValidationReport:
    caller_rng = capture_rng_state(device)
    stage_38_hash_before = _sha256_file(stage_38_checkpoint_path)
    stage_39_hash_before = _sha256_file(source.path)
    stage_38_model: GPTLanguageModel | None = None
    stage_39_model: GPTLanguageModel | None = None
    try:
        stage_38_source = _STAGE_39.validate_and_load_stage_38_source(
            stage_38_checkpoint_path,
            data,
            config,
            n_head,
            n_layer,
            ExtendedLearningRateSchedule(),
            device,
            training_batch_seed=training_batch_seed,
            fixed_panel_seed=fixed_panel_seed,
            training_rng_seed=training_rng_seed,
        )
        if stage_38_source.sha256 != source.stage_38_source.sha256:
            raise ValueError(
                "fresh Stage 38 checkpoint does not match Stage 39 provenance"
            )
        stage_38_model = build_model(data, config, n_head, n_layer, device)
        stage_39_model = build_model(data, config, n_head, n_layer, device)
        stage_38_state = _require_mapping(
            stage_38_source.checkpoint.get("model_state_dict"),
            name="Stage 38 model_state_dict",
        )
        stage_39_state = _require_mapping(
            source.checkpoint.get("model_state_dict"),
            name="Stage 39 model_state_dict",
        )
        stage_38_model.load_state_dict(stage_38_state, strict=True)
        stage_39_model.load_state_dict(stage_39_state, strict=True)
        verify_model_masks(stage_38_model)
        verify_model_masks(stage_39_model)
        paired_losses, final_generator_hash = evaluate_fresh_paired_models(
            (stage_38_model, stage_39_model),
            data,
            config,
            device,
            eval_iters=eval_iters,
            seed=seed,
            fixed_panel_seed=fixed_panel_seed,
        )
        stage_38_mean, stage_38_se = _mean_and_standard_error(paired_losses[0])
        stage_39_mean, stage_39_se = _mean_and_standard_error(paired_losses[1])
        stage_38_result = FreshCheckpointResult(
            name="Stage 38 @ 25,000",
            checkpoint_path=stage_38_checkpoint_path.resolve(),
            checkpoint_sha256=stage_38_source.sha256,
            checkpoint_step=stage_38_source.step,
            fixed_panel_loss=stage_38_source.snapshot_val_loss,
            mean_loss=stage_38_mean,
            standard_error=stage_38_se,
            batch_losses=paired_losses[0],
        )
        stage_39_result = FreshCheckpointResult(
            name="Stage 39 @ 32,000",
            checkpoint_path=source.path,
            checkpoint_sha256=source.sha256,
            checkpoint_step=source.step,
            fixed_panel_loss=source.snapshot_val_loss,
            mean_loss=stage_39_mean,
            standard_error=stage_39_se,
            batch_losses=paired_losses[1],
        )
        return FreshValidationReport(
            eval_iters=eval_iters,
            seed=seed,
            stage_38=stage_38_result,
            stage_39=stage_39_result,
            delta_32k_minus_25k=_fresh_paired_delta(
                stage_39_result,
                stage_38_result,
            ),
            final_generator_sha256=final_generator_hash,
        )
    finally:
        del stage_38_model
        del stage_39_model
        clear_accelerator_cache(device)
        _restore_rng_state(caller_rng, device)
        if _sha256_file(stage_38_checkpoint_path) != stage_38_hash_before:
            raise RuntimeError("Stage 38 checkpoint changed during fresh validation")
        if _sha256_file(source.path) != stage_39_hash_before:
            raise RuntimeError("Stage 39 checkpoint changed during fresh validation")


@dataclass(slots=True)
class Stage40ValidationCheckpoint(BestValidationCheckpoint):
    """Persist one branch with its common-fork and LR-only provenance."""

    branch: BranchSpec | None = None
    schedule: LateLearningRateSchedule | None = None
    target_step: int = 0
    source: Stage39Source | None = None
    fresh_validation: FreshValidationReport | None = None
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
    fork_model_sha256: str = ""
    fork_optimizer_sha256: str = ""
    fork_batch_generator_sha256: str = ""
    fork_rng_state_sha256: str = ""

    def _decorate_payload(
        self,
        payload: dict[str, Any],
        *,
        step: int,
        checkpoint_kind: CheckpointKind,
        snapshot_losses: LossEstimate,
    ) -> dict[str, Any]:
        if (
            self.branch is None
            or self.schedule is None
            or self.source is None
            or self.best_step is None
            or self.best_train_loss is None
        ):
            raise RuntimeError("incomplete Stage 40 checkpoint metadata")
        optimizer_state = _require_mapping(
            payload.get("optimizer_state_dict"), name="optimizer_state_dict"
        )
        if _optimizer_rates(optimizer_state) != {self.branch.learning_rate}:
            raise RuntimeError(
                f"optimizer LR at step {step} must be "
                f"{self.branch.learning_rate:g}"
            )
        payload["checkpoint_kind"] = checkpoint_kind
        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "head_size": self.config.n_embd // self.n_head,
                "feed_forward_width": 4 * self.config.n_embd,
                "residual_dropout": DEFAULT_DROPOUT,
                "dropout_placement": (
                    _STAGE_39._STAGE_38._STAGE_37._STAGE_36.DROPOUT_PLACEMENT
                ),
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
                "learning_rate": self.branch.learning_rate,
                "max_iters": self.target_step,
                "target_step": self.target_step,
                "source_checkpoint_step": self.source.step,
                "source_learning_rate": DEFAULT_SOURCE_LEARNING_RATE,
                "branch_learning_rate": self.branch.learning_rate,
                "learning_rate_schedule": self.schedule.as_metadata(
                    target_step=self.target_step
                ),
                "continuation_segment": {
                    "start_update": self.source.step,
                    "end_update_exclusive": self.target_step,
                    "learning_rate": self.branch.learning_rate,
                },
                "residual_dropout": DEFAULT_DROPOUT,
                "training_batch_seed": self.training_batch_seed,
                "fixed_panel_seed": self.fixed_panel_seed,
                "training_rng_seed": self.training_rng_seed,
                "targets_per_update": self.targets_per_update,
                "total_target_presentations": self.targets_per_update
                * self.target_step,
                "added_target_presentations_after_stage_39": (
                    self.targets_per_update * (self.target_step - self.source.step)
                ),
            }
        )
        payload["source_checkpoint"] = {
            "stage": _STAGE_39.EXPERIMENT_STAGE,
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
        payload["fork_state"] = {
            "captured_before_lr_override": True,
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
            "kind": "restored_stage_39_terminal_complete_shakespeare_state",
            "fresh_initial_state_sha256": self.initial_state_sha256,
            "architecture_changed": False,
            "corpus_changed": False,
        }
        payload["experiment"] = {
            "stage": EXPERIMENT_STAGE,
            "title": EXPERIMENT_TITLE,
            "protocol": PROTOCOL_NAME,
            "comparison_variable": COMPARISON_VARIABLE,
            "branch_key": self.branch.key,
            "branch_name": self.branch.name,
            "from_scratch": False,
            "same_model_as_stage_39": True,
            "same_corpus_as_stage_39": True,
            "same_split_as_stage_39": True,
            "same_sampler_across_branches": True,
            "same_fixed_evaluation_panel_across_branches": True,
            "same_training_batches_across_branches": True,
            "same_dropout_rng_stream_across_branches": True,
            "same_optimizer_state_at_fork": True,
            "optimizer_state_preserved": True,
            "optimizer_restarted": False,
            "rng_state_preserved": True,
            "architecture_frozen": True,
            "only_intended_post_fork_difference": "learning_rate",
            "trained_branches": 2,
        }
        runtime_type, runtime_name = _runtime_identity(self.device)
        payload["runtime"] = {
            "device_type": runtime_type,
            "device_name": runtime_name,
            "torch_version": str(torch.__version__),
            "matches_stage_39_source": (
                runtime_type == self.source.runtime_device_type
                and runtime_name == self.source.runtime_device_name
                and str(torch.__version__) == self.source.torch_version
            ),
        }
        payload["evaluation_protocol"] = {
            "selection_metric": "branch_fixed_panel_validation_loss",
            "fixed_panel_seed": self.fixed_panel_seed,
            "eval_interval": self.config.eval_interval,
            "eval_iters": self.config.eval_iters,
            "strict_improvement": True,
            "dropout_enabled": False,
            "absolute_step_numbering": True,
            "branch_comparison_uses_matched_steps": True,
        }
        payload["fresh_validation_25k_vs_32k"] = _fresh_summary(
            self.fresh_validation
        )
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
            super(Stage40ValidationCheckpoint, self)._payload(step),
            step=step,
            checkpoint_kind="best",
            snapshot_losses=LossEstimate(
                train=self.best_train_loss,
                val=self.best_val_loss,
            ),
        )

    def save_final(self, path: Path, record: EvaluationRecord) -> None:
        payload = self._decorate_payload(
            super(Stage40ValidationCheckpoint, self)._payload(record.step),
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


def train_branch(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    schedule: LateLearningRateSchedule,
    *,
    start_step: int,
    fixed_panel_seed: int,
    on_evaluation: Callable[[EvaluationRecord], None],
) -> TrainingResult:
    if start_step != DEFAULT_SOURCE_STEP:
        raise ValueError(
            f"Stage 40 must start at step {DEFAULT_SOURCE_STEP:,}, got "
            f"{start_step:,}"
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
        if learning_rate != schedule.branch_learning_rate:
            raise RuntimeError(
                f"Stage 40 update {update_index} selected a pre-fork LR"
            )
        _set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_stage_40_batch(
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
    best: Stage40ValidationCheckpoint,
    source: Stage39Source,
    branch: BranchSpec,
    *,
    final_checkpoint_path: Path,
) -> tuple[Callable[[EvaluationRecord], None], list[SourceRestoreAudit]]:
    source_audits: list[SourceRestoreAudit] = []

    def record_evaluation(record: EvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr {branch.learning_rate:g} | "
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
                    "the restored step-32,000 fixed-panel loss does not match "
                    "the Stage 39 source at four decimals: expected "
                    f"{source.snapshot_train_loss:.4f} / "
                    f"{source.snapshot_val_loss:.4f}, got "
                    f"{record.losses.train:.4f} / {record.losses.val:.4f}"
                )
            best.source_audit_verified = True
            # Materialize a resumable branch-specific source-step winner even
            # if this branch never improves. Its optimizer already has the
            # branch LR, but all moments and step counters are inherited.
            if best.best_step == record.step:
                best.save(record.step)
            source_audits.append(audit)

        if record.step == best.target_step:
            if not best.source_audit_verified:
                raise RuntimeError("training reached its target without source audit")
            best.save_final(final_checkpoint_path, record)

    return record_evaluation, source_audits


def validate_stage_40_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    source: Stage39Source,
    branch: BranchSpec,
    fresh_validation: FreshValidationReport | None,
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
        raise ValueError("unsupported Stage 40 checkpoint version")
    if checkpoint.get("checkpoint_kind") != expected_kind:
        raise ValueError(
            f"expected a {expected_kind!r} Stage 40 checkpoint, got "
            f"{checkpoint.get('checkpoint_kind')!r}"
        )
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 40 checkpoint must not restart AdamW")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 40 optimizer provenance must remain known")
    step = _require_int(checkpoint.get("step"), name="step")
    best_step = _require_int(checkpoint.get("best_step"), name="best_step")
    best_val_loss = _require_finite_float(
        checkpoint.get("best_val_loss"), name="best_val_loss"
    )
    best_train_loss = _require_finite_float(
        checkpoint.get("best_training_loss"), name="best_training_loss"
    )
    if expected_kind == "best" and step != best_step:
        raise ValueError("Stage 40 best checkpoint must store its selected step")
    if expected_kind == "final" and step != config.max_iters:
        raise ValueError("Stage 40 final checkpoint is not at the target step")
    if not (source.step <= best_step <= step):
        raise ValueError("Stage 40 best step is outside the saved trajectory")

    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    expected_experiment = {
        "stage": EXPERIMENT_STAGE,
        "protocol": PROTOCOL_NAME,
        "comparison_variable": COMPARISON_VARIABLE,
        "branch_key": branch.key,
        "branch_name": branch.name,
        "same_training_batches_across_branches": True,
        "same_dropout_rng_stream_across_branches": True,
        "same_optimizer_state_at_fork": True,
        "optimizer_state_preserved": True,
        "optimizer_restarted": False,
        "architecture_frozen": True,
        "only_intended_post_fork_difference": "learning_rate",
        "trained_branches": 2,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(f"Stage 40 experiment field {key!r} changed")

    source_metadata = _require_mapping(
        checkpoint.get("source_checkpoint"), name="source_checkpoint"
    )
    for key, expected in (
        ("stage", _STAGE_39.EXPERIMENT_STAGE),
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
            raise ValueError(f"Stage 40 source field {key!r} mismatch")

    fork_state = _require_mapping(checkpoint.get("fork_state"), name="fork_state")
    if fork_state.get("captured_before_lr_override") is not True:
        raise ValueError("Stage 40 fork state must precede the LR override")
    source_model_state = _require_mapping(
        source.checkpoint.get("model_state_dict"),
        name="source model_state_dict",
    )
    source_optimizer_state = _require_mapping(
        source.checkpoint.get("optimizer_state_dict"),
        name="source optimizer_state_dict",
    )
    source_generator_state = source.checkpoint.get("training_generator_state")
    if not isinstance(source_generator_state, torch.Tensor):
        raise ValueError("Stage 39 source generator state is missing")
    expected_fork_state = {
        "model_sha256": tensor_mapping_sha256(source_model_state),
        "optimizer_sha256": structured_state_sha256(source_optimizer_state),
        "training_generator_sha256": tensor_sha256(source_generator_state),
        "torch_rng_sha256": structured_state_sha256(
            source.checkpoint.get("rng_state")
        ),
    }
    for key, expected in expected_fork_state.items():
        if fork_state.get(key) != expected:
            raise ValueError(f"Stage 40 fork hash {key!r} does not match source")

    architecture = _require_mapping(checkpoint.get("architecture"), name="architecture")
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
                f"Stage 40 architecture {key!r} mismatch: expected "
                f"{expected!r}, got {observed!r}"
            )

    training = _require_mapping(
        checkpoint.get("training_config"), name="training_config"
    )
    expected_training = {
        "batch_size": config.batch_size,
        "learning_rate": branch.learning_rate,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": config.max_iters,
        "target_step": config.max_iters,
        "source_checkpoint_step": source.step,
        "source_learning_rate": DEFAULT_SOURCE_LEARNING_RATE,
        "branch_learning_rate": branch.learning_rate,
        "learning_rate_schedule": branch.schedule.as_metadata(
            target_step=config.max_iters
        ),
        "continuation_segment": {
            "start_update": source.step,
            "end_update_exclusive": config.max_iters,
            "learning_rate": branch.learning_rate,
        },
        "training_batch_seed": training_batch_seed,
        "fixed_panel_seed": fixed_panel_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Stage 40 training field {key!r} mismatch")

    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("Stage 40 checkpoint corpus fingerprint mismatch")
    corpus = _require_mapping(checkpoint.get("corpus"), name="corpus")
    for key, expected in (
        ("data_fingerprint", fingerprint_data(data)),
        ("decoded_characters", data.num_characters),
        ("train_characters", len(data.train_data)),
        ("validation_characters", len(data.val_data)),
        ("vocabulary_size", data.vocabulary.size),
    ):
        if corpus.get(key) != expected:
            raise ValueError(f"Stage 40 corpus field {key!r} mismatch")

    runtime = _require_mapping(checkpoint.get("runtime"), name="runtime")
    runtime_type = runtime.get("device_type")
    if runtime_type != source.runtime_device_type:
        raise ValueError("Stage 40 runtime type does not match Stage 39")
    if runtime.get("device_name") != source.runtime_device_name:
        raise ValueError("Stage 40 runtime device does not match Stage 39")
    if runtime.get("torch_version") != source.torch_version:
        raise ValueError("Stage 40 torch version changed during the fork")
    if runtime.get("matches_stage_39_source") is not True:
        raise ValueError("Stage 40 runtime must match the Stage 39 source")
    assert isinstance(runtime_type, str)
    _validate_rng_state(checkpoint.get("rng_state"), runtime_device_type=runtime_type)

    optimizer_state = _validate_adamw_state(
        checkpoint.get("optimizer_state_dict"),
        expected_learning_rate=branch.learning_rate,
    )
    _validate_optimizer_steps(optimizer_state, expected_step=step)
    _validate_optimizer_hyperparameters(optimizer_state)
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 40 training-generator state must be a tensor")
    expected_generator_state = _expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator_state):
        raise ValueError("Stage 40 batch-generator state is inconsistent")

    if checkpoint.get("fresh_validation_25k_vs_32k") != _fresh_summary(
        fresh_validation
    ):
        raise ValueError("Stage 40 fresh-validation metadata changed")

    snapshot_losses = _require_mapping(
        checkpoint.get("snapshot_losses"), name="snapshot_losses"
    )
    snapshot_train = _require_finite_float(
        snapshot_losses.get("train"), name="snapshot train loss"
    )
    snapshot_val = _require_finite_float(
        snapshot_losses.get("validation"), name="snapshot validation loss"
    )
    if expected_kind == "best":
        if snapshot_train != best_train_loss or snapshot_val != best_val_loss:
            raise ValueError("Stage 40 best snapshot losses are inconsistent")

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    with torch.random.fork_rng(devices=[]):
        model = build_model(data, config, n_head, n_layer)
    model.load_state_dict(model_state, strict=True)
    verify_model_masks(model)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if architecture.get("parameter_count") != parameter_count:
        raise ValueError("Stage 40 checkpoint parameter count mismatch")
    return checkpoint


def optimizer_lineage_sha256(optimizer_state: Mapping[str, object]) -> str:
    """Hash all AdamW state and options except the treatment variable, LR."""

    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list):
        raise ValueError("optimizer param_groups must be a list")
    normalized_groups: list[dict[object, object]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("optimizer parameter group must be a mapping")
        normalized_groups.append(
            {key: value for key, value in group.items() if key != "lr"}
        )
    normalized = {
        key: (normalized_groups if key == "param_groups" else value)
        for key, value in optimizer_state.items()
    }
    return structured_state_sha256(normalized)


def run_branch(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    source: Stage39Source,
    spec: BranchSpec,
    fresh_validation: FreshValidationReport,
    *,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> BranchReport:
    branch_source = reload_stage_39_source(source)
    clear_accelerator_cache(device)
    seed_everything(config.seed)
    model = build_model(data, config, n_head, n_layer, device)
    verify_model_masks(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DEFAULT_SOURCE_LEARNING_RATE
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    restore_stage_39_source(
        branch_source,
        model,
        optimizer,
        training_generator,
        device,
    )

    # Capture the independently restored common state before touching LR.
    fork_model_sha256 = tensor_mapping_sha256(model.state_dict())
    fork_optimizer_state = optimizer.state_dict()
    fork_optimizer_sha256 = structured_state_sha256(fork_optimizer_state)
    fork_optimizer_lineage_sha256 = optimizer_lineage_sha256(
        fork_optimizer_state
    )
    fork_batch_generator_sha256 = tensor_sha256(training_generator.get_state())
    fork_rng_state_sha256 = structured_state_sha256(capture_rng_state(device))

    # AdamW.load_state_dict restores the source's 3e-5 param-group LR. Apply
    # the treatment only after the full restore; no moments or step counters
    # are recreated.
    _set_optimizer_learning_rate(optimizer, spec.learning_rate)
    if optimizer_lineage_sha256(optimizer.state_dict()) != (
        fork_optimizer_lineage_sha256
    ):
        raise RuntimeError("changing branch LR altered non-LR optimizer state")
    if _optimizer_rates(optimizer.state_dict()) != {spec.learning_rate}:
        raise RuntimeError("branch LR override did not reach every parameter group")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if source.canonical and parameter_count != DEFAULT_PARAMETER_COUNT:
        raise RuntimeError("canonical Stage 40 parameter count changed")

    source_corpus = _require_mapping(
        branch_source.checkpoint.get("corpus"), name="source corpus"
    )
    source_initialization = _require_mapping(
        branch_source.checkpoint.get("initialization"),
        name="source initialization",
    )
    initial_state_sha256 = source_initialization.get("fresh_initial_state_sha256")
    if not isinstance(initial_state_sha256, str):
        raise ValueError("Stage 39 initialization hash is missing")
    best = Stage40ValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=spec.best_checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        best_val_loss=branch_source.best_val_loss,
        best_step=branch_source.best_step,
        branch=spec,
        schedule=spec.schedule,
        target_step=config.max_iters,
        source=branch_source,
        fresh_validation=fresh_validation,
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
        best_train_loss=branch_source.best_train_loss,
        fork_model_sha256=fork_model_sha256,
        fork_optimizer_sha256=fork_optimizer_sha256,
        fork_batch_generator_sha256=fork_batch_generator_sha256,
        fork_rng_state_sha256=fork_rng_state_sha256,
    )
    callback, source_audits = make_evaluation_callback(
        best,
        branch_source,
        spec,
        final_checkpoint_path=spec.final_checkpoint_path,
    )

    print(f"\n{spec.name} (lr={spec.learning_rate:g})")
    print(f"  source:                 {branch_source.path}")
    print(f"  source SHA-256:         {branch_source.sha256}")
    print(f"  restore step:           {branch_source.step:,}")
    print(f"  target step:            {config.max_iters:,}")
    print("  optimizer lineage:      restored; LR override only; no restart")
    print("  batch/dropout streams:  restored independently from common source")
    result = train_branch(
        model,
        optimizer,
        data,
        config,
        device,
        training_generator,
        spec.schedule,
        start_step=branch_source.step,
        fixed_panel_seed=fixed_panel_seed,
        on_evaluation=callback,
    )
    if len(source_audits) != 1:
        raise RuntimeError(f"{spec.name} did not perform exactly one source audit")
    source_audit = source_audits[0]
    assert best.best_step is not None
    assert best.best_train_loss is not None

    final_batch_generator_sha256 = tensor_sha256(training_generator.get_state())
    final_rng_state_sha256 = structured_state_sha256(capture_rng_state(device))
    if _sha256_file(branch_source.path) != branch_source.sha256:
        raise RuntimeError("Stage 39 source changed while training a branch")
    for artifact_path, kind in (
        (spec.best_checkpoint_path, "best"),
        (spec.final_checkpoint_path, "final"),
    ):
        validate_stage_40_checkpoint(
            artifact_path,
            data,
            config,
            n_head,
            n_layer,
            branch_source,
            spec,
            fresh_validation,
            expected_kind=kind,
            training_batch_seed=training_batch_seed,
            fixed_panel_seed=fixed_panel_seed,
            training_rng_seed=training_rng_seed,
        )
    final_payload = _require_mapping(
        torch.load(
            spec.final_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        ),
        name="final Stage 40 checkpoint",
    )
    if structured_state_sha256(final_payload.get("rng_state")) != (
        final_rng_state_sha256
    ):
        raise RuntimeError("terminal checkpoint captured the wrong Torch RNG state")

    selected = _require_mapping(
        torch.load(
            spec.best_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        ),
        name="selected Stage 40 checkpoint",
    )
    selected_state = _require_mapping(
        selected.get("model_state_dict"), name="selected model_state_dict"
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
    report = BranchReport(
        spec=spec,
        training=result,
        source_audit=source_audit,
        parameter_count=parameter_count,
        best_train_loss=best.best_train_loss,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        fork_model_sha256=fork_model_sha256,
        fork_optimizer_sha256=fork_optimizer_sha256,
        fork_batch_generator_sha256=fork_batch_generator_sha256,
        fork_rng_state_sha256=fork_rng_state_sha256,
        final_batch_generator_sha256=final_batch_generator_sha256,
        final_rng_state_sha256=final_rng_state_sha256,
        best_checkpoint_sha256=checkpoint_sha256(spec.best_checkpoint_path),
        final_checkpoint_sha256=checkpoint_sha256(spec.final_checkpoint_path),
        sample=sample,
    )
    del callback
    del best
    del optimizer
    del model
    del branch_source
    clear_accelerator_cache(device)
    return report


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    source_checkpoint_path: Path,
    stage_38_checkpoint_path: Path,
    control: BranchSpec,
    treatment: BranchSpec,
    *,
    source_data_path: Path | None,
    fresh_eval_iters: int,
    fresh_eval_seed: int,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> Stage40Report:
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    validate_protocol(
        data,
        config,
        n_head,
        n_layer,
        control,
        treatment,
        fresh_eval_iters=fresh_eval_iters,
        fresh_eval_seed=fresh_eval_seed,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    validate_artifact_paths(
        source_checkpoint_path=source_checkpoint_path,
        stage_38_checkpoint_path=stage_38_checkpoint_path,
        control_best_checkpoint_path=control.best_checkpoint_path,
        control_final_checkpoint_path=control.final_checkpoint_path,
        treatment_best_checkpoint_path=treatment.best_checkpoint_path,
        treatment_final_checkpoint_path=treatment.final_checkpoint_path,
        source_data_path=source_data_path,
    )
    source = validate_and_load_stage_39_source(
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

    print("\nFresh paired 25k vs 32k validation")
    print(
        f"  {fresh_eval_iters} shared Complete-Shakespeare validation batches "
        f"from seed {fresh_eval_seed:,}"
    )
    fresh_validation = run_fresh_validation(
        data,
        config,
        n_head,
        n_layer,
        device,
        stage_38_checkpoint_path,
        source,
        eval_iters=fresh_eval_iters,
        seed=fresh_eval_seed,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    fresh_delta = fresh_validation.delta_32k_minus_25k
    print(
        f"  25k fresh val: {fresh_validation.stage_38.mean_loss:.6f} "
        f"+/- {fresh_validation.stage_38.standard_error:.6f} SE"
    )
    print(
        f"  32k fresh val: {fresh_validation.stage_39.mean_loss:.6f} "
        f"+/- {fresh_validation.stage_39.standard_error:.6f} SE"
    )
    print(
        f"  paired delta (32k-25k): {fresh_delta.mean_delta:+.6f} "
        f"[{fresh_delta.confidence_low:+.6f}, "
        f"{fresh_delta.confidence_high:+.6f}]"
    )

    print("\nStage 40 paired branches")
    print(
        f"  B={config.batch_size}, T={config.block_size}, C={config.n_embd}, "
        f"H={n_head}, D={config.n_embd // n_head}, "
        f"FF={4 * config.n_embd}, L={n_layer}, dropout={DEFAULT_DROPOUT:g}"
    )
    print("  attention: W23, W23, W23, W23, Full, Full")
    control_report = run_branch(
        data,
        config,
        n_head,
        n_layer,
        device,
        sample_length,
        source,
        control,
        fresh_validation,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    treatment_report = run_branch(
        data,
        config,
        n_head,
        n_layer,
        device,
        sample_length,
        source,
        treatment,
        fresh_validation,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    if _sha256_file(source.path) != source_hash_before:
        raise RuntimeError("Stage 39 source changed during Stage 40")
    if control_report.parameter_count != treatment_report.parameter_count:
        raise RuntimeError("Stage 40 branch parameter counts differ")
    control_steps = tuple(record.step for record in control_report.training.history)
    treatment_steps = tuple(
        record.step for record in treatment_report.training.history
    )
    if control_steps != treatment_steps:
        raise RuntimeError("Stage 40 branches were not evaluated at matching steps")
    if control_report.training.initial != treatment_report.training.initial:
        raise RuntimeError("Stage 40 branches did not share the same initial losses")
    for field_name in (
        "fork_model_sha256",
        "fork_optimizer_sha256",
        "fork_batch_generator_sha256",
        "fork_rng_state_sha256",
        "final_batch_generator_sha256",
        "final_rng_state_sha256",
    ):
        if getattr(control_report, field_name) != getattr(
            treatment_report, field_name
        ):
            raise RuntimeError(
                f"Stage 40 paired-state mismatch for {field_name}"
            )
    if control_report.fork_batch_generator_sha256 != DEFAULT_SOURCE_GENERATOR_SHA256 and (
        source.canonical
    ):
        raise RuntimeError("canonical Stage 40 fork generator hash changed")

    final_val_delta = (
        treatment_report.training.final.val - control_report.training.final.val
    )
    best_val_delta = treatment_report.best_val_loss - control_report.best_val_loss
    return Stage40Report(
        source=source,
        fresh_validation=fresh_validation,
        control=control_report,
        treatment=treatment_report,
        final_val_delta=final_val_delta,
        best_val_delta=best_val_delta,
    )


def print_final_summary(report: Stage40Report) -> None:
    delta = report.fresh_validation.delta_32k_minus_25k
    print("\nStage 40 summary")
    print(
        f"  common source:            train "
        f"{report.source.snapshot_train_loss:.4f} / val "
        f"{report.source.snapshot_val_loss:.4f} @ {report.source.step:,}"
    )
    print(
        f"  fresh 32k-25k val delta: {delta.mean_delta:+.6f} "
        f"(95% panel interval {delta.confidence_low:+.6f}, "
        f"{delta.confidence_high:+.6f})"
    )
    for branch in (report.control, report.treatment):
        print(f"\n  {branch.spec.name} (lr {branch.spec.learning_rate:g})")
        print(
            f"    selected: train {branch.best_train_loss:.4f} / "
            f"val {branch.best_val_loss:.4f} @ {branch.best_step:,}"
        )
        print(
            f"    final:    train {branch.training.final.train:.4f} / "
            f"val {branch.training.final.val:.4f} @ {branch.final_step:,}"
        )
        print(f"    best checkpoint:  {branch.spec.best_checkpoint_path}")
        print(f"    best SHA-256:     {branch.best_checkpoint_sha256}")
        print(f"    final checkpoint: {branch.spec.final_checkpoint_path}")
        print(f"    final SHA-256:    {branch.final_checkpoint_sha256}")
    print(
        f"\n  final delta (1e-5 - 3e-5): {report.final_val_delta:+.6f} "
        "(negative favors 1e-5)"
    )
    print(
        f"  best delta (1e-5 - 3e-5):  {report.best_val_delta:+.6f}"
    )
    print(f"  final-step winner:           {report.final_winner}")
    print("  paired batch streams:        verified identical")
    print("  paired Torch RNG streams:    verified identical")
    print("\nControl selected-best generated text")
    print(report.control.sample)
    print("\nTreatment selected-best generated text")
    print(report.treatment.sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 40: independently fork the exact Stage 39 step-32,000 "
            "state and compare 3e-5 with 1e-5 through step 38,000."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_COMPLETE_DATA_PATH)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT_PATH
    )
    parser.add_argument(
        "--stage-38-checkpoint",
        type=Path,
        default=DEFAULT_STAGE_38_CHECKPOINT_PATH,
    )
    parser.add_argument("--target-step", type=int, default=DEFAULT_TARGET_STEP)
    parser.add_argument(
        "--fresh-eval-iters", type=int, default=DEFAULT_FRESH_VALIDATION_ITERS
    )
    parser.add_argument(
        "--fresh-eval-seed", type=int, default=DEFAULT_FRESH_VALIDATION_SEED
    )
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument(
        "--control-best-checkpoint-path",
        type=Path,
        default=DEFAULT_CONTROL_BEST_CHECKPOINT_PATH,
    )
    parser.add_argument("--control-final-checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--treatment-best-checkpoint-path",
        type=Path,
        default=DEFAULT_TREATMENT_BEST_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--treatment-final-checkpoint-path", type=Path, default=None
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _default_final_checkpoint_path(branch_key: str, target_step: int) -> Path:
    if branch_key == "control":
        if target_step == DEFAULT_TARGET_STEP:
            return DEFAULT_CONTROL_FINAL_CHECKPOINT_PATH
        prefix = "stage_40_control_3e_5"
    elif branch_key == "treatment":
        if target_step == DEFAULT_TARGET_STEP:
            return DEFAULT_TREATMENT_FINAL_CHECKPOINT_PATH
        prefix = "stage_40_treatment_1e_5"
    else:
        raise ValueError(f"unknown Stage 40 branch key: {branch_key!r}")
    return CHECKPOINT_DIRECTORY / f"{prefix}_step_{target_step}_final_checkpoint.pt"


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        batch_size=DEFAULT_BATCH_SIZE,
        block_size=DEFAULT_BLOCK_SIZE,
        n_embd=DEFAULT_N_EMBD,
        learning_rate=DEFAULT_SOURCE_LEARNING_RATE,
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
    control = BranchSpec(
        key="control",
        name="A: continue 3e-5",
        learning_rate=DEFAULT_CONTROL_LEARNING_RATE,
        best_checkpoint_path=args.control_best_checkpoint_path,
        final_checkpoint_path=(
            _default_final_checkpoint_path("control", args.target_step)
            if args.control_final_checkpoint_path is None
            else args.control_final_checkpoint_path
        ),
    )
    treatment = BranchSpec(
        key="treatment",
        name="B: refine at 1e-5",
        learning_rate=DEFAULT_TREATMENT_LEARNING_RATE,
        best_checkpoint_path=args.treatment_best_checkpoint_path,
        final_checkpoint_path=(
            _default_final_checkpoint_path("treatment", args.target_step)
            if args.treatment_final_checkpoint_path is None
            else args.treatment_final_checkpoint_path
        ),
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
        f"Protocol: exact fork at {DEFAULT_SOURCE_STEP:,}, matched target "
        f"{config.max_iters:,}"
    )
    print(
        f"Branch LRs: A={control.learning_rate:g}, B={treatment.learning_rate:g}"
    )

    report = run_experiment(
        data,
        config,
        DEFAULT_N_HEAD,
        DEFAULT_N_LAYER,
        device,
        args.sample_length,
        args.source_checkpoint,
        args.stage_38_checkpoint,
        control,
        treatment,
        source_data_path=args.data,
        fresh_eval_iters=args.fresh_eval_iters,
        fresh_eval_seed=args.fresh_eval_seed,
        training_batch_seed=DEFAULT_TRAINING_BATCH_SEED,
        fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED,
        training_rng_seed=DEFAULT_TRAINING_RNG_SEED,
    )
    print_final_summary(report)


if __name__ == "__main__":
    main()
