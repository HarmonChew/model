from __future__ import annotations

import argparse
import hashlib
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


# Stage 38 extends the completed Stage 37 run without changing its model,
# corpus, sampler, fixed evaluation panel, AdamW lineage, or stochastic stream.
_STAGE_37_PATH = Path(__file__).with_name("37_data_scaling.py")
_STAGE_37_MODULE_NAME = "stage_37_data_scaling_for_stage_38"
_STAGE_37_SPEC = importlib.util.spec_from_file_location(
    _STAGE_37_MODULE_NAME,
    _STAGE_37_PATH,
)
assert _STAGE_37_SPEC is not None and _STAGE_37_SPEC.loader is not None
_STAGE_37 = importlib.util.module_from_spec(_STAGE_37_SPEC)
sys.modules[_STAGE_37_MODULE_NAME] = _STAGE_37
_STAGE_37_SPEC.loader.exec_module(_STAGE_37)


GPTLanguageModel = _STAGE_37.GPTLanguageModel
BestValidationCheckpoint = _STAGE_37.BestValidationCheckpoint
checkpoint_sha256 = _STAGE_37.checkpoint_sha256
clear_accelerator_cache = _STAGE_37.clear_accelerator_cache
fingerprint_data = _STAGE_37.fingerprint_data
generate_from_final_model = _STAGE_37.generate_from_final_model
seed_everything = _STAGE_37.seed_everything
tensor_mapping_sha256 = _STAGE_37.tensor_mapping_sha256
tensor_sha256 = _STAGE_37._STAGE_36.tensor_sha256


EXPERIMENT_STAGE = 38
EXPERIMENT_TITLE = "Extended Complete Shakespeare training"
PROTOCOL_NAME = "stage_37_exact_state_replay_then_low_lr_extension"
COMPARISON_VARIABLE = "additional_optimization_at_lower_learning_rate"

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
DEFAULT_COMPLETE_DATA_PATH = _STAGE_37.DEFAULT_COMPLETE_DATA_PATH
CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_SOURCE_CHECKPOINT_PATH = _STAGE_37.DEFAULT_CHECKPOINT_PATH
DEFAULT_BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_38_complete_shakespeare_extended_best_checkpoint.pt"
)
DEFAULT_REPLAY_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_38_complete_shakespeare_step_18000_reconstructed_checkpoint.pt"
)
DEFAULT_FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "stage_38_complete_shakespeare_step_25000_final_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = _STAGE_37.DEFAULT_BATCH_SIZE
DEFAULT_BLOCK_SIZE = _STAGE_37.DEFAULT_BLOCK_SIZE
DEFAULT_LOCAL_CONTEXT = _STAGE_37.DEFAULT_LOCAL_CONTEXT
DEFAULT_LAYER_CONTEXTS = _STAGE_37.DEFAULT_LAYER_CONTEXTS
DEFAULT_GLOBAL_LAYER_INDICES = _STAGE_37.DEFAULT_GLOBAL_LAYER_INDICES
DEFAULT_GLOBAL_LAYER_NUMBERS = _STAGE_37.DEFAULT_GLOBAL_LAYER_NUMBERS
DEFAULT_N_EMBD = _STAGE_37.DEFAULT_N_EMBD
DEFAULT_N_HEAD = _STAGE_37.DEFAULT_N_HEAD
DEFAULT_HEAD_SIZE = _STAGE_37.DEFAULT_HEAD_SIZE
DEFAULT_FEED_FORWARD_WIDTH = _STAGE_37.DEFAULT_FEED_FORWARD_WIDTH
DEFAULT_N_LAYER = _STAGE_37.DEFAULT_N_LAYER
DEFAULT_DROPOUT = _STAGE_37.DEFAULT_DROPOUT

DEFAULT_INITIAL_LEARNING_RATE = _STAGE_37.DEFAULT_INITIAL_LEARNING_RATE
DEFAULT_FIRST_DECAY_STEP = _STAGE_37.DEFAULT_FIRST_DECAY_STEP
DEFAULT_MIDDLE_LEARNING_RATE = _STAGE_37.DEFAULT_MIDDLE_LEARNING_RATE
DEFAULT_SECOND_DECAY_STEP = _STAGE_37.DEFAULT_SECOND_DECAY_STEP
DEFAULT_PRE_EXTENSION_LEARNING_RATE = _STAGE_37.DEFAULT_FINAL_LEARNING_RATE
DEFAULT_EXTENSION_START_STEP = _STAGE_37.DEFAULT_MAX_ITERS
DEFAULT_EXTENSION_LEARNING_RATE = 3e-5
DEFAULT_TARGET_STEP = 25_000
DEFAULT_EVAL_INTERVAL = _STAGE_37.DEFAULT_EVAL_INTERVAL
DEFAULT_EVAL_ITERS = _STAGE_37.DEFAULT_EVAL_ITERS

DEFAULT_SEED = _STAGE_37.DEFAULT_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_37.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_FIXED_PANEL_SEED = _STAGE_37.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_37.DEFAULT_TRAINING_RNG_SEED
DEFAULT_SAMPLE_SEED = _STAGE_37.DEFAULT_SAMPLE_SEED
DEFAULT_TRAIN_FRACTION = _STAGE_37.DEFAULT_TRAIN_FRACTION

# The Stage 37 CUDA reproduction selected its best at the final evaluation,
# so its saved best checkpoint already sits on the reconstruction boundary.
# No updates are replayed; the boundary audit re-evaluates that exact state.
DEFAULT_SOURCE_STEP = 18_000
DEFAULT_SOURCE_BEST_TRAIN_LOSS = 1.2188
DEFAULT_SOURCE_BEST_VAL_LOSS = 1.4352
DEFAULT_STAGE_37_FINAL_TRAIN_LOSS = 1.2188
DEFAULT_STAGE_37_FINAL_VAL_LOSS = 1.4352
DEFAULT_STAGE_37_REPORTED_DECIMALS = 4
DEFAULT_SOURCE_CHECKPOINT_SHA256 = (
    "b163476632ef4d5954f0fe28dbf3f2c2c67ca4eb922f62762b29b989214b4168"
)
DEFAULT_SOURCE_TORCH_VERSION = "2.14.0+cu130"
DEFAULT_SOURCE_GENERATOR_SHA256 = (
    "5accd4ace34596d50183f50f83eaa846436e240bb691ad7f1ae06de63ab7401d"
)
DEFAULT_REPLAY_GENERATOR_SHA256 = (
    "5accd4ace34596d50183f50f83eaa846436e240bb691ad7f1ae06de63ab7401d"
)

DEFAULT_COMPLETE_DATA_FINGERPRINT = _STAGE_37.DEFAULT_COMPLETE_DATA_FINGERPRINT
DEFAULT_COMPLETE_NUM_CHARACTERS = _STAGE_37.DEFAULT_COMPLETE_NUM_CHARACTERS
DEFAULT_COMPLETE_TRAIN_CHARACTERS = _STAGE_37.DEFAULT_COMPLETE_TRAIN_CHARACTERS
DEFAULT_COMPLETE_VAL_CHARACTERS = _STAGE_37.DEFAULT_COMPLETE_VAL_CHARACTERS
DEFAULT_COMPLETE_VOCAB_SIZE = _STAGE_37.DEFAULT_COMPLETE_VOCAB_SIZE
DEFAULT_PARAMETER_COUNT = _STAGE_37.DEFAULT_PARAMETER_COUNT
DEFAULT_INITIAL_STATE_SHA256 = _STAGE_37.DEFAULT_INITIAL_STATE_SHA256
DEFAULT_TARGETS_PER_UPDATE = _STAGE_37.DEFAULT_TARGETS_PER_UPDATE
DEFAULT_TOTAL_TARGET_CHARACTERS = DEFAULT_TARGET_STEP * DEFAULT_TARGETS_PER_UPDATE
DEFAULT_ADDED_TARGET_CHARACTERS = (
    (DEFAULT_TARGET_STEP - DEFAULT_EXTENSION_START_STEP)
    * DEFAULT_TARGETS_PER_UPDATE
)

CheckpointKind = Literal["best", "reconstructed_18000", "final"]


@dataclass(frozen=True, slots=True)
class ExtendedLearningRateSchedule:
    initial_learning_rate: float = DEFAULT_INITIAL_LEARNING_RATE
    first_decay_step: int = DEFAULT_FIRST_DECAY_STEP
    middle_learning_rate: float = DEFAULT_MIDDLE_LEARNING_RATE
    second_decay_step: int = DEFAULT_SECOND_DECAY_STEP
    pre_extension_learning_rate: float = DEFAULT_PRE_EXTENSION_LEARNING_RATE
    extension_start_step: int = DEFAULT_EXTENSION_START_STEP
    extension_learning_rate: float = DEFAULT_EXTENSION_LEARNING_RATE

    def __post_init__(self) -> None:
        rates = (
            self.initial_learning_rate,
            self.middle_learning_rate,
            self.pre_extension_learning_rate,
            self.extension_learning_rate,
        )
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("all learning rates must be finite and positive")
        if not all(left > right for left, right in zip(rates, rates[1:])):
            raise ValueError("learning rates must strictly decrease by phase")
        steps = (
            self.first_decay_step,
            self.second_decay_step,
            self.extension_start_step,
        )
        if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
            raise ValueError("learning-rate phase boundaries must be integers")
        if not (0 < steps[0] < steps[1] < steps[2]):
            raise ValueError(
                "learning-rate phase boundaries must be strictly increasing"
            )

    def for_update(self, update_index: int) -> float:
        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise ValueError("update_index must be an integer")
        if update_index < 0:
            raise ValueError("update_index must be non-negative")
        if update_index < self.first_decay_step:
            return self.initial_learning_rate
        if update_index < self.second_decay_step:
            return self.middle_learning_rate
        if update_index < self.extension_start_step:
            return self.pre_extension_learning_rate
        return self.extension_learning_rate

    def validate_target_step(self, target_step: int) -> None:
        if isinstance(target_step, bool) or not isinstance(target_step, int):
            raise ValueError("target_step must be an integer")
        if target_step <= self.extension_start_step:
            raise ValueError(
                f"target_step must be above {self.extension_start_step:,}, "
                f"got {target_step:,}"
            )

    def stage_37_schedule(self) -> object:
        return _STAGE_37.LearningRateSchedule(
            initial_learning_rate=self.initial_learning_rate,
            first_decay_step=self.first_decay_step,
            middle_learning_rate=self.middle_learning_rate,
            second_decay_step=self.second_decay_step,
            final_learning_rate=self.pre_extension_learning_rate,
        )

    def as_metadata(self, *, target_step: int) -> list[dict[str, int | float]]:
        self.validate_target_step(target_step)
        return [
            {
                "start_update": 0,
                "end_update_exclusive": self.first_decay_step,
                "learning_rate": self.initial_learning_rate,
            },
            {
                "start_update": self.first_decay_step,
                "end_update_exclusive": self.second_decay_step,
                "learning_rate": self.middle_learning_rate,
            },
            {
                "start_update": self.second_decay_step,
                "end_update_exclusive": self.extension_start_step,
                "learning_rate": self.pre_extension_learning_rate,
            },
            {
                "start_update": self.extension_start_step,
                "end_update_exclusive": target_step,
                "learning_rate": self.extension_learning_rate,
            },
        ]


@dataclass(frozen=True, slots=True)
class Stage37Source:
    path: Path
    sha256: str
    checkpoint: Mapping[str, object]
    step: int
    best_step: int
    best_train_loss: float
    best_val_loss: float
    runtime_device_type: str
    runtime_device_name: str
    torch_version: str
    canonical: bool

    @property
    def replay_updates(self) -> int:
        return DEFAULT_EXTENSION_START_STEP - self.step


@dataclass(frozen=True, slots=True)
class ReplayAudit:
    record: EvaluationRecord
    train_matches_stage_37: bool
    val_matches_stage_37: bool
    source_runtime_matched: bool

    @property
    def verified(self) -> bool:
        return (
            self.train_matches_stage_37
            and self.val_matches_stage_37
            and self.source_runtime_matched
        )


@dataclass(frozen=True, slots=True)
class Stage38Report:
    source: Stage37Source
    training: TrainingResult
    replay_audit: ReplayAudit
    best_train_loss: float
    best_val_loss: float
    best_step: int
    final_batch_generator_sha256: str
    best_checkpoint_path: Path
    best_checkpoint_sha256: str
    replay_checkpoint_path: Path
    replay_checkpoint_sha256: str
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


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint {name} must be an integer")
    return value


def _require_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"checkpoint {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"checkpoint {name} must be finite")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(device: torch.device) -> tuple[str, str]:
    if device.type == "xpu":
        return "xpu", torch.xpu.get_device_name(device.index or 0)
    if device.type == "cuda":
        return "cuda", torch.cuda.get_device_name(device.index or 0)
    return device.type, device.type


def _capture_rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {"cpu": torch.get_rng_state().clone()}
    if device.type == "xpu":
        state["xpu"] = [
            item.cpu().clone() for item in torch.xpu.get_rng_state_all()
        ]
    elif device.type == "cuda":
        state["cuda"] = [
            item.cpu().clone() for item in torch.cuda.get_rng_state_all()
        ]
    return state


def _validate_rng_state(
    state: object,
    *,
    runtime_device_type: str,
) -> Mapping[str, object]:
    rng = _require_mapping(state, name="rng_state")
    cpu_state = rng.get("cpu")
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.numel() == 0:
        raise ValueError("checkpoint rng_state['cpu'] must be a non-empty tensor")
    if runtime_device_type in ("xpu", "cuda"):
        accelerator_state = rng.get(runtime_device_type)
        if (
            not isinstance(accelerator_state, list)
            or not accelerator_state
            or any(
                not isinstance(item, torch.Tensor) or item.numel() == 0
                for item in accelerator_state
            )
        ):
            raise ValueError(
                f"checkpoint rng_state['{runtime_device_type}'] must be a "
                "non-empty tensor list"
            )
    return rng


def _restore_rng_state(state: Mapping[str, object], device: torch.device) -> None:
    cpu_state = state["cpu"]
    assert isinstance(cpu_state, torch.Tensor)
    torch.set_rng_state(cpu_state.cpu())
    if device.type == "xpu":
        accelerator_state = state["xpu"]
        assert isinstance(accelerator_state, list)
        torch.xpu.set_rng_state_all([item.cpu() for item in accelerator_state])
    elif device.type == "cuda":
        accelerator_state = state["cuda"]
        assert isinstance(accelerator_state, list)
        torch.cuda.set_rng_state_all([item.cpu() for item in accelerator_state])


def _optimizer_rates(optimizer_state: Mapping[str, object]) -> set[float]:
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("checkpoint optimizer must contain parameter groups")
    rates: set[float] = set()
    for group in groups:
        if not isinstance(group, Mapping) or "lr" not in group:
            raise ValueError("checkpoint optimizer parameter group has no LR")
        rates.add(_require_finite_float(group["lr"], name="optimizer LR"))
    return rates


def _validate_adamw_state(
    optimizer_state: object,
    *,
    expected_learning_rate: float,
) -> Mapping[str, object]:
    saved = _require_mapping(optimizer_state, name="optimizer_state_dict")
    if _optimizer_rates(saved) != {expected_learning_rate}:
        raise ValueError(
            "checkpoint optimizer learning rate is inconsistent with its step"
        )
    moments = saved.get("state")
    if not isinstance(moments, Mapping) or not moments:
        raise ValueError("checkpoint AdamW state must contain saved moments")
    groups = saved["param_groups"]
    assert isinstance(groups, list)
    parameter_ids: set[object] = set()
    for group in groups:
        assert isinstance(group, Mapping)
        group_parameters = group.get("params")
        if not isinstance(group_parameters, list) or not group_parameters:
            raise ValueError("checkpoint optimizer group must contain parameters")
        parameter_ids.update(group_parameters)
    if set(moments) != parameter_ids:
        missing = tuple(sorted(parameter_ids - set(moments), key=repr))
        extra = tuple(sorted(set(moments) - parameter_ids, key=repr))
        raise ValueError(
            "checkpoint AdamW moments do not cover every parameter: "
            f"missing={missing!r}, extra={extra!r}"
        )
    for parameter_id, state in moments.items():
        entry = _require_mapping(
            state,
            name=f"optimizer state for parameter {parameter_id}",
        )
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value = entry.get(key)
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                raise ValueError(
                    f"checkpoint AdamW state {parameter_id!r} has no {key}"
                )
    return saved


def _reported_value_matches(observed: float, reported: float) -> bool:
    return f"{observed:.{DEFAULT_STAGE_37_REPORTED_DECIMALS}f}" == (
        f"{reported:.{DEFAULT_STAGE_37_REPORTED_DECIMALS}f}"
    )


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    return _STAGE_37.build_model(data, config, n_head, n_layer, device)


def verify_model_masks(model: GPTLanguageModel) -> None:
    _STAGE_37.verify_model_masks(model)


def get_stage_37_batch(
    data: CharacterData,
    split: str,
    config: TrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _STAGE_37.get_stage_36_matched_batch(
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
    return _STAGE_37.evaluate_on_fixed_batches(
        model,
        data,
        config,
        device,
        fixed_panel_seed=fixed_panel_seed,
    )


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _source_training_config(
    config: TrainingConfig,
    schedule: ExtendedLearningRateSchedule,
) -> TrainingConfig:
    return TrainingConfig(
        batch_size=config.batch_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        learning_rate=schedule.initial_learning_rate,
        max_iters=schedule.extension_start_step,
        eval_interval=config.eval_interval,
        eval_iters=config.eval_iters,
        seed=config.seed,
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
        and config.learning_rate == DEFAULT_EXTENSION_LEARNING_RATE
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


def is_canonical_stage_37_source_contract(
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
    """Identify the authenticated source independently of Stage 38's target."""

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
        and schedule.initial_learning_rate == DEFAULT_INITIAL_LEARNING_RATE
        and schedule.first_decay_step == DEFAULT_FIRST_DECAY_STEP
        and schedule.middle_learning_rate == DEFAULT_MIDDLE_LEARNING_RATE
        and schedule.second_decay_step == DEFAULT_SECOND_DECAY_STEP
        and schedule.pre_extension_learning_rate
        == DEFAULT_PRE_EXTENSION_LEARNING_RATE
        and schedule.extension_start_step == DEFAULT_EXTENSION_START_STEP
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
    schedule.validate_target_step(config.max_iters)
    if config.learning_rate != schedule.extension_learning_rate:
        raise ValueError(
            "config learning rate must equal the extension learning rate"
        )
    if n_layer != DEFAULT_N_LAYER:
        raise ValueError(f"Stage 38 requires L={DEFAULT_N_LAYER}, got {n_layer}")
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
            "Stage 38 requires an even batch size for the preserved Stage 36/37 "
            "long-panel sampler"
        )
    if schedule.extension_start_step % config.eval_interval:
        raise ValueError("the 18,000 reconstruction boundary must be evaluated")
    if config.max_iters % config.eval_interval:
        raise ValueError("the target step must align with the evaluation interval")
    for split, source in (("train", data.train_data), ("val", data.val_data)):
        panel_size = 2 * config.block_size
        if len(source) <= panel_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but Stage 38 "
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
    replay_checkpoint_path: Path,
    final_checkpoint_path: Path,
    source_data_path: Path | None,
) -> None:
    paths = {
        "Stage 37 source checkpoint": source_checkpoint_path.resolve(),
        "Stage 38 best checkpoint": best_checkpoint_path.resolve(),
        "Stage 38 replay checkpoint": replay_checkpoint_path.resolve(),
        "Stage 38 final checkpoint": final_checkpoint_path.resolve(),
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


def _validate_optimizer_steps(
    optimizer_state: Mapping[str, object],
    *,
    expected_step: int,
) -> None:
    moments = optimizer_state["state"]
    assert isinstance(moments, Mapping)
    for parameter_id, state in moments.items():
        assert isinstance(state, Mapping)
        step = state["step"]
        assert isinstance(step, torch.Tensor)
        if step.numel() != 1 or float(step.item()) != float(expected_step):
            raise ValueError(
                f"checkpoint AdamW parameter {parameter_id!r} has step "
                f"{step!r}, expected {expected_step}"
            )


def _validate_optimizer_hyperparameters(
    optimizer_state: Mapping[str, object],
) -> None:
    groups = optimizer_state["param_groups"]
    assert isinstance(groups, list)
    expected = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "maximize": False,
    }
    for group in groups:
        assert isinstance(group, Mapping)
        for key, value in expected.items():
            if group.get(key) != value:
                raise ValueError(
                    f"checkpoint AdamW option {key!r} changed: "
                    f"expected {value!r}, got {group.get(key)!r}"
                )


def _expected_training_generator_state(
    data: CharacterData,
    config: TrainingConfig,
    *,
    training_batch_seed: int,
    completed_steps: int,
) -> torch.Tensor:
    _, panel = _STAGE_37.make_sampling_specs(config)
    return _STAGE_37._STAGE_36.expected_training_generator_state(
        data,
        panel,
        training_batch_seed=training_batch_seed,
        completed_steps=completed_steps,
    )


def validate_and_load_stage_37_source(
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
) -> Stage37Source:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 37 source checkpoint not found: {path}")
    before_hash = _sha256_file(path)
    checkpoint = _require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        name="root",
    )
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported Stage 37 checkpoint version")
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("Stage 38 must resume a Stage 37 best checkpoint")
    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    if experiment.get("stage") != _STAGE_37.EXPERIMENT_STAGE:
        raise ValueError("source checkpoint is not a Stage 37 artifact")
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 37 source must have uninterrupted AdamW lineage")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 37 source optimizer provenance must be known")

    step = _require_int(checkpoint.get("step"), name="step")
    best_step = _require_int(checkpoint.get("best_step"), name="best_step")
    if step != best_step:
        raise ValueError("Stage 37 source must store its selected-best state")
    if step > schedule.extension_start_step:
        raise ValueError(
            "Stage 37 source step must not exceed the 18,000 reconstruction "
            "boundary"
        )
    if step % config.eval_interval:
        raise ValueError("Stage 37 source step must align with fixed evaluations")

    source_training = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    source_architecture = _require_mapping(
        checkpoint.get("architecture"),
        name="architecture",
    )
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "layer_contexts": DEFAULT_LAYER_CONTEXTS,
        "residual_dropout": DEFAULT_DROPOUT,
        "attention_mask": "W23,W23,W23,W23,Full,Full",
    }
    for key, expected in expected_architecture.items():
        observed = source_architecture.get(key)
        if key == "layer_contexts" and observed is not None:
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(
                f"Stage 37 source architecture {key!r} mismatch: "
                f"expected {expected!r}, got {observed!r}"
            )
    for key, expected in (
        ("batch_size", config.batch_size),
        ("eval_interval", config.eval_interval),
        ("eval_iters", config.eval_iters),
        ("seed", config.seed),
        ("max_iters", schedule.extension_start_step),
        ("training_batch_seed", training_batch_seed),
        ("fixed_panel_seed", fixed_panel_seed),
        ("training_rng_seed", training_rng_seed),
        ("targets_per_update", config.batch_size * config.block_size),
    ):
        if source_training.get(key) != expected:
            raise ValueError(
                f"Stage 37 source training field {key!r} mismatch: "
                f"expected {expected!r}, got {source_training.get(key)!r}"
            )
    expected_stage_37_schedule = schedule.stage_37_schedule().as_metadata(
        max_iters=schedule.extension_start_step
    )
    if source_training.get("learning_rate_schedule") != expected_stage_37_schedule:
        raise ValueError("Stage 37 source learning-rate schedule changed")
    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("Stage 37 source corpus fingerprint mismatch")

    runtime = _require_mapping(checkpoint.get("runtime"), name="runtime")
    runtime_device_type = runtime.get("device_type")
    runtime_device_name = runtime.get("device_name")
    saved_torch_version = runtime.get("torch_version")
    if not isinstance(runtime_device_type, str):
        raise ValueError("Stage 37 runtime device type is missing")
    if not isinstance(runtime_device_name, str):
        raise ValueError("Stage 37 runtime device name is missing")
    if not isinstance(saved_torch_version, str):
        raise ValueError("Stage 37 torch version is missing")
    current_device_type, current_device_name = _runtime_identity(device)
    if (
        runtime_device_type != current_device_type
        or runtime_device_name != current_device_name
    ):
        raise ValueError(
            "exact Stage 37 replay requires the recorded device/backend: "
            f"saved {runtime_device_type}/{runtime_device_name!r}, current "
            f"{current_device_type}/{current_device_name!r}"
        )
    if saved_torch_version != str(torch.__version__):
        raise ValueError(
            "exact Stage 37 replay requires the recorded torch version: "
            f"saved {saved_torch_version!r}, current {str(torch.__version__)!r}"
        )

    expected_rate = schedule.for_update(max(0, step - 1))
    optimizer_state = _validate_adamw_state(
        checkpoint.get("optimizer_state_dict"),
        expected_learning_rate=expected_rate,
    )
    _validate_optimizer_steps(optimizer_state, expected_step=step)
    _validate_optimizer_hyperparameters(optimizer_state)
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor) or generator_state.numel() == 0:
        raise ValueError(
            "Stage 37 source training-generator state must be a non-empty tensor"
        )
    expected_generator_state = _expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator_state):
        raise ValueError("Stage 37 source batch-generator state is inconsistent")
    _validate_rng_state(
        checkpoint.get("rng_state"),
        runtime_device_type=runtime_device_type,
    )

    source_config = _source_training_config(config, schedule)
    _STAGE_37.validate_stage_37_checkpoint(
        path,
        data,
        source_config,
        n_head,
        n_layer,
        schedule.stage_37_schedule(),
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )

    canonical_source = is_canonical_stage_37_source_contract(
        data,
        config,
        n_head,
        n_layer,
        schedule,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    if canonical_source:
        if step != DEFAULT_SOURCE_STEP:
            raise ValueError(
                f"canonical Stage 37 source must be step {DEFAULT_SOURCE_STEP:,}"
            )
        if before_hash != DEFAULT_SOURCE_CHECKPOINT_SHA256:
            raise ValueError(
                "canonical Stage 37 source hash mismatch: expected "
                f"{DEFAULT_SOURCE_CHECKPOINT_SHA256}, got {before_hash}"
            )
        if saved_torch_version != DEFAULT_SOURCE_TORCH_VERSION:
            raise ValueError("canonical Stage 37 source torch version changed")
        if tensor_sha256(generator_state) != DEFAULT_SOURCE_GENERATOR_SHA256:
            raise ValueError("canonical Stage 37 source generator state changed")

    after_hash = _sha256_file(path)
    if after_hash != before_hash:
        raise RuntimeError("Stage 37 source changed during validation")
    best_train_loss = _require_finite_float(
        checkpoint.get("best_training_loss"),
        name="best_training_loss",
    )
    best_val_loss = _require_finite_float(
        checkpoint.get("best_val_loss"),
        name="best_val_loss",
    )
    return Stage37Source(
        path=path.resolve(),
        sha256=before_hash,
        checkpoint=checkpoint,
        step=step,
        best_step=best_step,
        best_train_loss=best_train_loss,
        best_val_loss=best_val_loss,
        runtime_device_type=runtime_device_type,
        runtime_device_name=runtime_device_name,
        torch_version=saved_torch_version,
        canonical=canonical_source,
    )


def restore_stage_37_source(
    source: Stage37Source,
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
    # Restore last, after model construction and all validation diagnostics,
    # so the next dropout draw is exactly the one following source.step.
    _restore_rng_state(rng_state, device)


def _atomic_save_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(dict(payload), temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@dataclass(slots=True)
class Stage38ValidationCheckpoint(BestValidationCheckpoint):
    """Persist selected, reconstructed-boundary, and terminal full states."""

    schedule: ExtendedLearningRateSchedule | None = None
    target_step: int = 0
    source: Stage37Source | None = None
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
    replay_verified: bool = False

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
            raise RuntimeError("incomplete Stage 38 checkpoint metadata")
        expected_rate = self.schedule.for_update(max(0, step - 1))
        optimizer_state = _require_mapping(
            payload.get("optimizer_state_dict"),
            name="optimizer_state_dict",
        )
        if _optimizer_rates(optimizer_state) != {expected_rate}:
            raise RuntimeError(
                f"optimizer LR at step {step} must be {expected_rate:g}"
            )
        payload["checkpoint_kind"] = checkpoint_kind
        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "head_size": self.config.n_embd // self.n_head,
                "feed_forward_width": 4 * self.config.n_embd,
                "residual_dropout": DEFAULT_DROPOUT,
                "dropout_placement": _STAGE_37._STAGE_36.DROPOUT_PLACEMENT,
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
                "learning_rate": expected_rate,
                "max_iters": self.target_step,
                "target_step": self.target_step,
                "source_checkpoint_step": self.source.step,
                "reconstruction_step": self.schedule.extension_start_step,
                "replayed_updates": (
                    self.schedule.extension_start_step - self.source.step
                ),
                "extension_learning_rate": (
                    self.schedule.extension_learning_rate
                ),
                "learning_rate_schedule": self.schedule.as_metadata(
                    target_step=self.target_step
                ),
                "residual_dropout": DEFAULT_DROPOUT,
                "training_batch_seed": self.training_batch_seed,
                "fixed_panel_seed": self.fixed_panel_seed,
                "training_rng_seed": self.training_rng_seed,
                "targets_per_update": self.targets_per_update,
                "total_target_presentations": (
                    self.targets_per_update * self.target_step
                ),
                "added_target_presentations_after_stage_37": (
                    self.targets_per_update
                    * (self.target_step - self.schedule.extension_start_step)
                ),
            }
        )
        payload["source_checkpoint"] = {
            "stage": _STAGE_37.EXPERIMENT_STAGE,
            "path_name": self.source.path.name,
            "sha256": self.source.sha256,
            "checkpoint_kind": "best",
            "step": self.source.step,
            "best_step": self.source.best_step,
            "best_training_loss": self.source.best_train_loss,
            "best_validation_loss": self.source.best_val_loss,
            "optimizer_state_restored": True,
            "training_generator_restored": True,
            "cpu_rng_restored": True,
            "accelerator_rng_restored": self.device.type in ("xpu", "cuda"),
            "optimizer_restart": False,
            "runtime_device_type": self.source.runtime_device_type,
            "runtime_device_name": self.source.runtime_device_name,
            "torch_version": self.source.torch_version,
        }
        payload["reconstruction"] = {
            "reason": "Stage 37 did not persist its step-18000 endpoint",
            "source_step": self.source.step,
            "target_step": self.schedule.extension_start_step,
            "updates_replayed": self.schedule.extension_start_step - self.source.step,
            "learning_rate": self.schedule.pre_extension_learning_rate,
            "next_update_uses_extension_learning_rate": True,
            "same_runtime_required": True,
            "byte_identity_to_lost_endpoint_provable": False,
            "reported_endpoint_train_loss": DEFAULT_STAGE_37_FINAL_TRAIN_LOSS,
            "reported_endpoint_validation_loss": DEFAULT_STAGE_37_FINAL_VAL_LOSS,
            "reported_decimal_places": DEFAULT_STAGE_37_REPORTED_DECIMALS,
            "replay_verified_at_snapshot": self.replay_verified,
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
            "origin_stage": _STAGE_37.EXPERIMENT_STAGE,
            "kind": "restored_stage_37_complete_shakespeare_state",
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
            "same_model_as_stage_37": True,
            "same_corpus_as_stage_37": True,
            "same_split_as_stage_37": True,
            "same_sampler_as_stage_37": True,
            "same_fixed_evaluation_panel_as_stage_37": True,
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
            "matches_stage_37_source": (
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
            super(Stage38ValidationCheckpoint, self)._payload(step),
            step=step,
            checkpoint_kind="best",
            snapshot_losses=LossEstimate(
                train=self.best_train_loss,
                val=self.best_val_loss,
            ),
        )

    def save_snapshot(
        self,
        path: Path,
        record: EvaluationRecord,
        *,
        checkpoint_kind: Literal["reconstructed_18000", "final"],
    ) -> None:
        payload = self._decorate_payload(
            super(Stage38ValidationCheckpoint, self)._payload(record.step),
            step=record.step,
            checkpoint_kind=checkpoint_kind,
            snapshot_losses=record.losses,
        )
        _atomic_save_payload(path, payload)

    def consider(self, record: EvaluationRecord) -> bool:
        if not math.isfinite(record.losses.val):
            raise ValueError(
                f"validation loss at step {record.step} must be finite"
            )
        if record.losses.val >= self.best_val_loss:
            return False
        self.best_val_loss = record.losses.val
        self.best_step = record.step
        self.best_train_loss = record.losses.train
        self.save(record.step)
        return True


def train_extended(
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
    if start_step < 0 or start_step > schedule.extension_start_step:
        raise ValueError(
            "start_step must not exceed the reconstruction boundary, got "
            f"{start_step}"
        )
    if start_step % config.eval_interval:
        raise ValueError("start_step must align with the evaluation interval")
    schedule.validate_target_step(config.max_iters)
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
        _set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_stage_37_batch(
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
            or completed_step == schedule.extension_start_step
            or completed_step == config.max_iters
        ):
            record(completed_step)

    return TrainingResult(
        initial=initial_record.losses,
        final=history[-1].losses,
        history=tuple(history),
        benchmark=None,
    )


def make_evaluation_callback(
    best: Stage38ValidationCheckpoint,
    schedule: ExtendedLearningRateSchedule,
    source: Stage37Source,
    *,
    replay_checkpoint_path: Path,
    final_checkpoint_path: Path,
) -> tuple[
    Callable[[EvaluationRecord], None],
    list[ReplayAudit],
]:
    replay_audits: list[ReplayAudit] = []

    def record_evaluation(record: EvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        learning_rate = schedule.for_update(max(0, record.step - 1))
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr {learning_rate:g} | "
            f"train {record.losses.train:.4f} | "
            f"val {record.losses.val:.4f} | "
            f"gap {record.losses.val - record.losses.train:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )

        if record.step == schedule.extension_start_step:
            train_matches = (
                not source.canonical
                or _reported_value_matches(
                    record.losses.train,
                    DEFAULT_STAGE_37_FINAL_TRAIN_LOSS,
                )
            )
            val_matches = (
                not source.canonical
                or _reported_value_matches(
                    record.losses.val,
                    DEFAULT_STAGE_37_FINAL_VAL_LOSS,
                )
            )
            audit = ReplayAudit(
                record=record,
                train_matches_stage_37=train_matches,
                val_matches_stage_37=val_matches,
                source_runtime_matched=True,
            )
            if source.canonical and not audit.verified:
                raise RuntimeError(
                    "the reconstructed step-18,000 fixed-panel loss does not "
                    "match Stage 37 at four decimals: expected "
                    f"{DEFAULT_STAGE_37_FINAL_TRAIN_LOSS:.4f} / "
                    f"{DEFAULT_STAGE_37_FINAL_VAL_LOSS:.4f}, got "
                    f"{record.losses.train:.4f} / {record.losses.val:.4f}"
                )
            generator_hash = tensor_sha256(best.training_generator.get_state())
            if (
                source.canonical
                and generator_hash != DEFAULT_REPLAY_GENERATOR_SHA256
            ):
                raise RuntimeError(
                    "the reconstructed step-18,000 batch-generator state "
                    "does not match the canonical Stage 37 stream"
                )
            best.replay_verified = True
            best.save_snapshot(
                replay_checkpoint_path,
                record,
                checkpoint_kind="reconstructed_18000",
            )
            if best.best_step == record.step:
                # consider() ran before the endpoint audit. Refresh a newly
                # selected boundary checkpoint so its provenance records the
                # now-completed replay verification.
                best.save(record.step)
            replay_audits.append(audit)

        if record.step == best.target_step:
            if not best.replay_verified:
                raise RuntimeError("training reached its target without replay audit")
            best.save_snapshot(
                final_checkpoint_path,
                record,
                checkpoint_kind="final",
            )

    return record_evaluation, replay_audits


def validate_stage_38_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: ExtendedLearningRateSchedule,
    source: Stage37Source,
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
        raise ValueError("unsupported Stage 38 checkpoint version")
    if checkpoint.get("checkpoint_kind") != expected_kind:
        raise ValueError(
            f"expected a {expected_kind!r} Stage 38 checkpoint, got "
            f"{checkpoint.get('checkpoint_kind')!r}"
        )
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 38 checkpoint must not restart AdamW")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 38 optimizer provenance must remain known")
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
        raise ValueError("Stage 38 best checkpoint must store its selected step")
    if (
        expected_kind == "reconstructed_18000"
        and step != schedule.extension_start_step
    ):
        raise ValueError("Stage 38 replay checkpoint is not at step 18,000")
    if expected_kind == "final" and step != config.max_iters:
        raise ValueError("Stage 38 final checkpoint is not at the target step")
    if not (source.step <= best_step <= step):
        raise ValueError("Stage 38 best step is outside the saved trajectory")

    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    if experiment.get("stage") != EXPERIMENT_STAGE:
        raise ValueError("checkpoint is not a Stage 38 artifact")
    for key, expected in (
        ("same_model_as_stage_37", True),
        ("same_corpus_as_stage_37", True),
        ("same_split_as_stage_37", True),
        ("same_sampler_as_stage_37", True),
        ("same_fixed_evaluation_panel_as_stage_37", True),
        ("optimizer_state_preserved", True),
        ("optimizer_restarted", False),
        ("rng_state_preserved", True),
        ("architecture_frozen", True),
        ("trained_branches", 1),
    ):
        if experiment.get(key) != expected:
            raise ValueError(f"Stage 38 experiment field {key!r} changed")

    source_metadata = _require_mapping(
        checkpoint.get("source_checkpoint"),
        name="source_checkpoint",
    )
    for key, expected in (
        ("stage", _STAGE_37.EXPERIMENT_STAGE),
        ("sha256", source.sha256),
        ("checkpoint_kind", "best"),
        ("step", source.step),
        ("best_step", source.best_step),
        ("optimizer_state_restored", True),
        ("training_generator_restored", True),
        ("cpu_rng_restored", True),
        ("optimizer_restart", False),
        ("runtime_device_type", source.runtime_device_type),
        ("runtime_device_name", source.runtime_device_name),
        ("torch_version", source.torch_version),
    ):
        if source_metadata.get(key) != expected:
            raise ValueError(f"Stage 38 source field {key!r} mismatch")

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
                f"Stage 38 architecture {key!r} mismatch: expected "
                f"{expected!r}, got {observed!r}"
            )

    training = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    expected_training = {
        "batch_size": config.batch_size,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": config.max_iters,
        "target_step": config.max_iters,
        "source_checkpoint_step": source.step,
        "reconstruction_step": schedule.extension_start_step,
        "replayed_updates": schedule.extension_start_step - source.step,
        "extension_learning_rate": schedule.extension_learning_rate,
        "learning_rate_schedule": schedule.as_metadata(
            target_step=config.max_iters
        ),
        "training_batch_seed": training_batch_seed,
        "fixed_panel_seed": fixed_panel_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Stage 38 training field {key!r} mismatch")
    expected_rate = schedule.for_update(max(0, step - 1))
    if training.get("learning_rate") != expected_rate:
        raise ValueError("Stage 38 snapshot learning rate is inconsistent")

    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("Stage 38 checkpoint corpus fingerprint mismatch")
    corpus = _require_mapping(checkpoint.get("corpus"), name="corpus")
    for key, expected in (
        ("data_fingerprint", fingerprint_data(data)),
        ("decoded_characters", data.num_characters),
        ("train_characters", len(data.train_data)),
        ("validation_characters", len(data.val_data)),
        ("vocabulary_size", data.vocabulary.size),
    ):
        if corpus.get(key) != expected:
            raise ValueError(f"Stage 38 corpus field {key!r} mismatch")

    runtime = _require_mapping(checkpoint.get("runtime"), name="runtime")
    runtime_type = runtime.get("device_type")
    runtime_name = runtime.get("device_name")
    if not isinstance(runtime_type, str):
        raise ValueError("Stage 38 runtime device type is missing")
    if not isinstance(runtime_name, str):
        raise ValueError("Stage 38 runtime device name is missing")
    if (
        runtime_type != source.runtime_device_type
        or runtime_name != source.runtime_device_name
    ):
        raise ValueError("Stage 38 runtime does not match the Stage 37 source")
    if runtime.get("matches_stage_37_source") is not True:
        raise ValueError("Stage 38 runtime must match the Stage 37 source")
    if runtime.get("torch_version") != source.torch_version:
        raise ValueError("Stage 38 torch version changed during continuation")
    _validate_rng_state(
        checkpoint.get("rng_state"),
        runtime_device_type=runtime_type,
    )

    optimizer_state = _validate_adamw_state(
        checkpoint.get("optimizer_state_dict"),
        expected_learning_rate=expected_rate,
    )
    _validate_optimizer_steps(optimizer_state, expected_step=step)
    _validate_optimizer_hyperparameters(optimizer_state)
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 38 training-generator state must be a tensor")
    expected_generator_state = _expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator_state):
        raise ValueError("Stage 38 batch-generator state is inconsistent")
    if (
        source.canonical
        and expected_kind == "reconstructed_18000"
        and tensor_sha256(generator_state) != DEFAULT_REPLAY_GENERATOR_SHA256
    ):
        raise ValueError("canonical Stage 38 replay generator hash changed")

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
            raise ValueError("Stage 38 best snapshot losses are inconsistent")

    reconstruction = _require_mapping(
        checkpoint.get("reconstruction"),
        name="reconstruction",
    )
    if reconstruction.get("byte_identity_to_lost_endpoint_provable") is not False:
        raise ValueError("Stage 38 must preserve the replay identity caveat")
    if expected_kind in ("reconstructed_18000", "final"):
        if reconstruction.get("replay_verified_at_snapshot") is not True:
            raise ValueError("Stage 38 replay was not verified before snapshot")

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
        raise ValueError("Stage 38 checkpoint parameter count mismatch")
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
    replay_checkpoint_path: Path,
    final_checkpoint_path: Path,
    *,
    source_data_path: Path | None,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> Stage38Report:
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
        replay_checkpoint_path=replay_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
        source_data_path=source_data_path,
    )
    source = validate_and_load_stage_37_source(
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
        lr=schedule.for_update(max(0, source.step - 1)),
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    restore_stage_37_source(
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
        raise RuntimeError("canonical Stage 38 parameter count changed")

    source_corpus = _require_mapping(
        source.checkpoint.get("corpus"),
        name="source corpus",
    )
    source_initialization = _require_mapping(
        source.checkpoint.get("initialization"),
        name="source initialization",
    )
    initial_state_sha256 = source_initialization.get("state_sha256")
    if not isinstance(initial_state_sha256, str):
        raise ValueError("Stage 37 initialization hash is missing")
    best = Stage38ValidationCheckpoint(
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
    # Materialize the inherited Stage 37 winner immediately. The Stage 38 best
    # artifact therefore exists even when the low-LR extension never improves.
    best.save(source.step)
    callback, replay_audits = make_evaluation_callback(
        best,
        schedule,
        source,
        replay_checkpoint_path=replay_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
    )

    print("\nComplete Shakespeare continuation (the only Stage 38 branch)")
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
        f"  reconstruct to:         {schedule.extension_start_step:,} "
        f"at lr {schedule.pre_extension_learning_rate:g}"
    )
    print(
        f"  extend to:              {config.max_iters:,} "
        f"at lr {schedule.extension_learning_rate:g}"
    )
    print("  optimizer lineage:      restored; no restart")
    print("  batch/dropout streams:  restored")

    result = train_extended(
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
    if len(replay_audits) != 1:
        raise RuntimeError("Stage 38 did not perform exactly one replay audit")
    replay_audit = replay_audits[0]
    assert best.best_step is not None
    assert best.best_train_loss is not None

    source_hash_after_training = _sha256_file(source.path)
    if source_hash_after_training != source_hash_before_training:
        raise RuntimeError("Stage 37 source checkpoint changed during Stage 38")
    for path, kind in (
        (best_checkpoint_path, "best"),
        (replay_checkpoint_path, "reconstructed_18000"),
        (final_checkpoint_path, "final"),
    ):
        validate_stage_38_checkpoint(
            path,
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
        name="selected Stage 38 checkpoint",
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
    report = Stage38Report(
        source=source,
        training=result,
        replay_audit=replay_audit,
        best_train_loss=best.best_train_loss,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        final_batch_generator_sha256=tensor_sha256(
            training_generator.get_state()
        ),
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_sha256=checkpoint_sha256(best_checkpoint_path),
        replay_checkpoint_path=replay_checkpoint_path,
        replay_checkpoint_sha256=checkpoint_sha256(replay_checkpoint_path),
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


def print_final_summary(report: Stage38Report) -> None:
    print("\nStage 38 summary")
    print(
        f"  Stage 37 source best:       train "
        f"{report.source.best_train_loss:.4f} / val "
        f"{report.source.best_val_loss:.4f} @ {report.source.step:,}"
    )
    print(
        f"  reconstructed Stage 37 end: train "
        f"{report.replay_audit.record.losses.train:.4f} / val "
        f"{report.replay_audit.record.losses.val:.4f} @ "
        f"{report.replay_audit.record.step:,}"
    )
    print(f"  replay audit verified:      {report.replay_audit.verified}")
    print(
        f"  extended selected:          train {report.best_train_loss:.4f} / "
        f"val {report.best_val_loss:.4f} @ {report.best_step:,}"
    )
    print(
        f"  extended final:             train "
        f"{report.training.final.train:.4f} / val "
        f"{report.training.final.val:.4f} @ {report.final_step:,}"
    )
    print(f"  final generalization gap:   {report.generalization_gap:.4f}")
    print(
        f"  val reversal from best:     {report.validation_reversal:+.4f}"
    )
    print(f"  best checkpoint:            {report.best_checkpoint_path}")
    print(f"  best SHA-256:               {report.best_checkpoint_sha256}")
    print(f"  reconstructed checkpoint:   {report.replay_checkpoint_path}")
    print(f"  reconstructed SHA-256:      {report.replay_checkpoint_sha256}")
    print(f"  final checkpoint:           {report.final_checkpoint_path}")
    print(f"  final SHA-256:              {report.final_checkpoint_sha256}")
    print("\nComplete Shakespeare selected-best generated text")
    print(report.sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 38: restore the authenticated Stage 37 Complete Shakespeare "
            "best state, reconstruct step 18,000 at 1e-4, then extend the same "
            "model with a fourth 3e-5 learning-rate phase."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_COMPLETE_DATA_PATH)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINT_PATH,
    )
    parser.add_argument("--target-step", type=int, default=DEFAULT_TARGET_STEP)
    parser.add_argument(
        "--extension-learning-rate",
        type=float,
        default=DEFAULT_EXTENSION_LEARNING_RATE,
    )
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument(
        "--best-checkpoint-path",
        type=Path,
        default=DEFAULT_BEST_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--replay-checkpoint-path",
        type=Path,
        default=DEFAULT_REPLAY_CHECKPOINT_PATH,
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
        / f"stage_38_complete_shakespeare_step_{target_step}_final_checkpoint.pt"
    )


def main() -> None:
    args = parse_args()
    schedule = ExtendedLearningRateSchedule(
        extension_learning_rate=args.extension_learning_rate,
    )
    config = TrainingConfig(
        batch_size=DEFAULT_BATCH_SIZE,
        block_size=DEFAULT_BLOCK_SIZE,
        n_embd=DEFAULT_N_EMBD,
        learning_rate=schedule.extension_learning_rate,
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
    print(
        f"Train / validation: {len(data.train_data):,} / {len(data.val_data):,}"
    )
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Protocol: restore {DEFAULT_SOURCE_STEP:,}, reconstruct "
        f"{DEFAULT_EXTENSION_START_STEP:,}, extend to {config.max_iters:,}"
    )
    print(
        f"Fourth LR phase: [{DEFAULT_EXTENSION_START_STEP:,},"
        f"{config.max_iters:,}) at {schedule.extension_learning_rate:g}"
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
        args.replay_checkpoint_path,
        final_checkpoint_path,
        source_data_path=args.data,
        training_batch_seed=DEFAULT_TRAINING_BATCH_SEED,
        fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED,
        training_rng_seed=DEFAULT_TRAINING_RNG_SEED,
    )
    print_final_summary(report)


if __name__ == "__main__":
    main()
