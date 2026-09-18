from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

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


# Stage 24 keeps the Stage 22 C=128, p=0.10 champion as a pinned read-only
# control and makes transformer depth the only architectural treatment.  The
# established evaluation and statistical helpers remain shared, while this
# module owns depth-specific initialization, benchmarking, and provenance.
_STAGE_23_PATH = Path(__file__).with_name("23_dropout_strength.py")
_STAGE_23_MODULE_NAME = "stage_23_dropout_strength_for_stage_24"
_STAGE_23_SPEC = importlib.util.spec_from_file_location(
    _STAGE_23_MODULE_NAME,
    _STAGE_23_PATH,
)
assert _STAGE_23_SPEC is not None and _STAGE_23_SPEC.loader is not None
_STAGE_23 = importlib.util.module_from_spec(_STAGE_23_SPEC)
sys.modules[_STAGE_23_MODULE_NAME] = _STAGE_23
_STAGE_23_SPEC.loader.exec_module(_STAGE_23)

GPTLanguageModel = _STAGE_23.GPTLanguageModel
LearningRateSchedule = _STAGE_23.LearningRateSchedule
PositionLosses = _STAGE_23.PositionLosses
PositionLossEstimate = _STAGE_23.PositionLossEstimate
EvaluationRecord = _STAGE_23.EvaluationRecord
TrainingResult = _STAGE_23.TrainingResult
PreciseBranchResult = _STAGE_23.PreciseBranchResult
PreciseValidationReport = _STAGE_23.PreciseValidationReport
ReferenceCheckpoint = _STAGE_23.ReferenceCheckpoint
BestValidationCheckpoint = _STAGE_23._STAGE_22._STAGE_21.BestValidationCheckpoint

checkpoint_sha256 = _STAGE_23.checkpoint_sha256
clear_accelerator_cache = _STAGE_23.clear_accelerator_cache
count_parameters = _STAGE_23.count_parameters
estimate_validation_precise = _STAGE_23.estimate_validation_precise
evaluate_on_fixed_batches = _STAGE_23.evaluate_on_fixed_batches
expected_parameter_count = _STAGE_23.expected_parameter_count
expected_training_generator_state = _STAGE_23.expected_training_generator_state
fingerprint_data = _STAGE_23.fingerprint_data
generate_from_final_model = _STAGE_23.generate_from_final_model
inspect_attention_shape = _STAGE_23.inspect_attention_shape
make_position_bands = _STAGE_23.make_position_bands
set_optimizer_learning_rate = (
    _STAGE_23._STAGE_22._STAGE_21._STAGE_20.set_optimizer_learning_rate
)
tensor_mapping_sha256 = _STAGE_23.tensor_mapping_sha256
tensor_sha256 = _STAGE_23.tensor_sha256
train_with_schedule = _STAGE_23.train_with_schedule

_assert_position_losses_close = (
    _STAGE_23._STAGE_22._STAGE_21._assert_position_losses_close
)
_clone_state = _STAGE_23._STAGE_22._STAGE_21._clone_state
_is_evaluation_step = _STAGE_23._STAGE_22._STAGE_21._is_evaluation_step
_load_and_measure_fixed_control = _STAGE_23._load_and_measure_fixed_control
_measure_precise_checkpoint = _STAGE_23._measure_precise_checkpoint
_paired_delta = _STAGE_23._paired_delta
_position_statistics = _STAGE_23._position_statistics
_require_mapping = _STAGE_23._require_mapping
_serialize_position_losses = _STAGE_23._STAGE_22._STAGE_21._serialize_position_losses
_validate_non_negative_integer = _STAGE_23._validate_non_negative_integer
_validate_serialized_losses = _STAGE_23._STAGE_22._STAGE_21._validate_serialized_losses


CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_22_dropout_010_best_checkpoint.pt"
)
DEFAULT_TREATMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_24_depth_6_best_checkpoint.pt"
)
DEFAULT_CONTROL_CHECKPOINT_SHA256 = (
    "8811ba1c355c69b880c6df26b2be0b6daf205f95466a2604860ce53b56b20284"
)
DEFAULT_CONTROL_BEST_STEP = 15_500
DEFAULT_CONTROL_INITIAL_STATE_SHA256 = _STAGE_23.DEFAULT_INITIAL_STATE_SHA256
DEFAULT_TREATMENT_INITIAL_STATE_SHA256 = (
    "b4bf85310a542f248a51a8eb8e290bbb9274adcce6a2d9898f89cf2e78665a71"
)

DEFAULT_BATCH_SIZE = _STAGE_23.DEFAULT_BATCH_SIZE
DEFAULT_BLOCK_SIZE = _STAGE_23.DEFAULT_BLOCK_SIZE
DEFAULT_N_EMBD = _STAGE_23.DEFAULT_N_EMBD
DEFAULT_N_HEAD = _STAGE_23.DEFAULT_N_HEAD
DEFAULT_CONTROL_N_LAYER = 4
DEFAULT_TREATMENT_N_LAYER = 6
DEFAULT_MAX_ITERS = _STAGE_23.DEFAULT_MAX_ITERS
DEFAULT_INITIAL_LEARNING_RATE = _STAGE_23.DEFAULT_INITIAL_LEARNING_RATE
DEFAULT_FIRST_DECAY_STEP = _STAGE_23.DEFAULT_FIRST_DECAY_STEP
DEFAULT_MIDDLE_LEARNING_RATE = _STAGE_23.DEFAULT_MIDDLE_LEARNING_RATE
DEFAULT_SECOND_DECAY_STEP = _STAGE_23.DEFAULT_SECOND_DECAY_STEP
DEFAULT_FINAL_LEARNING_RATE = _STAGE_23.DEFAULT_FINAL_LEARNING_RATE
DEFAULT_PRECISE_EVAL_ITERS = _STAGE_23.DEFAULT_PRECISE_EVAL_ITERS
DEFAULT_DROPOUT = 0.10
DROPOUT_PLACEMENT = _STAGE_23.DROPOUT_PLACEMENT
INITIALIZATION_KIND = "architecture_native_same_seed"


@dataclass(frozen=True, slots=True)
class DepthSpec:
    """One transformer-depth branch with otherwise shared architecture."""

    name: str
    n_layer: int
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch name must not be empty")
        if (
            isinstance(self.n_layer, bool)
            or not isinstance(self.n_layer, int)
            or self.n_layer <= 0
        ):
            raise ValueError(
                f"n_layer must be a positive integer, got {self.n_layer!r}"
            )


@dataclass(frozen=True, slots=True)
class PairedInitialStates:
    """Architecture-native L=control/L=treatment states from one seed."""

    control: dict[str, torch.Tensor]
    treatment: dict[str, torch.Tensor]
    control_state_sha256: str
    treatment_state_sha256: str
    control_parameter_count: int
    treatment_parameter_count: int
    initialization_seed: int


@dataclass(frozen=True, slots=True)
class ControlReport:
    spec: DepthSpec
    reference: ReferenceCheckpoint
    losses_at_best: PositionLossEstimate

    @property
    def best_step(self) -> int:
        return self.reference.checkpoint_step

    @property
    def best_val_loss(self) -> float:
        return self.reference.fixed_panel_loss

    @property
    def generalization_gap_at_best(self) -> float:
        return self.losses_at_best.val.overall - self.losses_at_best.train.overall


@dataclass(frozen=True, slots=True)
class TreatmentReport:
    spec: DepthSpec
    parameter_count: int
    initial_state_sha256: str
    training: TrainingResult
    best_training_losses: PositionLosses
    best_validation_losses: PositionLosses
    best_val_loss: float
    best_step: int
    final_batch_generator_sha256: str
    attention_shape: tuple[int, ...]
    sample: str

    @property
    def generalization_gap_at_best(self) -> float:
        return self.best_validation_losses.overall - self.best_training_losses.overall

    @property
    def final_generalization_gap(self) -> float:
        return self.training.final.val.overall - self.training.final.train.overall


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    initialization: PairedInitialStates
    initialization_seed: int
    training_batch_seed: int
    training_rng_seed: int
    schedule: LearningRateSchedule
    targets_per_update: int
    bands: tuple[tuple[int, int], ...]
    control_initial_losses: PositionLossEstimate
    treatment_initial_losses: PositionLossEstimate
    control: ControlReport
    treatment: TreatmentReport
    control_benchmark: BenchmarkStats | None
    treatment_benchmark: BenchmarkStats | None
    treatment_checkpoint_sha256: str

    @property
    def best_val_delta(self) -> float:
        return self.treatment.best_val_loss - self.control.best_val_loss

    @property
    def best_winner(self) -> str:
        if self.best_val_delta < 0:
            return self.treatment.spec.name
        if self.best_val_delta > 0:
            return self.control.spec.name
        return "tie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 24: compare transformer depth L=4 with L=6 at C=128 and "
            "residual dropout p=0.10, reusing the pinned Stage 22 L=4 best "
            "checkpoint and training only the architecture-native L=6 branch."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument(
        "--control-layers",
        "--control-n-layer",
        dest="control_layers",
        type=int,
        default=DEFAULT_CONTROL_N_LAYER,
    )
    parser.add_argument(
        "--treatment-layers",
        "--treatment-n-layer",
        dest="treatment_layers",
        type=int,
        default=DEFAULT_TREATMENT_N_LAYER,
    )
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
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Stage 24 protocol requires residual dropout to remain exactly 0.10.",
    )
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument(
        "--training-batch-seed",
        type=int,
        default=None,
        help="Defaults to seed + 1, matching the Stage 22 batch stream.",
    )
    parser.add_argument(
        "--training-rng-seed",
        type=int,
        default=None,
        help="Defaults to seed + 3 and is reset immediately before update 0.",
    )
    parser.add_argument(
        "--precise-eval-iters",
        type=int,
        default=DEFAULT_PRECISE_EVAL_ITERS,
        help="Fresh paired validation batches after training; 0 skips.",
    )
    parser.add_argument(
        "--precise-eval-seed",
        type=int,
        default=None,
        help="Defaults to seed + 7, matching the Stage 22 fresh panel.",
    )
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument(
        "--control-checkpoint-path",
        type=Path,
        default=DEFAULT_CONTROL_CHECKPOINT_PATH,
        help="Read-only Stage 22 C=128, L=4, p=0.10 best checkpoint.",
    )
    parser.add_argument(
        "--control-checkpoint-sha256",
        type=str,
        default=None,
        help="Required with a custom control path; the canonical path is pinned.",
    )
    parser.add_argument(
        "--control-checkpoint-step",
        type=int,
        default=None,
        help="Required with a custom control path; the canonical path is pinned.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_TREATMENT_CHECKPOINT_PATH,
        help="Output path for the Stage 24 L=6 best checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _pin_for_default_path(path: Path) -> tuple[str | None, int | None]:
    if path.resolve() == DEFAULT_CONTROL_CHECKPOINT_PATH.resolve():
        return DEFAULT_CONTROL_CHECKPOINT_SHA256, DEFAULT_CONTROL_BEST_STEP
    return None, None


def resolve_control_pin(
    path: Path,
    explicit_sha256: str | None,
    explicit_step: int | None,
) -> tuple[str, int]:
    default_sha256, default_step = _pin_for_default_path(path)
    if explicit_sha256 is None and explicit_step is None:
        if default_sha256 is None or default_step is None:
            raise ValueError(
                "a custom Stage 22 control requires both "
                "--control-checkpoint-sha256 and --control-checkpoint-step"
            )
        return default_sha256, default_step
    if explicit_sha256 is None or explicit_step is None:
        raise ValueError(
            "control checkpoint SHA-256 and step must be provided together"
        )
    normalized_sha256 = explicit_sha256.lower()
    if len(normalized_sha256) != 64:
        raise ValueError("control checkpoint SHA-256 must contain 64 hex digits")
    try:
        int(normalized_sha256, 16)
    except ValueError as error:
        raise ValueError(
            "control checkpoint SHA-256 must contain 64 hex digits"
        ) from error
    _validate_non_negative_integer(explicit_step, name="control checkpoint step")
    if default_sha256 is not None and (
        normalized_sha256 != default_sha256 or explicit_step != default_step
    ):
        raise ValueError(
            "explicit control pin does not match the canonical Stage 22 pin"
        )
    return normalized_sha256, explicit_step


def validate_depth_protocol(
    config: TrainingConfig,
    control: DepthSpec,
    treatment: DepthSpec,
    n_head: int,
    dropout: float,
    schedule: LearningRateSchedule,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    if control.name == treatment.name:
        raise ValueError("control and treatment names must differ")
    if treatment.n_layer != control.n_layer + 2:
        raise ValueError(
            "Stage 24 requires treatment depth to be exactly two layers "
            "greater than control depth"
        )
    if dropout != DEFAULT_DROPOUT:
        raise ValueError("Stage 24 residual dropout must be exactly 0.10")
    if isinstance(n_head, bool) or not isinstance(n_head, int) or n_head <= 0:
        raise ValueError(f"n_head must be a positive integer, got {n_head!r}")
    if config.n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if config.block_size % 4 != 0:
        raise ValueError(
            "Stage 24 block_size must be divisible by four for position bands"
        )
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError("config learning rate must equal the schedule initial LR")
    schedule.validate_target_step(config.max_iters)
    _STAGE_23._STAGE_22._STAGE_21.validate_distinct_paths(
        treatment.checkpoint_path,
        control.checkpoint_path,
    )
    return config.batch_size * config.block_size, make_position_bands(config.block_size)


def _build_model(
    data: CharacterData,
    config: TrainingConfig,
    spec: DepthSpec,
    n_head: int,
    dropout: float = DEFAULT_DROPOUT,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=spec.n_layer,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model


def create_paired_initial_states(
    data: CharacterData,
    config: TrainingConfig,
    control: DepthSpec,
    treatment: DepthSpec,
    n_head: int,
    dropout: float = DEFAULT_DROPOUT,
) -> PairedInitialStates:
    """Recreate each architecture's native initialization from one seed.

    The deeper model is not made by copying or extending trained control
    weights.
    Resetting the seed before each constructor makes both initializations
    deterministic while preserving each architecture's normal construction.
    Their complete states and parameter counts are intentionally different.
    """

    states: list[dict[str, torch.Tensor]] = []
    counts: list[int] = []
    for spec in (control, treatment):
        seed_everything(config.seed)
        model = _build_model(data, config, spec, n_head, dropout)
        states.append(_clone_state(model))
        counts.append(count_parameters(model))
        del model
    hashes = tuple(tensor_mapping_sha256(state) for state in states)
    if hashes[0] == hashes[1] or counts[0] == counts[1]:
        raise RuntimeError("different depths must have distinct states and counts")
    is_canonical_architecture = (
        data.vocabulary.size == 65
        and config.block_size == DEFAULT_BLOCK_SIZE
        and config.n_embd == DEFAULT_N_EMBD
        and config.seed == 1_337
        and n_head == DEFAULT_N_HEAD
        and control.n_layer == DEFAULT_CONTROL_N_LAYER
        and treatment.n_layer == DEFAULT_TREATMENT_N_LAYER
        and dropout == DEFAULT_DROPOUT
    )
    if is_canonical_architecture and hashes != (
        DEFAULT_CONTROL_INITIAL_STATE_SHA256,
        DEFAULT_TREATMENT_INITIAL_STATE_SHA256,
    ):
        raise RuntimeError("canonical Stage 24 initialization hashes changed")
    return PairedInitialStates(
        control=states[0],
        treatment=states[1],
        control_state_sha256=hashes[0],
        treatment_state_sha256=hashes[1],
        control_parameter_count=counts[0],
        treatment_parameter_count=counts[1],
        initialization_seed=config.seed,
    )


@dataclass(slots=True)
class DepthValidationCheckpoint(BestValidationCheckpoint):
    branch_spec: DepthSpec | None = None
    control_spec: DepthSpec | None = None
    treatment_spec: DepthSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    dropout: float = DEFAULT_DROPOUT
    initialization_seed: int = 0
    branch_initial_state_sha256: str = ""
    control_initial_state_sha256: str = ""
    treatment_initial_state_sha256: str = ""
    control_parameter_count: int = 0
    treatment_parameter_count: int = 0
    training_batch_seed: int = 0
    training_rng_seed: int = 0
    control_checkpoint_sha256: str = ""
    control_checkpoint_step: int = 0
    attention_shape: tuple[int, ...] = ()
    bands: tuple[tuple[int, int], ...] = ()
    best_training_losses: PositionLosses | None = None
    best_validation_losses: PositionLosses | None = None

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(DepthValidationCheckpoint, self)._payload(step)
        if (
            self.branch_spec is None
            or self.control_spec is None
            or self.treatment_spec is None
            or self.schedule is None
            or self.best_training_losses is None
            or self.best_validation_losses is None
        ):
            raise RuntimeError("incomplete Stage 24 checkpoint metadata")
        rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(rates) != 1:
            raise RuntimeError("all optimizer groups must share one LR")
        learning_rate = rates.pop()
        expected_rate = self.schedule.for_update(max(0, step - 1))
        if learning_rate != expected_rate:
            raise RuntimeError(
                f"optimizer LR at step {step} must be {expected_rate:g}, "
                f"got {learning_rate:g}"
            )
        schedule_metadata = self.schedule.as_metadata(max_iters=self.target_step)
        payload["checkpoint_kind"] = "best"
        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "head_size": self.config.n_embd // self.n_head,
                "feed_forward_width": 4 * self.config.n_embd,
                "residual_dropout": self.dropout,
                "dropout_placement": DROPOUT_PLACEMENT,
                "attention_mask": "full_causal",
                "attention_mask_scope": "per_layer",
                "max_context": None,
                "parameter_count": count_parameters(self.model),
            }
        )
        group = self.optimizer.param_groups[0]
        training = payload["training_config"]
        assert isinstance(training, dict)
        training.update(
            {
                "learning_rate": learning_rate,
                "max_iters": self.target_step,
                "learning_rate_schedule": schedule_metadata,
                "residual_dropout": self.dropout,
                "training_batch_seed": self.training_batch_seed,
                "training_rng_seed": self.training_rng_seed,
                "targets_per_update": self.config.batch_size * self.config.block_size,
                "total_target_characters": (
                    self.config.batch_size * self.config.block_size * self.target_step
                ),
                "optimizer": "AdamW",
                "optimizer_betas": tuple(group["betas"]),
                "optimizer_eps": float(group["eps"]),
                "optimizer_weight_decay": float(group["weight_decay"]),
            }
        )
        payload["initialization"] = {
            "kind": INITIALIZATION_KIND,
            "seed": self.initialization_seed,
            "branch_state_sha256": self.branch_initial_state_sha256,
            "control_state_sha256": self.control_initial_state_sha256,
            "treatment_state_sha256": self.treatment_initial_state_sha256,
            "identical_initialization_seed": True,
            "identical_complete_model_state": False,
            "architecture_native": True,
        }
        payload["experiment"] = {
            "stage": 24,
            "branch": self.branch_spec.name,
            "from_scratch": True,
            "comparison_variable": "transformer_depth",
            "branch_n_layer": self.branch_spec.n_layer,
            "control_n_layer": self.control_spec.n_layer,
            "treatment_n_layer": self.treatment_spec.n_layer,
            "batch_size": self.config.batch_size,
            "block_size": self.config.block_size,
            "n_embd": self.config.n_embd,
            "n_head": self.n_head,
            "head_size": self.config.n_embd // self.n_head,
            "feed_forward_width": 4 * self.config.n_embd,
            "residual_dropout": self.dropout,
            "dropout_placement": DROPOUT_PLACEMENT,
            "identical_parameter_count": False,
            "identical_learned_initialization": False,
            "identical_initialization_seed": True,
            "identical_batch_geometry": True,
            "identical_targets_per_update": True,
            "identical_training_batches": True,
            "identical_fixed_training_batches": True,
            "identical_fixed_validation_batches": True,
            "identical_attention_shapes": True,
            "identical_dropout_masks": False,
            "same_optimizer_configuration": True,
            "same_update_count": True,
            "same_learning_rate_schedule": True,
            "fresh_paired_validation_protocol": True,
            "reused_control": True,
            "control_stage": 22,
            "control_branch": "dropout-p010",
            "control_checkpoint_sha256": self.control_checkpoint_sha256,
            "control_checkpoint_step": self.control_checkpoint_step,
            "control_parameter_count": self.control_parameter_count,
            "treatment_parameter_count": self.treatment_parameter_count,
            "initialization_seed": self.initialization_seed,
            "control_initial_state_sha256": self.control_initial_state_sha256,
            "treatment_initial_state_sha256": self.treatment_initial_state_sha256,
            "training_batch_seed": self.training_batch_seed,
            "training_rng_seed": self.training_rng_seed,
            "targets_per_update": self.config.batch_size * self.config.block_size,
            "total_target_characters": (
                self.config.batch_size * self.config.block_size * self.target_step
            ),
            "learning_rate_schedule": schedule_metadata,
            "position_bands": self.bands,
            "attention_shape": self.attention_shape,
            "attention_shape_per_layer": self.attention_shape,
        }
        payload["evaluation_protocol"] = {
            "fixed_panel_seed": self.config.seed + 2,
            "model_mode": "eval",
            "dropout_enabled": False,
            "gradients_enabled": False,
            "split_order": ("train", "val"),
            "batches_per_split": self.config.eval_iters,
            "reported_train_loss": "fixed_training_panel_eval_mode",
            "reported_validation_loss": "fixed_validation_panel_eval_mode",
        }
        payload["best_training_losses"] = _serialize_position_losses(
            self.best_training_losses,
            self.bands,
        )
        payload["best_validation_losses"] = _serialize_position_losses(
            self.best_validation_losses,
            self.bands,
        )
        return payload

    def consider(self, record: EvaluationRecord) -> bool:
        val_loss = record.losses.val.overall
        if not math.isfinite(val_loss):
            raise ValueError(f"validation loss at step {record.step} must be finite")
        if val_loss >= self.best_val_loss:
            return False
        self.best_val_loss = val_loss
        self.best_step = record.step
        self.best_training_losses = record.losses.train
        self.best_validation_losses = record.losses.val
        self.save(record.step)
        return True


def validate_stage_24_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    spec: DepthSpec,
    control: DepthSpec,
    treatment: DepthSpec,
    n_head: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    control_checkpoint_sha256: str,
    control_checkpoint_step: int,
    dropout: float = DEFAULT_DROPOUT,
) -> Mapping[str, object]:
    """Strictly validate a selected Stage 24 depth-treatment checkpoint."""

    if spec != treatment:
        raise ValueError("Stage 24 checkpoint spec must be the treatment branch")
    if path.resolve() != spec.checkpoint_path.resolve():
        raise ValueError("checkpoint path does not match the treatment spec")
    validate_depth_protocol(config, control, treatment, n_head, dropout, schedule)
    if initialization.initialization_seed != config.seed:
        raise ValueError("initialization seed does not match the Stage 24 config")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    before_hash = checkpoint_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = _require_mapping(checkpoint, name="Stage 24 checkpoint")

    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported checkpoint version")
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("Stage 24 requires a best checkpoint")
    step = checkpoint.get("step")
    _validate_non_negative_integer(step, name="checkpoint step")
    assert isinstance(step, int)
    if step > config.max_iters or not _is_evaluation_step(step, config, schedule):
        raise ValueError("checkpoint step is not a Stage 24 evaluation step")
    if checkpoint.get("best_step") != step:
        raise ValueError("best checkpoint step and best_step must match")
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 24 checkpoints must not restart the optimizer")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 24 optimizer provenance must be known")

    expected_parameter_count = initialization.treatment_parameter_count
    architecture = _require_mapping(
        checkpoint.get("architecture"),
        name="architecture",
    )
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": spec.n_layer,
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "residual_dropout": dropout,
        "dropout_placement": DROPOUT_PLACEMENT,
        "attention_mask": "full_causal",
        "attention_mask_scope": "per_layer",
        "max_context": None,
        "parameter_count": expected_parameter_count,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"checkpoint architecture.{key} must be {expected!r}")

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
        "residual_dropout": dropout,
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
            raise ValueError(f"checkpoint training_config.{key} must be {expected!r}")

    initialization_metadata = _require_mapping(
        checkpoint.get("initialization"),
        name="initialization",
    )
    expected_initialization = {
        "kind": INITIALIZATION_KIND,
        "seed": config.seed,
        "branch_state_sha256": initialization.treatment_state_sha256,
        "control_state_sha256": initialization.control_state_sha256,
        "treatment_state_sha256": initialization.treatment_state_sha256,
        "identical_initialization_seed": True,
        "identical_complete_model_state": False,
        "architecture_native": True,
    }
    for key, expected in expected_initialization.items():
        if initialization_metadata.get(key) != expected:
            raise ValueError(f"checkpoint initialization.{key} must be {expected!r}")

    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    expected_experiment = {
        "stage": 24,
        "branch": spec.name,
        "from_scratch": True,
        "comparison_variable": "transformer_depth",
        "branch_n_layer": spec.n_layer,
        "control_n_layer": control.n_layer,
        "treatment_n_layer": treatment.n_layer,
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "residual_dropout": dropout,
        "dropout_placement": DROPOUT_PLACEMENT,
        "identical_parameter_count": False,
        "identical_learned_initialization": False,
        "identical_initialization_seed": True,
        "identical_batch_geometry": True,
        "identical_targets_per_update": True,
        "identical_training_batches": True,
        "identical_fixed_training_batches": True,
        "identical_fixed_validation_batches": True,
        "identical_attention_shapes": True,
        "identical_dropout_masks": False,
        "same_optimizer_configuration": True,
        "same_update_count": True,
        "same_learning_rate_schedule": True,
        "fresh_paired_validation_protocol": True,
        "reused_control": True,
        "control_stage": 22,
        "control_branch": "dropout-p010",
        "control_checkpoint_sha256": control_checkpoint_sha256,
        "control_checkpoint_step": control_checkpoint_step,
        "control_parameter_count": initialization.control_parameter_count,
        "treatment_parameter_count": initialization.treatment_parameter_count,
        "initialization_seed": config.seed,
        "control_initial_state_sha256": initialization.control_state_sha256,
        "treatment_initial_state_sha256": initialization.treatment_state_sha256,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
        "total_target_characters": (
            config.batch_size * config.block_size * config.max_iters
        ),
        "learning_rate_schedule": schedule.as_metadata(max_iters=config.max_iters),
        "position_bands": bands,
        "attention_shape": (
            config.batch_size,
            n_head,
            config.block_size,
            config.block_size,
        ),
        "attention_shape_per_layer": (
            config.batch_size,
            n_head,
            config.block_size,
            config.block_size,
        ),
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(f"checkpoint experiment.{key} must be {expected!r}")

    evaluation = _require_mapping(
        checkpoint.get("evaluation_protocol"),
        name="evaluation_protocol",
    )
    expected_evaluation = {
        "fixed_panel_seed": config.seed + 2,
        "model_mode": "eval",
        "dropout_enabled": False,
        "gradients_enabled": False,
        "split_order": ("train", "val"),
        "batches_per_split": config.eval_iters,
        "reported_train_loss": "fixed_training_panel_eval_mode",
        "reported_validation_loss": "fixed_validation_panel_eval_mode",
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(
                f"checkpoint evaluation_protocol.{key} must be {expected!r}"
            )

    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("checkpoint data fingerprint does not match")
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("training_generator_state must be a tensor")
    expected_generator = expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator):
        raise ValueError("checkpoint training generator state does not match protocol")
    rng_state = _require_mapping(checkpoint.get("rng_state"), name="rng_state")
    if not isinstance(rng_state.get("cpu"), torch.Tensor):
        raise ValueError("checkpoint CPU RNG state must be a tensor")
    for accelerator_name in ("xpu", "cuda"):
        accelerator_states = rng_state.get(accelerator_name)
        if accelerator_states is not None and (
            not isinstance(accelerator_states, list)
            or not accelerator_states
            or not all(isinstance(state, torch.Tensor) for state in accelerator_states)
        ):
            raise ValueError(f"checkpoint {accelerator_name} RNG states are malformed")

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    with torch.random.fork_rng(devices=[]):
        model = _build_model(data, config, spec, n_head, dropout)
        try:
            model.load_state_dict(model_state, strict=True)
        except RuntimeError as error:
            raise ValueError("checkpoint model state is incompatible") from error
    if count_parameters(model) != expected_parameter_count:
        raise ValueError("checkpoint model parameter count is incorrect")
    expected_mask = torch.tril(torch.ones(config.block_size, config.block_size))
    saved_mask_names = tuple(name for name in model_state if name.endswith(".tril"))
    if len(saved_mask_names) != n_head * spec.n_layer:
        raise ValueError("checkpoint has an incorrect number of causal masks")
    for name in saved_mask_names:
        saved_mask = model_state[name]
        if not isinstance(saved_mask, torch.Tensor):
            raise ValueError(f"checkpoint mask {name!r} must be a tensor")
        try:
            torch.testing.assert_close(
                saved_mask.cpu(),
                expected_mask,
                rtol=0,
                atol=0,
            )
        except AssertionError as error:
            raise ValueError(
                f"checkpoint contains an incorrect causal mask: {name}"
            ) from error
    dropout_modules = [
        module for module in model.modules() if isinstance(module, nn.Dropout)
    ]
    if len(dropout_modules) != 2 * spec.n_layer or any(
        module.p != dropout for module in dropout_modules
    ):
        raise ValueError("checkpoint model dropout sites do not match protocol")

    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"),
        name="optimizer_state_dict",
    )
    parameter_groups = optimizer_state.get("param_groups")
    if not isinstance(parameter_groups, list) or not parameter_groups:
        raise ValueError("optimizer checkpoint must contain parameter groups")
    if len(parameter_groups) != 1:
        raise ValueError("Stage 24 optimizer must contain exactly one parameter group")
    if not all(
        isinstance(group, Mapping)
        and all(
            key in group for key in ("lr", "betas", "eps", "weight_decay", "params")
        )
        for group in parameter_groups
    ):
        raise ValueError("optimizer parameter group metadata is incomplete")
    try:
        optimizer_rates = {
            float(group["lr"])
            for group in parameter_groups
            if isinstance(group, Mapping) and "lr" in group
        }
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint optimizer LR must be numeric") from error
    if optimizer_rates != {expected_learning_rate}:
        raise ValueError(f"checkpoint optimizer LR must be {expected_learning_rate:g}")
    optimizer_group = parameter_groups[0]
    assert isinstance(optimizer_group, Mapping)
    try:
        optimizer_betas = tuple(float(value) for value in optimizer_group["betas"])
        optimizer_eps = float(optimizer_group["eps"])
        optimizer_weight_decay = float(optimizer_group["weight_decay"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint optimizer settings must be numeric") from error
    if optimizer_betas != (0.9, 0.999):
        raise ValueError("checkpoint optimizer betas do not match Stage 24")
    if optimizer_eps != 1e-8:
        raise ValueError("checkpoint optimizer epsilon does not match Stage 24")
    if optimizer_weight_decay != 0.01:
        raise ValueError("checkpoint optimizer weight decay does not match Stage 24")
    optimizer = torch.optim.AdamW(model.parameters(), lr=expected_learning_rate)
    try:
        optimizer.load_state_dict(dict(optimizer_state))
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint optimizer state is incompatible") from error
    parameters = tuple(model.parameters())
    if step == 0 and optimizer.state:
        raise ValueError("step-zero checkpoint must have empty optimizer state")
    if step > 0 and len(optimizer.state) != len(parameters):
        raise ValueError("checkpoint optimizer state is incomplete")
    for parameter, state in optimizer.state.items():
        optimizer_step = state.get("step")
        if isinstance(optimizer_step, torch.Tensor):
            if optimizer_step.numel() != 1:
                raise ValueError("optimizer step must be scalar")
            optimizer_step = optimizer_step.item()
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, (int, float))
            or float(optimizer_step) != step
        ):
            raise ValueError("checkpoint optimizer step is inconsistent")
        for key, item in state.items():
            if key != "step" and (
                not isinstance(item, torch.Tensor) or item.shape != parameter.shape
            ):
                raise ValueError(f"checkpoint optimizer {key} shape is incompatible")

    best_train = _validate_serialized_losses(
        checkpoint.get("best_training_losses"),
        bands,
        name="best_training_losses",
    )
    best_val = _validate_serialized_losses(
        checkpoint.get("best_validation_losses"),
        bands,
        name="best_validation_losses",
    )
    best_val_loss = checkpoint.get("best_val_loss")
    if (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
        or float(best_val_loss) != best_val.overall
    ):
        raise ValueError("checkpoint best validation loss is inconsistent")
    if not math.isfinite(best_train.overall):
        raise ValueError("checkpoint best training loss is not finite")
    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError("Stage 24 checkpoint changed during validation")
    del optimizer, model
    return checkpoint


def make_evaluation_callback(
    best: DepthValidationCheckpoint,
    schedule: LearningRateSchedule,
    control: ReferenceCheckpoint,
) -> tuple[Callable[[EvaluationRecord], None], set[int]]:
    verified_reference_steps: set[int] = set()

    def record_evaluation(record: EvaluationRecord) -> None:
        if record.step == control.checkpoint_step:
            if not torch.equal(
                best.training_generator.get_state().cpu(),
                control.training_generator_state,
            ):
                raise RuntimeError(
                    "treatment batch-generator state does not match the "
                    f"Stage 22 control at step {record.step}"
                )
            verified_reference_steps.add(record.step)
        is_best = best.consider(record)
        assert best.best_step is not None
        learning_rate = schedule.for_update(max(0, record.step - 1))
        train = record.losses.train
        val = record.losses.val
        band_text = " | ".join(
            f"eval-val {start:03d}-{end - 1:03d} {loss:.4f}"
            for (start, end), loss in zip(best.bands, val.bands, strict=True)
        )
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr {learning_rate:g} | "
            f"eval-train {train.overall:.4f} | eval-val {val.overall:.4f} | "
            f"eval-val first {val.first_half:.4f} | "
            f"eval-val second {val.second_half:.4f} | "
            f"{band_text} | eval-gap {val.overall - train.overall:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )

    return record_evaluation, verified_reference_steps


def benchmark_depth_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: DepthSpec,
    n_head: int,
    schedule: LearningRateSchedule,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
    dropout: float = DEFAULT_DROPOUT,
) -> BenchmarkStats:
    """Benchmark a disposable model on the real Stage 24 optimizer path."""

    clear_accelerator_cache(device)
    model = _build_model(data, config, spec, n_head, dropout, device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.for_update(0))
    generator = torch.Generator().manual_seed(training_batch_seed)
    seed_everything(training_rng_seed)
    model.train()
    total_steps = benchmark.num_warmup + benchmark.num_steps
    if total_steps > config.max_iters:
        raise ValueError(
            f"benchmark requires {total_steps} updates, but max_iters is "
            f"{config.max_iters}"
        )

    started_at: float | None = None
    for update_index in range(total_steps):
        if update_index == benchmark.num_warmup:
            sync_device(device)
            reset_peak_memory_stats(device)
            started_at = time.perf_counter()
        set_optimizer_learning_rate(optimizer, schedule.for_update(update_index))
        inputs, targets = data.get_batch(
            "train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()

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
    del optimizer, model
    clear_accelerator_cache(device)
    return stats


def aggregate_benchmark_stats(
    runs: Sequence[BenchmarkStats],
    *,
    steps_per_run: int,
) -> BenchmarkStats:
    if not runs:
        raise ValueError("at least one benchmark run is required")
    if steps_per_run <= 0:
        raise ValueError("steps_per_run must be positive")
    elapsed = sum(run.seconds for run in runs)
    iterations_per_sec = len(runs) * steps_per_run / elapsed
    tokens_per_iteration = runs[0].tokens_per_sec / runs[0].iterations_per_sec
    return BenchmarkStats(
        seconds=elapsed,
        iterations_per_sec=iterations_per_sec,
        tokens_per_sec=iterations_per_sec * tokens_per_iteration,
        peak_allocated_mb=max(run.peak_allocated_mb for run in runs),
        peak_reserved_mb=max(run.peak_reserved_mb for run in runs),
    )


def run_counterbalanced_benchmarks(
    data: CharacterData,
    config: TrainingConfig,
    control: DepthSpec,
    treatment: DepthSpec,
    n_head: int,
    schedule: LearningRateSchedule,
    device: torch.device,
    initialization: PairedInitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
    dropout: float = DEFAULT_DROPOUT,
) -> tuple[BenchmarkStats, BenchmarkStats]:
    """Run adjacent control-treatment-treatment-control disposable trials."""

    control_runs: list[BenchmarkStats] = []
    treatment_runs: list[BenchmarkStats] = []
    ordered_runs = (
        (control, initialization.control, control_runs),
        (treatment, initialization.treatment, treatment_runs),
        (treatment, initialization.treatment, treatment_runs),
        (control, initialization.control, control_runs),
    )
    for spec, state, destination in ordered_runs:
        destination.append(
            benchmark_depth_branch(
                data,
                config,
                spec,
                n_head,
                schedule,
                device,
                state,
                training_batch_seed=training_batch_seed,
                training_rng_seed=training_rng_seed,
                benchmark=benchmark,
                dropout=dropout,
            )
        )
    return (
        aggregate_benchmark_stats(control_runs, steps_per_run=benchmark.num_steps),
        aggregate_benchmark_stats(
            treatment_runs,
            steps_per_run=benchmark.num_steps,
        ),
    )


def _print_benchmark(name: str, stats: BenchmarkStats) -> None:
    print(f"\n{name} disposable training benchmark")
    print(f"  seconds:               {stats.seconds:.3f}")
    print(f"  iterations/sec:        {stats.iterations_per_sec:.3f}")
    print(f"  target characters/sec: {stats.tokens_per_sec:.3f}")
    print(f"  peak allocated MiB:    {stats.peak_allocated_mb:.3f}")
    print(f"  peak reserved MiB:     {stats.peak_reserved_mb:.3f}")


def run_treatment(
    data: CharacterData,
    config: TrainingConfig,
    control_spec: DepthSpec,
    treatment_spec: DepthSpec,
    n_head: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initialization: PairedInitialStates,
    control: ReferenceCheckpoint,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    dropout: float = DEFAULT_DROPOUT,
) -> TreatmentReport:
    clear_accelerator_cache(device)
    model = _build_model(
        data,
        config,
        treatment_spec,
        n_head,
        dropout,
        device,
    )
    model.load_state_dict(initialization.treatment, strict=True)
    if (
        tensor_mapping_sha256(model.state_dict())
        != initialization.treatment_state_sha256
    ):
        raise RuntimeError(
            f"L={treatment_spec.n_layer} did not load its exact native initialization"
        )
    parameter_count = count_parameters(model)
    if parameter_count != initialization.treatment_parameter_count:
        raise RuntimeError(
            f"L={treatment_spec.n_layer} treatment parameter count changed"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.for_update(0))
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    attention_shape = (
        config.batch_size,
        n_head,
        config.block_size,
        config.block_size,
    )
    best = DepthValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=treatment_spec.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=treatment_spec.n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        branch_spec=treatment_spec,
        control_spec=control_spec,
        treatment_spec=treatment_spec,
        schedule=schedule,
        target_step=config.max_iters,
        dropout=dropout,
        initialization_seed=config.seed,
        branch_initial_state_sha256=(initialization.treatment_state_sha256),
        control_initial_state_sha256=(initialization.control_state_sha256),
        treatment_initial_state_sha256=(initialization.treatment_state_sha256),
        control_parameter_count=initialization.control_parameter_count,
        treatment_parameter_count=initialization.treatment_parameter_count,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        control_checkpoint_sha256=control.checkpoint_sha256,
        control_checkpoint_step=control.checkpoint_step,
        attention_shape=attention_shape,
        bands=bands,
    )
    callback, verified_reference_steps = make_evaluation_callback(
        best,
        schedule,
        control,
    )
    print(
        f"\n{treatment_spec.name} treatment (B={config.batch_size}, "
        f"T={config.block_size}, C={config.n_embd}, H={n_head}, "
        f"D={config.n_embd // n_head}, FF={4 * config.n_embd}, "
        f"L={treatment_spec.n_layer}, residual dropout={dropout:g})"
    )
    print(f"  native initialization: {initialization.treatment_state_sha256}")
    print(f"  parameters:            {parameter_count:,}")
    print(f"  training-batch seed:   {training_batch_seed}")
    print(f"  dropout/model RNG seed:{training_rng_seed:>8}")

    # Model construction consumes the global RNG. Reset immediately before
    # update zero; the explicit batch generator remains independent.
    seed_everything(training_rng_seed)
    result = train_with_schedule(
        model,
        optimizer,
        data,
        config,
        device,
        training_generator,
        schedule,
        bands,
        on_evaluation=callback,
    )
    if control.checkpoint_step not in verified_reference_steps:
        raise RuntimeError(
            f"the Stage 22 L={control_spec.n_layer} control step was not "
            "encountered in Stage 24 evaluations"
        )
    expected_final_generator = expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=config.max_iters,
    )
    if not torch.equal(training_generator.get_state(), expected_final_generator):
        raise RuntimeError(
            f"L={treatment_spec.n_layer} did not consume the exact Stage 22 "
            "batch stream"
        )
    measured_attention_shape = inspect_attention_shape(
        model,
        config,
        n_head,
        device,
    )
    if measured_attention_shape != attention_shape:
        raise RuntimeError("attention shape changed during treatment training")
    assert best.best_step is not None
    assert best.best_training_losses is not None
    assert best.best_validation_losses is not None

    checkpoint = torch.load(
        treatment_spec.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_mapping(checkpoint, name="Stage 24 checkpoint")
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    model.load_state_dict(model_state, strict=True)
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=config.seed + 5,
    )
    report = TreatmentReport(
        spec=treatment_spec,
        parameter_count=parameter_count,
        initial_state_sha256=initialization.treatment_state_sha256,
        training=result,
        best_training_losses=best.best_training_losses,
        best_validation_losses=best.best_validation_losses,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        final_batch_generator_sha256=tensor_sha256(training_generator.get_state()),
        attention_shape=measured_attention_shape,
        sample=sample,
    )
    del checkpoint, callback, best, optimizer, model
    clear_accelerator_cache(device)
    return report


def replay_treatment_fixed_panel(
    treatment: DepthSpec,
    expected_checkpoint_sha256: str,
    expected_losses: PositionLossEstimate,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    *,
    dropout: float = DEFAULT_DROPOUT,
) -> PositionLossEstimate:
    """Reload the selected treatment weights and tie saved metrics to them."""

    if checkpoint_sha256(treatment.checkpoint_path) != expected_checkpoint_sha256:
        raise RuntimeError("Stage 24 checkpoint changed before fixed-panel replay")
    checkpoint = torch.load(
        treatment.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_mapping(checkpoint, name="Stage 24 checkpoint")
    model = _build_model(data, config, treatment, n_head, dropout, device)
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    model.load_state_dict(model_state, strict=True)
    actual = evaluate_on_fixed_batches(model, data, config, device, bands)
    _assert_position_losses_close(
        actual.train,
        expected_losses.train,
        name="training",
    )
    _assert_position_losses_close(
        actual.val,
        expected_losses.val,
        name="validation",
    )
    if checkpoint_sha256(treatment.checkpoint_path) != expected_checkpoint_sha256:
        raise RuntimeError("Stage 24 checkpoint changed during fixed-panel replay")
    del checkpoint, model
    clear_accelerator_cache(device)
    return actual


def _measure_initial_losses(
    data: CharacterData,
    config: TrainingConfig,
    spec: DepthSpec,
    n_head: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    state: Mapping[str, torch.Tensor],
    *,
    dropout: float,
) -> PositionLossEstimate:
    model = _build_model(data, config, spec, n_head, dropout, device)
    model.load_state_dict(state, strict=True)
    losses = evaluate_on_fixed_batches(model, data, config, device, bands)
    del model
    clear_accelerator_cache(device)
    return losses


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    control: DepthSpec,
    treatment: DepthSpec,
    n_head: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_control_sha256: str | None = None,
    expected_control_step: int | None = None,
    benchmark: BenchmarkConfig | None = None,
    dropout: float = DEFAULT_DROPOUT,
) -> ExperimentReport:
    if expected_control_sha256 is None or expected_control_step is None:
        raise ValueError(
            "Stage 24 requires an expected control checkpoint SHA-256 and step"
        )
    targets_per_update, bands = validate_depth_protocol(
        config,
        control,
        treatment,
        n_head,
        dropout,
        schedule,
    )
    _validate_non_negative_integer(sample_length, name="sample length")
    _validate_non_negative_integer(training_batch_seed, name="training batch seed")
    _validate_non_negative_integer(training_rng_seed, name="training RNG seed")
    if benchmark is not None:
        benchmark_steps = benchmark.num_warmup + benchmark.num_steps
        if benchmark_steps > config.max_iters:
            raise ValueError(
                f"benchmark requires {benchmark_steps} steps, but max_iters "
                f"is {config.max_iters}"
            )
    for split, source in (("train", data.train_data), ("val", data.val_data)):
        if len(source) <= config.block_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but block_size "
                f"is {config.block_size}"
            )

    initialization = create_paired_initial_states(
        data,
        config,
        control,
        treatment,
        n_head,
        dropout,
    )
    expected_control_count = expected_parameter_count(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_layer=control.n_layer,
    )
    expected_treatment_count = expected_parameter_count(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_layer=treatment.n_layer,
    )
    if initialization.control_parameter_count != expected_control_count:
        raise RuntimeError(
            f"L={control.n_layer} parameter count must be {expected_control_count:,}"
        )
    if initialization.treatment_parameter_count != expected_treatment_count:
        raise RuntimeError(
            f"L={treatment.n_layer} parameter count must be "
            f"{expected_treatment_count:,}"
        )

    control_reference = _STAGE_23.validate_stage_22_reference(
        control.checkpoint_path,
        data,
        config,
        n_head,
        control.n_layer,
        schedule,
        bands,
        initial_state_sha256=initialization.control_state_sha256,
        parameter_count=initialization.control_parameter_count,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_checkpoint_sha256=expected_control_sha256,
        expected_step=expected_control_step,
    )
    control_hash_before = control_reference.checkpoint_sha256
    control_losses = _load_and_measure_fixed_control(
        control_reference,
        data,
        config,
        n_head,
        control.n_layer,
        device,
        bands,
    )
    control_initial_losses = _measure_initial_losses(
        data,
        config,
        control,
        n_head,
        device,
        bands,
        initialization.control,
        dropout=dropout,
    )
    treatment_initial_losses = _measure_initial_losses(
        data,
        config,
        treatment,
        n_head,
        device,
        bands,
        initialization.treatment,
        dropout=dropout,
    )

    print("\nControlled C=128 transformer-depth protocol")
    print(
        f"  The Stage 22 L={control.n_layer}, p={dropout:g} best checkpoint "
        "is validated and remains read-only."
    )
    print(
        f"  The L={treatment.n_layer} treatment uses architecture-native "
        "initialization recreated from the same seed and a fresh AdamW optimizer."
    )
    print(
        "  Training batches, fixed panels, targets/update, update count, "
        "optimizer settings, and LR schedule match Stage 22 exactly."
    )
    print(
        "  Fixed train and validation panels run in eval mode with dropout "
        "disabled; eval-gap is apples-to-apples."
    )
    print(
        "  The model RNG seed is shared, but depth changes dropout RNG "
        "consumption; masks are not claimed identical."
    )
    print(
        f"  L={control.n_layer} initialization SHA-256:   "
        f"{initialization.control_state_sha256}"
    )
    print(
        f"  L={treatment.n_layer} initialization SHA-256: "
        f"{initialization.treatment_state_sha256}"
    )
    print(f"  Stage 22 control SHA-256:      {control_hash_before}")
    print(
        f"  Targets per update: {targets_per_update:,}; total per branch: "
        f"{targets_per_update * config.max_iters:,}."
    )

    control_benchmark: BenchmarkStats | None = None
    treatment_benchmark: BenchmarkStats | None = None
    if benchmark is not None:
        print("\nCounterbalanced disposable training benchmark")
        print(
            f"  Order: {control.name}, {treatment.name}, "
            f"{treatment.name}, {control.name}"
        )
        print(
            f"  Each run: {benchmark.num_warmup} warmup + "
            f"{benchmark.num_steps} timed updates"
        )
        control_benchmark, treatment_benchmark = run_counterbalanced_benchmarks(
            data,
            config,
            control,
            treatment,
            n_head,
            schedule,
            device,
            initialization,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            benchmark=benchmark,
            dropout=dropout,
        )
        _print_benchmark(control.name, control_benchmark)
        _print_benchmark(treatment.name, treatment_benchmark)

    treatment_report = run_treatment(
        data,
        config,
        control,
        treatment,
        n_head,
        device,
        sample_length,
        schedule,
        bands,
        initialization,
        control_reference,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        dropout=dropout,
    )
    if treatment_report.training.initial != treatment_initial_losses:
        raise RuntimeError(f"L={treatment.n_layer} initial fixed-panel replay changed")
    if checkpoint_sha256(control.checkpoint_path) != control_hash_before:
        raise RuntimeError("Stage 22 control checkpoint changed during Stage 24")
    treatment_hash = checkpoint_sha256(treatment.checkpoint_path)
    validate_stage_24_checkpoint(
        treatment.checkpoint_path,
        data,
        config,
        treatment,
        control,
        treatment,
        n_head,
        schedule,
        bands,
        initialization,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        control_checkpoint_sha256=control_reference.checkpoint_sha256,
        control_checkpoint_step=control_reference.checkpoint_step,
        dropout=dropout,
    )
    replay_treatment_fixed_panel(
        treatment,
        treatment_hash,
        PositionLossEstimate(
            train=treatment_report.best_training_losses,
            val=treatment_report.best_validation_losses,
        ),
        data,
        config,
        n_head,
        device,
        bands,
        dropout=dropout,
    )
    if checkpoint_sha256(control.checkpoint_path) != control_hash_before:
        raise RuntimeError("Stage 22 control checkpoint changed during validation")
    if checkpoint_sha256(treatment.checkpoint_path) != treatment_hash:
        raise RuntimeError("Stage 24 treatment checkpoint changed during validation")
    return ExperimentReport(
        initialization=initialization,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        targets_per_update=targets_per_update,
        bands=bands,
        control_initial_losses=control_initial_losses,
        treatment_initial_losses=treatment_initial_losses,
        control=ControlReport(
            spec=control,
            reference=control_reference,
            losses_at_best=control_losses,
        ),
        treatment=treatment_report,
        control_benchmark=control_benchmark,
        treatment_benchmark=treatment_benchmark,
        treatment_checkpoint_sha256=treatment_hash,
    )


def run_precise_validation(
    data: CharacterData,
    config: TrainingConfig,
    control: DepthSpec,
    treatment: DepthSpec,
    n_head: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    experiment: ExperimentReport,
    *,
    eval_iters: int = DEFAULT_PRECISE_EVAL_ITERS,
    seed: int,
    dropout: float = DEFAULT_DROPOUT,
) -> PreciseValidationReport:
    if eval_iters < 2:
        raise ValueError(f"eval_iters must be at least 2, got {eval_iters}")
    _validate_non_negative_integer(seed, name="precise evaluation seed")
    if seed == config.seed + 2:
        raise ValueError("precise evaluation seed must differ from fixed-panel seed")
    _, bands = validate_depth_protocol(
        config,
        control,
        treatment,
        n_head,
        dropout,
        schedule,
    )
    if bands != experiment.bands:
        raise ValueError("precise-validation bands differ from training protocol")
    if control != experiment.control.spec or treatment != experiment.treatment.spec:
        raise ValueError("precise-validation depth specs differ from the experiment")
    recreated_initialization = create_paired_initial_states(
        data,
        config,
        control,
        treatment,
        n_head,
        dropout,
    )
    recreated_identity = (
        recreated_initialization.control_state_sha256,
        recreated_initialization.treatment_state_sha256,
        recreated_initialization.control_parameter_count,
        recreated_initialization.treatment_parameter_count,
        recreated_initialization.initialization_seed,
    )
    reported_identity = (
        experiment.initialization.control_state_sha256,
        experiment.initialization.treatment_state_sha256,
        experiment.initialization.control_parameter_count,
        experiment.initialization.treatment_parameter_count,
        experiment.initialization.initialization_seed,
    )
    if recreated_identity != reported_identity:
        raise ValueError("Stage 24 initialization report does not match the protocol")

    control_reference = _STAGE_23.validate_stage_22_reference(
        control.checkpoint_path,
        data,
        config,
        n_head,
        control.n_layer,
        schedule,
        bands,
        initial_state_sha256=experiment.initialization.control_state_sha256,
        parameter_count=experiment.initialization.control_parameter_count,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        expected_checkpoint_sha256=(experiment.control.reference.checkpoint_sha256),
        expected_step=experiment.control.reference.checkpoint_step,
    )
    treatment_hash = checkpoint_sha256(treatment.checkpoint_path)
    if treatment_hash != experiment.treatment_checkpoint_sha256:
        raise RuntimeError(
            "Stage 24 treatment checkpoint changed after checkpoint selection: "
            f"expected {experiment.treatment_checkpoint_sha256}, got "
            f"{treatment_hash}"
        )
    treatment_checkpoint = validate_stage_24_checkpoint(
        treatment.checkpoint_path,
        data,
        config,
        treatment,
        control,
        treatment,
        n_head,
        schedule,
        bands,
        experiment.initialization,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        control_checkpoint_sha256=control_reference.checkpoint_sha256,
        control_checkpoint_step=control_reference.checkpoint_step,
        dropout=dropout,
    )
    control_checkpoint = torch.load(
        control.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    control_checkpoint = _require_mapping(
        control_checkpoint,
        name="Stage 22 control checkpoint",
    )
    control_result = _measure_precise_checkpoint(
        name=f"L={control.n_layer}",
        dropout=dropout,
        path=control.checkpoint_path,
        expected_sha256=control_reference.checkpoint_sha256,
        checkpoint=control_checkpoint,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=control.n_layer,
        device=device,
        bands=bands,
        eval_iters=eval_iters,
        seed=seed,
    )
    treatment_result = _measure_precise_checkpoint(
        name=f"L={treatment.n_layer}",
        dropout=dropout,
        path=treatment.checkpoint_path,
        expected_sha256=treatment_hash,
        checkpoint=treatment_checkpoint,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=treatment.n_layer,
        device=device,
        bands=bands,
        eval_iters=eval_iters,
        seed=seed,
    )
    if (
        checkpoint_sha256(control.checkpoint_path)
        != control_reference.checkpoint_sha256
    ):
        raise RuntimeError(
            "Stage 22 control checkpoint changed during fresh validation"
        )
    if checkpoint_sha256(treatment.checkpoint_path) != treatment_hash:
        raise RuntimeError(
            "Stage 24 treatment checkpoint changed during fresh validation"
        )
    return PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        bands=bands,
        results=(control_result, treatment_result),
        delta=_paired_delta(treatment_result, control_result),
    )


def print_experiment_summary(report: ExperimentReport) -> None:
    control = report.control
    treatment = report.treatment
    parameter_delta = treatment.parameter_count - control.reference.parameter_count
    parameter_ratio = treatment.parameter_count / control.reference.parameter_count
    print("\nStage 24 summary")
    print(f"  targets/update:               {report.targets_per_update:,}")
    print(
        f"  L={control.spec.n_layer} / L={treatment.spec.n_layer} parameters:       "
        f"{control.reference.parameter_count:,} / {treatment.parameter_count:,} "
        f"({parameter_delta:+,}; {parameter_ratio:.3f}x)"
    )
    print(
        f"  L={control.spec.n_layer} initial eval train/val:    "
        f"{report.control_initial_losses.train.overall:.4f} / "
        f"{report.control_initial_losses.val.overall:.4f}"
    )
    print(
        f"  L={treatment.spec.n_layer} initial eval train/val:    "
        f"{report.treatment_initial_losses.train.overall:.4f} / "
        f"{report.treatment_initial_losses.val.overall:.4f}"
    )
    print(
        f"  L={control.spec.n_layer} best eval-train/val/gap:     "
        f"{control.losses_at_best.train.overall:.4f} / "
        f"{control.losses_at_best.val.overall:.4f} / "
        f"{control.generalization_gap_at_best:.4f} @ {control.best_step}"
    )
    print(
        f"  L={treatment.spec.n_layer} best eval-train/val/gap:     "
        f"{treatment.best_training_losses.overall:.4f} / "
        f"{treatment.best_validation_losses.overall:.4f} / "
        f"{treatment.generalization_gap_at_best:.4f} @ {treatment.best_step}"
    )
    print(
        f"  best fixed-panel delta:       {report.best_val_delta:+.6f} "
        f"(L={treatment.spec.n_layer} - L={control.spec.n_layer}); "
        f"nominal winner: {report.best_winner}"
    )
    print(
        f"  L={treatment.spec.n_layer} final eval-train/val/gap:    "
        f"{treatment.training.final.train.overall:.4f} / "
        f"{treatment.training.final.val.overall:.4f} / "
        f"{treatment.final_generalization_gap:.4f}"
    )
    if report.control_benchmark is not None and report.treatment_benchmark is not None:
        iteration_ratio = (
            report.treatment_benchmark.iterations_per_sec
            / report.control_benchmark.iterations_per_sec
        )
        print(
            f"  L={treatment.spec.n_layer}/L={control.spec.n_layer} "
            f"training-throughput ratio: {iteration_ratio:.3f}x"
        )
        if report.control_benchmark.peak_allocated_mb > 0:
            allocated_ratio = (
                report.treatment_benchmark.peak_allocated_mb
                / report.control_benchmark.peak_allocated_mb
            )
            print(
                f"  L={treatment.spec.n_layer}/L={control.spec.n_layer} "
                f"peak-allocated ratio:      {allocated_ratio:.3f}x"
            )
        if report.control_benchmark.peak_reserved_mb > 0:
            reserved_ratio = (
                report.treatment_benchmark.peak_reserved_mb
                / report.control_benchmark.peak_reserved_mb
            )
            print(
                f"  L={treatment.spec.n_layer}/L={control.spec.n_layer} "
                f"peak-reserved ratio:       {reserved_ratio:.3f}x"
            )
    print("  Fixed-panel train and val losses use model.eval(); dropout is off.")
    print(
        f"  read-only L={control.spec.n_layer} checkpoint:  "
        f"{control.reference.checkpoint_path}"
    )
    print(
        f"  L={treatment.spec.n_layer} checkpoint:            "
        f"{treatment.spec.checkpoint_path}"
    )
    print(
        f"  L={treatment.spec.n_layer} checkpoint SHA-256:    "
        f"{report.treatment_checkpoint_sha256}"
    )
    print(
        f"\nL={treatment.spec.n_layer} best-checkpoint generated text "
        f"(step {treatment.best_step})"
    )
    print(treatment.sample)


def main() -> None:
    args = parse_args()
    for name, value in (
        ("seed", args.seed),
        ("sample length", args.sample_length),
        ("benchmark warmup", args.benchmark_warmup),
        ("benchmark steps", args.benchmark_steps),
        ("precise eval iters", args.precise_eval_iters),
    ):
        _validate_non_negative_integer(value, name=name)
    if args.precise_eval_iters == 1:
        raise ValueError("precise-eval-iters must be 0 or at least 2")

    schedule = LearningRateSchedule(
        initial_learning_rate=args.initial_learning_rate,
        first_decay_step=args.first_decay_step,
        middle_learning_rate=args.middle_learning_rate,
        second_decay_step=args.second_decay_step,
        final_learning_rate=args.final_learning_rate,
    )
    config = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        learning_rate=args.initial_learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    control = DepthSpec(
        name=f"depth-{args.control_layers}",
        n_layer=args.control_layers,
        checkpoint_path=args.control_checkpoint_path,
    )
    treatment = DepthSpec(
        name=f"depth-{args.treatment_layers}",
        n_layer=args.treatment_layers,
        checkpoint_path=args.checkpoint_path,
    )
    validate_depth_protocol(
        config,
        control,
        treatment,
        args.n_head,
        args.dropout,
        schedule,
    )
    training_batch_seed = (
        args.seed + 1 if args.training_batch_seed is None else args.training_batch_seed
    )
    training_rng_seed = (
        args.seed + 3 if args.training_rng_seed is None else args.training_rng_seed
    )
    precise_eval_seed = (
        args.seed + 7 if args.precise_eval_seed is None else args.precise_eval_seed
    )
    for name, value in (
        ("training batch seed", training_batch_seed),
        ("training RNG seed", training_rng_seed),
        ("precise evaluation seed", precise_eval_seed),
    ):
        _validate_non_negative_integer(value, name=name)
    if args.precise_eval_iters > 0 and precise_eval_seed == args.seed + 2:
        raise ValueError("precise evaluation seed must differ from fixed-panel seed")
    control_pin, control_step_pin = resolve_control_pin(
        args.control_checkpoint_path,
        args.control_checkpoint_sha256,
        args.control_checkpoint_step,
    )
    benchmark = None
    if args.benchmark_steps > 0:
        benchmark = BenchmarkConfig(
            num_warmup=args.benchmark_warmup,
            num_steps=args.benchmark_steps,
        )
        if benchmark.num_warmup + benchmark.num_steps > config.max_iters:
            raise ValueError("benchmark warmup + timed steps must not exceed max-iters")

    data = CharacterData.from_file(args.data)
    device = resolve_device(args.device)
    print(
        f"Stage 24: C={config.n_embd}, p={args.dropout:g} depth scaling "
        f"L={control.n_layer} -> L={treatment.n_layer}"
    )
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Shared B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={args.n_head}, "
        f"D={config.n_embd // args.n_head}, FF={4 * config.n_embd}, "
        f"residual dropout={args.dropout:g}"
    )
    print(
        f"Treatment: reused L={control.n_layer} best checkpoint vs "
        f"new L={treatment.n_layer} branch"
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
    report = run_experiment(
        data,
        config,
        control,
        treatment,
        args.n_head,
        device,
        args.sample_length,
        schedule,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_control_sha256=control_pin,
        expected_control_step=control_step_pin,
        benchmark=benchmark,
        dropout=args.dropout,
    )
    print_experiment_summary(report)
    if args.precise_eval_iters > 0:
        precise = run_precise_validation(
            data,
            config,
            control,
            treatment,
            args.n_head,
            device,
            schedule,
            report,
            eval_iters=args.precise_eval_iters,
            seed=precise_eval_seed,
            dropout=args.dropout,
        )
        _STAGE_23._STAGE_22._STAGE_21.print_precise_validation(
            precise,
            stage_label="Stage 22/24",
        )


if __name__ == "__main__":
    main()
