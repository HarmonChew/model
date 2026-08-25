from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import seed_everything


# Stage 21 changes only residual-path dropout for the Stage 20 C=128 model.
# Numeric stage filenames are not importable normally, so load the tested
# Stage 20 implementation explicitly and reuse its training/evaluation/statistical
# machinery.
_STAGE_20_PATH = Path(__file__).with_name("20_scale_width.py")
_STAGE_20_MODULE_NAME = "stage_20_scale_width_for_stage_21"
_STAGE_20_SPEC = importlib.util.spec_from_file_location(
    _STAGE_20_MODULE_NAME,
    _STAGE_20_PATH,
)
assert _STAGE_20_SPEC is not None and _STAGE_20_SPEC.loader is not None
_STAGE_20 = importlib.util.module_from_spec(_STAGE_20_SPEC)
sys.modules[_STAGE_20_MODULE_NAME] = _STAGE_20
_STAGE_20_SPEC.loader.exec_module(_STAGE_20)

GPTLanguageModel = _STAGE_20.GPTLanguageModel
LearningRateSchedule = _STAGE_20.LearningRateSchedule
BestValidationCheckpoint = _STAGE_20.BestValidationCheckpoint
PositionLosses = _STAGE_20.PositionLosses
PositionLossEstimate = _STAGE_20.PositionLossEstimate
EvaluationRecord = _STAGE_20.WidthEvaluationRecord
TrainingResult = _STAGE_20.WidthTrainingResult
PreciseBranchResult = _STAGE_20.PreciseBranchResult
PreciseValidationReport = _STAGE_20.PreciseValidationReport

checkpoint_sha256 = _STAGE_20.checkpoint_sha256
clear_accelerator_cache = _STAGE_20.clear_accelerator_cache
count_parameters = _STAGE_20.count_parameters
estimate_validation_precise = _STAGE_20.estimate_validation_precise
evaluate_on_fixed_batches = _STAGE_20.evaluate_on_fixed_batches
expected_parameter_count = _STAGE_20.expected_parameter_count
expected_training_generator_state = _STAGE_20.expected_training_generator_state
fingerprint_data = _STAGE_20.fingerprint_data
generate_from_final_model = _STAGE_20.generate_from_final_model
inspect_attention_shape = _STAGE_20.inspect_attention_shape
make_position_bands = _STAGE_20.make_position_bands
tensor_mapping_sha256 = _STAGE_20.tensor_mapping_sha256
tensor_sha256 = _STAGE_20.tensor_sha256
train_with_schedule = _STAGE_20.train_with_schedule
_paired_delta = _STAGE_20._paired_delta
_position_statistics = _STAGE_20._position_statistics


CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_20_width_128_best_checkpoint.pt"
)
DEFAULT_TREATMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_21_dropout_005_best_checkpoint.pt"
)
DEFAULT_CONTROL_CHECKPOINT_SHA256 = (
    "fc42fd0b58d3e197db3cb4d687e841b9db3f8c38a4208f9b9db2cf329ab1d3e9"
)
DEFAULT_CONTROL_BEST_STEP = 10_500
DEFAULT_INITIAL_STATE_SHA256 = (
    "ea5ac1a1b7fc912a1f5535e6bbabc43216b48aafbd86869dffd5372e7fc0bfad"
)

DEFAULT_BATCH_SIZE = _STAGE_20.DEFAULT_BATCH_SIZE
DEFAULT_BLOCK_SIZE = _STAGE_20.DEFAULT_BLOCK_SIZE
DEFAULT_N_EMBD = _STAGE_20.DEFAULT_TREATMENT_WIDTH
DEFAULT_N_HEAD = _STAGE_20.DEFAULT_N_HEAD
DEFAULT_N_LAYER = _STAGE_20.DEFAULT_N_LAYER
DEFAULT_MAX_ITERS = _STAGE_20.DEFAULT_MAX_ITERS
DEFAULT_INITIAL_LEARNING_RATE = _STAGE_20.DEFAULT_INITIAL_LEARNING_RATE
DEFAULT_FIRST_DECAY_STEP = _STAGE_20.DEFAULT_FIRST_DECAY_STEP
DEFAULT_MIDDLE_LEARNING_RATE = _STAGE_20.DEFAULT_MIDDLE_LEARNING_RATE
DEFAULT_SECOND_DECAY_STEP = _STAGE_20.DEFAULT_SECOND_DECAY_STEP
DEFAULT_FINAL_LEARNING_RATE = _STAGE_20.DEFAULT_FINAL_LEARNING_RATE
DEFAULT_PRECISE_EVAL_ITERS = _STAGE_20.DEFAULT_PRECISE_EVAL_ITERS
DEFAULT_CONTROL_DROPOUT = 0.0
DEFAULT_TREATMENT_DROPOUT = 0.05
DROPOUT_PLACEMENT = _STAGE_20.DROPOUT_PLACEMENT


@dataclass(frozen=True, slots=True)
class DropoutSpec:
    name: str
    dropout: float
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch name must not be empty")
        if not math.isfinite(float(self.dropout)) or not 0 <= self.dropout < 1:
            raise ValueError(
                f"dropout must be finite and in [0, 1), got {self.dropout!r}"
            )


@dataclass(frozen=True, slots=True)
class ReferenceCheckpoint:
    name: str
    dropout: float
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_step: int
    fixed_panel_loss: float
    best_validation_losses: PositionLosses
    training_generator_state: torch.Tensor
    initial_state_sha256: str
    parameter_count: int


@dataclass(frozen=True, slots=True)
class ControlReport:
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
    spec: DropoutSpec
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
    parameter_count: int
    initial_state_sha256: str
    initialization_seed: int
    training_batch_seed: int
    training_rng_seed: int
    schedule: LearningRateSchedule
    bands: tuple[tuple[int, int], ...]
    initial_losses: PositionLossEstimate
    control: ControlReport
    treatment: TreatmentReport

    @property
    def best_val_delta(self) -> float:
        return self.treatment.best_val_loss - self.control.best_val_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 21: train only C=128 residual dropout p=0.05 from the "
            "exact Stage 20 initialization and batch stream, reusing the "
            "saved Stage 20 C=128 p=0 best checkpoint as a read-only control."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
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
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_TREATMENT_DROPOUT,
        help="Stage 21 treatment probability; the protocol requires exactly 0.05.",
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
        help="Defaults to seed + 1, matching Stage 20.",
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
        help="Defaults to seed + 7, the same fresh panel used by Stage 20.",
    )
    parser.add_argument(
        "--control-checkpoint-path",
        type=Path,
        default=DEFAULT_CONTROL_CHECKPOINT_PATH,
        help="Read-only Stage 20 C=128 p=0 best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_TREATMENT_CHECKPOINT_PATH,
        help="Output path for the Stage 21 p=0.05 best checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _validate_non_negative_integer(value: int, *, name: str) -> int:
    return _STAGE_20._validate_non_negative_integer(value, name=name)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    return _STAGE_20._require_mapping(value, name=name)


def _pin_for_default_path(
    path: Path,
    default_path: Path = DEFAULT_CONTROL_CHECKPOINT_PATH,
    expected_sha256: str = DEFAULT_CONTROL_CHECKPOINT_SHA256,
    expected_step: int = DEFAULT_CONTROL_BEST_STEP,
) -> tuple[str | None, int | None]:
    if path.resolve() == default_path.resolve():
        return expected_sha256, expected_step
    return None, None


def _output_paths(path: Path) -> tuple[Path, Path]:
    return (
        path.resolve(),
        path.with_name(f".{path.name}.tmp").resolve(),
    )


def validate_distinct_paths(
    treatment_checkpoint: Path,
    control_checkpoint: Path,
) -> None:
    paths = {
        "treatment checkpoint": _output_paths(treatment_checkpoint)[0],
        "treatment temporary checkpoint": _output_paths(treatment_checkpoint)[1],
        "control checkpoint": _output_paths(control_checkpoint)[0],
        "control temporary checkpoint": _output_paths(control_checkpoint)[1],
    }
    labels = tuple(paths)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            first_path = paths[first_label]
            second_path = paths[second_label]
            if first_path == second_path:
                raise ValueError(
                    f"{first_label} and {second_label} must use different paths"
                )
            if (
                first_path.exists()
                and second_path.exists()
                and first_path.samefile(second_path)
            ):
                raise ValueError(
                    f"{first_label} and {second_label} must not refer to the same file"
                )


def validate_protocol(
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    treatment: DropoutSpec,
    control_checkpoint_path: Path,
    schedule: LearningRateSchedule,
) -> tuple[tuple[int, int], ...]:
    if treatment.dropout != DEFAULT_TREATMENT_DROPOUT:
        raise ValueError("Stage 21 treatment dropout must be exactly 0.05")
    if config.n_embd <= 0 or config.n_embd % 2 != 0:
        raise ValueError("Stage 21 n_embd must be a positive even integer")
    if isinstance(n_head, bool) or not isinstance(n_head, int) or n_head <= 0:
        raise ValueError(f"n_head must be a positive integer, got {n_head!r}")
    if config.n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer <= 0:
        raise ValueError(f"n_layer must be a positive integer, got {n_layer!r}")
    if config.block_size % 4 != 0:
        raise ValueError("block_size must be divisible by four for position bands")
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError("config learning rate must equal the schedule initial LR")
    schedule.validate_target_step(config.max_iters)
    validate_distinct_paths(treatment.checkpoint_path, control_checkpoint_path)
    return make_position_bands(config.block_size)


def _build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    dropout: float,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model


def _clone_state(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def create_shared_initial_state(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
) -> tuple[dict[str, torch.Tensor], str, int]:
    """Recreate one standard initialization shared by p=0 and p=0.05."""

    states: list[dict[str, torch.Tensor]] = []
    counts: list[int] = []
    for dropout in (DEFAULT_CONTROL_DROPOUT, DEFAULT_TREATMENT_DROPOUT):
        seed_everything(config.seed)
        model = _build_model(data, config, n_head, n_layer, dropout)
        states.append(_clone_state(model))
        counts.append(count_parameters(model))
        del model
    if states[0].keys() != states[1].keys():
        raise RuntimeError("dropout changed the model state keys")
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    first_hash = tensor_mapping_sha256(states[0])
    second_hash = tensor_mapping_sha256(states[1])
    if first_hash != second_hash or counts[0] != counts[1]:
        raise RuntimeError("dropout changed initialization or parameter count")
    return states[0], first_hash, counts[0]


def _stage_20_protocol(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    control_checkpoint_path: Path,
) -> tuple[
    TrainingConfig,
    _STAGE_20.WidthSpec,
    _STAGE_20.WidthSpec,
    _STAGE_20.PairedInitialStates,
    tuple[tuple[int, int], ...],
]:
    narrow_config = replace(config, n_embd=config.n_embd // 2)
    narrow = _STAGE_20.WidthSpec(
        name=f"width-{narrow_config.n_embd}",
        n_embd=narrow_config.n_embd,
        checkpoint_path=control_checkpoint_path.with_name("unused_narrow.pt"),
    )
    wide = _STAGE_20.WidthSpec(
        name=f"width-{config.n_embd}",
        n_embd=config.n_embd,
        checkpoint_path=control_checkpoint_path,
    )
    _, bands = _STAGE_20.validate_width_protocol(
        narrow_config,
        narrow,
        wide,
        n_head,
        n_layer,
    )
    initialization = _STAGE_20.create_paired_initial_states(
        data,
        narrow_config,
        config,
        narrow,
        wide,
        n_head,
        n_layer,
    )
    return narrow_config, narrow, wide, initialization, bands


def _position_losses_from_checkpoint(
    checkpoint: Mapping[str, object],
    bands: tuple[tuple[int, int], ...],
) -> PositionLosses:
    saved = _require_mapping(
        checkpoint.get("best_validation_losses"),
        name="best_validation_losses",
    )
    saved_bands = saved.get("bands")
    if not isinstance(saved_bands, list) or len(saved_bands) != len(bands):
        raise ValueError("best validation position bands are incomplete")
    values: list[float] = []
    for item, (start, end) in zip(saved_bands, bands, strict=True):
        mapping = _require_mapping(item, name="best validation band")
        if mapping.get("start") != start or mapping.get("end") != end:
            raise ValueError("best validation band boundaries do not match")
        values.append(float(mapping["loss"]))
    return PositionLosses(
        overall=float(saved["overall"]),
        first_half=float(saved["first_half"]),
        second_half=float(saved["second_half"]),
        bands=tuple(values),
    )


def validate_stage_20_reference(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    *,
    initial_state_sha256: str,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_checkpoint_sha256: str | None = None,
    expected_step: int | None = None,
) -> ReferenceCheckpoint:
    before_hash = checkpoint_sha256(path)
    if expected_checkpoint_sha256 is not None and before_hash != expected_checkpoint_sha256:
        raise ValueError(
            "Stage 20 C=128 checkpoint SHA-256 must be "
            f"{expected_checkpoint_sha256}, got {before_hash}"
        )
    narrow_config, narrow, wide, initialization, bands = _stage_20_protocol(
        data,
        config,
        n_head,
        n_layer,
        path,
    )
    del narrow_config
    if initialization.treatment_state_sha256 != initial_state_sha256:
        raise ValueError("Stage 20 C=128 initialization does not match Stage 21")
    checkpoint = _STAGE_20.validate_stage_20_checkpoint(
        path,
        data,
        config,
        wide,
        narrow,
        wide,
        n_head,
        n_layer,
        schedule,
        bands,
        initialization_seed=config.seed,
        branch_initial_state_sha256=initial_state_sha256,
        control_parameter_count=initialization.control_parameter_count,
        treatment_parameter_count=initialization.treatment_parameter_count,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )
    step = int(checkpoint["step"])
    if expected_step is not None and step != expected_step:
        raise ValueError(
            f"Stage 20 C=128 checkpoint step must be {expected_step}, got {step}"
        )
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 20 checkpoint training generator state is not a tensor")
    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError(f"checkpoint changed during validation: {path}")
    return ReferenceCheckpoint(
        name="p=0",
        dropout=DEFAULT_CONTROL_DROPOUT,
        checkpoint_path=path,
        checkpoint_sha256=before_hash,
        checkpoint_step=step,
        fixed_panel_loss=float(checkpoint["best_val_loss"]),
        best_validation_losses=_position_losses_from_checkpoint(checkpoint, bands),
        training_generator_state=generator_state.detach().cpu().clone(),
        initial_state_sha256=initial_state_sha256,
        parameter_count=initialization.treatment_parameter_count,
    )


@dataclass(slots=True)
class TreatmentValidationCheckpoint(BestValidationCheckpoint):
    treatment_spec: DropoutSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initial_state_sha256: str = ""
    initialization_seed: int = 0
    training_batch_seed: int = 0
    training_rng_seed: int = 0
    control_checkpoint_sha256: str = ""
    control_checkpoint_step: int = 0
    parameter_count: int = 0
    attention_shape: tuple[int, ...] = ()
    bands: tuple[tuple[int, int], ...] = ()
    best_training_losses: PositionLosses | None = None
    best_validation_losses: PositionLosses | None = None

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(TreatmentValidationCheckpoint, self)._payload(step)
        if (
            self.treatment_spec is None
            or self.schedule is None
            or self.best_training_losses is None
            or self.best_validation_losses is None
        ):
            raise RuntimeError("incomplete Stage 21 checkpoint metadata")
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
                "residual_dropout": self.treatment_spec.dropout,
                "dropout_placement": DROPOUT_PLACEMENT,
                "attention_mask": "full_causal",
                "attention_mask_scope": "per_layer",
                "max_context": None,
                "parameter_count": self.parameter_count,
            }
        )
        training = payload["training_config"]
        assert isinstance(training, dict)
        group = self.optimizer.param_groups[0]
        training.update(
            {
                "learning_rate": learning_rate,
                "max_iters": self.target_step,
                "learning_rate_schedule": schedule_metadata,
                "residual_dropout": self.treatment_spec.dropout,
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
            "kind": "shared_with_stage_20_width_128_control",
            "seed": self.initialization_seed,
            "state_sha256": self.initial_state_sha256,
            "identical_learned_initialization": True,
            "dropout_has_no_state": True,
        }
        payload["experiment"] = {
            "stage": 21,
            "branch": self.treatment_spec.name,
            "from_scratch": True,
            "comparison_variable": "residual_dropout",
            "control_residual_dropout": DEFAULT_CONTROL_DROPOUT,
            "treatment_residual_dropout": self.treatment_spec.dropout,
            "dropout_placement": DROPOUT_PLACEMENT,
            "n_embd": self.config.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "identical_parameter_count": True,
            "identical_learned_initialization": True,
            "identical_training_batches": True,
            "identical_fixed_validation_batches": True,
            "same_update_count": True,
            "same_learning_rate_schedule": True,
            "reused_control": True,
            "control_stage": 20,
            "control_branch": f"width-{self.config.n_embd}",
            "control_checkpoint_sha256": self.control_checkpoint_sha256,
            "control_checkpoint_step": self.control_checkpoint_step,
            "initialization_seed": self.initialization_seed,
            "initial_state_sha256": self.initial_state_sha256,
            "training_batch_seed": self.training_batch_seed,
            "training_rng_seed": self.training_rng_seed,
            "learning_rate_schedule": schedule_metadata,
            "position_bands": self.bands,
            "attention_shape": self.attention_shape,
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


def _serialize_position_losses(
    losses: PositionLosses,
    bands: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    return {
        "overall": losses.overall,
        "first_half": losses.first_half,
        "second_half": losses.second_half,
        "bands": [
            {"start": start, "end": end, "loss": loss}
            for (start, end), loss in zip(bands, losses.bands, strict=True)
        ],
    }


def make_evaluation_callback(
    best: TreatmentValidationCheckpoint,
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
                    "treatment batch-generator state does not match the Stage 20 "
                    f"control at step {record.step}"
                )
            verified_reference_steps.add(record.step)
        is_best = best.consider(record)
        assert best.best_step is not None
        learning_rate = schedule.for_update(max(0, record.step - 1))
        train = record.losses.train
        val = record.losses.val
        band_text = " | ".join(
            f"val {start:03d}-{end - 1:03d} {loss:.4f}"
            for (start, end), loss in zip(best.bands, val.bands, strict=True)
        )
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr {learning_rate:g} | "
            f"train {train.overall:.4f} | val {val.overall:.4f} | "
            f"val first {val.first_half:.4f} | val second {val.second_half:.4f} | "
            f"{band_text} | gap {val.overall - train.overall:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )

    return record_evaluation, verified_reference_steps


def _is_evaluation_step(
    step: int,
    config: TrainingConfig,
    schedule: LearningRateSchedule,
) -> bool:
    return (
        step == 0
        or step == config.max_iters
        or step == schedule.first_decay_step
        or step == schedule.second_decay_step
        or step % config.eval_interval == 0
    )


def _validate_serialized_losses(
    value: object,
    bands: tuple[tuple[int, int], ...],
    *,
    name: str,
) -> PositionLosses:
    mapping = _require_mapping(value, name=name)
    for key in ("overall", "first_half", "second_half"):
        item = mapping.get(key)
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{name}.{key} must be finite")
    saved_bands = mapping.get("bands")
    if not isinstance(saved_bands, list) or len(saved_bands) != len(bands):
        raise ValueError(f"{name}.bands are incomplete")
    band_losses: list[float] = []
    for item, (start, end) in zip(saved_bands, bands, strict=True):
        saved = _require_mapping(item, name=f"{name} band")
        if saved.get("start") != start or saved.get("end") != end:
            raise ValueError(f"{name} band boundaries do not match")
        loss = saved.get("loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ValueError(f"{name} band loss must be finite")
        band_losses.append(float(loss))
    return PositionLosses(
        overall=float(mapping["overall"]),
        first_half=float(mapping["first_half"]),
        second_half=float(mapping["second_half"]),
        bands=tuple(band_losses),
    )


def validate_stage_21_checkpoint(
    path: Path,
    treatment: DropoutSpec,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    *,
    initial_state_sha256: str,
    parameter_count: int,
    training_batch_seed: int,
    training_rng_seed: int,
    control_checkpoint_sha256: str,
    control_checkpoint_step: int,
) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 21 checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = _require_mapping(checkpoint, name="checkpoint")
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported Stage 21 checkpoint version")
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("Stage 21 requires a best checkpoint")
    step = checkpoint.get("step")
    _validate_non_negative_integer(step, name="checkpoint step")
    assert isinstance(step, int)
    if step > config.max_iters or not _is_evaluation_step(step, config, schedule):
        raise ValueError("Stage 21 checkpoint step is not a scheduled evaluation")
    if checkpoint.get("best_step") != step:
        raise ValueError("checkpoint best_step must equal its saved step")
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 21 checkpoint must use uninterrupted AdamW")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 21 optimizer provenance must be known")

    architecture = _require_mapping(checkpoint.get("architecture"), name="architecture")
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "residual_dropout": treatment.dropout,
        "dropout_placement": DROPOUT_PLACEMENT,
        "attention_mask": "full_causal",
        "attention_mask_scope": "per_layer",
        "max_context": None,
        "parameter_count": parameter_count,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"checkpoint architecture.{key} must be {expected!r}")

    schedule_metadata = schedule.as_metadata(max_iters=config.max_iters)
    training = _require_mapping(
        checkpoint.get("training_config"), name="training_config"
    )
    expected_training = {
        "batch_size": config.batch_size,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": config.max_iters,
        "learning_rate": schedule.for_update(max(0, step - 1)),
        "learning_rate_schedule": schedule_metadata,
        "residual_dropout": treatment.dropout,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
        "total_target_characters": config.batch_size * config.block_size * config.max_iters,
        "optimizer": "AdamW",
        "optimizer_betas": (0.9, 0.999),
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0.01,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"checkpoint training_config.{key} must be {expected!r}")

    initialization = _require_mapping(
        checkpoint.get("initialization"), name="initialization"
    )
    expected_initialization = {
        "kind": "shared_with_stage_20_width_128_control",
        "seed": config.seed,
        "state_sha256": initial_state_sha256,
        "identical_learned_initialization": True,
        "dropout_has_no_state": True,
    }
    for key, expected in expected_initialization.items():
        if initialization.get(key) != expected:
            raise ValueError(f"checkpoint initialization.{key} must be {expected!r}")

    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    expected_experiment = {
        "stage": 21,
        "branch": treatment.name,
        "from_scratch": True,
        "comparison_variable": "residual_dropout",
        "control_residual_dropout": DEFAULT_CONTROL_DROPOUT,
        "treatment_residual_dropout": treatment.dropout,
        "dropout_placement": DROPOUT_PLACEMENT,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "identical_parameter_count": True,
        "identical_learned_initialization": True,
        "identical_training_batches": True,
        "identical_fixed_validation_batches": True,
        "same_update_count": True,
        "same_learning_rate_schedule": True,
        "reused_control": True,
        "control_stage": 20,
        "control_branch": f"width-{config.n_embd}",
        "control_checkpoint_sha256": control_checkpoint_sha256,
        "control_checkpoint_step": control_checkpoint_step,
        "initialization_seed": config.seed,
        "initial_state_sha256": initial_state_sha256,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "learning_rate_schedule": schedule_metadata,
        "position_bands": bands,
        "attention_shape": (
            config.batch_size,
            n_head,
            config.block_size,
            config.block_size,
        ),
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(f"checkpoint experiment.{key} must be {expected!r}")
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
            raise ValueError(
                f"checkpoint {accelerator_name} RNG states are malformed"
            )

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    with torch.random.fork_rng(devices=[]):
        model = _build_model(
            data,
            config,
            n_head,
            n_layer,
            treatment.dropout,
        )
        model.load_state_dict(model_state, strict=True)
    if count_parameters(model) != parameter_count:
        raise ValueError("checkpoint model parameter count is incorrect")
    expected_mask = torch.tril(torch.ones(config.block_size, config.block_size))
    saved_mask_names = tuple(name for name in model_state if name.endswith(".tril"))
    if len(saved_mask_names) != n_head * n_layer:
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
    if len(dropout_modules) != 2 * n_layer or any(
        module.p != treatment.dropout for module in dropout_modules
    ):
        raise ValueError("checkpoint model dropout sites do not match protocol")

    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"), name="optimizer_state_dict"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=expected_training["learning_rate"])
    try:
        optimizer.load_state_dict(dict(optimizer_state))
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint optimizer state is incompatible") from error
    if step == 0 and optimizer.state:
        raise ValueError("step-zero checkpoint must have empty optimizer state")
    if step > 0 and len(optimizer.state) != len(tuple(model.parameters())):
        raise ValueError("checkpoint optimizer state is incomplete")
    for parameter, state in optimizer.state.items():
        optimizer_step = state.get("step")
        if isinstance(optimizer_step, torch.Tensor):
            if optimizer_step.numel() != 1:
                raise ValueError("optimizer step must be scalar")
            optimizer_step = optimizer_step.item()
        if float(optimizer_step) != step:
            raise ValueError("checkpoint optimizer step is inconsistent")
        for key, item in state.items():
            if key != "step" and (
                not isinstance(item, torch.Tensor) or item.shape != parameter.shape
            ):
                raise ValueError(f"checkpoint optimizer {key} shape is incompatible")

    best_train = _validate_serialized_losses(
        checkpoint.get("best_training_losses"), bands, name="best_training_losses"
    )
    best_val = _validate_serialized_losses(
        checkpoint.get("best_validation_losses"), bands, name="best_validation_losses"
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
    del optimizer
    del model
    return checkpoint


def _load_and_measure_fixed_control(
    reference: ReferenceCheckpoint,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
) -> PositionLossEstimate:
    before_hash = checkpoint_sha256(reference.checkpoint_path)
    checkpoint = torch.load(
        reference.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_mapping(checkpoint, name="Stage 20 checkpoint")
    model = _build_model(
        data,
        config,
        n_head,
        n_layer,
        reference.dropout,
        device,
    )
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    model.load_state_dict(model_state, strict=True)
    losses = evaluate_on_fixed_batches(model, data, config, device, bands)
    if not math.isclose(
        losses.val.overall,
        reference.fixed_panel_loss,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "replayed Stage 20 fixed validation loss does not match checkpoint"
        )
    if checkpoint_sha256(reference.checkpoint_path) != before_hash:
        raise RuntimeError("Stage 20 checkpoint changed during fixed-panel replay")
    del model
    del checkpoint
    clear_accelerator_cache(device)
    return losses


def _assert_position_losses_close(
    actual: PositionLosses,
    expected: PositionLosses,
    *,
    name: str,
) -> None:
    actual_values = (
        actual.overall,
        actual.first_half,
        actual.second_half,
        *actual.bands,
    )
    expected_values = (
        expected.overall,
        expected.first_half,
        expected.second_half,
        *expected.bands,
    )
    for index, (actual_value, expected_value) in enumerate(
        zip(actual_values, expected_values, strict=True)
    ):
        if not math.isclose(actual_value, expected_value, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(
                f"replayed Stage 21 {name} loss {index} does not match the "
                "selected checkpoint"
            )


def replay_treatment_fixed_panel(
    treatment: DropoutSpec,
    expected_checkpoint_sha256: str,
    expected_losses: PositionLossEstimate,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
) -> PositionLossEstimate:
    """Reload the selected treatment weights and tie saved metrics to them."""

    if checkpoint_sha256(treatment.checkpoint_path) != expected_checkpoint_sha256:
        raise RuntimeError("Stage 21 checkpoint changed before fixed-panel replay")
    checkpoint = torch.load(
        treatment.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_mapping(checkpoint, name="Stage 21 checkpoint")
    model = _build_model(
        data,
        config,
        n_head,
        n_layer,
        treatment.dropout,
        device,
    )
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
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
        raise RuntimeError("Stage 21 checkpoint changed during fixed-panel replay")
    del model
    del checkpoint
    clear_accelerator_cache(device)
    return actual


def run_treatment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    parameter_count: int,
    treatment: DropoutSpec,
    control: ReferenceCheckpoint,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
) -> TreatmentReport:
    clear_accelerator_cache(device)
    model = _build_model(
        data,
        config,
        n_head,
        n_layer,
        treatment.dropout,
        device,
    )
    model.load_state_dict(initial_state, strict=True)
    if tensor_mapping_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("treatment did not load the exact Stage 20 initialization")
    if count_parameters(model) != parameter_count:
        raise RuntimeError("treatment parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.for_update(0))
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    attention_shape = (
        config.batch_size,
        n_head,
        config.block_size,
        config.block_size,
    )
    best = TreatmentValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=treatment.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        treatment_spec=treatment,
        schedule=schedule,
        target_step=config.max_iters,
        initial_state_sha256=initial_state_sha256,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        control_checkpoint_sha256=control.checkpoint_sha256,
        control_checkpoint_step=control.checkpoint_step,
        parameter_count=parameter_count,
        attention_shape=attention_shape,
        bands=bands,
    )
    callback, verified_reference_steps = make_evaluation_callback(
        best,
        schedule,
        control,
    )
    print(
        f"\n{treatment.name} treatment (B={config.batch_size}, "
        f"T={config.block_size}, C={config.n_embd}, H={n_head}, "
        f"D={config.n_embd // n_head}, FF={4 * config.n_embd}, L={n_layer}, "
        f"residual dropout={treatment.dropout:g})"
    )
    print(f"  shared initialization: {initial_state_sha256}")
    print(f"  parameters:            {parameter_count:,}")
    print(f"  training-batch seed:   {training_batch_seed}")
    print(f"  dropout/model RNG seed:{training_rng_seed:>8}")

    # Construction and reference replay consumed global RNG. Match Stage 20 by
    # resetting immediately before update 0. The batch generator is separate.
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
            "the Stage 20 control step was not encountered in Stage 21 evaluations"
        )
    expected_final_generator = expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=config.max_iters,
    )
    if not torch.equal(training_generator.get_state(), expected_final_generator):
        raise RuntimeError("treatment did not consume the exact training batch stream")
    measured_attention_shape = inspect_attention_shape(model, config, n_head, device)
    if measured_attention_shape != attention_shape:
        raise RuntimeError("attention shape changed during treatment training")
    assert best.best_step is not None
    assert best.best_training_losses is not None
    assert best.best_validation_losses is not None

    checkpoint = torch.load(
        treatment.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_mapping(checkpoint, name="Stage 21 checkpoint")
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
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
        spec=treatment,
        parameter_count=parameter_count,
        initial_state_sha256=initial_state_sha256,
        training=result,
        best_training_losses=best.best_training_losses,
        best_validation_losses=best.best_validation_losses,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        final_batch_generator_sha256=tensor_sha256(training_generator.get_state()),
        attention_shape=measured_attention_shape,
        sample=sample,
    )
    del checkpoint
    del callback
    del best
    del optimizer
    del model
    clear_accelerator_cache(device)
    return report


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    treatment: DropoutSpec,
    control_checkpoint_path: Path,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_control_sha256: str | None = None,
    expected_control_step: int | None = None,
) -> ExperimentReport:
    bands = validate_protocol(
        config,
        n_head,
        n_layer,
        treatment,
        control_checkpoint_path,
        schedule,
    )
    _validate_non_negative_integer(sample_length, name="sample length")
    _validate_non_negative_integer(training_batch_seed, name="training batch seed")
    _validate_non_negative_integer(training_rng_seed, name="training RNG seed")
    for split, source in (("train", data.train_data), ("val", data.val_data)):
        if len(source) <= config.block_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but block_size "
                f"is {config.block_size}"
            )

    initial_state, initial_hash, parameter_count = create_shared_initial_state(
        data,
        config,
        n_head,
        n_layer,
    )
    expected_count = expected_parameter_count(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_layer=n_layer,
    )
    if parameter_count != expected_count:
        raise RuntimeError(
            f"parameter count must be {expected_count:,}, got {parameter_count:,}"
        )
    control = validate_stage_20_reference(
        control_checkpoint_path,
        data,
        config,
        n_head,
        n_layer,
        schedule,
        initial_state_sha256=initial_hash,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_checkpoint_sha256=expected_control_sha256,
        expected_step=expected_control_step,
    )
    control_hash_before = control.checkpoint_sha256
    control_losses = _load_and_measure_fixed_control(
        control,
        data,
        config,
        n_head,
        n_layer,
        device,
        bands,
    )

    initial_model = _build_model(
        data,
        config,
        n_head,
        n_layer,
        DEFAULT_CONTROL_DROPOUT,
        device,
    )
    initial_model.load_state_dict(initial_state, strict=True)
    initial_losses = evaluate_on_fixed_batches(
        initial_model,
        data,
        config,
        device,
        bands,
    )
    del initial_model
    clear_accelerator_cache(device)

    print("\nControlled C=128 residual-dropout protocol")
    print("  The Stage 20 p=0 best checkpoint is validated and remains read-only.")
    print(
        "  The p=0.05 treatment starts from the byte-identical initialization, "
        "with a fresh AdamW optimizer."
    )
    print(
        "  Training batches, fixed panels, updates, and LR schedule match "
        "Stage 20 exactly."
    )
    print(f"  Shared initialization SHA-256: {initial_hash}")
    print(f"  Stage 20 control SHA-256:      {control.checkpoint_sha256}")

    treatment_report = run_treatment(
        data,
        config,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        bands,
        initial_state,
        initial_hash,
        parameter_count,
        treatment,
        control,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )
    if treatment_report.training.initial != initial_losses:
        raise RuntimeError("p=0 and p=0.05 initial fixed-panel losses differ")
    if checkpoint_sha256(control_checkpoint_path) != control_hash_before:
        raise RuntimeError("Stage 20 control checkpoint changed during training")
    treatment_checkpoint_sha256 = checkpoint_sha256(treatment.checkpoint_path)
    validate_stage_21_checkpoint(
        treatment.checkpoint_path,
        treatment,
        data,
        config,
        n_head,
        n_layer,
        schedule,
        bands,
        initial_state_sha256=initial_hash,
        parameter_count=parameter_count,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        control_checkpoint_sha256=control.checkpoint_sha256,
        control_checkpoint_step=control.checkpoint_step,
    )
    replay_treatment_fixed_panel(
        treatment,
        treatment_checkpoint_sha256,
        PositionLossEstimate(
            train=treatment_report.best_training_losses,
            val=treatment_report.best_validation_losses,
        ),
        data,
        config,
        n_head,
        n_layer,
        device,
        bands,
    )
    if checkpoint_sha256(control_checkpoint_path) != control_hash_before:
        raise RuntimeError("Stage 20 control checkpoint changed during validation")
    return ExperimentReport(
        parameter_count=parameter_count,
        initial_state_sha256=initial_hash,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        bands=bands,
        initial_losses=initial_losses,
        control=ControlReport(reference=control, losses_at_best=control_losses),
        treatment=treatment_report,
    )


def _measure_precise_checkpoint(
    *,
    name: str,
    dropout: float,
    path: Path,
    expected_sha256: str,
    checkpoint: Mapping[str, object],
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    eval_iters: int,
    seed: int,
) -> PreciseBranchResult:
    if checkpoint_sha256(path) != expected_sha256:
        raise RuntimeError(f"checkpoint changed before evaluation: {path}")
    model = _build_model(data, config, n_head, n_layer, dropout, device)
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    model.load_state_dict(model_state, strict=True)
    batch_losses = estimate_validation_precise(
        model,
        data,
        config,
        device,
        bands,
        eval_iters=eval_iters,
        seed=seed,
    )
    if checkpoint_sha256(path) != expected_sha256:
        raise RuntimeError(f"checkpoint changed during evaluation: {path}")
    result = PreciseBranchResult(
        name=name,
        checkpoint_path=path,
        checkpoint_sha256=expected_sha256,
        checkpoint_step=int(checkpoint["step"]),
        fixed_panel_loss=float(checkpoint["best_val_loss"]),
        statistics=_position_statistics(batch_losses),
        batch_losses=batch_losses,
    )
    del model
    clear_accelerator_cache(device)
    return result


def run_precise_validation(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    treatment: DropoutSpec,
    experiment: ExperimentReport,
    *,
    eval_iters: int = DEFAULT_PRECISE_EVAL_ITERS,
    seed: int,
) -> PreciseValidationReport:
    if eval_iters < 2:
        raise ValueError(f"eval_iters must be at least 2, got {eval_iters}")
    _validate_non_negative_integer(seed, name="precise evaluation seed")
    if seed == config.seed + 2:
        raise ValueError("precise evaluation seed must differ from fixed-panel seed")
    bands = validate_protocol(
        config,
        n_head,
        n_layer,
        treatment,
        experiment.control.reference.checkpoint_path,
        schedule,
    )
    if bands != experiment.bands:
        raise ValueError("precise-validation bands differ from training protocol")
    control = validate_stage_20_reference(
        experiment.control.reference.checkpoint_path,
        data,
        config,
        n_head,
        n_layer,
        schedule,
        initial_state_sha256=experiment.initial_state_sha256,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        expected_checkpoint_sha256=(experiment.control.reference.checkpoint_sha256),
        expected_step=experiment.control.reference.checkpoint_step,
    )
    treatment_hash = checkpoint_sha256(treatment.checkpoint_path)
    treatment_checkpoint = validate_stage_21_checkpoint(
        treatment.checkpoint_path,
        treatment,
        data,
        config,
        n_head,
        n_layer,
        schedule,
        bands,
        initial_state_sha256=experiment.initial_state_sha256,
        parameter_count=experiment.parameter_count,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        control_checkpoint_sha256=control.checkpoint_sha256,
        control_checkpoint_step=control.checkpoint_step,
    )
    control_checkpoint = torch.load(
        control.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    control_checkpoint = _require_mapping(control_checkpoint, name="control checkpoint")
    control_result = _measure_precise_checkpoint(
        name="p=0",
        dropout=DEFAULT_CONTROL_DROPOUT,
        path=control.checkpoint_path,
        expected_sha256=control.checkpoint_sha256,
        checkpoint=control_checkpoint,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=bands,
        eval_iters=eval_iters,
        seed=seed,
    )
    treatment_result = _measure_precise_checkpoint(
        name="p=0.05",
        dropout=treatment.dropout,
        path=treatment.checkpoint_path,
        expected_sha256=treatment_hash,
        checkpoint=treatment_checkpoint,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        bands=bands,
        eval_iters=eval_iters,
        seed=seed,
    )
    if checkpoint_sha256(control.checkpoint_path) != control.checkpoint_sha256:
        raise RuntimeError("Stage 20 control checkpoint changed during fresh validation")
    return PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        bands=bands,
        results=(control_result, treatment_result),
        delta=_paired_delta(treatment_result, control_result),
    )


def print_precise_validation(report: PreciseValidationReport) -> None:
    print("\nFresh paired validation of Stage 20/21 best checkpoints")
    print(
        f"  {report.eval_iters} paired validation batches per checkpoint, "
        f"seed={report.seed}"
    )
    print("  Dropout is disabled for every measurement.")
    for result in report.results:
        mean = result.statistics.mean
        se = result.statistics.standard_error
        print(
            f"  {result.name:8s} | step {result.checkpoint_step:5d} | "
            f"fixed {result.fixed_panel_loss:.4f} | overall "
            f"{mean.overall:.6f} +/- {se.overall:.6f} SE | first "
            f"{mean.first_half:.6f} +/- {se.first_half:.6f} | second "
            f"{mean.second_half:.6f} +/- {se.second_half:.6f}"
        )
        for (start, end), band_mean, band_se in zip(
            report.bands,
            mean.bands,
            se.bands,
            strict=True,
        ):
            print(
                f"    positions {start:03d}-{end - 1:03d}: "
                f"{band_mean:.6f} +/- {band_se:.6f} SE"
            )
        print(f"    checkpoint SHA-256: {result.checkpoint_sha256}")
    delta = report.delta
    print(
        f"  Paired delta: {delta.candidate} - {delta.baseline}; negative favors "
        f"{delta.candidate}."
    )
    print(
        "  Overall is primary; half and position-band intervals are "
        "exploratory and not multiplicity-adjusted."
    )
    print(
        "  Intervals quantify validation-batch sampling uncertainty "
        "conditional on these checkpoints, not training-seed or dropout-mask "
        "uncertainty."
    )
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
        for index, (start, end) in enumerate(report.bands)
    )
    for label, mean, standard_error, low, high in rows:
        print(
            f"    {label:21s}: {mean:+.6f} +/- {standard_error:.6f} SE; "
            f"95% CI [{low:+.6f}, {high:+.6f}]"
        )


def print_experiment_summary(report: ExperimentReport) -> None:
    control = report.control
    treatment = report.treatment
    print("\nStage 21 summary")
    print(f"  parameters per branch:       {report.parameter_count:,}")
    print(
        f"  shared initial train / val:  "
        f"{report.initial_losses.train.overall:.4f} / "
        f"{report.initial_losses.val.overall:.4f}"
    )
    print(
        f"  p=0 best train / val / gap:  "
        f"{control.losses_at_best.train.overall:.4f} / "
        f"{control.losses_at_best.val.overall:.4f} / "
        f"{control.generalization_gap_at_best:.4f} @ {control.best_step}"
    )
    print(
        f"  p=0.05 best train/val/gap:   "
        f"{treatment.best_training_losses.overall:.4f} / "
        f"{treatment.best_validation_losses.overall:.4f} / "
        f"{treatment.generalization_gap_at_best:.4f} @ {treatment.best_step}"
    )
    print(
        f"  best fixed-panel delta:      {report.best_val_delta:+.6f} "
        "(p=0.05 - p=0)"
    )
    print(
        f"  p=0.05 final train/val/gap:  "
        f"{treatment.training.final.train.overall:.4f} / "
        f"{treatment.training.final.val.overall:.4f} / "
        f"{treatment.final_generalization_gap:.4f}"
    )
    print(
        "  p=0 final-step weights were not saved by Stage 20; its recorded "
        "18k curve is transcript evidence, not reloaded here."
    )
    print(f"  read-only p=0 checkpoint:    {control.reference.checkpoint_path}")
    print(f"  p=0.05 checkpoint:           {treatment.spec.checkpoint_path}")
    print(f"\np=0.05 best-checkpoint generated text (step {treatment.best_step})")
    print(treatment.sample)


def main() -> None:
    args = parse_args()
    for name, value in (
        ("seed", args.seed),
        ("sample length", args.sample_length),
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
    treatment = DropoutSpec(
        name="dropout-p005",
        dropout=args.dropout,
        checkpoint_path=args.checkpoint_path,
    )
    validate_protocol(
        config,
        args.n_head,
        args.n_layer,
        treatment,
        args.control_checkpoint_path,
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
    control_pin, control_step_pin = _pin_for_default_path(
        args.control_checkpoint_path
    )
    data = CharacterData.from_file(args.data)
    device = resolve_device(args.device)
    print(
        f"Stage 21: C={config.n_embd} residual dropout p=0.05 "
        "from initialization"
    )
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Architecture: B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={args.n_head}, D={config.n_embd // args.n_head}, "
        f"FF={4 * config.n_embd}, L={args.n_layer}"
    )
    print(
        f"Treatment: reused p=0 best checkpoint vs new residual dropout "
        f"p={treatment.dropout:g}"
    )
    print(
        "Schedule by zero-based update: "
        f"[0, {schedule.first_decay_step:,}) lr={schedule.initial_learning_rate:g}; "
        f"[{schedule.first_decay_step:,}, {schedule.second_decay_step:,}) "
        f"lr={schedule.middle_learning_rate:g}; "
        f"[{schedule.second_decay_step:,}, {config.max_iters:,}) "
        f"lr={schedule.final_learning_rate:g}"
    )
    report = run_experiment(
        data,
        config,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        schedule,
        treatment,
        args.control_checkpoint_path,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_control_sha256=control_pin,
        expected_control_step=control_step_pin,
    )
    print_experiment_summary(report)
    if args.precise_eval_iters > 0:
        precise = run_precise_validation(
            data,
            config,
            args.n_head,
            args.n_layer,
            device,
            schedule,
            treatment,
            report,
            eval_iters=args.precise_eval_iters,
            seed=precise_eval_seed,
        )
        print_precise_validation(precise)


if __name__ == "__main__":
    main()
