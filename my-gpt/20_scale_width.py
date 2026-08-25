from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Callable, Mapping
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


# Stage 20 keeps the Stage 18/19 evaluation and statistical machinery, but
# returns to T=64 and makes representation width the only architectural
# treatment. Importing the tested helpers avoids drifting from the established
# fixed-panel, fresh-panel, checkpoint, and benchmark conventions.
_STAGE_18_PATH = Path(__file__).with_name("18_attention_span.py")
_STAGE_18_MODULE_NAME = "stage_18_attention_span_for_stage_20"
_STAGE_18_SPEC = importlib.util.spec_from_file_location(
    _STAGE_18_MODULE_NAME,
    _STAGE_18_PATH,
)
assert _STAGE_18_SPEC is not None and _STAGE_18_SPEC.loader is not None
_STAGE_18 = importlib.util.module_from_spec(_STAGE_18_SPEC)
sys.modules[_STAGE_18_MODULE_NAME] = _STAGE_18
_STAGE_18_SPEC.loader.exec_module(_STAGE_18)

GPTLanguageModel = _STAGE_18.GPTLanguageModel
LearningRateSchedule = _STAGE_18.LearningRateSchedule
BestValidationCheckpoint = _STAGE_18.BestValidationCheckpoint

PositionLosses = _STAGE_18.PositionLosses
PositionLossEstimate = _STAGE_18.PositionLossEstimate
WidthEvaluationRecord = _STAGE_18.AttentionEvaluationRecord
WidthTrainingResult = _STAGE_18.AttentionTrainingResult
PositionStatistics = _STAGE_18.PositionStatistics
PreciseBranchResult = _STAGE_18.PreciseBranchResult
PreciseDelta = _STAGE_18.PreciseDelta
PreciseValidationReport = _STAGE_18.PreciseValidationReport

aggregate_benchmark_stats = _STAGE_18.aggregate_benchmark_stats
checkpoint_sha256 = _STAGE_18.checkpoint_sha256
clear_accelerator_cache = _STAGE_18.clear_accelerator_cache
fingerprint_data = _STAGE_18.fingerprint_data
generate_from_final_model = _STAGE_18.generate_from_final_model
make_position_bands = _STAGE_18.make_position_bands
position_cross_entropy = _STAGE_18.position_cross_entropy
set_optimizer_learning_rate = _STAGE_18.set_optimizer_learning_rate
tensor_mapping_sha256 = _STAGE_18.tensor_mapping_sha256
tensor_sha256 = _STAGE_18.tensor_sha256

_as_position_losses = _STAGE_18._as_position_losses
_mean_position_losses = _STAGE_18._mean_position_losses
_paired_delta = _STAGE_18._paired_delta
_paired_t_95_critical_value = _STAGE_18._paired_t_95_critical_value
_position_statistics = _STAGE_18._position_statistics


CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_20_width_64_best_checkpoint.pt"
)
DEFAULT_TREATMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_20_width_128_best_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = 32
DEFAULT_BLOCK_SIZE = 64
DEFAULT_CONTROL_WIDTH = 64
DEFAULT_TREATMENT_WIDTH = 128
DEFAULT_N_HEAD = 4
DEFAULT_N_LAYER = 4
DEFAULT_MAX_ITERS = 18_000
DEFAULT_INITIAL_LEARNING_RATE = 1e-3
DEFAULT_FIRST_DECAY_STEP = 10_000
DEFAULT_MIDDLE_LEARNING_RATE = 3e-4
DEFAULT_SECOND_DECAY_STEP = 13_000
DEFAULT_FINAL_LEARNING_RATE = 1e-4
DEFAULT_PRECISE_EVAL_ITERS = 500
DEFAULT_DROPOUT = 0.0
DROPOUT_PLACEMENT = _STAGE_18.DROPOUT_PLACEMENT


@dataclass(frozen=True, slots=True)
class WidthSpec:
    """One width branch using the shared batch and context geometry."""

    name: str
    n_embd: int
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch name must not be empty")
        if (
            isinstance(self.n_embd, bool)
            or not isinstance(self.n_embd, int)
            or self.n_embd <= 0
        ):
            raise ValueError(f"n_embd must be a positive integer, got {self.n_embd!r}")


@dataclass(frozen=True, slots=True)
class PairedInitialStates:
    """Architecture-native initial states recreated from one seed."""

    control: dict[str, torch.Tensor]
    treatment: dict[str, torch.Tensor]
    control_state_sha256: str
    treatment_state_sha256: str
    control_parameter_count: int
    treatment_parameter_count: int
    initialization_seed: int


@dataclass(frozen=True, slots=True)
class BranchReport:
    spec: WidthSpec
    parameter_count: int
    initial_state_sha256: str
    training: WidthTrainingResult
    best_val_loss: float
    best_step: int
    best_validation_losses: PositionLosses
    final_batch_generator_sha256: str
    attention_shape: tuple[int, ...]
    sample: str

    @property
    def generalization_gap(self) -> float:
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
    control: BranchReport
    treatment: BranchReport
    control_benchmark: BenchmarkStats | None
    treatment_benchmark: BenchmarkStats | None
    final_val_delta: float
    best_val_delta: float

    @property
    def final_winner(self) -> str:
        if self.final_val_delta < 0:
            return self.treatment.spec.name
        if self.final_val_delta > 0:
            return self.control.spec.name
        return "tie"

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
            "Stage 20: compare C=64 with C=128 while keeping B=32, "
            "T=64, H=4, L=4, dropout=0, the optimizer schedule, and "
            "every sampled target character fixed."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--control-width",
        "--control-n-embd",
        dest="control_width",
        type=int,
        default=DEFAULT_CONTROL_WIDTH,
    )
    parser.add_argument(
        "--treatment-width",
        "--treatment-n-embd",
        dest="treatment_width",
        type=int,
        default=DEFAULT_TREATMENT_WIDTH,
    )
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
        "--final-learning-rate",
        type=float,
        default=DEFAULT_FINAL_LEARNING_RATE,
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
        help="Defaults to seed + 1 and is recreated for each width.",
    )
    parser.add_argument(
        "--training-rng-seed",
        type=int,
        default=None,
        help="Defaults to seed + 3 and is reset before each width.",
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
        help="Defaults to seed + 7, distinct from the fixed panel.",
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


def _validate_non_negative_integer(value: int, *, name: str) -> int:
    return _STAGE_18._validate_non_negative_integer(value, name=name)


def _resolved_output_paths(spec: WidthSpec) -> tuple[Path, Path]:
    checkpoint = spec.checkpoint_path.resolve()
    temporary = spec.checkpoint_path.with_name(
        f".{spec.checkpoint_path.name}.tmp"
    ).resolve()
    return checkpoint, temporary


def validate_width_protocol(
    config: TrainingConfig,
    control: WidthSpec,
    treatment: WidthSpec,
    n_head: int,
    n_layer: int,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    if control.name == treatment.name:
        raise ValueError("control and treatment names must differ")
    if treatment.n_embd != 2 * control.n_embd:
        raise ValueError(
            "Stage 20 requires treatment width to be exactly twice control width"
        )
    if isinstance(n_head, bool) or not isinstance(n_head, int) or n_head <= 0:
        raise ValueError(f"n_head must be a positive integer, got {n_head!r}")
    for spec in (control, treatment):
        if spec.n_embd % n_head != 0:
            raise ValueError(
                f"{spec.name} width ({spec.n_embd}) must be divisible by "
                f"n_head ({n_head})"
            )
    if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer <= 0:
        raise ValueError(f"n_layer must be a positive integer, got {n_layer!r}")
    if config.block_size % 4 != 0:
        raise ValueError(
            "Stage 20 block_size must be divisible by four for position bands"
        )

    paths = {
        f"{control.name} checkpoint": _resolved_output_paths(control)[0],
        f"{control.name} temporary checkpoint": _resolved_output_paths(control)[1],
        f"{treatment.name} checkpoint": _resolved_output_paths(treatment)[0],
        f"{treatment.name} temporary checkpoint": _resolved_output_paths(treatment)[1],
    }
    labels = tuple(paths)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            first_path = paths[first_label]
            second_path = paths[second_label]
            if first_path == second_path:
                raise ValueError(
                    f"{first_label} and {second_label} must use distinct paths"
                )
            if (
                first_path.exists()
                and second_path.exists()
                and first_path.samefile(second_path)
            ):
                raise ValueError(
                    f"{first_label} and {second_label} must not be the same file"
                )

    return config.batch_size * config.block_size, make_position_bands(config.block_size)


def _build_model(
    data: CharacterData,
    config: TrainingConfig,
    spec: WidthSpec,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    if config.n_embd != spec.n_embd:
        raise ValueError(
            f"{spec.name} config width {config.n_embd} does not match spec "
            f"width {spec.n_embd}"
        )
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=spec.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=DEFAULT_DROPOUT,
    )
    if device is not None:
        model = model.to(device)
    return model


def _clone_state(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def expected_parameter_count(
    *,
    vocab_size: int,
    block_size: int,
    n_embd: int,
    n_layer: int,
) -> int:
    """Return the exact count for the repository's untied GPT architecture."""

    embeddings = vocab_size * n_embd + block_size * n_embd
    per_block = 12 * n_embd**2 + 10 * n_embd
    final_layer_norm = 2 * n_embd
    language_model_head = vocab_size * n_embd + vocab_size
    return embeddings + n_layer * per_block + final_layer_norm + language_model_head


def create_paired_initial_states(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    control: WidthSpec,
    treatment: WidthSpec,
    n_head: int,
    n_layer: int,
) -> PairedInitialStates:
    """Create native initializations for both widths from the same seed.

    Width-dependent layers use different fan-in bounds, so copying or slicing
    a wide state into the narrow model would cease to be the standard
    initialization for at least one branch. Each model is therefore freshly
    initialized after resetting the same recorded seed. The resulting states
    are deterministic, independent objects, and intentionally not identical.
    """

    if control_config.seed != treatment_config.seed:
        raise ValueError("branch initialization seeds must match")

    states: list[dict[str, torch.Tensor]] = []
    counts: list[int] = []
    for config, spec in (
        (control_config, control),
        (treatment_config, treatment),
    ):
        seed_everything(config.seed)
        model = _build_model(data, config, spec, n_head, n_layer)
        states.append(_clone_state(model))
        counts.append(count_parameters(model))
        del model

    return PairedInitialStates(
        control=states[0],
        treatment=states[1],
        control_state_sha256=tensor_mapping_sha256(states[0]),
        treatment_state_sha256=tensor_mapping_sha256(states[1]),
        control_parameter_count=counts[0],
        treatment_parameter_count=counts[1],
        initialization_seed=control_config.seed,
    )


def estimate_position_loss(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    split: str,
    bands: tuple[tuple[int, int], ...],
    eval_iters: int,
    generator: torch.Generator,
) -> PositionLosses:
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    values: list[PositionLosses] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(eval_iters):
                inputs, targets = data.get_batch(
                    split,
                    batch_size=config.batch_size,
                    block_size=config.block_size,
                    device=device,
                    generator=generator,
                )
                logits, _ = model(inputs)
                values.append(
                    _as_position_losses(position_cross_entropy(logits, targets, bands))
                )
    finally:
        model.train(was_training)
    return _mean_position_losses(values)


def evaluate_on_fixed_batches(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
) -> PositionLossEstimate:
    # Recreating this generator makes every width and every checkpoint report
    # use precisely the same fixed train/validation examples.
    generator = torch.Generator().manual_seed(config.seed + 2)
    train = estimate_position_loss(
        model,
        data,
        config,
        device,
        split="train",
        bands=bands,
        eval_iters=config.eval_iters,
        generator=generator,
    )
    val = estimate_position_loss(
        model,
        data,
        config,
        device,
        split="val",
        bands=bands,
        eval_iters=config.eval_iters,
        generator=generator,
    )
    return PositionLossEstimate(train=train, val=val)


@dataclass(slots=True)
class WidthValidationCheckpoint(BestValidationCheckpoint):
    """Persist a best Stage 20 state with complete width provenance."""

    branch_spec: WidthSpec | None = None
    control_spec: WidthSpec | None = None
    treatment_spec: WidthSpec | None = None
    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initialization_seed: int = 0
    branch_initial_state_sha256: str = ""
    control_parameter_count: int = 0
    treatment_parameter_count: int = 0
    training_batch_seed: int = 0
    training_rng_seed: int = 0
    attention_shape: tuple[int, ...] = ()
    bands: tuple[tuple[int, int], ...] = ()
    best_validation_losses: PositionLosses | None = None

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(WidthValidationCheckpoint, self)._payload(step)
        if (
            self.branch_spec is None
            or self.control_spec is None
            or self.treatment_spec is None
            or self.schedule is None
            or self.best_validation_losses is None
        ):
            raise RuntimeError("incomplete Stage 20 checkpoint metadata")

        optimizer_rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(optimizer_rates) != 1:
            raise RuntimeError("all optimizer groups must use one LR")
        optimizer_rate = optimizer_rates.pop()
        expected_rate = self.schedule.for_update(max(0, step - 1))
        if optimizer_rate != expected_rate:
            raise RuntimeError(
                f"optimizer LR at step {step} must be {expected_rate:g}, "
                f"got {optimizer_rate:g}"
            )

        schedule_metadata = self.schedule.as_metadata(max_iters=self.target_step)
        payload["checkpoint_kind"] = "best"
        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "head_size": self.branch_spec.n_embd // self.n_head,
                "feed_forward_width": 4 * self.branch_spec.n_embd,
                "residual_dropout": DEFAULT_DROPOUT,
                "dropout_placement": DROPOUT_PLACEMENT,
                "attention_mask": "full_causal",
                "attention_mask_scope": "per_layer",
                "max_context": None,
                "parameter_count": count_parameters(self.model),
            }
        )
        training_config = payload["training_config"]
        assert isinstance(training_config, dict)
        training_config.update(
            {
                "learning_rate": optimizer_rate,
                "max_iters": self.target_step,
                "learning_rate_schedule": schedule_metadata,
                "residual_dropout": DEFAULT_DROPOUT,
                "training_batch_seed": self.training_batch_seed,
                "training_rng_seed": self.training_rng_seed,
                "targets_per_update": (self.config.batch_size * self.config.block_size),
                "total_target_characters": (
                    self.config.batch_size * self.config.block_size * self.target_step
                ),
            }
        )
        payload["initialization"] = {
            "kind": "architecture_native_same_seed",
            "seed": self.initialization_seed,
            "branch_state_sha256": self.branch_initial_state_sha256,
            "identical_initialization_seed": True,
            "identical_model_state": False,
            "width_specific_fan_in_preserved": True,
        }
        payload["experiment"] = {
            "stage": 20,
            "branch": self.branch_spec.name,
            "from_scratch": True,
            "comparison_variable": "embedding_width",
            "branch_n_embd": self.branch_spec.n_embd,
            "control_n_embd": self.control_spec.n_embd,
            "treatment_n_embd": self.treatment_spec.n_embd,
            "batch_size": self.config.batch_size,
            "block_size": self.config.block_size,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "residual_dropout": DEFAULT_DROPOUT,
            "identical_parameter_count": False,
            "identical_learned_initialization": False,
            "identical_position_embeddings": False,
            "identical_batch_geometry": True,
            "identical_targets_per_update": True,
            "identical_training_batches": True,
            "identical_attention_shapes": True,
            "identical_fixed_validation_batches": True,
            "fresh_paired_validation_protocol": True,
            "targets_per_update": (self.config.batch_size * self.config.block_size),
            "total_target_characters": (
                self.config.batch_size * self.config.block_size * self.target_step
            ),
            "control_parameter_count": self.control_parameter_count,
            "treatment_parameter_count": self.treatment_parameter_count,
            "initialization_seed": self.initialization_seed,
            "training_batch_seed": self.training_batch_seed,
            "training_rng_seed": self.training_rng_seed,
            "learning_rate_schedule": schedule_metadata,
            "position_bands": self.bands,
            "attention_shape": self.attention_shape,
        }
        payload["best_validation_losses"] = {
            "overall": self.best_validation_losses.overall,
            "first_half": self.best_validation_losses.first_half,
            "second_half": self.best_validation_losses.second_half,
            "bands": [
                {"start": start, "end": end, "loss": loss}
                for (start, end), loss in zip(
                    self.bands,
                    self.best_validation_losses.bands,
                    strict=True,
                )
            ],
        }
        return payload

    def consider(self, record: WidthEvaluationRecord) -> bool:
        val_loss = record.losses.val.overall
        if not math.isfinite(val_loss):
            raise ValueError(f"validation loss at step {record.step} must be finite")
        if val_loss >= self.best_val_loss:
            return False
        self.best_val_loss = val_loss
        self.best_step = record.step
        self.best_validation_losses = record.losses.val
        self.save(record.step)
        return True


def make_evaluation_callback(
    best: WidthValidationCheckpoint,
    schedule: LearningRateSchedule,
) -> Callable[[WidthEvaluationRecord], None]:
    def record_evaluation(record: WidthEvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        update_index = max(0, record.step - 1)
        learning_rate = schedule.for_update(update_index)
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
            f"val first {val.first_half:.4f} | "
            f"val second {val.second_half:.4f} | {band_text} | "
            f"gap {val.overall - train.overall:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )

    return record_evaluation


def train_with_schedule(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    *,
    on_evaluation: Callable[[WidthEvaluationRecord], None],
    on_update: Callable[[int, float, torch.Tensor, torch.Tensor], None] | None = None,
) -> WidthTrainingResult:
    schedule.validate_target_step(config.max_iters)
    history: list[WidthEvaluationRecord] = []

    def record(step: int) -> WidthEvaluationRecord:
        evaluation = WidthEvaluationRecord(
            step=step,
            losses=evaluate_on_fixed_batches(
                model,
                data,
                config,
                device,
                bands,
            ),
        )
        history.append(evaluation)
        on_evaluation(evaluation)
        return evaluation

    model.train()
    initial_record = record(0)
    for update_index in range(config.max_iters):
        learning_rate = schedule.for_update(update_index)
        set_optimizer_learning_rate(optimizer, learning_rate)
        inputs, targets = data.get_batch(
            "train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=training_generator,
        )
        if on_update is not None:
            on_update(update_index, learning_rate, inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()

        completed_step = update_index + 1
        if (
            completed_step % config.eval_interval == 0
            or completed_step == schedule.first_decay_step
            or completed_step == schedule.second_decay_step
            or completed_step == config.max_iters
        ):
            record(completed_step)

    return WidthTrainingResult(
        initial=initial_record.losses,
        final=history[-1].losses,
        history=tuple(history),
        benchmark=None,
    )


def inspect_attention_shape(
    model: GPTLanguageModel,
    config: TrainingConfig,
    n_head: int,
    device: torch.device,
) -> tuple[int, ...]:
    was_training = model.training
    model.eval()
    try:
        inputs = torch.zeros(
            (config.batch_size, config.block_size),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            weights = model.get_attention_weights(inputs, block_index=0)
        expected = (
            config.batch_size,
            n_head,
            config.block_size,
            config.block_size,
        )
        shape = tuple(weights.shape)
        if shape != expected:
            raise RuntimeError(f"attention shape must be {expected}, got {shape}")
        return shape
    finally:
        model.train(was_training)


def benchmark_width_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: WidthSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    device: torch.device,
    initial_state: Mapping[str, torch.Tensor],
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> BenchmarkStats:
    """Benchmark a disposable model on the real Stage 20 update path."""

    clear_accelerator_cache(device)
    model = _build_model(data, config, spec, n_head, n_layer, device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(training_batch_seed)
    seed_everything(training_rng_seed)
    model.train()

    started_at: float | None = None
    total_steps = benchmark.num_warmup + benchmark.num_steps
    schedule.validate_target_step(config.max_iters)
    if total_steps > config.max_iters:
        raise ValueError(
            f"benchmark requires {total_steps} updates, but max_iters is "
            f"{config.max_iters}"
        )
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

    del optimizer
    del model
    clear_accelerator_cache(device)
    return stats


def run_counterbalanced_benchmarks(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    control: WidthSpec,
    treatment: WidthSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    device: torch.device,
    initialization: PairedInitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    benchmark: BenchmarkConfig,
) -> tuple[BenchmarkStats, BenchmarkStats]:
    """Run adjacent control-treatment-treatment-control disposable trials."""

    control_runs: list[BenchmarkStats] = []
    treatment_runs: list[BenchmarkStats] = []
    ordered_runs = (
        (control_config, control, initialization.control, control_runs),
        (
            treatment_config,
            treatment,
            initialization.treatment,
            treatment_runs,
        ),
        (
            treatment_config,
            treatment,
            initialization.treatment,
            treatment_runs,
        ),
        (control_config, control, initialization.control, control_runs),
    )
    for config, spec, state, destination in ordered_runs:
        destination.append(
            benchmark_width_branch(
                data,
                config,
                spec,
                n_head,
                n_layer,
                schedule,
                device,
                state,
                training_batch_seed=training_batch_seed,
                training_rng_seed=training_rng_seed,
                benchmark=benchmark,
            )
        )

    return (
        aggregate_benchmark_stats(control_runs, steps_per_run=benchmark.num_steps),
        aggregate_benchmark_stats(treatment_runs, steps_per_run=benchmark.num_steps),
    )


def _print_benchmark(name: str, stats: BenchmarkStats) -> None:
    print(f"\n{name} disposable training benchmark")
    print(f"  seconds:               {stats.seconds:.3f}")
    print(f"  iterations/sec:        {stats.iterations_per_sec:.3f}")
    print(f"  target characters/sec: {stats.tokens_per_sec:.3f}")
    print(f"  peak allocated MiB:    {stats.peak_allocated_mb:.3f}")
    print(f"  peak reserved MiB:     {stats.peak_reserved_mb:.3f}")


def _validate_branch_config(
    config: TrainingConfig,
    spec: WidthSpec,
    schedule: LearningRateSchedule,
) -> None:
    if config.n_embd != spec.n_embd:
        raise ValueError(
            f"{spec.name} config width {config.n_embd} does not match "
            f"spec width {spec.n_embd}"
        )
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError(
            f"{spec.name} config learning rate must equal schedule initial LR"
        )
    schedule.validate_target_step(config.max_iters)


def run_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: WidthSpec,
    control: WidthSpec,
    treatment: WidthSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    initialization: PairedInitialStates,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
) -> BranchReport:
    _validate_branch_config(config, spec, schedule)
    clear_accelerator_cache(device)
    model = _build_model(data, config, spec, n_head, n_layer, device)
    model.load_state_dict(initial_state, strict=True)
    if tensor_mapping_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError(f"{spec.name} did not load its exact initial state")

    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.for_update(0))
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    attention_shape = (
        config.batch_size,
        n_head,
        config.block_size,
        config.block_size,
    )
    best = WidthValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=spec.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        branch_spec=spec,
        control_spec=control,
        treatment_spec=treatment,
        schedule=schedule,
        target_step=config.max_iters,
        initialization_seed=initialization.initialization_seed,
        branch_initial_state_sha256=initial_state_sha256,
        control_parameter_count=initialization.control_parameter_count,
        treatment_parameter_count=initialization.treatment_parameter_count,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        attention_shape=attention_shape,
        bands=bands,
    )
    callback = make_evaluation_callback(best, schedule)

    print(
        f"\n{spec.name} branch (B={config.batch_size}, "
        f"T={config.block_size}, C={spec.n_embd}, H={n_head}, "
        f"D={spec.n_embd // n_head}, FF={4 * spec.n_embd}, L={n_layer})"
    )
    print(f"  native initialization: {initial_state_sha256}")
    print(f"  parameters:            {parameter_count:,}")
    print(f"  training-batch seed:   {training_batch_seed}")
    print(f"  training RNG seed:     {training_rng_seed}")

    # Construction consumed global CPU RNG. Reset immediately before updates;
    # the explicit batch generator remains separate from model RNG state.
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
    measured_attention_shape = inspect_attention_shape(model, config, n_head, device)
    if measured_attention_shape != attention_shape:
        raise RuntimeError("attention shape changed during training")

    assert best.best_step is not None
    assert best.best_validation_losses is not None
    final_batch_generator_sha256 = tensor_sha256(training_generator.get_state())

    # Text samples represent the independently selected overall-best state.
    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
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

    report = BranchReport(
        spec=spec,
        parameter_count=parameter_count,
        initial_state_sha256=initial_state_sha256,
        training=result,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        best_validation_losses=best.best_validation_losses,
        final_batch_generator_sha256=final_batch_generator_sha256,
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
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    control: WidthSpec,
    treatment: WidthSpec,
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
    targets_per_update, bands = validate_width_protocol(
        control_config, control, treatment, n_head, n_layer
    )
    _validate_branch_config(control_config, control, schedule)
    _validate_branch_config(treatment_config, treatment, schedule)
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if (
        control_config.batch_size != treatment_config.batch_size
        or control_config.block_size != treatment_config.block_size
        or control_config.max_iters != treatment_config.max_iters
        or control_config.eval_interval != treatment_config.eval_interval
        or control_config.eval_iters != treatment_config.eval_iters
        or control_config.seed != treatment_config.seed
    ):
        raise ValueError("branches may differ only in n_embd within TrainingConfig")
    if benchmark is not None:
        benchmark_steps = benchmark.num_warmup + benchmark.num_steps
        if benchmark_steps > control_config.max_iters:
            raise ValueError(
                f"benchmark requires {benchmark_steps} steps, but max_iters "
                f"is {control_config.max_iters}"
            )
    for split, source in (
        ("train", data.train_data),
        ("val", data.val_data),
    ):
        if len(source) <= control_config.block_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but block_size "
                f"is {control_config.block_size}"
            )
    _validate_non_negative_integer(training_batch_seed, name="training batch seed")
    _validate_non_negative_integer(training_rng_seed, name="training RNG seed")

    initialization = create_paired_initial_states(
        data,
        control_config,
        treatment_config,
        control,
        treatment,
        n_head,
        n_layer,
    )
    print("\nControlled width-scaling protocol")
    print(
        "  Both widths use architecture-native initialization recreated "
        "from the same seed; their tensors are not claimed identical."
    )
    print(
        "  Training batches, fixed evaluation batches, update count, and "
        "learning-rate schedule are paired exactly."
    )
    print(
        f"  Targets per update: {targets_per_update:,}; total per branch: "
        f"{targets_per_update * control_config.max_iters:,}."
    )
    print(
        f"  {control.name} initialization SHA-256: "
        f"{initialization.control_state_sha256}"
    )
    print(
        f"  {treatment.name} initialization SHA-256: "
        f"{initialization.treatment_state_sha256}"
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
            control_config,
            treatment_config,
            control,
            treatment,
            n_head,
            n_layer,
            schedule,
            device,
            initialization,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            benchmark=benchmark,
        )
        _print_benchmark(control.name, control_benchmark)
        _print_benchmark(treatment.name, treatment_benchmark)

    control_report = run_branch(
        data,
        control_config,
        control,
        control,
        treatment,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        bands,
        initialization.control,
        initialization.control_state_sha256,
        initialization,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )
    treatment_report = run_branch(
        data,
        treatment_config,
        treatment,
        control,
        treatment,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        bands,
        initialization.treatment,
        initialization.treatment_state_sha256,
        initialization,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )
    if (
        control_report.final_batch_generator_sha256
        != treatment_report.final_batch_generator_sha256
    ):
        raise RuntimeError("width branches consumed different training batches")

    return ExperimentReport(
        initialization=initialization,
        initialization_seed=control_config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        targets_per_update=targets_per_update,
        bands=bands,
        control=control_report,
        treatment=treatment_report,
        control_benchmark=control_benchmark,
        treatment_benchmark=treatment_benchmark,
        final_val_delta=(
            treatment_report.training.final.val.overall
            - control_report.training.final.val.overall
        ),
        best_val_delta=(treatment_report.best_val_loss - control_report.best_val_loss),
    )


def expected_training_generator_state(
    data: CharacterData,
    config: TrainingConfig,
    *,
    training_batch_seed: int,
    completed_steps: int,
) -> torch.Tensor:
    _validate_non_negative_integer(completed_steps, name="completed steps")
    maximum_start = len(data.train_data) - config.block_size
    if maximum_start <= 0:
        raise ValueError("training split is too short for block_size")
    generator = torch.Generator().manual_seed(training_batch_seed)
    for _ in range(completed_steps):
        torch.randint(
            0,
            maximum_start,
            (config.batch_size,),
            generator=generator,
        )
    return generator.get_state().clone()


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_stage_20_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    spec: WidthSpec,
    control: WidthSpec,
    treatment: WidthSpec,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    bands: tuple[tuple[int, int], ...],
    *,
    initialization_seed: int,
    branch_initial_state_sha256: str,
    control_parameter_count: int,
    treatment_parameter_count: int,
    training_batch_seed: int,
    training_rng_seed: int,
) -> Mapping[str, object]:
    if spec not in (control, treatment):
        raise ValueError("checkpoint branch spec is not part of this protocol")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = _require_mapping(checkpoint, name="checkpoint")

    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported checkpoint version")
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("Stage 20 requires a best checkpoint")
    step = checkpoint.get("step")
    _validate_non_negative_integer(step, name="checkpoint step")
    assert isinstance(step, int)
    if step > config.max_iters:
        raise ValueError("checkpoint step exceeds target max_iters")
    if checkpoint.get("best_step") != step:
        raise ValueError("best checkpoint step and best_step must match")
    best_val_loss_value = checkpoint.get("best_val_loss")
    if (
        isinstance(best_val_loss_value, bool)
        or not isinstance(best_val_loss_value, (int, float))
        or not math.isfinite(float(best_val_loss_value))
    ):
        raise ValueError("checkpoint best_val_loss must be finite")
    best_val_loss = float(best_val_loss_value)
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 20 checkpoints must not restart the optimizer")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 20 optimizer provenance must be known")

    architecture = _require_mapping(checkpoint.get("architecture"), name="architecture")
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": spec.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "head_size": spec.n_embd // n_head,
        "feed_forward_width": 4 * spec.n_embd,
        "residual_dropout": DEFAULT_DROPOUT,
        "dropout_placement": DROPOUT_PLACEMENT,
        "attention_mask": "full_causal",
        "attention_mask_scope": "per_layer",
        "max_context": None,
        "parameter_count": (
            control_parameter_count if spec == control else treatment_parameter_count
        ),
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"checkpoint architecture {key}={architecture.get(key)!r} "
                f"does not match {expected!r}"
            )

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
        "learning_rate_schedule": schedule.as_metadata(max_iters=config.max_iters),
        "residual_dropout": DEFAULT_DROPOUT,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "targets_per_update": config.batch_size * config.block_size,
        "total_target_characters": (
            config.batch_size * config.block_size * config.max_iters
        ),
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"checkpoint training_config {key}={training.get(key)!r} "
                f"does not match {expected!r}"
            )

    initialization = _require_mapping(
        checkpoint.get("initialization"), name="initialization"
    )
    expected_initialization = {
        "kind": "architecture_native_same_seed",
        "seed": initialization_seed,
        "branch_state_sha256": branch_initial_state_sha256,
        "identical_initialization_seed": True,
        "identical_model_state": False,
        "width_specific_fan_in_preserved": True,
    }
    for key, expected in expected_initialization.items():
        if initialization.get(key) != expected:
            raise ValueError(f"checkpoint initialization {key} does not match protocol")

    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    expected_experiment = {
        "stage": 20,
        "branch": spec.name,
        "from_scratch": True,
        "comparison_variable": "embedding_width",
        "branch_n_embd": spec.n_embd,
        "control_n_embd": control.n_embd,
        "treatment_n_embd": treatment.n_embd,
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "n_head": n_head,
        "n_layer": n_layer,
        "residual_dropout": DEFAULT_DROPOUT,
        "identical_parameter_count": False,
        "identical_learned_initialization": False,
        "identical_position_embeddings": False,
        "identical_batch_geometry": True,
        "identical_targets_per_update": True,
        "identical_training_batches": True,
        "identical_attention_shapes": True,
        "identical_fixed_validation_batches": True,
        "fresh_paired_validation_protocol": True,
        "targets_per_update": config.batch_size * config.block_size,
        "total_target_characters": (
            config.batch_size * config.block_size * config.max_iters
        ),
        "control_parameter_count": control_parameter_count,
        "treatment_parameter_count": treatment_parameter_count,
        "initialization_seed": initialization_seed,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "learning_rate_schedule": schedule.as_metadata(max_iters=config.max_iters),
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
            raise ValueError(
                f"checkpoint experiment {key}={experiment.get(key)!r} "
                f"does not match {expected!r}"
            )

    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("checkpoint data fingerprint does not match corpus")
    rng_state = _require_mapping(checkpoint.get("rng_state"), name="rng_state")
    cpu_rng_state = rng_state.get("cpu")
    if not isinstance(cpu_rng_state, torch.Tensor):
        raise ValueError("checkpoint CPU RNG state must be a tensor")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(training_rng_seed)
        expected_cpu_rng_state = torch.get_rng_state().clone()
    if not torch.equal(cpu_rng_state.cpu(), expected_cpu_rng_state):
        raise ValueError("checkpoint CPU RNG state does not match the protocol")
    for accelerator_name in ("xpu", "cuda"):
        accelerator_states = rng_state.get(accelerator_name)
        if accelerator_states is not None and (
            not isinstance(accelerator_states, list)
            or not accelerator_states
            or not all(isinstance(state, torch.Tensor) for state in accelerator_states)
        ):
            raise ValueError(f"checkpoint {accelerator_name} RNG states are malformed")
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("training_generator_state must be a tensor")
    expected_generator_state = expected_training_generator_state(
        data,
        config,
        training_batch_seed=training_batch_seed,
        completed_steps=step,
    )
    if not torch.equal(generator_state.cpu(), expected_generator_state):
        raise ValueError(
            "checkpoint training generator state does not match the paired "
            f"batch stream at step {step}"
        )

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"), name="model_state_dict"
    )
    with torch.random.fork_rng(devices=[]):
        model = _build_model(data, config, spec, n_head, n_layer)
        model.load_state_dict(model_state, strict=True)
        if count_parameters(model) != expected_architecture["parameter_count"]:
            raise ValueError("checkpoint model has an incorrect parameter count")
    expected_mask = torch.tril(torch.ones(config.block_size, config.block_size))
    saved_mask_names = tuple(name for name in model_state if name.endswith(".tril"))
    if len(saved_mask_names) != n_head * n_layer:
        raise ValueError("checkpoint has an incorrect number of causal masks")
    for name in saved_mask_names:
        saved_mask = model_state[name]
        if not isinstance(saved_mask, torch.Tensor):
            raise ValueError(f"checkpoint mask {name!r} must be a tensor")
        try:
            torch.testing.assert_close(saved_mask.cpu(), expected_mask, rtol=0, atol=0)
        except AssertionError as error:
            raise ValueError(
                f"checkpoint contains an incorrect causal mask: {name}"
            ) from error

    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"), name="optimizer_state_dict"
    )
    parameter_groups = optimizer_state.get("param_groups")
    if not isinstance(parameter_groups, list) or not parameter_groups:
        raise ValueError("optimizer checkpoint must contain parameter groups")
    if not all(
        isinstance(group, Mapping) and "lr" in group for group in parameter_groups
    ):
        raise ValueError("every optimizer parameter group must contain an LR")
    try:
        optimizer_rates = {
            float(group["lr"])
            for group in parameter_groups
            if isinstance(group, Mapping)
        }
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint optimizer LR must be numeric") from error
    expected_rate = schedule.for_update(max(0, step - 1))
    if optimizer_rates != {expected_rate}:
        raise ValueError(
            f"checkpoint optimizer LR must be {expected_rate:g}, got "
            f"{sorted(optimizer_rates)}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=expected_rate)
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
                raise ValueError("checkpoint optimizer step must be scalar")
            optimizer_step = optimizer_step.item()
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, (int, float))
            or float(optimizer_step) != step
        ):
            raise ValueError("checkpoint optimizer step is inconsistent")
        for key, value in state.items():
            if key == "step":
                continue
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"checkpoint optimizer {key} shape is incompatible")

    best_losses = _require_mapping(
        checkpoint.get("best_validation_losses"),
        name="best_validation_losses",
    )
    for key in ("overall", "first_half", "second_half"):
        value = best_losses.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"best validation loss {key!r} must be finite")
    if float(best_losses["overall"]) != best_val_loss:
        raise ValueError("best validation loss metadata is inconsistent")
    saved_bands = best_losses.get("bands")
    if not isinstance(saved_bands, list) or len(saved_bands) != len(bands):
        raise ValueError("best validation band metadata is incomplete")
    for saved, (start, end) in zip(saved_bands, bands, strict=True):
        mapping = _require_mapping(saved, name="position band")
        if mapping.get("start") != start or mapping.get("end") != end:
            raise ValueError("best validation band boundaries are inconsistent")
        loss = mapping.get("loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ValueError("best validation band loss must be finite")

    del optimizer
    del model
    return checkpoint


@torch.no_grad()
def estimate_validation_precise(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    bands: tuple[tuple[int, int], ...],
    *,
    eval_iters: int,
    seed: int,
) -> tuple[PositionLosses, ...]:
    if eval_iters < 2:
        raise ValueError(
            f"eval_iters must be at least 2 for paired inference, got {eval_iters}"
        )
    _validate_non_negative_integer(seed, name="precise evaluation seed")
    if seed == config.seed + 2:
        raise ValueError(
            "precise evaluation seed must differ from the fixed-panel seed"
        )
    generator = torch.Generator().manual_seed(seed)
    values: list[PositionLosses] = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(eval_iters):
            inputs, targets = data.get_batch(
                "val",
                batch_size=config.batch_size,
                block_size=config.block_size,
                device=device,
                generator=generator,
            )
            logits, _ = model(inputs)
            values.append(
                _as_position_losses(position_cross_entropy(logits, targets, bands))
            )
    finally:
        model.train(was_training)
    return tuple(values)


def run_precise_validation(
    data: CharacterData,
    control_config: TrainingConfig,
    treatment_config: TrainingConfig,
    control: WidthSpec,
    treatment: WidthSpec,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    initialization: PairedInitialStates,
    bands: tuple[tuple[int, int], ...],
    *,
    initialization_seed: int,
    training_batch_seed: int,
    training_rng_seed: int,
    eval_iters: int,
    seed: int,
) -> PreciseValidationReport:
    if eval_iters < 2:
        raise ValueError(
            f"eval_iters must be at least 2 for paired inference, got {eval_iters}"
        )
    _validate_non_negative_integer(seed, name="precise evaluation seed")
    _, expected_bands = validate_width_protocol(
        control_config,
        control,
        treatment,
        n_head,
        n_layer,
    )
    _validate_branch_config(control_config, control, schedule)
    _validate_branch_config(treatment_config, treatment, schedule)
    if (
        control_config.batch_size != treatment_config.batch_size
        or control_config.block_size != treatment_config.block_size
        or control_config.max_iters != treatment_config.max_iters
        or control_config.eval_interval != treatment_config.eval_interval
        or control_config.eval_iters != treatment_config.eval_iters
        or control_config.seed != treatment_config.seed
    ):
        raise ValueError(
            "precise-validation branches must share batch and evaluation geometry"
        )
    if bands != expected_bands:
        raise ValueError("precise-validation position bands are inconsistent")
    if seed in (control_config.seed + 2, treatment_config.seed + 2):
        raise ValueError(
            "precise evaluation seed must differ from the fixed-panel seed"
        )
    results: list[PreciseBranchResult] = []
    branches = (
        (
            control,
            control_config,
            initialization.control_state_sha256,
        ),
        (
            treatment,
            treatment_config,
            initialization.treatment_state_sha256,
        ),
    )
    for spec, config, initial_hash in branches:
        before_hash = checkpoint_sha256(spec.checkpoint_path)
        checkpoint = validate_stage_20_checkpoint(
            spec.checkpoint_path,
            data,
            config,
            spec,
            control,
            treatment,
            n_head,
            n_layer,
            schedule,
            bands,
            initialization_seed=initialization_seed,
            branch_initial_state_sha256=initial_hash,
            control_parameter_count=initialization.control_parameter_count,
            treatment_parameter_count=initialization.treatment_parameter_count,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
        )
        model = _build_model(data, config, spec, n_head, n_layer, device)
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
        if checkpoint_sha256(spec.checkpoint_path) != before_hash:
            raise RuntimeError(
                f"checkpoint changed during evaluation: {spec.checkpoint_path}"
            )
        results.append(
            PreciseBranchResult(
                name=spec.name,
                checkpoint_path=spec.checkpoint_path,
                checkpoint_sha256=before_hash,
                checkpoint_step=int(checkpoint["step"]),
                fixed_panel_loss=float(checkpoint["best_val_loss"]),
                statistics=_position_statistics(batch_losses),
                batch_losses=batch_losses,
            )
        )
        del model
        clear_accelerator_cache(device)

    return PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        bands=bands,
        results=tuple(results),
        delta=_paired_delta(results[1], results[0]),
    )


def print_precise_validation(report: PreciseValidationReport) -> None:
    print("\nFresh paired validation of Stage 20 best checkpoints")
    print(
        f"  {report.eval_iters} paired validation batches per checkpoint, "
        f"seed={report.seed}"
    )
    for result in report.results:
        mean = result.statistics.mean
        se = result.statistics.standard_error
        print(
            f"  {result.name:12s} | step {result.checkpoint_step:5d} | "
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
        f"  Paired delta: {delta.candidate} - {delta.baseline}; "
        f"two-sided Student-t 95% intervals (critical "
        f"{delta.critical_value:.6f})"
    )
    print(f"  Negative deltas favor {delta.candidate}.")
    print(
        "  Overall is primary; half and position-band intervals are "
        "exploratory and not multiplicity-adjusted."
    )
    print(
        "  Intervals quantify validation-batch sampling uncertainty "
        "conditional on these checkpoints, not training-seed uncertainty."
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
    print("\nStage 20 experimental summary")
    print(
        f"  paired targets/update: {report.targets_per_update:,}; "
        f"training-batch state identical: "
        f"{report.control.final_batch_generator_sha256 == report.treatment.final_batch_generator_sha256}"
    )
    for branch in (report.control, report.treatment):
        final = branch.training.final
        print(
            f"  {branch.spec.name:12s} | C={branch.spec.n_embd:3d} | "
            f"params {branch.parameter_count:9,d} | final train "
            f"{final.train.overall:.4f} | final val {final.val.overall:.4f} | "
            f"best {branch.best_val_loss:.4f} @ {branch.best_step}"
        )
    print(
        f"  final treatment - control validation delta: "
        f"{report.final_val_delta:+.6f}; lower fixed-panel point estimate: "
        f"{report.final_winner}"
    )
    print(
        f"  best treatment - control validation delta:  "
        f"{report.best_val_delta:+.6f}; nominal lower point estimate: "
        f"{report.best_winner}"
    )
    if report.control_benchmark is not None and report.treatment_benchmark is not None:
        throughput_ratio = (
            report.treatment_benchmark.iterations_per_sec
            / report.control_benchmark.iterations_per_sec
        )
        print(f"  treatment/control iteration-rate ratio: {throughput_ratio:.3f}x")
    print(f"  control checkpoint:   {report.control.spec.checkpoint_path}")
    print(f"  treatment checkpoint: {report.treatment.spec.checkpoint_path}")
    print(
        f"\n{report.control.spec.name} best-checkpoint generated text "
        f"(step {report.control.best_step})"
    )
    print(report.control.sample)
    print(
        f"\n{report.treatment.spec.name} best-checkpoint generated text "
        f"(step {report.treatment.best_step})"
    )
    print(report.treatment.sample)


def main() -> None:
    args = parse_args()
    _validate_non_negative_integer(args.seed, name="seed")
    _validate_non_negative_integer(args.sample_length, name="sample length")
    _validate_non_negative_integer(args.benchmark_warmup, name="benchmark warmup")
    _validate_non_negative_integer(args.benchmark_steps, name="benchmark steps")
    _validate_non_negative_integer(args.precise_eval_iters, name="precise eval iters")
    if args.precise_eval_iters == 1:
        raise ValueError("precise-eval-iters must be 0 or at least 2")

    schedule = LearningRateSchedule(
        initial_learning_rate=args.initial_learning_rate,
        first_decay_step=args.first_decay_step,
        middle_learning_rate=args.middle_learning_rate,
        second_decay_step=args.second_decay_step,
        final_learning_rate=args.final_learning_rate,
    )
    control_config = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.control_width,
        learning_rate=args.initial_learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    treatment_config = replace(control_config, n_embd=args.treatment_width)
    control = WidthSpec(
        f"width-{args.control_width}",
        args.control_width,
        args.control_checkpoint_path,
    )
    treatment = WidthSpec(
        f"width-{args.treatment_width}",
        args.treatment_width,
        args.treatment_checkpoint_path,
    )
    validate_width_protocol(
        control_config,
        control,
        treatment,
        args.n_head,
        args.n_layer,
    )
    schedule.validate_target_step(args.max_iters)
    training_batch_seed = (
        args.seed + 1 if args.training_batch_seed is None else args.training_batch_seed
    )
    training_rng_seed = (
        args.seed + 3 if args.training_rng_seed is None else args.training_rng_seed
    )
    precise_eval_seed = (
        args.seed + 7 if args.precise_eval_seed is None else args.precise_eval_seed
    )
    _validate_non_negative_integer(training_batch_seed, name="training batch seed")
    _validate_non_negative_integer(training_rng_seed, name="training RNG seed")
    _validate_non_negative_integer(precise_eval_seed, name="precise eval seed")
    if args.precise_eval_iters > 0 and precise_eval_seed == args.seed + 2:
        raise ValueError(
            f"precise-eval-seed must differ from the fixed-panel seed ({args.seed + 2})"
        )

    data = CharacterData.from_file(args.data)
    device = resolve_device(args.device)
    benchmark = None
    if args.benchmark_steps > 0:
        benchmark = BenchmarkConfig(
            num_warmup=args.benchmark_warmup,
            num_steps=args.benchmark_steps,
        )

    print(
        "Stage 20: controlled width scaling "
        f"C={args.control_width} -> C={args.treatment_width}"
    )
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Shared B={args.batch_size}, T={args.block_size}, H={args.n_head}, "
        f"L={args.n_layer}, dropout={DEFAULT_DROPOUT:g}"
    )
    print(
        f"Control C={control.n_embd}, D={control.n_embd // args.n_head}, "
        f"FF={4 * control.n_embd}"
    )
    print(
        f"Treatment C={treatment.n_embd}, "
        f"D={treatment.n_embd // args.n_head}, "
        f"FF={4 * treatment.n_embd}"
    )
    print(
        "Schedule by zero-based update: "
        f"[0, {schedule.first_decay_step:,}) "
        f"lr={schedule.initial_learning_rate:g}; "
        f"[{schedule.first_decay_step:,}, "
        f"{schedule.second_decay_step:,}) "
        f"lr={schedule.middle_learning_rate:g}; "
        f"[{schedule.second_decay_step:,}, {args.max_iters:,}) "
        f"lr={schedule.final_learning_rate:g}"
    )

    report = run_experiment(
        data,
        control_config,
        treatment_config,
        control,
        treatment,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        schedule,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        benchmark=benchmark,
    )
    print_experiment_summary(report)

    if args.precise_eval_iters > 0:
        precise = run_precise_validation(
            data,
            control_config,
            treatment_config,
            control,
            treatment,
            args.n_head,
            args.n_layer,
            device,
            schedule,
            report.initialization,
            report.bands,
            initialization_seed=args.seed,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            eval_iters=args.precise_eval_iters,
            seed=precise_eval_seed,
        )
        print_precise_validation(precise)


if __name__ == "__main__":
    main()
