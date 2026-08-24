from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import EvaluationRecord, TrainingResult, seed_everything


# Stage 15 changes the training protocol, not the architecture. Import the
# exact residual-path dropout implementation tested in Stage 14 so placement
# remains the only established pair of sites: attention output projection and
# FFN output projection, immediately before their residual additions.
_STAGE_14_PATH = Path(__file__).with_name("14_dropout.py")
_STAGE_14_MODULE_NAME = "stage_14_dropout_for_stage_15"
_STAGE_14_SPEC = importlib.util.spec_from_file_location(
    _STAGE_14_MODULE_NAME,
    _STAGE_14_PATH,
)
assert _STAGE_14_SPEC is not None and _STAGE_14_SPEC.loader is not None
_STAGE_14 = importlib.util.module_from_spec(_STAGE_14_SPEC)
sys.modules[_STAGE_14_MODULE_NAME] = _STAGE_14
_STAGE_14_SPEC.loader.exec_module(_STAGE_14)

GPTLanguageModel = _STAGE_14.GPTLanguageModel
BranchSpec = _STAGE_14.BranchSpec
PreciseValidationResult = _STAGE_14.PreciseValidationResult
PreciseDelta = _STAGE_14.PreciseDelta
PreciseValidationReport = _STAGE_14.PreciseValidationReport

BestValidationCheckpoint = _STAGE_14._STAGE_11.BestValidationCheckpoint
checkpoint_sha256 = _STAGE_14.checkpoint_sha256
clear_accelerator_cache = _STAGE_14.clear_accelerator_cache
estimate_validation_precise = _STAGE_14.estimate_validation_precise
evaluate_on_fixed_batches = _STAGE_14.evaluate_on_fixed_batches
fingerprint_data = _STAGE_14.fingerprint_data
generate_from_final_model = _STAGE_14.generate_from_final_model

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_15_control_best_checkpoint.pt"
)
DEFAULT_DROPOUT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_15_dropout_best_checkpoint.pt"
)
DEFAULT_MAX_ITERS = 18_000
DEFAULT_INITIAL_LEARNING_RATE = 1e-3
DEFAULT_FIRST_DECAY_STEP = 10_000
DEFAULT_MIDDLE_LEARNING_RATE = 3e-4
DEFAULT_SECOND_DECAY_STEP = 13_000
DEFAULT_FINAL_LEARNING_RATE = 1e-4
DEFAULT_CONTROL_DROPOUT = 0.0
DEFAULT_RESIDUAL_DROPOUT = 0.1
DEFAULT_PRECISE_EVAL_ITERS = 500
DROPOUT_PLACEMENT = _STAGE_14.DROPOUT_PLACEMENT


def _validate_learning_rate(value: float, *, name: str) -> float:
    learning_rate = float(value)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return learning_rate


def _validate_non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class LearningRateSchedule:
    """The three-phase schedule, indexed by zero-based optimizer update."""

    initial_learning_rate: float = DEFAULT_INITIAL_LEARNING_RATE
    first_decay_step: int = DEFAULT_FIRST_DECAY_STEP
    middle_learning_rate: float = DEFAULT_MIDDLE_LEARNING_RATE
    second_decay_step: int = DEFAULT_SECOND_DECAY_STEP
    final_learning_rate: float = DEFAULT_FINAL_LEARNING_RATE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_learning_rate",
            _validate_learning_rate(
                self.initial_learning_rate,
                name="initial learning rate",
            ),
        )
        object.__setattr__(
            self,
            "middle_learning_rate",
            _validate_learning_rate(
                self.middle_learning_rate,
                name="middle learning rate",
            ),
        )
        object.__setattr__(
            self,
            "final_learning_rate",
            _validate_learning_rate(
                self.final_learning_rate,
                name="final learning rate",
            ),
        )
        _validate_non_negative_integer(
            self.first_decay_step,
            name="first decay step",
        )
        _validate_non_negative_integer(
            self.second_decay_step,
            name="second decay step",
        )
        if self.first_decay_step == 0:
            raise ValueError("first decay step must be positive")
        if self.second_decay_step <= self.first_decay_step:
            raise ValueError("second decay step must be greater than first decay step")
        if not (
            self.initial_learning_rate
            > self.middle_learning_rate
            > self.final_learning_rate
        ):
            raise ValueError(
                "learning rates must strictly decrease across the schedule"
            )

    def for_update(self, update_index: int) -> float:
        """Return the LR for a zero-based optimizer update index.

        With the defaults, update indices 0..9,999 use 1e-3,
        10,000..12,999 use 3e-4, and 13,000 onward use 1e-4. Thus the
        model state called step 10,000 has completed exactly 10,000 updates
        at 1e-3; its next update uses 3e-4.
        """

        index = _validate_non_negative_integer(
            update_index,
            name="update index",
        )
        if index < self.first_decay_step:
            return self.initial_learning_rate
        if index < self.second_decay_step:
            return self.middle_learning_rate
        return self.final_learning_rate

    def validate_target_step(self, max_iters: int) -> None:
        target = _validate_non_negative_integer(max_iters, name="max iters")
        if target <= self.second_decay_step:
            raise ValueError(
                f"max iters ({target}) must exceed second decay step "
                f"({self.second_decay_step})"
            )

    def as_metadata(self, *, max_iters: int) -> list[dict[str, object]]:
        self.validate_target_step(max_iters)
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
                "end_update_exclusive": max_iters,
                "learning_rate": self.final_learning_rate,
            },
        ]


@dataclass(slots=True)
class PreparedBranch:
    model: GPTLanguageModel
    optimizer: torch.optim.Optimizer
    training_generator: torch.Generator
    spec: BranchSpec


@dataclass(slots=True)
class BranchValidationCheckpoint(BestValidationCheckpoint):
    """Persist a best Stage 15 state with the complete paired protocol."""

    branch_name: str = ""
    branch_dropout: float = 0.0
    initial_state_sha256: str = ""
    initialization_seed: int = 0
    training_batch_seed: int = 0
    training_rng_seed: int = 0
    schedule: LearningRateSchedule | None = None
    target_step: int = 0

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(BranchValidationCheckpoint, self)._payload(step)
        if self.schedule is None:
            raise RuntimeError("Stage 15 checkpoint requires an LR schedule")

        learning_rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(learning_rates) != 1:
            raise RuntimeError(
                "all optimizer parameter groups must share one learning rate"
            )
        optimizer_learning_rate = learning_rates.pop()
        last_update_index = max(0, step - 1)
        expected_learning_rate = self.schedule.for_update(last_update_index)
        if optimizer_learning_rate != expected_learning_rate:
            raise RuntimeError(
                f"optimizer learning rate at completed step {step} must be "
                f"{expected_learning_rate:g}, got "
                f"{optimizer_learning_rate:g}"
            )
        schedule_metadata = self.schedule.as_metadata(max_iters=self.target_step)

        payload["checkpoint_kind"] = "best"

        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "residual_dropout": self.branch_dropout,
                "dropout_placement": DROPOUT_PLACEMENT,
            }
        )

        training_config = payload["training_config"]
        assert isinstance(training_config, dict)
        training_config.update(
            {
                "learning_rate": optimizer_learning_rate,
                "max_iters": self.target_step,
                "learning_rate_schedule": schedule_metadata,
                "residual_dropout": self.branch_dropout,
                "training_batch_seed": self.training_batch_seed,
                "training_rng_seed": self.training_rng_seed,
            }
        )

        payload["initialization"] = {
            "kind": "shared_random_initialization",
            "seed": self.initialization_seed,
            "state_sha256": self.initial_state_sha256,
        }
        payload["experiment"] = {
            "stage": 15,
            "branch": self.branch_name,
            "from_scratch": True,
            "comparison_variable": "residual_dropout",
            "branch_residual_dropout": self.branch_dropout,
            "dropout_placement": DROPOUT_PLACEMENT,
            "identical_initialization": True,
            "initial_state_sha256": self.initial_state_sha256,
            "initialization_seed": self.initialization_seed,
            "identical_training_batches": True,
            "training_batch_seed": self.training_batch_seed,
            "training_rng_seed": self.training_rng_seed,
            "learning_rate_schedule": schedule_metadata,
        }
        return payload


@dataclass(frozen=True, slots=True)
class BranchReport:
    name: str
    dropout: float
    training: TrainingResult
    generalization_gap: float
    best_val_loss: float
    best_step: int
    checkpoint_path: Path
    final_batch_generator_sha256: str
    sample: str


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    parameter_count: int
    initial_state_sha256: str
    initialization_seed: int
    training_batch_seed: int
    training_rng_seed: int
    schedule: LearningRateSchedule
    control: BranchReport
    dropout: BranchReport
    final_val_delta: float
    best_val_delta: float

    @property
    def final_winner(self) -> str:
        if self.final_val_delta < 0:
            return self.dropout.name
        if self.final_val_delta > 0:
            return self.control.name
        return "tie"

    @property
    def best_winner(self) -> str:
        if self.best_val_delta < 0:
            return self.dropout.name
        if self.best_val_delta > 0:
            return self.control.name
        return "tie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train identical models from the same initialization and on the "
            "same batches, comparing residual dropout p=0.0 versus p=0.1."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument(
        "--initial-learning-rate",
        type=float,
        default=DEFAULT_INITIAL_LEARNING_RATE,
    )
    parser.add_argument(
        "--first-decay-step",
        type=int,
        default=DEFAULT_FIRST_DECAY_STEP,
        help="First zero-based update index that uses the middle LR.",
    )
    parser.add_argument(
        "--middle-learning-rate",
        type=float,
        default=DEFAULT_MIDDLE_LEARNING_RATE,
    )
    parser.add_argument(
        "--second-decay-step",
        type=int,
        default=DEFAULT_SECOND_DECAY_STEP,
        help="First zero-based update index that uses the final LR.",
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=DEFAULT_FINAL_LEARNING_RATE,
    )
    parser.add_argument(
        "--control-dropout",
        type=float,
        default=DEFAULT_CONTROL_DROPOUT,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_RESIDUAL_DROPOUT,
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
        help="Defaults to seed + 1; owns only training-batch sampling.",
    )
    parser.add_argument(
        "--training-rng-seed",
        type=int,
        default=None,
        help="Defaults to seed + 3; reset before each branch's updates.",
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
        help="Defaults to seed + 4, distinct from checkpoint selection.",
    )
    parser.add_argument(
        "--control-checkpoint-path",
        type=Path,
        default=DEFAULT_CONTROL_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--dropout-checkpoint-path",
        type=Path,
        default=DEFAULT_DROPOUT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def tensor_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and exact bytes in a state dict."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} must be a tensor")
        contiguous = tensor.detach().cpu().contiguous()
        raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        for component in (
            name.encode("utf-8"),
            str(contiguous.dtype).encode("ascii"),
            repr(tuple(contiguous.shape)).encode("ascii"),
            raw,
        ):
            digest.update(len(component).to_bytes(8, "big"))
            digest.update(component)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return tensor_mapping_sha256({"tensor": tensor})


def create_shared_initial_state(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
) -> tuple[dict[str, torch.Tensor], str, int]:
    """Create one CPU initialization that both branches will clone."""

    seed_everything(config.seed)
    base_model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=0.0,
    )
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in base_model.state_dict().items()
    }
    parameter_count = sum(
        parameter.numel()
        for parameter in base_model.parameters()
        if parameter.requires_grad
    )
    state_hash = tensor_mapping_sha256(state)
    del base_model
    return state, state_hash, parameter_count


def prepare_branch(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    initial_state: Mapping[str, torch.Tensor],
    spec: BranchSpec,
    *,
    training_batch_seed: int,
) -> PreparedBranch:
    """Build an independent model, optimizer, and batch RNG for one branch."""

    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=spec.dropout,
    ).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.for_update(0),
    )
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    return PreparedBranch(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        spec=spec,
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    rate = _validate_learning_rate(learning_rate, name="learning rate")
    if not optimizer.param_groups:
        raise ValueError("optimizer must contain at least one parameter group")
    for group in optimizer.param_groups:
        group["lr"] = rate


def train_with_schedule(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    schedule: LearningRateSchedule,
    *,
    on_evaluation: Callable[[EvaluationRecord], None],
    on_update: Callable[[int, float], None] | None = None,
) -> TrainingResult:
    """Train from step zero while applying LR before each indexed update."""

    schedule.validate_target_step(config.max_iters)
    history: list[EvaluationRecord] = []

    def record(step: int) -> EvaluationRecord:
        evaluation = EvaluationRecord(
            step=step,
            losses=evaluate_on_fixed_batches(model, data, config, device),
        )
        history.append(evaluation)
        on_evaluation(evaluation)
        return evaluation

    model.train()
    initial_record = record(0)
    for update_index in range(config.max_iters):
        learning_rate = schedule.for_update(update_index)
        set_optimizer_learning_rate(optimizer, learning_rate)
        if on_update is not None:
            on_update(update_index, learning_rate)

        inputs, targets = data.get_batch(
            "train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=training_generator,
        )
        _, loss = model(inputs, targets)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
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

    return TrainingResult(
        initial=initial_record.losses,
        final=history[-1].losses,
        history=tuple(history),
    )


def validate_distinct_output_paths(
    control_checkpoint: Path,
    dropout_checkpoint: Path,
) -> None:
    labeled_paths = {
        "control checkpoint": control_checkpoint.resolve(),
        "control temporary checkpoint": control_checkpoint.with_name(
            f".{control_checkpoint.name}.tmp"
        ).resolve(),
        "dropout checkpoint": dropout_checkpoint.resolve(),
        "dropout temporary checkpoint": dropout_checkpoint.with_name(
            f".{dropout_checkpoint.name}.tmp"
        ).resolve(),
    }
    labels = tuple(labeled_paths)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            if labeled_paths[first_label] == labeled_paths[second_label]:
                raise ValueError(
                    f"{first_label} and {second_label} must use different "
                    f"paths: {labeled_paths[first_label]}"
                )


def make_evaluation_callback(
    best: BranchValidationCheckpoint,
    schedule: LearningRateSchedule,
) -> Callable[[EvaluationRecord], None]:
    def record_evaluation(record: EvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        marker = "  * best" if is_best else ""
        losses = record.losses
        gap = losses.val - losses.train
        update_index = max(0, record.step - 1)
        learning_rate = schedule.for_update(update_index)
        print(
            f"  step {record.step:5d} | lr {learning_rate:g} | "
            f"train {losses.train:.4f} | val {losses.val:.4f} | "
            f"gap {gap:.4f} | best {best.best_val_loss:.4f} "
            f"@ {best.best_step}{marker}"
        )

    return record_evaluation


def run_branch(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    initial_state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    spec: BranchSpec,
    *,
    initialization_seed: int,
    training_batch_seed: int,
    training_rng_seed: int,
) -> tuple[BranchReport, int]:
    branch = prepare_branch(
        data,
        config,
        n_head,
        n_layer,
        device,
        schedule,
        initial_state,
        spec,
        training_batch_seed=training_batch_seed,
    )
    loaded_hash = tensor_mapping_sha256(branch.model.state_dict())
    if loaded_hash != initial_state_sha256:
        raise RuntimeError(f"{spec.name} did not load the exact shared initialization")

    parameter_count = sum(
        parameter.numel()
        for parameter in branch.model.parameters()
        if parameter.requires_grad
    )
    best = BranchValidationCheckpoint(
        model=branch.model,
        optimizer=branch.optimizer,
        training_generator=branch.training_generator,
        path=spec.checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        branch_name=spec.name,
        branch_dropout=spec.dropout,
        initial_state_sha256=initial_state_sha256,
        initialization_seed=initialization_seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        target_step=config.max_iters,
    )
    record_evaluation = make_evaluation_callback(best, schedule)

    print(f"\n{spec.name} branch (residual dropout={spec.dropout:g})")
    print(f"  shared initialization: {initial_state_sha256}")
    print(f"  training-batch seed:   {training_batch_seed}")
    print(f"  training RNG seed:     {training_rng_seed}")

    # Model construction consumes the global CPU RNG. Reset only after all
    # construction/loading work, immediately before each branch's updates.
    # The batch generator above is separate, so dropout masks cannot move its
    # training-example stream.
    seed_everything(training_rng_seed)
    result = train_with_schedule(
        branch.model,
        branch.optimizer,
        data,
        config,
        device,
        branch.training_generator,
        schedule,
        on_evaluation=record_evaluation,
    )

    assert best.best_step is not None
    final_batch_generator_sha256 = tensor_sha256(branch.training_generator.get_state())
    sample = generate_from_final_model(
        branch.model,
        data,
        device,
        sample_length=sample_length,
        seed=config.seed + 5,
    )
    report = BranchReport(
        name=spec.name,
        dropout=spec.dropout,
        training=result,
        generalization_gap=result.final.val - result.final.train,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        checkpoint_path=spec.checkpoint_path,
        final_batch_generator_sha256=final_batch_generator_sha256,
        sample=sample,
    )

    del record_evaluation
    del best
    del branch
    clear_accelerator_cache(device)
    return report, parameter_count


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    control_spec: BranchSpec,
    dropout_spec: BranchSpec,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
) -> ExperimentReport:
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if n_head <= 0 or config.n_embd % n_head != 0:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({n_head})"
        )
    if n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {n_layer}")
    schedule.validate_target_step(config.max_iters)
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError(
            "config learning rate must equal the schedule's initial rate: "
            f"{config.learning_rate} != {schedule.initial_learning_rate}"
        )
    if control_spec.dropout != 0.0:
        raise ValueError("Stage 15 control dropout must be exactly 0.0")
    if dropout_spec.dropout <= control_spec.dropout:
        raise ValueError("dropout branch probability must exceed the control")
    if control_spec.name == dropout_spec.name:
        raise ValueError("branch names must be different")
    _validate_non_negative_integer(
        training_batch_seed,
        name="training batch seed",
    )
    _validate_non_negative_integer(
        training_rng_seed,
        name="training RNG seed",
    )
    validate_distinct_output_paths(
        control_spec.checkpoint_path,
        dropout_spec.checkpoint_path,
    )

    initial_state, initial_hash, expected_parameter_count = create_shared_initial_state(
        data, config, n_head, n_layer
    )
    print("\nPaired from-initialization protocol")
    print("  Both branches load one exact initial state.")
    print(
        "  Both branches own generators with the same batch seed; dropout "
        "uses only the separately reset global/device RNG."
    )
    print(
        "  Every measurement uses the same fixed train/validation panels "
        "with dropout disabled."
    )
    print(f"  initial state SHA-256: {initial_hash}")

    control, control_parameter_count = run_branch(
        data,
        config,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        initial_state,
        initial_hash,
        control_spec,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )
    dropout, dropout_parameter_count = run_branch(
        data,
        config,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        initial_state,
        initial_hash,
        dropout_spec,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )

    if control_parameter_count != expected_parameter_count:
        raise RuntimeError("control parameter count changed from initialization")
    if dropout_parameter_count != expected_parameter_count:
        raise RuntimeError("dropout parameter count changed from initialization")
    if control.training.initial != dropout.training.initial:
        raise RuntimeError("branches were not inference-identical at step zero")
    if control.final_batch_generator_sha256 != (dropout.final_batch_generator_sha256):
        raise RuntimeError(
            "branches did not consume identical training-batch RNG streams"
        )
    control_steps = tuple(record.step for record in control.training.history)
    dropout_steps = tuple(record.step for record in dropout.training.history)
    if control_steps != dropout_steps:
        raise RuntimeError("branches were not evaluated at matching steps")

    final_val_delta = dropout.training.final.val - control.training.final.val
    best_val_delta = dropout.best_val_loss - control.best_val_loss

    print("\nStage 15 experimental summary")
    print(
        f"  B / T / C / H / D / FF / L: "
        f"{config.batch_size} / {config.block_size} / {config.n_embd} / "
        f"{n_head} / {config.n_embd // n_head} / "
        f"{4 * config.n_embd} / {n_layer}"
    )
    print(f"  parameter count:             {expected_parameter_count:,}")
    print(f"  shared initialization:       {initial_hash}")
    print(f"  shared training-batch seed:  {training_batch_seed}")
    print(
        f"  control final train / val:   "
        f"{control.training.final.train:.4f} / "
        f"{control.training.final.val:.4f} (p={control.dropout:g})"
    )
    print(
        f"  dropout final train / val:   "
        f"{dropout.training.final.train:.4f} / "
        f"{dropout.training.final.val:.4f} (p={dropout.dropout:g})"
    )
    print(
        f"  final val delta (p-control): {final_val_delta:+.4f} "
        "(negative favors dropout)"
    )
    print(
        f"  control best val:            {control.best_val_loss:.4f} "
        f"@ {control.best_step}"
    )
    print(
        f"  dropout best val:            {dropout.best_val_loss:.4f} "
        f"@ {dropout.best_step}"
    )
    print(
        f"  best val delta (p-control):  {best_val_delta:+.4f} "
        "(negative favors dropout)"
    )
    print(f"  control checkpoint:          {control.checkpoint_path}")
    print(f"  dropout checkpoint:          {dropout.checkpoint_path}")
    print("\nControl final-step generated text")
    print(control.sample)
    print("\nDropout final-step generated text")
    print(dropout.sample)

    return ExperimentReport(
        parameter_count=expected_parameter_count,
        initial_state_sha256=initial_hash,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        control=control,
        dropout=dropout,
        final_val_delta=final_val_delta,
        best_val_delta=best_val_delta,
    )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def validate_stage_15_checkpoint(
    path: Path,
    spec: BranchSpec,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    *,
    initial_state_sha256: str,
    training_batch_seed: int,
    training_rng_seed: int,
) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 15 checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = _require_mapping(checkpoint, name=str(path))

    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError(f"{spec.name} must be a best checkpoint")
    step = checkpoint.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= config.max_iters
    ):
        raise ValueError(
            f"checkpoint step must be in [0, {config.max_iters}], got {step!r}"
        )
    if checkpoint.get("best_step") != step:
        raise ValueError("checkpoint best_step must equal its saved step")
    best_val_loss = checkpoint.get("best_val_loss")
    if (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise ValueError("checkpoint best_val_loss must be finite")

    architecture = _require_mapping(
        checkpoint.get("architecture"),
        name="architecture",
    )
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "residual_dropout": spec.dropout,
        "dropout_placement": DROPOUT_PLACEMENT,
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"checkpoint architecture.{key} must be {expected!r}, "
                f"got {architecture.get(key)!r}"
            )

    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("checkpoint data fingerprint does not match")

    expected_schedule = schedule.as_metadata(max_iters=config.max_iters)
    initialization = _require_mapping(
        checkpoint.get("initialization"),
        name="initialization",
    )
    expected_initialization = {
        "kind": "shared_random_initialization",
        "seed": config.seed,
        "state_sha256": initial_state_sha256,
    }
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
        "stage": 15,
        "branch": spec.name,
        "from_scratch": True,
        "comparison_variable": "residual_dropout",
        "branch_residual_dropout": spec.dropout,
        "dropout_placement": DROPOUT_PLACEMENT,
        "identical_initialization": True,
        "initial_state_sha256": initial_state_sha256,
        "initialization_seed": config.seed,
        "identical_training_batches": True,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
        "learning_rate_schedule": expected_schedule,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"checkpoint experiment.{key} must be {expected!r}, "
                f"got {experiment.get(key)!r}"
            )

    training_config = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    expected_training_metadata = {
        "batch_size": config.batch_size,
        "eval_interval": config.eval_interval,
        "eval_iters": config.eval_iters,
        "seed": config.seed,
        "max_iters": config.max_iters,
        "learning_rate_schedule": expected_schedule,
        "residual_dropout": spec.dropout,
        "training_batch_seed": training_batch_seed,
        "training_rng_seed": training_rng_seed,
    }
    for key, expected in expected_training_metadata.items():
        if training_config.get(key) != expected:
            raise ValueError(
                f"checkpoint training_config.{key} must be {expected!r}, "
                f"got {training_config.get(key)!r}"
            )

    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"),
        name="optimizer_state_dict",
    )
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("checkpoint optimizer must contain parameter groups")
    if not all(isinstance(group, Mapping) and "lr" in group for group in param_groups):
        raise ValueError(
            "every checkpoint optimizer parameter group must contain an LR"
        )
    optimizer_rates = {float(group["lr"]) for group in param_groups}
    saved_learning_rate = training_config.get("learning_rate")
    if (
        isinstance(saved_learning_rate, bool)
        or not isinstance(saved_learning_rate, (int, float))
        or not math.isfinite(float(saved_learning_rate))
        or float(saved_learning_rate) <= 0
    ):
        raise ValueError(
            "checkpoint training_config.learning_rate must be finite and positive"
        )
    expected_learning_rate = schedule.for_update(max(0, step - 1))
    if float(saved_learning_rate) != expected_learning_rate:
        raise ValueError(
            f"checkpoint learning rate at completed step {step} must be "
            f"{expected_learning_rate:g}, got {saved_learning_rate!r}"
        )
    if optimizer_rates != {float(saved_learning_rate)}:
        raise ValueError("checkpoint optimizer LR does not match training metadata")

    _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    return checkpoint


def run_precise_validation(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    schedule: LearningRateSchedule,
    checkpoint_specs: Sequence[BranchSpec],
    *,
    initial_state_sha256: str,
    training_batch_seed: int,
    training_rng_seed: int,
    eval_iters: int = DEFAULT_PRECISE_EVAL_ITERS,
    seed: int,
) -> PreciseValidationReport:
    if len(checkpoint_specs) < 2:
        raise ValueError("precise comparison requires at least two checkpoints")
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    _validate_non_negative_integer(seed, name="precise evaluation seed")

    results: list[PreciseValidationResult] = []
    for spec in checkpoint_specs:
        checkpoint_hash = checkpoint_sha256(spec.checkpoint_path)
        checkpoint = validate_stage_15_checkpoint(
            spec.checkpoint_path,
            spec,
            data,
            config,
            n_head,
            n_layer,
            schedule,
            initial_state_sha256=initial_state_sha256,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
        )
        model = GPTLanguageModel(
            vocab_size=data.vocabulary.size,
            block_size=config.block_size,
            n_embd=config.n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=spec.dropout,
        ).to(device)
        model_state = _require_mapping(
            checkpoint.get("model_state_dict"),
            name="model_state_dict",
        )
        model.load_state_dict(model_state, strict=True)
        batch_losses = estimate_validation_precise(
            model,
            data,
            config,
            device,
            eval_iters=eval_iters,
            seed=seed,
        )
        if checkpoint_sha256(spec.checkpoint_path) != checkpoint_hash:
            raise RuntimeError(
                f"checkpoint changed during precise evaluation: {spec.checkpoint_path}"
            )
        mean, standard_error = _STAGE_14._mean_and_standard_error(batch_losses)
        results.append(
            PreciseValidationResult(
                name=spec.name,
                checkpoint_path=spec.checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                checkpoint_step=int(checkpoint["step"]),
                fixed_panel_loss=float(checkpoint["best_val_loss"]),
                mean_loss=mean,
                standard_error=standard_error,
                batch_losses=batch_losses,
            )
        )
        del model
        clear_accelerator_cache(device)

    adjacent_deltas = tuple(
        _STAGE_14._paired_delta(candidate, baseline)
        for baseline, candidate in zip(results, results[1:], strict=False)
    )
    return PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        results=tuple(results),
        adjacent_deltas=adjacent_deltas,
    )


def print_precise_validation(report: PreciseValidationReport) -> None:
    print("\nFresh paired validation of Stage 15 best checkpoints")
    print(
        f"  {report.eval_iters} validation batches per checkpoint, "
        f"shared panel seed={report.seed}"
    )
    print("  Dropout is disabled for every measurement.")
    for result in report.results:
        print(
            f"  {result.name:18s} | step {result.checkpoint_step:5d} | "
            f"fixed {result.fixed_panel_loss:.4f} | fresh "
            f"{result.mean_loss:.6f} +/- {result.standard_error:.6f} SE"
        )
    for delta in report.adjacent_deltas:
        print(
            f"  delta {delta.candidate} - {delta.baseline}: "
            f"{delta.mean_delta:+.6f} +/- {delta.standard_error:.6f} SE "
            f"(95% CI [{delta.confidence_low:+.6f}, "
            f"{delta.confidence_high:+.6f}]); negative favors dropout"
        )


def main() -> None:
    args = parse_args()
    if args.sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if args.precise_eval_iters < 0:
        raise ValueError(
            f"precise-eval-iters must be non-negative, got {args.precise_eval_iters}"
        )

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
        learning_rate=schedule.initial_learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    schedule.validate_target_step(config.max_iters)
    training_batch_seed = (
        args.training_batch_seed
        if args.training_batch_seed is not None
        else config.seed + 1
    )
    training_rng_seed = (
        args.training_rng_seed
        if args.training_rng_seed is not None
        else config.seed + 3
    )
    precise_eval_seed = (
        args.precise_eval_seed
        if args.precise_eval_seed is not None
        else config.seed + 4
    )
    for name, value in (
        ("training batch seed", training_batch_seed),
        ("training RNG seed", training_rng_seed),
        ("precise evaluation seed", precise_eval_seed),
    ):
        _validate_non_negative_integer(value, name=name)

    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)
    control_spec = BranchSpec(
        name="control",
        dropout=args.control_dropout,
        checkpoint_path=args.control_checkpoint_path,
    )
    dropout_spec = BranchSpec(
        name="dropout",
        dropout=args.dropout,
        checkpoint_path=args.dropout_checkpoint_path,
    )

    print("Stage 15: residual dropout from initialization")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Architecture: B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={args.n_head}, "
        f"D={config.n_embd // args.n_head}, FF={4 * config.n_embd}, "
        f"L={args.n_layer}"
    )
    print(
        "Schedule by zero-based update: "
        f"[0, {schedule.first_decay_step:,}) "
        f"lr={schedule.initial_learning_rate:g}; "
        f"[{schedule.first_decay_step:,}, "
        f"{schedule.second_decay_step:,}) "
        f"lr={schedule.middle_learning_rate:g}; "
        f"[{schedule.second_decay_step:,}, {config.max_iters:,}) "
        f"lr={schedule.final_learning_rate:g}"
    )
    print(
        f"Treatment: control p={control_spec.dropout:g}, "
        f"dropout p={dropout_spec.dropout:g}"
    )

    report = run_experiment(
        data,
        config,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        schedule,
        control_spec,
        dropout_spec,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )

    if args.precise_eval_iters > 0:
        precise_report = run_precise_validation(
            data,
            config,
            args.n_head,
            args.n_layer,
            device,
            schedule,
            (control_spec, dropout_spec),
            initial_state_sha256=report.initial_state_sha256,
            training_batch_seed=training_batch_seed,
            training_rng_seed=training_rng_seed,
            eval_iters=args.precise_eval_iters,
            seed=precise_eval_seed,
        )
        print_precise_validation(precise_report)


if __name__ == "__main__":
    main()
