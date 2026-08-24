from __future__ import annotations

import argparse
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
from training import EvaluationRecord, seed_everything


# Stage 16 changes only the experiment driver. Reuse the exact Stage 15 model,
# schedule, fixed-panel evaluator, and training loop so p is the sole treatment
# variable. Numeric stage filenames are not regular Python module names, hence
# the explicit import.
_STAGE_15_PATH = Path(__file__).with_name("15_dropout_from_initialization.py")
_STAGE_15_MODULE_NAME = "stage_15_dropout_from_initialization_for_stage_16"
_STAGE_15_SPEC = importlib.util.spec_from_file_location(
    _STAGE_15_MODULE_NAME,
    _STAGE_15_PATH,
)
assert _STAGE_15_SPEC is not None and _STAGE_15_SPEC.loader is not None
_STAGE_15 = importlib.util.module_from_spec(_STAGE_15_SPEC)
sys.modules[_STAGE_15_MODULE_NAME] = _STAGE_15
_STAGE_15_SPEC.loader.exec_module(_STAGE_15)

GPTLanguageModel = _STAGE_15.GPTLanguageModel
BranchSpec = _STAGE_15.BranchSpec
BranchReport = _STAGE_15.BranchReport
LearningRateSchedule = _STAGE_15.LearningRateSchedule
PreciseDelta = _STAGE_15.PreciseDelta
PreciseValidationResult = _STAGE_15.PreciseValidationResult

checkpoint_sha256 = _STAGE_15.checkpoint_sha256
clear_accelerator_cache = _STAGE_15.clear_accelerator_cache
create_shared_initial_state = _STAGE_15.create_shared_initial_state
estimate_validation_precise = _STAGE_15.estimate_validation_precise
fingerprint_data = _STAGE_15.fingerprint_data
generate_from_final_model = _STAGE_15.generate_from_final_model
prepare_branch = _STAGE_15.prepare_branch
tensor_mapping_sha256 = _STAGE_15.tensor_mapping_sha256
tensor_sha256 = _STAGE_15.tensor_sha256
train_with_schedule = _STAGE_15.train_with_schedule

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_15_control_best_checkpoint.pt"
)
DEFAULT_HIGH_DROPOUT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_15_dropout_best_checkpoint.pt"
)
DEFAULT_TREATMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_16_dropout_best_checkpoint.pt"
)

# Pin the default inputs so the normal Stage 16 command cannot silently compare
# against a different, merely metadata-compatible Stage 15 run. Custom paths
# are supported for tests and deliberately constructed replications.
DEFAULT_CONTROL_CHECKPOINT_SHA256 = (
    "4d05d242344f6020a56d93c21507e8b0bc7a73a23d51d65e673982192b180168"
)
DEFAULT_HIGH_DROPOUT_CHECKPOINT_SHA256 = (
    "8b3d781476cfbdbe304a54ff6883264697f16d2291ce724997ead29cc28ecfd6"
)
DEFAULT_STAGE_15_BEST_STEP = 17_000

DEFAULT_MAX_ITERS = _STAGE_15.DEFAULT_MAX_ITERS
DEFAULT_INITIAL_LEARNING_RATE = _STAGE_15.DEFAULT_INITIAL_LEARNING_RATE
DEFAULT_FIRST_DECAY_STEP = _STAGE_15.DEFAULT_FIRST_DECAY_STEP
DEFAULT_MIDDLE_LEARNING_RATE = _STAGE_15.DEFAULT_MIDDLE_LEARNING_RATE
DEFAULT_SECOND_DECAY_STEP = _STAGE_15.DEFAULT_SECOND_DECAY_STEP
DEFAULT_FINAL_LEARNING_RATE = _STAGE_15.DEFAULT_FINAL_LEARNING_RATE
DEFAULT_CONTROL_DROPOUT = _STAGE_15.DEFAULT_CONTROL_DROPOUT
DEFAULT_RESIDUAL_DROPOUT = 0.05
STAGE_15_HIGH_DROPOUT = _STAGE_15.DEFAULT_RESIDUAL_DROPOUT
DEFAULT_PRECISE_EVAL_ITERS = _STAGE_15.DEFAULT_PRECISE_EVAL_ITERS
DROPOUT_PLACEMENT = _STAGE_15.DROPOUT_PLACEMENT


@dataclass(frozen=True, slots=True)
class ReferenceCheckpoint:
    """A validated, read-only Stage 15 comparison checkpoint."""

    branch_name: str
    display_name: str
    dropout: float
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_step: int
    fixed_panel_loss: float
    training_generator_state: torch.Tensor


@dataclass(slots=True)
class TreatmentValidationCheckpoint(_STAGE_15.BranchValidationCheckpoint):
    """Persist the Stage 16 treatment with its reused-control provenance."""

    control_checkpoint_sha256: str = ""
    high_dropout_checkpoint_sha256: str = ""

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(TreatmentValidationCheckpoint, self)._payload(step)
        if self.schedule is None:
            raise RuntimeError("Stage 16 checkpoint requires an LR schedule")

        schedule_metadata = self.schedule.as_metadata(max_iters=self.target_step)
        payload["experiment"] = {
            "stage": 16,
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
            "reused_control": True,
            "control_stage": 15,
            "control_branch": "control",
            "control_residual_dropout": DEFAULT_CONTROL_DROPOUT,
            "control_checkpoint_sha256": self.control_checkpoint_sha256,
            "high_dropout_reference_stage": 15,
            "high_dropout_reference_branch": "dropout",
            "high_dropout_reference_probability": STAGE_15_HIGH_DROPOUT,
            "high_dropout_reference_checkpoint_sha256": (
                self.high_dropout_checkpoint_sha256
            ),
        }
        return payload


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    parameter_count: int
    initial_state_sha256: str
    initialization_seed: int
    training_batch_seed: int
    training_rng_seed: int
    schedule: LearningRateSchedule
    control: ReferenceCheckpoint
    treatment: BranchReport
    high_dropout: ReferenceCheckpoint
    treatment_best_delta: float
    high_dropout_best_delta: float

    @property
    def best_winner(self) -> str:
        losses = {
            self.control.display_name: self.control.fixed_panel_loss,
            self.treatment.name: self.treatment.best_val_loss,
            self.high_dropout.display_name: self.high_dropout.fixed_panel_loss,
        }
        return min(losses, key=losses.__getitem__)


@dataclass(frozen=True, slots=True)
class DoseResponseValidationReport:
    eval_iters: int
    seed: int
    results: tuple[PreciseValidationResult, ...]
    deltas: tuple[PreciseDelta, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only residual dropout p=0.05 from the exact Stage 15 "
            "initialization and batch stream, reusing the saved p=0 control."
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
        "--dropout",
        type=float,
        default=DEFAULT_RESIDUAL_DROPOUT,
        help="Stage 16 treatment probability (default: 0.05).",
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
        help="Defaults to seed + 1; must match the Stage 15 references.",
    )
    parser.add_argument(
        "--training-rng-seed",
        type=int,
        default=None,
        help="Defaults to seed + 3; reset immediately before updates.",
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
        help=("Defaults to seed + 6, a new panel after Stage 15's seed + 4 panel."),
    )
    parser.add_argument(
        "--control-checkpoint-path",
        type=Path,
        default=DEFAULT_CONTROL_CHECKPOINT_PATH,
        help="Read-only Stage 15 p=0 best checkpoint.",
    )
    parser.add_argument(
        "--high-dropout-checkpoint-path",
        type=Path,
        default=DEFAULT_HIGH_DROPOUT_CHECKPOINT_PATH,
        help="Read-only Stage 15 p=0.1 best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_TREATMENT_CHECKPOINT_PATH,
        help="Output path for the Stage 16 p=0.05 best checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _validate_non_negative_integer(value: int, *, name: str) -> int:
    return _STAGE_15._validate_non_negative_integer(value, name=name)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    return _STAGE_15._require_mapping(value, name=name)


def _pin_for_default_path(
    path: Path,
    default_path: Path,
    expected_sha256: str,
) -> tuple[str | None, int | None]:
    if path.resolve() == default_path.resolve():
        return expected_sha256, DEFAULT_STAGE_15_BEST_STEP
    return None, None


def validate_distinct_paths(
    treatment_checkpoint: Path,
    control_checkpoint: Path,
    high_dropout_checkpoint: Path,
) -> None:
    paths = {
        "treatment checkpoint": treatment_checkpoint.resolve(),
        "treatment temporary checkpoint": treatment_checkpoint.with_name(
            f".{treatment_checkpoint.name}.tmp"
        ).resolve(),
        "control checkpoint": control_checkpoint.resolve(),
        "control temporary checkpoint": control_checkpoint.with_name(
            f".{control_checkpoint.name}.tmp"
        ).resolve(),
        "high-dropout checkpoint": high_dropout_checkpoint.resolve(),
        "high-dropout temporary checkpoint": high_dropout_checkpoint.with_name(
            f".{high_dropout_checkpoint.name}.tmp"
        ).resolve(),
    }
    labels = tuple(paths)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            if paths[first_label] == paths[second_label]:
                raise ValueError(
                    f"{first_label} and {second_label} must use different "
                    f"paths: {paths[first_label]}"
                )
            first_path = paths[first_label]
            second_path = paths[second_label]
            if (
                first_path.exists()
                and second_path.exists()
                and first_path.samefile(second_path)
            ):
                raise ValueError(
                    f"{first_label} and {second_label} must not refer to the "
                    f"same file: {first_path}"
                )


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


def validate_stage_15_reference(
    path: Path,
    *,
    branch_name: str,
    display_name: str,
    dropout: float,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    initial_state_sha256: str,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_checkpoint_sha256: str | None = None,
    expected_step: int | None = None,
) -> ReferenceCheckpoint:
    before_hash = checkpoint_sha256(path)
    if expected_checkpoint_sha256 is not None and before_hash != (
        expected_checkpoint_sha256
    ):
        raise ValueError(
            f"Stage 15 {branch_name} checkpoint SHA-256 must be "
            f"{expected_checkpoint_sha256}, got {before_hash}"
        )

    spec = BranchSpec(
        name=branch_name,
        dropout=dropout,
        checkpoint_path=path,
    )
    checkpoint = _STAGE_15.validate_stage_15_checkpoint(
        path,
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
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("Stage 15 reference checkpoint_version must be 1")
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 15 reference must have uninterrupted AdamW state")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 15 reference optimizer provenance must be known")
    step = int(checkpoint["step"])
    if expected_step is not None and step != expected_step:
        raise ValueError(
            f"Stage 15 {branch_name} checkpoint step must be "
            f"{expected_step}, got {step}"
        )
    if not _is_evaluation_step(step, config, schedule):
        raise ValueError(
            f"Stage 15 {branch_name} checkpoint step {step} is not a "
            "scheduled fixed-panel evaluation step"
        )

    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    compatibility_model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )
    compatibility_model.load_state_dict(model_state, strict=True)
    del compatibility_model

    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError(
            f"Stage 15 {branch_name} checkpoint must contain a tensor "
            "training_generator_state"
        )
    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError(f"checkpoint changed during validation: {path}")

    return ReferenceCheckpoint(
        branch_name=branch_name,
        display_name=display_name,
        dropout=dropout,
        checkpoint_path=path,
        checkpoint_sha256=before_hash,
        checkpoint_step=step,
        fixed_panel_loss=float(checkpoint["best_val_loss"]),
        training_generator_state=generator_state.detach().cpu().clone(),
    )


def expected_training_generator_sha256(
    data: CharacterData,
    config: TrainingConfig,
    *,
    training_batch_seed: int,
) -> str:
    generator = torch.Generator().manual_seed(training_batch_seed)
    _STAGE_15._STAGE_14._STAGE_11.advance_training_generator(
        generator,
        num_steps=config.max_iters,
        batch_size=config.batch_size,
        num_train_tokens=len(data.train_data),
        block_size=config.block_size,
    )
    return tensor_sha256(generator.get_state())


def make_evaluation_callback(
    best: TreatmentValidationCheckpoint,
    schedule: LearningRateSchedule,
    reference_checkpoints: Sequence[ReferenceCheckpoint],
) -> tuple[Callable[[EvaluationRecord], None], set[str]]:
    base_callback = _STAGE_15.make_evaluation_callback(best, schedule)
    verified_references: set[str] = set()

    def record_evaluation(record: EvaluationRecord) -> None:
        for reference in reference_checkpoints:
            if record.step != reference.checkpoint_step:
                continue
            actual_state = best.training_generator.get_state()
            if not torch.equal(
                actual_state.cpu(),
                reference.training_generator_state,
            ):
                raise RuntimeError(
                    "treatment batch-generator state does not match Stage 15 "
                    f"{reference.branch_name} at step {record.step}"
                )
            verified_references.add(reference.branch_name)
        base_callback(record)

    return record_evaluation, verified_references


def run_treatment(
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
    control: ReferenceCheckpoint,
    high_dropout: ReferenceCheckpoint,
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
        raise RuntimeError("Stage 16 did not load the exact Stage 15 initialization")

    parameter_count = sum(
        parameter.numel()
        for parameter in branch.model.parameters()
        if parameter.requires_grad
    )
    best = TreatmentValidationCheckpoint(
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
        control_checkpoint_sha256=control.checkpoint_sha256,
        high_dropout_checkpoint_sha256=high_dropout.checkpoint_sha256,
    )
    record_evaluation, verified_references = make_evaluation_callback(
        best,
        schedule,
        (control, high_dropout),
    )

    print(f"\n{spec.name} treatment (residual dropout={spec.dropout:g})")
    print(f"  shared initialization: {initial_state_sha256}")
    print(f"  training-batch seed:   {training_batch_seed}")
    print(f"  training RNG seed:     {training_rng_seed}")

    # Reference validation and model construction consume the global CPU RNG.
    # Reset immediately before optimizer updates, exactly as Stage 15 did.
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

    expected_verified = {control.branch_name, high_dropout.branch_name}
    if verified_references != expected_verified:
        missing = sorted(expected_verified - verified_references)
        raise RuntimeError(
            "did not reach the Stage 15 reference checkpoint step(s): "
            + ", ".join(missing)
        )

    final_generator_hash = tensor_sha256(branch.training_generator.get_state())
    expected_generator_hash = expected_training_generator_sha256(
        data,
        config,
        training_batch_seed=training_batch_seed,
    )
    if final_generator_hash != expected_generator_hash:
        raise RuntimeError(
            "Stage 16 did not consume the expected deterministic training batch stream"
        )

    assert best.best_step is not None
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
        final_batch_generator_sha256=final_generator_hash,
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
    treatment_spec: BranchSpec,
    control_checkpoint_path: Path,
    high_dropout_checkpoint_path: Path,
    *,
    training_batch_seed: int,
    training_rng_seed: int,
    expected_control_sha256: str | None = None,
    expected_control_step: int | None = None,
    expected_high_dropout_sha256: str | None = None,
    expected_high_dropout_step: int | None = None,
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
    if treatment_spec.dropout != DEFAULT_RESIDUAL_DROPOUT:
        raise ValueError(
            "Stage 16 treatment dropout must be exactly "
            f"{DEFAULT_RESIDUAL_DROPOUT:g}, got {treatment_spec.dropout:g}"
        )
    _validate_non_negative_integer(
        training_batch_seed,
        name="training batch seed",
    )
    _validate_non_negative_integer(
        training_rng_seed,
        name="training RNG seed",
    )
    validate_distinct_paths(
        treatment_spec.checkpoint_path,
        control_checkpoint_path,
        high_dropout_checkpoint_path,
    )

    initial_state, initial_hash, expected_parameter_count = create_shared_initial_state(
        data, config, n_head, n_layer
    )
    control = validate_stage_15_reference(
        control_checkpoint_path,
        branch_name="control",
        display_name="p=0",
        dropout=DEFAULT_CONTROL_DROPOUT,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        schedule=schedule,
        initial_state_sha256=initial_hash,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_checkpoint_sha256=expected_control_sha256,
        expected_step=expected_control_step,
    )
    high_dropout = validate_stage_15_reference(
        high_dropout_checkpoint_path,
        branch_name="dropout",
        display_name="p=0.10",
        dropout=STAGE_15_HIGH_DROPOUT,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        schedule=schedule,
        initial_state_sha256=initial_hash,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_checkpoint_sha256=expected_high_dropout_sha256,
        expected_step=expected_high_dropout_step,
    )

    print("\nStage 16 treatment-only protocol")
    print("  The Stage 15 p=0 control is validated and never trained or written.")
    print("  The p=0.05 treatment regenerates the exact Stage 15 initial state.")
    print("  Its independent batch generator uses the Stage 15 batch seed.")
    print("  Every curve measurement reuses the Stage 15 fixed panels.")
    print(f"  initial state SHA-256: {initial_hash}")
    print(f"  control checkpoint SHA-256: {control.checkpoint_sha256}")
    print(f"  p=0.10 checkpoint SHA-256: {high_dropout.checkpoint_sha256}")

    treatment, treatment_parameter_count = run_treatment(
        data,
        config,
        n_head,
        n_layer,
        device,
        sample_length,
        schedule,
        initial_state,
        initial_hash,
        treatment_spec,
        control,
        high_dropout,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
    )
    if treatment_parameter_count != expected_parameter_count:
        raise RuntimeError("Stage 16 parameter count changed from initialization")
    if checkpoint_sha256(control.checkpoint_path) != control.checkpoint_sha256:
        raise RuntimeError("Stage 15 control checkpoint changed during Stage 16")
    if checkpoint_sha256(high_dropout.checkpoint_path) != (
        high_dropout.checkpoint_sha256
    ):
        raise RuntimeError("Stage 15 p=0.10 checkpoint changed during Stage 16")

    treatment_best_delta = treatment.best_val_loss - control.fixed_panel_loss
    high_dropout_best_delta = high_dropout.fixed_panel_loss - control.fixed_panel_loss
    print("\nStage 16 experimental summary")
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
        f"  treatment final train / val: "
        f"{treatment.training.final.train:.4f} / "
        f"{treatment.training.final.val:.4f} "
        f"(p={treatment.dropout:g})"
    )
    print(
        f"  p=0 best fixed val:          {control.fixed_panel_loss:.4f} "
        f"@ {control.checkpoint_step}"
    )
    print(
        f"  p=0.05 best fixed val:       {treatment.best_val_loss:.4f} "
        f"@ {treatment.best_step}"
    )
    print(
        f"  p=0.10 best fixed val:       {high_dropout.fixed_panel_loss:.4f} "
        f"@ {high_dropout.checkpoint_step}"
    )
    print(
        f"  best delta (p=.05 - p=0):    {treatment_best_delta:+.4f} "
        "(negative favors p=.05)"
    )
    print(f"  Stage 15 delta (p=.10-p=0):  {high_dropout_best_delta:+.4f}")
    print(f"  reused control checkpoint:   {control.checkpoint_path}")
    print(f"  treatment checkpoint:        {treatment.checkpoint_path}")
    print("\nTreatment final-step generated text")
    print(treatment.sample)

    return ExperimentReport(
        parameter_count=expected_parameter_count,
        initial_state_sha256=initial_hash,
        initialization_seed=config.seed,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        schedule=schedule,
        control=control,
        treatment=treatment,
        high_dropout=high_dropout,
        treatment_best_delta=treatment_best_delta,
        high_dropout_best_delta=high_dropout_best_delta,
    )


def validate_stage_16_checkpoint(
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
    control_checkpoint_sha256: str,
    high_dropout_checkpoint_sha256: str,
) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 16 checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = _require_mapping(checkpoint, name=str(path))

    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("Stage 16 checkpoint_version must be 1")
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("Stage 16 treatment must be a best checkpoint")
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError("Stage 16 checkpoint must have uninterrupted AdamW state")
    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError("Stage 16 optimizer provenance must be known")
    step = checkpoint.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= config.max_iters
    ):
        raise ValueError(
            f"checkpoint step must be in [0, {config.max_iters}], got {step!r}"
        )
    if not _is_evaluation_step(step, config, schedule):
        raise ValueError(
            f"Stage 16 checkpoint step {step} is not a scheduled evaluation step"
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
        "stage": 16,
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
        "reused_control": True,
        "control_stage": 15,
        "control_branch": "control",
        "control_residual_dropout": DEFAULT_CONTROL_DROPOUT,
        "control_checkpoint_sha256": control_checkpoint_sha256,
        "high_dropout_reference_stage": 15,
        "high_dropout_reference_branch": "dropout",
        "high_dropout_reference_probability": STAGE_15_HIGH_DROPOUT,
        "high_dropout_reference_checkpoint_sha256": (high_dropout_checkpoint_sha256),
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
    expected_training = {
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
    for key, expected in expected_training.items():
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
    optimizer_rates = {float(group["lr"]) for group in param_groups}
    if optimizer_rates != {float(saved_learning_rate)}:
        raise ValueError("checkpoint optimizer LR does not match training metadata")

    _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )
    if not isinstance(checkpoint.get("training_generator_state"), torch.Tensor):
        raise ValueError(
            "Stage 16 checkpoint must contain a tensor training_generator_state"
        )
    return checkpoint


def _measure_checkpoint(
    *,
    name: str,
    dropout: float,
    checkpoint_path: Path,
    checkpoint: Mapping[str, object],
    expected_checkpoint_sha256: str,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    eval_iters: int,
    seed: int,
) -> PreciseValidationResult:
    if checkpoint_sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise RuntimeError(f"checkpoint changed before evaluation: {checkpoint_path}")
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
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
    if checkpoint_sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise RuntimeError(f"checkpoint changed during evaluation: {checkpoint_path}")
    mean, standard_error = _STAGE_15._STAGE_14._mean_and_standard_error(batch_losses)
    result = PreciseValidationResult(
        name=name,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_step=int(checkpoint["step"]),
        fixed_panel_loss=float(checkpoint["best_val_loss"]),
        mean_loss=mean,
        standard_error=standard_error,
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
    treatment_spec: BranchSpec,
    experiment: ExperimentReport,
    *,
    eval_iters: int = DEFAULT_PRECISE_EVAL_ITERS,
    seed: int,
) -> DoseResponseValidationReport:
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    _validate_non_negative_integer(seed, name="precise evaluation seed")

    control = validate_stage_15_reference(
        experiment.control.checkpoint_path,
        branch_name="control",
        display_name="p=0",
        dropout=DEFAULT_CONTROL_DROPOUT,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        schedule=schedule,
        initial_state_sha256=experiment.initial_state_sha256,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        expected_checkpoint_sha256=experiment.control.checkpoint_sha256,
        expected_step=experiment.control.checkpoint_step,
    )
    high_dropout = validate_stage_15_reference(
        experiment.high_dropout.checkpoint_path,
        branch_name="dropout",
        display_name="p=0.10",
        dropout=STAGE_15_HIGH_DROPOUT,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        schedule=schedule,
        initial_state_sha256=experiment.initial_state_sha256,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        expected_checkpoint_sha256=experiment.high_dropout.checkpoint_sha256,
        expected_step=experiment.high_dropout.checkpoint_step,
    )
    treatment_hash = checkpoint_sha256(treatment_spec.checkpoint_path)
    treatment_checkpoint = validate_stage_16_checkpoint(
        treatment_spec.checkpoint_path,
        treatment_spec,
        data,
        config,
        n_head,
        n_layer,
        schedule,
        initial_state_sha256=experiment.initial_state_sha256,
        training_batch_seed=experiment.training_batch_seed,
        training_rng_seed=experiment.training_rng_seed,
        control_checkpoint_sha256=control.checkpoint_sha256,
        high_dropout_checkpoint_sha256=high_dropout.checkpoint_sha256,
    )
    if checkpoint_sha256(treatment_spec.checkpoint_path) != treatment_hash:
        raise RuntimeError("Stage 16 checkpoint changed during validation")

    control_checkpoint = torch.load(
        control.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    control_checkpoint = _require_mapping(
        control_checkpoint,
        name=str(control.checkpoint_path),
    )
    high_checkpoint = torch.load(
        high_dropout.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    high_checkpoint = _require_mapping(
        high_checkpoint,
        name=str(high_dropout.checkpoint_path),
    )

    control_result = _measure_checkpoint(
        name="p=0",
        dropout=DEFAULT_CONTROL_DROPOUT,
        checkpoint_path=control.checkpoint_path,
        checkpoint=control_checkpoint,
        expected_checkpoint_sha256=control.checkpoint_sha256,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        eval_iters=eval_iters,
        seed=seed,
    )
    treatment_result = _measure_checkpoint(
        name="p=0.05",
        dropout=treatment_spec.dropout,
        checkpoint_path=treatment_spec.checkpoint_path,
        checkpoint=treatment_checkpoint,
        expected_checkpoint_sha256=treatment_hash,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        eval_iters=eval_iters,
        seed=seed,
    )
    high_result = _measure_checkpoint(
        name="p=0.10",
        dropout=STAGE_15_HIGH_DROPOUT,
        checkpoint_path=high_dropout.checkpoint_path,
        checkpoint=high_checkpoint,
        expected_checkpoint_sha256=high_dropout.checkpoint_sha256,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        device=device,
        eval_iters=eval_iters,
        seed=seed,
    )
    paired_delta = _STAGE_15._STAGE_14._paired_delta
    deltas = (
        paired_delta(treatment_result, control_result),
        paired_delta(high_result, control_result),
        paired_delta(high_result, treatment_result),
    )
    return DoseResponseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        results=(control_result, treatment_result, high_result),
        deltas=deltas,
    )


def print_precise_validation(report: DoseResponseValidationReport) -> None:
    print("\nFresh paired validation of the Stage 15/16 dose response")
    print(
        f"  {report.eval_iters} validation batches per checkpoint, "
        f"shared new-panel seed={report.seed}"
    )
    print("  Dropout is disabled for every measurement.")
    for result in report.results:
        print(
            f"  {result.name:8s} | step {result.checkpoint_step:5d} | "
            f"fixed {result.fixed_panel_loss:.4f} | fresh "
            f"{result.mean_loss:.6f} +/- {result.standard_error:.6f} SE"
        )
    for delta in report.deltas:
        print(
            f"  delta {delta.candidate} - {delta.baseline}: "
            f"{delta.mean_delta:+.6f} +/- {delta.standard_error:.6f} SE "
            f"(95% CI [{delta.confidence_low:+.6f}, "
            f"{delta.confidence_high:+.6f}]); negative favors "
            f"{delta.candidate}"
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
        else config.seed + 6
    )
    for name, value in (
        ("training batch seed", training_batch_seed),
        ("training RNG seed", training_rng_seed),
        ("precise evaluation seed", precise_eval_seed),
    ):
        _validate_non_negative_integer(value, name=name)

    control_pin, control_step_pin = _pin_for_default_path(
        args.control_checkpoint_path,
        DEFAULT_CONTROL_CHECKPOINT_PATH,
        DEFAULT_CONTROL_CHECKPOINT_SHA256,
    )
    high_pin, high_step_pin = _pin_for_default_path(
        args.high_dropout_checkpoint_path,
        DEFAULT_HIGH_DROPOUT_CHECKPOINT_PATH,
        DEFAULT_HIGH_DROPOUT_CHECKPOINT_SHA256,
    )
    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)
    treatment_spec = BranchSpec(
        name="dropout_p005",
        dropout=args.dropout,
        checkpoint_path=args.checkpoint_path,
    )

    print("Stage 16: residual-dropout strength p=0.05")
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
        f"Dose response: reused p=0, new p={treatment_spec.dropout:g}, "
        f"reused p={STAGE_15_HIGH_DROPOUT:g}"
    )

    report = run_experiment(
        data,
        config,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        schedule,
        treatment_spec,
        args.control_checkpoint_path,
        args.high_dropout_checkpoint_path,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        expected_control_sha256=control_pin,
        expected_control_step=control_step_pin,
        expected_high_dropout_sha256=high_pin,
        expected_high_dropout_step=high_step_pin,
    )

    if args.precise_eval_iters > 0:
        precise_report = run_precise_validation(
            data,
            config,
            args.n_head,
            args.n_layer,
            device,
            schedule,
            treatment_spec,
            report,
            eval_iters=args.precise_eval_iters,
            seed=precise_eval_seed,
        )
        print_precise_validation(precise_report)


if __name__ == "__main__":
    main()
