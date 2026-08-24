from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import EvaluationRecord


# Stage 13 reuses the tested Stage 12 fork machinery, but owns its CLI,
# source-contract checks, reporting, and checkpoint provenance. This avoids
# mislabeling a second decay experiment as another Stage 12 run.
_STAGE_12_PATH = Path(__file__).with_name("12_learning_rate_drop.py")
_STAGE_12_MODULE_NAME = "stage_12_learning_rate_drop_for_stage_13"
_STAGE_12_SPEC = importlib.util.spec_from_file_location(
    _STAGE_12_MODULE_NAME,
    _STAGE_12_PATH,
)
assert _STAGE_12_SPEC is not None and _STAGE_12_SPEC.loader is not None
_STAGE_12 = importlib.util.module_from_spec(_STAGE_12_SPEC)
sys.modules[_STAGE_12_MODULE_NAME] = _STAGE_12
_STAGE_12_SPEC.loader.exec_module(_STAGE_12)

GPTLanguageModel = _STAGE_12.GPTLanguageModel
ResumeState = _STAGE_12.ResumeState
BranchSpec = _STAGE_12.BranchSpec
LoadedBranch = _STAGE_12.LoadedBranch
BranchReport = _STAGE_12.BranchReport
ExperimentReport = _STAGE_12.ExperimentReport
checkpoint_sha256 = _STAGE_12.checkpoint_sha256
clear_accelerator_cache = _STAGE_12.clear_accelerator_cache
evaluate_on_fixed_batches = _STAGE_12.evaluate_on_fixed_batches
fingerprint_data = _STAGE_12.fingerprint_data
generate_from_final_model = _STAGE_12.generate_from_final_model
override_optimizer_learning_rate = _STAGE_12.override_optimizer_learning_rate
train_until = _STAGE_12.train_until
validate_distinct_paths = _STAGE_12.validate_distinct_paths

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_SOURCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_12_lr_drop_best_checkpoint.pt"
)
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_13_control_best_checkpoint.pt"
)
DEFAULT_LR_DROP_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_13_lr_drop_best_checkpoint.pt"
)
DEFAULT_SOURCE_STEP = 13_000
DEFAULT_TARGET_STEP = 18_000
DEFAULT_SOURCE_LEARNING_RATE = 3e-4
DEFAULT_REDUCED_LEARNING_RATE = 1e-4


@dataclass(slots=True)
class BranchValidationCheckpoint(_STAGE_12.BranchValidationCheckpoint):
    """Save a full Stage 13 checkpoint with second-decay provenance."""

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(BranchValidationCheckpoint, self)._payload(step)
        experiment = payload["experiment"]
        assert isinstance(experiment, dict)
        experiment.update(
            {
                "stage": 13,
                "source_stage": 12,
                "source_branch": "lr_drop",
            }
        )
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fork the exact Stage 12 step-13,000 LR-drop checkpoint and "
            "compare 5,000 more steps at 3e-4 versus 1e-4."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument(
        "--source-learning-rate",
        type=float,
        default=DEFAULT_SOURCE_LEARNING_RATE,
        help="Learning rate recorded in the Stage 12 source checkpoint.",
    )
    parser.add_argument(
        "--control-learning-rate",
        type=float,
        default=DEFAULT_SOURCE_LEARNING_RATE,
    )
    parser.add_argument(
        "--reduced-learning-rate",
        type=float,
        default=DEFAULT_REDUCED_LEARNING_RATE,
    )
    parser.add_argument("--source-step", type=int, default=DEFAULT_SOURCE_STEP)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_TARGET_STEP)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--control-checkpoint-path",
        type=Path,
        default=DEFAULT_CONTROL_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--lr-drop-checkpoint-path",
        type=Path,
        default=DEFAULT_LR_DROP_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"source checkpoint {name} must be a mapping")
    return value


def validate_source_checkpoint_contract(
    path: Path,
    *,
    expected_step: int,
    expected_learning_rate: float,
) -> Mapping[str, object]:
    """Require the exact, clean Stage 12 LR-drop winner used by Stage 13."""

    if not math.isfinite(expected_learning_rate) or expected_learning_rate <= 0:
        raise ValueError(
            "source learning rate must be finite and positive, got "
            f"{expected_learning_rate}"
        )

    checkpoint = _STAGE_12.validate_source_checkpoint_contract(
        path,
        expected_step=expected_step,
    )
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("Stage 13 requires a Stage 12 best checkpoint")

    training_config = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    if training_config.get("learning_rate") != expected_learning_rate:
        raise ValueError(
            "source checkpoint learning rate does not match requested "
            f"{expected_learning_rate}: "
            f"got {training_config.get('learning_rate')!r}"
        )

    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"),
        name="optimizer_state_dict",
    )
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("source optimizer_state_dict must contain parameter groups")
    loaded_learning_rates: set[float] = set()
    for group in param_groups:
        if not isinstance(group, Mapping) or "lr" not in group:
            raise ValueError(
                "every source optimizer parameter group must contain an lr"
            )
        loaded_learning_rates.add(float(group["lr"]))
    if loaded_learning_rates != {expected_learning_rate}:
        raise ValueError(
            "source optimizer learning rate does not match requested "
            f"{expected_learning_rate}: got {sorted(loaded_learning_rates)}"
        )

    rng_state = _require_mapping(
        checkpoint.get("rng_state"),
        name="rng_state",
    )
    for accelerator in ("xpu", "cuda"):
        if accelerator not in rng_state:
            continue
        accelerator_states = rng_state[accelerator]
        if (
            not isinstance(accelerator_states, list)
            or not accelerator_states
            or not all(isinstance(state, torch.Tensor) for state in accelerator_states)
        ):
            raise ValueError(
                f"source checkpoint rng_state['{accelerator}'] must be a "
                "non-empty tensor list"
            )

    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    if experiment.get("stage") != 12:
        raise ValueError("Stage 13 source must have experiment.stage == 12")
    if experiment.get("branch") != "lr_drop":
        raise ValueError("Stage 13 source must be the Stage 12 lr_drop branch")
    if experiment.get("branch_learning_rate") != expected_learning_rate:
        raise ValueError(
            "Stage 12 branch learning rate does not match requested "
            f"{expected_learning_rate}"
        )
    if experiment.get("learning_rate_changed") is not True:
        raise ValueError("Stage 13 source must record learning_rate_changed as true")

    predecessor_step = experiment.get("source_step")
    if (
        isinstance(predecessor_step, bool)
        or not isinstance(predecessor_step, int)
        or predecessor_step < 0
        or predecessor_step >= expected_step
    ):
        raise ValueError("Stage 12 source_step must be a non-negative earlier step")
    predecessor_learning_rate = experiment.get("source_learning_rate")
    if (
        isinstance(predecessor_learning_rate, bool)
        or not isinstance(predecessor_learning_rate, (int, float))
        or not math.isfinite(float(predecessor_learning_rate))
        or float(predecessor_learning_rate) <= expected_learning_rate
    ):
        raise ValueError("Stage 12 source learning rate must exceed its branch rate")

    return checkpoint


def validate_device_rng_state(
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> None:
    """Require saved RNG state for the accelerator used by the experiment."""

    if device.type not in ("xpu", "cuda"):
        return
    rng_state = _require_mapping(
        checkpoint.get("rng_state"),
        name="rng_state",
    )
    if device.type not in rng_state:
        raise ValueError(
            f"Stage 13 on {device.type} requires saved {device.type} RNG state"
        )


def load_branch(
    data: CharacterData,
    source_config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    source_checkpoint: Path,
    spec: BranchSpec,
    *,
    expected_source_step: int,
) -> LoadedBranch:
    return _STAGE_12.load_branch(
        data,
        source_config,
        n_head,
        n_layer,
        device,
        source_checkpoint,
        spec,
        expected_source_step=expected_source_step,
    )


def make_evaluation_callback(
    best: BranchValidationCheckpoint,
) -> Callable[[EvaluationRecord], None]:
    def record_evaluation(record: EvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        marker = "  * best" if is_best else ""
        losses = record.losses
        gap = losses.val - losses.train
        print(
            f"  step {record.step:5d} | train {losses.train:.4f} | "
            f"val {losses.val:.4f} | gap {gap:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )

    return record_evaluation


def run_branch(
    data: CharacterData,
    source_config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    source_checkpoint: Path,
    source_checkpoint_hash: str,
    spec: BranchSpec,
    *,
    expected_source_step: int,
) -> tuple[BranchReport, int]:
    if checkpoint_sha256(source_checkpoint) != source_checkpoint_hash:
        raise RuntimeError("source checkpoint changed before branch loading")

    branch = load_branch(
        data,
        source_config,
        n_head,
        n_layer,
        device,
        source_checkpoint,
        spec,
        expected_source_step=expected_source_step,
    )
    if checkpoint_sha256(source_checkpoint) != source_checkpoint_hash:
        raise RuntimeError("source checkpoint changed while loading a branch")

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
        config=branch.config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=branch.resume.optimizer_restart_step,
        optimizer_provenance_known=branch.resume.optimizer_provenance_known,
        best_val_loss=branch.resume.best_val_loss,
        best_step=branch.resume.best_step,
        branch_name=spec.name,
        source_checkpoint_sha256=source_checkpoint_hash,
        source_step=branch.resume.start_step,
        source_learning_rate=source_config.learning_rate,
        branch_learning_rate=spec.learning_rate,
    )

    # Ensure even a branch with no improvement has a complete, resumable
    # checkpoint under its active learning rate.
    best.save(branch.resume.start_step)
    record_evaluation = make_evaluation_callback(best)

    print(f"\n{spec.name} branch (lr={spec.learning_rate:g})")
    print(f"  source step: {branch.resume.start_step}")
    print(f"  target step: {branch.config.max_iters}")
    result = train_until(
        branch.model,
        branch.optimizer,
        data,
        branch.config,
        device,
        branch.training_generator,
        start_step=branch.resume.start_step,
        on_evaluation=record_evaluation,
    )

    assert best.best_step is not None
    sample = generate_from_final_model(
        branch.model,
        data,
        device,
        sample_length=sample_length,
        seed=source_config.seed + 2,
    )
    report = BranchReport(
        name=spec.name,
        learning_rate=spec.learning_rate,
        training=result,
        resume=branch.resume,
        generalization_gap=result.final.val - result.final.train,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        checkpoint_path=spec.checkpoint_path,
        sample=sample,
    )

    del record_evaluation
    del best
    del branch
    clear_accelerator_cache(device)
    return report, parameter_count


def run_experiment(
    data: CharacterData,
    source_config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    source_checkpoint: Path,
    control_spec: BranchSpec,
    lr_drop_spec: BranchSpec,
    *,
    expected_source_step: int = DEFAULT_SOURCE_STEP,
) -> ExperimentReport:
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if source_config.max_iters <= expected_source_step:
        raise ValueError(
            f"target step {source_config.max_iters} must exceed source step "
            f"{expected_source_step}"
        )
    if control_spec.learning_rate != source_config.learning_rate:
        raise ValueError(
            "control learning rate must equal the source learning rate: "
            f"{control_spec.learning_rate} != {source_config.learning_rate}"
        )
    if lr_drop_spec.learning_rate >= control_spec.learning_rate:
        raise ValueError(
            "reduced learning rate must be below the control learning rate"
        )
    if control_spec.name == lr_drop_spec.name:
        raise ValueError("branch names must be different")

    validate_distinct_paths(
        source_checkpoint,
        control_spec.checkpoint_path,
        lr_drop_spec.checkpoint_path,
    )
    hash_before_validation = checkpoint_sha256(source_checkpoint)
    source_payload = validate_source_checkpoint_contract(
        source_checkpoint,
        expected_step=expected_source_step,
        expected_learning_rate=source_config.learning_rate,
    )
    validate_device_rng_state(source_payload, device)
    source_hash = checkpoint_sha256(source_checkpoint)
    if source_hash != hash_before_validation:
        raise RuntimeError("source checkpoint changed during validation")

    print("\nPaired fixed-batch evaluation")
    print(
        "  Every measurement and both branches use the same sampled train "
        "and validation blocks."
    )
    print(f"  source checkpoint SHA-256: {source_hash}")

    control, control_parameter_count = run_branch(
        data,
        source_config,
        n_head,
        n_layer,
        device,
        sample_length,
        source_checkpoint,
        source_hash,
        control_spec,
        expected_source_step=expected_source_step,
    )
    lr_drop, lr_drop_parameter_count = run_branch(
        data,
        source_config,
        n_head,
        n_layer,
        device,
        sample_length,
        source_checkpoint,
        source_hash,
        lr_drop_spec,
        expected_source_step=expected_source_step,
    )

    if checkpoint_sha256(source_checkpoint) != source_hash:
        raise RuntimeError("source checkpoint changed during the experiment")
    if control_parameter_count != lr_drop_parameter_count:
        raise RuntimeError("branch parameter counts do not match")
    if control.training.initial != lr_drop.training.initial:
        raise RuntimeError("branches did not start from the same losses")

    control_steps = tuple(record.step for record in control.training.history)
    lr_drop_steps = tuple(record.step for record in lr_drop.training.history)
    if control_steps != lr_drop_steps:
        raise RuntimeError("branches were not evaluated at matching steps")

    source_val_loss = float(source_payload["best_val_loss"])
    final_val_delta = lr_drop.training.final.val - control.training.final.val
    best_val_delta = lr_drop.best_val_loss - control.best_val_loss

    print("\nStage 13 experimental summary")
    print(
        f"  B / T / C / H / D / FF / L: "
        f"{source_config.batch_size} / {source_config.block_size} / "
        f"{source_config.n_embd} / {n_head} / "
        f"{source_config.n_embd // n_head} / "
        f"{4 * source_config.n_embd} / {n_layer}"
    )
    print(f"  parameter count:           {control_parameter_count:,}")
    print(
        f"  common source:             step {expected_source_step:,}, "
        f"val {source_val_loss:.4f}, lr {source_config.learning_rate:g}"
    )
    print(
        f"  control final val:         {control.training.final.val:.4f} "
        f"(lr {control.learning_rate:g})"
    )
    print(
        f"  second-drop final val:     {lr_drop.training.final.val:.4f} "
        f"(lr {lr_drop.learning_rate:g})"
    )
    print(
        f"  final delta (drop-control): {final_val_delta:+.4f} "
        "(negative favors second drop)"
    )
    print(
        f"  control best val:          {control.best_val_loss:.4f} "
        f"@ {control.best_step}"
    )
    print(
        f"  second-drop best val:      {lr_drop.best_val_loss:.4f} "
        f"@ {lr_drop.best_step}"
    )
    print(
        f"  best delta (drop-control):  {best_val_delta:+.4f} "
        "(negative favors second drop)"
    )
    print(f"  control checkpoint:        {control.checkpoint_path}")
    print(f"  second-drop checkpoint:    {lr_drop.checkpoint_path}")
    print("\nControl final-step generated text")
    print(control.sample)
    print("\nSecond-drop final-step generated text")
    print(lr_drop.sample)

    return ExperimentReport(
        parameter_count=control_parameter_count,
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=source_hash,
        source_step=expected_source_step,
        source_val_loss=source_val_loss,
        control=control,
        lr_drop=lr_drop,
        final_val_delta=final_val_delta,
        best_val_delta=best_val_delta,
    )


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        learning_rate=args.source_learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    if args.n_head <= 0 or config.n_embd % args.n_head != 0:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({args.n_head})"
        )
    if args.n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {args.n_layer}")

    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)
    control_spec = BranchSpec(
        name="control",
        learning_rate=args.control_learning_rate,
        checkpoint_path=args.control_checkpoint_path,
    )
    lr_drop_spec = BranchSpec(
        name="lr_drop",
        learning_rate=args.reduced_learning_rate,
        checkpoint_path=args.lr_drop_checkpoint_path,
    )

    print("Stage 13: second controlled learning-rate drop")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Fork step={args.source_step:,}, target step={config.max_iters:,}, "
        f"control lr={control_spec.learning_rate:g}, "
        f"second-drop lr={lr_drop_spec.learning_rate:g}"
    )
    print(f"Checkpoint input: {args.source_checkpoint}")

    run_experiment(
        data,
        config,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        args.source_checkpoint,
        control_spec,
        lr_drop_spec,
        expected_source_step=args.source_step,
    )


if __name__ == "__main__":
    main()
