from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import EvaluationRecord, TrainingResult, seed_everything


# Stage 12 keeps the complete Stage 11 state fixed at the fork, including
# AdamW's moments and the explicit training-batch generator. Each branch loads
# that checkpoint independently; only the optimizer learning rate is changed.
_STAGE_11_PATH = Path(__file__).with_name("11_train_longer.py")
_STAGE_11_MODULE_NAME = "stage_11_train_longer_for_stage_12"
_STAGE_11_SPEC = importlib.util.spec_from_file_location(
    _STAGE_11_MODULE_NAME,
    _STAGE_11_PATH,
)
assert _STAGE_11_SPEC is not None and _STAGE_11_SPEC.loader is not None
_STAGE_11 = importlib.util.module_from_spec(_STAGE_11_SPEC)
sys.modules[_STAGE_11_MODULE_NAME] = _STAGE_11
_STAGE_11_SPEC.loader.exec_module(_STAGE_11)

GPTLanguageModel = _STAGE_11.GPTLanguageModel
ResumeState = _STAGE_11.ResumeState
BestValidationCheckpoint = _STAGE_11.BestValidationCheckpoint
evaluate_on_fixed_batches = _STAGE_11.evaluate_on_fixed_batches
fingerprint_data = _STAGE_11.fingerprint_data
train_until = _STAGE_11.train_until

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_SOURCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_11_best_checkpoint.pt"
)
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_12_control_best_checkpoint.pt"
)
DEFAULT_LR_DROP_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_12_lr_drop_best_checkpoint.pt"
)
DEFAULT_SOURCE_STEP = 10_000
DEFAULT_TARGET_STEP = 15_000
DEFAULT_SOURCE_LEARNING_RATE = 1e-3
DEFAULT_REDUCED_LEARNING_RATE = 3e-4


@dataclass(frozen=True, slots=True)
class BranchSpec:
    name: str
    learning_rate: float
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch name must not be empty")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(
                "branch learning rate must be finite and positive, got "
                f"{self.learning_rate}"
            )


@dataclass(slots=True)
class LoadedBranch:
    model: GPTLanguageModel
    optimizer: torch.optim.Optimizer
    training_generator: torch.Generator
    config: TrainingConfig
    resume: ResumeState


@dataclass(slots=True)
class BranchValidationCheckpoint(BestValidationCheckpoint):
    """Add the learning-rate fork provenance to a Stage 11 checkpoint."""

    branch_name: str = ""
    source_checkpoint_sha256: str = ""
    source_step: int = 0
    source_learning_rate: float = 0.0
    branch_learning_rate: float = 0.0

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(BranchValidationCheckpoint, self)._payload(step)
        payload["checkpoint_kind"] = "best"
        payload["experiment"] = {
            "stage": 12,
            "branch": self.branch_name,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_step": self.source_step,
            "source_learning_rate": self.source_learning_rate,
            "branch_learning_rate": self.branch_learning_rate,
            "learning_rate_changed": (
                self.branch_learning_rate != self.source_learning_rate
            ),
        }
        return payload


@dataclass(frozen=True, slots=True)
class BranchReport:
    name: str
    learning_rate: float
    training: TrainingResult
    resume: ResumeState
    generalization_gap: float
    best_val_loss: float
    best_step: int
    checkpoint_path: Path
    sample: str


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    parameter_count: int
    source_checkpoint: Path
    source_checkpoint_sha256: str
    source_step: int
    source_val_loss: float
    control: BranchReport
    lr_drop: BranchReport
    final_val_delta: float
    best_val_delta: float

    @property
    def final_winner(self) -> str:
        if self.final_val_delta < 0:
            return self.lr_drop.name
        if self.final_val_delta > 0:
            return self.control.name
        return "tie"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fork the exact Stage 11 step-10,000 checkpoint and compare "
            "5,000 more steps at 1e-3 versus 3e-4."
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
        help="Learning rate recorded in the source checkpoint.",
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


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_distinct_paths(
    source_checkpoint: Path,
    control_checkpoint: Path,
    lr_drop_checkpoint: Path,
) -> None:
    control_temporary = control_checkpoint.with_name(
        f".{control_checkpoint.name}.tmp"
    )
    lr_drop_temporary = lr_drop_checkpoint.with_name(
        f".{lr_drop_checkpoint.name}.tmp"
    )
    labeled_paths = {
        "source checkpoint": source_checkpoint.resolve(),
        "control checkpoint": control_checkpoint.resolve(),
        "control temporary checkpoint": control_temporary.resolve(),
        "LR-drop checkpoint": lr_drop_checkpoint.resolve(),
        "LR-drop temporary checkpoint": lr_drop_temporary.resolve(),
    }
    labels = tuple(labeled_paths)
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1 :]:
            if labeled_paths[first_label] == labeled_paths[second_label]:
                raise ValueError(
                    f"{first_label} and {second_label} must use different "
                    f"paths: {labeled_paths[first_label]}"
                )


def validate_source_checkpoint_contract(
    path: Path,
    *,
    expected_step: int,
) -> Mapping[str, object]:
    if expected_step < 0:
        raise ValueError(
            f"source step must be non-negative, got {expected_step}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage 12 requires a full mapping checkpoint")

    required_keys = {
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "best_val_loss",
        "best_step",
        "training_generator_state",
        "rng_state",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            "Stage 12 requires an exact resumable checkpoint; missing: "
            f"{missing}"
        )

    step = checkpoint["step"]
    best_step = checkpoint["best_step"]
    if step != expected_step:
        raise ValueError(
            f"source checkpoint must be step {expected_step}, got {step!r}"
        )
    if best_step != step:
        raise ValueError(
            "source checkpoint must contain the exact branch-point model; "
            f"best_step {best_step!r} does not equal step {step!r}"
        )

    generator_state = checkpoint["training_generator_state"]
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError(
            "source checkpoint training_generator_state must be a tensor"
        )
    rng_state = checkpoint["rng_state"]
    if (
        not isinstance(rng_state, Mapping)
        or not isinstance(rng_state.get("cpu"), torch.Tensor)
    ):
        raise ValueError("source checkpoint must contain the CPU RNG state")

    if checkpoint.get("optimizer_provenance_known") is not True:
        raise ValueError(
            "Stage 12 requires known optimizer provenance from Stage 11"
        )
    if checkpoint.get("optimizer_restart_step") is not None:
        raise ValueError(
            "Stage 12 requires uninterrupted AdamW lineage; checkpoint "
            f"records a restart at step {checkpoint['optimizer_restart_step']}"
        )

    optimizer_state = checkpoint["optimizer_state_dict"]
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("source optimizer_state_dict must be a mapping")
    moment_states = optimizer_state.get("state")
    if not isinstance(moment_states, Mapping) or not moment_states:
        raise ValueError(
            "source checkpoint must contain non-empty AdamW moment state"
        )

    return checkpoint


def override_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(
            "learning rate must be finite and positive, got "
            f"{learning_rate}"
        )
    if not optimizer.param_groups:
        raise ValueError("optimizer must contain at least one parameter group")

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def clear_accelerator_cache(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def validate_clean_resume(
    resume: ResumeState,
    *,
    expected_step: int,
) -> None:
    if resume.mode != "full":
        raise ValueError("Stage 12 requires a full Stage 11 checkpoint")
    if resume.start_step != expected_step:
        raise ValueError(
            f"expected source step {expected_step}, got {resume.start_step}"
        )
    if not resume.optimizer_restored:
        raise ValueError("Stage 12 requires restored AdamW state")
    if not resume.training_generator_restored:
        raise ValueError(
            "Stage 12 requires the saved training-batch generator state"
        )
    if not resume.optimizer_provenance_known:
        raise ValueError("Stage 12 requires known AdamW provenance")
    if resume.optimizer_restart_step is not None:
        raise ValueError(
            "Stage 12 requires uninterrupted AdamW lineage, not a restart at "
            f"step {resume.optimizer_restart_step}"
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
    # Constructing then restoring each branch independently prevents the first
    # branch from leaking changed weights, moments, batches, or RNG into the
    # second branch.
    seed_everything(source_config.seed)
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=source_config.block_size,
        n_embd=source_config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=source_config.learning_rate,
    )
    training_generator = torch.Generator().manual_seed(source_config.seed + 1)

    resume = _STAGE_11.load_resume_checkpoint(
        source_checkpoint,
        model,
        optimizer,
        training_generator,
        data,
        source_config,
        device,
        allow_optimizer_restart=False,
        legacy_step=expected_source_step,
        n_head=n_head,
        n_layer=n_layer,
    )
    validate_clean_resume(resume, expected_step=expected_source_step)

    # optimizer.load_state_dict above restores the checkpoint's 1e-3 value.
    # Mutate every group only after that exact state has been recovered.
    override_optimizer_learning_rate(optimizer, spec.learning_rate)
    branch_config = replace(
        source_config,
        learning_rate=spec.learning_rate,
    )

    return LoadedBranch(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        config=branch_config,
        resume=resume,
    )


def generate_from_final_model(
    model: GPTLanguageModel,
    data: CharacterData,
    device: torch.device,
    *,
    sample_length: int,
    seed: int,
) -> str:
    model.eval()
    seed_everything(seed)
    start_id = data.vocabulary.stoi.get("\n", 0)
    context = torch.tensor([[start_id]], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=sample_length)
    return data.vocabulary.decode(generated[0].cpu().tolist())


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

    # Materialize the common branch point under the branch's active LR before
    # any update. A branch that never improves still gets a valid checkpoint.
    best.save(branch.resume.start_step)

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
    source_payload = validate_source_checkpoint_contract(
        source_checkpoint,
        expected_step=expected_source_step,
    )
    source_hash = checkpoint_sha256(source_checkpoint)

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

    control_steps = tuple(record.step for record in control.training.history)
    lr_drop_steps = tuple(record.step for record in lr_drop.training.history)
    if control_steps != lr_drop_steps:
        raise RuntimeError("branches were not evaluated at matching steps")

    source_val_loss = float(source_payload["best_val_loss"])
    final_val_delta = lr_drop.training.final.val - control.training.final.val
    best_val_delta = lr_drop.best_val_loss - control.best_val_loss

    print("\nStage 12 experimental summary")
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
        f"  LR-drop final val:         {lr_drop.training.final.val:.4f} "
        f"(lr {lr_drop.learning_rate:g})"
    )
    print(
        f"  final delta (drop-control): {final_val_delta:+.4f} "
        "(negative favors LR drop)"
    )
    print(
        f"  control best val:          {control.best_val_loss:.4f} "
        f"@ {control.best_step}"
    )
    print(
        f"  LR-drop best val:          {lr_drop.best_val_loss:.4f} "
        f"@ {lr_drop.best_step}"
    )
    print(f"  control checkpoint:        {control.checkpoint_path}")
    print(f"  LR-drop checkpoint:        {lr_drop.checkpoint_path}")
    print("\nControl final-step generated text")
    print(control.sample)
    print("\nLR-drop final-step generated text")
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

    print("Stage 12: controlled learning-rate drop")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Fork step={args.source_step:,}, target step={config.max_iters:,}, "
        f"control lr={control_spec.learning_rate:g}, "
        f"drop lr={lr_drop_spec.learning_rate:g}"
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
