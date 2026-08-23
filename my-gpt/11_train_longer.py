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
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import (
    BenchmarkConfig,
    BenchmarkStats,
    EvaluationRecord,
    LossEstimate,
    TrainingResult,
    estimate_loss,
    seed_everything,
    sync_device,
)


# Stage 11 deliberately reuses the Stage 10 model. The only experimental
# change in the default run is the total training duration: 5,000 -> 10,000.
_STAGE_10_PATH = Path(__file__).with_name("10_scale_width.py")
_STAGE_10_MODULE_NAME = "stage_10_scale_width_for_stage_11"
_STAGE_10_SPEC = importlib.util.spec_from_file_location(
    _STAGE_10_MODULE_NAME,
    _STAGE_10_PATH,
)
assert _STAGE_10_SPEC is not None and _STAGE_10_SPEC.loader is not None
_STAGE_10 = importlib.util.module_from_spec(_STAGE_10_SPEC)
sys.modules[_STAGE_10_MODULE_NAME] = _STAGE_10
_STAGE_10_SPEC.loader.exec_module(_STAGE_10)

GPTLanguageModel = _STAGE_10.GPTLanguageModel
ShapeAndNormReport = _STAGE_10.ShapeAndNormReport
GradientReport = _STAGE_10.GradientReport
print_parameter_report = _STAGE_10.print_parameter_report
print_shape_and_norm_report = _STAGE_10.print_shape_and_norm_report
print_gradient_report = _STAGE_10.print_gradient_report
run_independent_benchmark = _STAGE_10.run_independent_benchmark
print_benchmark = _STAGE_10.print_benchmark

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "stage_11_best_checkpoint.pt"
)

ResumeMode = Literal["fresh", "full", "legacy_weights"]


@dataclass(frozen=True, slots=True)
class ResumeState:
    mode: ResumeMode
    start_step: int
    best_val_loss: float
    best_step: int | None
    optimizer_restored: bool
    training_generator_restored: bool
    optimizer_restart_step: int | None
    optimizer_provenance_known: bool

    @property
    def description(self) -> str:
        if self.mode == "fresh":
            return "fresh uninterrupted run"
        if self.mode == "full":
            if not self.optimizer_provenance_known:
                return "full checkpoint; prior optimizer lineage unknown"
            if self.optimizer_restart_step is not None:
                return (
                    "full checkpoint; AdamW lineage restarted at step "
                    f"{self.optimizer_restart_step}"
                )
            return "full model + AdamW checkpoint (uninterrupted lineage)"
        return "Stage-10 weights + restarted AdamW state"


@dataclass(slots=True)
class BestValidationCheckpoint:
    """Track the best validation result and save a resumable checkpoint."""

    model: GPTLanguageModel
    optimizer: torch.optim.Optimizer
    training_generator: torch.Generator
    path: Path
    device: torch.device
    config: TrainingConfig
    n_head: int
    n_layer: int
    data_fingerprint: str | None = None
    optimizer_restart_step: int | None = None
    optimizer_provenance_known: bool = True
    best_val_loss: float = float("inf")
    best_step: int | None = None

    def _payload(self, step: int) -> dict[str, Any]:
        # XPU work and device-to-host transfers are asynchronous. Persist an
        # owned CPU snapshot so later optimizer steps cannot race serialization
        # and so checkpoints remain portable across device types.
        sync_device(self.device)
        return {
            "checkpoint_version": 1,
            "step": step,
            "model_state_dict": clone_to_cpu(self.model.state_dict()),
            "optimizer_state_dict": clone_to_cpu(
                self.optimizer.state_dict()
            ),
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "training_generator_state": (
                self.training_generator.get_state().clone()
            ),
            "rng_state": capture_rng_state(self.device),
            "architecture": {
                "block_size": self.config.block_size,
                "n_embd": self.config.n_embd,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
            },
            "training_config": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "eval_interval": self.config.eval_interval,
                "eval_iters": self.config.eval_iters,
                "seed": self.config.seed,
            },
            "data_fingerprint": self.data_fingerprint,
            "optimizer_restart_step": self.optimizer_restart_step,
            "optimizer_provenance_known": self.optimizer_provenance_known,
        }

    def save(self, step: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")

        try:
            torch.save(self._payload(step), temporary_path)
            temporary_path.replace(self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def consider(self, record: EvaluationRecord) -> bool:
        if not math.isfinite(record.losses.val):
            raise ValueError(
                "validation loss must be finite at step "
                f"{record.step}, got {record.losses.val}"
            )

        if record.losses.val >= self.best_val_loss:
            return False

        self.best_val_loss = record.losses.val
        self.best_step = record.step
        self.save(record.step)
        return True


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    parameter_count: int
    shapes_and_norms: ShapeAndNormReport
    gradients: GradientReport
    training: TrainingResult
    benchmark: BenchmarkStats | None
    resume: ResumeState
    generalization_gap: float
    best_val_loss: float
    best_step: int
    checkpoint_path: Path
    sample: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the unchanged Stage 10 model for 10,000 total steps to "
            "measure convergence and overfitting."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Optional full checkpoint to resume. A legacy weights-only "
            "checkpoint also requires --allow-optimizer-restart."
        ),
    )
    parser.add_argument(
        "--allow-optimizer-restart",
        action="store_true",
        help=(
            "Allow a weights-only warm start with a fresh AdamW optimizer. "
            "This is not an exact continuation."
        ),
    )
    parser.add_argument(
        "--legacy-step",
        type=int,
        default=5_000,
        help="Step represented by a legacy weights-only checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return value


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state().clone()}

    if device.type == "xpu":
        state["xpu"] = [
            device_state.cpu().clone()
            for device_state in torch.xpu.get_rng_state_all()
        ]
    elif device.type == "cuda":
        state["cuda"] = [
            device_state.cpu().clone()
            for device_state in torch.cuda.get_rng_state_all()
        ]

    return state


def restore_rng_state(state: object, device: torch.device) -> bool:
    if not isinstance(state, Mapping) or "cpu" not in state:
        return False

    cpu_state = state["cpu"]
    if not isinstance(cpu_state, torch.Tensor):
        raise ValueError("checkpoint rng_state['cpu'] must be a tensor")
    torch.set_rng_state(cpu_state.cpu())

    if device.type == "xpu" and "xpu" in state:
        xpu_state = state["xpu"]
        if not isinstance(xpu_state, list):
            raise ValueError("checkpoint rng_state['xpu'] must be a list")
        torch.xpu.set_rng_state_all(
            [device_state.cpu() for device_state in xpu_state]
        )
    elif device.type == "cuda" and "cuda" in state:
        cuda_state = state["cuda"]
        if not isinstance(cuda_state, list):
            raise ValueError("checkpoint rng_state['cuda'] must be a list")
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in cuda_state]
        )

    return True


def is_legacy_model_state(checkpoint: object) -> bool:
    return (
        isinstance(checkpoint, Mapping)
        and bool(checkpoint)
        and all(isinstance(key, str) for key in checkpoint)
        and all(isinstance(value, torch.Tensor) for value in checkpoint.values())
    )


def _require_step(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def fingerprint_data(data: CharacterData) -> str:
    """Identify the vocabulary, split boundary, and token contents."""

    digest = hashlib.sha256()
    digest.update("\0".join(data.vocabulary.chars).encode("utf-8"))
    digest.update(len(data.train_data).to_bytes(8, "little"))
    digest.update(data.train_data.contiguous().numpy().tobytes())
    digest.update(len(data.val_data).to_bytes(8, "little"))
    digest.update(data.val_data.contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, object],
    *,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int | None,
    n_layer: int | None,
) -> None:
    version = checkpoint.get("checkpoint_version")
    if version is not None and version != 1:
        raise ValueError(f"unsupported checkpoint_version: {version!r}")

    architecture = checkpoint.get("architecture")
    if architecture is not None:
        if not isinstance(architecture, Mapping):
            raise ValueError("checkpoint architecture must be a mapping")
        expected_architecture = {
            "block_size": config.block_size,
            "n_embd": config.n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
        }
        for key, expected in expected_architecture.items():
            if expected is not None and architecture.get(key) != expected:
                raise ValueError(
                    f"checkpoint architecture {key}={architecture.get(key)!r} "
                    f"does not match requested {expected!r}"
                )

    training_config = checkpoint.get("training_config")
    if training_config is not None:
        if not isinstance(training_config, Mapping):
            raise ValueError("checkpoint training_config must be a mapping")
        expected_training = {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "eval_interval": config.eval_interval,
            "eval_iters": config.eval_iters,
            "seed": config.seed,
        }
        for key, expected in expected_training.items():
            if training_config.get(key) != expected:
                raise ValueError(
                    f"checkpoint training_config {key}="
                    f"{training_config.get(key)!r} does not match requested "
                    f"{expected!r}"
                )

    saved_fingerprint = checkpoint.get("data_fingerprint")
    if (
        saved_fingerprint is not None
        and saved_fingerprint != fingerprint_data(data)
    ):
        raise ValueError("checkpoint data fingerprint does not match the corpus")


def advance_training_generator(
    generator: torch.Generator,
    *,
    num_steps: int,
    batch_size: int,
    num_train_tokens: int,
    block_size: int,
) -> None:
    """Reconstruct the explicit Stage 10 batch stream through num_steps."""

    maximum_start = num_train_tokens - block_size
    if maximum_start <= 0:
        raise ValueError(
            f"training split has {num_train_tokens} tokens, but block_size "
            f"is {block_size}"
        )

    for _ in range(num_steps):
        torch.randint(
            0,
            maximum_start,
            (batch_size,),
            generator=generator,
        )


def load_resume_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    allow_optimizer_restart: bool,
    legacy_step: int,
    n_head: int | None = None,
    n_layer: int | None = None,
) -> ResumeState:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if is_legacy_model_state(checkpoint):
        if not allow_optimizer_restart:
            raise ValueError(
                f"{path} contains model weights only; it cannot exactly "
                "resume AdamW. Pass --allow-optimizer-restart to run the "
                "explicitly labeled warm-start experiment, or omit "
                "--resume-from for a clean uninterrupted run."
            )

        start_step = _require_step(legacy_step, name="legacy_step")
        if start_step >= config.max_iters:
            raise ValueError(
                f"legacy_step {start_step} must be below target step "
                f"{config.max_iters}"
            )
        model.load_state_dict(checkpoint)
        advance_training_generator(
            training_generator,
            num_steps=start_step,
            batch_size=config.batch_size,
            num_train_tokens=len(data.train_data),
            block_size=config.block_size,
        )
        return ResumeState(
            mode="legacy_weights",
            start_step=start_step,
            best_val_loss=float("inf"),
            best_step=None,
            optimizer_restored=False,
            training_generator_restored=False,
            optimizer_restart_step=start_step,
            optimizer_provenance_known=True,
        )

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"unsupported checkpoint object: {type(checkpoint).__name__}")

    required_keys = {
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "best_val_loss",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"checkpoint is missing required keys: {missing}")

    validate_checkpoint_metadata(
        checkpoint,
        data=data,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
    )

    start_step = _require_step(checkpoint["step"], name="checkpoint step")
    if start_step >= config.max_iters:
        raise ValueError(
            f"checkpoint step {start_step} must be below target step "
            f"{config.max_iters}"
        )
    best_val_loss = float(checkpoint["best_val_loss"])
    if not math.isfinite(best_val_loss):
        raise ValueError(
            f"checkpoint best_val_loss must be finite, got {best_val_loss}"
        )

    best_step_value = checkpoint.get("best_step", start_step)
    best_step = _require_step(best_step_value, name="checkpoint best_step")
    if best_step != start_step:
        raise ValueError(
            "Stage 11 checkpoints contain best-state weights, so "
            f"best_step {best_step} must equal step {start_step}"
        )

    provenance_value = checkpoint.get(
        "optimizer_provenance_known",
        "optimizer_restart_step" in checkpoint,
    )
    if not isinstance(provenance_value, bool):
        raise ValueError("optimizer_provenance_known must be a boolean")
    provenance_known = provenance_value
    restart_step_value = checkpoint.get("optimizer_restart_step")
    optimizer_restart_step = None
    if restart_step_value is not None:
        optimizer_restart_step = _require_step(
            restart_step_value,
            name="optimizer_restart_step",
        )
        if optimizer_restart_step > start_step:
            raise ValueError(
                f"optimizer_restart_step {optimizer_restart_step} cannot "
                f"exceed checkpoint step {start_step}"
            )
    if optimizer_restart_step is not None and not provenance_known:
        raise ValueError(
            "optimizer restart step is present but provenance is marked unknown"
        )

    model_state = checkpoint["model_state_dict"]
    optimizer_state = checkpoint["optimizer_state_dict"]
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer_state_dict must be a mapping")

    model.load_state_dict(model_state)
    optimizer.load_state_dict(dict(optimizer_state))
    loaded_learning_rates = {
        float(group["lr"]) for group in optimizer.param_groups
    }
    if loaded_learning_rates != {config.learning_rate}:
        raise ValueError(
            "checkpoint optimizer learning rate does not match requested "
            f"{config.learning_rate}: got {sorted(loaded_learning_rates)}"
        )

    generator_state = checkpoint.get("training_generator_state")
    generator_restored = isinstance(generator_state, torch.Tensor)
    if generator_restored:
        training_generator.set_state(generator_state.cpu())
    else:
        advance_training_generator(
            training_generator,
            num_steps=start_step,
            batch_size=config.batch_size,
            num_train_tokens=len(data.train_data),
            block_size=config.block_size,
        )

    restore_rng_state(checkpoint.get("rng_state"), device)
    return ResumeState(
        mode="full",
        start_step=start_step,
        best_val_loss=best_val_loss,
        best_step=best_step,
        optimizer_restored=True,
        training_generator_restored=generator_restored,
        optimizer_restart_step=optimizer_restart_step,
        optimizer_provenance_known=provenance_known,
    )


def evaluate_on_fixed_batches(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
) -> LossEstimate:
    # The same sampled blocks are used at every measurement, matching Stage 10.
    evaluation_generator = torch.Generator().manual_seed(config.seed + 2)
    return estimate_loss(
        model,
        data,
        config,
        device,
        evaluation_generator,
    )


def train_until(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    *,
    start_step: int,
    on_evaluation: Callable[[EvaluationRecord], None],
) -> TrainingResult:
    if start_step < 0 or start_step > config.max_iters:
        raise ValueError(
            f"start_step must be between 0 and {config.max_iters}, "
            f"got {start_step}"
        )

    history: list[EvaluationRecord] = []

    def record(step: int) -> EvaluationRecord:
        evaluation = EvaluationRecord(
            step=step,
            losses=evaluate_on_fixed_batches(model, data, config, device),
        )
        history.append(evaluation)
        on_evaluation(evaluation)
        return evaluation

    initial_record = record(start_step)
    for completed_step in range(start_step + 1, config.max_iters + 1):
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

        if (
            completed_step % config.eval_interval == 0
            or completed_step == config.max_iters
        ):
            record(completed_step)

    return TrainingResult(
        initial=initial_record.losses,
        final=history[-1].losses,
        history=tuple(history),
    )


def print_evaluation(
    record: EvaluationRecord,
    *,
    best_val_loss: float,
    best_step: int,
    is_best: bool,
) -> None:
    marker = "  * best" if is_best else ""
    gap = record.losses.val - record.losses.train
    print(
        f"  step {record.step:5d} | "
        f"train {record.losses.train:.4f} | "
        f"val {record.losses.val:.4f} | "
        f"gap {gap:.4f} | "
        f"best {best_val_loss:.4f} @ {best_step}{marker}"
    )


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    benchmark: BenchmarkConfig | None,
    checkpoint_path: Path,
    *,
    resume_from: Path | None,
    allow_optimizer_restart: bool,
    legacy_step: int,
) -> ExperimentReport:
    benchmark_stats = None
    if benchmark is not None:
        benchmark_stats = run_independent_benchmark(
            data,
            config,
            n_head,
            n_layer,
            device,
            benchmark,
        )
        print_benchmark(benchmark_stats)

    seed_everything(config.seed)
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    training_generator = torch.Generator().manual_seed(config.seed + 1)

    if resume_from is None:
        resume = ResumeState(
            mode="fresh",
            start_step=0,
            best_val_loss=float("inf"),
            best_step=None,
            optimizer_restored=False,
            training_generator_restored=False,
            optimizer_restart_step=None,
            optimizer_provenance_known=True,
        )
    else:
        if not resume_from.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
        resume = load_resume_checkpoint(
            resume_from,
            model,
            optimizer,
            training_generator,
            data,
            config,
            device,
            allow_optimizer_restart=allow_optimizer_restart,
            legacy_step=legacy_step,
            n_head=n_head,
            n_layer=n_layer,
        )

        if (
            resume.mode == "legacy_weights"
            and resume_from.resolve() == checkpoint_path.resolve()
        ):
            raise ValueError(
                "a legacy source and Stage 11 output must use different paths"
            )

    if resume.start_step >= config.max_iters:
        raise ValueError(
            f"checkpoint step {resume.start_step} must be below target step "
            f"{config.max_iters}"
        )

    parameter_count = print_parameter_report(model)
    shapes_and_norms = print_shape_and_norm_report(
        model,
        data,
        config,
        device,
    )
    gradients = print_gradient_report(model, data, config, device)

    # Diagnostics above may consume global RNG. A full checkpoint carries the
    # exact state needed by future stochastic layers; restore it once more.
    if resume_from is not None and resume.mode == "full":
        checkpoint_for_rng = torch.load(
            resume_from,
            map_location="cpu",
            weights_only=True,
        )
        restore_rng_state(checkpoint_for_rng.get("rng_state"), device)
        del checkpoint_for_rng

    best = BestValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=fingerprint_data(data),
        optimizer_restart_step=resume.optimizer_restart_step,
        optimizer_provenance_known=resume.optimizer_provenance_known,
        best_val_loss=resume.best_val_loss,
        best_step=resume.best_step,
    )

    # A full resume source is already the best model at its saved step. When
    # the caller selects a different output path, materialize that state there
    # so generation still works even if validation never improves again.
    if (
        resume.mode == "full"
        and resume_from is not None
        and resume_from.resolve() != checkpoint_path.resolve()
    ):
        best.save(resume.start_step)

    def record_evaluation(record: EvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        print_evaluation(
            record,
            best_val_loss=best.best_val_loss,
            best_step=best.best_step,
            is_best=is_best,
        )

    print("\nTraining")
    print(f"  mode: {resume.description}")
    print(f"  start step: {resume.start_step}")
    print(f"  target step: {config.max_iters}")
    result = train_until(
        model,
        optimizer,
        data,
        config,
        device,
        training_generator,
        start_step=resume.start_step,
        on_evaluation=record_evaluation,
    )

    assert best.best_step is not None
    generalization_gap = result.final.val - result.final.train

    best_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    del best_checkpoint
    model.eval()
    seed_everything(config.seed + 2)
    start_id = data.vocabulary.stoi.get("\n", 0)
    context = torch.tensor([[start_id]], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=sample_length)
    sample = data.vocabulary.decode(generated[0].cpu().tolist())

    print("\nStage 11 experimental summary")
    print(
        f"  B / T / C / H / D / FF / L: "
        f"{config.batch_size} / {config.block_size} / {config.n_embd} / "
        f"{n_head} / {config.n_embd // n_head} / "
        f"{4 * config.n_embd} / {n_layer}"
    )
    print(f"  training mode:             {resume.description}")
    print(f"  optimizer restored:        {resume.optimizer_restored}")
    if not resume.optimizer_provenance_known:
        print("  optimizer lineage:         unknown before this checkpoint")
    elif resume.optimizer_restart_step is None:
        print("  optimizer lineage:         uninterrupted")
    else:
        print(
            f"  optimizer lineage:         restarted at step "
            f"{resume.optimizer_restart_step}"
        )
    print(f"  parameter count:           {parameter_count:,}")
    print(f"  final train loss:          {result.final.train:.4f}")
    print(f"  final validation loss:     {result.final.val:.4f}")
    print(f"  generalization gap:        {generalization_gap:.4f}")
    print(
        f"  best validation loss:      {best.best_val_loss:.4f} "
        f"(step {best.best_step})"
    )
    if benchmark_stats is None:
        print("  benchmark:                 disabled")
    else:
        print(
            f"  iterations/sec:            "
            f"{benchmark_stats.iterations_per_sec:.3f}"
        )
        print(
            f"  tokens/sec:                "
            f"{benchmark_stats.tokens_per_sec:.3f}"
        )
        print(
            f"  peak allocated memory:     "
            f"{benchmark_stats.peak_allocated_mb:.3f} MB"
        )
        print(
            f"  peak reserved memory:      "
            f"{benchmark_stats.peak_reserved_mb:.3f} MB"
        )
    print(f"  best checkpoint:           {checkpoint_path}")
    print("\nGenerated text")
    print(sample)

    return ExperimentReport(
        parameter_count=parameter_count,
        shapes_and_norms=shapes_and_norms,
        gradients=gradients,
        training=result,
        benchmark=benchmark_stats,
        resume=resume,
        generalization_gap=generalization_gap,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        checkpoint_path=checkpoint_path,
        sample=sample,
    )


def main() -> None:
    args = parse_args()

    if args.sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    if args.benchmark_warmup < 0:
        raise ValueError("benchmark-warmup must be non-negative")
    if args.benchmark_steps < 0:
        raise ValueError("benchmark-steps must be non-negative")
    if args.legacy_step < 0:
        raise ValueError("legacy-step must be non-negative")

    config = TrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    data = CharacterData.from_file(args.data)

    if args.n_head <= 0 or config.n_embd % args.n_head != 0:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({args.n_head})"
        )
    if args.n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {args.n_layer}")

    benchmark = None
    if args.benchmark_steps > 0:
        benchmark = BenchmarkConfig(
            num_warmup=args.benchmark_warmup,
            num_steps=args.benchmark_steps,
        )

    print("Stage 11: train the unchanged Stage 10 model to convergence")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"B={config.batch_size}, T={config.block_size}, "
        f"C={config.n_embd}, H={args.n_head}, "
        f"D={config.n_embd // args.n_head}, FF={4 * config.n_embd}, "
        f"L={args.n_layer}, lr={config.learning_rate:g}"
    )
    if args.resume_from is None:
        print(
            f"Checkpoint input: none (clean {config.max_iters:,}-step run)"
        )
    else:
        print(f"Checkpoint input: {args.resume_from}")

    run_experiment(
        data,
        config,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        benchmark,
        args.checkpoint_path,
        resume_from=args.resume_from,
        allow_optimizer_restart=args.allow_optimizer_restart,
        legacy_step=args.legacy_step,
    )


if __name__ == "__main__":
    main()
