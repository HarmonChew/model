from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn

from config import TrainingConfig
from data_utils import CharacterData, DEFAULT_DATA_PATH
from train import resolve_device
from training import EvaluationRecord, TrainingResult, seed_everything


# Stage 14 changes the model for the first time since Stage 10, so it owns a
# dropout-aware architecture while reusing the exact-resume primitives proven
# by Stages 11-13. Keeping the historical Stage 8 model untouched preserves
# every earlier checkpoint as the lesson that was originally run.
_STAGE_8_PATH = Path(__file__).with_name("08_transformer_block.py")
_STAGE_8_MODULE_NAME = "stage_8_transformer_block_for_stage_14"
_STAGE_8_SPEC = importlib.util.spec_from_file_location(
    _STAGE_8_MODULE_NAME,
    _STAGE_8_PATH,
)
assert _STAGE_8_SPEC is not None and _STAGE_8_SPEC.loader is not None
_STAGE_8 = importlib.util.module_from_spec(_STAGE_8_SPEC)
sys.modules[_STAGE_8_MODULE_NAME] = _STAGE_8
_STAGE_8_SPEC.loader.exec_module(_STAGE_8)

_STAGE_13_PATH = Path(__file__).with_name("13_second_learning_rate_drop.py")
_STAGE_13_MODULE_NAME = "stage_13_second_learning_rate_drop_for_stage_14"
_STAGE_13_SPEC = importlib.util.spec_from_file_location(
    _STAGE_13_MODULE_NAME,
    _STAGE_13_PATH,
)
assert _STAGE_13_SPEC is not None and _STAGE_13_SPEC.loader is not None
_STAGE_13 = importlib.util.module_from_spec(_STAGE_13_SPEC)
sys.modules[_STAGE_13_MODULE_NAME] = _STAGE_13
_STAGE_13_SPEC.loader.exec_module(_STAGE_13)

_STAGE_12 = _STAGE_13._STAGE_12
_STAGE_11 = _STAGE_12._STAGE_11

ResumeState = _STAGE_13.ResumeState
checkpoint_sha256 = _STAGE_13.checkpoint_sha256
clear_accelerator_cache = _STAGE_13.clear_accelerator_cache
evaluate_on_fixed_batches = _STAGE_13.evaluate_on_fixed_batches
fingerprint_data = _STAGE_13.fingerprint_data
generate_from_final_model = _STAGE_13.generate_from_final_model
train_until = _STAGE_13.train_until

CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_SOURCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_13_lr_drop_best_checkpoint.pt"
)
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_14_control_best_checkpoint.pt"
)
DEFAULT_DROPOUT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_14_dropout_best_checkpoint.pt"
)
DEFAULT_STAGE_12_BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_12_lr_drop_best_checkpoint.pt"
)
DEFAULT_STAGE_13_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_13_control_best_checkpoint.pt"
)
DEFAULT_SOURCE_STEP = 17_000
DEFAULT_TARGET_STEP = 22_000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_CONTROL_DROPOUT = 0.0
DEFAULT_RESIDUAL_DROPOUT = 0.1
DEFAULT_PRECISE_EVAL_ITERS = 500
DROPOUT_PLACEMENT = "attention_projection_and_ffn_output"
STAGE_13_PREDECESSOR_STEP = 13_000
STAGE_13_PREDECESSOR_LEARNING_RATE = 3e-4


def _validate_dropout(value: float, *, name: str = "dropout") -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1), got {value!r}")
    return probability


class MultiHeadAttention(_STAGE_8.MultiHeadAttention):
    """Projected causal attention with dropout on its residual update."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
        )
        self.dropout = nn.Dropout(_validate_dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        result = super().forward(x, return_weights=return_weights)
        if isinstance(result, tuple):
            output, weights = result
            return self.dropout(output), weights
        return self.dropout(result)


class FeedForward(_STAGE_8.FeedForward):
    """Position-wise FFN with dropout after its final C projection."""

    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__(n_embd=n_embd)
        # Appending preserves the checkpoint names of the existing projections:
        # net.0 and net.2. Dropout has no parameters or persistent buffers.
        self.net.append(nn.Dropout(_validate_dropout(dropout)))


class Block(nn.Module):
    """One pre-norm block with dropout on both residual branch outputs."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        probability = _validate_dropout(dropout)
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=probability,
        )
        self.ffwd = FeedForward(n_embd=n_embd, dropout=probability)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_update = self.sa(self.ln1(x))
        assert isinstance(attention_update, torch.Tensor)
        x = x + attention_update
        return x + self.ffwd(self.ln2(x))


class GPTLanguageModel(_STAGE_8.GPTLanguageModel):
    """The Stage 13 architecture plus configurable residual-path dropout."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.0,
    ) -> None:
        # This mirrors Stage 8's construction order exactly. nn.Dropout does
        # not consume RNG, so a seeded p=0 model has byte-for-byte compatible
        # model tensors and parameter traversal order with Stage 13.
        nn.Module.__init__(self)

        positive_values = {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
            )

        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = _validate_dropout(dropout)

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=self.dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


@dataclass(frozen=True, slots=True)
class BranchSpec:
    name: str
    dropout: float
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("branch name must not be empty")
        object.__setattr__(
            self,
            "dropout",
            _validate_dropout(self.dropout, name="branch dropout"),
        )


@dataclass(slots=True)
class LoadedBranch:
    model: GPTLanguageModel
    optimizer: torch.optim.Optimizer
    training_generator: torch.Generator
    config: TrainingConfig
    resume: ResumeState
    dropout: float


@dataclass(slots=True)
class BranchValidationCheckpoint(_STAGE_11.BestValidationCheckpoint):
    """Save a complete Stage 14 checkpoint with dropout provenance."""

    branch_name: str = ""
    source_checkpoint_sha256: str = ""
    source_step: int = 0
    learning_rate: float = 0.0
    source_dropout: float = 0.0
    branch_dropout: float = 0.0

    def _payload(self, step: int) -> dict[str, object]:
        payload = super(BranchValidationCheckpoint, self)._payload(step)
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
        training_config["residual_dropout"] = self.branch_dropout

        payload["experiment"] = {
            "stage": 14,
            "branch": self.branch_name,
            "source_stage": 13,
            "source_branch": "lr_drop",
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_step": self.source_step,
            "source_learning_rate": self.learning_rate,
            "branch_learning_rate": self.learning_rate,
            "learning_rate_changed": False,
            "source_residual_dropout": self.source_dropout,
            "branch_residual_dropout": self.branch_dropout,
            "dropout_changed": self.branch_dropout != self.source_dropout,
            "dropout_placement": DROPOUT_PLACEMENT,
        }
        return payload


@dataclass(frozen=True, slots=True)
class BranchReport:
    name: str
    dropout: float
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


@dataclass(frozen=True, slots=True)
class PreciseCheckpointSpec:
    name: str
    checkpoint_path: Path
    expected_step: int | None = None
    expected_stage: int | None = None
    expected_branch: str | None = None
    expected_learning_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("precise checkpoint name must not be empty")
        if self.expected_step is not None and self.expected_step < 0:
            raise ValueError("expected precise checkpoint step must be non-negative")
        if self.expected_learning_rate is not None:
            expected_rate = float(self.expected_learning_rate)
            if not math.isfinite(expected_rate) or expected_rate <= 0:
                raise ValueError(
                    "expected precise checkpoint learning rate must be "
                    f"finite and positive, got {self.expected_learning_rate!r}"
                )


@dataclass(frozen=True, slots=True)
class PreciseValidationResult:
    name: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_step: int
    fixed_panel_loss: float
    mean_loss: float
    standard_error: float
    batch_losses: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class PreciseDelta:
    candidate: str
    baseline: str
    mean_delta: float
    standard_error: float
    confidence_low: float
    confidence_high: float


@dataclass(frozen=True, slots=True)
class PreciseValidationReport:
    eval_iters: int
    seed: int
    results: tuple[PreciseValidationResult, ...]
    adjacent_deltas: tuple[PreciseDelta, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fork the exact Stage 13 step-17,000 winner and compare 5,000 "
            "more steps at residual dropout p=0.0 versus p=0.1."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
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
    parser.add_argument("--source-step", type=int, default=DEFAULT_SOURCE_STEP)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_TARGET_STEP)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument(
        "--precise-eval-iters",
        type=int,
        default=DEFAULT_PRECISE_EVAL_ITERS,
        help="Fresh paired validation batches per preflight checkpoint; 0 skips.",
    )
    parser.add_argument(
        "--precise-eval-seed",
        type=int,
        default=None,
        help="Defaults to seed + 3, distinct from the fixed selection panel.",
    )
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
        "--dropout-checkpoint-path",
        type=Path,
        default=DEFAULT_DROPOUT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--stage-12-best-checkpoint",
        type=Path,
        default=DEFAULT_STAGE_12_BEST_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--stage-13-control-checkpoint",
        type=Path,
        default=DEFAULT_STAGE_13_CONTROL_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def validate_distinct_paths(
    source_checkpoint: Path,
    control_checkpoint: Path,
    dropout_checkpoint: Path,
    *,
    additional_input_paths: Sequence[Path] = (),
) -> None:
    labeled_paths = {
        "source checkpoint": source_checkpoint.resolve(),
        "control checkpoint": control_checkpoint.resolve(),
        "control temporary checkpoint": control_checkpoint.with_name(
            f".{control_checkpoint.name}.tmp"
        ).resolve(),
        "dropout checkpoint": dropout_checkpoint.resolve(),
        "dropout temporary checkpoint": dropout_checkpoint.with_name(
            f".{dropout_checkpoint.name}.tmp"
        ).resolve(),
    }
    for index, path in enumerate(additional_input_paths, start=1):
        labeled_paths[f"additional input checkpoint {index}"] = path.resolve()
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
    expected_learning_rate: float,
) -> Mapping[str, object]:
    """Require the exact clean Stage 13 second-decay winner."""

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
        raise ValueError("Stage 14 requires a Stage 13 best checkpoint")

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

    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    expected_provenance = {
        "stage": 13,
        "branch": "lr_drop",
        "source_stage": 12,
        "source_branch": "lr_drop",
        "branch_learning_rate": expected_learning_rate,
        "learning_rate_changed": True,
    }
    for key, expected in expected_provenance.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"Stage 14 source experiment.{key} must be {expected!r}, "
                f"got {experiment.get(key)!r}"
            )

    predecessor_step = experiment.get("source_step")
    predecessor_learning_rate = experiment.get("source_learning_rate")
    if (
        expected_step == DEFAULT_SOURCE_STEP
        and expected_learning_rate == DEFAULT_LEARNING_RATE
    ):
        if predecessor_step != STAGE_13_PREDECESSOR_STEP:
            raise ValueError(
                "Stage 13 source_step must be the exact Stage 12 winner at "
                f"{STAGE_13_PREDECESSOR_STEP}, got {predecessor_step!r}"
            )
        if predecessor_learning_rate != STAGE_13_PREDECESSOR_LEARNING_RATE:
            raise ValueError(
                "Stage 13 source learning rate must be the exact Stage 12 "
                f"rate {STAGE_13_PREDECESSOR_LEARNING_RATE:g}, got "
                f"{predecessor_learning_rate!r}"
            )
    else:
        if (
            isinstance(predecessor_step, bool)
            or not isinstance(predecessor_step, int)
            or predecessor_step < 0
            or predecessor_step >= expected_step
        ):
            raise ValueError("Stage 13 source_step must be a non-negative earlier step")
        if (
            isinstance(predecessor_learning_rate, bool)
            or not isinstance(predecessor_learning_rate, (int, float))
            or not math.isfinite(float(predecessor_learning_rate))
            or float(predecessor_learning_rate) <= expected_learning_rate
        ):
            raise ValueError(
                "Stage 13 source learning rate must exceed its branch rate"
            )

    architecture = _require_mapping(
        checkpoint.get("architecture"),
        name="architecture",
    )
    recorded_architecture_dropout = architecture.get("residual_dropout")
    if recorded_architecture_dropout not in (None, 0.0):
        raise ValueError("Stage 13 source must have residual dropout 0.0")
    if "dropout_placement" in architecture:
        raise ValueError(
            "Stage 13 source must predate explicit dropout placement metadata"
        )
    recorded_training_dropout = training_config.get("residual_dropout")
    if recorded_training_dropout not in (None, 0.0):
        raise ValueError("Stage 13 training dropout must be 0.0")

    return checkpoint


def validate_device_rng_state(
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> None:
    if device.type == "mps":
        raise ValueError(
            "Stage 14 exact dropout continuation does not support MPS because "
            "the inherited checkpoints do not capture MPS RNG state"
        )
    if device.type not in ("xpu", "cuda"):
        return
    rng_state = _require_mapping(checkpoint.get("rng_state"), name="rng_state")
    accelerator_states = rng_state.get(device.type)
    if (
        not isinstance(accelerator_states, list)
        or not accelerator_states
        or not all(isinstance(state, torch.Tensor) for state in accelerator_states)
    ):
        raise ValueError(
            f"Stage 14 on {device.type} requires saved {device.type} RNG state"
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
    # Both branches independently reconstruct and restore the entire source.
    # Dropout consumes global/device RNG only after training begins; the batch
    # stream remains paired because it uses its own restored Generator.
    seed_everything(source_config.seed)
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=source_config.block_size,
        n_embd=source_config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=spec.dropout,
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
    _STAGE_12.validate_clean_resume(
        resume,
        expected_step=expected_source_step,
    )

    return LoadedBranch(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        config=source_config,
        resume=resume,
        dropout=spec.dropout,
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
        learning_rate=source_config.learning_rate,
        source_dropout=0.0,
        branch_dropout=spec.dropout,
    )

    # Materialize a resumable output even if the branch never beats the source.
    best.save(branch.resume.start_step)
    record_evaluation = make_evaluation_callback(best)

    print(
        f"\n{spec.name} branch "
        f"(dropout={spec.dropout:g}, lr={source_config.learning_rate:g})"
    )
    print(f"  source step: {branch.resume.start_step}")
    print(f"  target step: {branch.config.max_iters}")
    branch.model.train()
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
        dropout=spec.dropout,
        learning_rate=source_config.learning_rate,
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
    dropout_spec: BranchSpec,
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
    if control_spec.dropout != 0.0:
        raise ValueError("Stage 14 control dropout must be exactly 0.0")
    if dropout_spec.dropout <= control_spec.dropout:
        raise ValueError("dropout branch probability must exceed the control")
    if control_spec.name == dropout_spec.name:
        raise ValueError("branch names must be different")

    validate_distinct_paths(
        source_checkpoint,
        control_spec.checkpoint_path,
        dropout_spec.checkpoint_path,
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
        "and validation blocks; dropout is disabled while measuring."
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
    dropout, dropout_parameter_count = run_branch(
        data,
        source_config,
        n_head,
        n_layer,
        device,
        sample_length,
        source_checkpoint,
        source_hash,
        dropout_spec,
        expected_source_step=expected_source_step,
    )

    if checkpoint_sha256(source_checkpoint) != source_hash:
        raise RuntimeError("source checkpoint changed during the experiment")
    if control_parameter_count != dropout_parameter_count:
        raise RuntimeError("branch parameter counts do not match")
    if control.training.initial != dropout.training.initial:
        raise RuntimeError("branches did not start from the same losses")

    control_steps = tuple(record.step for record in control.training.history)
    dropout_steps = tuple(record.step for record in dropout.training.history)
    if control_steps != dropout_steps:
        raise RuntimeError("branches were not evaluated at matching steps")

    source_val_loss = float(source_payload["best_val_loss"])
    if not math.isclose(
        control.training.initial.val,
        source_val_loss,
        rel_tol=1e-6,
        abs_tol=1e-5,
    ):
        raise RuntimeError(
            "Stage 14 model is not inference-equivalent to the saved source "
            f"panel: evaluated {control.training.initial.val}, checkpoint "
            f"records {source_val_loss}"
        )
    final_val_delta = dropout.training.final.val - control.training.final.val
    best_val_delta = dropout.best_val_loss - control.best_val_loss

    print("\nStage 14 experimental summary")
    print(
        f"  B / T / C / H / D / FF / L: "
        f"{source_config.batch_size} / {source_config.block_size} / "
        f"{source_config.n_embd} / {n_head} / "
        f"{source_config.n_embd // n_head} / "
        f"{4 * source_config.n_embd} / {n_layer}"
    )
    print(f"  parameter count:            {control_parameter_count:,}")
    print(
        f"  common source:              step {expected_source_step:,}, "
        f"val {source_val_loss:.4f}, lr {source_config.learning_rate:g}"
    )
    print(
        f"  control final train / val:  "
        f"{control.training.final.train:.4f} / "
        f"{control.training.final.val:.4f} (p={control.dropout:g})"
    )
    print(
        f"  dropout final train / val:  "
        f"{dropout.training.final.train:.4f} / "
        f"{dropout.training.final.val:.4f} (p={dropout.dropout:g})"
    )
    print(
        f"  final val delta (p-control): {final_val_delta:+.4f} "
        "(negative favors dropout)"
    )
    print(
        f"  control best val:           {control.best_val_loss:.4f} "
        f"@ {control.best_step}"
    )
    print(
        f"  dropout best val:           {dropout.best_val_loss:.4f} "
        f"@ {dropout.best_step}"
    )
    print(
        f"  best val delta (p-control):  {best_val_delta:+.4f} "
        "(negative favors dropout)"
    )
    print(f"  control checkpoint:         {control.checkpoint_path}")
    print(f"  dropout checkpoint:         {dropout.checkpoint_path}")
    print("\nControl final-step generated text")
    print(control.sample)
    print("\nDropout final-step generated text")
    print(dropout.sample)

    return ExperimentReport(
        parameter_count=control_parameter_count,
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=source_hash,
        source_step=expected_source_step,
        source_val_loss=source_val_loss,
        control=control,
        dropout=dropout,
        final_val_delta=final_val_delta,
        best_val_delta=best_val_delta,
    )


@torch.no_grad()
def estimate_validation_precise(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    eval_iters: int = DEFAULT_PRECISE_EVAL_ITERS,
    seed: int,
) -> tuple[float, ...]:
    """Measure validation loss on a fresh, reproducible batch panel."""

    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    try:
        for _ in range(eval_iters):
            inputs, targets = data.get_batch(
                "val",
                batch_size=config.batch_size,
                block_size=config.block_size,
                device=device,
                generator=generator,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            losses.append(loss.item())
    finally:
        model.train(was_training)
    return tuple(losses)


def _mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one loss value is required")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    mean = tensor.mean().item()
    if len(values) == 1:
        return mean, 0.0
    standard_error = (tensor.std(unbiased=True) / math.sqrt(len(values))).item()
    return mean, standard_error


def _paired_delta(
    candidate: PreciseValidationResult,
    baseline: PreciseValidationResult,
) -> PreciseDelta:
    if len(candidate.batch_losses) != len(baseline.batch_losses):
        raise ValueError("paired validation results must have equal lengths")
    differences = tuple(
        candidate_loss - baseline_loss
        for candidate_loss, baseline_loss in zip(
            candidate.batch_losses,
            baseline.batch_losses,
            strict=True,
        )
    )
    mean, standard_error = _mean_and_standard_error(differences)
    margin = 1.96 * standard_error
    return PreciseDelta(
        candidate=candidate.name,
        baseline=baseline.name,
        mean_delta=mean,
        standard_error=standard_error,
        confidence_low=mean - margin,
        confidence_high=mean + margin,
    )


def _load_precise_checkpoint_model(
    spec: PreciseCheckpointSpec,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
) -> tuple[GPTLanguageModel, int, float]:
    if not spec.checkpoint_path.is_file():
        raise FileNotFoundError(
            f"precise validation checkpoint not found: {spec.checkpoint_path}"
        )
    checkpoint = torch.load(
        spec.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_mapping(checkpoint, name=str(spec.checkpoint_path))
    saved_training_config = _require_mapping(
        checkpoint.get("training_config"),
        name="training_config",
    )
    saved_learning_rate = saved_training_config.get("learning_rate")
    if (
        isinstance(saved_learning_rate, bool)
        or not isinstance(saved_learning_rate, (int, float))
        or not math.isfinite(float(saved_learning_rate))
        or float(saved_learning_rate) <= 0
    ):
        raise ValueError(
            "precise checkpoint learning rate must be finite and positive, "
            f"got {saved_learning_rate!r}"
        )
    if (
        spec.expected_learning_rate is not None
        and float(saved_learning_rate) != spec.expected_learning_rate
    ):
        raise ValueError(
            f"{spec.name} learning rate must be "
            f"{spec.expected_learning_rate:g}, got {saved_learning_rate!r}"
        )
    # Learning rate affects optimizer continuation, not a weights-only forward
    # pass. Validate every other saved training and architecture field while
    # accepting the rate that belongs to each historical checkpoint.
    checkpoint_config = replace(
        config,
        learning_rate=float(saved_learning_rate),
    )
    _STAGE_11.validate_checkpoint_metadata(
        checkpoint,
        data=data,
        config=checkpoint_config,
        n_head=n_head,
        n_layer=n_layer,
    )

    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(
            f"checkpoint step must be a non-negative integer, got {step!r}"
        )
    if spec.expected_step is not None and step != spec.expected_step:
        raise ValueError(
            f"{spec.name} checkpoint step must be {spec.expected_step}, got {step}"
        )
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError(f"{spec.name} must be a best checkpoint")
    if checkpoint.get("best_step") != step:
        raise ValueError(f"{spec.name} best_step must equal its saved step {step}")
    experiment = _require_mapping(
        checkpoint.get("experiment"),
        name="experiment",
    )
    if (
        spec.expected_stage is not None
        and experiment.get("stage") != spec.expected_stage
    ):
        raise ValueError(
            f"{spec.name} experiment.stage must be {spec.expected_stage}, "
            f"got {experiment.get('stage')!r}"
        )
    if (
        spec.expected_branch is not None
        and experiment.get("branch") != spec.expected_branch
    ):
        raise ValueError(
            f"{spec.name} experiment.branch must be "
            f"{spec.expected_branch!r}, got {experiment.get('branch')!r}"
        )
    fixed_panel_loss = float(checkpoint.get("best_val_loss", float("nan")))
    if not math.isfinite(fixed_panel_loss):
        raise ValueError("checkpoint best_val_loss must be finite")
    model_state = _require_mapping(
        checkpoint.get("model_state_dict"),
        name="model_state_dict",
    )

    seed_everything(config.seed)
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    return model, step, fixed_panel_loss


def run_precise_validation(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    checkpoint_specs: Sequence[PreciseCheckpointSpec],
    *,
    eval_iters: int = DEFAULT_PRECISE_EVAL_ITERS,
    seed: int,
) -> PreciseValidationReport:
    if len(checkpoint_specs) < 2:
        raise ValueError("precise comparison requires at least two checkpoints")

    results: list[PreciseValidationResult] = []
    for spec in checkpoint_specs:
        checkpoint_hash = checkpoint_sha256(spec.checkpoint_path)
        model, step, fixed_panel_loss = _load_precise_checkpoint_model(
            spec,
            data,
            config,
            n_head,
            n_layer,
            device,
        )
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
        mean, standard_error = _mean_and_standard_error(batch_losses)
        results.append(
            PreciseValidationResult(
                name=spec.name,
                checkpoint_path=spec.checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                checkpoint_step=step,
                fixed_panel_loss=fixed_panel_loss,
                mean_loss=mean,
                standard_error=standard_error,
                batch_losses=batch_losses,
            )
        )
        del model
        clear_accelerator_cache(device)

    adjacent_deltas = tuple(
        _paired_delta(candidate, baseline)
        for baseline, candidate in zip(results, results[1:], strict=False)
    )
    return PreciseValidationReport(
        eval_iters=eval_iters,
        seed=seed,
        results=tuple(results),
        adjacent_deltas=adjacent_deltas,
    )


def print_precise_validation(report: PreciseValidationReport) -> None:
    print("\nFresh paired validation preflight")
    print(
        f"  {report.eval_iters} validation batches per checkpoint, "
        f"shared panel seed={report.seed}"
    )
    print("  This is a new sample from the same validation split, not a new split.")
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
            f"{delta.confidence_high:+.6f}])"
        )


def main() -> None:
    args = parse_args()
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
    if args.n_head <= 0 or config.n_embd % args.n_head != 0:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({args.n_head})"
        )
    if args.n_layer <= 0:
        raise ValueError(f"n_layer must be positive, got {args.n_layer}")
    if args.precise_eval_iters < 0:
        raise ValueError(
            f"precise-eval-iters must be non-negative, got {args.precise_eval_iters}"
        )

    device = resolve_device(args.device)
    if device.type == "mps":
        raise ValueError(
            "Stage 14 exact dropout continuation does not support MPS because "
            "the inherited checkpoints do not capture MPS RNG state"
        )
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

    print("Stage 14: controlled residual-path dropout")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Characters: {data.num_characters:,}")
    print(f"Vocabulary size: {data.vocabulary.size}")
    print(f"Uniform-loss baseline: {math.log(data.vocabulary.size):.4f}")
    print(
        f"Fork step={args.source_step:,}, target step={config.max_iters:,}, "
        f"lr={config.learning_rate:g}, control p={control_spec.dropout:g}, "
        f"dropout p={dropout_spec.dropout:g}"
    )
    print(f"Checkpoint input: {args.source_checkpoint}")

    if args.precise_eval_iters > 0:
        validate_distinct_paths(
            args.source_checkpoint,
            args.control_checkpoint_path,
            args.dropout_checkpoint_path,
            additional_input_paths=(
                args.stage_12_best_checkpoint,
                args.stage_13_control_checkpoint,
            ),
        )
        precise_seed = (
            args.precise_eval_seed
            if args.precise_eval_seed is not None
            else config.seed + 3
        )
        precise_report = run_precise_validation(
            data,
            config,
            args.n_head,
            args.n_layer,
            device,
            (
                PreciseCheckpointSpec(
                    "stage_12_best",
                    args.stage_12_best_checkpoint,
                    expected_step=13_000,
                    expected_stage=12,
                    expected_branch="lr_drop",
                    expected_learning_rate=3e-4,
                ),
                PreciseCheckpointSpec(
                    "stage_13_control",
                    args.stage_13_control_checkpoint,
                    expected_step=17_000,
                    expected_stage=13,
                    expected_branch="control",
                    expected_learning_rate=3e-4,
                ),
                PreciseCheckpointSpec(
                    "stage_13_lr_drop",
                    args.source_checkpoint,
                    expected_step=17_000,
                    expected_stage=13,
                    expected_branch="lr_drop",
                    expected_learning_rate=1e-4,
                ),
            ),
            eval_iters=args.precise_eval_iters,
            seed=precise_seed,
        )
        print_precise_validation(precise_report)

    run_experiment(
        data,
        config,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        args.source_checkpoint,
        control_spec,
        dropout_spec,
        expected_source_step=args.source_step,
    )


if __name__ == "__main__":
    main()
