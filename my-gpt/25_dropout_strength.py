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
from training import seed_everything


# Stage 25 changes only residual-path dropout for the Stage 24 L=6 model.
# Reuse the tested training, evaluation, checkpoint, and inference machinery,
# while giving this stage its own provenance boundary and read-only control.
_STAGE_24_PATH = Path(__file__).with_name("24_scale_depth.py")
_STAGE_24_MODULE_NAME = "stage_24_scale_depth_for_stage_25"
_STAGE_24_SPEC = importlib.util.spec_from_file_location(
    _STAGE_24_MODULE_NAME,
    _STAGE_24_PATH,
)
assert _STAGE_24_SPEC is not None and _STAGE_24_SPEC.loader is not None
_STAGE_24 = importlib.util.module_from_spec(_STAGE_24_SPEC)
sys.modules[_STAGE_24_MODULE_NAME] = _STAGE_24
_STAGE_24_SPEC.loader.exec_module(_STAGE_24)

GPTLanguageModel = _STAGE_24.GPTLanguageModel
LearningRateSchedule = _STAGE_24.LearningRateSchedule
PositionLosses = _STAGE_24.PositionLosses
PositionLossEstimate = _STAGE_24.PositionLossEstimate
EvaluationRecord = _STAGE_24.EvaluationRecord
TrainingResult = _STAGE_24.TrainingResult
PreciseBranchResult = _STAGE_24.PreciseBranchResult
PreciseValidationReport = _STAGE_24.PreciseValidationReport
DropoutSpec = _STAGE_24._STAGE_23.DropoutSpec
ReferenceCheckpoint = _STAGE_24.ReferenceCheckpoint
ControlReport = _STAGE_24._STAGE_23.ControlReport
TreatmentReport = _STAGE_24._STAGE_23.TreatmentReport
BaseExperimentReport = _STAGE_24._STAGE_23.BaseExperimentReport
DepthSpec = _STAGE_24.DepthSpec

checkpoint_sha256 = _STAGE_24.checkpoint_sha256
clear_accelerator_cache = _STAGE_24.clear_accelerator_cache
count_parameters = _STAGE_24.count_parameters
evaluate_on_fixed_batches = _STAGE_24.evaluate_on_fixed_batches
expected_parameter_count = _STAGE_24.expected_parameter_count
expected_training_generator_state = _STAGE_24.expected_training_generator_state
fingerprint_data = _STAGE_24.fingerprint_data
generate_from_final_model = _STAGE_24.generate_from_final_model
inspect_attention_shape = _STAGE_24.inspect_attention_shape
make_position_bands = _STAGE_24.make_position_bands
tensor_mapping_sha256 = _STAGE_24.tensor_mapping_sha256
tensor_sha256 = _STAGE_24.tensor_sha256
train_with_schedule = _STAGE_24.train_with_schedule
_clone_state = _STAGE_24._clone_state
_load_and_measure_fixed_control = _STAGE_24._load_and_measure_fixed_control
_measure_precise_checkpoint = _STAGE_24._measure_precise_checkpoint
_paired_delta = _STAGE_24._paired_delta
_replay_treatment_fixed_panel = (
    _STAGE_24._STAGE_23._STAGE_22._STAGE_21.replay_treatment_fixed_panel
)
_require_mapping = _STAGE_24._require_mapping
_validate_checkpoint_base = (
    _STAGE_24._STAGE_23._STAGE_22._STAGE_21.validate_stage_21_checkpoint
)
_validate_non_negative_integer = _STAGE_24._validate_non_negative_integer


CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CONTROL_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_24_depth_6_best_checkpoint.pt"
)
DEFAULT_TREATMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_25_depth_6_dropout_015_best_checkpoint.pt"
)
DEFAULT_CONTROL_CHECKPOINT_SHA256 = (
    "b44f591123bb3679557b07135dd1c44ca0392ecfa4f5b96f1ffc24f4c65a2727"
)
DEFAULT_CONTROL_BEST_STEP = 15_500
DEFAULT_INITIAL_STATE_SHA256 = _STAGE_24.DEFAULT_TREATMENT_INITIAL_STATE_SHA256

DEFAULT_BATCH_SIZE = _STAGE_24.DEFAULT_BATCH_SIZE
DEFAULT_BLOCK_SIZE = _STAGE_24.DEFAULT_BLOCK_SIZE
DEFAULT_N_EMBD = _STAGE_24.DEFAULT_N_EMBD
DEFAULT_N_HEAD = _STAGE_24.DEFAULT_N_HEAD
DEFAULT_N_LAYER = _STAGE_24.DEFAULT_TREATMENT_N_LAYER
DEFAULT_MAX_ITERS = _STAGE_24.DEFAULT_MAX_ITERS
DEFAULT_INITIAL_LEARNING_RATE = _STAGE_24.DEFAULT_INITIAL_LEARNING_RATE
DEFAULT_FIRST_DECAY_STEP = _STAGE_24.DEFAULT_FIRST_DECAY_STEP
DEFAULT_MIDDLE_LEARNING_RATE = _STAGE_24.DEFAULT_MIDDLE_LEARNING_RATE
DEFAULT_SECOND_DECAY_STEP = _STAGE_24.DEFAULT_SECOND_DECAY_STEP
DEFAULT_FINAL_LEARNING_RATE = _STAGE_24.DEFAULT_FINAL_LEARNING_RATE
DEFAULT_PRECISE_EVAL_ITERS = _STAGE_24.DEFAULT_PRECISE_EVAL_ITERS
DEFAULT_CONTROL_DROPOUT = 0.10
DEFAULT_TREATMENT_DROPOUT = 0.15
DROPOUT_PLACEMENT = _STAGE_24.DROPOUT_PLACEMENT
INITIALIZATION_KIND = "recreated_stage_24_depth_6_shared_initialization"


@dataclass(frozen=True, slots=True)
class ExperimentReport(BaseExperimentReport):
    treatment_checkpoint_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 25: train only the L=6 residual-dropout p=0.15 branch "
            "from the exact Stage 24 L=6 initialization and batch stream, "
            "reusing the saved Stage 24 p=0.10 best checkpoint as a "
            "read-only control."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument(
        "--n-layer",
        type=int,
        default=DEFAULT_N_LAYER,
        help=(
            "Fixed depth for both branches. The canonical Stage 24 control "
            "requires 6; alternate values require a matching custom control."
        ),
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
        default=DEFAULT_TREATMENT_DROPOUT,
        help="Stage 25 treatment probability; the protocol requires exactly 0.15.",
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
        help="Defaults to seed + 1, matching Stages 20 and 21.",
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
        help="Defaults to seed + 7, matching the Stage 24 fresh panel.",
    )
    parser.add_argument(
        "--control-checkpoint-path",
        type=Path,
        default=DEFAULT_CONTROL_CHECKPOINT_PATH,
        help="Read-only Stage 24 C=128, L=6, p=0.10 best checkpoint.",
    )
    parser.add_argument(
        "--control-checkpoint-sha256",
        type=str,
        default=None,
        help=(
            "Required with a custom control path; the canonical control is "
            "already pinned."
        ),
    )
    parser.add_argument(
        "--control-checkpoint-step",
        type=int,
        default=None,
        help=(
            "Required with a custom control path; the canonical control is "
            "already pinned."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_TREATMENT_CHECKPOINT_PATH,
        help="Output path for the Stage 25 L=6, p=0.15 best checkpoint.",
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
                "a custom Stage 24 control requires both "
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
            "explicit control pin does not match the canonical Stage 24 pin"
        )
    return normalized_sha256, explicit_step


def validate_protocol(
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    treatment: DropoutSpec,
    control_checkpoint_path: Path,
    schedule: LearningRateSchedule,
) -> tuple[tuple[int, int], ...]:
    if treatment.dropout != DEFAULT_TREATMENT_DROPOUT:
        raise ValueError("Stage 25 treatment dropout must be exactly 0.15")
    if config.n_embd <= 0 or config.n_embd % 2 != 0:
        raise ValueError("Stage 25 n_embd must be a positive even integer")
    if isinstance(n_head, bool) or not isinstance(n_head, int) or n_head <= 0:
        raise ValueError(f"n_head must be a positive integer, got {n_head!r}")
    if config.n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer < 3:
        raise ValueError(
            "n_layer must be an integer of at least 3 so the Stage 24 "
            f"two-layer depth comparison can be reconstructed, got {n_layer!r}"
        )
    if (
        control_checkpoint_path.resolve() == DEFAULT_CONTROL_CHECKPOINT_PATH.resolve()
        and n_layer != DEFAULT_N_LAYER
    ):
        raise ValueError("the canonical Stage 24 control requires n_layer=6")
    if config.block_size % 4 != 0:
        raise ValueError("block_size must be divisible by four for position bands")
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError("config learning rate must equal the schedule initial LR")
    schedule.validate_target_step(config.max_iters)
    if treatment.checkpoint_path.resolve() == control_checkpoint_path.resolve():
        raise ValueError("control and treatment checkpoint paths must differ")
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


def create_shared_initial_state(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
) -> tuple[dict[str, torch.Tensor], str, int]:
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
    hashes = tuple(tensor_mapping_sha256(state) for state in states)
    if hashes[0] != hashes[1] or counts[0] != counts[1]:
        raise RuntimeError("dropout changed initialization or parameter count")
    is_canonical_architecture = (
        data.vocabulary.size == 65
        and config.block_size == DEFAULT_BLOCK_SIZE
        and config.n_embd == DEFAULT_N_EMBD
        and config.seed == 1_337
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
    )
    if is_canonical_architecture and hashes[0] != DEFAULT_INITIAL_STATE_SHA256:
        raise RuntimeError("canonical Stage 25 initialization hash changed")
    return states[0], hashes[0], counts[0]


def validate_stage_24_reference(
    path: Path,
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
    expected_checkpoint_sha256: str | None = None,
    expected_step: int | None = None,
) -> ReferenceCheckpoint:
    before_hash = checkpoint_sha256(path)
    if (
        expected_checkpoint_sha256 is not None
        and before_hash != expected_checkpoint_sha256
    ):
        raise ValueError(
            "Stage 24 L=6, p=0.10 checkpoint SHA-256 must be "
            f"{expected_checkpoint_sha256}, got {before_hash}"
        )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw = _require_mapping(raw, name="Stage 24 checkpoint")
    experiment = _require_mapping(raw.get("experiment"), name="experiment")
    initialization = _require_mapping(raw.get("initialization"), name="initialization")
    if initialization.get("branch_state_sha256") != initial_state_sha256:
        raise ValueError("Stage 24 L=6 initialization does not match Stage 25")
    control_hash = experiment.get("control_checkpoint_sha256")
    control_step = experiment.get("control_checkpoint_step")
    if not isinstance(control_hash, str):
        raise ValueError("Stage 24 source-control checkpoint hash is missing")
    _validate_non_negative_integer(control_step, name="Stage 24 source-control step")
    assert isinstance(control_step, int)

    source_control = DepthSpec(
        name=f"depth-{n_layer - 2}",
        n_layer=n_layer - 2,
        checkpoint_path=path.with_name(f"{path.name}.source-control-placeholder"),
    )
    source_treatment = DepthSpec(
        name=f"depth-{n_layer}",
        n_layer=n_layer,
        checkpoint_path=path,
    )
    paired_initialization = _STAGE_24.create_paired_initial_states(
        data,
        config,
        source_control,
        source_treatment,
        n_head,
        DEFAULT_CONTROL_DROPOUT,
    )
    if (
        paired_initialization.treatment_state_sha256 != initial_state_sha256
        or paired_initialization.treatment_parameter_count != parameter_count
    ):
        raise ValueError("Stage 24 L=6 initialization identity does not match")
    checkpoint = _STAGE_24.validate_stage_24_checkpoint(
        path,
        data,
        config,
        source_treatment,
        source_control,
        source_treatment,
        n_head,
        schedule,
        bands,
        paired_initialization,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        control_checkpoint_sha256=control_hash,
        control_checkpoint_step=control_step,
        dropout=DEFAULT_CONTROL_DROPOUT,
    )
    step = int(checkpoint["step"])
    if expected_step is not None and step != expected_step:
        raise ValueError(
            f"Stage 24 L=6, p=0.10 checkpoint step must be {expected_step}, got {step}"
        )
    generator_state = checkpoint.get("training_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Stage 24 training generator state is not a tensor")
    if checkpoint_sha256(path) != before_hash:
        raise RuntimeError(f"checkpoint changed during validation: {path}")
    return ReferenceCheckpoint(
        name=f"L={n_layer},p=0.10",
        dropout=DEFAULT_CONTROL_DROPOUT,
        checkpoint_path=path,
        checkpoint_sha256=before_hash,
        checkpoint_step=step,
        fixed_panel_loss=float(checkpoint["best_val_loss"]),
        best_validation_losses=_STAGE_24._validate_serialized_losses(
            checkpoint.get("best_validation_losses"),
            bands,
            name="best_validation_losses",
        ),
        training_generator_state=generator_state.detach().cpu().clone(),
        initial_state_sha256=initial_state_sha256,
        parameter_count=parameter_count,
    )


@dataclass(slots=True)
class TreatmentValidationCheckpoint(_STAGE_24._STAGE_23.TreatmentValidationCheckpoint):
    def _payload(self, step: int) -> dict[str, object]:
        payload = super()._payload(step)
        initialization = payload["initialization"]
        assert isinstance(initialization, dict)
        initialization.update(
            {
                "kind": INITIALIZATION_KIND,
                "source_stage": 24,
                "source_branch": f"depth-{self.n_layer}",
                "identical_complete_model_state": True,
            }
        )
        experiment = payload["experiment"]
        assert isinstance(experiment, dict)
        experiment.update(
            {
                "stage": 25,
                "control_residual_dropout": DEFAULT_CONTROL_DROPOUT,
                "control_stage": 24,
                "control_branch": f"depth-{self.n_layer}",
                "identical_fixed_training_batches": True,
                "fixed_transformer_depth": self.n_layer,
                "capacity_regularization_interaction": True,
                "source_depth_scaling_stage": 24,
                "identical_dropout_masks": False,
                "identical_complete_model_state": True,
                "same_training_rng_seed": True,
                "same_optimizer_configuration": True,
            }
        )
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
        return payload


def validate_stage_25_checkpoint(
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
    checkpoint = _validate_checkpoint_base(
        path,
        treatment,
        data,
        config,
        n_head,
        n_layer,
        schedule,
        bands,
        initial_state_sha256=initial_state_sha256,
        parameter_count=parameter_count,
        training_batch_seed=training_batch_seed,
        training_rng_seed=training_rng_seed,
        control_checkpoint_sha256=control_checkpoint_sha256,
        control_checkpoint_step=control_checkpoint_step,
        checkpoint_stage=25,
        control_stage=24,
        control_branch=f"depth-{n_layer}",
        control_dropout=DEFAULT_CONTROL_DROPOUT,
        initialization_kind=INITIALIZATION_KIND,
    )
    experiment = _require_mapping(checkpoint.get("experiment"), name="experiment")
    expected_experiment = {
        "identical_fixed_training_batches": True,
        "fixed_transformer_depth": n_layer,
        "capacity_regularization_interaction": True,
        "source_depth_scaling_stage": 24,
        "identical_dropout_masks": False,
        "identical_complete_model_state": True,
        "same_training_rng_seed": True,
        "same_optimizer_configuration": True,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(f"checkpoint experiment.{key} must be {expected!r}")
    initialization = _require_mapping(
        checkpoint.get("initialization"), name="initialization"
    )
    if initialization.get("source_stage") != 24:
        raise ValueError("checkpoint initialization.source_stage must be 24")
    if initialization.get("source_branch") != f"depth-{n_layer}":
        raise ValueError(
            f"checkpoint initialization.source_branch must be 'depth-{n_layer}'"
        )
    if initialization.get("identical_complete_model_state") is not True:
        raise ValueError(
            "checkpoint initialization.identical_complete_model_state must be True"
        )
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
    return checkpoint


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
                    "treatment batch-generator state does not match the Stage 24 "
                    f"control at step {record.step}"
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
        raise RuntimeError("treatment did not load the exact shared initialization")
    if count_parameters(model) != parameter_count:
        raise RuntimeError("treatment parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.for_update(0),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
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
            "the Stage 24 control step was not encountered in Stage 25 evaluations"
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
    checkpoint = _require_mapping(checkpoint, name="Stage 25 checkpoint")
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
    del checkpoint, callback, best, optimizer, model
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
    if expected_control_sha256 is None or expected_control_step is None:
        raise ValueError(
            "Stage 25 requires an expected control checkpoint SHA-256 and step"
        )
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
    control = validate_stage_24_reference(
        control_checkpoint_path,
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

    print("\nControlled L=6 capacity-regularization interaction protocol")
    print(
        "  The Stage 24 L=6, p=0.10 best checkpoint is validated and remains read-only."
    )
    print(
        "  The L=6, p=0.15 treatment starts from the byte-identical L=6 "
        "initialization, with a fresh AdamW optimizer."
    )
    print(
        "  Training batches, fixed panels, updates, and LR schedule match "
        "Stage 24 exactly."
    )
    print(
        "  Fixed train and validation panels run in eval mode with dropout "
        "disabled; eval-gap is therefore apples-to-apples."
    )
    print(f"  Shared initialization SHA-256: {initial_hash}")
    print(f"  Stage 24 control SHA-256:      {control.checkpoint_sha256}")

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
        raise RuntimeError("p=0.10 and p=0.15 initial fixed-panel losses differ")
    if checkpoint_sha256(control_checkpoint_path) != control_hash_before:
        raise RuntimeError("Stage 24 control checkpoint changed during training")
    treatment_hash = checkpoint_sha256(treatment.checkpoint_path)
    validate_stage_25_checkpoint(
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
    _replay_treatment_fixed_panel(
        treatment,
        treatment_hash,
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
        raise RuntimeError("Stage 24 control checkpoint changed during validation")
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
        treatment_checkpoint_sha256=treatment_hash,
    )


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
    control = validate_stage_24_reference(
        experiment.control.reference.checkpoint_path,
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
        expected_checkpoint_sha256=experiment.control.reference.checkpoint_sha256,
        expected_step=experiment.control.reference.checkpoint_step,
    )
    treatment_hash = checkpoint_sha256(treatment.checkpoint_path)
    if treatment_hash != experiment.treatment_checkpoint_sha256:
        raise RuntimeError(
            "Stage 25 treatment checkpoint changed after checkpoint selection: "
            f"expected {experiment.treatment_checkpoint_sha256}, got "
            f"{treatment_hash}"
        )
    treatment_checkpoint = validate_stage_25_checkpoint(
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
        name="p=0.10",
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
        name="p=0.15",
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
        raise RuntimeError(
            "Stage 24 control checkpoint changed during fresh validation"
        )
    if checkpoint_sha256(treatment.checkpoint_path) != treatment_hash:
        raise RuntimeError(
            "Stage 25 treatment checkpoint changed during fresh validation"
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
    print("\nStage 25 summary")
    print(f"  parameters per branch:        {report.parameter_count:,}")
    print(
        f"  shared initial eval train/val: "
        f"{report.initial_losses.train.overall:.4f} / "
        f"{report.initial_losses.val.overall:.4f}"
    )
    print(
        f"  p=0.10 best eval-train/val/gap: "
        f"{control.losses_at_best.train.overall:.4f} / "
        f"{control.losses_at_best.val.overall:.4f} / "
        f"{control.generalization_gap_at_best:.4f} @ {control.best_step}"
    )
    print(
        f"  p=0.15 best eval-train/val/gap: "
        f"{treatment.best_training_losses.overall:.4f} / "
        f"{treatment.best_validation_losses.overall:.4f} / "
        f"{treatment.generalization_gap_at_best:.4f} @ {treatment.best_step}"
    )
    print(
        f"  best fixed-panel delta:       {report.best_val_delta:+.6f} "
        "(p=0.15 - p=0.10)"
    )
    print(
        f"  p=0.15 final eval-train/val/gap: "
        f"{treatment.training.final.train.overall:.4f} / "
        f"{treatment.training.final.val.overall:.4f} / "
        f"{treatment.final_generalization_gap:.4f}"
    )
    print("  Fixed-panel train and val losses use model.eval(); dropout is off.")
    print(f"  read-only p=0.10 checkpoint:  {control.reference.checkpoint_path}")
    print(f"  p=0.15 checkpoint:            {treatment.spec.checkpoint_path}")
    print(f"  p=0.15 checkpoint SHA-256:    {report.treatment_checkpoint_sha256}")
    print(f"\np=0.15 best-checkpoint generated text (step {treatment.best_step})")
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
        name="dropout-p015",
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
    control_pin, control_step_pin = resolve_control_pin(
        args.control_checkpoint_path,
        args.control_checkpoint_sha256,
        args.control_checkpoint_step,
    )
    data = CharacterData.from_file(args.data)
    device = resolve_device(args.device)
    print(
        f"Stage 25: C={config.n_embd}, L={args.n_layer} residual dropout "
        "p=0.10 -> p=0.15"
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
        f"Treatment: reused p={DEFAULT_CONTROL_DROPOUT:g} best checkpoint vs "
        f"new residual dropout p={treatment.dropout:g}"
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
        _STAGE_24._STAGE_23._STAGE_22._STAGE_21.print_precise_validation(
            precise,
            stage_label="Stage 24/25",
        )


if __name__ == "__main__":
    main()
