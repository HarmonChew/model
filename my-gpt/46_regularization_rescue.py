from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from config import TrainingConfig
from data_utils import CharacterData
from train import resolve_device
from training import EvaluationRecord, LossEstimate, TrainingResult


_SPEC = importlib.util.spec_from_file_location(
    "stage_45_for_stage_46", Path(__file__).with_name("45_width_scaling.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE_45 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE_45
_SPEC.loader.exec_module(_STAGE_45)
_STAGE_44 = _STAGE_45._STAGE_44
_STAGE_43 = _STAGE_45._STAGE_43
_STAGE_42 = _STAGE_45._STAGE_42
_STAGE_41 = _STAGE_45._STAGE_41
_STAGE_40 = _STAGE_45._STAGE_40
# The architecture owner for this lineage: Stage 36 builds the LLLLGG T=256
# model from Stage 30's factory with a module-global dropout of 0.10. Stage 46
# re-issues that same construction with dropout exposed as a parameter.
_STAGE_36 = _STAGE_43._STAGE_38._STAGE_37._STAGE_36

# Reuse the sampler, panel geometry, schedule, and statistics unchanged.
# Only the residual dropout probabilities and the artifact identity change, so
# this stage cannot drift from the Stage 45 training protocol.
build_layer_attention_masks = _STAGE_36.build_layer_attention_masks
apply_attention_masks = _STAGE_36._STAGE_32.apply_attention_masks
GPTLanguageModel = _STAGE_36.GPTLanguageModel
get_batch = _STAGE_43.get_batch
evaluate_on_fixed_batches = _STAGE_43.evaluate_on_fixed_batches
capture_rng_state = _STAGE_43.capture_rng_state
state_hash = _STAGE_43.state_hash
file_hash = _STAGE_43.file_hash
fingerprint_data = _STAGE_43.fingerprint_data
stream_hashes = _STAGE_43.stream_hashes
architecture_metadata = _STAGE_43.architecture_metadata
make_schedule = _STAGE_43.make_schedule
schedule_metadata = _STAGE_43.schedule_metadata

EXPERIMENT_STAGE = 46
PROTOCOL_NAME = "paired_architecture_native_regularization"
DEFAULT_WIDTH = 320
DEFAULT_CONTROL_DROPOUT = 0.10
DEFAULT_TREATMENT_DROPOUT = 0.15
DEFAULT_TARGET_STEP = 50_000
DEFAULT_FRESH_VALIDATION_ITERS = 500
DEFAULT_FRESH_VALIDATION_SEED = 1_360
DEFAULT_FIXED_PANEL_SEED = _STAGE_44.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_44.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_44.DEFAULT_TRAINING_RNG_SEED
DEFAULT_OUTPUT_DIRECTORY = _STAGE_41.CHECKPOINT_DIRECTORY
DEFAULT_COMPLETE_DATA_PATH = _STAGE_41.DEFAULT_COMPLETE_DATA_PATH
# A read-only cross-stage readout: Stage 45's fresh C256 control endpoint is
# the current width frontier, and Stage 46's own control re-derives Stage 45's
# C320 treatment. Comparing the three stored endpoints on one shared fresh
# panel answers whether the rescued C320 overtakes C256.
DEFAULT_FRONTIER_FINAL_CHECKPOINT_PATH = (
    DEFAULT_OUTPUT_DIRECTORY / "stage_45_control_c256_step_50000_final_checkpoint.pt"
)
DEFAULT_FRONTIER_FINAL_CHECKPOINT_SHA256 = (
    "e58ce4532a3ce0623ea622b91ea7ee04232c8d1f12d8b2c53591c7efec299e9f"
)
DEFAULT_CROSS_CHECK_SEED = 1_361
N_HEAD = _STAGE_44.N_HEAD
N_LAYER = _STAGE_44.N_LAYER
INITIALIZATION_POLICY = (
    "Both branches recreate the same architecture-native initialization from "
    "the same seed and corpus. Residual dropout has no learnable parameters and "
    "consumes no initialization RNG, so the paired initial weights are claimed "
    "and checked to be byte-identical."
)
DROPOUT_POLICY = (
    "Both branches begin from the same CPU and accelerator training RNG states, "
    "reset after native initialization, and consume the same sampled training "
    "panels. Different dropout probabilities draw different masks, so neither "
    "identical dropout masks nor identical terminal Torch RNG states are claimed."
)
UNCERTAINTY_SCOPE = (
    "Student-t 95% panel-sampling interval conditional on these checkpoints; "
    "not training-seed or corpus-IID uncertainty."
)
# Provenance references only. Stage 46 never opens or restores these artifacts.
HISTORICAL_LINEAGE = {
    "stage_41_authenticated_44k_ancestor_sha256": _STAGE_42.DEFAULT_SOURCE_CHECKPOINT_SHA256,
    "stage_44_empirical_best": copy.deepcopy(
        _STAGE_45.HISTORICAL_LINEAGE["stage_44_empirical_best"]
    ),
    "stage_45_c320_under_regularized_endpoint": {
        "branch": "treatment_c320",
        "dropout": DEFAULT_CONTROL_DROPOUT,
        "step": 50_000,
        "best_step": 39_000,
        "best_checkpoint_sha256": "dbcaffe6a4d186c3c3d8c99fab215aafd5a97e178f0c50ff1d1e0b7140da4c86",
        "final_checkpoint_sha256": "a6f908c18d02ea3d7a9060d4fb7886320bff41eea06a0e97e2196ea57f34984b",
        "fixed_panel_train_loss": 1.0492555099725722,
        "fixed_panel_validation_loss": 1.364953727722168,
        "role": "the state this stage's p=0.10 control must re-derive from zero",
    },
    "role": "historical reference only; none is a Stage 46 initialization",
}


def canonical_config(dropout: float = DEFAULT_CONTROL_DROPOUT) -> TrainingConfig:
    """The locked Stage 46 training config; dropout travels with BranchSpec."""

    if dropout not in (DEFAULT_CONTROL_DROPOUT, DEFAULT_TREATMENT_DROPOUT):
        raise ValueError("Stage 46 canonical dropout probabilities are 0.10 and 0.15")
    return replace(_STAGE_42.canonical_config(), n_embd=DEFAULT_WIDTH, learning_rate=1e-3)


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    dropout: float = DEFAULT_CONTROL_DROPOUT,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    """The Stage 45 architecture factory with residual dropout as a variable."""

    if config.n_embd % n_head:
        raise ValueError("width must be divisible by the head count")
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )
    apply_attention_masks(model, build_layer_attention_masks(config.block_size))
    if device is not None:
        model = model.to(device)
    return model


@dataclass(frozen=True, slots=True)
class BranchSpec:
    key: str
    width: int
    dropout: float
    best_path: Path
    final_path: Path

    def __post_init__(self) -> None:
        if self.key not in ("control", "treatment"):
            raise ValueError("Stage 46 branch must be control or treatment")
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise ValueError("width must be an integer")
        if self.width <= 0 or self.width % N_HEAD:
            raise ValueError("width must be positive and divisible by four heads")
        probability = self.dropout
        if isinstance(probability, bool) or not isinstance(probability, float):
            raise ValueError("dropout must be a float")
        if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
            raise ValueError(f"dropout must be finite and in [0, 1), got {probability!r}")


def branch_specs(directory: Path) -> tuple[BranchSpec, BranchSpec]:
    return tuple(
        BranchSpec(
            key,
            DEFAULT_WIDTH,
            dropout,
            directory
            / f"stage_46_{key}_c{DEFAULT_WIDTH}_{dropout_label(dropout)}_best_checkpoint.pt",
            directory
            / (
                f"stage_46_{key}_c{DEFAULT_WIDTH}_{dropout_label(dropout)}"
                "_step_50000_final_checkpoint.pt"
            ),
        )
        for key, dropout in (
            ("control", DEFAULT_CONTROL_DROPOUT),
            ("treatment", DEFAULT_TREATMENT_DROPOUT),
        )
    )


def dropout_label(dropout: float) -> str:
    """Name a residual-dropout probability the way earlier stages did: p010."""

    return f"p{round(dropout * 100):03d}"


def validate_branch_specs(specs: tuple[BranchSpec, ...]) -> None:
    expected = tuple(
        (key, dropout)
        for key, dropout in (
            ("control", DEFAULT_CONTROL_DROPOUT),
            ("treatment", DEFAULT_TREATMENT_DROPOUT),
        )
    )
    if tuple((spec.key, spec.dropout) for spec in specs) != expected or any(
        spec.width != DEFAULT_WIDTH for spec in specs
    ):
        raise ValueError("Stage 46 locks control p=0.10 against treatment p=0.15 at C320")


def validate_protocol(data: CharacterData, configs: tuple[TrainingConfig, ...]) -> None:
    if configs != (canonical_config(), canonical_config(DEFAULT_TREATMENT_DROPOUT)):
        raise ValueError("Stage 46 locks all settings except residual dropout")
    if fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Stage 46 requires unchanged Complete Shakespeare and split")


def validate_artifact_paths(
    inputs: tuple[Path, ...], outputs: tuple[Path, ...]
) -> None:
    paths = [path.resolve() for path in (*inputs, *outputs)]
    if len(paths) != len(set(paths)):
        raise ValueError("corpus, checkpoints, and reports must have distinct paths")
    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Stage 46 output exists; use a new --output-dir: {path}"
            )
        if path.with_name(f".{path.name}.tmp").exists():
            raise FileExistsError(f"Stage 46 temporary output exists: {path}")


def dropout_architecture_metadata(
    data: CharacterData, config: TrainingConfig, dropout: float
) -> dict[str, Any]:
    metadata = dict(architecture_metadata(data, config))
    if "residual_dropout" not in metadata:
        raise RuntimeError("inherited architecture metadata lost its dropout field")
    metadata["residual_dropout"] = dropout
    return metadata


def verify_control_factory_equivalence(
    data: CharacterData, config: TrainingConfig
) -> str:
    """Prove the exposed-dropout factory reproduces the Stage 45 p=0.10 model."""

    _STAGE_41.seed_everything(config.seed)
    reference = _STAGE_45.build_model(data, config, N_HEAD, N_LAYER, torch.device("cpu"))
    reference_hash = _STAGE_41.tensor_mapping_sha256(reference.state_dict())
    del reference
    _STAGE_41.seed_everything(config.seed)
    exposed = build_model(
        data, config, N_HEAD, N_LAYER, DEFAULT_CONTROL_DROPOUT, torch.device("cpu")
    )
    exposed_hash = _STAGE_41.tensor_mapping_sha256(exposed.state_dict())
    del exposed
    if exposed_hash != reference_hash:
        raise RuntimeError("Stage 46 p=0.10 factory diverged from the Stage 45 factory")
    return exposed_hash


def initialize_branch(
    data: CharacterData,
    config: TrainingConfig,
    dropout: float,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.Generator, dict[str, Any]]:
    if config.n_embd % N_HEAD or config.block_size < 23 or config.batch_size % 2:
        raise ValueError(
            "requires width divisible by four, T >= 23, and even batch size"
        )
    if config.learning_rate != make_schedule().for_update(0) or config.max_iters <= 0:
        raise ValueError(
            "fresh runs require the established initial LR and positive budget"
        )
    # Always construct on CPU from the same seed, then move the native model.
    # No tensor padding, checkpoint loading, or optimizer-moment transplant.
    _STAGE_41.seed_everything(config.seed)
    model = build_model(data, config, N_HEAD, N_LAYER, dropout, torch.device("cpu"))
    _STAGE_41.verify_model_masks(model)
    initial_hash = _STAGE_41.tensor_mapping_sha256(model.state_dict())
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
    )
    generator = torch.Generator().manual_seed(DEFAULT_TRAINING_BATCH_SEED)
    _STAGE_41.seed_everything(DEFAULT_TRAINING_RNG_SEED)
    initialization = {
        "kind": "fresh_architecture_native",
        "origin_stage": 46,
        "step": 0,
        "seed": config.seed,
        "device": "cpu",
        "state_sha256": initial_hash,
        "policy": INITIALIZATION_POLICY,
        "residual_dropout": dropout,
        "historical_weights_loaded": False,
        "historical_optimizer_loaded": False,
        "optimizer_state_entries_at_initialization": len(optimizer.state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, optimizer, generator, initialization


def checkpoint_payload(
    spec: BranchSpec,
    data: CharacterData,
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
    record: EvaluationRecord,
    best: EvaluationRecord,
    kind: str,
    initialization: dict[str, Any],
    initial_streams: dict[str, str],
) -> dict[str, Any]:
    device_type, device_name = _STAGE_41._runtime_identity(device)
    payload = {
        "checkpoint_version": 1,
        "checkpoint_kind": kind,
        "step": record.step,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "training_generator_state": generator.get_state().clone(),
        "rng_state": capture_rng_state(device),
        "optimizer_restart_step": 0,
        "optimizer_provenance_known": True,
        "snapshot_losses": {
            "train": record.losses.train,
            "validation": record.losses.val,
        },
        "best_step": best.step,
        "best_val_loss": best.losses.val,
        "best_training_loss": best.losses.train,
        "architecture": dropout_architecture_metadata(data, config, spec.dropout),
        "data_fingerprint": fingerprint_data(data),
        "corpus": {
            "train_characters": len(data.train_data),
            "validation_characters": len(data.val_data),
            "vocabulary_size": data.vocabulary.size,
            "train_fraction": 0.9,
        },
        "initialization": copy.deepcopy(initialization),
        "initial_streams": dict(initial_streams),
        "training_config": {
            **asdict(config),
            "learning_rate_schedule": schedule_metadata(config.max_iters),
            "training_batch_seed": DEFAULT_TRAINING_BATCH_SEED,
            "training_rng_seed": DEFAULT_TRAINING_RNG_SEED,
            "fixed_panel_seed": DEFAULT_FIXED_PANEL_SEED,
            "targets_per_update": config.batch_size * config.block_size,
            "total_target_presentations": record.step
            * config.batch_size
            * config.block_size,
            "target_total_presentations": config.max_iters
            * config.batch_size
            * config.block_size,
            "sampler": "B/2 contiguous 2T panels reshaped into B T-token rows",
        },
        "experiment": {
            "stage": 46,
            "protocol": PROTOCOL_NAME,
            "comparison_variable": "residual_dropout",
            "branch_key": spec.key,
            "branch_residual_dropout": spec.dropout,
            "control_residual_dropout": DEFAULT_CONTROL_DROPOUT,
            "treatment_residual_dropout": DEFAULT_TREATMENT_DROPOUT,
            "from_scratch": True,
            "trained_branches": 2,
            "canonical_protocol": (
                config == canonical_config()
                and spec.width == DEFAULT_WIDTH
                and spec.dropout in (DEFAULT_CONTROL_DROPOUT, DEFAULT_TREATMENT_DROPOUT)
                and fingerprint_data(data)
                == _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT
            ),
        },
        "evaluation_protocol": {
            "selection_metric": "strict_minimum_fixed_panel_validation_loss",
            "primary_metric": "fresh_paired_terminal_validation_treatment_minus_control",
            "fresh_panel_seed": DEFAULT_FRESH_VALIDATION_SEED,
            "fresh_panel_batches": DEFAULT_FRESH_VALIDATION_ITERS,
            "dropout_enabled": False,
        },
        "randomness_policy": DROPOUT_POLICY,
        "runtime": {
            "device_type": device_type,
            "device_name": device_name,
            "torch_version": str(torch.__version__),
        },
        "historical_lineage": copy.deepcopy(HISTORICAL_LINEAGE),
    }
    payload["payload_sha256"] = state_hash(payload)
    return payload


def validate_branch_checkpoint(
    path: Path,
    spec: BranchSpec,
    data: CharacterData,
    config: TrainingConfig,
    kind: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Stage 46 checkpoint must be a mapping")
    digest = payload.pop("payload_sha256", None)
    if digest != state_hash(payload):
        raise ValueError("Stage 46 checkpoint payload integrity mismatch")
    payload["payload_sha256"] = digest
    if payload["checkpoint_version"] != 1 or payload["checkpoint_kind"] != kind:
        raise ValueError("Stage 46 checkpoint version or kind mismatch")
    if (
        payload["experiment"]["stage"] != 46
        or payload["experiment"]["protocol"] != PROTOCOL_NAME
    ):
        raise ValueError("Stage 46 experiment mismatch")
    if (
        payload["experiment"]["branch_key"] != spec.key
        or payload["experiment"]["branch_residual_dropout"] != spec.dropout
        or config.n_embd != spec.width
    ):
        raise ValueError("Stage 46 checkpoint branch mismatch")
    if payload["architecture"] != dropout_architecture_metadata(
        data, config, spec.dropout
    ):
        raise ValueError("Stage 46 architecture changed, including residual dropout")
    if payload["data_fingerprint"] != fingerprint_data(data):
        raise ValueError("Stage 46 corpus or split changed")
    for key, value in asdict(config).items():
        if payload["training_config"].get(key) != value:
            raise ValueError(f"Stage 46 training config changed: {key}")
    if payload["training_config"]["learning_rate_schedule"] != schedule_metadata(
        config.max_iters
    ):
        raise ValueError("Stage 46 learning rate schedule changed")
    step, best_step = payload["step"], payload["best_step"]
    if (
        type(step) is not int
        or type(best_step) is not int
        or not 0 <= best_step <= step <= config.max_iters
    ):
        raise ValueError("Stage 46 checkpoint steps are invalid")
    if kind == "final" and step != config.max_iters:
        raise ValueError("Stage 46 final must be the matched terminal step")
    if kind == "best" and step != best_step:
        raise ValueError("Stage 46 best must contain its selected weights")
    if not all(
        math.isfinite(value)
        for value in (
            *payload["snapshot_losses"].values(),
            payload["best_val_loss"],
            payload["best_training_loss"],
        )
    ):
        raise ValueError("Stage 46 losses must be finite")
    initialization = payload["initialization"]
    if (
        initialization["kind"] != "fresh_architecture_native"
        or initialization["origin_stage"] != 46
        or initialization["step"] != 0
        or initialization["seed"] != config.seed
        or initialization["residual_dropout"] != spec.dropout
        or initialization["historical_weights_loaded"]
        or initialization["historical_optimizer_loaded"]
        or initialization["optimizer_state_entries_at_initialization"] != 0
        or payload["optimizer_restart_step"] != 0
    ):
        raise ValueError("Stage 46 requires fresh model and optimizer initialization")
    optimizer_state = payload["optimizer_state_dict"]
    groups, states = optimizer_state["param_groups"], optimizer_state["state"]
    if len(groups) != 1:
        raise ValueError("Stage 46 requires one AdamW parameter group")
    group = groups[0]
    if (
        group["lr"] != make_schedule().for_update(max(0, step - 1))
        or group["betas"] != (0.9, 0.999)
        or group["weight_decay"] != 0.01
        or group["eps"] != 1e-8
        or group["amsgrad"]
    ):
        raise ValueError("Stage 46 AdamW settings changed")
    if step == 0:
        if states:
            raise ValueError("Stage 46 step-zero optimizer must be empty")
    elif set(states) != set(group["params"]) or any(
        state["step"].item() != step for state in states.values()
    ):
        raise ValueError("Stage 46 AdamW counters do not match completed updates")
    return payload


@dataclass(frozen=True, slots=True)
class BranchReport:
    spec: BranchSpec
    training: TrainingResult
    best_step: int
    best_losses: LossEstimate
    initialization: dict[str, Any]
    initial_streams: dict[str, str]
    final_streams: dict[str, str]
    best_sha256: str
    final_sha256: str
    elapsed_seconds: float


def train_branch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    on_evaluation: Any,
) -> TrainingResult:
    """The Stage 45 update loop, re-evaluated against this module's globals."""
    history = []
    schedule = make_schedule()

    def evaluate(step: int) -> None:
        rng_hash = state_hash(capture_rng_state(device))
        batch_hash = state_hash(generator.get_state())
        losses = evaluate_on_fixed_batches(
            model,
            data,
            config,
            device,
            fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED,
        )
        if not all(math.isfinite(value) for value in (losses.train, losses.val)):
            raise ValueError(f"non-finite fixed-panel loss at step {step}")
        record = EvaluationRecord(step, losses)
        on_evaluation(record)
        if rng_hash != state_hash(capture_rng_state(device)):
            raise RuntimeError(
                "fixed evaluation or checkpoint saving advanced training RNG"
            )
        if batch_hash != state_hash(generator.get_state()):
            raise RuntimeError("fixed evaluation advanced the training batch stream")
        history.append(record)

    model.train()
    evaluate(0)
    for update in range(config.max_iters):
        _STAGE_40._set_optimizer_learning_rate(optimizer, schedule.for_update(update))
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_batch(data, "train", config, generator, device)
        logits, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss).item():
            raise ValueError(f"non-finite training loss at update {update}")
        loss.backward()
        optimizer.step()
        del inputs, targets, logits, loss
        if (update + 1) % config.eval_interval == 0 or update + 1 == config.max_iters:
            evaluate(update + 1)
    return TrainingResult(history[0].losses, history[-1].losses, tuple(history), None)


def run_branch(
    spec: BranchSpec, data: CharacterData, config: TrainingConfig, device: torch.device
) -> BranchReport:
    """Run from zero. Reduced configs are for internal tests, not the CLI protocol."""
    if spec.width != config.n_embd:
        raise ValueError("branch width does not match its configuration")
    validate_artifact_paths((), (spec.best_path, spec.final_path))
    _STAGE_41.clear_accelerator_cache(device)
    model, optimizer, generator, initialization = initialize_branch(
        data, config, spec.dropout, device
    )
    initial_streams = stream_hashes(model, optimizer, generator, device)
    best = None
    started = time.perf_counter()

    def on_evaluation(record: EvaluationRecord) -> None:
        nonlocal best
        improved = best is None or record.losses.val < best.losses.val
        if improved:
            best = record
        for kind, path, should_save in (
            ("best", spec.best_path, improved),
            ("final", spec.final_path, record.step == config.max_iters),
        ):
            if should_save:
                payload = checkpoint_payload(
                    spec,
                    data,
                    config,
                    model,
                    optimizer,
                    generator,
                    device,
                    record,
                    best,
                    kind,
                    initialization,
                    initial_streams,
                )
                _STAGE_40._atomic_save_payload(path, payload)
        print(
            f"  {spec.key} C{spec.width} p{spec.dropout:g} step {record.step:5d} | "
            f"lr {make_schedule().for_update(max(0, record.step - 1)):g} | "
            f"train {record.losses.train:.4f} | val {record.losses.val:.4f} | "
            f"best {best.losses.val:.4f} @ {best.step}",
            flush=True,
        )

    training = train_branch(
        model, optimizer, generator, data, config, device, on_evaluation
    )
    final_streams = stream_hashes(model, optimizer, generator, device)
    for kind, path in (("best", spec.best_path), ("final", spec.final_path)):
        payload = validate_branch_checkpoint(path, spec, data, config, kind)
        if payload["initialization"] != initialization:
            raise RuntimeError("checkpoint lost native initialization provenance")
        if kind == "final":
            restored_hashes = {
                "model_sha256": _STAGE_41.tensor_mapping_sha256(
                    payload["model_state_dict"]
                ),
                "optimizer_sha256": state_hash(payload["optimizer_state_dict"]),
                "training_generator_sha256": _STAGE_41.tensor_sha256(
                    payload["training_generator_state"]
                ),
                "torch_rng_sha256": state_hash(payload["rng_state"]),
            }
            if restored_hashes != final_streams:
                raise RuntimeError(
                    "Stage 46 terminal state failed checkpoint round trip"
                )
    assert best is not None
    report = BranchReport(
        spec,
        training,
        best.step,
        best.losses,
        initialization,
        initial_streams,
        final_streams,
        file_hash(spec.best_path),
        file_hash(spec.final_path),
        time.perf_counter() - started,
    )
    return report


def verify_paired_streams(
    control_spec: BranchSpec,
    treatment_spec: BranchSpec,
    control_initial: dict[str, str],
    treatment_initial: dict[str, str],
    control_final: dict[str, str],
    treatment_final: dict[str, str],
    control_parameters: int,
    treatment_parameters: int,
) -> None:
    """Spec, weight, and sampler checks shared by trained pairs and stored pairs."""

    if control_spec.key != "control" or treatment_spec.key != "treatment":
        raise RuntimeError("paired comparison requires control followed by treatment")
    if (control_spec.dropout, treatment_spec.dropout) != (
        DEFAULT_CONTROL_DROPOUT,
        DEFAULT_TREATMENT_DROPOUT,
    ):
        raise RuntimeError(
            "paired dropout treatment must be exactly 0.15 against 0.10"
        )
    if control_spec.width != treatment_spec.width:
        raise RuntimeError("paired dropout comparison must hold width fixed")
    if control_parameters != treatment_parameters:
        raise RuntimeError("paired dropout branches differ in parameter count")
    # Residual dropout consumes no initialization RNG, so a shared seed must
    # produce byte-identical starting weights for both probabilities.
    if control_initial["model_sha256"] != treatment_initial["model_sha256"]:
        raise RuntimeError("paired branches started from different initial weights")
    for key in ("training_generator_sha256", "torch_rng_sha256"):
        if control_initial[key] != treatment_initial[key]:
            raise RuntimeError(f"paired branches started with different {key}")
    if (
        control_final["training_generator_sha256"]
        != treatment_final["training_generator_sha256"]
    ):
        raise RuntimeError("paired branches consumed different training batches")
    # Different dropout probabilities draw different masks, so terminal Torch
    # RNG states and terminal model states are expected to differ.


def verify_paired_branches(control: Any, treatment: Any) -> None:
    verify_paired_streams(
        control.spec,
        treatment.spec,
        control.initial_streams,
        treatment.initial_streams,
        control.final_streams,
        treatment.final_streams,
        control.initialization["parameter_count"],
        treatment.initialization["parameter_count"],
    )
    if [record.step for record in control.training.history] != [
        record.step for record in treatment.training.history
    ]:
        raise RuntimeError("paired update/evaluation budgets differ")


def evaluate_checkpoint_pair(
    control: BranchReport,
    treatment: BranchReport,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    seed: int = DEFAULT_FRESH_VALIDATION_SEED,
    eval_iters: int = DEFAULT_FRESH_VALIDATION_ITERS,
) -> Any:
    if seed in (DEFAULT_FIXED_PANEL_SEED, 1354, 1355, 1356, 1357, 1358, 1359):
        raise ValueError("Stage 46 fresh validation must use an unused panel seed")
    if eval_iters < 2:
        raise ValueError("paired uncertainty requires at least two batches")
    verify_paired_branches(control, treatment)
    caller_rng = capture_rng_state(device)
    models, payloads = [], []
    try:
        for branch in (control, treatment):
            if file_hash(branch.spec.final_path) != branch.final_sha256:
                raise ValueError("Stage 46 endpoint checkpoint hash mismatch")
            payload = validate_branch_checkpoint(
                branch.spec.final_path,
                branch.spec,
                data,
                config,
                "final",
            )
            model = build_model(
                data, config, N_HEAD, N_LAYER, branch.spec.dropout, device
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            _STAGE_41.verify_model_masks(model)
            models.append(model)
            payloads.append(payload)
        losses, generator_hash = _STAGE_40.evaluate_fresh_paired_models(
            models,
            data,
            config,
            device,
            eval_iters=eval_iters,
            seed=seed,
            fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED,
        )
        results = []
        for branch, payload, values in zip(
            (control, treatment), payloads, losses, strict=True
        ):
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Stage 46 fresh validation produced non-finite loss")
            if file_hash(branch.spec.final_path) != branch.final_sha256:
                raise RuntimeError("Stage 46 endpoint changed during evaluation")
            mean, se = _STAGE_40._mean_and_standard_error(values)
            results.append(
                _STAGE_40.FreshCheckpointResult(
                    f"{branch.spec.key} C{branch.spec.width} "
                    f"p{branch.spec.dropout:g} @ {payload['step']}",
                    branch.spec.final_path.resolve(),
                    branch.final_sha256,
                    payload["step"],
                    payload["snapshot_losses"]["validation"],
                    mean,
                    se,
                    values,
                )
            )
        return _STAGE_42.PairedValidationReport(
            eval_iters,
            seed,
            results[0],
            results[1],
            _STAGE_40._fresh_paired_delta(results[1], results[0]),
            generator_hash,
        )
    finally:
        models.clear()
        _STAGE_40._restore_rng_state(caller_rng, device)
        _STAGE_41.clear_accelerator_cache(device)


def paired_outcome(report: Any) -> str:
    if report.delta.confidence_high < 0:
        return (
            "p=0.15 improves C320 validation at matched 50k updates on the "
            "fresh paired panel"
        )
    if report.delta.confidence_low > 0:
        return "p=0.10 beats p=0.15 at C320 on the fresh paired panel"
    return "unresolved within paired panel uncertainty; this does not prove equivalence"


@dataclass(frozen=True, slots=True)
class FrontierEndpoint:
    """A stored, authenticated endpoint read as a cross-stage peer."""

    path: Path
    payload: dict[str, Any]
    config: TrainingConfig

    def __post_init__(self) -> None:
        if type(self.payload) is not dict:
            raise ValueError("frontier payload must be a checkpoint mapping")
        if self.payload.get("step") != self.config.max_iters:
            raise ValueError("frontier endpoint must store its terminal step")
        if self.payload.get("model_state_dict") is None:
            raise ValueError("frontier payload must carry model weights")


def load_frontier_checkpoint(data: CharacterData, path: Path) -> FrontierEndpoint:
    """Authenticate Stage 45's fresh C256 control endpoint as a read-only peer."""

    if path.resolve() == DEFAULT_FRONTIER_FINAL_CHECKPOINT_PATH.resolve():
        if file_hash(path) != DEFAULT_FRONTIER_FINAL_CHECKPOINT_SHA256:
            raise ValueError("Stage 45 C256 frontier checkpoint hash changed")
    spec, _ = _STAGE_45.branch_specs(path.parent)
    config = _STAGE_45.canonical_config(spec.width)
    payload = _STAGE_45.validate_branch_checkpoint(path, spec, data, config, "final")
    return FrontierEndpoint(path, payload, config)


def evaluate_frontier_comparison(
    control_spec: BranchSpec,
    treatment_spec: BranchSpec,
    control_payload: dict[str, Any],
    treatment_payload: dict[str, Any],
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    frontier: FrontierEndpoint,
    *,
    seed: int = DEFAULT_CROSS_CHECK_SEED,
    eval_iters: int = DEFAULT_FRESH_VALIDATION_ITERS,
) -> dict[str, Any]:
    """Score both C320 endpoints and the stored C256 endpoint on one fresh panel.

    No checkpoint is modified. The panel is shared per batch, so each reported
    interval is a paired comparison at matched validation batches; dropout is
    inactive in eval mode, so only stored weights differ.
    """

    if seed in (DEFAULT_FIXED_PANEL_SEED, 1354, 1355, 1356, 1357, 1358, 1359, 1360):
        raise ValueError("cross-stage panel seed must be unused")
    if eval_iters < 2:
        raise ValueError("paired uncertainty requires at least two batches")
    verify_paired_streams(
        control_spec,
        treatment_spec,
        control_payload["initial_streams"],
        treatment_payload["initial_streams"],
        {
            "training_generator_sha256": _STAGE_41.tensor_sha256(
                control_payload["training_generator_state"]
            )
        },
        {
            "training_generator_sha256": _STAGE_41.tensor_sha256(
                treatment_payload["training_generator_state"]
            )
        },
        control_payload["initialization"]["parameter_count"],
        treatment_payload["initialization"]["parameter_count"],
    )
    frontier_payload, frontier_config = frontier.payload, frontier.config
    if frontier_config.n_embd == config.n_embd:
        raise ValueError("frontier peer must differ in width from the C320 branches")
    plan = [
        (
            "frontier_c256_p010",
            frontier.path,
            frontier_payload,
            frontier_config,
            DEFAULT_CONTROL_DROPOUT,
        ),
        (
            "control_c320_p010",
            control_spec.final_path,
            control_payload,
            config,
            control_spec.dropout,
        ),
        (
            "treatment_c320_p015",
            treatment_spec.final_path,
            treatment_payload,
            config,
            treatment_spec.dropout,
        ),
    ]
    caller_rng = capture_rng_state(device)
    models = []
    try:
        for _, _, payload, branch_config, dropout in plan:
            model = build_model(data, branch_config, N_HEAD, N_LAYER, dropout, device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            _STAGE_41.verify_model_masks(model)
            models.append(model)
        losses, generator_hash = _STAGE_40.evaluate_fresh_paired_models(
            models,
            data,
            config,
            device,
            eval_iters=eval_iters,
            seed=seed,
            fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED,
        )
        results = {}
        for (key, path, payload, branch_config, dropout), values in zip(
            plan, losses, strict=True
        ):
            if not all(math.isfinite(value) for value in values):
                raise ValueError("frontier comparison produced non-finite loss")
            mean, se = _STAGE_40._mean_and_standard_error(values)
            results[key] = _STAGE_40.FreshCheckpointResult(
                f"C{branch_config.n_embd} p={dropout:g} @ {payload['step']}",
                path.resolve(),
                file_hash(path),
                payload["step"],
                payload["snapshot_losses"]["validation"],
                mean,
                se,
                values,
            )
        frontier_result = results["frontier_c256_p010"]
        under_result = results["control_c320_p010"]
        rescued_result = results["treatment_c320_p015"]
        comparisons = {
            key: asdict(_STAGE_40._fresh_paired_delta(candidate, baseline))
            for key, candidate, baseline in (
                ("rescued_minus_frontier", rescued_result, frontier_result),
                ("under_regularized_minus_frontier", under_result, frontier_result),
                ("rescued_minus_under_regularized", rescued_result, under_result),
            )
        }
        return {
            "seed": seed,
            "eval_iters": eval_iters,
            "frontier_checkpoint": str(frontier.path.resolve()),
            "frontier_checkpoint_sha256": file_hash(frontier.path),
            "shared_panel_per_batch": True,
            "dropout_enabled": False,
            "endpoints": {key: asdict(value) for key, value in results.items()},
            "comparisons": comparisons,
            "final_generator_sha256": generator_hash,
        }
    finally:
        models.clear()
        _STAGE_40._restore_rng_state(caller_rng, device)
        _STAGE_41.clear_accelerator_cache(device)


def frontier_outcome(report: dict[str, Any]) -> str:
    delta = report["comparisons"]["rescued_minus_frontier"]
    if delta["confidence_high"] < 0:
        return "rescued C320 p=0.15 overtakes C256 at matched 50k updates"
    if delta["confidence_low"] > 0:
        return "C256 still wins against rescued C320 p=0.15 at matched 50k updates"
    return "rescued C320 p=0.15 is unresolved against C256 on the shared panel"


def preflight(
    data: CharacterData,
    configs: tuple[TrainingConfig, ...],
    specs: tuple[BranchSpec, ...],
    device: torch.device,
) -> dict[str, Any]:
    validate_protocol(data, configs)
    validate_branch_specs(specs)
    caller_rng = capture_rng_state(device)
    audits = []
    try:
        factory_hash = verify_control_factory_equivalence(data, configs[0])
        for spec in specs:
            model, optimizer, generator, initialization = initialize_branch(
                data, configs[0], spec.dropout, device
            )
            audits.append(
                {
                    "branch_key": spec.key,
                    "residual_dropout": spec.dropout,
                    "architecture": dropout_architecture_metadata(
                        data, configs[0], spec.dropout
                    ),
                    "initialization": initialization,
                    "initial_streams": stream_hashes(
                        model, optimizer, generator, device
                    ),
                }
            )
            del model, optimizer, generator
        for key in ("model_sha256", "training_generator_sha256", "torch_rng_sha256"):
            if audits[0]["initial_streams"][key] != audits[1]["initial_streams"][key]:
                raise RuntimeError(f"preflight branch random streams differ in {key}")
        if audits[0]["initialization"]["parameter_count"] != audits[1]["initialization"][
            "parameter_count"
        ]:
            raise RuntimeError("preflight branches differ in parameter count")
        return {
            "stage": 46,
            "protocol": PROTOCOL_NAME,
            "training_executed": False,
            "branches": audits,
            "control_factory_equivalence": {
                "reference": "Stage 45 model factory at p=0.10",
                "state_sha256": factory_hash,
                "identical": True,
            },
            "updates_per_branch": DEFAULT_TARGET_STEP,
            "target_presentations_per_branch": DEFAULT_TARGET_STEP * 8 * 256,
            "parameter_count": audits[0]["initialization"]["parameter_count"],
            "learning_rate_schedule": schedule_metadata(DEFAULT_TARGET_STEP),
            "data_fingerprint": fingerprint_data(data),
            "initialization_policy": INITIALIZATION_POLICY,
            "randomness_policy": DROPOUT_POLICY,
            "historical_lineage": copy.deepcopy(HISTORICAL_LINEAGE),
        }
    finally:
        _STAGE_40._restore_rng_state(caller_rng, device)
        _STAGE_41.clear_accelerator_cache(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 46: fresh C320 at residual dropout p=0.10 versus p=0.15, "
            "paired 50k-update regularization experiment"
        ),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "xpu")
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_COMPLETE_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate corpus, architecture, and initialization without training",
    )
    parser.add_argument(
        "--frontier-comparison-only",
        action="store_true",
        help=(
            "score the stored C320 p=0.10, C320 p=0.15, and C256 p=0.10 endpoints "
            "on one shared fresh panel without training"
        ),
    )
    return parser.parse_args()


def run_frontier_comparison_only(args: argparse.Namespace) -> None:
    """Reuse the stored Stage 46 endpoints; never trains or rewrites them."""

    data = CharacterData.from_file(args.data, train_fraction=0.9)
    device = resolve_device(args.device)
    config = canonical_config()
    specs = branch_specs(args.output_dir)
    report_path = args.output_dir / "stage_46_c256_comparison.json"
    validate_artifact_paths((args.data,), (report_path,))
    inputs = (specs[0].final_path, specs[1].final_path)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Stage 46 endpoint not present: {path}")
    control_payload = validate_branch_checkpoint(
        specs[0].final_path, specs[0], data, config, "final"
    )
    treatment_payload = validate_branch_checkpoint(
        specs[1].final_path, specs[1], data, config, "final"
    )
    print(
        f"Stage 46 frontier on {device}: C256 p=0.10 versus both stored C320 "
        f"endpoints at {DEFAULT_CROSS_CHECK_SEED}",
        flush=True,
    )
    frontier = load_frontier_checkpoint(data, DEFAULT_FRONTIER_FINAL_CHECKPOINT_PATH)
    report = evaluate_frontier_comparison(
        specs[0],
        specs[1],
        control_payload,
        treatment_payload,
        data,
        config,
        device,
        frontier,
    )
    outcome = frontier_outcome(report)
    report["frontier_outcome"] = outcome
    report["uncertainty_scope"] = UNCERTAINTY_SCOPE
    report["budget_interpretation"] = (
        "one shared 500-batch validation panel; dropout inactive in eval mode; "
        "C256 was trained by Stage 45 and is read here as a stored endpoint"
    )
    _STAGE_42.write_report(report_path, report)
    for key, delta in report["comparisons"].items():
        print(
            f"  {delta['candidate']} - {delta['baseline']}: "
            f"{delta['mean_delta']:+.7f}; SE {delta['standard_error']:.7f}; "
            f"95% panel interval [{delta['confidence_low']:+.7f}, "
            f"{delta['confidence_high']:+.7f}]",
            flush=True,
        )
    print(f"{outcome}. Report: {report_path.resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    if args.frontier_comparison_only:
        run_frontier_comparison_only(args)
        return
    data = CharacterData.from_file(args.data, train_fraction=0.9)
    device = resolve_device(args.device)
    configs = (canonical_config(), canonical_config(DEFAULT_TREATMENT_DROPOUT))
    specs = branch_specs(args.output_dir)
    report_path = args.output_dir / "stage_46_report.json"
    outputs = (
        report_path,
        *(path for spec in specs for path in (spec.best_path, spec.final_path)),
    )
    validate_artifact_paths((args.data,), () if args.validate_only else outputs)
    audit = preflight(data, configs, specs, device)
    print(json.dumps(audit, indent=2, allow_nan=False), flush=True)
    if args.validate_only:
        return
    print(
        f"Stage 46 on {device}: native C320 at p=0.10 and p=0.15, "
        "50,000 updates per branch",
        flush=True,
    )
    control = run_branch(specs[0], data, configs[0], device)
    treatment = run_branch(specs[1], data, configs[1], device)
    verify_paired_branches(control, treatment)
    fresh = evaluate_checkpoint_pair(control, treatment, data, configs[0], device)
    _STAGE_42.print_pair(fresh)
    outcome = paired_outcome(fresh)
    control_gap = control.training.final.val - control.training.final.train
    treatment_gap = treatment.training.final.val - treatment.training.final.train
    report = {
        **audit,
        "training_executed": True,
        "paired_batch_streams_verified": True,
        "control": asdict(control),
        "treatment": asdict(treatment),
        "fresh_50k_branches": asdict(fresh),
        "uncertainty_scope": UNCERTAINTY_SCOPE,
        "final_fixed_panel_delta_treatment_minus_control": treatment.training.final.val
        - control.training.final.val,
        "selected_best_fixed_panel_delta_treatment_minus_control": treatment.best_losses.val
        - control.best_losses.val,
        "final_generalization_gap_delta_treatment_minus_control": treatment_gap
        - control_gap,
        "outcome": outcome,
        "budget_interpretation": (
            "equal updates, target tokens, and parameter count; only residual "
            "dropout differs"
        ),
    }
    _STAGE_42.write_report(report_path, report)
    print(f"{outcome}. Report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
