from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from config import TrainingConfig
from data_utils import CharacterData
from train import resolve_device
from training import EvaluationRecord, LossEstimate, TrainingResult


_SPEC = importlib.util.spec_from_file_location(
    "stage_41_for_stage_42", Path(__file__).with_name("41_final_compute_extension.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE_41 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE_41
_SPEC.loader.exec_module(_STAGE_41)
_STAGE_40 = _STAGE_41._STAGE_40

# Reuse the established model, sampler, fixed panels, RNG primitives, and
# paired statistics. Stage 40 and Stage 41 remain unchanged and runnable.
build_model = _STAGE_41.build_model
get_batch = _STAGE_41.get_stage_41_batch
evaluate_on_fixed_batches = _STAGE_41.evaluate_on_fixed_batches
state_hash = _STAGE_40.structured_state_sha256
file_hash = _STAGE_40.checkpoint_sha256
capture_rng_state = _STAGE_40.capture_rng_state

EXPERIMENT_STAGE = 42
PROTOCOL_NAME = "paired_stage_41_terminal_full_state_learning_rate_fork"
DEFAULT_SOURCE_STEP = 44_000
DEFAULT_TARGET_STEP = 50_000
DEFAULT_CONTROL_LEARNING_RATE = 3e-5
DEFAULT_TREATMENT_LEARNING_RATE = 1e-5
DEFAULT_FRESH_VALIDATION_ITERS = 500
# Independent of the fixed panel and Stage 40's seed 1354. The endpoint
# comparison also gets a separate panel from the historical comparison.
DEFAULT_HISTORY_VALIDATION_SEED = 1_355
DEFAULT_BRANCH_VALIDATION_SEED = 1_356
DEFAULT_SOURCE_CHECKPOINT_PATH = _STAGE_41.DEFAULT_FINAL_CHECKPOINT_PATH
DEFAULT_STAGE_40_CHECKPOINT_PATH = _STAGE_41.DEFAULT_SOURCE_CHECKPOINT_PATH
DEFAULT_SOURCE_CHECKPOINT_SHA256 = (
    "c6e6a5c5eecf0d4ec90b128820dbf0f7a6d46df335f456649e230c3bb9d5b1d5"
)
DEFAULT_SOURCE_TRAIN_LOSS = 1.1989565205574035
DEFAULT_SOURCE_VAL_LOSS = 1.4180302500724793
DEFAULT_OUTPUT_DIRECTORY = _STAGE_41.CHECKPOINT_DIRECTORY
DEFAULT_FIXED_PANEL_SEED = _STAGE_41.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_41.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_41.DEFAULT_TRAINING_RNG_SEED
N_HEAD = _STAGE_41.DEFAULT_N_HEAD
N_LAYER = _STAGE_41.DEFAULT_N_LAYER


def canonical_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=_STAGE_41.DEFAULT_BATCH_SIZE,
        block_size=_STAGE_41.DEFAULT_BLOCK_SIZE,
        n_embd=_STAGE_41.DEFAULT_N_EMBD,
        learning_rate=DEFAULT_CONTROL_LEARNING_RATE,
        max_iters=DEFAULT_TARGET_STEP,
        eval_interval=_STAGE_41.DEFAULT_EVAL_INTERVAL,
        eval_iters=_STAGE_41.DEFAULT_EVAL_ITERS,
        seed=_STAGE_41.DEFAULT_SEED,
    )


@dataclass(frozen=True, slots=True)
class FinalLearningRateSchedule:
    branch_learning_rate: float

    def __post_init__(self) -> None:
        if self.branch_learning_rate not in (
            DEFAULT_CONTROL_LEARNING_RATE,
            DEFAULT_TREATMENT_LEARNING_RATE,
        ):
            raise ValueError("Stage 42 branch LR must be 3e-5 or 1e-5")

    def for_update(self, update_index: int) -> float:
        if (
            isinstance(update_index, bool)
            or not isinstance(update_index, int)
            or update_index < 0
        ):
            raise ValueError("update index must be a non-negative integer")
        if update_index < DEFAULT_SOURCE_STEP:
            return _STAGE_41.FinalComputeExtensionSchedule().for_update(update_index)
        return self.branch_learning_rate

    def as_metadata(self, *, target_step: int) -> list[dict[str, int | float]]:
        if target_step <= DEFAULT_SOURCE_STEP:
            raise ValueError("target must follow the 44k fork")
        return [
            *_STAGE_41.FinalComputeExtensionSchedule().as_metadata(
                target_step=DEFAULT_SOURCE_STEP
            ),
            {
                "start_update": DEFAULT_SOURCE_STEP,
                "end_update_exclusive": target_step,
                "learning_rate": self.branch_learning_rate,
            },
        ]


@dataclass(frozen=True, slots=True)
class BranchSpec:
    key: str
    learning_rate: float
    best_path: Path
    final_path: Path

    def __post_init__(self) -> None:
        expected = {
            "control": DEFAULT_CONTROL_LEARNING_RATE,
            "treatment": DEFAULT_TREATMENT_LEARNING_RATE,
        }
        if self.key not in expected or self.learning_rate != expected[self.key]:
            raise ValueError("Stage 42 requires control=3e-5 and treatment=1e-5")

    @property
    def schedule(self) -> FinalLearningRateSchedule:
        return FinalLearningRateSchedule(self.learning_rate)


def branch_specs(directory: Path) -> tuple[BranchSpec, BranchSpec]:
    return tuple(
        BranchSpec(
            key,
            rate,
            directory / f"stage_42_{key}_{label}_best_checkpoint.pt",
            directory / f"stage_42_{key}_{label}_step_50000_final_checkpoint.pt",
        )
        for key, rate, label in (
            ("control", DEFAULT_CONTROL_LEARNING_RATE, "3e_5"),
            ("treatment", DEFAULT_TREATMENT_LEARNING_RATE, "1e_5"),
        )
    )


@dataclass(frozen=True, slots=True)
class ForkSource:
    path: Path
    sha256: str
    checkpoint: dict[str, Any]
    runtime_device_type: str
    runtime_device_name: str
    torch_version: str

    @property
    def step(self) -> int:
        return self.checkpoint["step"]

    @property
    def losses(self) -> LossEstimate:
        losses = self.checkpoint["snapshot_losses"]
        return LossEstimate(train=losses["train"], val=losses["validation"])


def validate_artifact_paths(
    inputs: tuple[Path, ...], outputs: tuple[Path, ...]
) -> None:
    paths = [path.resolve() for path in (*inputs, *outputs)]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "corpus, sources, checkpoints, and report paths must be distinct"
        )
    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Stage 42 output already exists; use a new --output-dir: {path}"
            )


def load_sources(
    source_path: Path,
    stage_40_path: Path,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[ForkSource, Any]:
    if config != canonical_config():
        raise ValueError(
            "Stage 42 locks architecture, update budget, and fixed evaluation settings"
        )
    if file_hash(source_path) != DEFAULT_SOURCE_CHECKPOINT_SHA256:
        raise ValueError(
            "Stage 42 requires the authenticated Stage 41 44k final checkpoint"
        )
    if file_hash(stage_40_path) != _STAGE_41.DEFAULT_SOURCE_CHECKPOINT_SHA256:
        raise ValueError(
            "fresh 38k comparison requires Stage 40 A-final, not selected-best"
        )
    if _STAGE_41.fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError(
            "Stage 42 requires the unchanged Complete-Shakespeare corpus and split"
        )
    source_config = replace(config, max_iters=DEFAULT_SOURCE_STEP)
    seeds = dict(
        training_batch_seed=DEFAULT_TRAINING_BATCH_SEED,
        fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED,
        training_rng_seed=DEFAULT_TRAINING_RNG_SEED,
    )
    prior = _STAGE_41.validate_and_load_stage_40_source(
        stage_40_path,
        data,
        source_config,
        N_HEAD,
        N_LAYER,
        device,
        inherited_best_path=None,
        **seeds,
    )
    checkpoint = _STAGE_41.validate_stage_41_checkpoint(
        source_path,
        data,
        source_config,
        N_HEAD,
        N_LAYER,
        _STAGE_41.FinalComputeExtensionSchedule(),
        prior,
        expected_kind="final",
        **seeds,
    )
    if checkpoint["best_step"] != DEFAULT_SOURCE_STEP or checkpoint[
        "snapshot_losses"
    ] != {"train": DEFAULT_SOURCE_TRAIN_LOSS, "validation": DEFAULT_SOURCE_VAL_LOSS}:
        raise ValueError("Stage 41 source losses or selected step changed")
    if file_hash(source_path) != DEFAULT_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("Stage 41 source changed while validating")
    source = ForkSource(
        source_path.resolve(),
        DEFAULT_SOURCE_CHECKPOINT_SHA256,
        dict(checkpoint),
        prior.runtime_device_type,
        prior.runtime_device_name,
        prior.torch_version,
    )
    return source, prior


def reload_source(source: ForkSource) -> ForkSource:
    if file_hash(source.path) != source.sha256:
        raise RuntimeError("common fork checkpoint changed")
    checkpoint = torch.load(source.path, map_location="cpu", weights_only=True)
    if file_hash(source.path) != source.sha256:
        raise RuntimeError("common fork checkpoint changed while loading")
    return replace(source, checkpoint=checkpoint)


def stream_hashes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, str]:
    return {
        "model_sha256": _STAGE_41.tensor_mapping_sha256(model.state_dict()),
        "optimizer_sha256": state_hash(optimizer.state_dict()),
        "training_generator_sha256": _STAGE_41.tensor_sha256(generator.get_state()),
        "torch_rng_sha256": state_hash(capture_rng_state(device)),
    }


def restore_fork(
    source: ForkSource,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
    learning_rate: float,
) -> dict[str, str]:
    FinalLearningRateSchedule(learning_rate)
    # The inherited restore routine loads model -> AdamW -> batch generator
    # -> global Torch RNG last. Nothing draws from dropout after this point.
    _STAGE_41.restore_stage_40_source(source, model, optimizer, generator, device)
    fork = stream_hashes(model, optimizer, generator, device)
    saved = source.checkpoint
    expected = {
        "model_sha256": _STAGE_41.tensor_mapping_sha256(saved["model_state_dict"]),
        "optimizer_sha256": state_hash(saved["optimizer_state_dict"]),
        "training_generator_sha256": _STAGE_41.tensor_sha256(
            saved["training_generator_state"]
        ),
        "torch_rng_sha256": state_hash(saved["rng_state"]),
    }
    if fork != expected:
        raise RuntimeError("full-state fork did not exactly reproduce the source")
    lineage = _STAGE_40.optimizer_lineage_sha256(optimizer.state_dict())
    _STAGE_40._set_optimizer_learning_rate(optimizer, learning_rate)
    if _STAGE_40.optimizer_lineage_sha256(optimizer.state_dict()) != lineage:
        raise RuntimeError("LR override changed non-LR optimizer state")
    return fork


@dataclass(frozen=True, slots=True)
class PairedValidationReport:
    eval_iters: int
    seed: int
    baseline: Any
    candidate: Any
    delta: Any
    final_generator_sha256: str


def evaluate_checkpoint_pair(
    baseline_path: Path,
    candidate_path: Path,
    baseline_hash: str,
    candidate_hash: str,
    baseline_name: str,
    candidate_name: str,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    seed: int,
    eval_iters: int = DEFAULT_FRESH_VALIDATION_ITERS,
) -> PairedValidationReport:
    if seed in (DEFAULT_FIXED_PANEL_SEED, _STAGE_40.DEFAULT_FRESH_VALIDATION_SEED):
        raise ValueError("Stage 42 fresh validation must use a new panel seed")
    caller_rng = capture_rng_state(device)
    models, payloads = [], []
    paths = (baseline_path, candidate_path)
    hashes = (baseline_hash, candidate_hash)
    try:
        for path, expected_hash in zip(paths, hashes, strict=True):
            if file_hash(path) != expected_hash:
                raise ValueError(f"fresh validation checkpoint hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = build_model(data, config, N_HEAD, N_LAYER, device)
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
        for name, path, sha256, payload, values in zip(
            (baseline_name, candidate_name),
            paths,
            hashes,
            payloads,
            losses,
            strict=True,
        ):
            if not all(math.isfinite(value) for value in values):
                raise ValueError("fresh validation produced a non-finite loss")
            if file_hash(path) != sha256:
                raise RuntimeError("checkpoint changed during fresh validation")
            mean, se = _STAGE_40._mean_and_standard_error(values)
            results.append(
                _STAGE_40.FreshCheckpointResult(
                    name,
                    path.resolve(),
                    sha256,
                    payload["step"],
                    payload["snapshot_losses"]["validation"],
                    mean,
                    se,
                    values,
                )
            )
        baseline, candidate = results
        return PairedValidationReport(
            eval_iters,
            seed,
            baseline,
            candidate,
            _STAGE_40._fresh_paired_delta(candidate, baseline),
            generator_hash,
        )
    finally:
        models.clear()
        _STAGE_40._restore_rng_state(caller_rng, device)
        _STAGE_41.clear_accelerator_cache(device)


def train_branch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    schedule: FinalLearningRateSchedule,
    on_evaluation: Any,
) -> TrainingResult:
    if config.max_iters <= DEFAULT_SOURCE_STEP:
        raise ValueError("target must follow the 44k fork")
    history = []

    def evaluate(step: int) -> None:
        losses = evaluate_on_fixed_batches(
            model, data, config, device, fixed_panel_seed=DEFAULT_FIXED_PANEL_SEED
        )
        if not math.isfinite(losses.train) or not math.isfinite(losses.val):
            raise ValueError(f"non-finite fixed-panel loss at step {step}")
        record = EvaluationRecord(step=step, losses=losses)
        on_evaluation(record)
        history.append(record)

    model.train()
    evaluate(DEFAULT_SOURCE_STEP)
    for update in range(DEFAULT_SOURCE_STEP, config.max_iters):
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


def checkpoint_payload(
    source: ForkSource,
    spec: BranchSpec,
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
    record: EvaluationRecord,
    best: EvaluationRecord,
    kind: str,
    fork: dict[str, str],
    fresh_history: PairedValidationReport | None,
) -> dict[str, Any]:
    saved = source.checkpoint
    training = copy.deepcopy(saved["training_config"])
    # Retain inherited boundaries in the schedule without stale Stage 41
    # treatment/compute fields on the new artifact.
    training.pop("continuation_learning_rate", None)
    training.pop("added_target_presentations_after_stage_40", None)
    training.update(
        learning_rate=spec.learning_rate,
        source_learning_rate=DEFAULT_CONTROL_LEARNING_RATE,
        branch_learning_rate=spec.learning_rate,
        max_iters=config.max_iters,
        target_step=config.max_iters,
        source_checkpoint_step=source.step,
        learning_rate_schedule=spec.schedule.as_metadata(target_step=config.max_iters),
        continuation_segment={
            "start_update": source.step,
            "end_update_exclusive": config.max_iters,
            "learning_rate": spec.learning_rate,
        },
        total_target_presentations=config.batch_size
        * config.block_size
        * config.max_iters,
        added_target_presentations_after_stage_41=(
            config.batch_size * config.block_size * (config.max_iters - source.step)
        ),
    )
    initialization = copy.deepcopy(saved["initialization"])
    initialization["kind"] = "restored_stage_41_terminal_complete_shakespeare_state"
    return {
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
        "optimizer_restart_step": None,
        "optimizer_provenance_known": True,
        "snapshot_losses": {
            "train": record.losses.train,
            "validation": record.losses.val,
        },
        "best_step": best.step,
        "best_val_loss": best.losses.val,
        "best_training_loss": best.losses.train,
        "architecture": copy.deepcopy(saved["architecture"]),
        "data_fingerprint": saved["data_fingerprint"],
        "corpus": copy.deepcopy(saved["corpus"]),
        "initialization": initialization,
        "training_config": training,
        "runtime": {
            "device_type": source.runtime_device_type,
            "device_name": source.runtime_device_name,
            "torch_version": source.torch_version,
            "matches_stage_41_source": True,
        },
        "source_checkpoint": {
            "stage": 41,
            "checkpoint_kind": "final",
            "step": source.step,
            "path_name": source.path.name,
            "sha256": source.sha256,
            "snapshot_training_loss": source.losses.train,
            "snapshot_validation_loss": source.losses.val,
            "model_state_restored": True,
            "optimizer_state_restored": True,
            "training_generator_restored": True,
            "cpu_rng_restored": True,
            "accelerator_rng_restored": device.type in ("xpu", "cuda"),
            "optimizer_restart": False,
            "restore_audit_verified_at_snapshot": True,
        },
        "fork_state": {"captured_before_lr_override": True, **fork},
        "experiment": {
            "stage": EXPERIMENT_STAGE,
            "protocol": PROTOCOL_NAME,
            "comparison_variable": "post_44000_learning_rate",
            "branch_key": spec.key,
            "trained_branches": 2,
            "optimizer_state_preserved": True,
            "optimizer_restarted": False,
            "rng_state_preserved": True,
            "architecture_frozen": True,
            "only_intended_post_fork_difference": "learning_rate",
        },
        "evaluation_protocol": {
            **copy.deepcopy(saved["evaluation_protocol"]),
            "selection_metric": "branch_fixed_panel_validation_loss",
            "branch_comparison_uses_matched_steps": True,
        },
        "fresh_validation_38k_vs_44k": None
        if fresh_history is None
        else plain_values(asdict(fresh_history)),
    }


def validate_branch_checkpoint(
    path: Path,
    source: ForkSource,
    spec: BranchSpec,
    config: TrainingConfig,
    *,
    kind: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") != 1 or payload.get("checkpoint_kind") != kind:
        raise ValueError("Stage 42 checkpoint kind/version mismatch")
    experiment = payload["experiment"]
    if (
        experiment.get("stage") != 42
        or experiment.get("protocol") != PROTOCOL_NAME
        or experiment.get("branch_key") != spec.key
    ):
        raise ValueError("Stage 42 checkpoint experiment/branch mismatch")
    if (
        payload["source_checkpoint"]["sha256"] != source.sha256
        or payload["source_checkpoint"]["step"] != DEFAULT_SOURCE_STEP
    ):
        raise ValueError("Stage 42 checkpoint source mismatch")
    if (
        payload.get("optimizer_restart_step") is not None
        or payload.get("optimizer_provenance_known") is not True
    ):
        raise ValueError("Stage 42 must preserve optimizer provenance")
    for key, expected in (
        ("optimizer_state_preserved", True),
        ("optimizer_restarted", False),
        ("rng_state_preserved", True),
        ("architecture_frozen", True),
        ("trained_branches", 2),
        ("comparison_variable", "post_44000_learning_rate"),
        ("only_intended_post_fork_difference", "learning_rate"),
    ):
        if experiment.get(key) != expected:
            raise ValueError(f"Stage 42 experiment field {key} mismatch")
    expected_fork = {
        "captured_before_lr_override": True,
        "model_sha256": _STAGE_41.tensor_mapping_sha256(
            source.checkpoint["model_state_dict"]
        ),
        "optimizer_sha256": state_hash(source.checkpoint["optimizer_state_dict"]),
        "training_generator_sha256": _STAGE_41.tensor_sha256(
            source.checkpoint["training_generator_state"]
        ),
        "torch_rng_sha256": state_hash(source.checkpoint["rng_state"]),
    }
    if payload["fork_state"] != expected_fork:
        raise ValueError("Stage 42 fork provenance mismatch")
    if payload["runtime"] != {
        "device_type": source.runtime_device_type,
        "device_name": source.runtime_device_name,
        "torch_version": source.torch_version,
        "matches_stage_41_source": True,
    }:
        raise ValueError("Stage 42 runtime mismatch")
    for key in ("architecture", "data_fingerprint", "corpus"):
        if payload[key] != source.checkpoint[key]:
            raise ValueError(f"Stage 42 changed {key}")
    training = payload["training_config"]
    if training["learning_rate_schedule"] != spec.schedule.as_metadata(
        target_step=config.max_iters
    ):
        raise ValueError("Stage 42 LR schedule mismatch")
    for key, value in (
        ("batch_size", config.batch_size),
        ("learning_rate", spec.learning_rate),
        ("branch_learning_rate", spec.learning_rate),
        ("max_iters", config.max_iters),
        ("eval_interval", config.eval_interval),
        ("eval_iters", config.eval_iters),
        ("training_batch_seed", DEFAULT_TRAINING_BATCH_SEED),
        ("fixed_panel_seed", DEFAULT_FIXED_PANEL_SEED),
        ("training_rng_seed", DEFAULT_TRAINING_RNG_SEED),
    ):
        if training.get(key) != value:
            raise ValueError(f"Stage 42 training field {key} mismatch")
    step, best_step = payload["step"], payload["best_step"]
    if not (DEFAULT_SOURCE_STEP <= best_step <= step <= config.max_iters):
        raise ValueError("Stage 42 snapshot/selection step mismatch")
    if (kind == "final" and step != config.max_iters) or (
        kind == "best" and step != best_step
    ):
        raise ValueError("Stage 42 checkpoint is at the wrong step")
    for value in (
        *payload["snapshot_losses"].values(),
        payload["best_val_loss"],
        payload["best_training_loss"],
    ):
        if not math.isfinite(value):
            raise ValueError("Stage 42 checkpoint contains non-finite losses")
    if payload["best_val_loss"] > payload["snapshot_losses"]["validation"]:
        raise ValueError("Stage 42 selected loss is worse than the snapshot")
    if kind == "best" and payload["snapshot_losses"] != {
        "train": payload["best_training_loss"],
        "validation": payload["best_val_loss"],
    }:
        raise ValueError("Stage 42 best checkpoint losses mismatch")
    model_state = payload["model_state_dict"]
    source_state = source.checkpoint["model_state_dict"]
    if model_state.keys() != source_state.keys():
        raise ValueError("Stage 42 model state keys changed")
    for key, value in model_state.items():
        if (
            value.shape != source_state[key].shape
            or value.dtype != source_state[key].dtype
        ):
            raise ValueError(f"Stage 42 model tensor {key} changed shape or dtype")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Stage 42 model tensor {key} is non-finite")
    optimizer = _STAGE_40._validate_adamw_state(
        payload["optimizer_state_dict"], expected_learning_rate=spec.learning_rate
    )
    _STAGE_40._validate_optimizer_steps(optimizer, expected_step=step)
    _STAGE_40._validate_optimizer_hyperparameters(optimizer)
    expected_generator = torch.Generator()
    expected_generator.set_state(source.checkpoint["training_generator_state"])
    # Exactly four T512 panels per B8/T256 update; this is the same draw
    # consumed by the inherited sampler, without allocating training batches.
    train_characters = source.checkpoint["corpus"]["train_characters"]
    for _ in range(source.step, step):
        torch.randint(
            0,
            train_characters - 2 * config.block_size,
            (config.batch_size // 2,),
            generator=expected_generator,
        )
    if not torch.equal(
        payload["training_generator_state"], expected_generator.get_state()
    ):
        raise ValueError("Stage 42 checkpoint batch stream mismatch")
    _STAGE_40._validate_rng_state(
        payload["rng_state"], runtime_device_type=source.runtime_device_type
    )
    return payload


@dataclass(frozen=True, slots=True)
class BranchReport:
    spec: BranchSpec
    training: TrainingResult
    best_step: int
    best_losses: LossEstimate
    fork: dict[str, str]
    final_state: dict[str, str]
    best_sha256: str
    final_sha256: str


def run_branch(
    source: ForkSource,
    spec: BranchSpec,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    fresh_history: PairedValidationReport | None,
) -> BranchReport:
    source = reload_source(source)
    _STAGE_41.clear_accelerator_cache(device)
    model = build_model(data, config, N_HEAD, N_LAYER, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_CONTROL_LEARNING_RATE)
    generator = torch.Generator()
    fork = restore_fork(source, model, optimizer, generator, device, spec.learning_rate)
    best = EvaluationRecord(step=source.step, losses=source.losses)
    source_audited = False

    def save(record: EvaluationRecord, kind: str, path: Path) -> None:
        if not source_audited:
            raise RuntimeError("cannot save a branch before source loss audit")
        payload = checkpoint_payload(
            source,
            spec,
            config,
            model,
            optimizer,
            generator,
            device,
            record,
            best,
            kind,
            fork,
            fresh_history,
        )
        _STAGE_40._atomic_save_payload(path, payload)

    def on_evaluation(record: EvaluationRecord) -> None:
        nonlocal best, source_audited
        if record.step == source.step:
            if not all(
                _STAGE_41._reported_value_matches(observed, expected)
                for observed, expected in (
                    (record.losses.train, source.losses.train),
                    (record.losses.val, source.losses.val),
                )
            ):
                raise RuntimeError(
                    "44k source fixed-panel audit failed before the first update"
                )
            if (
                stream_hashes(model, optimizer, generator, device)["torch_rng_sha256"]
                != fork["torch_rng_sha256"]
            ):
                raise RuntimeError("fixed evaluation advanced dropout RNG")
            source_audited = True
            # Source is also the selected best at 44k. Materialize an exact
            # branch-specific fallback even if no later validation improves.
            save(best, "best", spec.best_path)
        elif record.losses.val < best.losses.val:
            best = record
            save(record, "best", spec.best_path)
        print(
            f"  {spec.key} step {record.step:5d} | lr {spec.learning_rate:g} | "
            f"train {record.losses.train:.4f} | val {record.losses.val:.4f} | "
            f"best {best.losses.val:.4f} @ {best.step}",
            flush=True,
        )
        if record.step == config.max_iters:
            save(record, "final", spec.final_path)

    result = train_branch(
        model, optimizer, generator, data, config, device, spec.schedule, on_evaluation
    )
    final_state = stream_hashes(model, optimizer, generator, device)
    for kind, path in (("best", spec.best_path), ("final", spec.final_path)):
        payload = validate_branch_checkpoint(path, source, spec, config, kind=kind)
        if kind == "final":
            for key, observed in (
                (
                    "model_sha256",
                    _STAGE_41.tensor_mapping_sha256(payload["model_state_dict"]),
                ),
                ("optimizer_sha256", state_hash(payload["optimizer_state_dict"])),
                (
                    "training_generator_sha256",
                    _STAGE_41.tensor_sha256(payload["training_generator_state"]),
                ),
                ("torch_rng_sha256", state_hash(payload["rng_state"])),
            ):
                if final_state[key] != observed:
                    raise RuntimeError(f"terminal checkpoint failed {key} round trip")
    if file_hash(source.path) != source.sha256:
        raise RuntimeError("common source changed during training")
    report = BranchReport(
        spec,
        result,
        best.step,
        best.losses,
        fork,
        final_state,
        file_hash(spec.best_path),
        file_hash(spec.final_path),
    )
    return report


def verify_paired_branches(control: BranchReport, treatment: BranchReport) -> None:
    if control.fork != treatment.fork:
        raise RuntimeError("branches did not restore identical pre-LR states")
    for key in ("training_generator_sha256", "torch_rng_sha256"):
        if control.final_state[key] != treatment.final_state[key]:
            raise RuntimeError(f"branches consumed different {key} streams")
    if control.training.initial != treatment.training.initial:
        raise RuntimeError("branches did not reproduce the same initial losses")
    if tuple(r.step for r in control.training.history) != tuple(
        r.step for r in treatment.training.history
    ):
        raise RuntimeError("branches have mismatched update/evaluation budgets")


def paired_outcome(report: PairedValidationReport) -> str:
    if report.delta.confidence_high < 0:
        return "1e-5 wins on the fresh endpoint panel; a later stage can test refinement to plateau"
    if report.delta.confidence_low > 0:
        return "3e-5 wins on the fresh endpoint panel; stop this experiment at 50k and revisit capacity"
    return "unresolved within fresh paired uncertainty; stop at 50k (this does not prove equivalence)"


def print_pair(report: PairedValidationReport) -> None:
    print(f"  {report.baseline.name}: {report.baseline.mean_loss:.7f}")
    print(f"  {report.candidate.name}: {report.candidate.mean_loss:.7f}")
    delta = report.delta
    print(
        f"  paired candidate - baseline: {delta.mean_delta:+.7f}; "
        f"SE {delta.standard_error:.7f}; "
        f"95% panel interval [{delta.confidence_low:+.7f}, {delta.confidence_high:+.7f}]",
        flush=True,
    )


def write_report(path: Path, report: dict[str, Any]) -> None:
    # Report is separate from resumable snapshots; endpoint validation never
    # alters training RNG or selects new weights from the fixed-panel winners.
    text = json.dumps(plain_values(report), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(text)


def plain_values(value: Any) -> Any:
    """Keep reports JSON-safe and embedded metadata weights_only-loadable."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: plain_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_values(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 42: exact 44k -> 50k LR fork, 3e-5 vs 1e-5"
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "xpu")
    )
    parser.add_argument(
        "--data", type=Path, default=_STAGE_41.DEFAULT_COMPLETE_DATA_PATH
    )
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT_PATH
    )
    parser.add_argument(
        "--stage-40-checkpoint", type=Path, default=DEFAULT_STAGE_40_CHECKPOINT_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="authenticate sources and runtime without training",
    )
    mode.add_argument(
        "--fresh-validation-only",
        action="store_true",
        help="run only the 500-batch 38k vs 44k comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = canonical_config()
    data = CharacterData.from_file(
        args.data, train_fraction=_STAGE_41.DEFAULT_TRAIN_FRACTION
    )
    device = resolve_device(args.device)
    control_spec, treatment_spec = branch_specs(args.output_dir)
    report_path = args.output_dir / (
        "stage_42_fresh_38k_vs_44k.json"
        if args.fresh_validation_only
        else "stage_42_report.json"
    )
    outputs = () if args.validate_only else (report_path,)
    if not (args.validate_only or args.fresh_validation_only):
        outputs += (
            control_spec.best_path,
            control_spec.final_path,
            treatment_spec.best_path,
            treatment_spec.final_path,
        )
    validate_artifact_paths(
        (args.data, args.source_checkpoint, args.stage_40_checkpoint), outputs
    )
    source, prior = load_sources(
        args.source_checkpoint, args.stage_40_checkpoint, data, config, device
    )
    print(
        f"Stage 42: authenticated {source.step:,} source on {device}, SHA-256 {source.sha256}",
        flush=True,
    )
    if args.validate_only:
        print(
            "Source model, AdamW, batch generator, dropout RNG, corpus, and runtime validated."
        )
        return

    print(
        "Fresh paired Stage 40 A @ 38k vs Stage 41 @ 44k (500 shared batches)",
        flush=True,
    )
    fresh_history = evaluate_checkpoint_pair(
        prior.path,
        source.path,
        prior.sha256,
        source.sha256,
        "Stage 40 A @ 38k",
        "Stage 41 @ 44k",
        data,
        config,
        device,
        seed=DEFAULT_HISTORY_VALIDATION_SEED,
    )
    print_pair(fresh_history)
    report = {
        "stage": 42,
        "protocol": PROTOCOL_NAME,
        "source_path": source.path,
        "source_sha256": source.sha256,
        "fresh_38k_vs_44k": asdict(fresh_history),
        "uncertainty_scope": "Student-t 95% panel-sampling interval conditional on checkpoints; not training-seed or corpus-IID uncertainty",
    }
    if args.fresh_validation_only:
        report["training_executed"] = False
        write_report(report_path, report)
        return

    control = run_branch(source, control_spec, data, config, device, fresh_history)
    treatment = run_branch(source, treatment_spec, data, config, device, fresh_history)
    verify_paired_branches(control, treatment)
    print(
        "Fresh paired A vs B at the matched 50k endpoint (500 shared batches)",
        flush=True,
    )
    fresh_branches = evaluate_checkpoint_pair(
        control.spec.final_path,
        treatment.spec.final_path,
        control.final_sha256,
        treatment.final_sha256,
        "A 3e-5 @ 50k",
        "B 1e-5 @ 50k",
        data,
        config,
        device,
        seed=DEFAULT_BRANCH_VALIDATION_SEED,
    )
    print_pair(fresh_branches)
    fixed_delta = treatment.training.final.val - control.training.final.val
    best_delta = treatment.best_losses.val - control.best_losses.val
    outcome = paired_outcome(fresh_branches)
    report.update(
        training_executed=True,
        paired_streams_verified=True,
        control=asdict(control),
        treatment=asdict(treatment),
        fresh_50k_branches=asdict(fresh_branches),
        final_fixed_panel_delta_b_minus_a=fixed_delta,
        selected_best_fixed_panel_delta_b_minus_a=best_delta,
        outcome=outcome,
    )
    write_report(report_path, report)
    print(f"Fixed final B-A: {fixed_delta:+.7f}; selected-best B-A: {best_delta:+.7f}")
    print(f"Paired batch and dropout streams: verified. {outcome}.")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
