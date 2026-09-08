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
    "stage_42_for_stage_43", Path(__file__).with_name("42_final_learning_rate_fork.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE_42 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE_42
_SPEC.loader.exec_module(_STAGE_42)
_STAGE_41 = _STAGE_42._STAGE_41
_STAGE_40 = _STAGE_42._STAGE_40
_STAGE_38 = _STAGE_40._STAGE_39._STAGE_38

# Reuse architecture, panel geometry, and statistics, never a historical state.
build_model = _STAGE_42.build_model
get_batch = _STAGE_42.get_batch
evaluate_on_fixed_batches = _STAGE_42.evaluate_on_fixed_batches
capture_rng_state = _STAGE_42.capture_rng_state
state_hash = _STAGE_42.state_hash
file_hash = _STAGE_42.file_hash
fingerprint_data = _STAGE_41.fingerprint_data

EXPERIMENT_STAGE = 43
PROTOCOL_NAME = "paired_architecture_native_width_scaling"
DEFAULT_CONTROL_WIDTH = 128
DEFAULT_TREATMENT_WIDTH = 192
DEFAULT_TARGET_STEP = 50_000
DEFAULT_FRESH_VALIDATION_ITERS = 500
DEFAULT_FRESH_VALIDATION_SEED = 1_357
DEFAULT_FIXED_PANEL_SEED = _STAGE_42.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_42.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_42.DEFAULT_TRAINING_RNG_SEED
DEFAULT_OUTPUT_DIRECTORY = _STAGE_41.CHECKPOINT_DIRECTORY
DEFAULT_COMPLETE_DATA_PATH = _STAGE_41.DEFAULT_COMPLETE_DATA_PATH
N_HEAD = _STAGE_42.N_HEAD
N_LAYER = _STAGE_42.N_LAYER
INITIALIZATION_POLICY = (
    "Architecture-native initializations recreated from the same seed; "
    "their tensors are not claimed byte-identical."
)
DROPOUT_POLICY = (
    "Both branches begin from the same CPU and accelerator training RNG states, "
    "reset after native initialization. Different widths can subsequently "
    "consume different draws, including before later equal-shape calls. Neither identical "
    "dropout masks nor identical terminal Torch RNG states are claimed."
)
UNCERTAINTY_SCOPE = (
    "Student-t 95% panel-sampling interval conditional on these checkpoints; "
    "not training-seed or corpus-IID uncertainty."
)
# Provenance references only. Stage 43 never opens or restores these artifacts.
HISTORICAL_LINEAGE = {
    "stage_41_authenticated_44k_ancestor_sha256": _STAGE_42.DEFAULT_SOURCE_CHECKPOINT_SHA256,
    "stage_42_empirical_best": {
        "branch": "control_3e_5",
        "step": 50_000,
        "best_checkpoint_sha256": "183be907a891dd84fc31430c66c8313e70d38c87a278098cebe219c9904a49a6",
        "final_checkpoint_sha256": "7bfc120595760b538b9c4063e3580525b48762bcc779265cb09ba09e571a1441",
        "fixed_panel_train_loss": 1.1956539690,
        "fixed_panel_validation_loss": 1.4158585346,
    },
    "role": "historical reference only; neither is a Stage 43 initialization",
}


def canonical_config(width: int = DEFAULT_CONTROL_WIDTH) -> TrainingConfig:
    if width not in (DEFAULT_CONTROL_WIDTH, DEFAULT_TREATMENT_WIDTH):
        raise ValueError("Stage 43 canonical widths are 128 and 192")
    return replace(_STAGE_42.canonical_config(), n_embd=width, learning_rate=1e-3)


def make_schedule() -> Any:
    # The winning lineage kept 3e-5 continuously from update 18k through 50k.
    return _STAGE_38.ExtendedLearningRateSchedule()


def schedule_metadata(target_step: int) -> list[dict[str, int | float]]:
    # Clip phases for small internal smoke runs; production is locked to 50k.
    return [
        {
            **phase,
            "end_update_exclusive": min(target_step, phase["end_update_exclusive"]),
        }
        for phase in make_schedule().as_metadata(target_step=DEFAULT_TARGET_STEP)
        if phase["start_update"] < target_step
    ]


@dataclass(frozen=True, slots=True)
class BranchSpec:
    key: str
    width: int
    best_path: Path
    final_path: Path

    def __post_init__(self) -> None:
        if self.key not in ("control", "treatment"):
            raise ValueError("Stage 43 branch must be control or treatment")
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise ValueError("width must be an integer")
        if self.width <= 0 or self.width % N_HEAD:
            raise ValueError("width must be positive and divisible by four heads")


def branch_specs(directory: Path) -> tuple[BranchSpec, BranchSpec]:
    return tuple(
        BranchSpec(
            key,
            width,
            directory / f"stage_43_{key}_c{width}_best_checkpoint.pt",
            directory / f"stage_43_{key}_c{width}_step_50000_final_checkpoint.pt",
        )
        for key, width in (
            ("control", DEFAULT_CONTROL_WIDTH),
            ("treatment", DEFAULT_TREATMENT_WIDTH),
        )
    )


def validate_protocol(data: CharacterData, configs: tuple[TrainingConfig, ...]) -> None:
    if configs != (canonical_config(), canonical_config(DEFAULT_TREATMENT_WIDTH)):
        raise ValueError("Stage 43 locks all settings except C128 versus C192")
    if fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Stage 43 requires unchanged Complete Shakespeare and split")


def validate_artifact_paths(
    inputs: tuple[Path, ...], outputs: tuple[Path, ...]
) -> None:
    paths = [path.resolve() for path in (*inputs, *outputs)]
    if len(paths) != len(set(paths)):
        raise ValueError("corpus, checkpoints, and reports must have distinct paths")
    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Stage 43 output exists; use a new --output-dir: {path}"
            )
        if path.with_name(f".{path.name}.tmp").exists():
            raise FileExistsError(f"Stage 43 temporary output exists: {path}")


def architecture_metadata(
    data: CharacterData, config: TrainingConfig
) -> dict[str, Any]:
    width = config.n_embd
    return {
        "n_embd": width,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "head_size": width // N_HEAD,
        "feed_forward_width": 4 * width,
        "block_size": config.block_size,
        "residual_dropout": 0.10,
        "layer_contexts": (23, 23, 23, 23, None, None),
        "topology": "LLLLGG",
        "vocabulary_size": data.vocabulary.size,
    }


def initialize_branch(
    data: CharacterData,
    config: TrainingConfig,
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
    model = build_model(data, config, N_HEAD, N_LAYER, torch.device("cpu"))
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
        "origin_stage": 43,
        "step": 0,
        "seed": config.seed,
        "device": "cpu",
        "state_sha256": initial_hash,
        "policy": INITIALIZATION_POLICY,
        "cross_width_tensor_identity_claimed": False,
        "historical_weights_loaded": False,
        "historical_optimizer_loaded": False,
        "optimizer_state_entries_at_initialization": len(optimizer.state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, optimizer, generator, initialization


def stream_hashes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, str]:
    return _STAGE_42.stream_hashes(model, optimizer, generator, device)


def train_branch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    on_evaluation: Any,
) -> TrainingResult:
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
        "architecture": architecture_metadata(data, config),
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
            "stage": 43,
            "protocol": PROTOCOL_NAME,
            "comparison_variable": "model_width",
            "branch_key": spec.key,
            "from_scratch": True,
            "trained_branches": 2,
            "canonical_protocol": (
                config == canonical_config(spec.width)
                and fingerprint_data(data)
                == _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT
            )
            if spec.width in (128, 192)
            else False,
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
        raise ValueError("Stage 43 checkpoint must be a mapping")
    digest = payload.pop("payload_sha256", None)
    if digest != state_hash(payload):
        raise ValueError("Stage 43 checkpoint payload integrity mismatch")
    payload["payload_sha256"] = digest
    if payload["checkpoint_version"] != 1 or payload["checkpoint_kind"] != kind:
        raise ValueError("Stage 43 checkpoint version or kind mismatch")
    if (
        payload["experiment"]["stage"] != 43
        or payload["experiment"]["protocol"] != PROTOCOL_NAME
    ):
        raise ValueError("Stage 43 experiment mismatch")
    if payload["experiment"]["branch_key"] != spec.key or config.n_embd != spec.width:
        raise ValueError("Stage 43 checkpoint branch mismatch")
    if payload["architecture"] != architecture_metadata(data, config):
        raise ValueError("Stage 43 architecture changed")
    if payload["data_fingerprint"] != fingerprint_data(data):
        raise ValueError("Stage 43 corpus or split changed")
    for key, value in asdict(config).items():
        if payload["training_config"].get(key) != value:
            raise ValueError(f"Stage 43 training config changed: {key}")
    if payload["training_config"]["learning_rate_schedule"] != schedule_metadata(
        config.max_iters
    ):
        raise ValueError("Stage 43 learning rate schedule changed")
    step, best_step = payload["step"], payload["best_step"]
    if (
        type(step) is not int
        or type(best_step) is not int
        or not 0 <= best_step <= step <= config.max_iters
    ):
        raise ValueError("Stage 43 checkpoint steps are invalid")
    if kind == "final" and step != config.max_iters:
        raise ValueError("Stage 43 final must be the matched terminal step")
    if kind == "best" and step != best_step:
        raise ValueError("Stage 43 best must contain its selected weights")
    if not all(
        math.isfinite(value)
        for value in (
            *payload["snapshot_losses"].values(),
            payload["best_val_loss"],
            payload["best_training_loss"],
        )
    ):
        raise ValueError("Stage 43 losses must be finite")
    initialization = payload["initialization"]
    if (
        initialization["kind"] != "fresh_architecture_native"
        or initialization["origin_stage"] != 43
        or initialization["step"] != 0
        or initialization["seed"] != config.seed
        or initialization["historical_weights_loaded"]
        or initialization["historical_optimizer_loaded"]
        or initialization["optimizer_state_entries_at_initialization"] != 0
        or payload["optimizer_restart_step"] != 0
    ):
        raise ValueError("Stage 43 requires fresh model and optimizer initialization")
    optimizer_state = payload["optimizer_state_dict"]
    groups, states = optimizer_state["param_groups"], optimizer_state["state"]
    if len(groups) != 1:
        raise ValueError("Stage 43 requires one AdamW parameter group")
    group = groups[0]
    if (
        group["lr"] != make_schedule().for_update(max(0, step - 1))
        or group["betas"] != (0.9, 0.999)
        or group["weight_decay"] != 0.01
        or group["eps"] != 1e-8
        or group["amsgrad"]
    ):
        raise ValueError("Stage 43 AdamW settings changed")
    if step == 0:
        if states:
            raise ValueError("Stage 43 step-zero optimizer must be empty")
    elif set(states) != set(group["params"]) or any(
        state["step"].item() != step for state in states.values()
    ):
        raise ValueError("Stage 43 AdamW counters do not match completed updates")
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


def run_branch(
    spec: BranchSpec, data: CharacterData, config: TrainingConfig, device: torch.device
) -> BranchReport:
    """Run from zero. Reduced configs are for internal tests, not the CLI protocol."""
    if spec.width != config.n_embd:
        raise ValueError("branch width does not match its configuration")
    validate_artifact_paths((), (spec.best_path, spec.final_path))
    _STAGE_41.clear_accelerator_cache(device)
    model, optimizer, generator, initialization = initialize_branch(
        data, config, device
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
            f"  {spec.key} C{spec.width} step {record.step:5d} | "
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
                    "Stage 43 terminal state failed checkpoint round trip"
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


def verify_paired_branches(control: BranchReport, treatment: BranchReport) -> None:
    if control.spec.key != "control" or treatment.spec.key != "treatment":
        raise RuntimeError("paired comparison requires control followed by treatment")
    if treatment.spec.width * 2 != control.spec.width * 3:
        raise RuntimeError("paired width treatment must be exactly 1.5 times control")
    for key in ("training_generator_sha256", "torch_rng_sha256"):
        if control.initial_streams[key] != treatment.initial_streams[key]:
            raise RuntimeError(f"paired branches started with different {key}")
    if (
        control.final_streams["training_generator_sha256"]
        != treatment.final_streams["training_generator_sha256"]
    ):
        raise RuntimeError("paired branches consumed different training batches")
    if [record.step for record in control.training.history] != [
        record.step for record in treatment.training.history
    ]:
        raise RuntimeError("paired update/evaluation budgets differ")
    # Different widths need not have equal initial losses or terminal dropout RNG.


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
    if seed in (DEFAULT_FIXED_PANEL_SEED, 1354, 1355, 1356):
        raise ValueError("Stage 43 fresh validation must use an unused panel seed")
    if eval_iters < 2:
        raise ValueError("paired uncertainty requires at least two batches")
    verify_paired_branches(control, treatment)
    caller_rng = capture_rng_state(device)
    models, payloads = [], []
    try:
        for branch in (control, treatment):
            if file_hash(branch.spec.final_path) != branch.final_sha256:
                raise ValueError("Stage 43 endpoint checkpoint hash mismatch")
            branch_config = replace(config, n_embd=branch.spec.width)
            payload = validate_branch_checkpoint(
                branch.spec.final_path,
                branch.spec,
                data,
                branch_config,
                "final",
            )
            model = build_model(data, branch_config, N_HEAD, N_LAYER, device)
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
                raise ValueError("Stage 43 fresh validation produced non-finite loss")
            if file_hash(branch.spec.final_path) != branch.final_sha256:
                raise RuntimeError("Stage 43 endpoint changed during evaluation")
            mean, se = _STAGE_40._mean_and_standard_error(values)
            results.append(
                _STAGE_40.FreshCheckpointResult(
                    f"{branch.spec.key} C{branch.spec.width} @ {payload['step']}",
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
            "C192 improves validation at matched 50k updates on the fresh paired panel"
        )
    if report.delta.confidence_low > 0:
        return "C128 wins at matched 50k updates under the established schedule"
    return "unresolved within paired panel uncertainty; this does not prove equivalence"


def preflight(
    data: CharacterData, configs: tuple[TrainingConfig, ...], device: torch.device
) -> dict[str, Any]:
    validate_protocol(data, configs)
    caller_rng = capture_rng_state(device)
    audits = []
    try:
        for config in configs:
            model, optimizer, generator, initialization = initialize_branch(
                data, config, device
            )
            audits.append(
                {
                    "architecture": architecture_metadata(data, config),
                    "initialization": initialization,
                    "initial_streams": stream_hashes(
                        model, optimizer, generator, device
                    ),
                }
            )
            del model, optimizer, generator
        for key in ("training_generator_sha256", "torch_rng_sha256"):
            if audits[0]["initial_streams"][key] != audits[1]["initial_streams"][key]:
                raise RuntimeError("preflight branch random streams do not match")
        return {
            "stage": 43,
            "protocol": PROTOCOL_NAME,
            "training_executed": False,
            "branches": audits,
            "updates_per_branch": DEFAULT_TARGET_STEP,
            "target_presentations_per_branch": DEFAULT_TARGET_STEP * 8 * 256,
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
        description="Stage 43: fresh C128 vs C192, paired 50k-update width experiment",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = CharacterData.from_file(args.data, train_fraction=0.9)
    device = resolve_device(args.device)
    configs = (canonical_config(), canonical_config(DEFAULT_TREATMENT_WIDTH))
    specs = branch_specs(args.output_dir)
    report_path = args.output_dir / "stage_43_report.json"
    outputs = (
        report_path,
        *(path for spec in specs for path in (spec.best_path, spec.final_path)),
    )
    validate_artifact_paths((args.data,), () if args.validate_only else outputs)
    audit = preflight(data, configs, device)
    print(json.dumps(audit, indent=2, allow_nan=False), flush=True)
    if args.validate_only:
        return
    print(
        f"Stage 43 on {device}: native C128/C192, 50,000 updates per branch", flush=True
    )
    control = run_branch(specs[0], data, configs[0], device)
    treatment = run_branch(specs[1], data, configs[1], device)
    verify_paired_branches(control, treatment)
    fresh = evaluate_checkpoint_pair(control, treatment, data, configs[0], device)
    _STAGE_42.print_pair(fresh)
    outcome = paired_outcome(fresh)
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
        "parameter_count_ratio": treatment.initialization["parameter_count"]
        / control.initialization["parameter_count"],
        "outcome": outcome,
        "budget_interpretation": "equal updates and target tokens; FLOPs and wall time differ",
    }
    _STAGE_42.write_report(report_path, report)
    print(f"{outcome}. Report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
