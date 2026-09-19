from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from config import TrainingConfig
from data_utils import CharacterData
from train import resolve_device
from training import EvaluationRecord, LossEstimate, TrainingResult


_SPEC = importlib.util.spec_from_file_location(
    "stage_48_for_stage_49", Path(__file__).with_name("48_dropout_strength.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE_48 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE_48
_SPEC.loader.exec_module(_STAGE_48)
_STAGE_47 = _STAGE_48._STAGE_47
_STAGE_46 = _STAGE_48._STAGE_46
_STAGE_45 = _STAGE_48._STAGE_45
_STAGE_44 = _STAGE_48._STAGE_44
_STAGE_42 = _STAGE_48._STAGE_42
_STAGE_41 = _STAGE_48._STAGE_41
_STAGE_40 = _STAGE_48._STAGE_40

# The architecture owner for this lineage: Stage 46 rebuilt the LLLLGG T=256
# factory of Stage 36 with residual dropout exposed as a parameter. Stages 47
# and 48 re-used it unchanged at C320 and C256; Stage 49 re-uses it again at
# C256, because the two probabilities under test are branch parameters that
# never enter the factory itself.
build_model = _STAGE_48.build_model
dropout_architecture_metadata = _STAGE_48.dropout_architecture_metadata
dropout_label = _STAGE_48.dropout_label

# Reuse the sampler, panel geometry, schedule, and statistics unchanged.
# Only the two residual dropout probabilities and the artifact identity change,
# so this stage cannot drift from the Stage 43-48 training protocol.
get_batch = _STAGE_48.get_batch
evaluate_on_fixed_batches = _STAGE_48.evaluate_on_fixed_batches
capture_rng_state = _STAGE_48.capture_rng_state
state_hash = _STAGE_48.state_hash
file_hash = _STAGE_48.file_hash
fingerprint_data = _STAGE_48.fingerprint_data
stream_hashes = _STAGE_48.stream_hashes
make_schedule = _STAGE_48.make_schedule
schedule_metadata = _STAGE_48.schedule_metadata

EXPERIMENT_STAGE = 49
PROTOCOL_NAME = "paired_architecture_native_regularization"
DEFAULT_WIDTH = 256
DEFAULT_CONTROL_DROPOUT = 0.15
DEFAULT_TREATMENT_DROPOUT = 0.20
DEFAULT_TARGET_STEP = 50_000
DEFAULT_FRESH_VALIDATION_ITERS = 500
# Every seed from 1,354 through 1,365 was consumed by Stages 43-48.
DEFAULT_FRESH_VALIDATION_SEED = 1_366
DEFAULT_CHAMPION_PANEL_SEED = 1_367
DEFAULT_FIXED_PANEL_SEED = _STAGE_44.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_44.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_44.DEFAULT_TRAINING_RNG_SEED
DEFAULT_OUTPUT_DIRECTORY = _STAGE_41.CHECKPOINT_DIRECTORY
DEFAULT_COMPLETE_DATA_PATH = _STAGE_41.DEFAULT_COMPLETE_DATA_PATH
# Every branch of this lineage (Stages 43-48) was trained on one accelerator, and
# its recorded trajectories are only reproducible there: kernel selection depends
# on the device, so the same seed on a different GPU yields a different curve.
# This stage's p=0.15 control must reproduce the Stage 48 treatment exactly, so
# the canonical protocol requires the recorded device rather than any CUDA
# device. The host now exposes two GPUs and `cuda:0` is not the recorded one.
DEFAULT_LINEAGE_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
# The native initialization is dropout-invariant, so Stage 45's recorded C256
# hash is also this stage's p=0.15 control hash and its p=0.20 treatment hash.
# Preflight rebuilds it through the Stage 46 factory, through Stage 45's
# unchanged factory, and requires both to match this constant.
C256_NATIVE_INITIALIZATION_SHA256 = (
    "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e"
)
# A read-only cross-stage reference: Stage 48's fresh C256 p=0.15 treatment is
# the state this stage's p=0.15 control must re-derive from zero, and the stage
# the new treatment step is measured from. The stored terminal state is the
# target the trained control must reproduce exactly.
DEFAULT_STAGE_48_TREATMENT_FILENAME = (
    "stage_48_treatment_c256_p015_step_50000_final_checkpoint.pt"
)
DEFAULT_STAGE_48_TREATMENT_SHA256 = (
    "04952aeeab517f2a31ba4209737d4f8d495d40fb84aca8c9b5f3b58bb44f98b4"
)
DEFAULT_STAGE_48_TREATMENT_BEST_SHA256 = (
    "62d6b5ee7d9009cafd48561e5622c9a1e7b3254e3d26fbfc57c280662ce723ab"
)
DEFAULT_STAGE_48_TREATMENT_BEST_STEP = 48_500
DEFAULT_STAGE_48_TREATMENT_FIXED_PANEL_TRAIN_LOSS = 1.087766051888466
DEFAULT_STAGE_48_TREATMENT_FIXED_PANEL_VALIDATION_LOSS = 1.3507258701324463
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
# Provenance references only. Stage 49 never initializes from these artifacts;
# the two C320 endpoints are read once, read-only, as champion shootout peers.
# The Stage 43-48 entries are inherited verbatim from the stage this one extends.
HISTORICAL_LINEAGE = {
    **copy.deepcopy(_STAGE_48.HISTORICAL_LINEAGE),
    "stage_48_c256_control_p010_endpoint": {
        "branch": "control_c256",
        "dropout": _STAGE_48.DEFAULT_CONTROL_DROPOUT,
        "step": 50_000,
        "best_step": 48_500,
        "final_checkpoint_sha256": (
            "8c817fd5ea8a557e60f693a577cad68bbd9d483402d8e90fe343b23590c8090e"
        ),
        "fixed_panel_train_loss": 1.0690209597349167,
        "fixed_panel_validation_loss": 1.3538347363471985,
        "role": "the C256 endpoint Stage 48 displaced; not read by this stage",
    },
    "stage_48_c256_treatment_p015_endpoint": {
        "branch": "treatment_c256",
        "dropout": _STAGE_48.DEFAULT_TREATMENT_DROPOUT,
        "step": 50_000,
        "best_step": DEFAULT_STAGE_48_TREATMENT_BEST_STEP,
        "best_checkpoint_sha256": DEFAULT_STAGE_48_TREATMENT_BEST_SHA256,
        "final_checkpoint_sha256": DEFAULT_STAGE_48_TREATMENT_SHA256,
        "fixed_panel_train_loss": DEFAULT_STAGE_48_TREATMENT_FIXED_PANEL_TRAIN_LOSS,
        "fixed_panel_validation_loss": (
            DEFAULT_STAGE_48_TREATMENT_FIXED_PANEL_VALIDATION_LOSS
        ),
        "role": "the C256 p=0.15 state this stage's control must re-derive",
    },
    "role": "historical reference only; none is a Stage 49 initialization",
}
# The prior dropout steps, transcribed from the stage reports that measured them:
# the first C256 step (Stage 48) and both C320 steps (Stages 46 and 47). They are
# the other rows of the capacity-versus-regularization curve that this stage's
# own row is compared against; Stage 49 never retrains them.
PRIOR_DROPOUT_STEPS = (
    {
        "width": 256,
        "dropout_step": "p=0.1 -> p=0.15",
        "source_stage": 48,
        "fresh_panel_seed": 1_364,
        "mean_delta": -0.0015666978359222413,
        "standard_error": 0.0006895126702513343,
        "confidence_low": -0.0029214036446057468,
        "confidence_high": -0.0002119920272387356,
    },
    {
        "width": 320,
        "dropout_step": "p=0.1 -> p=0.15",
        "source_stage": 46,
        "fresh_panel_seed": 1_360,
        "mean_delta": -0.0050533,
        "standard_error": 0.0006239,
        "confidence_low": -0.0062791,
        "confidence_high": -0.0038274,
    },
    {
        "width": 320,
        "dropout_step": "p=0.15 -> p=0.2",
        "source_stage": 47,
        "fresh_panel_seed": 1_362,
        "mean_delta": 0.0000117,
        "standard_error": 0.0007087,
        "confidence_low": -0.0013807,
        "confidence_high": 0.0014042,
    },
)
# The two stored C320 endpoints that enter the champion shootout. Each is
# authenticated by its owning stage's validator and by its recorded file hash,
# terminal step, and fixed-panel losses before it is scored.
C320_CHAMPION_REFS = (
    {
        "key": "c320_p015",
        "owner": "stage_46",
        "width": 320,
        "dropout": 0.15,
        "filename": "stage_46_treatment_c320_p015_step_50000_final_checkpoint.pt",
        "file_sha256": (
            "e47016371b6a588448dbaa0bad6d05ff8564b7b21c6f6da81f20bbb56e916197"
        ),
        "best_step": 49_500,
        "fixed_panel_train_loss": 1.0690791565179825,
        "fixed_panel_validation_loss": 1.3597860205173493,
    },
    {
        "key": "c320_p020",
        "owner": "stage_47",
        "width": 320,
        "dropout": 0.20,
        "filename": "stage_47_treatment_c320_p020_step_50000_final_checkpoint.pt",
        "file_sha256": (
            "7289527c6078b92f52abc98f39c3871c30aed7fa56a2d8a2901d5f4f035b170f"
        ),
        "best_step": 49_500,
        "fixed_panel_train_loss": 1.085713002681732,
        "fixed_panel_validation_loss": 1.3601680147647857,
    },
)
# The one seed consumed by every width and dropout branch of this lineage. The
# sampler depends only on batch shape and seed, so all four endpoints in the
# champion shootout were trained on the same batch stream. Recorded by Stage 47
# as the batch generator of its C320 endpoints; re-derived here for all four.
DEFAULT_SHARED_TRAINING_GENERATOR_SHA256 = (
    "986ffc5650abe72ba270a01cc94ab5a247a946e1f4f040d1eef1508f8a7c86b9"
)


def resolve_lineage_device(requested: str) -> torch.device:
    """Resolve a device, preferring the accelerator the lineage recorded.

    Stage 43-47 trajectories are only reproducible on the device that produced
    them, so an unqualified CUDA request is satisfied by the recorded device when
    it is present. An explicit index is honoured and then checked separately.
    """

    if requested in ("cpu", "xpu", "mps"):
        return resolve_device(requested)
    if not torch.cuda.is_available():
        if requested == "auto":
            return resolve_device(requested)
        raise RuntimeError("CUDA was requested but is not available")
    if requested.startswith("cuda:"):
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as error:
            raise ValueError(f"malformed CUDA device selector: {requested!r}") from error
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {index} is not visible; "
                f"{torch.cuda.device_count()} device(s) are present"
            )
        return torch.device("cuda", index)
    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_name(index) == DEFAULT_LINEAGE_DEVICE_NAME:
            return torch.device("cuda", index)
    if requested == "auto":
        return resolve_device(requested)
    raise RuntimeError(
        f"CUDA was requested but no device named {DEFAULT_LINEAGE_DEVICE_NAME!r} "
        f"is present among {torch.cuda.device_count()} visible devices"
    )


def verify_lineage_device(device: torch.device) -> None:
    """The canonical protocol runs on the recorded accelerator or not at all."""

    if device.type != "cuda":
        return
    name = torch.cuda.get_device_name(device)
    if name != DEFAULT_LINEAGE_DEVICE_NAME:
        raise RuntimeError(
            f"Stage 49 requires {DEFAULT_LINEAGE_DEVICE_NAME!r} for the canonical "
            f"protocol; {device} is {name!r}. Kernel selection is device-dependent, "
            "so the Stage 48 control reproduction is only defined on the recorded "
            "device. Pass an explicit index such as --device cuda:1."
        )


def normalized_rng_sha256(
    rng_state: dict[str, Any], *, training_device_index: int | None
) -> str:
    """Hash the RNG state that actually governs training, not the host's GPU list.

    `capture_rng_state` stores every visible accelerator's state, so its container
    changes when the host's GPU set changes even though the training run does not:
    the same seed on the same accelerator yields the same CPU state and the same
    per-device state regardless of how many other GPUs are present. Cross-stage
    comparisons therefore project onto the CPU state and the training device's own
    entry, which is what reproducibility actually claims.
    """

    projection: dict[str, Any] = {"cpu": rng_state["cpu"]}
    entries = rng_state.get("cuda")
    if entries is not None:
        index = training_device_index
        if index is None:
            # A single-entry capture is a single-accelerator host, so its only
            # entry is the state of the device that trained.
            if len(entries) != 1:
                raise ValueError(
                    "a multi-device RNG capture must record its training device index"
                )
            index = 0
        if not 0 <= index < len(entries):
            raise ValueError(
                f"training device index {index} is outside a {len(entries)}-entry "
                "RNG capture"
            )
        projection["cuda_training_device"] = entries[index]
    return state_hash(projection)


def terminal_rng_normalization(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute an endpoint's training-device RNG projection from its own state."""

    rng_state = payload["rng_state"]
    runtime = payload.get("runtime", {})
    recorded = payload.get("rng_normalization", {})
    index = runtime.get("training_device_index", recorded.get("training_device_index"))
    entries = rng_state.get("cuda")
    if entries is not None and index is None and len(entries) != 1:
        raise ValueError(
            "endpoint captured several accelerators without recording which trained"
        )
    return {
        "training_device_name": runtime.get("device_name"),
        "training_device_index": index,
        "cuda_entry_count": len(entries) if entries is not None else 0,
        "terminal_training_rng_sha256": normalized_rng_sha256(
            rng_state, training_device_index=index
        ),
    }


def canonical_config(dropout: float = DEFAULT_CONTROL_DROPOUT) -> TrainingConfig:
    """The locked Stage 49 training config; dropout travels with BranchSpec."""

    if dropout not in (DEFAULT_CONTROL_DROPOUT, DEFAULT_TREATMENT_DROPOUT):
        raise ValueError("Stage 49 canonical dropout probabilities are 0.15 and 0.20")
    return replace(_STAGE_42.canonical_config(), n_embd=DEFAULT_WIDTH, learning_rate=1e-3)


@dataclass(frozen=True, slots=True)
class BranchSpec:
    key: str
    width: int
    dropout: float
    best_path: Path
    final_path: Path

    def __post_init__(self) -> None:
        if self.key not in ("control", "treatment"):
            raise ValueError("Stage 49 branch must be control or treatment")
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
            / f"stage_49_{key}_c{DEFAULT_WIDTH}_{dropout_label(dropout)}_best_checkpoint.pt",
            directory
            / (
                f"stage_49_{key}_c{DEFAULT_WIDTH}_{dropout_label(dropout)}"
                "_step_50000_final_checkpoint.pt"
            ),
        )
        for key, dropout in (
            ("control", DEFAULT_CONTROL_DROPOUT),
            ("treatment", DEFAULT_TREATMENT_DROPOUT),
        )
    )


def validate_branch_specs(specs: tuple[BranchSpec, ...]) -> None:
    expected = (
        ("control", DEFAULT_CONTROL_DROPOUT),
        ("treatment", DEFAULT_TREATMENT_DROPOUT),
    )
    if tuple((spec.key, spec.dropout) for spec in specs) != expected or any(
        spec.width != DEFAULT_WIDTH for spec in specs
    ):
        raise ValueError(
            "Stage 49 locks control p=0.15 against treatment p=0.20 at C256"
        )


def validate_protocol(data: CharacterData, configs: tuple[TrainingConfig, ...]) -> None:
    if configs != (canonical_config(), canonical_config(DEFAULT_TREATMENT_DROPOUT)):
        raise ValueError("Stage 49 locks all settings except residual dropout")
    if fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Stage 49 requires unchanged Complete Shakespeare and split")


def validate_artifact_paths(
    inputs: tuple[Path, ...], outputs: tuple[Path, ...]
) -> None:
    paths = [path.resolve() for path in (*inputs, *outputs)]
    if len(paths) != len(set(paths)):
        raise ValueError("corpus, checkpoints, and reports must have distinct paths")
    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Stage 49 output exists; use a new --output-dir: {path}"
            )
        if path.with_name(f".{path.name}.tmp").exists():
            raise FileExistsError(f"Stage 49 temporary output exists: {path}")


def verify_native_initialization(
    data: CharacterData,
    config: TrainingConfig,
    initialization: dict[str, Any],
) -> str:
    """Prove the C256 native initialization is the recorded one, twice over.

    Residual dropout adds no parameters and consumes no initialization RNG, so
    Stage 46's exposed-dropout factory, Stage 45's unchanged factory, and this
    stage must all build byte-identical native weights from the same seed. The
    hash Stage 45 recorded at C256 is required at the canonical width, which is
    this stage's only width, and it is the same hash at both p=0.15 and p=0.20.
    """

    if config.n_embd != DEFAULT_WIDTH:
        raise ValueError("Stage 49 native initialization is only defined at C256")
    digest = initialization["state_sha256"]
    _, _, _, reference = _STAGE_46.initialize_branch(
        data, config, DEFAULT_CONTROL_DROPOUT, torch.device("cpu")
    )
    if digest != reference["state_sha256"]:
        raise RuntimeError(
            "Stage 49 native initialization diverged from the Stage 46 factory"
        )
    inherited = _STAGE_46.verify_control_factory_equivalence(data, config)
    if inherited != digest:
        raise RuntimeError(
            "Stage 49 native initialization diverged from the Stage 45 factory"
        )
    if digest != C256_NATIVE_INITIALIZATION_SHA256:
        raise RuntimeError("Stage 49 native initialization hash changed at C256")
    return digest


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
        "origin_stage": EXPERIMENT_STAGE,
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
    rng_state = capture_rng_state(device)
    training_device_index = device.index
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
        "rng_state": rng_state,
        # The training device's own RNG projection, so a later stage on a host
        # with a different GPU set can still compare like with like.
        "rng_normalization": {
            "training_device_index": training_device_index,
            "cuda_entry_count": len(rng_state.get("cuda", ())),
            "terminal_training_rng_sha256": normalized_rng_sha256(
                rng_state, training_device_index=training_device_index
            ),
        },
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
            "stage": EXPERIMENT_STAGE,
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
            "device_index": training_device_index,
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
        raise ValueError("Stage 49 checkpoint must be a mapping")
    digest = payload.pop("payload_sha256", None)
    if digest != state_hash(payload):
        raise ValueError("Stage 49 checkpoint payload integrity mismatch")
    payload["payload_sha256"] = digest
    if payload["checkpoint_version"] != 1 or payload["checkpoint_kind"] != kind:
        raise ValueError("Stage 49 checkpoint version or kind mismatch")
    if (
        payload["experiment"]["stage"] != EXPERIMENT_STAGE
        or payload["experiment"]["protocol"] != PROTOCOL_NAME
    ):
        raise ValueError("Stage 49 experiment mismatch")
    if (
        payload["experiment"]["branch_key"] != spec.key
        or payload["experiment"]["branch_residual_dropout"] != spec.dropout
        or config.n_embd != spec.width
    ):
        raise ValueError("Stage 49 checkpoint branch mismatch")
    if payload["architecture"] != dropout_architecture_metadata(
        data, config, spec.dropout
    ):
        raise ValueError("Stage 49 architecture changed, including residual dropout")
    if payload["data_fingerprint"] != fingerprint_data(data):
        raise ValueError("Stage 49 corpus or split changed")
    for key, value in asdict(config).items():
        if payload["training_config"].get(key) != value:
            raise ValueError(f"Stage 49 training config changed: {key}")
    if payload["training_config"]["learning_rate_schedule"] != schedule_metadata(
        config.max_iters
    ):
        raise ValueError("Stage 49 learning rate schedule changed")
    step, best_step = payload["step"], payload["best_step"]
    if (
        type(step) is not int
        or type(best_step) is not int
        or not 0 <= best_step <= step <= config.max_iters
    ):
        raise ValueError("Stage 49 checkpoint steps are invalid")
    if kind == "final" and step != config.max_iters:
        raise ValueError("Stage 49 final must be the matched terminal step")
    if kind == "best" and step != best_step:
        raise ValueError("Stage 49 best must contain its selected weights")
    if not all(
        math.isfinite(value)
        for value in (
            *payload["snapshot_losses"].values(),
            payload["best_val_loss"],
            payload["best_training_loss"],
        )
    ):
        raise ValueError("Stage 49 losses must be finite")
    initialization = payload["initialization"]
    recorded_normalization = payload.get("rng_normalization")
    if not isinstance(recorded_normalization, dict):
        raise ValueError("Stage 49 checkpoint is missing its RNG normalization")
    recomputed = terminal_rng_normalization(payload)
    if (
        recorded_normalization.get("terminal_training_rng_sha256")
        != recomputed["terminal_training_rng_sha256"]
        or recorded_normalization.get("cuda_entry_count") != recomputed["cuda_entry_count"]
        or recorded_normalization.get("training_device_index")
        != recomputed["training_device_index"]
    ):
        raise ValueError("Stage 49 RNG normalization disagrees with its stored state")
    if payload["runtime"].get("device_index") != recorded_normalization.get(
        "training_device_index"
    ):
        raise ValueError("Stage 49 runtime device index disagrees with its normalization")
    if (
        initialization["kind"] != "fresh_architecture_native"
        or initialization["origin_stage"] != EXPERIMENT_STAGE
        or initialization["step"] != 0
        or initialization["seed"] != config.seed
        or initialization["residual_dropout"] != spec.dropout
        or initialization["historical_weights_loaded"]
        or initialization["historical_optimizer_loaded"]
        or initialization["optimizer_state_entries_at_initialization"] != 0
        or payload["optimizer_restart_step"] != 0
    ):
        raise ValueError("Stage 49 requires fresh model and optimizer initialization")
    optimizer_state = payload["optimizer_state_dict"]
    groups, states = optimizer_state["param_groups"], optimizer_state["state"]
    if len(groups) != 1:
        raise ValueError("Stage 49 requires one AdamW parameter group")
    group = groups[0]
    if (
        group["lr"] != make_schedule().for_update(max(0, step - 1))
        or group["betas"] != (0.9, 0.999)
        or group["weight_decay"] != 0.01
        or group["eps"] != 1e-8
        or group["amsgrad"]
    ):
        raise ValueError("Stage 49 AdamW settings changed")
    if step == 0:
        if states:
            raise ValueError("Stage 49 step-zero optimizer must be empty")
    elif set(states) != set(group["params"]) or any(
        state["step"].item() != step for state in states.values()
    ):
        raise ValueError("Stage 49 AdamW counters do not match completed updates")
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
                    "Stage 49 terminal state failed checkpoint round trip"
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
        raise RuntimeError("paired dropout treatment must be exactly 0.20 against 0.15")
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


@dataclass(frozen=True, slots=True)
class StoredEndpoint:
    """A stored, authenticated endpoint read as a read-only cross-stage peer."""

    key: str
    width: int
    dropout: float
    path: Path
    payload: dict[str, Any]
    config: TrainingConfig

    def __post_init__(self) -> None:
        if type(self.payload) is not dict:
            raise ValueError("stored endpoint payload must be a checkpoint mapping")
        if self.payload.get("step") != self.config.max_iters:
            raise ValueError("stored endpoint must record its terminal step")
        if self.payload.get("model_state_dict") is None:
            raise ValueError("stored endpoint payload must carry model weights")
        if self.config.n_embd != self.width:
            raise ValueError("stored endpoint width disagrees with its configuration")


def load_stage_48_treatment(data: CharacterData, directory: Path) -> StoredEndpoint:
    """Authenticate Stage 48's fresh C256 p=0.15 treatment as a read-only peer."""

    specs = _STAGE_48.branch_specs(directory)
    spec = next(spec for spec in specs if spec.key == "treatment")
    if spec.final_path.name != DEFAULT_STAGE_48_TREATMENT_FILENAME:
        raise ValueError("Stage 48 C256 treatment endpoint changed its name")
    if not spec.final_path.is_file():
        raise FileNotFoundError(f"Stage 48 endpoint not present: {spec.final_path}")
    if file_hash(spec.final_path) != DEFAULT_STAGE_48_TREATMENT_SHA256:
        raise ValueError("Stage 48 C256 treatment checkpoint hash changed")
    if not spec.best_path.is_file():
        raise FileNotFoundError(f"Stage 48 endpoint not present: {spec.best_path}")
    if file_hash(spec.best_path) != DEFAULT_STAGE_48_TREATMENT_BEST_SHA256:
        raise ValueError("Stage 48 C256 treatment best checkpoint hash changed")
    config = _STAGE_48.canonical_config(_STAGE_48.DEFAULT_TREATMENT_DROPOUT)
    payload = _STAGE_48.validate_branch_checkpoint(
        spec.final_path, spec, data, config, "final"
    )
    if (
        payload["best_step"] != DEFAULT_STAGE_48_TREATMENT_BEST_STEP
        or payload["snapshot_losses"]["train"]
        != DEFAULT_STAGE_48_TREATMENT_FIXED_PANEL_TRAIN_LOSS
        or payload["snapshot_losses"]["validation"]
        != DEFAULT_STAGE_48_TREATMENT_FIXED_PANEL_VALIDATION_LOSS
    ):
        raise ValueError("Stage 48 C256 treatment fixed-panel record changed")
    return StoredEndpoint(
        "c256_stage_48",
        spec.width,
        _STAGE_48.DEFAULT_TREATMENT_DROPOUT,
        spec.final_path,
        payload,
        config,
    )


def verify_stage_48_control_reproduction(
    spec: BranchSpec,
    data: CharacterData,
    config: TrainingConfig,
    reference: StoredEndpoint,
) -> dict[str, Any]:
    """Require the trained C256 p=0.15 control to reproduce Stage 48 exactly.

    The control is the endpoint this stage steps up from, so a control that is
    not that endpoint would silently invalidate the comparison and the ladder
    row it feeds. Both states are compared field by field: weights, AdamW state,
    batch-generator state, Torch RNG state, fixed-panel losses, and the selected
    best step.
    """

    payload = validate_branch_checkpoint(spec.final_path, spec, data, config, "final")
    recorded = {
        "step": payload["step"],
        "snapshot_losses": payload["snapshot_losses"],
        "best_step": payload["best_step"],
        "best_val_loss": payload["best_val_loss"],
        "best_training_loss": payload["best_training_loss"],
        "architecture": payload["architecture"],
        "data_fingerprint": payload["data_fingerprint"],
    }
    expected = {key: reference.payload[key] for key in recorded}
    mismatched = sorted(key for key, value in recorded.items() if value != expected[key])
    streams = {
        "model_sha256": (
            _STAGE_41.tensor_mapping_sha256(payload["model_state_dict"]),
            _STAGE_41.tensor_mapping_sha256(reference.payload["model_state_dict"]),
        ),
        "optimizer_sha256": (
            state_hash(payload["optimizer_state_dict"]),
            state_hash(reference.payload["optimizer_state_dict"]),
        ),
        "training_generator_sha256": (
            _STAGE_41.tensor_sha256(payload["training_generator_state"]),
            _STAGE_41.tensor_sha256(reference.payload["training_generator_state"]),
        ),
    }
    mismatched.extend(
        sorted(key for key, (actual, wanted) in streams.items() if actual != wanted)
    )
    # Torch RNG state is compared on the training device's own projection rather
    # than on the whole capture: the capture also holds every other visible
    # accelerator, so comparing containers would compare host topology instead of
    # the training run.
    ours = terminal_rng_normalization(payload)
    theirs = terminal_rng_normalization(reference.payload)
    if ours["training_device_name"] != theirs["training_device_name"]:
        mismatched.append("training_device_name")
    if (
        ours["terminal_training_rng_sha256"]
        != theirs["terminal_training_rng_sha256"]
    ):
        mismatched.append("training_device_rng_sha256")
    ours_cpu = state_hash(payload["rng_state"]["cpu"])
    theirs_cpu = state_hash(reference.payload["rng_state"]["cpu"])
    if ours_cpu != theirs_cpu:
        mismatched.append("cpu_rng_sha256")
    if mismatched:
        raise RuntimeError(
            "Stage 49 C256 p=0.15 control did not reproduce the Stage 48 treatment: "
            + ", ".join(mismatched)
        )
    return {
        "reference": "Stage 48 C256 p=0.15 treatment, step 50,000",
        "reference_path": str(reference.path.resolve()),
        "reference_sha256": DEFAULT_STAGE_48_TREATMENT_SHA256,
        "identical": True,
        "fields_checked": sorted(recorded) + sorted(streams) + [
            "cpu_rng_sha256",
            "training_device_rng_sha256",
            "training_device_name",
        ],
        "state_sha256": {
            **streams,
            "cpu_rng_sha256": (ours_cpu, theirs_cpu),
            "training_device_rng_sha256": (
                ours["terminal_training_rng_sha256"],
                theirs["terminal_training_rng_sha256"],
            ),
        },
        "rng_capture": {
            "training_device_name": ours["training_device_name"],
            "training_device_index": ours["training_device_index"],
            "cuda_entries_at_capture": ours["cuda_entry_count"],
            "reference_cuda_entries_at_capture": theirs["cuda_entry_count"],
            "container_identical": state_hash(payload["rng_state"])
            == state_hash(reference.payload["rng_state"]),
            "note": (
                "The full RNG capture also stores every other visible accelerator, "
                "so its container hash tracks the host's GPU set rather than the "
                "training run. Reproducibility is therefore asserted on the CPU "
                "state and the training device's own entry, which are byte-identical "
                "whether or not the containers match."
            ),
        },
    }


def load_c320_champion(
    ref: dict[str, Any], data: CharacterData, directory: Path
) -> StoredEndpoint:
    """Authenticate one stored C320 champion through its owning stage."""

    owner = {"stage_46": _STAGE_46, "stage_47": _STAGE_47}[ref["owner"]]
    specs = owner.branch_specs(directory)
    spec = next(spec for spec in specs if spec.final_path.name == ref["filename"])
    if not spec.final_path.is_file():
        raise FileNotFoundError(f"C320 champion not present: {spec.final_path}")
    if file_hash(spec.final_path) != ref["file_sha256"]:
        raise ValueError(f"C320 champion checkpoint hash changed: {ref['filename']}")
    config = owner.canonical_config()
    payload = owner.validate_branch_checkpoint(
        spec.final_path, spec, data, config, "final"
    )
    if (
        payload["architecture"]["residual_dropout"] != ref["dropout"]
        or config.n_embd != ref["width"]
    ):
        raise ValueError(f"C320 champion architecture changed: {ref['filename']}")
    if (
        payload["best_step"] != ref["best_step"]
        or payload["snapshot_losses"]["train"] != ref["fixed_panel_train_loss"]
        or payload["snapshot_losses"]["validation"]
        != ref["fixed_panel_validation_loss"]
    ):
        raise ValueError(f"C320 champion fixed-panel record changed: {ref['filename']}")
    return StoredEndpoint(
        ref["key"], ref["width"], ref["dropout"], spec.final_path, payload, config
    )


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
    if seed in (
        DEFAULT_FIXED_PANEL_SEED,
        1354,
        1355,
        1356,
        1357,
        1358,
        1359,
        1360,
        1361,
        1362,
        1363,
        1364,
        1365,
    ):
        raise ValueError("Stage 49 fresh validation must use an unused panel seed")
    if eval_iters < 2:
        raise ValueError("paired uncertainty requires at least two batches")
    verify_paired_branches(control, treatment)
    caller_rng = capture_rng_state(device)
    models, payloads = [], []
    try:
        for branch in (control, treatment):
            if file_hash(branch.spec.final_path) != branch.final_sha256:
                raise ValueError("Stage 49 endpoint checkpoint hash mismatch")
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
                raise ValueError("Stage 49 fresh validation produced non-finite loss")
            if file_hash(branch.spec.final_path) != branch.final_sha256:
                raise RuntimeError("Stage 49 endpoint changed during evaluation")
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
            "p=0.20 improves C256 validation at matched 50k updates on the "
            "fresh paired panel"
        )
    if report.delta.confidence_low > 0:
        return "p=0.15 beats p=0.20 at C256 on the fresh paired panel"
    return "unresolved within paired panel uncertainty; this does not prove equivalence"


def champion_plan(
    data: CharacterData,
    config: TrainingConfig,
    directory: Path,
    refs: Sequence[Any] = C320_CHAMPION_REFS,
    specs: Sequence[BranchSpec] | None = None,
) -> list[dict[str, Any]]:
    """The four champion endpoints, authenticated read-only and paired.

    The two C256 endpoints are this stage's fresh branches; the two C320
    endpoints are stored Stage 46 and Stage 47 treatments. Every endpoint is
    re-read from disk so the four-way panel is derivable without training, and
    each label carries its own branch width because the four do not share one.
    """

    specs = tuple(branch_specs(directory) if specs is None else specs)
    if any(spec.width != config.n_embd for spec in specs):
        raise ValueError("champion branch endpoints must match the evaluated width")
    plan: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.final_path.is_file():
            raise FileNotFoundError(f"Stage 49 endpoint not present: {spec.final_path}")
        plan.append(
            {
                "key": f"c{spec.width}_{dropout_label(spec.dropout)}",
                "path": spec.final_path,
                "payload": validate_branch_checkpoint(
                    spec.final_path, spec, data, config, "final"
                ),
                "config": config,
                "dropout": spec.dropout,
                "provenance": f"stage_49 {spec.key}, fresh",
            }
        )
    for ref in refs:
        endpoint = load_c320_champion(ref, data, directory)
        if endpoint.width == config.n_embd:
            raise ValueError("a stored champion peer must differ in width from the branches")
        plan.append(
            {
                "key": endpoint.key,
                "path": endpoint.path,
                "payload": endpoint.payload,
                "config": endpoint.config,
                "dropout": endpoint.dropout,
                "provenance": f"{ref['owner']} treatment, stored",
            }
        )
    for entry in plan:
        entry["label"] = (
            f"C{entry['config'].n_embd} p={entry['dropout']:g} "
            f"@ {entry['payload']['step']}"
        )
    return plan


def evaluate_champion_shootout(
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    directory: Path,
    *,
    refs: Sequence[Any] = C320_CHAMPION_REFS,
    specs: Sequence[BranchSpec] | None = None,
    expected_generator_sha256: str | None = DEFAULT_SHARED_TRAINING_GENERATOR_SHA256,
    seed: int = DEFAULT_CHAMPION_PANEL_SEED,
    eval_iters: int = DEFAULT_FRESH_VALIDATION_ITERS,
) -> dict[str, Any]:
    """Score the four width/dropout champions on one shared fresh panel.

    No checkpoint is modified. The panel is shared per batch, so every reported
    interval is paired at matched validation batches; dropout is inactive in
    eval mode, so only stored weights differ. All four branches were trained on
    one batch stream because the sampler depends only on batch shape and seed,
    which is checked here rather than assumed.
    """

    if seed in (
        DEFAULT_FIXED_PANEL_SEED,
        1354,
        1355,
        1356,
        1357,
        1358,
        1359,
        1360,
        1361,
        1362,
        1363,
        1364,
        1365,
    ):
        raise ValueError("champion panel seed must be unused")
    if eval_iters < 2:
        raise ValueError("paired uncertainty requires at least two batches")
    specs = tuple(branch_specs(directory) if specs is None else specs)
    plan = champion_plan(data, config, directory, refs, specs)
    entries = {entry["key"]: entry for entry in plan}
    control_key = f"c{specs[0].width}_{dropout_label(specs[0].dropout)}"
    treatment_key = f"c{specs[1].width}_{dropout_label(specs[1].dropout)}"
    verify_paired_streams(
        specs[0],
        specs[1],
        entries[control_key]["payload"]["initial_streams"],
        entries[treatment_key]["payload"]["initial_streams"],
        {
            "training_generator_sha256": _STAGE_41.tensor_sha256(
                entries[control_key]["payload"]["training_generator_state"]
            )
        },
        {
            "training_generator_sha256": _STAGE_41.tensor_sha256(
                entries[treatment_key]["payload"]["training_generator_state"]
            )
        },
        entries[control_key]["payload"]["initialization"]["parameter_count"],
        entries[treatment_key]["payload"]["initialization"]["parameter_count"],
    )
    generator_streams = {
        _STAGE_41.tensor_sha256(entry["payload"]["training_generator_state"])
        for entry in plan
    }
    if len(generator_streams) != 1:
        raise ValueError("champion endpoints were not trained on one batch stream")
    if (
        expected_generator_sha256 is not None
        and generator_streams != {expected_generator_sha256}
    ):
        raise ValueError(
            "champion batch stream is not the recorded shared lineage stream"
        )
    # Cross-stage panels must not compare RNG containers: the stored C320 peers
    # recorded a single-accelerator host while this run records the accelerators
    # it can see. The fresh pair's own containers are compared by
    # verify_paired_streams; across stages, compare the training device's
    # projection and its identity instead.
    projections = {
        entry["key"]: terminal_rng_normalization(entry["payload"]) for entry in plan
    }
    device_names = {value["training_device_name"] for value in projections.values()}
    if len(device_names) != 1:
        raise ValueError(
            "champion endpoints were not trained on the same device: "
            + ", ".join(sorted(str(name) for name in device_names))
        )
    terminal_rng = {
        value["terminal_training_rng_sha256"] for value in projections.values()
    }
    if len(terminal_rng) != 1:
        raise ValueError(
            "champion endpoints do not share a training-device RNG state"
        )
    caller_rng = capture_rng_state(device)
    models = []
    try:
        for entry in plan:
            model = build_model(
                data, entry["config"], N_HEAD, N_LAYER, entry["dropout"], device
            )
            model.load_state_dict(entry["payload"]["model_state_dict"], strict=True)
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
        for entry, values in zip(plan, losses, strict=True):
            if not all(math.isfinite(value) for value in values):
                raise ValueError("champion shootout produced non-finite loss")
            mean, se = _STAGE_40._mean_and_standard_error(values)
            results[entry["key"]] = _STAGE_40.FreshCheckpointResult(
                entry["label"],
                entry["path"].resolve(),
                file_hash(entry["path"]),
                entry["payload"]["step"],
                entry["payload"]["snapshot_losses"]["validation"],
                mean,
                se,
                values,
            )
        comparisons = {
            f"{candidate}_minus_{baseline}": asdict(
                _STAGE_40._fresh_paired_delta(results[candidate], results[baseline])
            )
            for candidate, baseline in itertools.combinations(results, 2)
        }
        ranking = sorted(results, key=lambda key: results[key].mean_loss)
        return {
            "seed": seed,
            "eval_iters": eval_iters,
            "shared_panel_per_batch": True,
            "dropout_enabled": False,
            "shared_training_batch_stream_verified": True,
            "shared_training_generator_sha256": DEFAULT_SHARED_TRAINING_GENERATOR_SHA256,
            "rng_projections": projections,
            "training_device_name": next(iter(device_names)),
            "endpoints": {key: asdict(value) for key, value in results.items()},
            "ranking": [
                {
                    "rank": rank,
                    "key": key,
                    "label": results[key].name,
                    "provenance": entries[key]["provenance"],
                    "mean_validation_loss": results[key].mean_loss,
                    "standard_error": results[key].standard_error,
                    "checkpoint": str(results[key].checkpoint_path),
                    "checkpoint_sha256": results[key].checkpoint_sha256,
                }
                for rank, key in enumerate(ranking, start=1)
            ],
            "comparisons": comparisons,
            "final_generator_sha256": generator_hash,
        }
    finally:
        models.clear()
        _STAGE_40._restore_rng_state(caller_rng, device)
        _STAGE_41.clear_accelerator_cache(device)


def compared_delta(
    comparisons: dict[str, Any], candidate: str, baseline: str
) -> dict[str, Any]:
    """Read a paired delta in the requested direction from either stored key.

    The panel stores each unordered pair once, so a candidate-baseline query may
    only exist as its mirror image. Mirroring negates the delta, its standard
    error is unchanged, and the interval negates and swaps endpoints.
    """

    forward = f"{candidate}_minus_{baseline}"
    if forward in comparisons:
        return comparisons[forward]
    reverse = comparisons.get(f"{baseline}_minus_{candidate}")
    if reverse is None:
        raise KeyError(f"champion panel recorded no comparison of {candidate} and {baseline}")
    return {
        **reverse,
        "candidate": candidate,
        "baseline": baseline,
        "mean_delta": -reverse["mean_delta"],
        "confidence_low": -reverse["confidence_high"],
        "confidence_high": -reverse["confidence_low"],
    }


def champion_outcome(report: dict[str, Any]) -> str:
    """Name the winner, and say whether the panel resolved it against all peers."""

    ranking = report["ranking"]
    champion = ranking[0]
    undefeated = all(
        compared_delta(report["comparisons"], entry["key"], champion["key"])[
            "confidence_low"
        ]
        > 0
        for entry in ranking[1:]
    )
    if undefeated:
        return (
            f"{champion['label']} is the resolved champion: it beats all "
            f"{len(ranking) - 1} peers on the shared fresh panel"
        )
    undecided = ", ".join(
        entry["label"]
        for entry in ranking[1:]
        if not compared_delta(report["comparisons"], entry["key"], champion["key"])[
            "confidence_low"
        ]
        > 0
    )
    return (
        f"{champion['label']} leads the shared fresh panel but the panel does "
        f"not resolve it against {undecided}"
    )


def capacity_regularization_ladder(paired: Any) -> list[dict[str, Any]]:
    """The width-versus-dropout curve, with this stage's step measured fresh."""

    measured = {
        "width": DEFAULT_WIDTH,
        "dropout_step": f"p={DEFAULT_CONTROL_DROPOUT:g} -> "
        f"p={DEFAULT_TREATMENT_DROPOUT:g}",
        "source_stage": EXPERIMENT_STAGE,
        "fresh_panel_seed": paired.seed,
        "mean_delta": paired.delta.mean_delta,
        "standard_error": paired.delta.standard_error,
        "confidence_low": paired.delta.confidence_low,
        "confidence_high": paired.delta.confidence_high,
    }
    return [measured, *(copy.deepcopy(step) for step in PRIOR_DROPOUT_STEPS)]


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
        initialization_hash = verify_native_initialization(
            data, configs[0], audits[0]["initialization"]
        )
        for key in ("model_sha256", "training_generator_sha256", "torch_rng_sha256"):
            if audits[0]["initial_streams"][key] != audits[1]["initial_streams"][key]:
                raise RuntimeError(f"preflight branch random streams differ in {key}")
        if audits[0]["initialization"]["parameter_count"] != audits[1]["initialization"][
            "parameter_count"
        ]:
            raise RuntimeError("preflight branches differ in parameter count")
        device_type, device_name = _STAGE_41._runtime_identity(device)
        return {
            "stage": EXPERIMENT_STAGE,
            "protocol": PROTOCOL_NAME,
            "training_executed": False,
            "runtime": {
                "device_type": device_type,
                "device_name": device_name,
                "torch_version": str(torch.__version__),
                "lineage_device_required": DEFAULT_LINEAGE_DEVICE_NAME,
                "lineage_device_verified": (
                    device.type != "cuda" or device_name == DEFAULT_LINEAGE_DEVICE_NAME
                ),
            },
            "branches": audits,
            "control_initialization_equivalence": {
                "reference": "Stage 45 recorded C256 native initialization (dropout-invariant)",
                "state_sha256": initialization_hash,
                "stage_46_factory_identical": True,
                "stage_45_factory_identical": True,
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
            "Stage 49: fresh C256 at residual dropout p=0.15 versus p=0.20, "
            "then a four-way C256-versus-C320 champion shootout"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "cpu, auto, or a CUDA selector; the canonical protocol requires the "
            f"recorded {DEFAULT_LINEAGE_DEVICE_NAME!r} device"
        ),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_COMPLETE_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate corpus, architecture, and initialization without training",
    )
    parser.add_argument(
        "--champion-panel-only",
        action="store_true",
        help=(
            "score the stored C256 p=0.15, C256 p=0.20, C320 p=0.15, and "
            "C320 p=0.20 endpoints on one shared fresh panel without training"
        ),
    )
    return parser.parse_args()


def print_champion_report(report: dict[str, Any]) -> None:
    for entry in report["ranking"]:
        print(
            f"  {entry['rank']}. {entry['label']}: "
            f"{entry['mean_validation_loss']:.7f} (SE {entry['standard_error']:.7f})",
            flush=True,
        )
    for key, delta in report["comparisons"].items():
        print(
            f"  {delta['candidate']} - {delta['baseline']}: "
            f"{delta['mean_delta']:+.7f}; SE {delta['standard_error']:.7f}; "
            f"95% panel interval [{delta['confidence_low']:+.7f}, "
            f"{delta['confidence_high']:+.7f}]",
            flush=True,
        )


def run_champion_panel_only(args: argparse.Namespace) -> None:
    """Reuse the stored Stage 49 endpoints; never trains or rewrites them."""

    data = CharacterData.from_file(args.data, train_fraction=0.9)
    device = resolve_lineage_device(args.device)
    verify_lineage_device(device)
    config = canonical_config()
    specs = branch_specs(args.output_dir)
    validate_branch_specs(specs)
    report_path = args.output_dir / "stage_49_champion_shootout.json"
    validate_artifact_paths((args.data,), (report_path,))
    for spec in specs:
        if not spec.final_path.is_file():
            raise FileNotFoundError(f"Stage 49 endpoint not present: {spec.final_path}")
    print(
        f"Stage 49 champion shootout on {device}: C256 p=0.15, C256 p=0.20, "
        f"C320 p=0.15, C320 p=0.20 at {DEFAULT_CHAMPION_PANEL_SEED}",
        flush=True,
    )
    report = evaluate_champion_shootout(data, config, device, args.output_dir)
    outcome = champion_outcome(report)
    report["champion_outcome"] = outcome
    report["uncertainty_scope"] = UNCERTAINTY_SCOPE
    report["budget_interpretation"] = (
        "one shared 500-batch validation panel; dropout inactive in eval mode; "
        "the two C256 endpoints are this stage's fresh branches and the two "
        "C320 endpoints are stored Stage 46 and Stage 47 treatments"
    )
    _STAGE_42.write_report(report_path, report)
    print_champion_report(report)
    print(f"{outcome}. Report: {report_path.resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    if args.champion_panel_only:
        run_champion_panel_only(args)
        return
    data = CharacterData.from_file(args.data, train_fraction=0.9)
    device = resolve_lineage_device(args.device)
    if not args.validate_only:
        verify_lineage_device(device)
    configs = (canonical_config(), canonical_config(DEFAULT_TREATMENT_DROPOUT))
    specs = branch_specs(args.output_dir)
    report_path = args.output_dir / "stage_49_report.json"
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
        f"Stage 49 on {device}: native C256 at p=0.15 and p=0.20, "
        "50,000 updates per branch",
        flush=True,
    )
    control = run_branch(specs[0], data, configs[0], device)
    reproduction = verify_stage_48_control_reproduction(
        specs[0], data, configs[0], load_stage_48_treatment(data, args.output_dir)
    )
    treatment = run_branch(specs[1], data, configs[1], device)
    verify_paired_branches(control, treatment)
    fresh = evaluate_checkpoint_pair(control, treatment, data, configs[0], device)
    _STAGE_42.print_pair(fresh)
    outcome = paired_outcome(fresh)
    shootout = evaluate_champion_shootout(data, configs[0], device, args.output_dir)
    print_champion_report(shootout)
    shootout_outcome = champion_outcome(shootout)
    control_gap = control.training.final.val - control.training.final.train
    treatment_gap = treatment.training.final.val - treatment.training.final.train
    report = {
        **audit,
        "training_executed": True,
        "paired_batch_streams_verified": True,
        "control": asdict(control),
        "treatment": asdict(treatment),
        "stage_48_control_reproduction": reproduction,
        "fresh_50k_branches": asdict(fresh),
        "champion_shootout": shootout,
        "capacity_regularization_ladder": capacity_regularization_ladder(fresh),
        "uncertainty_scope": UNCERTAINTY_SCOPE,
        "final_fixed_panel_delta_treatment_minus_control": treatment.training.final.val
        - control.training.final.val,
        "selected_best_fixed_panel_delta_treatment_minus_control": treatment.best_losses.val
        - control.best_losses.val,
        "final_generalization_gap_delta_treatment_minus_control": treatment_gap
        - control_gap,
        "outcome": outcome,
        "champion_outcome": shootout_outcome,
        "budget_interpretation": (
            "equal updates, target tokens, and parameter count; only residual "
            "dropout differs"
        ),
    }
    _STAGE_42.write_report(report_path, report)
    print(f"{outcome}. {shootout_outcome}. Report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
