from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from config import TrainingConfig
from data_utils import CharacterData
from train import resolve_device
from training import EvaluationRecord, LossEstimate, TrainingResult


# Stage 37 keeps the Stage 36 T=256 control architecture and training budget,
# but trains one fresh model on Complete Shakespeare. Stage 36 is loaded
# privately so the historical experiment remains unchanged.
_STAGE_36_PATH = Path(__file__).with_name("36_context_scaling.py")
_STAGE_36_MODULE_NAME = "stage_36_context_scaling_for_stage_37"
_STAGE_36_SPEC = importlib.util.spec_from_file_location(
    _STAGE_36_MODULE_NAME,
    _STAGE_36_PATH,
)
assert _STAGE_36_SPEC is not None and _STAGE_36_SPEC.loader is not None
_STAGE_36 = importlib.util.module_from_spec(_STAGE_36_SPEC)
sys.modules[_STAGE_36_MODULE_NAME] = _STAGE_36
_STAGE_36_SPEC.loader.exec_module(_STAGE_36)


GPTLanguageModel = _STAGE_36.GPTLanguageModel
LearningRateSchedule = _STAGE_36.LearningRateSchedule
BestValidationCheckpoint = _STAGE_36._STAGE_32.BestValidationCheckpoint
checkpoint_sha256 = _STAGE_36.checkpoint_sha256
clear_accelerator_cache = _STAGE_36.clear_accelerator_cache
fingerprint_data = _STAGE_36.fingerprint_data
generate_from_final_model = _STAGE_36._STAGE_32.generate_from_final_model
seed_everything = _STAGE_36.seed_everything
tensor_mapping_sha256 = _STAGE_36.tensor_mapping_sha256


EXPERIMENT_STAGE = 37
EXPERIMENT_TITLE = "TinyShakespeare to Complete Shakespeare data scaling"
COMPARISON_VARIABLE = "supplied_training_corpus_and_scale"
PROTOCOL_NAME = "complete_shakespeare_equal_training_token_budget"

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
# Preserve the repository's existing filename, including its historical typo.
DEFAULT_COMPLETE_DATA_PATH = DATA_DIRECTORY / "full-shakespreare.txt"
CHECKPOINT_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "stage_37_complete_shakespeare_best_checkpoint.pt"
)

DEFAULT_BATCH_SIZE = 8
DEFAULT_BLOCK_SIZE = 256
DEFAULT_LOCAL_CONTEXT = 23
DEFAULT_LAYER_CONTEXTS = (23, 23, 23, 23, None, None)
DEFAULT_GLOBAL_LAYER_INDICES = (4, 5)
DEFAULT_GLOBAL_LAYER_NUMBERS = (5, 6)
DEFAULT_N_EMBD = 128
DEFAULT_N_HEAD = 4
DEFAULT_HEAD_SIZE = 32
DEFAULT_FEED_FORWARD_WIDTH = 512
DEFAULT_N_LAYER = 6
DEFAULT_DROPOUT = 0.10
DEFAULT_MAX_ITERS = 18_000
DEFAULT_INITIAL_LEARNING_RATE = 1e-3
DEFAULT_FIRST_DECAY_STEP = 10_000
DEFAULT_MIDDLE_LEARNING_RATE = 3e-4
DEFAULT_SECOND_DECAY_STEP = 13_000
DEFAULT_FINAL_LEARNING_RATE = 1e-4
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_EVAL_ITERS = 100
DEFAULT_SEED = 1_337
DEFAULT_TRAINING_BATCH_SEED = 1_338
DEFAULT_FIXED_PANEL_SEED = 1_339
DEFAULT_TRAINING_RNG_SEED = 1_340
DEFAULT_SAMPLE_SEED = 1_342
DEFAULT_TRAIN_FRACTION = 0.9

DEFAULT_TARGETS_PER_UPDATE = 2_048
DEFAULT_TOTAL_TARGET_CHARACTERS = 36_864_000

# Audited decoded-corpus facts. Path.read_text() normalizes the source CRLFs,
# so these are the character counts actually tokenized by CharacterData.
DEFAULT_COMPLETE_SOURCE_BYTES = 5_589_865
DEFAULT_COMPLETE_SOURCE_SHA256 = (
    "9103ec54b761706256a2b66313311c52f2052d78e55d6e535a7cdc599feebed1"
)
DEFAULT_COMPLETE_LF_SOURCE_BYTES = 5_407_045
DEFAULT_COMPLETE_LF_SOURCE_SHA256 = (
    "811e63ec1ecac276e149193cac028d00888806c494aed70d772cf8670dd667e5"
)
DEFAULT_COMPLETE_NUM_CHARACTERS = 5_343_766
DEFAULT_COMPLETE_TRAIN_CHARACTERS = 4_809_389
DEFAULT_COMPLETE_VAL_CHARACTERS = 534_377
DEFAULT_COMPLETE_VOCAB_SIZE = 98
DEFAULT_COMPLETE_DATA_FINGERPRINT = (
    "c6cbdd246799ad495f7813e5d20afa52c03e6c339fe7dbe15912fd81a45d2b48"
)
DEFAULT_PARAMETER_COUNT = 1_245_538
DEFAULT_INITIAL_STATE_SHA256 = (
    "7a5b19780a98d02b4660b9e8f0d89715026ee9b7ae6439aaab320d41e8221726"
)
DEFAULT_PRE_GLOBAL_EFFECTIVE_RECEPTIVE_FIELD = 89
DEFAULT_LOGICAL_EDGES_BY_LAYER = (5_635, 5_635, 5_635, 5_635, 32_896, 32_896)
DEFAULT_COMPLETE_TRAIN_CORPUS_EQUIVALENTS = (
    DEFAULT_TOTAL_TARGET_CHARACTERS / DEFAULT_COMPLETE_TRAIN_CHARACTERS
)

# The control is an authenticated, read-only result from the exact Stage 36
# T=256 branch. It is reported but never loaded or retrained by Stage 37.
HISTORICAL_CONTROL_STAGE = 36
HISTORICAL_CONTROL_BRANCH = "T256 control"
HISTORICAL_CONTROL_CORPUS = "TinyShakespeare"
HISTORICAL_CONTROL_TRAIN_CHARACTERS = 1_003_854
HISTORICAL_CONTROL_VAL_CHARACTERS = 111_540
HISTORICAL_CONTROL_VOCAB_SIZE = 65
HISTORICAL_CONTROL_SELECTED_STEP = 15_000
HISTORICAL_CONTROL_SELECTED_TRAIN_LOSS = 1.1062
HISTORICAL_CONTROL_SELECTED_VAL_LOSS = 1.4854
HISTORICAL_CONTROL_FINAL_STEP = 18_000
HISTORICAL_CONTROL_FINAL_TRAIN_LOSS = 1.0923
HISTORICAL_CONTROL_FINAL_VAL_LOSS = 1.4903
HISTORICAL_CONTROL_CHECKPOINT_SHA256 = (
    "e8961efa7d8a39ee1eb79a0959bc3653ecadd8ca6a5eb53df279a7ff0116374c"
)
HISTORICAL_CONTROL_DEVICE = "Intel(R) Arc(TM) 140T GPU (16GB)"
HISTORICAL_CONTROL_TRAIN_CORPUS_EQUIVALENTS = (
    DEFAULT_TOTAL_TARGET_CHARACTERS / HISTORICAL_CONTROL_TRAIN_CHARACTERS
)

# Complete Shakespeare's vocabulary changes both the embedding/output tensors
# and the uniform-loss baseline. Keep this explicit rather than claiming that
# raw cross-corpus validation losses are a paired metric.
RAW_LOSS_CROSS_CORPUS_COMPARABLE = False
HISTORICAL_CONTROL_RERUN = False


@dataclass(frozen=True, slots=True)
class HistoricalControl:
    source_stage: int = HISTORICAL_CONTROL_STAGE
    branch: str = HISTORICAL_CONTROL_BRANCH
    corpus: str = HISTORICAL_CONTROL_CORPUS
    train_characters: int = HISTORICAL_CONTROL_TRAIN_CHARACTERS
    val_characters: int = HISTORICAL_CONTROL_VAL_CHARACTERS
    vocabulary_size: int = HISTORICAL_CONTROL_VOCAB_SIZE
    selected_step: int = HISTORICAL_CONTROL_SELECTED_STEP
    selected_train_loss: float = HISTORICAL_CONTROL_SELECTED_TRAIN_LOSS
    selected_val_loss: float = HISTORICAL_CONTROL_SELECTED_VAL_LOSS
    final_step: int = HISTORICAL_CONTROL_FINAL_STEP
    final_train_loss: float = HISTORICAL_CONTROL_FINAL_TRAIN_LOSS
    final_val_loss: float = HISTORICAL_CONTROL_FINAL_VAL_LOSS
    checkpoint_sha256: str = HISTORICAL_CONTROL_CHECKPOINT_SHA256

    @property
    def target_presentations(self) -> int:
        return DEFAULT_TOTAL_TARGET_CHARACTERS

    @property
    def train_corpus_equivalents(self) -> float:
        return self.target_presentations / self.train_characters


DEFAULT_HISTORICAL_CONTROL = HistoricalControl()


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    name: str
    source_path: Path | None
    source_bytes: int | None
    source_sha256: str | None
    data_fingerprint: str
    num_characters: int
    train_characters: int
    val_characters: int
    train_fraction: float
    vocabulary_size: int
    targets_per_update: int
    total_target_presentations: int
    train_corpus_equivalents: float
    train_pool_multiplier_vs_tiny: float


@dataclass(frozen=True, slots=True)
class Stage37Report:
    corpus: CorpusSummary
    historical_control: HistoricalControl
    parameter_count: int
    initial_state_sha256: str
    attention_shape: tuple[int, ...]
    training: TrainingResult
    best_train_loss: float
    best_val_loss: float
    best_step: int
    final_batch_generator_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    sample: str

    @property
    def generalization_gap(self) -> float:
        return self.training.final.val - self.training.final.train

    @property
    def validation_reversal(self) -> float:
        return self.training.final.val - self.best_val_loss


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_corpus(
    data: CharacterData,
    config: TrainingConfig,
    *,
    source_path: Path | None,
) -> CorpusSummary:
    targets_per_update = config.batch_size * config.block_size
    total_target_presentations = targets_per_update * config.max_iters
    train_characters = len(data.train_data)
    val_characters = len(data.val_data)
    if data.num_characters != train_characters + val_characters:
        raise ValueError(
            "corpus character count must equal train plus validation sizes"
        )
    if train_characters == 0:
        raise ValueError("training split must not be empty")

    resolved_path: Path | None = None
    source_bytes: int | None = None
    source_sha256: str | None = None
    if source_path is not None:
        resolved_path = source_path.resolve()
        source_bytes = resolved_path.stat().st_size
        source_sha256 = _sha256_file(resolved_path)

    return CorpusSummary(
        name="Complete Shakespeare",
        source_path=resolved_path,
        source_bytes=source_bytes,
        source_sha256=source_sha256,
        data_fingerprint=fingerprint_data(data),
        num_characters=data.num_characters,
        train_characters=train_characters,
        val_characters=val_characters,
        train_fraction=train_characters / data.num_characters,
        vocabulary_size=data.vocabulary.size,
        targets_per_update=targets_per_update,
        total_target_presentations=total_target_presentations,
        train_corpus_equivalents=(total_target_presentations / train_characters),
        train_pool_multiplier_vs_tiny=(
            train_characters / HISTORICAL_CONTROL_TRAIN_CHARACTERS
        ),
    )


def is_canonical_protocol(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    *,
    training_batch_seed: int = DEFAULT_TRAINING_BATCH_SEED,
    fixed_panel_seed: int = DEFAULT_FIXED_PANEL_SEED,
    training_rng_seed: int = DEFAULT_TRAINING_RNG_SEED,
) -> bool:
    return (
        fingerprint_data(data) == DEFAULT_COMPLETE_DATA_FINGERPRINT
        and config.batch_size == DEFAULT_BATCH_SIZE
        and config.block_size == DEFAULT_BLOCK_SIZE
        and config.n_embd == DEFAULT_N_EMBD
        and config.learning_rate == DEFAULT_INITIAL_LEARNING_RATE
        and config.max_iters == DEFAULT_MAX_ITERS
        and config.eval_interval == DEFAULT_EVAL_INTERVAL
        and config.eval_iters == DEFAULT_EVAL_ITERS
        and config.seed == DEFAULT_SEED
        and n_head == DEFAULT_N_HEAD
        and n_layer == DEFAULT_N_LAYER
        and training_batch_seed == DEFAULT_TRAINING_BATCH_SEED
        and fixed_panel_seed == DEFAULT_FIXED_PANEL_SEED
        and training_rng_seed == DEFAULT_TRAINING_RNG_SEED
        and schedule
        == LearningRateSchedule(
            initial_learning_rate=DEFAULT_INITIAL_LEARNING_RATE,
            first_decay_step=DEFAULT_FIRST_DECAY_STEP,
            middle_learning_rate=DEFAULT_MIDDLE_LEARNING_RATE,
            second_decay_step=DEFAULT_SECOND_DECAY_STEP,
            final_learning_rate=DEFAULT_FINAL_LEARNING_RATE,
        )
    )


def validate_protocol(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    *,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> None:
    if config.learning_rate != schedule.initial_learning_rate:
        raise ValueError("config learning rate must equal schedule initial LR")
    schedule.validate_target_step(config.max_iters)
    if n_layer != DEFAULT_N_LAYER:
        raise ValueError(f"Stage 37 requires L={DEFAULT_N_LAYER}, got {n_layer}")
    if n_head <= 0 or config.n_embd % n_head:
        raise ValueError(
            f"n_embd ({config.n_embd}) must be divisible by a positive "
            f"n_head ({n_head})"
        )
    if config.block_size < DEFAULT_LOCAL_CONTEXT:
        raise ValueError(
            f"block_size must be at least W{DEFAULT_LOCAL_CONTEXT}, "
            f"got {config.block_size}"
        )
    if config.batch_size % 2:
        raise ValueError(
            "Stage 37 requires an even batch size so B x T can be sampled "
            "as (B/2) contiguous 2T panels, matching the Stage 36 control"
        )
    for split, source in (("train", data.train_data), ("val", data.val_data)):
        panel_size = 2 * config.block_size
        if len(source) <= panel_size:
            raise ValueError(
                f"The {split} split has {len(source)} tokens, but Stage 37 "
                f"sampling requires a 2T panel of {panel_size} tokens"
            )
    for name, value in (
        ("training batch seed", training_batch_seed),
        ("fixed panel seed", fixed_panel_seed),
        ("training RNG seed", training_rng_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def validate_checkpoint_path(
    checkpoint_path: Path,
    *,
    source_path: Path | None,
) -> None:
    resolved = checkpoint_path.resolve()
    protected_paths = {
        DEFAULT_COMPLETE_DATA_PATH.resolve(),
        _STAGE_36.DEFAULT_CONTROL_CHECKPOINT_PATH.resolve(),
        _STAGE_36.DEFAULT_TREATMENT_CHECKPOINT_PATH.resolve(),
        (
            CHECKPOINT_DIRECTORY
            / "stage_35_last_two_global_context_128_best_checkpoint.pt"
        ).resolve(),
        (
            CHECKPOINT_DIRECTORY
            / "stage_35_last_two_global_context_256_best_checkpoint.pt"
        ).resolve(),
    }
    if source_path is not None:
        protected_paths.add(source_path.resolve())
    if resolved in protected_paths:
        raise ValueError(
            "Stage 37 checkpoint path collides with corpus or historical "
            f"artifact: {resolved}"
        )


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    return _STAGE_36.build_model(data, config, n_head, n_layer, device)


def build_layer_attention_masks(block_size: int) -> tuple[torch.Tensor, ...]:
    return _STAGE_36.build_layer_attention_masks(block_size)


def verify_model_masks(model: GPTLanguageModel) -> None:
    _STAGE_36.verify_model_masks(model)


def inspect_attention_shape(
    model: GPTLanguageModel,
    config: TrainingConfig,
    n_head: int,
    device: torch.device,
) -> tuple[int, ...]:
    control, _ = make_sampling_specs(config)
    return _STAGE_36.inspect_attention_shape(model, control, n_head, device)


def make_sampling_specs(
    config: TrainingConfig,
) -> tuple[object, object]:
    """Describe Stage 36's four-T512-to-eight-T256 sampling geometry."""

    control = _STAGE_36.ContextSpec(
        "Complete Shakespeare T256 treatment",
        config.batch_size,
        config.block_size,
        CHECKPOINT_DIRECTORY / ".stage_37_sampling_control.pt",
    )
    panel = _STAGE_36.ContextSpec(
        "sampling-only T512 panel",
        config.batch_size // 2,
        2 * config.block_size,
        CHECKPOINT_DIRECTORY / ".stage_37_sampling_panel.pt",
    )
    return control, panel


def get_stage_36_matched_batch(
    data: CharacterData,
    split: str,
    config: TrainingConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample long panels, then reshape them into the T256 control rows."""

    control, panel = make_sampling_specs(config)
    paired = _STAGE_36.get_paired_context_batch(
        data,
        split,
        control,
        panel,
        generator,
    )
    return paired.for_spec(control, device)


def evaluate_on_fixed_batches(
    model: GPTLanguageModel,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    *,
    fixed_panel_seed: int,
) -> LossEstimate:
    generator = torch.Generator().manual_seed(fixed_panel_seed)
    was_training = model.training
    model.eval()
    estimates: dict[str, float] = {}
    try:
        with torch.no_grad():
            for split in ("train", "val"):
                losses: list[float] = []
                for _ in range(config.eval_iters):
                    inputs, targets = get_stage_36_matched_batch(
                        data,
                        split,
                        config,
                        generator,
                        device,
                    )
                    _, loss = model(inputs, targets)
                    assert loss is not None
                    losses.append(loss.item())
                estimates[split] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return LossEstimate(train=estimates["train"], val=estimates["val"])


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def train_with_schedule(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    training_generator: torch.Generator,
    schedule: LearningRateSchedule,
    *,
    fixed_panel_seed: int,
    on_evaluation: Callable[[EvaluationRecord], None],
) -> TrainingResult:
    schedule.validate_target_step(config.max_iters)
    history: list[EvaluationRecord] = []

    def record(step: int) -> EvaluationRecord:
        evaluation = EvaluationRecord(
            step=step,
            losses=evaluate_on_fixed_batches(
                model,
                data,
                config,
                device,
                fixed_panel_seed=fixed_panel_seed,
            ),
        )
        history.append(evaluation)
        on_evaluation(evaluation)
        return evaluation

    model.train()
    initial_record = record(0)
    for update_index in range(config.max_iters):
        learning_rate = schedule.for_update(update_index)
        _set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_stage_36_matched_batch(
            data,
            "train",
            config,
            training_generator,
            device,
        )
        logits, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()
        del logits
        del loss
        del inputs
        del targets

        completed_step = update_index + 1
        if (
            completed_step % config.eval_interval == 0
            or completed_step == schedule.first_decay_step
            or completed_step == schedule.second_decay_step
            or completed_step == config.max_iters
        ):
            record(completed_step)

    return TrainingResult(
        initial=initial_record.losses,
        final=history[-1].losses,
        history=tuple(history),
        benchmark=None,
    )


@dataclass(slots=True)
class Stage37ValidationCheckpoint(BestValidationCheckpoint):
    """Save the best Complete Shakespeare state and its data-scale contract."""

    schedule: LearningRateSchedule | None = None
    target_step: int = 0
    initial_state_sha256: str = ""
    source_filename: str | None = None
    source_bytes: int | None = None
    source_sha256: str | None = None
    num_characters: int = 0
    train_characters: int = 0
    val_characters: int = 0
    vocabulary_size: int = 0
    train_fraction: float = 0.0
    targets_per_update: int = 0
    total_target_presentations: int = 0
    training_batch_seed: int = 0
    fixed_panel_seed: int = 0
    training_rng_seed: int = 0
    parameter_count: int = 0
    attention_shape: tuple[int, ...] = ()
    logical_edges_by_layer: tuple[int, ...] = ()
    pre_global_effective_receptive_field: int = 0
    best_train_loss: float | None = None

    def _payload(self, step: int) -> dict[str, Any]:
        payload = super(Stage37ValidationCheckpoint, self)._payload(step)
        if self.schedule is None or self.best_train_loss is None:
            raise RuntimeError("incomplete Stage 37 checkpoint metadata")

        optimizer_rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(optimizer_rates) != 1:
            raise RuntimeError("all optimizer groups must use one LR")
        optimizer_rate = optimizer_rates.pop()
        expected_rate = self.schedule.for_update(max(0, step - 1))
        if optimizer_rate != expected_rate:
            raise RuntimeError(
                f"optimizer LR at step {step} must be {expected_rate:g}, "
                f"got {optimizer_rate:g}"
            )

        schedule_metadata = self.schedule.as_metadata(max_iters=self.target_step)
        payload["checkpoint_kind"] = "best"
        architecture = payload["architecture"]
        assert isinstance(architecture, dict)
        architecture.update(
            {
                "head_size": self.config.n_embd // self.n_head,
                "feed_forward_width": 4 * self.config.n_embd,
                "residual_dropout": DEFAULT_DROPOUT,
                "dropout_placement": _STAGE_36.DROPOUT_PLACEMENT,
                "layer_contexts": DEFAULT_LAYER_CONTEXTS,
                "topology": "LLLLGG",
                "vocabulary_size": self.vocabulary_size,
                "parameter_count": self.parameter_count,
                "attention_mask": "W23,W23,W23,W23,Full,Full",
                "attention_mask_scope": "per_layer",
                "local_context": DEFAULT_LOCAL_CONTEXT,
                "global_layer_indices": DEFAULT_GLOBAL_LAYER_INDICES,
                "global_layer_numbers": DEFAULT_GLOBAL_LAYER_NUMBERS,
                "pre_global_effective_receptive_field": (
                    self.pre_global_effective_receptive_field
                ),
                "logical_edges_by_layer_per_head": self.logical_edges_by_layer,
                "attention_shape": self.attention_shape,
            }
        )
        training_config = payload["training_config"]
        assert isinstance(training_config, dict)
        training_config.update(
            {
                "learning_rate": optimizer_rate,
                "initial_learning_rate": self.schedule.initial_learning_rate,
                "max_iters": self.target_step,
                "learning_rate_schedule": schedule_metadata,
                "residual_dropout": DEFAULT_DROPOUT,
                "training_batch_seed": self.training_batch_seed,
                "fixed_panel_seed": self.fixed_panel_seed,
                "training_rng_seed": self.training_rng_seed,
                "targets_per_update": self.targets_per_update,
                "total_target_presentations": self.total_target_presentations,
            }
        )
        payload["corpus"] = {
            "name": "Complete Shakespeare",
            "source_filename": self.source_filename,
            "source_bytes": self.source_bytes,
            "source_sha256": self.source_sha256,
            "data_fingerprint": self.data_fingerprint,
            "decoded_characters": self.num_characters,
            "train_characters": self.train_characters,
            "validation_characters": self.val_characters,
            "train_fraction": self.train_fraction,
            "vocabulary_size": self.vocabulary_size,
            "train_corpus_equivalents": (
                self.total_target_presentations / self.train_characters
            ),
            "train_pool_multiplier_vs_tiny": (
                self.train_characters / HISTORICAL_CONTROL_TRAIN_CHARACTERS
            ),
        }
        payload["initialization"] = {
            "kind": "fresh_corpus_specific_initialization",
            "seed": self.config.seed,
            "state_sha256": self.initial_state_sha256,
            "reused_tiny_shakespeare_weights": False,
            "cross_corpus_initialization_identity_claimed": False,
            "reason": "vocabulary and output tensor shapes differ",
        }
        payload["historical_control"] = {
            "source_stage": HISTORICAL_CONTROL_STAGE,
            "branch": HISTORICAL_CONTROL_BRANCH,
            "corpus": HISTORICAL_CONTROL_CORPUS,
            "rerun_by_stage_37": HISTORICAL_CONTROL_RERUN,
            "train_characters": HISTORICAL_CONTROL_TRAIN_CHARACTERS,
            "validation_characters": HISTORICAL_CONTROL_VAL_CHARACTERS,
            "vocabulary_size": HISTORICAL_CONTROL_VOCAB_SIZE,
            "target_presentations": DEFAULT_TOTAL_TARGET_CHARACTERS,
            "selected_step": HISTORICAL_CONTROL_SELECTED_STEP,
            "selected_train_loss": HISTORICAL_CONTROL_SELECTED_TRAIN_LOSS,
            "selected_validation_loss": HISTORICAL_CONTROL_SELECTED_VAL_LOSS,
            "final_step": HISTORICAL_CONTROL_FINAL_STEP,
            "final_train_loss": HISTORICAL_CONTROL_FINAL_TRAIN_LOSS,
            "final_validation_loss": HISTORICAL_CONTROL_FINAL_VAL_LOSS,
            "checkpoint_sha256": HISTORICAL_CONTROL_CHECKPOINT_SHA256,
        }
        payload["experiment"] = {
            "stage": EXPERIMENT_STAGE,
            "title": EXPERIMENT_TITLE,
            "protocol": PROTOCOL_NAME,
            "branch": "Complete Shakespeare treatment",
            "trained_branches": 1,
            "from_scratch": True,
            "comparison_variable": COMPARISON_VARIABLE,
            "corpus_size_isolated": False,
            "historical_control_only": True,
            "tiny_shakespeare_rerun": False,
            "same_architecture_as_stage_36_t256": True,
            "same_update_count_as_historical_control": (
                self.target_step == HISTORICAL_CONTROL_FINAL_STEP
            ),
            "same_targets_per_update_as_historical_control": (
                self.targets_per_update == DEFAULT_TARGETS_PER_UPDATE
            ),
            "same_total_target_presentations_as_historical_control": (
                self.total_target_presentations == DEFAULT_TOTAL_TARGET_CHARACTERS
            ),
            "raw_loss_cross_corpus_comparable": RAW_LOSS_CROSS_CORPUS_COMPARABLE,
            "vocabulary_matched": False,
            "validation_split_matched": False,
            "training_stream_paired": False,
            "training_sampler_geometry_matched": True,
            "evaluation_panel_geometry_matched": True,
            "comparison_caveats": (
                "vocabulary_size_differs",
                "validation_text_distribution_differs",
                "historical_control_uses_a_different_training_stream",
                "single_training_seed_per_corpus",
                "backend_or_device_may_differ_from_historical_control",
            ),
        }
        if self.device.type == "xpu":
            device_name = torch.xpu.get_device_name(self.device.index or 0)
        elif self.device.type == "cuda":
            device_name = torch.cuda.get_device_name(self.device.index or 0)
        else:
            device_name = self.device.type
        payload["runtime"] = {
            "device_type": self.device.type,
            "device_name": device_name,
            "torch_version": str(torch.__version__),
            "historical_control_device_name": HISTORICAL_CONTROL_DEVICE,
            "device_matched_to_historical_control": (
                self.device.type == "xpu" and device_name == HISTORICAL_CONTROL_DEVICE
            ),
        }
        payload["evaluation_protocol"] = {
            "selection_metric": "overall_fixed_panel_validation_loss",
            "fixed_panel_seed": self.fixed_panel_seed,
            "eval_iters": self.config.eval_iters,
            "strict_improvement": True,
            "dropout_enabled": False,
        }
        payload["best_training_loss"] = self.best_train_loss
        return payload

    def consider(self, record: EvaluationRecord) -> bool:
        if not math.isfinite(record.losses.val):
            raise ValueError(f"validation loss at step {record.step} must be finite")
        if record.losses.val >= self.best_val_loss:
            return False
        self.best_val_loss = record.losses.val
        self.best_step = record.step
        self.best_train_loss = record.losses.train
        self.save(record.step)
        return True


def make_evaluation_callback(
    best: Stage37ValidationCheckpoint,
    schedule: LearningRateSchedule,
) -> Callable[[EvaluationRecord], None]:
    def record_evaluation(record: EvaluationRecord) -> None:
        is_best = best.consider(record)
        assert best.best_step is not None
        learning_rate = schedule.for_update(max(0, record.step - 1))
        marker = "  * best" if is_best else ""
        print(
            f"  step {record.step:5d} | lr {learning_rate:g} | "
            f"train {record.losses.train:.4f} | "
            f"val {record.losses.val:.4f} | "
            f"gap {record.losses.val - record.losses.train:.4f} | "
            f"best {best.best_val_loss:.4f} @ {best.best_step}{marker}"
        )

    return record_evaluation


def validate_stage_37_checkpoint(
    path: Path,
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    schedule: LearningRateSchedule,
    *,
    training_batch_seed: int | None = None,
    fixed_panel_seed: int | None = None,
    training_rng_seed: int | None = None,
) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage 37 checkpoint must be a mapping")
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported Stage 37 checkpoint version")
    if checkpoint.get("data_fingerprint") != fingerprint_data(data):
        raise ValueError("Stage 37 checkpoint corpus fingerprint mismatch")

    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("Stage 37 checkpoint architecture is missing")
    masks = build_layer_attention_masks(config.block_size)
    audit = _STAGE_36.audit_layerwise_reachability(masks)
    logical_edges = tuple(int(mask.sum().item()) for mask in masks)
    expected_architecture = {
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "head_size": config.n_embd // n_head,
        "feed_forward_width": 4 * config.n_embd,
        "residual_dropout": DEFAULT_DROPOUT,
        "layer_contexts": DEFAULT_LAYER_CONTEXTS,
        "topology": "LLLLGG",
        "vocabulary_size": data.vocabulary.size,
        "attention_mask": "W23,W23,W23,W23,Full,Full",
        "attention_mask_scope": "per_layer",
        "local_context": DEFAULT_LOCAL_CONTEXT,
        "global_layer_indices": DEFAULT_GLOBAL_LAYER_INDICES,
        "global_layer_numbers": DEFAULT_GLOBAL_LAYER_NUMBERS,
        "pre_global_effective_receptive_field": audit.widths_by_layer[3],
        "logical_edges_by_layer_per_head": logical_edges,
        "attention_shape": (
            config.batch_size,
            n_head,
            config.block_size,
            config.block_size,
        ),
    }
    tuple_fields = {
        "layer_contexts",
        "global_layer_indices",
        "global_layer_numbers",
        "logical_edges_by_layer_per_head",
        "attention_shape",
    }
    for key, expected in expected_architecture.items():
        observed = architecture.get(key)
        if key in tuple_fields and observed is not None:
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(
                f"Stage 37 checkpoint architecture {key!r} mismatch: "
                f"expected {expected!r}, got {observed!r}"
            )

    experiment = checkpoint.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("Stage 37 experiment metadata is missing")
    if experiment.get("stage") != EXPERIMENT_STAGE:
        raise ValueError("checkpoint is not a Stage 37 artifact")
    if experiment.get("tiny_shakespeare_rerun") is not False:
        raise ValueError("Stage 37 must not claim a TinyShakespeare rerun")
    if experiment.get("raw_loss_cross_corpus_comparable") is not False:
        raise ValueError("Stage 37 must preserve the raw-loss comparison caveat")
    if experiment.get("trained_branches") != 1:
        raise ValueError("Stage 37 checkpoint must describe exactly one branch")
    if experiment.get("historical_control_only") is not True:
        raise ValueError("Stage 37 control must remain historical-only")
    for key, expected in (
        ("vocabulary_matched", False),
        ("validation_split_matched", False),
        ("training_stream_paired", False),
        ("training_sampler_geometry_matched", True),
        ("evaluation_panel_geometry_matched", True),
    ):
        if experiment.get(key) is not expected:
            raise ValueError(f"Stage 37 experiment flag {key!r} changed")

    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError("Stage 37 training metadata is missing")
    if training_config.get("max_iters") != config.max_iters:
        raise ValueError("Stage 37 target step mismatch")
    expected_targets_per_update = config.batch_size * config.block_size
    if training_config.get("targets_per_update") != expected_targets_per_update:
        raise ValueError("Stage 37 targets-per-update mismatch")
    if training_config.get("total_target_presentations") != (
        expected_targets_per_update * config.max_iters
    ):
        raise ValueError("Stage 37 total target-presentation mismatch")
    if training_config.get("learning_rate_schedule") != schedule.as_metadata(
        max_iters=config.max_iters
    ):
        raise ValueError("Stage 37 learning-rate schedule mismatch")
    for key, expected in (
        ("training_batch_seed", training_batch_seed),
        ("fixed_panel_seed", fixed_panel_seed),
        ("training_rng_seed", training_rng_seed),
    ):
        if expected is not None and training_config.get(key) != expected:
            raise ValueError(f"Stage 37 {key.replace('_', ' ')} mismatch")

    corpus = checkpoint.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("Stage 37 corpus metadata is missing")
    expected_corpus = {
        "data_fingerprint": fingerprint_data(data),
        "decoded_characters": data.num_characters,
        "train_characters": len(data.train_data),
        "validation_characters": len(data.val_data),
        "vocabulary_size": data.vocabulary.size,
    }
    for key, expected in expected_corpus.items():
        if corpus.get(key) != expected:
            raise ValueError(f"Stage 37 corpus field {key!r} mismatch")

    historical = checkpoint.get("historical_control")
    if not isinstance(historical, Mapping):
        raise ValueError("Stage 37 historical-control metadata is missing")
    expected_historical = {
        "source_stage": HISTORICAL_CONTROL_STAGE,
        "branch": HISTORICAL_CONTROL_BRANCH,
        "corpus": HISTORICAL_CONTROL_CORPUS,
        "rerun_by_stage_37": False,
        "train_characters": HISTORICAL_CONTROL_TRAIN_CHARACTERS,
        "validation_characters": HISTORICAL_CONTROL_VAL_CHARACTERS,
        "vocabulary_size": HISTORICAL_CONTROL_VOCAB_SIZE,
        "target_presentations": DEFAULT_TOTAL_TARGET_CHARACTERS,
        "selected_step": HISTORICAL_CONTROL_SELECTED_STEP,
        "selected_train_loss": HISTORICAL_CONTROL_SELECTED_TRAIN_LOSS,
        "selected_validation_loss": HISTORICAL_CONTROL_SELECTED_VAL_LOSS,
        "final_step": HISTORICAL_CONTROL_FINAL_STEP,
        "final_train_loss": HISTORICAL_CONTROL_FINAL_TRAIN_LOSS,
        "final_validation_loss": HISTORICAL_CONTROL_FINAL_VAL_LOSS,
        "checkpoint_sha256": HISTORICAL_CONTROL_CHECKPOINT_SHA256,
    }
    for key, expected in expected_historical.items():
        if historical.get(key) != expected:
            raise ValueError(f"Stage 37 historical-control field {key!r} mismatch")

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("Stage 37 model state is missing")
    model = build_model(data, config, n_head, n_layer)
    model.load_state_dict(model_state, strict=True)
    verify_model_masks(model)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if architecture.get("parameter_count") != parameter_count:
        raise ValueError("Stage 37 checkpoint parameter count mismatch")
    initialization = checkpoint.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("Stage 37 initialization metadata is missing")
    if initialization.get("reused_tiny_shakespeare_weights") is not False:
        raise ValueError("Stage 37 must use a fresh corpus-specific initialization")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        initial_model = build_model(data, config, n_head, n_layer)
        expected_initial_state = tensor_mapping_sha256(initial_model.state_dict())
    if initialization.get("state_sha256") != expected_initial_state:
        raise ValueError("Stage 37 initialization hash mismatch")
    if checkpoint.get("step") != checkpoint.get("best_step"):
        raise ValueError("Stage 37 artifact must contain its selected-best step")
    return checkpoint


def run_experiment(
    data: CharacterData,
    config: TrainingConfig,
    n_head: int,
    n_layer: int,
    device: torch.device,
    sample_length: int,
    schedule: LearningRateSchedule,
    checkpoint_path: Path,
    *,
    source_path: Path | None,
    training_batch_seed: int,
    fixed_panel_seed: int,
    training_rng_seed: int,
) -> Stage37Report:
    if sample_length < 0:
        raise ValueError("sample-length must be non-negative")
    validate_protocol(
        data,
        config,
        n_head,
        n_layer,
        schedule,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    validate_checkpoint_path(checkpoint_path, source_path=source_path)
    corpus = summarize_corpus(data, config, source_path=source_path)

    canonical = is_canonical_protocol(
        data,
        config,
        n_head,
        n_layer,
        schedule,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
    )
    if canonical:
        expected_corpus = (
            DEFAULT_COMPLETE_NUM_CHARACTERS,
            DEFAULT_COMPLETE_TRAIN_CHARACTERS,
            DEFAULT_COMPLETE_VAL_CHARACTERS,
            DEFAULT_COMPLETE_VOCAB_SIZE,
            DEFAULT_TOTAL_TARGET_CHARACTERS,
        )
        observed_corpus = (
            corpus.num_characters,
            corpus.train_characters,
            corpus.val_characters,
            corpus.vocabulary_size,
            corpus.total_target_presentations,
        )
        if observed_corpus != expected_corpus:
            raise RuntimeError(
                "canonical Complete Shakespeare corpus changed: "
                f"expected {expected_corpus!r}, got {observed_corpus!r}"
            )

    clear_accelerator_cache(device)
    seed_everything(config.seed)
    model = build_model(data, config, n_head, n_layer, device)
    verify_model_masks(model)
    initial_state_sha256 = tensor_mapping_sha256(model.state_dict())
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    attention_shape = inspect_attention_shape(model, config, n_head, device)
    masks = build_layer_attention_masks(config.block_size)
    audit = _STAGE_36.audit_layerwise_reachability(masks)
    logical_edges = tuple(int(mask.sum().item()) for mask in masks)
    if canonical:
        if parameter_count != DEFAULT_PARAMETER_COUNT:
            raise RuntimeError(
                "canonical Stage 37 parameter count changed: "
                f"expected {DEFAULT_PARAMETER_COUNT:,}, got {parameter_count:,}"
            )
        if initial_state_sha256 != DEFAULT_INITIAL_STATE_SHA256:
            raise RuntimeError(
                "canonical Stage 37 initialization changed: expected "
                f"{DEFAULT_INITIAL_STATE_SHA256}, got {initial_state_sha256}"
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.for_update(0))
    training_generator = torch.Generator().manual_seed(training_batch_seed)
    best = Stage37ValidationCheckpoint(
        model=model,
        optimizer=optimizer,
        training_generator=training_generator,
        path=checkpoint_path,
        device=device,
        config=config,
        n_head=n_head,
        n_layer=n_layer,
        data_fingerprint=corpus.data_fingerprint,
        optimizer_restart_step=None,
        optimizer_provenance_known=True,
        schedule=schedule,
        target_step=config.max_iters,
        initial_state_sha256=initial_state_sha256,
        source_filename=(
            None if corpus.source_path is None else corpus.source_path.name
        ),
        source_bytes=corpus.source_bytes,
        source_sha256=corpus.source_sha256,
        num_characters=corpus.num_characters,
        train_characters=corpus.train_characters,
        val_characters=corpus.val_characters,
        vocabulary_size=corpus.vocabulary_size,
        train_fraction=corpus.train_fraction,
        targets_per_update=corpus.targets_per_update,
        total_target_presentations=corpus.total_target_presentations,
        training_batch_seed=training_batch_seed,
        fixed_panel_seed=fixed_panel_seed,
        training_rng_seed=training_rng_seed,
        parameter_count=parameter_count,
        attention_shape=attention_shape,
        logical_edges_by_layer=logical_edges,
        pre_global_effective_receptive_field=audit.widths_by_layer[3],
    )
    record_evaluation = make_evaluation_callback(best, schedule)

    print("\nComplete Shakespeare treatment (the only Stage 37 training branch)")
    print(
        f"  B={config.batch_size}, T={config.block_size}, C={config.n_embd}, "
        f"H={n_head}, D={config.n_embd // n_head}, "
        f"FF={4 * config.n_embd}, L={n_layer}, dropout={DEFAULT_DROPOUT:g}"
    )
    print("  attention: W23, W23, W23, W23, Full, Full")
    print(f"  parameters:             {parameter_count:,}")
    print(f"  initial state SHA-256:  {initial_state_sha256}")
    print(f"  training-batch seed:    {training_batch_seed}")
    print(f"  fixed-panel seed:       {fixed_panel_seed}")
    print(f"  training RNG seed:      {training_rng_seed}")

    # Construction consumes CPU RNG. Reset immediately before the initial
    # evaluation/training boundary so dropout has a reproducible stream.
    seed_everything(training_rng_seed)
    result = train_with_schedule(
        model,
        optimizer,
        data,
        config,
        device,
        training_generator,
        schedule,
        fixed_panel_seed=fixed_panel_seed,
        on_evaluation=record_evaluation,
    )
    assert best.best_step is not None
    assert best.best_train_loss is not None
    final_batch_generator_sha256 = _STAGE_36.tensor_sha256(
        training_generator.get_state()
    )
    selected_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    selected_state = selected_checkpoint.get("model_state_dict")
    if not isinstance(selected_state, Mapping):
        raise RuntimeError("selected Stage 37 checkpoint has no model state")
    model.load_state_dict(selected_state, strict=True)
    verify_model_masks(model)
    sample = generate_from_final_model(
        model,
        data,
        device,
        sample_length=sample_length,
        seed=DEFAULT_SAMPLE_SEED,
    )
    artifact_sha256 = checkpoint_sha256(checkpoint_path)

    report = Stage37Report(
        corpus=corpus,
        historical_control=DEFAULT_HISTORICAL_CONTROL,
        parameter_count=parameter_count,
        initial_state_sha256=initial_state_sha256,
        attention_shape=attention_shape,
        training=result,
        best_train_loss=best.best_train_loss,
        best_val_loss=best.best_val_loss,
        best_step=best.best_step,
        final_batch_generator_sha256=final_batch_generator_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=artifact_sha256,
        sample=sample,
    )
    del record_evaluation
    del best
    del optimizer
    del model
    clear_accelerator_cache(device)
    return report


def print_protocol_summary(
    corpus: CorpusSummary,
    historical: HistoricalControl = DEFAULT_HISTORICAL_CONTROL,
) -> None:
    print("\nEqual-compute data-scaling protocol")
    print(
        f"  historical control: Stage {historical.source_stage} "
        f"{historical.branch} on {historical.corpus} (not rerun)"
    )
    print("  treatment:          Complete Shakespeare (fresh Stage 37 run)")
    print(f"  targets/update:      {corpus.targets_per_update:,}")
    print(
        f"  updates:             {corpus.total_target_presentations // corpus.targets_per_update:,}"
    )
    print(f"  target presentations:{corpus.total_target_presentations:13,}")
    print(
        f"  train pools:         {historical.train_characters:,} -> "
        f"{corpus.train_characters:,} characters "
        f"({corpus.train_pool_multiplier_vs_tiny:.4f}x)"
    )
    print(
        f"  corpus-equivalents:  {historical.train_corpus_equivalents:.4f} -> "
        f"{corpus.train_corpus_equivalents:.4f}"
    )
    print(
        f"  vocabularies:        {historical.vocabulary_size} -> "
        f"{corpus.vocabulary_size} characters"
    )
    print(
        "  comparison caveat:  raw validation losses are not a paired delta; "
        "vocabulary and held-out text distribution differ"
    )


def print_final_summary(report: Stage37Report) -> None:
    control = report.historical_control
    print("\nStage 37 summary")
    print(
        f"  historical Tiny selected: train {control.selected_train_loss:.4f} / "
        f"val {control.selected_val_loss:.4f} @ {control.selected_step:,}"
    )
    print(
        f"  historical Tiny final:    train {control.final_train_loss:.4f} / "
        f"val {control.final_val_loss:.4f} @ {control.final_step:,}"
    )
    print(
        f"  Complete selected:         train {report.best_train_loss:.4f} / "
        f"val {report.best_val_loss:.4f} @ {report.best_step:,}"
    )
    print(
        f"  Complete final:            train {report.training.final.train:.4f} / "
        f"val {report.training.final.val:.4f} @ "
        f"{report.training.history[-1].step:,}"
    )
    print(f"  Complete final gap:        {report.generalization_gap:.4f}")
    print(
        f"  val reversal from best:    {report.validation_reversal:+.4f} "
        "(positive means validation worsened)"
    )
    print("  raw cross-corpus delta:    intentionally not reported")
    print(f"  best checkpoint:           {report.checkpoint_path}")
    print(f"  checkpoint SHA-256:        {report.checkpoint_sha256}")
    print("\nComplete Shakespeare selected-best generated text")
    print(report.sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 37: train the locked B8/T256 LLLLGG model only on Complete "
            "Shakespeare for the same 36,864,000 target presentations as the "
            "historical TinyShakespeare control."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_COMPLETE_DATA_PATH)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
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
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_EVAL_ITERS)
    parser.add_argument("--sample-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--training-batch-seed", type=int, default=DEFAULT_TRAINING_BATCH_SEED
    )
    parser.add_argument(
        "--fixed-panel-seed", type=int, default=DEFAULT_FIXED_PANEL_SEED
    )
    parser.add_argument(
        "--training-rng-seed", type=int, default=DEFAULT_TRAINING_RNG_SEED
    )
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    data = CharacterData.from_file(args.data, train_fraction=args.train_fraction)
    device = resolve_device(args.device)
    corpus = summarize_corpus(data, config, source_path=args.data)

    print(f"Stage {EXPERIMENT_STAGE}: {EXPERIMENT_TITLE}")
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"Accelerator: {torch.xpu.get_device_name(0)}")
    print(f"Source: {args.data.resolve()}")
    print(f"Decoded characters: {corpus.num_characters:,}")
    print(
        f"Train / validation: {corpus.train_characters:,} / {corpus.val_characters:,}"
    )
    print(f"Vocabulary size: {corpus.vocabulary_size}")
    print(f"Uniform-loss baseline: {math.log(corpus.vocabulary_size):.4f}")
    print_protocol_summary(corpus)

    report = run_experiment(
        data,
        config,
        args.n_head,
        args.n_layer,
        device,
        args.sample_length,
        schedule,
        args.checkpoint_path,
        source_path=args.data,
        training_batch_seed=args.training_batch_seed,
        fixed_panel_seed=args.fixed_panel_seed,
        training_rng_seed=args.training_rng_seed,
    )
    validate_stage_37_checkpoint(
        report.checkpoint_path,
        data,
        config,
        args.n_head,
        args.n_layer,
        schedule,
        training_batch_seed=args.training_batch_seed,
        fixed_panel_seed=args.fixed_panel_seed,
        training_rng_seed=args.training_rng_seed,
    )
    print_final_summary(report)


if __name__ == "__main__":
    main()
