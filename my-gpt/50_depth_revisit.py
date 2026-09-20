"""Stage 50: spend the next parameter budget on depth rather than on width.

Stages 43-49 resolved the C256 champion to the point where neither width nor
dropout moved the frontier: C320 loses to C256 at every measured dropout, and
C256's own dropout bracket is closed at p=0.15. The remaining question is how
that mature model should spend a larger parameter budget.

Control:   C=256, H=4, D=64, FF=1024, L=6, T=256, contexts LLLLGG, p=0.15.
Treatment: C=256, H=4, D=64, FF=1024, L=8, T=256, contexts LLLLLLGG, p=0.15.

Both branches keep every W23 local layer and the two consecutive final Full
layers that Stages 32 and 34 established as the topology; the treatment inserts
two more W23 layers before the same two global refiners. Each C256 block costs
788,992 parameters, so the treatment is 4,850,274 -> 6,428,258 parameters
(+32.53%), which places it between the C256 L=6 champion and the 7,537,378
parameter C320 L=6 model the budget shootout scores against it.

The Stage 43-49 protocol is reused unchanged: architecture-native initialization
from seed 1,337 on the CPU, the same sampler and Torch RNG seeds, the same
50,000-update / 102.4-million-target-presentation budget, the same fixed
evaluation panel, and the same lineage device. Both branches are therefore
trained on one batch stream, and the control is required to reproduce the stored
Stage 49 C256 p=0.15 champion byte for byte.
"""

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
from typing import Any, Mapping, Sequence

import torch

from config import TrainingConfig
from data_utils import CharacterData
from training import EvaluationRecord, LossEstimate, TrainingResult


_SPEC = importlib.util.spec_from_file_location(
    "stage_49_for_stage_50", Path(__file__).with_name("49_dropout_strength.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE_49 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE_49
_SPEC.loader.exec_module(_STAGE_49)
_STAGE_48 = _STAGE_49._STAGE_48
_STAGE_46 = _STAGE_49._STAGE_46
_STAGE_44 = _STAGE_49._STAGE_44
_STAGE_42 = _STAGE_49._STAGE_42
_STAGE_41 = _STAGE_49._STAGE_41
_STAGE_40 = _STAGE_49._STAGE_40

# The architecture owners for this lineage: Stage 32 owns the layerwise attention
# masks and the spec that names them, Stage 36 wrapped that factory in the
# LLLLGG T=256 topology every stage since 43 has trained, and Stage 46 exposed
# residual dropout on top of it. Depth is the one dimension none of them
# parameterized: Stage 46's factory hard-codes Stage 36's six-layer mask stack,
# so this stage re-issues the Stage 32 construction directly and proves it still
# reproduces the six-layer model byte for byte.
_STAGE_36 = _STAGE_46._STAGE_36
_STAGE_32 = _STAGE_36._STAGE_32
_STAGE_30 = _STAGE_32._STAGE_30
GPTLanguageModel = _STAGE_36.GPTLanguageModel
AttentionSpec = _STAGE_32.AttentionSpec
build_layer_attention_masks = _STAGE_32.build_layer_attention_masks
apply_attention_masks = _STAGE_32.apply_attention_masks
audit_layerwise_reachability = _STAGE_32.audit_layerwise_reachability
verify_global_reachability = _STAGE_32.verify_global_reachability

# Reuse the sampler, panel geometry, schedule, statistics, device resolution,
# hashing, and serialization unchanged. Only the architecture under test and the
# artifact identity change, so this stage cannot drift from the Stage 43-49
# training protocol.
get_batch = _STAGE_49.get_batch
evaluate_on_fixed_batches = _STAGE_49.evaluate_on_fixed_batches
capture_rng_state = _STAGE_49.capture_rng_state
state_hash = _STAGE_49.state_hash
file_hash = _STAGE_49.file_hash
fingerprint_data = _STAGE_49.fingerprint_data
stream_hashes = _STAGE_49.stream_hashes
make_schedule = _STAGE_49.make_schedule
schedule_metadata = _STAGE_49.schedule_metadata
resolve_lineage_device = _STAGE_49.resolve_lineage_device
verify_lineage_device = _STAGE_49.verify_lineage_device
normalized_rng_sha256 = _STAGE_49.normalized_rng_sha256
terminal_rng_normalization = _STAGE_49.terminal_rng_normalization

EXPERIMENT_STAGE = 50
PROTOCOL_NAME = "paired_architecture_native_depth"
DEFAULT_WIDTH = 256
DEFAULT_CONTROL_DEPTH = _STAGE_44.N_LAYER
DEFAULT_TREATMENT_DEPTH = 8
DEFAULT_DROPOUT = _STAGE_49.DEFAULT_CONTROL_DROPOUT
DEFAULT_LOCAL_CONTEXT = 23
DEFAULT_CONTROL_LAYER_CONTEXTS = (23, 23, 23, 23, None, None)
DEFAULT_TREATMENT_LAYER_CONTEXTS = (23, 23, 23, 23, 23, 23, None, None)
# The smallest T the mask builder is ever asked for in a structural check; the
# local window must stay below it.
_STRUCTURAL_BLOCK_SIZE = 24
DEFAULT_TARGET_STEP = 50_000
DEFAULT_FRESH_VALIDATION_ITERS = 500
# Every panel seed from 1,339 through 1,367 was consumed by Stages 43-49.
CONSUMED_PANEL_SEEDS = (1_339, *range(1_354, 1_368))
DEFAULT_FRESH_VALIDATION_SEED = 1_368
DEFAULT_BUDGET_PANEL_SEED = 1_369
DEFAULT_FIXED_PANEL_SEED = _STAGE_44.DEFAULT_FIXED_PANEL_SEED
DEFAULT_TRAINING_BATCH_SEED = _STAGE_44.DEFAULT_TRAINING_BATCH_SEED
DEFAULT_TRAINING_RNG_SEED = _STAGE_44.DEFAULT_TRAINING_RNG_SEED
DEFAULT_OUTPUT_DIRECTORY = _STAGE_41.CHECKPOINT_DIRECTORY
DEFAULT_COMPLETE_DATA_PATH = _STAGE_41.DEFAULT_COMPLETE_DATA_PATH
# The lineage device is pinned by name for the same reason Stage 49 pinned it:
# kernel selection is device-dependent, and this stage's control must reproduce a
# stored endpoint that was trained on the recorded accelerator.
DEFAULT_LINEAGE_DEVICE_NAME = _STAGE_49.DEFAULT_LINEAGE_DEVICE_NAME
# Architecture-native initialization fingerprints for the runtime that produced
# the recorded artifacts (torch 2.14.0+cu130). Residual dropout adds no
# parameters and consumes no initialization RNG, so one fingerprint covers both
# probabilities at a width; depth changes both the parameter count and the
# construction, so each depth has its own. The six-layer fingerprint is the one
# every stage since 45 has recorded at C256; the eight-layer fingerprint is new
# here and is proved two ways in preflight: through this stage's factory and
# through Stage 30's own unchanged factory.
C256_L6_NATIVE_INITIALIZATION_SHA256 = _STAGE_49.C256_NATIVE_INITIALIZATION_SHA256
C256_L8_NATIVE_INITIALIZATION_SHA256 = (
    "6bef5bb6fa54e908ca06f64d299b1c3b35db7cf8676a8ade7d33c2b9cdb2f8c3"
)
DEFAULT_CONTROL_PARAMETER_COUNT = 4_850_274
DEFAULT_TREATMENT_PARAMETER_COUNT = 6_428_258
DEFAULT_C320_PARAMETER_COUNT = 7_537_378
DEFAULT_BLOCK_PARAMETER_COUNT = 788_992
# A read-only cross-stage reference: Stage 49's fresh C256 p=0.15 control is the
# champion this stage's control must re-derive from zero. Stage 49 also
# reproduced Stage 48's treatment with it, so matching it again puts this stage's
# control on the trajectory of two stored endpoints.
DEFAULT_STAGE_49_CONTROL_FILENAME = (
    "stage_49_control_c256_p015_step_50000_final_checkpoint.pt"
)
DEFAULT_STAGE_49_CONTROL_SHA256 = (
    "8aaf70ae60e546168a976ecc640a6b1dac3044d135ccf7f9d822dfa5598affec"
)
DEFAULT_STAGE_49_CONTROL_BEST_FILENAME = "stage_49_control_c256_p015_best_checkpoint.pt"
DEFAULT_STAGE_49_CONTROL_BEST_SHA256 = (
    "a73a9912e150a2ba56aa29568f1abef3afd466dd8bb36aeccf973e9effa51c18"
)
DEFAULT_STAGE_49_CONTROL_BEST_STEP = 48_500
DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_TRAIN_LOSS = 1.087766051888466
DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_VALIDATION_LOSS = 1.3507258701324463
# One sampler, one seed, one batch shape. Both fresh branches and the stored C320
# peer were trained on this stream; the fresh pair's own states are re-derived
# here rather than assumed.
DEFAULT_SHARED_TRAINING_GENERATOR_SHA256 = (
    _STAGE_49.DEFAULT_SHARED_TRAINING_GENERATOR_SHA256
)
DEFAULT_SHARED_INITIAL_GENERATOR_SHA256 = (
    "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced"
)
# The provenance of the stored C320 peer: Stage 46's p=0.15 treatment, which is
# the wider spend this stage's depth treatment is measured against. It is
# authenticated by Stage 49's own loader, which delegates to Stage 46's
# validator, and by the file hash Stage 49 recorded for it.
C320_BUDGET_REF = _STAGE_49.C320_CHAMPION_REFS[0]
BUDGET_PEER_KEY = "c320_l6"
# The original depth result, from the same lineage's early, small-data,
# narrow-width era. Transcribed from the Stage 24 report; Stage 50 never
# retrains or re-reads it. It is the prior that motivates re-asking the question
# at the mature operating point, not a comparable measurement.
PRIOR_DEPTH_REFERENCE = {
    "source_stage": 24,
    "comparison": "L=4 -> L=6 at C=128, T=64, p=0.10",
    "control_parameter_count": 816_705,
    "treatment_parameter_count": 1_212_481,
    "parameter_ratio": 1.4846,
    "updates": 18_000,
    "eval_iters": 200,
    "fresh_panel_seed": None,
    "mean_delta": -0.011038,
    "standard_error": 0.000647,
    "confidence_low": -0.012309,
    "confidence_high": -0.009766,
    "role": "transcribed prior; a different corpus size, width, depth, and budget",
}
# Provenance references only. Stage 50 never initializes from these artifacts;
# the stored endpoints are read once, read-only, either to be reproduced or to
# be scored against.
HISTORICAL_LINEAGE = {
    **copy.deepcopy(_STAGE_49.HISTORICAL_LINEAGE),
    "stage_49_c256_p015_champion_endpoint": {
        "branch": "control_c256",
        "depth": DEFAULT_CONTROL_DEPTH,
        "dropout": DEFAULT_DROPOUT,
        "step": 50_000,
        "best_step": DEFAULT_STAGE_49_CONTROL_BEST_STEP,
        "best_checkpoint_sha256": DEFAULT_STAGE_49_CONTROL_BEST_SHA256,
        "final_checkpoint_sha256": DEFAULT_STAGE_49_CONTROL_SHA256,
        "fixed_panel_train_loss": DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_TRAIN_LOSS,
        "fixed_panel_validation_loss": (
            DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_VALIDATION_LOSS
        ),
        "role": "the champion this stage's control must re-derive and the depth "
        "step is measured from",
    },
    "stage_49_c256_p020_endpoint": {
        "branch": "treatment_c256",
        "dropout": 0.20,
        "step": 50_000,
        "final_checkpoint_sha256": (
            "cff587a1b79b03e946702947cd2fe6811b08c44c60488c02a43c97bc7fac5105"
        ),
        "role": "the over-regularized C256 endpoint Stage 49 displaced",
    },
    "stage_46_c320_p015_endpoint": {
        "branch": "treatment_c320",
        "dropout": 0.15,
        "step": 50_000,
        "final_checkpoint_sha256": C320_BUDGET_REF["file_sha256"],
        "parameter_count": DEFAULT_C320_PARAMETER_COUNT,
        "role": "the wider budget this stage's deeper treatment is scored against",
    },
    "role": "historical reference only; none is a Stage 50 initialization",
}
INITIALIZATION_POLICY = (
    "Both branches recreate their own architecture-native initialization from "
    "the same seed and corpus on the CPU. Residual dropout has no learnable "
    "parameters and consumes no initialization RNG. Depth does change the model, "
    "so byte-identical complete states are neither expected nor claimed; what is "
    "claimed and checked is that the treatment's six shared blocks, its two "
    "embedding tables, and its final LayerNorm are byte-identical to the "
    "control's, because the native construction draws in block order."
)
RANDOMNESS_POLICY = (
    "Both branches begin from the same CPU and accelerator training RNG states "
    "and consume the same sampled training panels, because the sampler depends "
    "only on batch shape and its seed. The treatment applies two more dropout "
    "sites per update, so its terminal Torch RNG state is expected to differ from "
    "the control's; neither identical dropout masks nor identical terminal Torch "
    "RNG states are claimed."
)
UNCERTAINTY_SCOPE = (
    "Student-t 95% panel-sampling interval conditional on these checkpoints; "
    "not training-seed or corpus-IID uncertainty."
)
# The architecture-record fields both this stage and Stage 49 write. Stage 50
# adds depth-derived fields Stage 49 had no notion of, so the control
# reproduction compares the intersection rather than the whole mapping.
SHARED_ARCHITECTURE_FIELDS = (
    "n_embd",
    "n_head",
    "n_layer",
    "head_size",
    "feed_forward_width",
    "block_size",
    "residual_dropout",
    "layer_contexts",
    "topology",
    "vocabulary_size",
)
# The only state-dict entries that are masks rather than learned parameters.
MASK_BUFFER_SUFFIX = ".tril"


def layer_contexts_for_depth(depth: int) -> tuple[int | None, ...]:
    """The canonical layerwise topology per depth: W23 locals, two Fulls last."""

    if depth == DEFAULT_CONTROL_DEPTH:
        return DEFAULT_CONTROL_LAYER_CONTEXTS
    if depth == DEFAULT_TREATMENT_DEPTH:
        return DEFAULT_TREATMENT_LAYER_CONTEXTS
    raise ValueError(
        f"Stage 50 canonical depths are {DEFAULT_CONTROL_DEPTH} and "
        f"{DEFAULT_TREATMENT_DEPTH}, got {depth!r}"
    )


def topology_label(layer_contexts: Sequence[int | None]) -> str:
    """Name a layerwise topology the way the width stages did: LLLLLLGG."""

    return "".join("G" if context is None else "L" for context in layer_contexts)


def canonical_config(depth: int = DEFAULT_CONTROL_DEPTH) -> TrainingConfig:
    """The locked Stage 50 training config; depth travels with BranchSpec."""

    layer_contexts_for_depth(depth)
    return replace(
        _STAGE_42.canonical_config(), n_embd=DEFAULT_WIDTH, learning_rate=1e-3
    )


def make_attention_spec(
    block_size: int, layer_contexts: Sequence[int | None]
) -> AttentionSpec:
    """The Stage 30/32 attention spec for one depth's layerwise mask stack.

    The spec constructor is also the mask policy's validator: it requires exactly
    two consecutive final Full layers above one common local window.
    """

    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError(f"block_size must be an integer, got {block_size!r}")
    if block_size < DEFAULT_LOCAL_CONTEXT:
        raise ValueError(
            f"block_size must be at least W{DEFAULT_LOCAL_CONTEXT}, got {block_size}"
        )
    contexts = tuple(layer_contexts)
    return AttentionSpec(
        name=(
            f"Last2Full {topology_label(contexts)} at T={block_size}, "
            f"L={len(contexts)}"
        ),
        layer_contexts=contexts,
        checkpoint_path=(
            _STAGE_41.CHECKPOINT_DIRECTORY
            / f".stage_50_architecture_t{block_size}_l{len(contexts)}.pt"
        ),
    )


def build_model(
    data: CharacterData,
    config: TrainingConfig,
    spec: AttentionSpec,
    n_head: int,
    device: torch.device | None = None,
) -> GPTLanguageModel:
    """The Stage 32 architecture factory with depth carried by its spec.

    Stage 46's factory re-used Stage 36's mask stack, which is fixed at six
    LLLLGG layers. This one takes the spec as the depth authority, which is the
    Stage 30/32 interface the six-layer wrapper was itself built on.
    """

    if config.n_embd % n_head:
        raise ValueError("width must be divisible by the head count")
    model = GPTLanguageModel(
        vocab_size=data.vocabulary.size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=n_head,
        n_layer=spec.n_layer,
        dropout=DEFAULT_DROPOUT,
    )
    apply_attention_masks(model, build_layer_attention_masks(config.block_size, spec))
    if device is not None:
        model = model.to(device)
    return model


def verify_model_masks(model: GPTLanguageModel, spec: AttentionSpec) -> None:
    """Check every layer's mask against this spec, not the six-layer default."""

    _STAGE_30.verify_model_masks(model, spec)


def spec_for(spec: "BranchSpec", block_size: int) -> AttentionSpec:
    return make_attention_spec(block_size, spec.layer_contexts)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def per_block_parameter_count(model: torch.nn.Module) -> int:
    if not model.blocks:
        raise ValueError("model has no blocks")
    return parameter_count(model.blocks[0])


def parameter_count_from_state_dict(state: Mapping[str, torch.Tensor]) -> int:
    """Count the learned parameters in a state dict, excluding mask buffers."""

    return sum(
        tensor.numel()
        for name, tensor in state.items()
        if not name.endswith(MASK_BUFFER_SUFFIX)
    )


def reachability_audit(block_size: int, layer_contexts: Sequence[int | None]) -> Any:
    """The composed receptive field of one topology, checked for full coverage."""

    audit = audit_layerwise_reachability(
        build_layer_attention_masks(block_size, make_attention_spec(block_size, layer_contexts))
    )
    verify_global_reachability(audit, block_size)
    return audit


def depth_architecture_metadata(
    data: CharacterData, config: TrainingConfig, spec: "BranchSpec"
) -> dict[str, Any]:
    """The control reproduction's shared fields plus this stage's depth record."""

    audit = reachability_audit(config.block_size, spec.layer_contexts)
    local_layers = sum(1 for context in spec.layer_contexts if context is not None)
    return {
        "n_embd": config.n_embd,
        "n_head": _STAGE_44.N_HEAD,
        "n_layer": spec.depth,
        "head_size": config.n_embd // _STAGE_44.N_HEAD,
        "feed_forward_width": 4 * config.n_embd,
        "block_size": config.block_size,
        "residual_dropout": spec.dropout,
        "layer_contexts": spec.layer_contexts,
        "topology": topology_label(spec.layer_contexts),
        "vocabulary_size": data.vocabulary.size,
        "local_layer_count": local_layers,
        "global_layer_count": spec.depth - local_layers,
        "global_layer_indices": tuple(
            index for index, context in enumerate(spec.layer_contexts) if context is None
        ),
        "direct_attention_spans": audit.direct_attention_spans,
        "receptive_field_by_layer": audit.widths_by_layer,
        "pre_global_receptive_field": audit.widths_by_layer[local_layers - 1],
        "first_global_reachability_layer": audit.first_global_reachability_layer,
    }


@dataclass(frozen=True, slots=True)
class BranchSpec:
    key: str
    width: int
    depth: int
    dropout: float
    layer_contexts: tuple[int | None, ...]
    best_path: Path
    final_path: Path

    def __post_init__(self) -> None:
        if self.key not in ("control", "treatment"):
            raise ValueError("Stage 50 branch must be control or treatment")
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise ValueError("width must be an integer")
        if self.width <= 0 or self.width % _STAGE_44.N_HEAD:
            raise ValueError("width must be positive and divisible by four heads")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise ValueError("depth must be an integer")
        contexts = tuple(self.layer_contexts)
        object.__setattr__(self, "layer_contexts", contexts)
        if self.depth != len(contexts):
            raise ValueError("depth must equal the number of layer contexts")
        if self.depth < 3:
            raise ValueError("a two-global-layer topology needs at least three layers")
        probability = self.dropout
        if isinstance(probability, bool) or not isinstance(probability, float):
            raise ValueError("dropout must be a float")
        if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
            raise ValueError(
                f"dropout must be finite and in [0, 1), got {probability!r}"
            )
        # The spec constructor is the mask policy's own validator.
        make_attention_spec(_STRUCTURAL_BLOCK_SIZE, contexts)


def branch_specs(directory: Path) -> tuple[BranchSpec, BranchSpec]:
    return tuple(
        BranchSpec(
            key,
            DEFAULT_WIDTH,
            depth,
            DEFAULT_DROPOUT,
            layer_contexts_for_depth(depth),
            directory
            / f"stage_50_{key}_c{DEFAULT_WIDTH}_l{depth}_best_checkpoint.pt",
            directory
            / (
                f"stage_50_{key}_c{DEFAULT_WIDTH}_l{depth}"
                "_step_50000_final_checkpoint.pt"
            ),
        )
        for key, depth in (
            ("control", DEFAULT_CONTROL_DEPTH),
            ("treatment", DEFAULT_TREATMENT_DEPTH),
        )
    )


def validate_branch_specs(specs: tuple[BranchSpec, ...]) -> None:
    expected = (
        ("control", DEFAULT_CONTROL_DEPTH),
        ("treatment", DEFAULT_TREATMENT_DEPTH),
    )
    if tuple((spec.key, spec.depth) for spec in specs) != expected or any(
        spec.width != DEFAULT_WIDTH
        or spec.dropout != DEFAULT_DROPOUT
        or spec.layer_contexts != layer_contexts_for_depth(spec.depth)
        for spec in specs
    ):
        raise ValueError(
            "Stage 50 locks L=6 against L=8 at C256 with residual dropout p=0.15 "
            "and the established LLLLGG / LLLLLLGG topologies"
        )


def validate_protocol(data: CharacterData, configs: tuple[TrainingConfig, ...]) -> None:
    if configs != (canonical_config(), canonical_config()):
        raise ValueError("Stage 50 locks every training setting")
    if fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Stage 50 requires unchanged Complete Shakespeare and split")


def validate_artifact_paths(
    inputs: tuple[Path, ...], outputs: tuple[Path, ...]
) -> None:
    paths = [path.resolve() for path in (*inputs, *outputs)]
    if len(paths) != len(set(paths)):
        raise ValueError("corpus, checkpoints, and reports must have distinct paths")
    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Stage 50 output exists; use a new --output-dir: {path}"
            )
        if path.with_name(f".{path.name}.tmp").exists():
            raise FileExistsError(f"Stage 50 temporary output exists: {path}")


def locked_initialization_fingerprint(spec: BranchSpec) -> tuple[int, str]:
    """The recorded (parameter count, native state hash) for one canonical depth."""

    if spec.depth == DEFAULT_CONTROL_DEPTH:
        return (DEFAULT_CONTROL_PARAMETER_COUNT, C256_L6_NATIVE_INITIALIZATION_SHA256)
    if spec.depth == DEFAULT_TREATMENT_DEPTH:
        return (
            DEFAULT_TREATMENT_PARAMETER_COUNT,
            C256_L8_NATIVE_INITIALIZATION_SHA256,
        )
    raise ValueError(f"Stage 50 has no recorded fingerprint for L={spec.depth}")


def expected_initialization_keys(
    control_spec: BranchSpec, treatment_spec: BranchSpec
) -> dict[str, Any]:
    """Name exactly which state-dict entries may differ between the branches.

    The native construction draws in block order, so every entry of the shallower
    model is expected to match the treatment's byte for byte except the language
    head (drawn after a different number of blocks) and the attention-mask
    buffers of the layers whose policy changed.
    """

    shallow, deep = sorted((control_spec, treatment_spec), key=lambda spec: spec.depth)
    changed_mask_layers = tuple(
        layer
        for layer in range(shallow.depth)
        if shallow.layer_contexts[layer] != deep.layer_contexts[layer]
    )
    changed_masks = tuple(
        f"blocks.{layer}.sa.heads.{head}{MASK_BUFFER_SUFFIX}"
        for layer in changed_mask_layers
        for head in range(_STAGE_44.N_HEAD)
    )
    return {
        "changed_mask_layers": changed_mask_layers,
        "expected_differing": (
            *changed_masks,
            "lm_head.weight",
            "lm_head.bias",
        ),
        "added_block_layers": tuple(range(shallow.depth, deep.depth)),
    }


def shared_initialization_audit(
    control_spec: BranchSpec,
    treatment_spec: BranchSpec,
    data: CharacterData,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Prove the treatment's depth is an extension of the control's draw order.

    Both models are rebuilt on the CPU from the same seed. The control's state
    must then appear inside the treatment's byte for byte, except for the
    language head and the attention-mask buffers of the layers whose policy
    changed, and the treatment must add nothing outside its new blocks.
    """

    # Each model is drawn from the same seed, exactly as the two branches are.
    _STAGE_41.seed_everything(config.seed)
    control_model = build_model(
        data,
        config,
        spec_for(control_spec, config.block_size),
        _STAGE_44.N_HEAD,
        torch.device("cpu"),
    )
    _STAGE_41.seed_everything(config.seed)
    treatment_model = build_model(
        data,
        config,
        spec_for(treatment_spec, config.block_size),
        _STAGE_44.N_HEAD,
        torch.device("cpu"),
    )
    verify_model_masks(control_model, spec_for(control_spec, config.block_size))
    verify_model_masks(treatment_model, spec_for(treatment_spec, config.block_size))
    control_state = {
        name: tensor.detach().cpu() for name, tensor in control_model.state_dict().items()
    }
    treatment_state = {
        name: tensor.detach().cpu()
        for name, tensor in treatment_model.state_dict().items()
    }
    expectations = expected_initialization_keys(control_spec, treatment_spec)
    differing_expected = set(expectations["expected_differing"])
    shared = set(control_state) & set(treatment_state)
    changed_but_expected = sorted(
        name
        for name in shared - differing_expected
        if not torch.equal(control_state[name], treatment_state[name])
    )
    if changed_but_expected:
        raise RuntimeError(
            "Stage 50 shared initialization prefix changed in: "
            + ", ".join(changed_but_expected[:6])
        )
    unchanged_but_expected = sorted(
        name
        for name in differing_expected
        if torch.equal(control_state[name], treatment_state[name])
    )
    if unchanged_but_expected:
        raise RuntimeError(
            "Stage 50 expected these initialization entries to differ and they "
            "did not: " + ", ".join(unchanged_but_expected)
        )
    missing = sorted(set(control_state) - set(treatment_state))
    if missing:
        raise RuntimeError(
            f"the treatment dropped {len(missing)} control entries, first {missing[0]}"
        )
    added = sorted(set(treatment_state) - set(control_state))
    prefixes = tuple(
        f"blocks.{layer}." for layer in expectations["added_block_layers"]
    )
    unexpected = [
        name for name in added if not any(name.startswith(prefix) for prefix in prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            "Stage 50 added initialization entries outside the new blocks: "
            + ", ".join(sorted(unexpected)[:6])
        )
    added_parameter_count = sum(
        parameter_count(treatment_model.blocks[layer])
        for layer in expectations["added_block_layers"]
    )
    if added_parameter_count != parameter_count(treatment_model) - parameter_count(
        control_model
    ):
        raise RuntimeError("the added blocks do not account for the parameter delta")
    identical = tuple(sorted(shared - differing_expected))
    return {
        "control_key_count": len(control_state),
        "treatment_key_count": len(treatment_state),
        "treatment_is_a_key_superset": not missing,
        "identical_key_count": len(identical),
        "control_identical_keys_sha256": _STAGE_41.tensor_mapping_sha256(
            {name: control_state[name] for name in identical}
        ),
        "treatment_identical_keys_sha256": _STAGE_41.tensor_mapping_sha256(
            {name: treatment_state[name] for name in identical}
        ),
        "expected_differing": expectations["expected_differing"],
        "changed_mask_layers": expectations["changed_mask_layers"],
        "added_key_count": len(added),
        "added_block_layers": expectations["added_block_layers"],
        "added_parameter_count": added_parameter_count,
        "per_added_block_parameter_count": per_block_parameter_count(treatment_model),
        "control_parameter_count": parameter_count(control_model),
        "treatment_parameter_count": parameter_count(treatment_model),
        "control_masks_sha256": _STAGE_41.tensor_mapping_sha256(
            {
                f"layer{index}": mask
                for index, mask in enumerate(
                    build_layer_attention_masks(
                        config.block_size, spec_for(control_spec, config.block_size)
                    )
                )
            }
        ),
        "treatment_masks_sha256": _STAGE_41.tensor_mapping_sha256(
            {
                f"layer{index}": mask
                for index, mask in enumerate(
                    build_layer_attention_masks(
                        config.block_size, spec_for(treatment_spec, config.block_size)
                    )
                )
            }
        ),
    }


def verify_native_initialization(
    data: CharacterData,
    config: TrainingConfig,
    spec: BranchSpec,
    initialization: dict[str, Any],
) -> str:
    """Prove one branch's native initialization is the recorded one, twice over.

    The recorded fingerprint is checked directly, the model is rebuilt through
    this stage's own factory, and the same state is rebuilt through Stage 30's
    unchanged factory. The six-layer branch additionally has to match the Stage 46
    exposed-dropout factory, which is the factory Stages 47-49 trained through, so
    the control is not merely a new construction that happens to look alike.
    """

    if config.n_embd != DEFAULT_WIDTH:
        raise ValueError("Stage 50 native initialization is only defined at C256")
    expected_count, expected_hash = locked_initialization_fingerprint(spec)
    digest = initialization["state_sha256"]
    if initialization["parameter_count"] != expected_count:
        raise RuntimeError("Stage 50 native initialization parameter count changed")
    if initialization["block_parameter_count"] != DEFAULT_BLOCK_PARAMETER_COUNT:
        raise RuntimeError("Stage 50 native per-block parameter count changed")
    _STAGE_41.seed_everything(config.seed)
    rebuilt = build_model(
        data,
        config,
        spec_for(spec, config.block_size),
        _STAGE_44.N_HEAD,
        torch.device("cpu"),
    )
    if _STAGE_41.tensor_mapping_sha256(rebuilt.state_dict()) != digest:
        raise RuntimeError("Stage 50 native initialization diverged from its factory")
    del rebuilt
    _STAGE_41.seed_everything(config.seed)
    independent = _STAGE_30.build_model(
        data,
        config,
        spec_for(spec, config.block_size),
        _STAGE_44.N_HEAD,
        spec.depth,
        torch.device("cpu"),
    )
    if _STAGE_41.tensor_mapping_sha256(independent.state_dict()) != digest:
        raise RuntimeError(
            "Stage 50 native initialization diverged from the Stage 30 factory"
        )
    del independent
    if spec.depth == DEFAULT_CONTROL_DEPTH:
        if _STAGE_46.verify_control_factory_equivalence(data, config) != digest:
            raise RuntimeError(
                "Stage 50 native initialization diverged from the Stage 46 factory"
            )
    if digest != expected_hash:
        raise RuntimeError(
            f"Stage 50 native initialization hash changed at L={spec.depth}"
        )
    return digest


def initialize_branch(
    data: CharacterData,
    config: TrainingConfig,
    spec: BranchSpec,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.Generator, dict[str, Any]]:
    if config.n_embd % _STAGE_44.N_HEAD or config.block_size < DEFAULT_LOCAL_CONTEXT:
        raise ValueError("requires width divisible by four and T >= 23")
    if config.batch_size % 2:
        raise ValueError("the sampler requires an even batch size")
    if config.learning_rate != make_schedule().for_update(0) or config.max_iters <= 0:
        raise ValueError(
            "fresh runs require the established initial LR and positive budget"
        )
    # Always construct on CPU from the same seed, then move the native model. No
    # tensor padding, checkpoint loading, or optimizer-moment transplant.
    _STAGE_41.seed_everything(config.seed)
    model = build_model(
        data,
        config,
        spec_for(spec, config.block_size),
        _STAGE_44.N_HEAD,
        torch.device("cpu"),
    )
    verify_model_masks(model, spec_for(spec, config.block_size))
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
        "depth": spec.depth,
        "layer_contexts": spec.layer_contexts,
        "residual_dropout": spec.dropout,
        "historical_weights_loaded": False,
        "historical_optimizer_loaded": False,
        "optimizer_state_entries_at_initialization": len(optimizer.state),
        "parameter_count": parameter_count(model),
        "block_parameter_count": per_block_parameter_count(model),
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
        "architecture": depth_architecture_metadata(data, config, spec),
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
            "comparison_variable": "transformer_depth",
            "branch_key": spec.key,
            "branch_depth": spec.depth,
            "branch_layer_contexts": spec.layer_contexts,
            "branch_residual_dropout": spec.dropout,
            "control_depth": DEFAULT_CONTROL_DEPTH,
            "treatment_depth": DEFAULT_TREATMENT_DEPTH,
            "from_scratch": True,
            "trained_branches": 2,
            "canonical_protocol": (
                config == canonical_config()
                and spec.width == DEFAULT_WIDTH
                and spec.depth in (DEFAULT_CONTROL_DEPTH, DEFAULT_TREATMENT_DEPTH)
                and spec.layer_contexts == layer_contexts_for_depth(spec.depth)
                and fingerprint_data(data)
                == _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT
            ),
        },
        "evaluation_protocol": {
            "selection_metric": "strict_minimum_fixed_panel_validation_loss",
            "primary_metric": (
                "fresh_paired_terminal_validation_treatment_minus_control"
            ),
            "fresh_panel_seed": DEFAULT_FRESH_VALIDATION_SEED,
            "fresh_panel_batches": DEFAULT_FRESH_VALIDATION_ITERS,
            "dropout_enabled": False,
        },
        "randomness_policy": RANDOMNESS_POLICY,
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
        raise ValueError("Stage 50 checkpoint must be a mapping")
    digest = payload.pop("payload_sha256", None)
    if digest != state_hash(payload):
        raise ValueError("Stage 50 checkpoint payload integrity mismatch")
    payload["payload_sha256"] = digest
    if payload["checkpoint_version"] != 1 or payload["checkpoint_kind"] != kind:
        raise ValueError("Stage 50 checkpoint version or kind mismatch")
    if (
        payload["experiment"]["stage"] != EXPERIMENT_STAGE
        or payload["experiment"]["protocol"] != PROTOCOL_NAME
    ):
        raise ValueError("Stage 50 experiment mismatch")
    if (
        payload["experiment"]["branch_key"] != spec.key
        or payload["experiment"]["branch_depth"] != spec.depth
        or payload["experiment"]["branch_layer_contexts"] != spec.layer_contexts
        or payload["experiment"]["branch_residual_dropout"] != spec.dropout
        or config.n_embd != spec.width
    ):
        raise ValueError("Stage 50 checkpoint branch mismatch")
    if payload["architecture"] != depth_architecture_metadata(data, config, spec):
        raise ValueError("Stage 50 architecture changed, including depth and dropout")
    if payload["data_fingerprint"] != fingerprint_data(data):
        raise ValueError("Stage 50 corpus or split changed")
    for key, value in asdict(config).items():
        if payload["training_config"].get(key) != value:
            raise ValueError(f"Stage 50 training config changed: {key}")
    if payload["training_config"]["learning_rate_schedule"] != schedule_metadata(
        config.max_iters
    ):
        raise ValueError("Stage 50 learning rate schedule changed")
    step, best_step = payload["step"], payload["best_step"]
    if (
        type(step) is not int
        or type(best_step) is not int
        or not 0 <= best_step <= step <= config.max_iters
    ):
        raise ValueError("Stage 50 checkpoint steps are invalid")
    if kind == "final" and step != config.max_iters:
        raise ValueError("Stage 50 final must be the matched terminal step")
    if kind == "best" and step != best_step:
        raise ValueError("Stage 50 best must contain its selected weights")
    if not all(
        math.isfinite(value)
        for value in (
            *payload["snapshot_losses"].values(),
            payload["best_val_loss"],
            payload["best_training_loss"],
        )
    ):
        raise ValueError("Stage 50 losses must be finite")
    recorded_normalization = payload.get("rng_normalization")
    if not isinstance(recorded_normalization, dict):
        raise ValueError("Stage 50 checkpoint is missing its RNG normalization")
    recomputed = terminal_rng_normalization(payload)
    if (
        recorded_normalization.get("terminal_training_rng_sha256")
        != recomputed["terminal_training_rng_sha256"]
        or recorded_normalization.get("cuda_entry_count")
        != recomputed["cuda_entry_count"]
        or recorded_normalization.get("training_device_index")
        != recomputed["training_device_index"]
    ):
        raise ValueError("Stage 50 RNG normalization disagrees with its stored state")
    if payload["runtime"].get("device_index") != recorded_normalization.get(
        "training_device_index"
    ):
        raise ValueError("Stage 50 runtime device index disagrees with its normalization")
    initialization = payload["initialization"]
    if (
        initialization["kind"] != "fresh_architecture_native"
        or initialization["origin_stage"] != EXPERIMENT_STAGE
        or initialization["step"] != 0
        or initialization["seed"] != config.seed
        or initialization["depth"] != spec.depth
        or initialization["layer_contexts"] != spec.layer_contexts
        or initialization["residual_dropout"] != spec.dropout
        or initialization["historical_weights_loaded"]
        or initialization["historical_optimizer_loaded"]
        or initialization["optimizer_state_entries_at_initialization"] != 0
        or payload["optimizer_restart_step"] != 0
    ):
        raise ValueError("Stage 50 requires fresh model and optimizer initialization")
    measured = parameter_count_from_state_dict(payload["model_state_dict"])
    if initialization["parameter_count"] != measured:
        raise ValueError(
            "Stage 50 parameter count disagrees with the stored model weights"
        )
    optimizer_state = payload["optimizer_state_dict"]
    groups, states = optimizer_state["param_groups"], optimizer_state["state"]
    if len(groups) != 1:
        raise ValueError("Stage 50 requires one AdamW parameter group")
    group = groups[0]
    if (
        group["lr"] != make_schedule().for_update(max(0, step - 1))
        or group["betas"] != (0.9, 0.999)
        or group["weight_decay"] != 0.01
        or group["eps"] != 1e-8
        or group["amsgrad"]
    ):
        raise ValueError("Stage 50 AdamW settings changed")
    if step == 0:
        if states:
            raise ValueError("Stage 50 step-zero optimizer must be empty")
    elif set(states) != set(group["params"]) or any(
        state["step"].item() != step for state in states.values()
    ):
        raise ValueError("Stage 50 AdamW counters do not match completed updates")
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
        data, config, spec, device
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
            f"  {spec.key} C{spec.width} L{spec.depth} p{spec.dropout:g} "
            f"step {record.step:5d} | "
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
                    "Stage 50 terminal state failed checkpoint round trip"
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


def verify_paired_specs(
    control_spec: BranchSpec,
    treatment_spec: BranchSpec,
    control_parameters: int,
    treatment_parameters: int,
    block_parameters: int,
) -> None:
    """Depth is the only licensed difference between the two branches."""

    if control_spec.key != "control" or treatment_spec.key != "treatment":
        raise RuntimeError("paired comparison requires control followed by treatment")
    if (control_spec.depth, treatment_spec.depth) != (
        DEFAULT_CONTROL_DEPTH,
        DEFAULT_TREATMENT_DEPTH,
    ):
        raise RuntimeError("paired depth treatment must be exactly L=8 against L=6")
    if control_spec.width != treatment_spec.width:
        raise RuntimeError("paired depth comparison must hold width fixed")
    if control_spec.dropout != treatment_spec.dropout:
        raise RuntimeError("paired depth comparison must hold dropout fixed")
    if control_spec.layer_contexts != layer_contexts_for_depth(DEFAULT_CONTROL_DEPTH):
        raise RuntimeError("the control must keep the LLLLGG topology")
    if treatment_spec.layer_contexts != layer_contexts_for_depth(
        DEFAULT_TREATMENT_DEPTH
    ):
        raise RuntimeError("the treatment must keep the LLLLLLGG topology")
    added = treatment_parameters - control_parameters
    if added != (
        DEFAULT_TREATMENT_DEPTH - DEFAULT_CONTROL_DEPTH
    ) * block_parameters:
        raise RuntimeError("paired depth branches differ by the wrong parameter count")


def verify_paired_initial_streams(
    control_initial: Mapping[str, str], treatment_initial: Mapping[str, str]
) -> None:
    """The sampler and the accelerator RNG start identical; the weights do not."""

    for key in ("training_generator_sha256", "torch_rng_sha256"):
        if control_initial[key] != treatment_initial[key]:
            raise RuntimeError(f"paired branches started with different {key}")
    if control_initial["model_sha256"] == treatment_initial["model_sha256"]:
        raise RuntimeError("paired depth branches started from identical weights")


def verify_distinct_draw_schedules(
    control_label: str, control_rng: str, treatment_label: str, treatment_rng: str
) -> None:
    """Two more blocks means two more dropout sites per update.

    The lineage's one-dropout-quantum-per-site schedule is what makes the
    terminal state comparable at all: Stages 46-49 recorded the same terminal
    device RNG across two widths and two probabilities, which is only possible if
    the advance tracks the number of dropout sites. A deeper model visits more
    sites, so equality here would mean the added layers were never trained.
    """

    if control_rng == treatment_rng:
        raise RuntimeError(
            f"{treatment_label} did not consume more dropout RNG than {control_label}"
        )


def verify_paired_branches(control: Any, treatment: Any) -> None:
    verify_paired_specs(
        control.spec,
        treatment.spec,
        control.initialization["parameter_count"],
        treatment.initialization["parameter_count"],
        control.initialization["block_parameter_count"],
    )
    verify_paired_initial_streams(control.initial_streams, treatment.initial_streams)
    if (
        control.final_streams["training_generator_sha256"]
        != treatment.final_streams["training_generator_sha256"]
    ):
        raise RuntimeError("paired branches consumed different training batches")
    if control.final_streams["model_sha256"] == treatment.final_streams["model_sha256"]:
        raise RuntimeError("paired depth branches ended with identical weights")
    verify_distinct_draw_schedules(
        "the L=6 control",
        control.final_streams["torch_rng_sha256"],
        "the L=8 treatment",
        treatment.final_streams["torch_rng_sha256"],
    )
    if [record.step for record in control.training.history] != [
        record.step for record in treatment.training.history
    ]:
        raise RuntimeError("paired update/evaluation budgets differ")


def verify_paired_endpoints(
    control_entry: dict[str, Any],
    treatment_entry: dict[str, Any],
    specs: Sequence[BranchSpec],
) -> None:
    """The same paired contract for two stored endpoints instead of live branches."""

    control_payload, treatment_payload = (
        control_entry["payload"],
        treatment_entry["payload"],
    )
    verify_paired_specs(
        specs[0],
        specs[1],
        control_payload["initialization"]["parameter_count"],
        treatment_payload["initialization"]["parameter_count"],
        control_payload["initialization"]["block_parameter_count"],
    )
    verify_paired_initial_streams(
        control_payload["initial_streams"], treatment_payload["initial_streams"]
    )
    control_generator = _STAGE_41.tensor_sha256(
        control_payload["training_generator_state"]
    )
    treatment_generator = _STAGE_41.tensor_sha256(
        treatment_payload["training_generator_state"]
    )
    if control_generator != treatment_generator:
        raise RuntimeError("paired endpoints consumed different training batches")
    if _STAGE_41.tensor_mapping_sha256(
        control_payload["model_state_dict"]
    ) == _STAGE_41.tensor_mapping_sha256(treatment_payload["model_state_dict"]):
        raise RuntimeError("paired depth endpoints hold identical weights")
    verify_distinct_draw_schedules(
        "the L=6 control",
        terminal_rng_normalization(control_payload)["terminal_training_rng_sha256"],
        "the L=8 treatment",
        terminal_rng_normalization(treatment_payload)["terminal_training_rng_sha256"],
    )


@dataclass(frozen=True, slots=True)
class StoredEndpoint:
    """A stored, authenticated endpoint read as a read-only cross-stage peer."""

    key: str
    width: int
    depth: int
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


def load_stage_49_control(data: CharacterData, directory: Path) -> StoredEndpoint:
    """Authenticate Stage 49's fresh C256 p=0.15 control as a read-only peer."""

    specs = _STAGE_49.branch_specs(directory)
    spec = next(spec for spec in specs if spec.key == "control")
    if spec.final_path.name != DEFAULT_STAGE_49_CONTROL_FILENAME:
        raise ValueError("Stage 49 C256 control endpoint changed its name")
    if not spec.final_path.is_file():
        raise FileNotFoundError(f"Stage 49 endpoint not present: {spec.final_path}")
    if file_hash(spec.final_path) != DEFAULT_STAGE_49_CONTROL_SHA256:
        raise ValueError("Stage 49 C256 control checkpoint hash changed")
    if spec.best_path.name != DEFAULT_STAGE_49_CONTROL_BEST_FILENAME:
        raise ValueError("Stage 49 C256 control best endpoint changed its name")
    if not spec.best_path.is_file():
        raise FileNotFoundError(f"Stage 49 endpoint not present: {spec.best_path}")
    if file_hash(spec.best_path) != DEFAULT_STAGE_49_CONTROL_BEST_SHA256:
        raise ValueError("Stage 49 C256 control best checkpoint hash changed")
    config = _STAGE_49.canonical_config(_STAGE_49.DEFAULT_CONTROL_DROPOUT)
    payload = _STAGE_49.validate_branch_checkpoint(
        spec.final_path, spec, data, config, "final"
    )
    if (
        payload["best_step"] != DEFAULT_STAGE_49_CONTROL_BEST_STEP
        or payload["snapshot_losses"]["train"]
        != DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_TRAIN_LOSS
        or payload["snapshot_losses"]["validation"]
        != DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_VALIDATION_LOSS
    ):
        raise ValueError("Stage 49 C256 control fixed-panel record changed")
    return StoredEndpoint(
        "c256_l6_stage_49",
        spec.width,
        _STAGE_44.N_LAYER,
        _STAGE_49.DEFAULT_CONTROL_DROPOUT,
        spec.final_path,
        payload,
        config,
    )


def verify_stage_49_control_reproduction(
    spec: BranchSpec,
    data: CharacterData,
    config: TrainingConfig,
    reference: StoredEndpoint,
) -> dict[str, Any]:
    """Require the trained L=6 control to reproduce Stage 49's champion exactly.

    The control *is* the endpoint the depth step is measured from, so a control
    that is not that endpoint would silently invalidate the comparison and every
    budget row it feeds. Both states are compared field by field: weights, AdamW
    state, batch-generator state, Torch RNG state, fixed-panel losses, the shared
    architecture fields, and the selected best step.
    """

    payload = validate_branch_checkpoint(spec.final_path, spec, data, config, "final")
    recorded = {
        "step": payload["step"],
        "snapshot_losses": payload["snapshot_losses"],
        "best_step": payload["best_step"],
        "best_val_loss": payload["best_val_loss"],
        "best_training_loss": payload["best_training_loss"],
        "data_fingerprint": payload["data_fingerprint"],
    }
    expected = {key: reference.payload[key] for key in recorded}
    mismatched = sorted(
        key for key, value in recorded.items() if value != expected[key]
    )
    # Stage 50 records depth fields Stage 49 had no notion of, so the architecture
    # record is compared on the fields both stages write.
    architecture = {
        key: payload["architecture"][key] for key in SHARED_ARCHITECTURE_FIELDS
    }
    reference_architecture = {
        key: reference.payload["architecture"][key]
        for key in SHARED_ARCHITECTURE_FIELDS
    }
    if architecture != reference_architecture:
        mismatched.append("architecture")
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
    if ours["terminal_training_rng_sha256"] != theirs["terminal_training_rng_sha256"]:
        mismatched.append("training_device_rng_sha256")
    ours_cpu = state_hash(payload["rng_state"]["cpu"])
    theirs_cpu = state_hash(reference.payload["rng_state"]["cpu"])
    if ours_cpu != theirs_cpu:
        mismatched.append("cpu_rng_sha256")
    if mismatched:
        raise RuntimeError(
            "Stage 50 L=6 control did not reproduce the Stage 49 champion: "
            + ", ".join(mismatched)
        )
    return {
        "reference": "Stage 49 C256 p=0.15 control, step 50,000",
        "reference_path": str(reference.path.resolve()),
        "reference_sha256": DEFAULT_STAGE_49_CONTROL_SHA256,
        "identical": True,
        "fields_checked": sorted(recorded)
        + ["architecture"]
        + sorted(streams)
        + [
            "cpu_rng_sha256",
            "training_device_rng_sha256",
            "training_device_name",
        ],
        "architecture_fields_checked": list(SHARED_ARCHITECTURE_FIELDS),
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


def load_c320_budget_peer(data: CharacterData, directory: Path) -> StoredEndpoint:
    """Authenticate Stage 46's C320 p=0.15 treatment through its owning stages."""

    endpoint = _STAGE_49.load_c320_champion(C320_BUDGET_REF, data, directory)
    if endpoint.width == DEFAULT_WIDTH:
        raise ValueError("the stored budget peer must be the wider C320 endpoint")
    return StoredEndpoint(
        BUDGET_PEER_KEY,
        endpoint.width,
        _STAGE_44.N_LAYER,
        endpoint.dropout,
        endpoint.path,
        endpoint.payload,
        endpoint.config,
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
    if seed in CONSUMED_PANEL_SEEDS or seed == DEFAULT_BUDGET_PANEL_SEED:
        raise ValueError("Stage 50 fresh validation must use an unused panel seed")
    if eval_iters < 2:
        raise ValueError("paired uncertainty requires at least two batches")
    verify_paired_branches(control, treatment)
    caller_rng = capture_rng_state(device)
    models, payloads = [], []
    try:
        for branch in (control, treatment):
            if file_hash(branch.spec.final_path) != branch.final_sha256:
                raise ValueError("Stage 50 endpoint checkpoint hash mismatch")
            payload = validate_branch_checkpoint(
                branch.spec.final_path,
                branch.spec,
                data,
                config,
                "final",
            )
            spec = spec_for(branch.spec, config.block_size)
            model = build_model(data, config, spec, _STAGE_44.N_HEAD, device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            verify_model_masks(model, spec)
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
                raise ValueError("Stage 50 fresh validation produced non-finite loss")
            if file_hash(branch.spec.final_path) != branch.final_sha256:
                raise RuntimeError("Stage 50 endpoint changed during evaluation")
            mean, se = _STAGE_40._mean_and_standard_error(values)
            results.append(
                _STAGE_40.FreshCheckpointResult(
                    branch_label(branch.spec, payload["step"]),
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


def branch_label(spec: BranchSpec, step: int) -> str:
    return f"C{spec.width} L={spec.depth} p={spec.dropout:g} @ {step}"


def paired_outcome(report: Any) -> str:
    if report.delta.confidence_high < 0:
        return (
            "L=8 improves C256 validation at matched 50k updates on the fresh "
            "paired panel"
        )
    if report.delta.confidence_low > 0:
        return "L=6 beats L=8 at C256 on the fresh paired panel"
    return "unresolved within paired panel uncertainty; this does not prove equivalence"


def budget_plan(
    data: CharacterData,
    config: TrainingConfig,
    directory: Path,
    specs: Sequence[BranchSpec] | None = None,
    peer_loader: Any = load_c320_budget_peer,
) -> list[dict[str, Any]]:
    """The three endpoints of the parameter-budget panel, authenticated read-only.

    The two C256 endpoints are this stage's fresh branches; the C320 endpoint is
    Stage 46's stored p=0.15 treatment. Every endpoint is re-read from disk so the
    panel is derivable without training, and each carries its own width, depth,
    and parameter count because the three do not share one architecture.
    """

    specs = tuple(branch_specs(directory) if specs is None else specs)
    if len(specs) != 2:
        raise ValueError("the budget panel needs exactly the two fresh branches")
    if any(spec.width != config.n_embd for spec in specs):
        raise ValueError("budget branch endpoints must match the evaluated width")
    plan: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.final_path.is_file():
            raise FileNotFoundError(f"Stage 50 endpoint not present: {spec.final_path}")
        plan.append(
            {
                "key": f"c{spec.width}_l{spec.depth}",
                "path": spec.final_path,
                "payload": validate_branch_checkpoint(
                    spec.final_path, spec, data, config, "final"
                ),
                "config": config,
                "attention_spec": spec_for(spec, config.block_size),
                "width": spec.width,
                "depth": spec.depth,
                "dropout": spec.dropout,
                "provenance": f"stage 50 {spec.key}, fresh",
            }
        )
    if peer_loader is not None:
        endpoint = peer_loader(data, directory)
        if endpoint.key != BUDGET_PEER_KEY:
            raise ValueError("the stored budget peer must be the recorded C320 endpoint")
        plan.append(
            {
                "key": endpoint.key,
                "path": endpoint.path,
                "payload": endpoint.payload,
                "config": endpoint.config,
                "attention_spec": make_attention_spec(
                    endpoint.config.block_size,
                    layer_contexts_for_depth(endpoint.depth),
                ),
                "width": endpoint.width,
                "depth": endpoint.depth,
                "dropout": endpoint.dropout,
                "provenance": "stage 46 treatment, stored",
            }
        )
    for entry in plan:
        entry["label"] = (
            f"C{entry['width']} L={entry['depth']} p={entry['dropout']:g} "
            f"@ {entry['payload']['step']}"
        )
        entry["parameter_count"] = entry["payload"]["initialization"]["parameter_count"]
    return plan


def evaluate_budget_shootout(
    data: CharacterData,
    config: TrainingConfig,
    device: torch.device,
    directory: Path,
    *,
    specs: Sequence[BranchSpec] | None = None,
    peer_loader: Any = load_c320_budget_peer,
    expected_generator_sha256: str | None = DEFAULT_SHARED_TRAINING_GENERATOR_SHA256,
    seed: int = DEFAULT_BUDGET_PANEL_SEED,
    eval_iters: int = DEFAULT_FRESH_VALIDATION_ITERS,
) -> dict[str, Any]:
    """Score the depth and width spends of one parameter budget on one panel.

    No checkpoint is modified. The panel is shared per batch, so every reported
    interval is paired at matched validation batches; dropout is inactive in eval
    mode, so only stored weights differ. All three endpoints were trained on one
    batch stream because the sampler depends only on batch shape and seed, which
    is checked here rather than assumed.
    """

    if seed in CONSUMED_PANEL_SEEDS or seed == DEFAULT_FRESH_VALIDATION_SEED:
        raise ValueError("budget panel seed must be unused")
    if eval_iters < 2:
        raise ValueError("paired uncertainty requires at least two batches")
    specs = tuple(branch_specs(directory) if specs is None else specs)
    plan = budget_plan(data, config, directory, specs, peer_loader)
    entries = {entry["key"]: entry for entry in plan}
    control_key = f"c{specs[0].width}_l{specs[0].depth}"
    treatment_key = f"c{specs[1].width}_l{specs[1].depth}"
    verify_paired_endpoints(entries[control_key], entries[treatment_key], specs)
    generator_streams = {
        _STAGE_41.tensor_sha256(entry["payload"]["training_generator_state"])
        for entry in plan
    }
    if len(generator_streams) != 1:
        raise ValueError("budget endpoints were not trained on one batch stream")
    if (
        expected_generator_sha256 is not None
        and generator_streams != {expected_generator_sha256}
    ):
        raise ValueError("budget batch stream is not the recorded shared lineage stream")
    # Cross-stage peers are compared through the training device's projection
    # rather than through RNG containers, because a stored peer captured a
    # different set of devices.
    projections = {
        entry["key"]: terminal_rng_normalization(entry["payload"]) for entry in plan
    }
    device_names = {value["training_device_name"] for value in projections.values()}
    if len(device_names) != 1:
        raise ValueError(
            "budget endpoints were not trained on the same device: "
            + ", ".join(sorted(str(name) for name in device_names))
        )
    verify_distinct_draw_schedules(
        "the L=6 control",
        projections[control_key]["terminal_training_rng_sha256"],
        "the L=8 treatment",
        projections[treatment_key]["terminal_training_rng_sha256"],
    )
    if BUDGET_PEER_KEY in projections and (
        projections[control_key]["terminal_training_rng_sha256"]
        != projections[BUDGET_PEER_KEY]["terminal_training_rng_sha256"]
    ):
        raise ValueError(
            "the L=6 C256 and C320 endpoints disagree on the training-device RNG "
            "state, so they were not trained on the same draw schedule"
        )
    caller_rng = capture_rng_state(device)
    models = []
    try:
        for entry in plan:
            spec = entry["attention_spec"]
            model = build_model(data, entry["config"], spec, _STAGE_44.N_HEAD, device)
            model.load_state_dict(entry["payload"]["model_state_dict"], strict=True)
            verify_model_masks(model, spec)
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
                raise ValueError("budget shootout produced non-finite loss")
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
            "shared_training_generator_sha256": (
                DEFAULT_SHARED_TRAINING_GENERATOR_SHA256
            ),
            "rng_projections": projections,
            "training_device_name": next(iter(device_names)),
            "panel_keys": {
                "control": control_key,
                "treatment": treatment_key,
                "budget_peer": BUDGET_PEER_KEY if BUDGET_PEER_KEY in entries else None,
            },
            "endpoints": {
                key: {
                    **asdict(value),
                    "width": entries[key]["width"],
                    "depth": entries[key]["depth"],
                    "dropout": entries[key]["dropout"],
                    "parameter_count": entries[key]["parameter_count"],
                    "parameter_ratio_vs_control": (
                        entries[key]["parameter_count"]
                        / entries[control_key]["parameter_count"]
                    ),
                    "provenance": entries[key]["provenance"],
                }
                for key, value in results.items()
            },
            "ranking": [
                {
                    "rank": rank,
                    "key": key,
                    "label": results[key].name,
                    "provenance": entries[key]["provenance"],
                    "width": entries[key]["width"],
                    "depth": entries[key]["depth"],
                    "parameter_count": entries[key]["parameter_count"],
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
    error is unchanged, and the interval negates and swaps endpoints. Either way
    the returned record names the requested candidate and baseline, so a caller
    never has to know which direction the panel happened to store; the endpoints'
    display names remain in the panel's own endpoint records.
    """

    forward = comparisons.get(f"{candidate}_minus_{baseline}")
    if forward is not None:
        return {**forward, "candidate": candidate, "baseline": baseline}
    reverse = comparisons.get(f"{baseline}_minus_{candidate}")
    if reverse is None:
        raise KeyError(
            f"budget panel recorded no comparison of {candidate} and {baseline}"
        )
    return {
        **reverse,
        "candidate": candidate,
        "baseline": baseline,
        "mean_delta": -reverse["mean_delta"],
        "confidence_low": -reverse["confidence_high"],
        "confidence_high": -reverse["confidence_low"],
    }


def substitution_outcome(report: dict[str, Any]) -> str:
    """Compare the same budget spent on depth (C256 L=8) versus width (C320 L=6)."""

    keys = report["panel_keys"]
    if keys["budget_peer"] is None:
        return "the depth-and-width budget substitution was not scored on this panel"
    delta = compared_delta(
        report["comparisons"], keys["treatment"], keys["budget_peer"]
    )
    depth_parameters = report["endpoints"][keys["treatment"]]["parameter_count"]
    width_parameters = report["endpoints"][keys["budget_peer"]]["parameter_count"]
    ratio = depth_parameters / width_parameters
    spend = (
        f"the {depth_parameters:,}-parameter L=8 model costs {ratio:.4f}x the "
        f"{width_parameters:,}-parameter C320 L=6 budget"
    )
    if delta["confidence_high"] < 0:
        return f"depth is the better spend: {spend} and verifies better on the shared panel"
    if delta["confidence_low"] > 0:
        return f"width is the better spend: {spend} and C320 L=6 verifies better"
    return (
        "the depth and width spends of this budget are unresolved on the shared "
        "panel; neither allocation is preferred"
    )


def budget_panel_outcome(report: dict[str, Any]) -> str:
    """Name the panel leader, then state what the depth-versus-width spend says."""

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
        leader = (
            f"{champion['label']} is the resolved champion: it beats all "
            f"{len(ranking) - 1} peers on the shared fresh panel"
        )
    else:
        undecided = ", ".join(
            entry["label"]
            for entry in ranking[1:]
            if not compared_delta(report["comparisons"], entry["key"], champion["key"])[
                "confidence_low"
            ]
            > 0
        )
        leader = (
            f"{champion['label']} leads the shared fresh panel but the panel does "
            f"not resolve it against {undecided}"
        )
    return f"{leader}. {substitution_outcome(report)}"


def depth_budget_ladder(paired: Any, panel: dict[str, Any]) -> dict[str, Any]:
    """The parameter-budget rows this stage measured, with its own depth step."""

    endpoints = panel["endpoints"]
    keys = panel["panel_keys"]
    control_key = keys["control"]
    rows = [
        {
            "key": key,
            "label": entry["name"],
            "width": entry["width"],
            "depth": entry["depth"],
            "parameters": entry["parameter_count"],
            "parameter_ratio_vs_control": entry["parameter_ratio_vs_control"],
            "source_stage": (
                EXPERIMENT_STAGE if key in (keys["control"], keys["treatment"]) else 46
            ),
            "provenance": entry["provenance"],
            "fresh_panel_seed": panel["seed"],
            "mean_validation_loss": entry["mean_loss"],
            "standard_error": entry["standard_error"],
        }
        for key, entry in endpoints.items()
    ]
    rows.sort(key=lambda row: row["parameters"])
    return {
        "rows": rows,
        "control_key": control_key,
        "measured_depth_step": {
            "width": DEFAULT_WIDTH,
            "depth_step": f"L={DEFAULT_CONTROL_DEPTH} -> L={DEFAULT_TREATMENT_DEPTH}",
            "source_stage": EXPERIMENT_STAGE,
            "fresh_panel_seed": paired.seed,
            "parameter_delta": DEFAULT_TREATMENT_PARAMETER_COUNT
            - DEFAULT_CONTROL_PARAMETER_COUNT,
            "parameter_ratio": DEFAULT_TREATMENT_PARAMETER_COUNT
            / DEFAULT_CONTROL_PARAMETER_COUNT,
            "mean_delta": paired.delta.mean_delta,
            "standard_error": paired.delta.standard_error,
            "confidence_low": paired.delta.confidence_low,
            "confidence_high": paired.delta.confidence_high,
        },
        "budget_substitution": (
            compared_delta(panel["comparisons"], keys["treatment"], keys["budget_peer"])
            if keys["budget_peer"] is not None
            else None
        ),
        "prior_depth_reference": copy.deepcopy(PRIOR_DEPTH_REFERENCE),
    }


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
                data, configs[0], spec, device
            )
            audits.append(
                {
                    "branch_key": spec.key,
                    "depth": spec.depth,
                    "layer_contexts": spec.layer_contexts,
                    "residual_dropout": spec.dropout,
                    "architecture": depth_architecture_metadata(
                        data, configs[0], spec
                    ),
                    "initialization": initialization,
                    "initial_streams": stream_hashes(model, optimizer, generator, device),
                }
            )
            del model, optimizer, generator
        shared = shared_initialization_audit(specs[0], specs[1], data, configs[0])
        for spec, audit in zip(specs, audits, strict=True):
            verify_native_initialization(
                data, configs[0], spec, audit["initialization"]
            )
        # The two stored endpoints the canonical run needs are read here, before
        # either branch spends its update budget: the control is compared against
        # one after training and the budget panel contains the other. Both are
        # authenticated by their owning stages and by their recorded file hashes.
        stage_49_reference = load_stage_49_control(data, DEFAULT_OUTPUT_DIRECTORY)
        c320_peer = load_c320_budget_peer(data, DEFAULT_OUTPUT_DIRECTORY)
        for key in ("training_generator_sha256", "torch_rng_sha256"):
            if audits[0]["initial_streams"][key] != audits[1]["initial_streams"][key]:
                raise RuntimeError(f"preflight branch random streams differ in {key}")
        if audits[0]["initial_streams"]["model_sha256"] == audits[1]["initial_streams"][
            "model_sha256"
        ]:
            raise RuntimeError("preflight branches started from the same weights")
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
            "shared_initialization": shared,
            "control_initialization_equivalence": {
                "reference": "Stage 49 recorded C256 L=6 native initialization",
                "state_sha256": audits[0]["initialization"]["state_sha256"],
                "stage_46_factory_identical": True,
                "stage_30_factory_identical": True,
                "identical": True,
            },
            "treatment_initialization_equivalence": {
                "reference": "Stage 50 recorded C256 L=8 native initialization",
                "state_sha256": audits[1]["initialization"]["state_sha256"],
                "stage_30_factory_identical": True,
                "identical": True,
            },
            "updates_per_branch": DEFAULT_TARGET_STEP,
            "target_presentations_per_branch": (
                DEFAULT_TARGET_STEP * configs[0].batch_size * configs[0].block_size
            ),
            "parameter_counts": {
                "control": audits[0]["initialization"]["parameter_count"],
                "treatment": audits[1]["initialization"]["parameter_count"],
                "added": audits[1]["initialization"]["parameter_count"]
                - audits[0]["initialization"]["parameter_count"],
                "added_fraction": (
                    audits[1]["initialization"]["parameter_count"]
                    / audits[0]["initialization"]["parameter_count"]
                    - 1.0
                ),
                "stored_c320_peer": DEFAULT_C320_PARAMETER_COUNT,
            },
            "learning_rate_schedule": schedule_metadata(DEFAULT_TARGET_STEP),
            "stored_peers": {
                "stage_49_control": {
                    "key": stage_49_reference.key,
                    "path": str(stage_49_reference.path.resolve()),
                    "file_sha256": DEFAULT_STAGE_49_CONTROL_SHA256,
                    "width": stage_49_reference.width,
                    "depth": stage_49_reference.depth,
                    "dropout": stage_49_reference.dropout,
                    "parameter_count": stage_49_reference.payload["initialization"][
                        "parameter_count"
                    ],
                    "best_step": stage_49_reference.payload["best_step"],
                    "role": "the champion the trained control must reproduce",
                },
                "c320_budget_peer": {
                    "key": c320_peer.key,
                    "path": str(c320_peer.path.resolve()),
                    "file_sha256": C320_BUDGET_REF["file_sha256"],
                    "width": c320_peer.width,
                    "depth": c320_peer.depth,
                    "dropout": c320_peer.dropout,
                    "parameter_count": c320_peer.payload["initialization"][
                        "parameter_count"
                    ],
                    "best_step": c320_peer.payload["best_step"],
                    "role": "the wider spend the depth treatment is scored against",
                },
            },
            "data_fingerprint": fingerprint_data(data),
            "initialization_policy": INITIALIZATION_POLICY,
            "randomness_policy": RANDOMNESS_POLICY,
            "historical_lineage": copy.deepcopy(HISTORICAL_LINEAGE),
        }
    finally:
        _STAGE_40._restore_rng_state(caller_rng, device)
        _STAGE_41.clear_accelerator_cache(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 50: fresh C256 L=6 versus L=8 at LLLLGG / LLLLLLGG and "
            "p=0.15, then a depth-versus-width parameter-budget shootout"
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
        "--budget-panel-only",
        action="store_true",
        help=(
            "score the stored C256 L=6, C256 L=8, and C320 L=6 endpoints on one "
            "shared fresh panel without training"
        ),
    )
    return parser.parse_args()


def print_budget_report(report: dict[str, Any]) -> None:
    for entry in report["ranking"]:
        print(
            f"  {entry['rank']}. {entry['label']} "
            f"[{entry['parameter_count']:,} params]: "
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


def run_budget_panel_only(args: argparse.Namespace) -> None:
    """Reuse the stored Stage 50 and Stage 46 endpoints; never trains or rewrites."""

    data = CharacterData.from_file(args.data, train_fraction=0.9)
    device = resolve_lineage_device(args.device)
    verify_lineage_device(device)
    config = canonical_config()
    specs = branch_specs(args.output_dir)
    validate_branch_specs(specs)
    report_path = args.output_dir / "stage_50_budget_shootout.json"
    validate_artifact_paths((args.data,), (report_path,))
    for spec in specs:
        if not spec.final_path.is_file():
            raise FileNotFoundError(f"Stage 50 endpoint not present: {spec.final_path}")
    print(
        f"Stage 50 budget shootout on {device}: C256 L=6, C256 L=8, C320 L=6 at "
        f"{DEFAULT_BUDGET_PANEL_SEED}",
        flush=True,
    )
    report = evaluate_budget_shootout(data, config, device, args.output_dir)
    outcome = budget_panel_outcome(report)
    report["budget_outcome"] = outcome
    report["uncertainty_scope"] = UNCERTAINTY_SCOPE
    report["budget_interpretation"] = (
        "one shared 500-batch validation panel; dropout inactive in eval mode; the "
        "two C256 endpoints are this stage's fresh branches and the C320 endpoint "
        "is Stage 46's stored p=0.15 treatment"
    )
    _STAGE_42.write_report(report_path, report)
    print_budget_report(report)
    print(f"{outcome}. Report: {report_path.resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    if args.budget_panel_only:
        run_budget_panel_only(args)
        return
    data = CharacterData.from_file(args.data, train_fraction=0.9)
    device = resolve_lineage_device(args.device)
    if not args.validate_only:
        verify_lineage_device(device)
    config = canonical_config()
    configs = (config, config)
    specs = branch_specs(args.output_dir)
    report_path = args.output_dir / "stage_50_report.json"
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
        f"Stage 50 on {device}: native C256 L=6 and L=8 at p=0.15, "
        "50,000 updates per branch",
        flush=True,
    )
    control = run_branch(specs[0], data, config, device)
    reproduction = verify_stage_49_control_reproduction(
        specs[0], data, config, load_stage_49_control(data, args.output_dir)
    )
    treatment = run_branch(specs[1], data, config, device)
    verify_paired_branches(control, treatment)
    fresh = evaluate_checkpoint_pair(control, treatment, data, config, device)
    _STAGE_42.print_pair(fresh)
    outcome = paired_outcome(fresh)
    panel = evaluate_budget_shootout(data, config, device, args.output_dir)
    print_budget_report(panel)
    budget_outcome = budget_panel_outcome(panel)
    control_gap = control.training.final.val - control.training.final.train
    treatment_gap = treatment.training.final.val - treatment.training.final.train
    report = {
        **audit,
        "training_executed": True,
        "paired_batch_streams_verified": True,
        "control": asdict(control),
        "treatment": asdict(treatment),
        "stage_49_control_reproduction": reproduction,
        "fresh_50k_branches": asdict(fresh),
        "budget_shootout": panel,
        "depth_budget_ladder": depth_budget_ladder(fresh, panel),
        "uncertainty_scope": UNCERTAINTY_SCOPE,
        "final_fixed_panel_delta_treatment_minus_control": treatment.training.final.val
        - control.training.final.val,
        "selected_best_fixed_panel_delta_treatment_minus_control": treatment.best_losses.val
        - control.best_losses.val,
        "final_generalization_gap_delta_treatment_minus_control": treatment_gap
        - control_gap,
        "outcome": outcome,
        "budget_outcome": budget_outcome,
        "budget_interpretation": (
            "equal updates, target tokens, width, dropout, and sampler; the "
            "treatment's extra parameters are exactly two W23 local layers"
        ),
    }
    _STAGE_42.write_report(report_path, report)
    print(f"{outcome}. {budget_outcome}. Report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
