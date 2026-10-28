"""Stage 51: measure the noise floor before trusting any more ~0.002 calls.

Two measurements, no new architecture:

1. Full-split scoring. Every stored endpoint is scored on the entire validation
   split, cut into consecutive non-overlapping T=256 windows, so every held-out
   target is scored exactly once. There is no panel seed, so the panel-to-panel
   disagreement Stage 50 found cannot occur.
2. Seed replicates. The champion (C256 L=6 p=0.15) is retrained from scratch on
   the lineage RTX 3090 with only its three seeds changed. Replicate 0 uses the
   lineage seeds and must reproduce the stored Stage 49 control byte for byte,
   which proves this short training loop is the Stage 43-50 protocol.

Each historical comparison is then read against the seed-to-seed spread.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data_utils import CharacterData


_SPEC = importlib.util.spec_from_file_location(
    "stage_50_for_stage_51", Path(__file__).with_name("50_depth_revisit.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE_50 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE_50
_SPEC.loader.exec_module(_STAGE_50)
_STAGE_49 = _STAGE_50._STAGE_49
_STAGE_44 = _STAGE_50._STAGE_44
_STAGE_41 = _STAGE_50._STAGE_41
_STAGE_40 = _STAGE_50._STAGE_40

EXPERIMENT_STAGE = 51
CHECKPOINTS = _STAGE_41.CHECKPOINT_DIRECTORY
REPORT_PATH = CHECKPOINTS / "stage_51_report.json"
REPLICATES = (0, 1, 2, 3)
SEED_OFFSET = 1_000
SCORING_BATCH_WINDOWS = 64

# Final (step 50,000) endpoints already on disk, keyed by a short name.
STORED_ENDPOINTS = {
    "c256_l6_p010": "stage_48_control_c256_p010_step_50000_final_checkpoint.pt",
    "c256_l6_p015": "stage_49_control_c256_p015_step_50000_final_checkpoint.pt",
    "c256_l6_p020": "stage_49_treatment_c256_p020_step_50000_final_checkpoint.pt",
    "c320_l6_p015": "stage_46_treatment_c320_p015_step_50000_final_checkpoint.pt",
    "c256_l6_p015_ti": "stage_50_control_c256_l6_step_50000_final_checkpoint.pt",
    "c256_l8_p015_ti": "stage_50_treatment_c256_l8_step_50000_final_checkpoint.pt",
}

# The calls earlier stages made, as (stage, question, candidate, baseline).
HISTORICAL_CALLS = (
    (48, "dropout 0.10 -> 0.15", "c256_l6_p015", "c256_l6_p010"),
    (49, "dropout 0.15 -> 0.20", "c256_l6_p020", "c256_l6_p015"),
    (50, "depth L=6 -> L=8 (Ti)", "c256_l8_p015_ti", "c256_l6_p015_ti"),
    (50, "device 3090 -> Ti", "c256_l6_p015_ti", "c256_l6_p015"),
    (46, "width C256 -> C320", "c320_l6_p015", "c256_l6_p015"),
)


def replicate_seeds(replicate: int) -> dict[str, int]:
    """Initialization, batch-stream, and dropout seeds; replicate 0 is the lineage."""

    offset = SEED_OFFSET * replicate
    return {
        "init_seed": _STAGE_50.canonical_config().seed + offset,
        "batch_seed": _STAGE_50.DEFAULT_TRAINING_BATCH_SEED + offset,
        "rng_seed": _STAGE_50.DEFAULT_TRAINING_RNG_SEED + offset,
    }


def replicate_path(replicate: int, directory: Path = CHECKPOINTS) -> Path:
    return directory / f"stage_51_c256_l6_p015_seed{replicate}_final_checkpoint.pt"


def build_endpoint(data: CharacterData, architecture: dict[str, Any]) -> torch.nn.Module:
    """Rebuild any LLLL..GG endpoint of this lineage from its recorded architecture."""

    config = _STAGE_50.canonical_config()
    config = _STAGE_50.replace(
        config, n_embd=architecture["n_embd"], block_size=architecture["block_size"]
    )
    spec = _STAGE_50.make_attention_spec(
        config.block_size, tuple(architecture["layer_contexts"])
    )
    return _STAGE_50.build_model(data, config, spec, architecture["n_head"])


def load_endpoint(
    data: CharacterData, path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["step"] != _STAGE_50.DEFAULT_TARGET_STEP:
        raise ValueError(f"{path.name} is not a step-50,000 endpoint")
    if payload["data_fingerprint"] != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError(f"{path.name} was trained on a different corpus or split")
    model = build_endpoint(data, payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    info = {
        "file": path.name,
        "trained_on": payload["runtime"]["device_name"],
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model.to(device), info


@torch.no_grad()
def score_full_split(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    block_size: int,
    device: torch.device,
    batch_windows: int = SCORING_BATCH_WINDOWS,
) -> list[float]:
    """Mean loss of each consecutive non-overlapping window, in order.

    Every target in the first ``windows * block_size`` positions is scored once.
    Windows are equal length, so the mean of the window losses is the mean loss
    over every scored target.
    """

    windows = (len(tokens) - 1) // block_size
    if windows < 2:
        raise ValueError("the split must hold at least two full windows")
    usable = windows * block_size
    inputs = tokens[:usable].view(windows, block_size)
    targets = tokens[1 : usable + 1].view(windows, block_size)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        for start in range(0, windows, batch_windows):
            x = inputs[start : start + batch_windows].to(device)
            y = targets[start : start + batch_windows].to(device)
            logits, _ = model(x)
            per_target = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).double(),
                y.reshape(-1),
                reduction="none",
            )
            losses.extend(per_target.view(len(x), block_size).mean(dim=1).tolist())
    finally:
        model.train(was_training)
    return losses


def paired_delta(candidate: list[float], baseline: list[float]) -> dict[str, float]:
    """Per-window paired difference with the lineage's Student-t 95% interval."""

    if len(candidate) != len(baseline):
        raise ValueError("paired scores must cover the same windows")
    differences = [c - b for c, b in zip(candidate, baseline, strict=True)]
    mean, se = _STAGE_40._mean_and_standard_error(differences)
    margin = _STAGE_40._paired_t_95_critical_value(len(differences)) * se
    return {"delta": mean, "se": se, "low": mean - margin, "high": mean + margin}


def seed_spread(losses: list[float]) -> dict[str, float]:
    if len(losses) < 2:
        raise ValueError("seed spread needs at least two replicates")
    values = torch.tensor(losses, dtype=torch.float64)
    sd = values.std(unbiased=True).item()
    return {
        "replicates": len(losses),
        "mean": values.mean().item(),
        "sd": sd,
        "min": values.min().item(),
        "max": values.max().item(),
        # Two independent single-seed runs differ with SD sqrt(2) * sd.
        "two_run_threshold": 2 * math.sqrt(2) * sd,
    }


def train_replicate(
    data: CharacterData, replicate: int, device: torch.device, directory: Path
) -> Path:
    """The Stage 43-50 update loop with the fixed-panel evaluations left out.

    Those evaluations are checked by every earlier stage not to touch the RNG or
    batch stream, so dropping them leaves the trajectory unchanged; replicate 0
    proves it against the stored Stage 49 control.
    """

    path = replicate_path(replicate, directory)
    if path.exists():
        raise FileExistsError(f"replicate already trained: {path}")
    seeds = replicate_seeds(replicate)
    config = _STAGE_50.replace(_STAGE_50.canonical_config(), seed=seeds["init_seed"])
    spec = _STAGE_50.make_attention_spec(
        config.block_size, _STAGE_50.DEFAULT_CONTROL_LAYER_CONTEXTS
    )
    _STAGE_41.seed_everything(config.seed)
    model = _STAGE_50.build_model(data, config, spec, _STAGE_44.N_HEAD, torch.device("cpu"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
    )
    generator = torch.Generator().manual_seed(seeds["batch_seed"])
    _STAGE_41.seed_everything(seeds["rng_seed"])
    schedule = _STAGE_50.make_schedule()
    started = time.perf_counter()
    model.train()
    for update in range(config.max_iters):
        _STAGE_40._set_optimizer_learning_rate(optimizer, schedule.for_update(update))
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = _STAGE_50.get_batch(data, "train", config, generator, device)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss).item():
            raise ValueError(f"non-finite training loss at update {update}")
        loss.backward()
        optimizer.step()
        if (update + 1) % 5_000 == 0:
            print(f"  seed{replicate} update {update + 1:6d} | train batch {loss.item():.4f}", flush=True)
    payload = {
        "step": config.max_iters,
        "model_state_dict": model.state_dict(),
        "model_sha256": _STAGE_41.tensor_mapping_sha256(model.state_dict()),
        "architecture": {
            "n_embd": config.n_embd,
            "n_head": _STAGE_44.N_HEAD,
            "block_size": config.block_size,
            "layer_contexts": _STAGE_50.DEFAULT_CONTROL_LAYER_CONTEXTS,
            "residual_dropout": _STAGE_50.DEFAULT_DROPOUT,
        },
        "data_fingerprint": _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT,
        "runtime": {"device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)},
        "replicate": replicate,
        "seeds": seeds,
        "seconds": time.perf_counter() - started,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def verify_replicate_zero(path: Path, directory: Path = CHECKPOINTS) -> str:
    """Replicate 0 must be the stored Stage 49 control, weight for weight."""

    trained = torch.load(path, map_location="cpu", weights_only=False)
    stored = torch.load(
        directory / STORED_ENDPOINTS["c256_l6_p015"], map_location="cpu", weights_only=False
    )
    digest = _STAGE_41.tensor_mapping_sha256(trained["model_state_dict"])
    if digest != _STAGE_41.tensor_mapping_sha256(stored["model_state_dict"]):
        raise RuntimeError(
            "replicate 0 did not reproduce the Stage 49 control; the short "
            "training loop is not the lineage protocol, so no replicate is valid"
        )
    return digest


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    if _STAGE_50.fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Complete Shakespeare or its split changed")
    device = _STAGE_50.resolve_lineage_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(f"Stage {EXPERIMENT_STAGE}: noise floor | device {device_name}", flush=True)

    if not args.score_only:
        _STAGE_50.verify_lineage_device(device)
        for replicate in REPLICATES:
            if replicate_path(replicate).exists():
                print(f"seed{replicate}: already trained, skipping", flush=True)
                continue
            print(f"seed{replicate}: training {replicate_seeds(replicate)}", flush=True)
            path = train_replicate(data, replicate, device, CHECKPOINTS)
            if replicate == 0:
                print(f"seed0: reproduces Stage 49 control, model {verify_replicate_zero(path)[:12]}", flush=True)

    endpoints = {key: CHECKPOINTS / name for key, name in STORED_ENDPOINTS.items()}
    for replicate in REPLICATES:
        if replicate_path(replicate).exists():
            endpoints[f"seed{replicate}"] = replicate_path(replicate)

    block_size = _STAGE_50.canonical_config().block_size
    scores: dict[str, list[float]] = {}
    info: dict[str, dict[str, Any]] = {}
    for key, path in endpoints.items():
        model, info[key] = load_endpoint(data, path, device)
        scores[key] = score_full_split(model, data.val_data, block_size, device)
        info[key]["full_split_loss"] = sum(scores[key]) / len(scores[key])
        del model
    windows = len(next(iter(scores.values())))

    print(f"\nFull validation split: {windows} windows x {block_size} = {windows * block_size} targets")
    print(f"{'endpoint':18s} {'trained on':26s} {'params':>10s} {'loss':>10s}")
    for key, row in info.items():
        print(f"{key:18s} {row['trained_on']:26s} {row['parameters']:10d} {row['full_split_loss']:10.6f}")

    seed_keys = [f"seed{r}" for r in REPLICATES if f"seed{r}" in scores]
    spread = None
    if len(seed_keys) >= 2:
        spread = seed_spread([info[key]["full_split_loss"] for key in seed_keys])
        print(
            f"\nSeed spread, C256 L=6 p=0.15 on the 3090 ({spread['replicates']} replicates): "
            f"mean {spread['mean']:.6f} | sd {spread['sd']:.6f} | "
            f"range {spread['min']:.6f}..{spread['max']:.6f}"
        )
        print(f"Two single-seed runs must differ by more than {spread['two_run_threshold']:.6f} (2*sqrt(2)*sd)")

    print("\nHistorical calls, rescored on the full split (candidate - baseline):")
    calls = []
    for stage, question, candidate, baseline in HISTORICAL_CALLS:
        result = paired_delta(scores[candidate], scores[baseline])
        verdict = None
        if spread is not None:
            verdict = (
                "clears seed noise"
                if abs(result["delta"]) > spread["two_run_threshold"]
                else "within seed noise"
            )
        calls.append({"stage": stage, "question": question, "candidate": candidate,
                      "baseline": baseline, **result, "verdict": verdict})
        print(
            f"  Stage {stage} {question:24s} {result['delta']:+.6f} "
            f"[{result['low']:+.6f}, {result['high']:+.6f}]"
            + (f"  {verdict}" if verdict else "")
        )

    report = {
        "stage": EXPERIMENT_STAGE,
        "scoring_device": device_name,
        "block_size": block_size,
        "windows": windows,
        "endpoints": info,
        "seed_spread": spread,
        "replicate_seeds": {f"seed{r}": replicate_seeds(r) for r in REPLICATES},
        "historical_calls": calls,
        "window_losses": scores,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", help="cuda resolves to the lineage RTX 3090")
    parser.add_argument("--score-only", action="store_true", help="skip training; score what exists")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
