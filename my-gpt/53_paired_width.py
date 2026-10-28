"""Stage 53: does the Stage 46 width result (C320 loses to C256) survive a new seed?

Same recipe as Stage 52: retrain the other arm of the call, C320 L=6 p=0.15, at
Stage 51's four seed sets and pair each run with the matching C256 replicate.
Seed set 0 must reproduce the stored Stage 46 treatment byte for byte.

Later seed sets (4+) train both arms here, so they can run on either card; each
pair must share one card, because the 3090 -> Ti shift is about 0.002.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from data_utils import CharacterData


_SPEC = importlib.util.spec_from_file_location(
    "stage_51_for_stage_53", Path(__file__).with_name("51_noise_floor.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE_51 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE_51
_SPEC.loader.exec_module(_STAGE_51)
_STAGE_50 = _STAGE_51._STAGE_50
_STAGE_44 = _STAGE_51._STAGE_44
_STAGE_41 = _STAGE_51._STAGE_41
_STAGE_40 = _STAGE_51._STAGE_40

EXPERIMENT_STAGE = 53
CHECKPOINTS = _STAGE_51.CHECKPOINTS
REPORT_PATH = CHECKPOINTS / "stage_53_report.json"
REPLICATES = _STAGE_51.REPLICATES
SCORED_SEEDS = range(8)
WIDTH = 320
STAGE_46_TREATMENT = _STAGE_51.STORED_ENDPOINTS["c320_l6_p015"]


def replicate_path(replicate: int, directory: Path = CHECKPOINTS) -> Path:
    return directory / f"stage_53_c320_l6_p015_seed{replicate}_final_checkpoint.pt"


def train_replicate(
    data: CharacterData, replicate: int, device: torch.device, directory: Path
) -> Path:
    """Stage 51's training loop at width 320; nothing else changes."""

    path = replicate_path(replicate, directory)
    if path.exists():
        raise FileExistsError(f"replicate already trained: {path}")
    seeds = _STAGE_51.replicate_seeds(replicate)
    config = _STAGE_50.replace(
        _STAGE_50.canonical_config(), seed=seeds["init_seed"], n_embd=WIDTH
    )
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
            "n_embd": WIDTH,
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
    """Seed set 0 at C320 must be the stored Stage 46 treatment, weight for weight."""

    trained = torch.load(path, map_location="cpu", weights_only=False)
    stored = torch.load(directory / STAGE_46_TREATMENT, map_location="cpu", weights_only=False)
    digest = _STAGE_41.tensor_mapping_sha256(trained["model_state_dict"])
    if digest != _STAGE_41.tensor_mapping_sha256(stored["model_state_dict"]):
        raise RuntimeError(
            "seed0 did not reproduce the Stage 46 treatment; this loop is not the "
            "Stage 46 arm, so no replicate is valid"
        )
    return digest


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    if _STAGE_50.fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Complete Shakespeare or its split changed")
    device = _STAGE_50.resolve_lineage_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(f"Stage {EXPERIMENT_STAGE}: paired width C320 vs C256 | device {device_name}", flush=True)

    if not args.score_only:
        if any(seed in REPLICATES for seed in args.seeds):
            _STAGE_50.verify_lineage_device(device)
        for replicate in args.seeds:
            if not _STAGE_51.replicate_path(replicate).exists():
                print(f"seed{replicate}: training C256 {_STAGE_51.replicate_seeds(replicate)}", flush=True)
                _STAGE_51.train_replicate(data, replicate, device, CHECKPOINTS)
            if replicate_path(replicate).exists():
                print(f"seed{replicate}: C{WIDTH} already trained, skipping", flush=True)
                continue
            print(f"seed{replicate}: training C{WIDTH} {_STAGE_51.replicate_seeds(replicate)}", flush=True)
            path = train_replicate(data, replicate, device, CHECKPOINTS)
            if replicate == 0:
                print(f"seed0: reproduces Stage 46 treatment, model {verify_replicate_zero(path)[:12]}", flush=True)
        if args.train_only:
            return {}

    block_size = _STAGE_50.canonical_config().block_size
    rows = []
    for replicate in SCORED_SEEDS:
        treated, control = replicate_path(replicate), _STAGE_51.replicate_path(replicate)
        if not (treated.exists() and control.exists()):
            continue
        scores, trained_on = {}, {}
        for arm, path in (("c320", treated), ("c256", control)):
            model, info = _STAGE_51.load_endpoint(data, path, device)
            scores[arm] = _STAGE_51.score_full_split(model, data.val_data, block_size, device)
            trained_on[arm] = info["trained_on"]
            del model
        if trained_on["c320"] != trained_on["c256"]:
            raise RuntimeError(f"seed{replicate} arms were trained on different cards: {trained_on}")
        result = _STAGE_51.paired_delta(scores["c320"], scores["c256"])
        rows.append({
            "replicate": replicate,
            "trained_on": trained_on["c256"],
            "c256_loss": sum(scores["c256"]) / len(scores["c256"]),
            "c320_loss": sum(scores["c320"]) / len(scores["c320"]),
            **result,
        })

    print("\nFull-split loss per seed set (C320 - C256, positive favours C256):")
    print(f"{'seed':6s} {'trained on':26s} {'C256':>10s} {'C320':>10s} {'delta':>10s}  95% window interval")
    for row in rows:
        print(
            f"seed{row['replicate']:<2d} {row['trained_on']:26s} {row['c256_loss']:10.6f} {row['c320_loss']:10.6f} "
            f"{row['delta']:+10.6f}  [{row['low']:+.6f}, {row['high']:+.6f}]"
        )

    summary = None
    if len(rows) >= 2:
        deltas = torch.tensor([row["delta"] for row in rows], dtype=torch.float64)
        mean, se = _STAGE_40._mean_and_standard_error(deltas.tolist())
        margin = _STAGE_40._paired_t_95_critical_value(len(rows)) * se
        summary = {
            "seed_sets": len(rows),
            "mean_delta": mean,
            "sd_delta": deltas.std(unbiased=True).item(),
            "low": mean - margin,
            "high": mean + margin,
            "favours_c256": int((deltas > 0).sum().item()),
        }
        print(
            f"\nAcross {summary['seed_sets']} seed sets: mean {mean:+.6f} | "
            f"sd {summary['sd_delta']:.6f} | 95% seed interval "
            f"[{summary['low']:+.6f}, {summary['high']:+.6f}] | "
            f"C256 better in {summary['favours_c256']}/{summary['seed_sets']}"
        )

    report = {
        "stage": EXPERIMENT_STAGE,
        "scoring_device": device_name,
        "width": WIDTH,
        "replicate_seeds": {f"seed{row['replicate']}": _STAGE_51.replicate_seeds(row["replicate"]) for row in rows},
        "paired_seed_sets": rows,
        "summary": summary,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", help="cuda resolves to the lineage RTX 3090")
    parser.add_argument("--score-only", action="store_true", help="skip training; score what exists")
    parser.add_argument("--train-only", action="store_true", help="skip scoring")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(REPLICATES), help="seed sets to train")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
