"""Stage 57: does the lower first-phase learning rate help the C320 champion?

Stage 56 halved the first schedule phase (1e-3 -> 5e-4, updates 0-9,999) at C384: the
mean tied the champion but the seed-to-seed spread fell about fivefold. This stage
applies the identical schedule to the champion itself, C320 L=6 p=0.15, at the same
four seed sets, paired against Stage 53's C320 runs. If it helps, the schedule is too
hot at every width; if not, the Stage 56 effect is specific to C384.

Each pair must share one card, because the 3090 -> Ti shift is about 0.002; the
baseline seed sets 0-1 were trained on the 3090 and 6-7 on the Ti.
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


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_STAGE_56 = _load("stage_56_for_stage_57", "56_learning_rate_width.py")
_STAGE_53 = _STAGE_56._STAGE_53
_STAGE_51 = _STAGE_56._STAGE_51
_STAGE_50 = _STAGE_51._STAGE_50
_STAGE_44 = _STAGE_51._STAGE_44
_STAGE_41 = _STAGE_51._STAGE_41
_STAGE_40 = _STAGE_51._STAGE_40

EXPERIMENT_STAGE = 57
CHECKPOINTS = _STAGE_51.CHECKPOINTS
REPORT_PATH = CHECKPOINTS / "stage_57_report.json"
SCORED_SEEDS = range(8)
WIDTH = 320
LOW_LEARNING_RATE = _STAGE_56.LOW_LEARNING_RATE
make_schedule = _STAGE_56.make_schedule


def replicate_path(replicate: int, directory: Path = CHECKPOINTS) -> Path:
    return directory / f"stage_57_c320_l6_p015_lr5e4_seed{replicate}_final_checkpoint.pt"


def train_replicate(
    data: CharacterData, replicate: int, device: torch.device, directory: Path
) -> Path:
    """Stage 56's training loop at width 320."""

    path = replicate_path(replicate, directory)
    if path.exists():
        raise FileExistsError(f"replicate already trained: {path}")
    seeds = _STAGE_51.replicate_seeds(replicate)
    config = _STAGE_50.replace(
        _STAGE_50.canonical_config(),
        seed=seeds["init_seed"],
        n_embd=WIDTH,
        learning_rate=LOW_LEARNING_RATE,
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
    schedule = make_schedule()
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
        "peak_learning_rate": LOW_LEARNING_RATE,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    if _STAGE_50.fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Complete Shakespeare or its split changed")
    device = _STAGE_50.resolve_lineage_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(f"Stage {EXPERIMENT_STAGE}: C{WIDTH} peak learning rate {LOW_LEARNING_RATE:g} | device {device_name}", flush=True)

    if not args.score_only:
        make_schedule()
        for replicate in args.seeds:
            if replicate_path(replicate).exists():
                print(f"seed{replicate}: already trained, skipping", flush=True)
                continue
            print(f"seed{replicate}: training C{WIDTH} lr{LOW_LEARNING_RATE:g} {_STAGE_51.replicate_seeds(replicate)}", flush=True)
            train_replicate(data, replicate, device, CHECKPOINTS)
        if args.train_only:
            return {}

    block_size = _STAGE_50.canonical_config().block_size
    arms = {
        "c320_lr5e4": replicate_path,
        "c320_lr1e3": _STAGE_53.replicate_path,
        "c384_lr5e4": _STAGE_56.replicate_path,
    }
    rows = []
    for replicate in SCORED_SEEDS:
        paths = {arm: fn(replicate) for arm, fn in arms.items()}
        if not all(p.exists() for p in paths.values()):
            continue
        scores, trained_on = {}, {}
        for arm, path in paths.items():
            model, info = _STAGE_51.load_endpoint(data, path, device)
            scores[arm] = _STAGE_51.score_full_split(model, data.val_data, block_size, device)
            trained_on[arm] = info["trained_on"]
            del model
        if len(set(trained_on.values())) != 1:
            raise RuntimeError(f"seed{replicate} arms were trained on different cards: {trained_on}")
        row = {
            "replicate": replicate,
            "trained_on": trained_on["c320_lr1e3"],
            **{f"{arm}_loss": sum(s) / len(s) for arm, s in scores.items()},
            "vs_c320_lr1e3": _STAGE_51.paired_delta(scores["c320_lr5e4"], scores["c320_lr1e3"]),
            "vs_c384_lr5e4": _STAGE_51.paired_delta(scores["c320_lr5e4"], scores["c384_lr5e4"]),
        }
        rows.append(row)

    print("\nFull-split loss per seed set:")
    print(f"{'seed':6s} {'trained on':26s} {'C320 1e-3':>10s} {'C320 5e-4':>10s} {'C384 5e-4':>10s}")
    for row in rows:
        print(
            f"seed{row['replicate']:<2d} {row['trained_on']:26s} {row['c320_lr1e3_loss']:10.6f} "
            f"{row['c320_lr5e4_loss']:10.6f} {row['c384_lr5e4_loss']:10.6f}"
        )

    comparisons = {}
    for key, label, baseline, favours in (
        ("vs_c320_lr1e3", "C320 lr5e-4 - C320 lr1e-3 champion", "C320 lr1e-3", "lower_lr_better"),
        ("vs_c384_lr5e4", "C320 lr5e-4 - C384 lr5e-4 (Stage 56)", "C384 lr5e-4", "c320_better"),
    ):
        print(f"\n{label} (negative favours C320 lr5e-4):")
        print(f"{'seed':6s} {'trained on':26s} {'delta':>10s}  95% window interval")
        for row in rows:
            d = row[key]
            print(f"seed{row['replicate']:<2d} {row['trained_on']:26s} {d['delta']:+10.6f}  [{d['low']:+.6f}, {d['high']:+.6f}]")
        summary = None
        if len(rows) >= 2:
            summary = _STAGE_56.summarize([row[key]["delta"] for row in rows], favours)
            print(
                f"Across {summary['seed_sets']} seed sets: mean {summary['mean_delta']:+.6f} | "
                f"sd {summary['sd_delta']:.6f} | 95% seed interval "
                f"[{summary['low']:+.6f}, {summary['high']:+.6f}] | "
                f"C320 lr5e-4 better than {baseline} in {summary[favours]}/{summary['seed_sets']}"
            )
        comparisons[key] = {"label": label, "summary": summary}

    spread = {
        arm: _STAGE_56.sample_sd([row[f"{arm}_loss"] for row in rows])
        for arm in ("c320_lr5e4", "c320_lr1e3", "c384_lr5e4")
    }
    if len(rows) >= 2:
        print(f"\nSeed-to-seed sd of loss over {len(rows)} seed sets:")
        print(f"C320 lr5e-4 {spread['c320_lr5e4']:.6f} | C320 lr1e-3 {spread['c320_lr1e3']:.6f} | C384 lr5e-4 (Stage 56) {spread['c384_lr5e4']:.6f}")

    report = {
        "stage": EXPERIMENT_STAGE,
        "scoring_device": device_name,
        "width": WIDTH,
        "peak_learning_rate": LOW_LEARNING_RATE,
        "replicate_seeds": {f"seed{row['replicate']}": _STAGE_51.replicate_seeds(row["replicate"]) for row in rows},
        "paired_seed_sets": rows,
        "comparisons": comparisons,
        "seed_sd": spread,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", help="cuda resolves to the lineage RTX 3090")
    parser.add_argument("--score-only", action="store_true", help="skip training; score what exists")
    parser.add_argument("--train-only", action="store_true", help="skip scoring")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 6, 7], help="seed sets to train")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
