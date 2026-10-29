"""Stage 59: how much worse is the C320 champion with half the training text?

Stages 56-58 found the learning rate and weight decay flat to within about 0.002 around
the champion, C320 L=6 p=0.15, while its train/val gap stays near 0.28. This stage asks
whether the data is the binding constraint: it trains the unchanged champion recipe
(same schedule, 50,000 updates, weight decay 0.01) on half of the Complete Shakespeare
training split, at the same four seed sets, paired against Stage 53's full-data runs.
The vocabulary and the validation split are unchanged.

The half is every other 100,000-character block of the training split, so it samples
the whole corpus rather than only its first plays. The kept blocks are joined, which
creates 23 seams; a training window crosses one with probability below 1%.

Each pair must share one card, because the 3090 -> Ti shift is about 0.002; the
baseline seed sets 0-1 were trained on the 3090 and 6-7 on the Ti.
"""

from __future__ import annotations

import argparse
import dataclasses
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


_STAGE_56 = _load("stage_56_for_stage_59", "56_learning_rate_width.py")
_STAGE_53 = _STAGE_56._STAGE_53
_STAGE_51 = _STAGE_56._STAGE_51
_STAGE_50 = _STAGE_51._STAGE_50
_STAGE_44 = _STAGE_51._STAGE_44
_STAGE_41 = _STAGE_51._STAGE_41
_STAGE_40 = _STAGE_51._STAGE_40

EXPERIMENT_STAGE = 59
CHECKPOINTS = _STAGE_51.CHECKPOINTS
REPORT_PATH = CHECKPOINTS / "stage_59_report.json"
SCORED_SEEDS = range(8)
WIDTH = 320
WEIGHT_DECAY = 0.01
BLOCK_CHARACTERS = 100_000
make_schedule = _STAGE_50.make_schedule


def replicate_path(replicate: int, directory: Path = CHECKPOINTS) -> Path:
    return directory / f"stage_59_c320_l6_p015_half_seed{replicate}_final_checkpoint.pt"


def halve_training_split(data: CharacterData) -> CharacterData:
    """Keep every other block of the training split; vocabulary and val are unchanged."""

    blocks = torch.split(data.train_data, BLOCK_CHARACTERS)
    half = torch.cat(blocks[0::2])
    return dataclasses.replace(data, train_data=half)


def train_replicate(
    data: CharacterData, replicate: int, device: torch.device, directory: Path
) -> Path:
    """Stage 58's training loop with the champion's weight decay, on the given data."""

    path = replicate_path(replicate, directory)
    if path.exists():
        raise FileExistsError(f"replicate already trained: {path}")
    seeds = _STAGE_51.replicate_seeds(replicate)
    config = _STAGE_50.replace(
        _STAGE_50.canonical_config(),
        seed=seeds["init_seed"],
        n_embd=WIDTH,
    )
    schedule = make_schedule()
    assert config.learning_rate == schedule.for_update(0) == 1e-3
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
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
    )
    generator = torch.Generator().manual_seed(seeds["batch_seed"])
    _STAGE_41.seed_everything(seeds["rng_seed"])
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
        "weight_decay": WEIGHT_DECAY,
        "train_characters": len(data.train_data),
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
    full = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    if _STAGE_50.fingerprint_data(full) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Complete Shakespeare or its split changed")
    half = halve_training_split(full)
    device = _STAGE_50.resolve_lineage_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(
        f"Stage {EXPERIMENT_STAGE}: C{WIDTH} on half the training split "
        f"({len(half.train_data):,} of {len(full.train_data):,} characters) | device {device_name}",
        flush=True,
    )

    if not args.score_only:
        for replicate in args.seeds:
            if replicate_path(replicate).exists():
                print(f"seed{replicate}: already trained, skipping", flush=True)
                continue
            print(f"seed{replicate}: training C{WIDTH} half data {_STAGE_51.replicate_seeds(replicate)}", flush=True)
            train_replicate(half, replicate, device, CHECKPOINTS)
        if args.train_only:
            return {}

    block_size = _STAGE_50.canonical_config().block_size
    arms = {"c320_half": replicate_path, "c320_full": _STAGE_53.replicate_path}
    # Each arm's train loss is scored on the start of the text it was trained on.
    train_slices = {
        "c320_half": half.train_data[: len(full.val_data)],
        "c320_full": full.train_data[: len(full.val_data)],
    }
    rows = []
    for replicate in SCORED_SEEDS:
        paths = {arm: fn(replicate) for arm, fn in arms.items()}
        if not all(p.exists() for p in paths.values()):
            continue
        scores, train_loss, trained_on = {}, {}, {}
        for arm, path in paths.items():
            model, info = _STAGE_51.load_endpoint(full, path, device)
            scores[arm] = _STAGE_51.score_full_split(model, full.val_data, block_size, device)
            train = _STAGE_51.score_full_split(model, train_slices[arm], block_size, device)
            train_loss[arm] = sum(train) / len(train)
            trained_on[arm] = info["trained_on"]
            del model
        if len(set(trained_on.values())) != 1:
            raise RuntimeError(f"seed{replicate} arms were trained on different cards: {trained_on}")
        row = {"replicate": replicate, "trained_on": trained_on["c320_full"]}
        for arm in arms:
            val = sum(scores[arm]) / len(scores[arm])
            row[f"{arm}_loss"] = val
            row[f"{arm}_train_loss"] = train_loss[arm]
            row[f"{arm}_gap"] = val - train_loss[arm]
        row["vs_c320_full"] = _STAGE_51.paired_delta(scores["c320_half"], scores["c320_full"])
        rows.append(row)

    print("\nFull-split loss per seed set (val | train | gap = val - train):")
    print(f"{'seed':6s} {'trained on':26s} {'arm':>10s} {'val':>10s} {'train':>10s} {'gap':>10s}")
    for row in rows:
        for arm, name in (("c320_full", "full"), ("c320_half", "half")):
            print(
                f"seed{row['replicate']:<2d} {row['trained_on']:26s} {name:>10s} {row[f'{arm}_loss']:10.6f} "
                f"{row[f'{arm}_train_loss']:10.6f} {row[f'{arm}_gap']:10.6f}"
            )

    comparisons = {}
    key, label = "vs_c320_full", "C320 half data - C320 full data champion"
    print(f"\n{label} (positive means half data is worse):")
    print(f"{'seed':6s} {'trained on':26s} {'delta':>10s}  95% window interval")
    for row in rows:
        d = row[key]
        print(f"seed{row['replicate']:<2d} {row['trained_on']:26s} {d['delta']:+10.6f}  [{d['low']:+.6f}, {d['high']:+.6f}]")
    summary = None
    if len(rows) >= 2:
        summary = _STAGE_56.summarize([row[key]["delta"] for row in rows], "half_better")
        print(
            f"Across {summary['seed_sets']} seed sets: mean {summary['mean_delta']:+.6f} | "
            f"sd {summary['sd_delta']:.6f} | 95% seed interval "
            f"[{summary['low']:+.6f}, {summary['high']:+.6f}] | "
            f"half data better than full in {summary['half_better']}/{summary['seed_sets']}"
        )
    comparisons[key] = {"label": label, "summary": summary}

    spread = {arm: _STAGE_56.sample_sd([row[f"{arm}_loss"] for row in rows]) for arm in arms}
    mean_gap = {arm: sum(row[f"{arm}_gap"] for row in rows) / len(rows) if rows else None for arm in arms}
    if len(rows) >= 2:
        print(f"\nSeed-to-seed sd of val loss over {len(rows)} seed sets:")
        print(f"C320 half {spread['c320_half']:.6f} | C320 full {spread['c320_full']:.6f}")
        print("\nMean generalization gap (val - train):")
        print(f"C320 half {mean_gap['c320_half']:.6f} | C320 full {mean_gap['c320_full']:.6f}")

    report = {
        "stage": EXPERIMENT_STAGE,
        "scoring_device": device_name,
        "width": WIDTH,
        "train_characters": {"half": len(half.train_data), "full": len(full.train_data)},
        "block_characters": BLOCK_CHARACTERS,
        "replicate_seeds": {f"seed{row['replicate']}": _STAGE_51.replicate_seeds(row["replicate"]) for row in rows},
        "paired_seed_sets": rows,
        "comparisons": comparisons,
        "seed_sd": spread,
        "mean_gap": mean_gap,
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
