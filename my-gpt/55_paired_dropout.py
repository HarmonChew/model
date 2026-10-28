"""Stage 55: is p=0.15 still the right dropout for the C320 champion?

Stage 53 made C320 L=6 p=0.15 the champion, but p=0.15 was tuned on C256 at the
biased lineage seed set. This stage trains C320 at p=0.10 and p=0.20 and pairs
each run with the Stage 53 C320 p=0.15 run at the same seed set and card.

Seed set 0 must reproduce the stored Stage 46 control (p=0.10) and Stage 47
treatment (p=0.20) byte for byte.
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
    "stage_53_for_stage_55", Path(__file__).with_name("53_paired_width.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_STAGE_53 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE_53
_SPEC.loader.exec_module(_STAGE_53)
_STAGE_51 = _STAGE_53._STAGE_51
_STAGE_50 = _STAGE_51._STAGE_50
_STAGE_44 = _STAGE_51._STAGE_44
_STAGE_41 = _STAGE_51._STAGE_41
_STAGE_40 = _STAGE_51._STAGE_40

EXPERIMENT_STAGE = 55
CHECKPOINTS = _STAGE_51.CHECKPOINTS
REPORT_PATH = CHECKPOINTS / "stage_55_report.json"
WIDTH = _STAGE_53.WIDTH
SCORED_SEEDS = range(8)
# Each dropout and the stored seed-set-0 endpoint it must reproduce.
DROPOUTS = {
    0.10: "stage_46_control_c320_p010_step_50000_final_checkpoint.pt",
    0.20: "stage_47_treatment_c320_p020_step_50000_final_checkpoint.pt",
}


def tag(dropout: float) -> str:
    return f"p{round(dropout * 100):03d}"


def replicate_path(dropout: float, replicate: int, directory: Path = CHECKPOINTS) -> Path:
    return directory / f"stage_55_c320_l6_{tag(dropout)}_seed{replicate}_final_checkpoint.pt"


def set_dropout(model: torch.nn.Module, probability: float) -> None:
    """nn.Dropout holds no parameters or RNG state, so initialization is unchanged."""

    modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    if not modules:
        raise RuntimeError("model has no dropout modules")
    for module in modules:
        module.p = probability
    model.dropout = probability


def train_replicate(
    data: CharacterData, dropout: float, replicate: int, device: torch.device, directory: Path
) -> Path:
    """Stage 53's C320 training loop with only the dropout probability changed."""

    path = replicate_path(dropout, replicate, directory)
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
    set_dropout(model, dropout)
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
            print(f"  {tag(dropout)} seed{replicate} update {update + 1:6d} | train batch {loss.item():.4f}", flush=True)
    payload = {
        "step": config.max_iters,
        "model_state_dict": model.state_dict(),
        "model_sha256": _STAGE_41.tensor_mapping_sha256(model.state_dict()),
        "architecture": {
            "n_embd": WIDTH,
            "n_head": _STAGE_44.N_HEAD,
            "block_size": config.block_size,
            "layer_contexts": _STAGE_50.DEFAULT_CONTROL_LAYER_CONTEXTS,
            "residual_dropout": dropout,
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


def verify_replicate_zero(dropout: float, path: Path, directory: Path = CHECKPOINTS) -> str:
    """Seed set 0 must be the stored endpoint at this dropout, weight for weight."""

    trained = torch.load(path, map_location="cpu", weights_only=False)
    stored = torch.load(directory / DROPOUTS[dropout], map_location="cpu", weights_only=False)
    digest = _STAGE_41.tensor_mapping_sha256(trained["model_state_dict"])
    if digest != _STAGE_41.tensor_mapping_sha256(stored["model_state_dict"]):
        raise RuntimeError(
            f"seed0 at {tag(dropout)} did not reproduce {DROPOUTS[dropout]}; "
            "this loop is not the stored arm, so no replicate is valid"
        )
    return digest


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    if _STAGE_50.fingerprint_data(data) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Complete Shakespeare or its split changed")
    device = _STAGE_50.resolve_lineage_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(f"Stage {EXPERIMENT_STAGE}: paired dropout at C{WIDTH} | device {device_name}", flush=True)

    if not args.score_only:
        if 0 in args.seeds:
            _STAGE_50.verify_lineage_device(device)
        for replicate in args.seeds:
            for dropout in DROPOUTS:
                if replicate_path(dropout, replicate).exists():
                    print(f"seed{replicate}: {tag(dropout)} already trained, skipping", flush=True)
                    continue
                print(f"seed{replicate}: training {tag(dropout)} {_STAGE_51.replicate_seeds(replicate)}", flush=True)
                path = train_replicate(data, dropout, replicate, device, CHECKPOINTS)
                if replicate == 0:
                    digest = verify_replicate_zero(dropout, path)
                    print(f"seed0: {tag(dropout)} reproduces {DROPOUTS[dropout]}, model {digest[:12]}", flush=True)
        if args.train_only:
            return {}

    block_size = _STAGE_50.canonical_config().block_size
    cache: dict[Path, tuple[list[float], str]] = {}

    def score(path: Path) -> tuple[list[float], str]:
        if path not in cache:
            model, info = _STAGE_51.load_endpoint(data, path, device)
            cache[path] = (_STAGE_51.score_full_split(model, data.val_data, block_size, device), info["trained_on"])
            del model
        return cache[path]

    comparisons = {}
    for dropout in DROPOUTS:
        rows = []
        for replicate in SCORED_SEEDS:
            treated, control = replicate_path(dropout, replicate), _STAGE_53.replicate_path(replicate)
            if not (treated.exists() and control.exists()):
                continue
            (treated_scores, treated_card), (control_scores, control_card) = score(treated), score(control)
            if treated_card != control_card:
                raise RuntimeError(f"seed{replicate} arms were trained on different cards")
            rows.append({
                "replicate": replicate,
                "trained_on": control_card,
                "p015_loss": sum(control_scores) / len(control_scores),
                "loss": sum(treated_scores) / len(treated_scores),
                **_STAGE_51.paired_delta(treated_scores, control_scores),
            })

        print(f"\n{tag(dropout)} vs p015 at C{WIDTH} (negative favours {tag(dropout)}):")
        print(f"{'seed':6s} {'trained on':26s} {'p015':>10s} {tag(dropout):>10s} {'delta':>10s}  95% window interval")
        for row in rows:
            print(
                f"seed{row['replicate']:<2d} {row['trained_on']:26s} {row['p015_loss']:10.6f} {row['loss']:10.6f} "
                f"{row['delta']:+10.6f}  [{row['low']:+.6f}, {row['high']:+.6f}]"
            )
        summary = None
        if len(rows) >= 2:
            deltas = [row["delta"] for row in rows]
            mean, se = _STAGE_40._mean_and_standard_error(deltas)
            margin = _STAGE_40._paired_t_95_critical_value(len(rows)) * se
            summary = {
                "seed_sets": len(rows),
                "mean_delta": mean,
                "sd_delta": torch.tensor(deltas, dtype=torch.float64).std(unbiased=True).item(),
                "low": mean - margin,
                "high": mean + margin,
                "favours_p015": sum(d > 0 for d in deltas),
            }
            print(
                f"Across {summary['seed_sets']} seed sets: mean {mean:+.6f} | "
                f"sd {summary['sd_delta']:.6f} | 95% seed interval "
                f"[{summary['low']:+.6f}, {summary['high']:+.6f}] | "
                f"p015 better in {summary['favours_p015']}/{summary['seed_sets']}"
            )
        comparisons[tag(dropout)] = {"paired_seed_sets": rows, "summary": summary}

    report = {"stage": EXPERIMENT_STAGE, "scoring_device": device_name, "width": WIDTH, "comparisons": comparisons}
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
