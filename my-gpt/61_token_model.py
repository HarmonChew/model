"""Stage 61: does the champion predict Shakespeare better from BPE tokens than characters?

Stage 60 built a BPE tokenizer; at 1,024 tokens the text is 2.45x shorter. This stage
trains the unchanged champion architecture (C320 L=6 p=0.15, LLLLGG with W23 local
windows, 256 positions, the Stage 50 schedule, 50,000 updates of 8x256 targets) on
vocab-1,024 tokens instead of characters, at the same four seed sets.

The two models predict different units, so loss per token and loss per character are
not comparable. Both are converted to bits per character (bpc): total scored loss in
bits divided by the number of characters those scored targets cover. Lower is better.

Same compute per update means the token model sees about 2.45x more text per update
and makes about 52 passes over its training tokens instead of 21. The full validation
split is scored every 5,000 updates, and the best of those checkpoints is kept too,
in case the extra repetition overfits; the final endpoint is the primary result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from bpe import BPETokenizer
from data_utils import CharacterData


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_STAGE_56 = _load("stage_56_for_stage_61", "56_learning_rate_width.py")
_STAGE_60 = _load("stage_60_for_stage_61", "60_bpe_tokenizer.py")
_STAGE_53 = _STAGE_56._STAGE_53
_STAGE_51 = _STAGE_56._STAGE_51
_STAGE_50 = _STAGE_51._STAGE_50
_STAGE_44 = _STAGE_51._STAGE_44
_STAGE_41 = _STAGE_51._STAGE_41
_STAGE_40 = _STAGE_51._STAGE_40

EXPERIMENT_STAGE = 61
CHECKPOINTS = _STAGE_51.CHECKPOINTS
REPORT_PATH = CHECKPOINTS / "stage_61_report.json"
SCORED_SEEDS = range(8)
WIDTH = 320
VOCAB_SIZE = 1024
EVAL_EVERY = 5_000
LN2 = math.log(2)


@dataclass(frozen=True)
class TokenVocabulary:
    size: int


@dataclass(frozen=True)
class TokenData:
    """The slice of CharacterData's interface the lineage sampler and builder use."""

    vocabulary: TokenVocabulary
    train_data: torch.Tensor
    val_data: torch.Tensor
    symbol_lengths: torch.Tensor


def load_token_data(full: CharacterData) -> tuple[TokenData, BPETokenizer]:
    tokenizer = BPETokenizer.load(_STAGE_60.TOKENIZER_PATH).truncated(VOCAB_SIZE)
    train, val = _STAGE_60.split_text(full)
    data = TokenData(
        vocabulary=TokenVocabulary(tokenizer.vocab_size),
        train_data=torch.tensor(tokenizer.encode(train), dtype=torch.long),
        val_data=torch.tensor(tokenizer.encode(val), dtype=torch.long),
        symbol_lengths=torch.tensor([len(s) for s in tokenizer.symbols], dtype=torch.long),
    )
    return data, tokenizer


def replicate_path(replicate: int, kind: str = "final", directory: Path = CHECKPOINTS) -> Path:
    return directory / f"stage_61_c320_l6_p015_bpe1024_seed{replicate}_{kind}_checkpoint.pt"


def scored_characters(tokens: torch.Tensor, block_size: int, symbol_lengths: torch.Tensor) -> int:
    """Characters covered by the targets score_full_split scores."""

    usable = ((len(tokens) - 1) // block_size) * block_size
    return int(symbol_lengths[tokens[1 : usable + 1]].sum())


def bits_per_character(window_losses: list[float], block_size: int, characters: int) -> float:
    """Total loss in nats over every scored target, converted to bits per character."""

    return sum(window_losses) * block_size / characters / LN2


def token_bpc(model: torch.nn.Module, data: TokenData, block_size: int, device: torch.device) -> float:
    losses = _STAGE_51.score_full_split(model, data.val_data, block_size, device)
    return bits_per_character(losses, block_size, scored_characters(data.val_data, block_size, data.symbol_lengths))


def architecture() -> dict[str, Any]:
    return {
        "n_embd": WIDTH,
        "n_head": _STAGE_44.N_HEAD,
        "block_size": _STAGE_50.canonical_config().block_size,
        "layer_contexts": _STAGE_50.DEFAULT_CONTROL_LAYER_CONTEXTS,
        "residual_dropout": _STAGE_50.DEFAULT_DROPOUT,
        "vocab_size": VOCAB_SIZE,
    }


def train_replicate(data: TokenData, replicate: int, device: torch.device) -> None:
    final_path, best_path = replicate_path(replicate, "final"), replicate_path(replicate, "best")
    if final_path.exists():
        raise FileExistsError(f"replicate already trained: {final_path}")
    seeds = _STAGE_51.replicate_seeds(replicate)
    config = _STAGE_50.replace(_STAGE_50.canonical_config(), seed=seeds["init_seed"], n_embd=WIDTH)
    schedule = _STAGE_50.make_schedule()
    assert config.learning_rate == schedule.for_update(0) == 1e-3
    spec = _STAGE_50.make_attention_spec(config.block_size, _STAGE_50.DEFAULT_CONTROL_LAYER_CONTEXTS)
    _STAGE_41.seed_everything(config.seed)
    model = _STAGE_50.build_model(data, config, spec, _STAGE_44.N_HEAD, torch.device("cpu")).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False
    )
    generator = torch.Generator().manual_seed(seeds["batch_seed"])
    _STAGE_41.seed_everything(seeds["rng_seed"])
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    started = time.perf_counter()
    best_bpc, curve = math.inf, []

    def save(path: Path, step: int, bpc: float) -> None:
        payload = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "architecture": architecture(),
            "val_bpc": bpc,
            "data_fingerprint": _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT,
            "tokenizer": _STAGE_60.TOKENIZER_PATH.name,
            "runtime": {"device_name": device_name},
            "replicate": replicate,
            "seeds": seeds,
            "seconds": time.perf_counter() - started,
            "curve": curve,
        }
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

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
        step = update + 1
        if step % EVAL_EVERY == 0:
            # score_full_split switches to eval mode and restores train mode afterwards.
            bpc = token_bpc(model, data, config.block_size, device)
            curve.append({"step": step, "val_bpc": bpc, "train_batch_loss": loss.item()})
            print(f"  seed{replicate} update {step:6d} | train batch {loss.item():.4f} | val bpc {bpc:.4f}", flush=True)
            if bpc < best_bpc:
                best_bpc = bpc
                save(best_path, step, bpc)
    save(final_path, config.max_iters, curve[-1]["val_bpc"])


def load_token_endpoint(data: TokenData, path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _STAGE_51.build_endpoint(data, payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device), payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    full = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    if _STAGE_50.fingerprint_data(full) != _STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT:
        raise ValueError("Complete Shakespeare or its split changed")
    data, tokenizer = load_token_data(full)
    device = _STAGE_50.resolve_lineage_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    block_size = _STAGE_50.canonical_config().block_size
    print(
        f"Stage {EXPERIMENT_STAGE}: C{WIDTH} on BPE-{VOCAB_SIZE} tokens | train {len(data.train_data):,} tokens | "
        f"val {len(data.val_data):,} tokens | device {device_name}",
        flush=True,
    )

    if not args.score_only:
        for replicate in args.seeds:
            if replicate_path(replicate).exists():
                print(f"seed{replicate}: already trained, skipping", flush=True)
                continue
            print(f"seed{replicate}: training {_STAGE_51.replicate_seeds(replicate)}", flush=True)
            train_replicate(data, replicate, device)
        if args.train_only:
            return {}

    char_chars = ((len(full.val_data) - 1) // block_size) * block_size
    rows = []
    for replicate in SCORED_SEEDS:
        char_path = _STAGE_53.replicate_path(replicate)
        final_path, best_path = replicate_path(replicate, "final"), replicate_path(replicate, "best")
        if not (char_path.exists() and final_path.exists() and best_path.exists()):
            continue
        char_model, char_info = _STAGE_51.load_endpoint(full, char_path, device)
        char_losses = _STAGE_51.score_full_split(char_model, full.val_data, block_size, device)
        del char_model
        row = {
            "replicate": replicate,
            "trained_on": char_info["trained_on"],
            "char_bpc": bits_per_character(char_losses, block_size, char_chars),
            "char_val_characters": char_chars,
        }
        for kind, path in (("final", final_path), ("best", best_path)):
            model, payload = load_token_endpoint(data, path, device)
            if payload["runtime"]["device_name"] != char_info["trained_on"]:
                raise RuntimeError(f"seed{replicate}: token and character models were trained on different cards")
            row[f"token_{kind}_bpc"] = token_bpc(model, data, block_size, device)
            row[f"token_{kind}_step"] = payload["step"]
            row["token_parameters"] = sum(p.numel() for p in model.parameters())
            del model
        row["token_val_characters"] = scored_characters(data.val_data, block_size, data.symbol_lengths)
        row["delta_final"] = row["token_final_bpc"] - row["char_bpc"]
        row["delta_best"] = row["token_best_bpc"] - row["char_bpc"]
        rows.append(row)

    print("\nValidation bits per character (lower is better):")
    print(f"{'seed':6s} {'trained on':26s} {'characters':>10s} {'tokens final':>12s} {'tokens best':>12s} {'best step':>9s}")
    for r in rows:
        print(
            f"seed{r['replicate']:<2d} {r['trained_on']:26s} {r['char_bpc']:10.4f} {r['token_final_bpc']:12.4f} "
            f"{r['token_best_bpc']:12.4f} {r['token_best_step']:9d}"
        )
    if rows:
        print(
            f"\nScored characters: character model {rows[0]['char_val_characters']:,} | "
            f"token model {rows[0]['token_val_characters']:,} | "
            f"parameters: token model {rows[0]['token_parameters']:,}"
        )

    comparisons = {}
    for key, label in (("delta_final", "tokens (final) - characters"), ("delta_best", "tokens (best checkpoint) - characters")):
        if len(rows) < 2:
            break
        summary = _STAGE_56.summarize([r[key] for r in rows], "tokens_better")
        comparisons[key] = {"label": label, "summary": summary}
        print(
            f"\n{label} (negative favours tokens), same seed set and card:\n"
            f"Across {summary['seed_sets']} seed sets: mean {summary['mean_delta']:+.4f} bpc | "
            f"sd {summary['sd_delta']:.4f} | 95% seed interval [{summary['low']:+.4f}, {summary['high']:+.4f}] | "
            f"tokens better in {summary['tokens_better']}/{summary['seed_sets']}"
        )

    if rows:
        curves = [load_token_endpoint(data, replicate_path(r["replicate"]), torch.device("cpu"))[1]["curve"] for r in rows]
        print("\nToken model validation bpc during training (mean over seed sets):")
        for i, point in enumerate(curves[0]):
            mean = sum(c[i]["val_bpc"] for c in curves) / len(curves)
            print(f"  update {point['step']:6d}: {mean:.4f}")

    report = {
        "stage": EXPERIMENT_STAGE,
        "scoring_device": device_name,
        "vocab_size": VOCAB_SIZE,
        "train_tokens": len(data.train_data),
        "val_tokens": len(data.val_data),
        "rows": rows,
        "comparisons": comparisons,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: my-gpt/checkpoints/{REPORT_PATH.name}")
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
