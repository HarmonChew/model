"""Stage 62: what the champion's attention heads look at.

Every stage so far trained a model and measured its loss. This one trains nothing:
it opens the Stage 53 champion (C320, 6 layers x 4 heads = 24 heads) and records
where each head puts its attention on 64 validation passages of 256 characters.

Each head is scored on three simple patterns. For every query character, how much
of its attention lands on:

- fixed offset: the character exactly k places back, for the head's best k in 0-8
  (k = 1 is "the previous character"),
- word start:   the most recent space or newline before it (the start of its word),
- line start:   the most recent newline before it (the start of its line).

It also reports how far back the head looks on average. A head is labelled with
its strongest pattern when that pattern gets at least 30% of its attention;
otherwise it is "spread". Note that layers 1-4 of the champion can only see the
last 23 characters (its local/global design from Stage 29); layers 5-6 see all 256.

Two figures: all 24 heads on one passage, and a labelled close-up of the
clearest heads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from data_utils import CharacterData  # noqa: E402


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_STAGE_51 = _load("stage_51_for_stage_62", "51_noise_floor.py")
_STAGE_41 = _STAGE_51._STAGE_41

ROOT = Path(__file__).resolve().parent
CHAMPION_PATH = ROOT / "checkpoints" / "stage_53_c320_l6_p015_seed0_final_checkpoint.pt"
REPORT_PATH = ROOT / "checkpoints" / "stage_62_report.json"
GRID_FIGURE = ROOT / "figures" / "stage_62_attention_grid.png"
CLOSEUP_FIGURE = ROOT / "figures" / "stage_62_attention_closeup.png"
PASSAGES = 64
BATCH = 8
MIN_QUERY = 32  # skip the first queries, which have almost nothing to look back at
LABEL_THRESHOLD = 0.30
PATTERNS = ("offset", "word start", "line start")
MAX_OFFSET = 8
SHOW_CHARACTERS = 64  # the figures show the last 64 characters of the example passage
EXAMPLE_PASSAGE = 5


@torch.no_grad()
def forward_with_attention(model: torch.nn.Module, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's own forward pass, also returning attention: (B, layer, head, T, T)."""
    x = model._representations(idx)
    weights = []
    for block in model.blocks:
        update, w = block.sa(block.ln1(x), return_weights=True)
        x = x + update
        x = x + block.ffwd(block.ln2(x))
        weights.append(w)
    logits = model.lm_head(model.ln_f(x))
    return logits, torch.stack(weights, dim=1)


def last_position_of(idx: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """For each position, the latest earlier position holding one of `ids` (-1 if none)."""
    batch, length = idx.shape
    result = torch.full((batch, length), -1, dtype=torch.long)
    last = torch.full((batch,), -1, dtype=torch.long)
    is_match = torch.isin(idx, torch.tensor(ids))
    for t in range(length):
        result[:, t] = last
        last = torch.where(is_match[:, t], torch.full_like(last, t), last)
    return result


def head_scores(weights: torch.Tensor, idx: torch.Tensor, space_ids: list[int], newline_ids: list[int]) -> dict[str, torch.Tensor]:
    """Average share of attention on each pattern, per (layer, head)."""
    _, _, _, length, _ = weights.shape
    queries = torch.arange(length)

    def share_at(keys: torch.Tensor) -> torch.Tensor:
        # keys: (B, T) key position per query, -1 = none (scores 0 there).
        gathered = weights.gather(-1, keys.clamp(min=0)[:, None, None, :, None].expand(*weights.shape[:4], 1))
        return gathered.squeeze(-1) * (keys >= 0)[:, None, None, :]

    batch = idx.shape[0]
    shares = {
        f"offset {k}": share_at((queries - k).masked_fill(queries < k, -1).expand(batch, length))
        for k in range(MAX_OFFSET + 1)
    }
    shares |= {
        "word start": share_at(last_position_of(idx, space_ids)),
        "line start": share_at(last_position_of(idx, newline_ids)),
    }
    distance = (queries[:, None] - queries[None, :]).clamp(min=0).float()
    shares["distance"] = (weights * distance).sum(-1)
    return {name: share[..., MIN_QUERY:].mean(dim=(0, 3)) for name, share in shares.items()}


def pick_best_offset(scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace the per-offset shares with each head's best offset and its share."""
    scores = dict(scores)
    by_offset = torch.stack([scores.pop(f"offset {k}") for k in range(MAX_OFFSET + 1)])
    scores["offset"], scores["best offset"] = by_offset.max(dim=0)
    return scores


def label(scores: dict[str, float]) -> str:
    best = max(PATTERNS, key=lambda name: scores[name])
    if scores[best] < LABEL_THRESHOLD:
        return "spread"
    return f"offset {scores['best offset']}" if best == "offset" else best


def show(ch: str) -> str:
    return {"\n": "↵", " ": "␣"}.get(ch, ch)


def draw_grid(weights: torch.Tensor, path: Path) -> None:
    n_layer, n_head = weights.shape[:2]
    fig, axes = plt.subplots(n_layer, n_head, figsize=(2.2 * n_head, 2.2 * n_layer))
    for layer in range(n_layer):
        for head in range(n_head):
            ax = axes[layer, head]
            ax.imshow(weights[layer, head, -SHOW_CHARACTERS:, -SHOW_CHARACTERS:], cmap="Blues", vmin=0, vmax=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L{layer + 1} H{head + 1}", fontsize=9)
    fig.suptitle("Attention of all 24 heads on the last 64 characters of one passage\n(rows = query, columns = key)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_closeup(weights: torch.Tensor, text: str, heads: list[tuple[int, int, str]], path: Path) -> None:
    chars = [show(c) for c in text[-SHOW_CHARACTERS:]]
    ticks = range(SHOW_CHARACTERS)
    fig, axes = plt.subplots(1, len(heads), figsize=(7 * len(heads), 7.4))
    for ax, (layer, head, name) in zip(axes, heads):
        ax.imshow(weights[layer, head, -SHOW_CHARACTERS:, -SHOW_CHARACTERS:], cmap="Blues", vmin=0, vmax=0.6)
        ax.set_xticks(ticks, chars, fontsize=6, family="monospace")
        ax.set_yticks(ticks, chars, fontsize=6, family="monospace")
        ax.set_title(f"Layer {layer + 1}, head {head + 1}: {name}")
        ax.set_xlabel("looked at (key)")
        ax.set_ylabel("reading (query)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    model, info = _STAGE_51.load_endpoint(data, CHAMPION_PATH, torch.device("cpu"))
    model.eval()
    length = model.block_size
    print(f"Model: {info['file']} ({info['parameters']:,} parameters, {len(model.blocks)} layers x {model.n_head} heads)")

    stride = (len(data.val_data) - length) // PASSAGES
    passages = torch.stack([data.val_data[i * stride : i * stride + length] for i in range(PASSAGES)])
    space_ids = data.vocabulary.encode(" \n")
    newline_ids = data.vocabulary.encode("\n")
    print(f"Passages: {PASSAGES} x {length} validation characters, scoring queries {MIN_QUERY}-{length - 1}")

    totals: dict[str, torch.Tensor] = {}
    loss_sum = 0.0
    for start in range(0, PASSAGES, BATCH):
        idx = passages[start : start + BATCH]
        logits, weights = forward_with_attention(model, idx[:, :-1])
        model_logits, _ = model(idx[:, :-1])
        if not torch.allclose(logits, model_logits, atol=1e-5):
            raise AssertionError("forward_with_attention does not match the model's forward pass")
        loss_sum += torch.nn.functional.cross_entropy(logits.flatten(0, 1), idx[:, 1:].flatten()).item() * len(idx)
        for name, value in head_scores(weights, idx[:, :-1], space_ids, newline_ids).items():
            totals[name] = totals.get(name, 0) + value * len(idx)
        if start == (EXAMPLE_PASSAGE // BATCH) * BATCH:
            example_weights = weights[EXAMPLE_PASSAGE - start]
    scores = pick_best_offset({name: total / PASSAGES for name, total in totals.items()})
    print(f"Loss on these passages: {loss_sum / PASSAGES:.4f} (the champion's usual validation loss is about 1.35)")
    print()

    print("| Head | Best offset | Offset share | Word start | Line start | Mean distance | Label |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    rows = []
    for layer in range(len(model.blocks)):
        for head in range(model.n_head):
            row = {name: float(scores[name][layer, head]) for name in (*PATTERNS, "distance")}
            row["best offset"] = int(scores["best offset"][layer, head])
            row.update(layer=layer + 1, head=head + 1, label=label(row))
            rows.append(row)
            print(
                f"| L{layer + 1} H{head + 1} | {row['best offset']} | {row['offset']:.2f} | {row['word start']:.2f} "
                f"| {row['line start']:.2f} | {row['distance']:.1f} | {row['label']} |"
            )
    print()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print("Heads per label: " + ", ".join(f"{name} {count}" for name, count in counts.items()))

    closeup = []
    for name in PATTERNS:
        best = max(rows, key=lambda row: row[name])
        title = f"offset {best['best offset']}" if name == "offset" else name
        closeup.append((best["layer"] - 1, best["head"] - 1, f"{title} ({best[name]:.0%})"))
        print(f"Clearest {title} head: L{best['layer']} H{best['head']} ({best[name]:.2f})")

    example_text = data.vocabulary.decode(passages[EXAMPLE_PASSAGE, :-1].tolist())
    print()
    print(f"Example passage (last {SHOW_CHARACTERS} characters):")
    print(repr(example_text[-SHOW_CHARACTERS:]))

    GRID_FIGURE.parent.mkdir(exist_ok=True)
    draw_grid(example_weights, GRID_FIGURE)
    draw_closeup(example_weights, example_text, closeup, CLOSEUP_FIGURE)
    REPORT_PATH.write_text(json.dumps({"model": info, "heads": rows}, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {GRID_FIGURE.relative_to(ROOT)}, {CLOSEUP_FIGURE.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
