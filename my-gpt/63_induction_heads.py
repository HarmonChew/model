"""Stage 63: does the champion copy? An induction-head test.

Stage 62 found that layers 5-6 look about 55 characters back in a way no simple pattern
explained. The leading idea is induction: "the current character appeared before; copy
what came after it".

This stage trains nothing. It builds 64 sequences of 100 random lowercase letters, each
repeated once (200 characters). Random letters cannot be predicted from Shakespeare
(a uniform guesser would pay ln 26 = 3.26 per character). 100 is more than the 23
characters layers 1-4 can see, so only layers 5-6 can reach the first copy.

The loss can also drift down along a sequence for reasons that have nothing to do with
copying. So the same is done on 64 control sequences of 200 random letters with no
repeat. Copying = (first-copy loss - second-copy loss) on the repeated sequences, minus
the same drop on the controls.

Each of the 24 heads is scored on the second copy, averaged over its queries:

- induction: attention on the position 99 back (the character right after the earlier
  copy of the current character),
- duplicate: attention on the position 100 back (the earlier copy of the current
  character itself).

A head is labelled when a score is at least 0.3.

Two figures: loss by position, and the attention map of the strongest induction head.
"""

from __future__ import annotations

import importlib.util
import json
import math
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


_STAGE_62 = _load("stage_62_for_stage_63", "62_attention_maps.py")
_STAGE_51 = _STAGE_62._STAGE_51
_STAGE_41 = _STAGE_62._STAGE_41

ROOT = Path(__file__).resolve().parent
CHAMPION_PATH = _STAGE_62.CHAMPION_PATH
REPORT_PATH = ROOT / "checkpoints" / "stage_63_report.json"
LOSS_FIGURE = ROOT / "figures" / "stage_63_loss_by_position.png"
HEAD_FIGURE = ROOT / "figures" / "stage_63_induction_head.png"
SEQUENCES = 64
HALF = 100  # random characters before the repeat
LABEL_THRESHOLD = 0.30
SEED = 0
LETTERS = "abcdefghijklmnopqrstuvwxyz"
EXAMPLE_SEQUENCE = 0


def random_letters(letter_ids: list[int], count: int, length: int, seed: int) -> torch.Tensor:
    """`count` sequences of `length` letters drawn uniformly from `letter_ids`."""
    generator = torch.Generator().manual_seed(seed)
    return torch.tensor(letter_ids)[torch.randint(len(letter_ids), (count, length), generator=generator)]


def make_sequences(letter_ids: list[int], count: int, half: int, seed: int) -> torch.Tensor:
    """`count` sequences of `half` random letters, each followed by an exact repeat."""
    first = random_letters(letter_ids, count, half, seed)
    return torch.cat([first, first], dim=1)


def copy_losses(logits: torch.Tensor, sequences: torch.Tensor, half: int) -> tuple[torch.Tensor, float, float]:
    """Per-position loss, then the mean loss on the first half and on the second half.

    loss[i] is the loss on predicting character i + 1. Character `half` (the first one of
    the second half) is left out: nothing before it says what it will be.
    """
    loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), sequences[:, 1:], reduction="none").mean(0)
    return loss, loss[: half - 1].mean().item(), loss[half:].mean().item()


def repeat_scores(weights: torch.Tensor, half: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Average attention, per (layer, head), on `half - 1` back (induction) and `half` back (duplicate)."""
    length = weights.shape[-1]
    queries = torch.arange(half, length)  # queries in the second copy
    induction = weights[..., queries, queries - (half - 1)].mean(dim=(0, 3))
    duplicate = weights[..., queries, queries - half].mean(dim=(0, 3))
    return induction, duplicate


def label(induction: float, duplicate: float) -> str:
    names = [name for name, score in (("induction", induction), ("duplicate", duplicate)) if score >= LABEL_THRESHOLD]
    return " + ".join(names) if names else "-"


def draw_loss(loss: torch.Tensor, control_loss: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(1, len(loss) + 1), control_loss, linewidth=1.0, color="silver", label="control: no repeat")
    ax.plot(range(1, len(loss) + 1), loss, linewidth=1.2, label="repeated sequences")
    ax.axvline(HALF, color="gray", linestyle="--", label="start of the repeat")
    ax.axhline(math.log(26), color="red", linestyle=":", label="guessing: ln 26 = 3.26")
    ax.set_xlabel("position in the sequence")
    ax.set_ylabel("loss (average over 64 sequences)")
    ax.set_title("Loss on 100 random letters, then the same 100 again")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_head(weights: torch.Tensor, layer: int, head: int, score: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    image = ax.imshow(weights[layer, head], cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Layer {layer + 1}, head {head + 1}: attention on one repeated sequence\n(stripe 99 back = induction, {score:.0%} on average)")
    ax.set_xlabel("looked at (key position)")
    ax.set_ylabel("reading (query position)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(4)
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    model, info = _STAGE_51.load_endpoint(data, CHAMPION_PATH, torch.device("cpu"))
    model.eval()
    print(f"Model: {info['file']} ({info['parameters']:,} parameters, {len(model.blocks)} layers x {model.n_head} heads)")

    letter_ids = data.vocabulary.encode(LETTERS)
    sequences = make_sequences(letter_ids, SEQUENCES, HALF, SEED)
    controls = random_letters(letter_ids, SEQUENCES, 2 * HALF, SEED + 1)
    print(f"Sequences: {SEQUENCES} x {HALF} random letters, repeated once ({2 * HALF} characters)")
    print(f"Controls: {SEQUENCES} x {2 * HALF} random letters, no repeat")

    logits, weights = _STAGE_62.forward_with_attention(model, sequences[:, :-1])
    loss, first, second = copy_losses(logits, sequences, HALF)
    control_loss, control_first, control_second = copy_losses(model(controls[:, :-1])[0], controls, HALF)
    copying = (first - second) - (control_first - control_second)
    print(f"Guessing baseline ln(26): {math.log(26):.4f}")
    print()
    print(f"| Sequences | First half (positions 1-{HALF - 1}) | Second half (positions {HALF + 1}-{2 * HALF - 1}) | Drop |")
    print("| --- | ---: | ---: | ---: |")
    print(f"| Repeated | {first:.4f} | {second:.4f} | {first - second:.4f} |")
    print(f"| Control, no repeat | {control_first:.4f} | {control_second:.4f} | {control_first - control_second:.4f} |")
    print()
    print(f"Copying (repeated drop minus control drop): {copying:.4f}")
    print()

    induction, duplicate = repeat_scores(weights, HALF)
    print("| Head | Induction | Duplicate | Label |")
    print("| --- | ---: | ---: | --- |")
    rows = []
    for layer in range(len(model.blocks)):
        for head in range(model.n_head):
            row = {"layer": layer + 1, "head": head + 1, "induction": float(induction[layer, head]), "duplicate": float(duplicate[layer, head])}
            row["label"] = label(row["induction"], row["duplicate"])
            rows.append(row)
            print(f"| L{layer + 1} H{head + 1} | {row['induction']:.2f} | {row['duplicate']:.2f} | {row['label']} |")
    print()
    best = max(rows, key=lambda row: row["induction"])
    print(f"Strongest induction head: L{best['layer']} H{best['head']} ({best['induction']:.2f})")
    print(f"Induction heads: {sum('induction' in row['label'] for row in rows)}, duplicate heads: {sum('duplicate' in row['label'] for row in rows)}")

    LOSS_FIGURE.parent.mkdir(exist_ok=True)
    draw_loss(loss, control_loss, LOSS_FIGURE)
    draw_head(weights[EXAMPLE_SEQUENCE], best["layer"] - 1, best["head"] - 1, best["induction"], HEAD_FIGURE)
    report = {"model": info, "first_copy_loss": first, "second_copy_loss": second,
              "control_first_loss": control_first, "control_second_loss": control_second, "copying": copying, "baseline": math.log(26), "heads": rows}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {LOSS_FIGURE.relative_to(ROOT)}, {HEAD_FIGURE.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
