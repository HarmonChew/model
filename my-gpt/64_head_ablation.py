"""Stage 64: which heads does the champion need? Switch each one off and see.

Stages 62-63 looked at where heads put their attention. Attention shows where a head
looks, not whether the model needs it. This stage trains nothing. It switches heads off
(sets their output to zero) and measures how much worse the model gets.

Two measurements for every switch-off:

- real text: the loss on the same 64 validation passages of 256 characters as Stage 62,
- copying: on the Stage 63 test, the drop in loss from the first copy to the second on
  the repeated sequences, minus the same drop on the unrepeated controls. With nothing
  switched off it is about 0.46. If a head does the copying, switching it off should
  bring this towards 0.

First each of the 24 heads alone, then each layer's 4 heads together.

One figure: for each head, the loss increase on real text and the copying left.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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


_STAGE_63 = _load("stage_63_for_stage_64", "63_induction_heads.py")
_STAGE_62 = _STAGE_63._STAGE_62
_STAGE_51 = _STAGE_62._STAGE_51
_STAGE_41 = _STAGE_62._STAGE_41

ROOT = Path(__file__).resolve().parent
CHAMPION_PATH = _STAGE_62.CHAMPION_PATH
REPORT_PATH = ROOT / "checkpoints" / "stage_64_report.json"
FIGURE = ROOT / "figures" / "stage_64_head_ablation.png"
PASSAGES = _STAGE_62.PASSAGES
HALF = _STAGE_63.HALF


@contextmanager
def heads_off(model: torch.nn.Module, heads: list[tuple[int, int]]) -> Iterator[None]:
    """Inside this block, each (layer, head) in `heads` outputs zeros.

    Only for the normal forward pass: a head asked for its attention weights returns a
    tuple, which this does not handle.
    """
    hooks = [
        model.blocks[layer].sa.heads[head].register_forward_hook(lambda _module, _inputs, output: torch.zeros_like(output))
        for layer, head in heads
    ]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def text_loss(model: torch.nn.Module, passages: torch.Tensor) -> float:
    logits, _ = model(passages[:, :-1])
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), passages[:, 1:].flatten()).item()


@torch.no_grad()
def copying(model: torch.nn.Module, sequences: torch.Tensor, controls: torch.Tensor, half: int) -> float:
    """First-half minus second-half loss on the repeated sequences, minus the same on the controls."""
    _, first, second = _STAGE_63.copy_losses(model(sequences[:, :-1])[0], sequences, half)
    _, control_first, control_second = _STAGE_63.copy_losses(model(controls[:, :-1])[0], controls, half)
    return (first - second) - (control_first - control_second)


def draw(increase: torch.Tensor, copied: torch.Tensor, base_copying: float, path: Path) -> None:
    n_layer, n_head = increase.shape
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    panels = [
        (axes[0], increase, "Reds", 0.05, "+.3f", lambda v: v > 0.03, "Loss increase on real text\n(colour capped at 0.05)"),
        (axes[1], copied, "Blues_r", base_copying, ".2f", lambda v: v < 0.2, f"Copying left when switched off\n(nothing off: {base_copying:.2f}; lower = head was copying)"),
    ]
    for ax, values, cmap, vmax, fmt, dark, title in panels:
        ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax)
        for layer in range(n_layer):
            for head in range(n_head):
                ax.text(head, layer, format(value := float(values[layer, head]), fmt), ha="center", va="center",
                        fontsize=10, color="white" if dark(value) else "black")
        ax.set_xticks(range(n_head), [f"H{h + 1}" for h in range(n_head)])
        ax.set_yticks(range(n_layer), [f"L{l + 1}" for l in range(n_layer)])
        ax.set_title(title)
    fig.suptitle("Switching off one head at a time")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    model, info = _STAGE_51.load_endpoint(data, CHAMPION_PATH, torch.device("cpu"))
    model.eval()
    n_layer, n_head, length = len(model.blocks), model.n_head, model.block_size
    print(f"Model: {info['file']} ({info['parameters']:,} parameters, {n_layer} layers x {n_head} heads)")

    stride = (len(data.val_data) - length) // PASSAGES
    passages = torch.stack([data.val_data[i * stride : i * stride + length] for i in range(PASSAGES)])
    letter_ids = data.vocabulary.encode(_STAGE_63.LETTERS)
    sequences = _STAGE_63.make_sequences(letter_ids, _STAGE_63.SEQUENCES, HALF, _STAGE_63.SEED)
    controls = _STAGE_63.random_letters(letter_ids, _STAGE_63.SEQUENCES, 2 * HALF, _STAGE_63.SEED + 1)
    print(f"Real text: {PASSAGES} x {length} validation characters (as Stage 62)")
    print(f"Copying: {len(sequences)} x {HALF} random letters repeated once, and {len(controls)} unrepeated controls (as Stage 63)")

    base_loss, base_copying = text_loss(model, passages), copying(model, sequences, controls, HALF)
    print(f"Nothing switched off: real-text loss {base_loss:.4f}, copying {base_copying:.4f}")
    print()

    def measure(heads: list[tuple[int, int]]) -> dict[str, float]:
        with heads_off(model, heads):
            return {"loss": text_loss(model, passages), "copying": copying(model, sequences, controls, HALF)}

    print("| Switched off | Real-text loss | Increase | Copying |")
    print("| --- | ---: | ---: | ---: |")
    rows = []
    for layer in range(n_layer):
        for head in range(n_head):
            row = {"layer": layer + 1, "head": head + 1} | measure([(layer, head)])
            rows.append(row)
            print(f"| L{layer + 1} H{head + 1} | {row['loss']:.4f} | {row['loss'] - base_loss:+.4f} | {row['copying']:.4f} |")
    print()

    print("| Switched off | Real-text loss | Increase | Copying |")
    print("| --- | ---: | ---: | ---: |")
    layers = []
    for layer in range(n_layer):
        row = {"layer": layer + 1} | measure([(layer, head) for head in range(n_head)])
        layers.append(row)
        print(f"| all of L{layer + 1} | {row['loss']:.4f} | {row['loss'] - base_loss:+.4f} | {row['copying']:.4f} |")
    print()

    ranked = sorted(rows, key=lambda row: row["loss"], reverse=True)
    print("Most needed heads: " + ", ".join(f"L{r['layer']} H{r['head']} ({r['loss'] - base_loss:+.3f})" for r in ranked[:5]))
    print("Least needed heads: " + ", ".join(f"L{r['layer']} H{r['head']} ({r['loss'] - base_loss:+.3f})" for r in ranked[-5:]))
    copier = min(rows, key=lambda row: row["copying"])
    print(f"Head whose removal cuts copying most: L{copier['layer']} H{copier['head']} ({base_copying:.4f} -> {copier['copying']:.4f})")

    increase = torch.tensor([row["loss"] - base_loss for row in rows]).view(n_layer, n_head)
    copied = torch.tensor([row["copying"] for row in rows]).view(n_layer, n_head)
    FIGURE.parent.mkdir(exist_ok=True)
    draw(increase, copied, base_copying, FIGURE)
    report = {"model": info, "base_loss": base_loss, "base_copying": base_copying, "heads": rows, "layers": layers}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {FIGURE.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
