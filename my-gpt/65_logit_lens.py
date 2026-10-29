"""Stage 65: the logit lens. Where does the champion's answer form?

Stage 64 switched heads off. This stage trains nothing and switches nothing off. It
reads the model's prediction part-way through: after the embeddings (layer 0) and after
each of the 6 blocks, it puts the residual stream through the model's own final layer
norm and output layer (`ln_f` + `lm_head`), as if the model stopped there. After layer 6
this is exactly the model's normal output.

Two measurements for every layer:

- real text: the loss and top-1 accuracy on the same 64 validation passages of 256
  characters as Stage 62,
- copying: the Stage 63 test, as in Stage 64. The drop in loss from the first copy to the
  second on the repeated random letters, minus the same drop on the unrepeated controls.

One figure: loss on real text and copying, layer by layer.
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


_STAGE_63 = _load("stage_63_for_stage_65", "63_induction_heads.py")
_STAGE_62 = _STAGE_63._STAGE_62
_STAGE_51 = _STAGE_62._STAGE_51
_STAGE_41 = _STAGE_62._STAGE_41

ROOT = Path(__file__).resolve().parent
CHAMPION_PATH = _STAGE_62.CHAMPION_PATH
REPORT_PATH = ROOT / "checkpoints" / "stage_65_report.json"
FIGURE = ROOT / "figures" / "stage_65_logit_lens.png"
PASSAGES = _STAGE_62.PASSAGES
HALF = _STAGE_63.HALF


def layer_name(layer: int) -> str:
    return "embeddings" if layer == 0 else f"layer {layer}"


@torch.no_grad()
def lens(model: torch.nn.Module, idx: torch.Tensor) -> list[torch.Tensor]:
    """Logits read through `ln_f` + `lm_head` after the embeddings and after each block."""
    x = model._representations(idx)
    readings = [model.lm_head(model.ln_f(x))]
    for block in model.blocks:
        x = block(x)
        readings.append(model.lm_head(model.ln_f(x)))
    return readings


def draw(losses: list[float], copied: list[float], path: Path) -> None:
    layers = range(len(losses))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (axes[0], losses, ".2f", "loss", "Loss on real text, read after each layer"),
        (axes[1], copied, ".2f", "copying (repeated drop minus control drop)", "Copying on repeated random letters"),
    ]
    for ax, values, fmt, ylabel, title in panels:
        ax.plot(layers, values, marker="o", linewidth=2)
        for layer, value in zip(layers, values):
            ax.annotate(format(value, fmt), (layer, value), textcoords="offset points", xytext=(0, 8 if value >= 0 else -15), ha="center", fontsize=9)
        ax.set_xticks(list(layers), ["emb"] + [f"L{layer}" for layer in layers[1:]])
        ax.set_xlabel("read after")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.margins(y=0.15)
    axes[1].axhline(0, color="gray", linewidth=1)
    fig.suptitle("Logit lens: the champion's prediction after each layer")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(4)
    data = CharacterData.from_file(_STAGE_41.DEFAULT_COMPLETE_DATA_PATH)
    model, info = _STAGE_51.load_endpoint(data, CHAMPION_PATH, torch.device("cpu"))
    model.eval()
    n_layer, length = len(model.blocks), model.block_size
    print(f"Model: {info['file']} ({info['parameters']:,} parameters, {n_layer} layers x {model.n_head} heads)")

    stride = (len(data.val_data) - length) // PASSAGES
    passages = torch.stack([data.val_data[i * stride : i * stride + length] for i in range(PASSAGES)])
    letter_ids = data.vocabulary.encode(_STAGE_63.LETTERS)
    sequences = _STAGE_63.make_sequences(letter_ids, _STAGE_63.SEQUENCES, HALF, _STAGE_63.SEED)
    controls = _STAGE_63.random_letters(letter_ids, _STAGE_63.SEQUENCES, 2 * HALF, _STAGE_63.SEED + 1)
    print(f"Real text: {PASSAGES} x {length} validation characters (as Stage 62)")
    print(f"Copying: {len(sequences)} x {HALF} random letters repeated once, and {len(controls)} unrepeated controls (as Stage 63)")

    text = lens(model, passages[:, :-1])
    exact = torch.equal(text[-1], model(passages[:, :-1])[0])
    print(f"Layer-6 reading equals the model's own output exactly: {exact}")
    assert exact
    print()

    targets = passages[:, 1:]
    rows = []
    for layer, logits in enumerate(text):
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()).item()
        accuracy = (logits.argmax(-1) == targets).float().mean().item()
        improvement = rows[-1]["loss"] - loss if rows else None
        rows.append({"layer": layer, "loss": loss, "improvement": improvement, "accuracy": accuracy})

    print("| Read after | Real-text loss | Improvement | Top-1 accuracy |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        improvement = "-" if row["improvement"] is None else f"{row['improvement']:+.4f}"
        print(f"| {layer_name(row['layer'])} | {row['loss']:.4f} | {improvement} | {row['accuracy']:.1%} |")
    print()

    repeated, unrepeated = lens(model, sequences[:, :-1]), lens(model, controls[:, :-1])
    print("| Read after | Repeated first | Repeated second | Control first | Control second | Copying |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row, logits, control_logits in zip(rows, repeated, unrepeated):
        _, first, second = _STAGE_63.copy_losses(logits, sequences, HALF)
        _, control_first, control_second = _STAGE_63.copy_losses(control_logits, controls, HALF)
        row |= {"first": first, "second": second, "control_first": control_first, "control_second": control_second,
                "copying": (first - second) - (control_first - control_second)}
        print(f"| {layer_name(row['layer'])} | {first:.4f} | {second:.4f} | {control_first:.4f} | {control_second:.4f} | {row['copying']:.4f} |")
    print()

    total = rows[0]["loss"] - rows[-1]["loss"]
    biggest = max(rows[1:], key=lambda row: row["improvement"])
    print(f"Total improvement from embeddings to layer {n_layer}: {total:.4f}")
    print(f"Biggest single-layer improvement: layer {biggest['layer']} ({biggest['improvement']:.4f}, {biggest['improvement'] / total:.0%} of the total)")
    print("Copying per layer: " + ", ".join(f"{row['layer']}: {row['copying']:.3f}" for row in rows))

    FIGURE.parent.mkdir(exist_ok=True)
    draw([row["loss"] for row in rows], [row["copying"] for row in rows], FIGURE)
    report = {"model": info, "layers": rows}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {FIGURE.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
