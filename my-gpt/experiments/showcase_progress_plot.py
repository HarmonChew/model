"""Render the portfolio progress chart from docs/stage_results.csv only."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "showcase_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "docs" / "stage_results.csv"
OUT_PATH = ROOT / "figures" / "showcase_progress.png"
COLORS = {"tiny": "#2768a5", "complete": "#d46625"}
LABELS = {"tiny": "TinyShakespeare (1.1M characters)", "complete": "Complete Shakespeare (5.3M characters)"}
# stage -> (text, (dx, dy) offset in points)
MILESTONES = {
    8: ("Stage 8: first stacked\ntransformer blocks", (10, 30)),
    24: ("Stage 24: six layers", (-10, -42)),
    26: ("Stage 26: 128-character\ncontext champion", (28, 38)),
    53: ("Stage 53: C320 champion\n(8-seed mean 1.3458)", (-60, -42)),
    59: ("Stage 59: half the text\nis 0.098 worse", (-95, 22)),
}


def load_rows() -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if r["best_val_loss"]]


def render() -> None:
    rows = load_rows()
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for corpus, color in COLORS.items():
        sub = [r for r in rows if r["corpus"] == corpus]
        xs = [int(r["stage"]) for r in sub]
        ys = [float(r["best_val_loss"]) for r in sub]
        ax.plot(xs, ys, color=color, alpha=0.35, linewidth=1.2, zorder=1)
        kept = [r["kept"] == "yes" for r in sub]
        ax.scatter([x for x, k in zip(xs, kept) if not k], [y for y, k in zip(ys, kept) if not k],
                   s=28, facecolors="white", edgecolors=color, linewidths=1.2, zorder=2)
        ax.scatter([x for x, k in zip(xs, kept) if k], [y for y, k in zip(ys, kept) if k],
                   s=64, color=color, edgecolors="black", linewidths=0.8, zorder=3, label=LABELS[corpus])
    by_stage = {int(r["stage"]): float(r["best_val_loss"]) for r in rows}
    for stage, (text, offset) in MILESTONES.items():
        ax.annotate(text, (stage, by_stage[stage]), xytext=offset, textcoords="offset points",
                    fontsize=8.5, arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8})
    ax.axvline(37, color="#555555", linestyle="--", linewidth=1.2, zorder=0)
    ax.text(37.4, ax.get_ylim()[1] - 0.01, "Stage 37: switch to Complete\nShakespeare (not comparable)",
            va="top", fontsize=8.5, color="#333333")
    ax.scatter([], [], s=28, facecolors="white", edgecolors="#555555", label="change was not kept (hollow)")
    ax.scatter([], [], s=64, color="#555555", edgecolors="black", label="change became part of the champion (filled)")
    ax.set_xlabel("Stage (each stage changes one thing)")
    ax.set_ylabel("Best validation loss (lower is better)")
    ax.set_title("A small GPT learning Shakespeare, one experiment at a time\n"
                 "Validation loss by stage (Stages 1-7 reported no loss; the two corpora cannot be compared)",
                 fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), fontsize=8.5, frameon=False)
    ax.set_xlim(0, 60.5)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)


if __name__ == "__main__":
    render()
