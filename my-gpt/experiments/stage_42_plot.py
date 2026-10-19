"""Render the completed Stage 42 report without loading model checkpoints.

The report already carries both branches' full fixed-panel histories and the
fresh paired endpoint interval, so no checkpoint reload is required.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stage42_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def render(report_path: Path, output_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["stage"] != 42 or not report["training_executed"]:
        raise ValueError("requires a completed Stage 42 report")
    fresh = report["fresh_50k_branches"]
    if fresh["eval_iters"] != 500:
        raise ValueError("requires the canonical fresh 500-batch endpoint panel")
    colors = {"control": "#2768a5", "treatment": "#d46625"}
    labels = {"control": "A · continue 3e-5", "treatment": "B · refine at 1e-5"}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axes = plt.subplots(
            1, 3, figsize=(14, 4.7), gridspec_kw={"width_ratios": [1.2, 1.2, 1]}
        )
        for key in ("control", "treatment"):
            history = report[key]["training"]["history"]
            if history[-1]["step"] != 50000:
                raise ValueError("requires both matched 50k endpoints")
            for axis, start in zip(axes[:2], (44000, 48000), strict=True):
                records = [record for record in history if record["step"] >= start]
                axis.plot(
                    [record["step"] for record in records],
                    [record["losses"]["val"] for record in records],
                    color=colors[key],
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                    label=labels[key],
                )
        axes[0].set_title("Validation over the fork interval")
        axes[1].set_title("Validation over the final 2,000 updates")
        for axis, start in zip(axes[:2], (44000, 48000), strict=True):
            axis.set_xlim(start, 50000)
            axis.set_xlabel("Completed updates")
            axis.set_ylabel("Fixed-panel validation loss")
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value / 1000:g}k")
            )
            axis.grid(alpha=0.18)
        axes[0].legend(frameon=False)
        delta = fresh["delta"]
        mean, low, high = (
            delta[key] for key in ("mean_delta", "confidence_low", "confidence_high")
        )
        axes[2].axvline(0, color="#777777", linewidth=1, linestyle="--")
        axes[2].errorbar(
            mean,
            0,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=colors["treatment"],
            capsize=6,
            markersize=7,
            linewidth=2,
        )
        span = max(abs(low), abs(high), 1e-6)
        axes[2].set_xlim(min(low, 0) - 0.2 * span, max(high, 0) + 0.2 * span)
        axes[2].set_ylim(-0.8, 0.8)
        axes[2].set_yticks([])
        axes[2].set_title("Fresh paired 50k endpoint")
        axes[2].set_xlabel("Validation loss: 1e-5 minus 3e-5")
        axes[2].text(
            0.5,
            0.2,
            f"Delta {mean:+.6f}\n95% interval [{low:+.6f}, {high:+.6f}]",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        axes[2].grid(axis="x", alpha=0.18)
        figure.suptitle("Stage 42 · exact 44k fork · 3e-5 vs 1e-5", fontsize=15)
        figure.text(
            0.5,
            0.035,
            "6,000 matched updates and 12.288 million target presentations per branch. "
            "Fresh interval: 500 shared batches, conditional on these checkpoints.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.94))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, facecolor="white")
        plt.close(figure)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=project / "checkpoints" / "stage_42_report.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "figures" / "stage_42_learning_rate_fork.png",
    )
    args = parser.parse_args()
    render(args.report, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
