"""Render the completed Stage 44 report without loading model checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Keep the font cache in the writable temporary directory on restricted hosts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stage44_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def render(report_path: Path, output_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["stage"] != 44 or not report["training_executed"]:
        raise ValueError("requires a completed Stage 44 report")
    fresh = report["fresh_50k_branches"]
    if fresh["eval_iters"] != 500:
        raise ValueError("requires the canonical fresh 500-batch endpoint panel")
    colors = {"control": "#2768a5", "treatment": "#d46625"}
    labels = {"control": "C192 control", "treatment": "C256 treatment"}
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
            for axis, start in zip(axes[:2], (500, 18000), strict=True):
                records = [record for record in history if record["step"] >= start]
                axis.plot(
                    [record["step"] for record in records],
                    [record["losses"]["val"] for record in records],
                    color=colors[key],
                    linewidth=1.8,
                    label=labels[key],
                )
        axes[0].set_title("Validation throughout training")
        axes[1].set_title("Validation after the final LR drop")
        for axis, start in zip(axes[:2], (500, 18000), strict=True):
            axis.set_xlim(start, 50000)
            axis.set_xlabel("Completed updates")
            axis.set_ylabel("Fixed-panel validation loss")
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value / 1000:g}k")
            )
            axis.grid(alpha=0.18)
        for boundary in (10000, 13000, 18000):
            axes[0].axvline(boundary, color="#999999", alpha=0.35, linewidth=0.8)
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
        axes[2].set_xlabel("Validation loss: C256 minus C192")
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
        figure.suptitle("Stage 44 · fresh initialization · C192 vs C256", fontsize=15)
        figure.text(
            0.5,
            0.035,
            "50,000 updates and 102.4 million target presentations per branch. "
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
        "--report", type=Path, default=project / "checkpoints" / "stage_44_report.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "figures" / "stage_44_width_scaling.png",
    )
    args = parser.parse_args()
    render(args.report, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
