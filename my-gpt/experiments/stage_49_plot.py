"""Render the completed Stage 49 report without loading model checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Keep the font cache in the writable temporary directory on restricted hosts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stage49_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


CHAMPION_COLORS = {
    "c256_p015": "#5aa2d1",
    "c256_p020": "#2768a5",
    "c320_p015": "#d46625",
    "c320_p020": "#8c3f13",
}


def render(report_path: Path, output_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["stage"] != 49 or not report["training_executed"]:
        raise ValueError("requires a completed Stage 49 report")
    fresh = report["fresh_50k_branches"]
    if fresh["eval_iters"] != 500:
        raise ValueError("requires the canonical fresh 500-batch endpoint panel")
    champion = report["champion_shootout"]
    if champion["eval_iters"] != 500 or champion["seed"] != 1367:
        raise ValueError("requires the canonical four-way champion panel")
    colors = {"control": "#2768a5", "treatment": "#d46625"}
    labels = {"control": "C256 p=0.15 control", "treatment": "C256 p=0.20 treatment"}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
        train_axis, paired_axis = axes[0]
        champion_axis, ladder_axis = axes[1]

        # Panel 1: both branches' fixed-panel trajectories, with their selected
        # best checkpoints marked at the step the selection actually used.
        train_axis.set_title("Fixed-panel validation throughout training")
        for key in ("control", "treatment"):
            history = report[key]["training"]["history"]
            if history[-1]["step"] != 50000:
                raise ValueError("requires both matched 50k endpoints")
            records = [record for record in history if record["step"] >= 500]
            steps = [record["step"] for record in records]
            train_axis.plot(
                steps,
                [record["losses"]["train"] for record in records],
                color=colors[key],
                linewidth=1.4,
                alpha=0.55,
                linestyle=":",
            )
            train_axis.plot(
                steps,
                [record["losses"]["val"] for record in records],
                color=colors[key],
                linewidth=1.9,
                label=labels[key],
            )
            best_step = report[key]["best_step"]
            train_axis.plot(
                [best_step],
                [report[key]["best_losses"]["val"]],
                marker="o",
                markersize=7,
                color=colors[key],
                markeredgecolor="white",
                markeredgewidth=1.1,
                zorder=5,
            )
            train_axis.annotate(
                f"best {report[key]['best_losses']['val']:.6f}\n@ {best_step:,}",
                (best_step, report[key]["best_losses"]["val"]),
                textcoords="offset points",
                xytext=(6, -30 if key == "control" else -52),
                fontsize=8,
                color=colors[key],
            )
        for boundary in (10000, 13000, 18000):
            train_axis.axvline(boundary, color="#999999", alpha=0.35, linewidth=0.8)
        train_axis.set_xlim(500, 50000)
        train_axis.set_xlabel("Completed updates")
        train_axis.set_ylabel("Fixed-panel loss (dotted: train)")
        train_axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1000:g}k")
        )
        train_axis.grid(alpha=0.18)
        train_axis.legend(frameon=False, loc="upper right")

        # Panel 2: the fresh paired endpoint comparison on its own panel.
        delta = fresh["delta"]
        mean, low, high = (
            delta[key] for key in ("mean_delta", "confidence_low", "confidence_high")
        )
        paired_axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
        paired_axis.errorbar(
            mean,
            0,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=colors["treatment"],
            capsize=6,
            markersize=7,
            linewidth=2,
        )
        spread = max(abs(low), abs(high), 1e-6)
        paired_axis.set_xlim(min(low, 0) - 0.2 * spread, max(high, 0) + 0.2 * spread)
        paired_axis.set_ylim(-0.8, 0.8)
        paired_axis.set_yticks([])
        # A delta panel spans thousandths, so default ticks collide; keep four.
        paired_axis.locator_params(axis="x", nbins=4)
        paired_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.4f}"))
        paired_axis.set_title(
            f"Fresh paired 50k endpoint · panel seed {fresh['seed']}"
        )
        paired_axis.set_xlabel("Validation loss: p=0.20 minus p=0.15")
        paired_axis.text(
            0.5,
            0.22,
            f"p=0.15 {fresh['baseline']['mean_loss']:.6f}\n"
            f"p=0.20 {fresh['candidate']['mean_loss']:.6f}\n"
            f"Delta {mean:+.6f}\n95% [{low:+.6f}, {high:+.6f}]",
            transform=paired_axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        paired_axis.grid(axis="x", alpha=0.18)

        # Panel 3: the four-way champion shootout, ranked, each against the
        # leader so the interval shown is always leader-minus-endpoint.
        ranking = champion["ranking"]
        leader = ranking[0]["key"]
        leader_mean = ranking[0]["mean_validation_loss"]
        positions = list(range(len(ranking)))
        champion_axis.set_title(
            f"Four-way champion shootout · panel seed {champion['seed']}"
        )
        champion_axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
        for position, entry in zip(positions, ranking, strict=True):
            key = entry["key"]
            if key == leader:
                champion_axis.plot(
                    [0], [position], marker="D", markersize=8, color=CHAMPION_COLORS[key]
                )
                champion_axis.text(
                    0.0,
                    position + 0.32,
                    f"leader · {entry['mean_validation_loss']:.6f}",
                    ha="center",
                    fontsize=8,
                    color=CHAMPION_COLORS[key],
                )
                continue
            comparison = champion["comparisons"].get(f"{key}_minus_{leader}")
            if comparison is None:
                comparison = champion["comparisons"][f"{leader}_minus_{key}"]
                point = -comparison["mean_delta"]
                lower, upper = -comparison["confidence_high"], -comparison["confidence_low"]
            else:
                point = comparison["mean_delta"]
                lower, upper = comparison["confidence_low"], comparison["confidence_high"]
            champion_axis.errorbar(
                point,
                position,
                xerr=[[point - lower], [upper - point]],
                fmt="o",
                color=CHAMPION_COLORS[key],
                capsize=5,
                markersize=7,
                linewidth=1.8,
            )
            champion_axis.text(
                upper,
                position,
                f"  {entry['mean_validation_loss']:.6f} (rank {entry['rank']})",
                va="center",
                fontsize=8,
                color=CHAMPION_COLORS[key],
            )
        champion_axis.set_yticks(positions)
        champion_axis.set_yticklabels(
            [
                f"C{entry['label'].split()[0].lstrip('C')} {entry['label'].split()[1]}\n"
                f"({entry['provenance'].split(',')[0]})"
                for entry in ranking
            ],
            fontsize=8,
        )
        champion_axis.invert_yaxis()
        champion_axis.set_xlabel(
            f"Champion-panel validation loss minus {leader} (negative is better)"
        )
        champion_axis.grid(axis="x", alpha=0.18)

        # Panel 4: the capacity-versus-regularization ladder.
        ladder = report["capacity_regularization_ladder"]
        ladder_axis.set_title("Capacity-versus-regularization dropout curve")
        ladder_axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
        ladder_positions = list(range(len(ladder)))
        ladder_colors = ["#2768a5", "#5aa2d1", "#d46625", "#8c3f13"]
        for position, row in zip(ladder_positions, ladder, strict=True):
            point, lower, high_value = (
                row["mean_delta"],
                row["confidence_low"],
                row["confidence_high"],
            )
            ladder_axis.errorbar(
                point,
                position,
                xerr=[[point - lower], [high_value - point]],
                fmt="o",
                color=ladder_colors[position],
                capsize=5,
                markersize=7,
                linewidth=1.8,
            )
            ladder_axis.text(
                high_value,
                position,
                f"  {point:+.6f}",
                va="center",
                fontsize=8,
                color=ladder_colors[position],
            )
        ladder_axis.set_yticks(
            ladder_positions,
            [
                f"C{row['width']} {row['dropout_step']}\n(stage {row['source_stage']})"
                for row in ladder
            ],
            fontsize=8,
        )
        ladder_axis.invert_yaxis()
        ladder_axis.set_xlabel(
            "Fresh paired validation delta for the dropout step (negative is better)"
        )
        ladder_axis.grid(axis="x", alpha=0.18)

        figure.suptitle(
            "Stage 49 · fresh initialization · C256 p=0.15 vs p=0.20, "
            "then the C256-versus-C320 champion shootout",
            fontsize=14,
        )
        figure.text(
            0.5,
            0.015,
            "50,000 updates and 102.4 million target presentations per branch. "
            "Intervals are 500-shared-batch panel-sampling uncertainty conditional "
            "on these checkpoints; dropout is inactive in eval mode.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.955))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, facecolor="white")
        plt.close(figure)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=project / "checkpoints" / "stage_49_report.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "figures" / "stage_49_dropout_strength.png",
    )
    args = parser.parse_args()
    render(args.report, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
