"""Render the completed Stage 50 report without loading model checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Keep the font cache in the writable temporary directory on restricted hosts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stage50_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


BUDGET_COLORS = {
    "c256_l6": "#2768a5",
    "c256_l8": "#d46625",
    "c320_l6": "#7d7d7d",
}


def render(report_path: Path, output_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["stage"] != 50 or not report["training_executed"]:
        raise ValueError("requires a completed Stage 50 report")
    fresh = report["fresh_50k_branches"]
    if fresh["eval_iters"] != 500:
        raise ValueError("requires the canonical fresh 500-batch endpoint panel")
    panel = report["budget_shootout"]
    if panel["eval_iters"] != 500 or panel["seed"] != 1369:
        raise ValueError("requires the canonical three-way budget panel")
    keys = panel["panel_keys"]
    colors = {keys["control"]: BUDGET_COLORS["c256_l6"],
              keys["treatment"]: BUDGET_COLORS["c256_l8"],
              keys["budget_peer"]: BUDGET_COLORS["c320_l6"]}
    labels = {
        "control": "C256 L=6 p=0.15 control",
        "treatment": "C256 L=8 p=0.15 treatment",
    }
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
        budget_axis, ladder_axis = axes[1]

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
                color=colors[keys[key]],
                linewidth=1.4,
                alpha=0.55,
                linestyle=":",
            )
            train_axis.plot(
                steps,
                [record["losses"]["val"] for record in records],
                color=colors[keys[key]],
                linewidth=1.9,
                label=labels[key],
            )
            best_step = report[key]["best_step"]
            train_axis.plot(
                [best_step],
                [report[key]["best_losses"]["val"]],
                marker="o",
                markersize=7,
                color=colors[keys[key]],
                markeredgecolor="white",
                markeredgewidth=1.1,
                zorder=5,
            )
            train_axis.annotate(
                f"best {report[key]['best_losses']['val']:.6f}\n@ {best_step:,}",
                (best_step, report[key]["best_losses"]["val"]),
                textcoords="offset points",
                xytext=(-8, -28 if key == "control" else -66),
                ha="right",
                fontsize=8,
                color=colors[keys[key]],
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
            color=colors[keys["treatment"]],
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
        paired_axis.set_xlabel("Validation loss: L=8 minus L=6")
        paired_axis.text(
            0.5,
            0.22,
            f"L=6 {fresh['baseline']['mean_loss']:.6f}\n"
            f"L=8 {fresh['candidate']['mean_loss']:.6f}\n"
            f"Delta {mean:+.6f}\n95% [{low:+.6f}, {high:+.6f}]",
            transform=paired_axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        paired_axis.grid(axis="x", alpha=0.18)

        # Panel 3: the three-way budget panel, ranked, each against the leader so
        # the interval shown is always leader-minus-endpoint.
        ranking = panel["ranking"]
        leader = ranking[0]["key"]
        leader_label = ranking[0]["label"]
        positions = list(range(len(ranking)))
        budget_axis.set_title(
            f"Depth-versus-width budget panel · panel seed {panel['seed']}"
        )
        budget_axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
        label_extent = 0.0
        plot_extent = 0.0
        for position, entry in zip(positions, ranking, strict=True):
            key = entry["key"]
            if key == leader:
                budget_axis.plot(
                    [0], [position], marker="D", markersize=8, color=colors[key]
                )
                budget_axis.text(
                    0.0,
                    position + 0.32,
                    f"leader · {entry['mean_validation_loss']:.6f}",
                    ha="center",
                    fontsize=8,
                    color=colors[key],
                )
                continue
            comparison = panel["comparisons"].get(f"{key}_minus_{leader}")
            if comparison is None:
                comparison = panel["comparisons"][f"{leader}_minus_{key}"]
                point = -comparison["mean_delta"]
                lower, upper = -comparison["confidence_high"], -comparison["confidence_low"]
            else:
                point = comparison["mean_delta"]
                lower, upper = comparison["confidence_low"], comparison["confidence_high"]
            budget_axis.errorbar(
                point,
                position,
                xerr=[[point - lower], [upper - point]],
                fmt="o",
                color=colors[key],
                capsize=5,
                markersize=7,
                linewidth=1.8,
            )
            budget_axis.text(
                upper,
                position,
                f"  {entry['mean_validation_loss']:.6f} "
                f"({entry['parameter_count'] / 1e6:.2f}M)",
                va="center",
                fontsize=8,
                color=colors[key],
            )
            label_extent = max(label_extent, upper)
            plot_extent = min(plot_extent, lower)
        budget_axis.set_yticks(positions)
        budget_axis.set_yticklabels(
            [
                f"C{entry['width']} L={entry['depth']}\n{entry['parameter_count'] / 1e6:.2f}M params"
                for entry in ranking
            ],
            fontsize=8,
        )
        budget_axis.invert_yaxis()
        # Reserve room on the right for the value labels rather than letting the
        # widest one run into the axis.
        span = max(label_extent - plot_extent, 1e-6)
        budget_axis.set_xlim(plot_extent - 0.05 * span, label_extent + 0.35 * span)
        budget_axis.set_xlabel(
            f"Budget-panel validation loss minus {leader_label} (negative is better)"
        )
        budget_axis.grid(axis="x", alpha=0.18)

        # Panel 4: the parameter-budget ladder this stage was run to read.
        ladder = report["depth_budget_ladder"]
        ladder_axis.set_title("Validation loss against parameter budget")
        rows = sorted(ladder["rows"], key=lambda row: row["parameters"])
        for row in rows:
            ladder_axis.errorbar(
                row["parameters"] / 1e6,
                row["mean_validation_loss"],
                yerr=row["standard_error"],
                fmt="o",
                color=colors[row["key"]],
                capsize=4,
                markersize=8,
                linewidth=1.8,
            )
            ladder_axis.annotate(
                f"C{row['width']} L={row['depth']} · {row['parameters'] / 1e6:.2f}M\n"
                f"{row['mean_validation_loss']:.6f}",
                (row["parameters"] / 1e6, row["mean_validation_loss"]),
                textcoords="offset points",
                xytext=(0, 16),
                ha="center",
                fontsize=8,
                color=colors[row["key"]],
            )
        control = next(row for row in rows if row["key"] == ladder["control_key"])
        ladder_axis.axhline(
            control["mean_validation_loss"],
            color=colors[ladder["control_key"]],
            alpha=0.3,
            linewidth=1,
            linestyle=":",
        )
        measured = ladder["measured_depth_step"]
        ladder_axis.set_xlabel("Learned parameters (millions)")
        ladder_axis.set_ylabel("Fresh-panel validation loss (lower is better)")
        ladder_axis.set_xlim(
            min(row["parameters"] for row in rows) / 1e6 - 0.5,
            max(row["parameters"] for row in rows) / 1e6 + 0.7,
        )
        losses = [row["mean_validation_loss"] for row in rows]
        errors = [row["standard_error"] for row in rows]
        low = min(value - error for value, error in zip(losses, errors, strict=True))
        high = max(value + error for value, error in zip(losses, errors, strict=True))
        pad = 0.5 * (high - low)
        ladder_axis.set_ylim(low - 0.35 * pad, high + pad)
        ladder_axis.text(
            0.98,
            0.05,
            f"depth {measured['depth_step']}: {measured['mean_delta']:+.6f}\n"
            f"95% [{measured['confidence_low']:+.6f}, "
            f"{measured['confidence_high']:+.6f}]\n"
            f"+{measured['parameter_delta']:,} params "
            f"({measured['parameter_ratio']:.4f}x control)",
            transform=ladder_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
        ladder_axis.grid(alpha=0.18)

        figure.suptitle(
            "Stage 50 · fresh initialization · C256 L=6 versus L=8 at p=0.15, "
            "then the depth-versus-width budget",
            fontsize=14,
        )
        figure.text(
            0.5,
            0.052,
            "50,000 updates and 102.4 million target presentations per branch, "
            "one shared batch stream.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        figure.text(
            0.5,
            0.022,
            "Intervals are 500-shared-batch panel-sampling uncertainty conditional "
            "on these checkpoints; dropout is inactive in eval mode.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        figure.tight_layout(rect=(0, 0.065, 1, 0.955))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, facecolor="white")
        plt.close(figure)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=project / "checkpoints" / "stage_50_report.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "figures" / "stage_50_depth_revisit.png",
    )
    args = parser.parse_args()
    render(args.report, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
