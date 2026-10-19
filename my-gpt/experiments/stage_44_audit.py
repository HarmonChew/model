"""Read-only CPU audit of a completed canonical Stage 44 run.

The runner's model factory and serialization hash primitives are reused, but
protocol constants, optimizer checks, sampler replay, and paired statistics are
checked here. No checkpoint is modified and no model forward pass is run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DATA_FINGERPRINT = "c6cbdd246799ad495f7813e5d20afa52c03e6c339fe7dbe15912fd81a45d2b48"
SCHEDULE = [
    {"start_update": start, "end_update_exclusive": end, "learning_rate": rate}
    for start, end, rate in (
        (0, 10000, 1e-3),
        (10000, 13000, 3e-4),
        (13000, 18000, 1e-4),
        (18000, 50000, 3e-5),
    )
]
# Architecture-native initialization fingerprints for the runtime that
# produced the recorded artifacts (torch 2.14.0+cu130).
INITIALIZATIONS = {
    192: (2752994, "a0ea9075d4b5eab4e4996fb6d3a5aa2ef4bf53c548445f9f4c3dcc05e86950fa"),
    256: (4850274, "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e"),
}
HISTORICAL_HASHES = {
    "stage_41_complete_shakespeare_step_44000_final_checkpoint.pt": "666c9e72345399cf63b412128a1427d1c778d58794473c0e8503516d2235db36",
    "stage_42_control_3e_5_best_checkpoint.pt": "1a66c6bcae77aede45af77432d93cdf8b4c51b72a100fd04b6cdb84cafbe39e6",
    "stage_42_control_3e_5_step_50000_final_checkpoint.pt": "57024bd7bb9e698ba3c16c23533aa5e5d52562eebe87841ac25a992def2b9baf",
    "stage_43_treatment_c192_best_checkpoint.pt": "de624fde73bef1a641d971809041ed6c1779270b6d0e3371a75e9fbafaf2aaa4",
    "stage_43_treatment_c192_step_50000_final_checkpoint.pt": "7506703a1643510812a1bbbb94a6a3327073dc77be8397d6a25c69297a4248f5",
}
# Stage 43's C192 treatment is byte-identical to this stage's fresh C192
# control only if the training pipeline is deterministic across stages.
STAGE_43_C192_FINAL = (
    PROJECT / "checkpoints" / "stage_43_treatment_c192_step_50000_final_checkpoint.pt"
)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def close(actual: float, expected: float, label: str, *, atol: float = 1e-12) -> None:
    require(math.isfinite(actual) and math.isfinite(expected), f"{label}: nonfinite")
    require(
        math.isclose(actual, expected, rel_tol=1e-11, abs_tol=atol),
        f"{label}: actual {actual!r}, expected {expected!r}",
    )


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta, evaluated by Lentz's method."""
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) >= tiny else tiny)
    value = d
    for m in range(1, 10001):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1 + numerator * d
            c = 1 + numerator / c
            d = 1.0 / (d if abs(d) >= tiny else tiny)
            c = c if abs(c) >= tiny else tiny
            delta = d * c
            value *= delta
        if abs(delta - 1) < 2e-14:
            return value
    raise ArithmeticError("incomplete beta did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    scale = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return scale * beta_fraction(a, b, x) / a
    return 1 - scale * beta_fraction(b, a, 1 - x) / b


def student_t_975(df: int) -> float:
    """Invert the Student-t CDF; independent of the runner's asymptotic formula."""
    require(df >= 1, "Student-t requires positive degrees of freedom")

    def cdf(value: float) -> float:
        return 1 - 0.5 * regularized_beta(df / (df + value * value), df / 2, 0.5)

    low, high = 0.0, 2.0
    while cdf(high) < 0.975:
        high *= 2
    for _ in range(70):
        midpoint = (low + high) / 2
        if cdf(midpoint) < 0.975:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def mean_se(values: list[float]) -> tuple[float, float]:
    require(
        len(values) >= 2 and all(math.isfinite(x) for x in values),
        "paired samples must be finite and contain at least two observations",
    )
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def audit_pair(pair: dict[str, Any], *, seed: int | None = None) -> dict[str, Any]:
    require(pair["eval_iters"] == 500, "fresh panel must contain 500 paired batches")
    if seed is not None:
        require(pair["seed"] == seed, "fresh panel seed changed")
    values = []
    for key in ("baseline", "candidate"):
        branch = pair[key]
        require(len(branch["batch_losses"]) == 500, f"{key}: batch count changed")
        mean, se = mean_se(branch["batch_losses"])
        close(branch["mean_loss"], mean, f"{key}: mean")
        close(branch["standard_error"], se, f"{key}: SE")
        values.append(branch["batch_losses"])
    differences = [b - a for a, b in zip(*values, strict=True)]
    mean, se = mean_se(differences)
    critical = student_t_975(499)
    low, high = mean - critical * se, mean + critical * se
    reported = pair["delta"]
    close(reported["mean_delta"], mean, "paired mean")
    close(reported["standard_error"], se, "paired SE")
    # The runner uses a third-order approximation. At df=499 its critical
    # value is within 1e-9 of the independent numerical CDF inversion.
    close(reported["critical_value"], critical, "paired t critical", atol=1e-9)
    close(reported["confidence_low"], low, "paired lower bound", atol=1e-10)
    close(reported["confidence_high"], high, "paired upper bound", atol=1e-10)
    require(reported["candidate"] == pair["candidate"]["name"], "candidate label")
    require(reported["baseline"] == pair["baseline"]["name"], "baseline label")
    return {
        "batches": 500,
        "seed": pair["seed"],
        "control_mean": statistics.fmean(values[0]),
        "treatment_mean": statistics.fmean(values[1]),
        "mean_treatment_minus_control": mean,
        "paired_standard_error": se,
        "independent_student_t_critical_value": critical,
        "independent_student_t_95_interval": [low, high],
        "reported_critical_value_error": reported["critical_value"] - critical,
        "treatment_lower_loss_batches": sum(x < 0 for x in differences),
        "scope": "panel-sampling uncertainty conditional on stored endpoints",
    }


def load_runtime() -> tuple[Any, Any]:
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location(
        "stage_44_for_independent_audit", PROJECT / "44_width_scaling.py"
    )
    require(spec is not None and spec.loader is not None, "runner import unavailable")
    stage = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage
    spec.loader.exec_module(stage)
    return torch, stage


def audit_run(report_path: Path, data_path: Path | None) -> dict[str, Any]:
    require(report_path.is_file(), f"completed report not present: {report_path}")
    report_hash = file_sha256(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(
        report["stage"] == 44 and report["training_executed"] is True,
        "report must describe a completed Stage 44 training run",
    )
    require(
        report["protocol"] == "paired_architecture_native_width_scaling", "protocol"
    )
    require(report["updates_per_branch"] == 50000, "update budget")
    require(report["target_presentations_per_branch"] == 102400000, "token budget")
    require(report["learning_rate_schedule"] == SCHEDULE, "LR schedule")
    require(report["paired_batch_streams_verified"] is True, "paired verification flag")
    torch, stage = load_runtime()
    data = stage.CharacterData.from_file(
        data_path or stage.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
    )
    require(
        stage.fingerprint_data(data) == DATA_FINGERPRINT, "current corpus fingerprint"
    )
    require(report["data_fingerprint"] == DATA_FINGERPRINT, "report corpus fingerprint")
    require(data.vocabulary.size == 98, "vocabulary size")

    # Replay only the sampler's four panel-start draws, without allocating
    # batches or performing training. Save all evaluation-boundary states.
    generator = torch.Generator().manual_seed(1338)
    batch_hashes = {0: stage._STAGE_41.tensor_sha256(generator.get_state())}
    for step in range(1, 50001):
        torch.randint(0, len(data.train_data) - 512, (4,), generator=generator)
        if step % 500 == 0:
            batch_hashes[step] = stage._STAGE_41.tensor_sha256(generator.get_state())
    findings = {}
    checkpoints = {}
    for index, (key, width) in enumerate((("control", 192), ("treatment", 256))):
        branch = report[key]
        require(
            branch["spec"]["key"] == key and branch["spec"]["width"] == width,
            f"{key}: branch specification",
        )
        config = stage.TrainingConfig(
            batch_size=8,
            block_size=256,
            n_embd=width,
            learning_rate=1e-3,
            max_iters=50000,
            eval_interval=500,
            eval_iters=100,
            seed=1337,
        )
        require(
            config == stage.canonical_config(width), f"{key}: canonical config changed"
        )
        # Seed only CPU; no accelerator RNG inspection or execution is needed.
        torch.set_rng_state(torch.Generator().manual_seed(1337).get_state())
        model = stage.build_model(data, config, 4, 6, torch.device("cpu"))
        stage._STAGE_41.verify_model_masks(model)
        count = sum(x.numel() for x in model.parameters())
        native_hash = stage._STAGE_41.tensor_mapping_sha256(model.state_dict())
        require(
            (count, native_hash) == INITIALIZATIONS[width],
            f"{key}: native initialization",
        )
        init = branch["initialization"]
        require(
            init == report["branches"][index]["initialization"],
            f"{key}: preflight init",
        )
        for name, value in {
            "state_sha256": native_hash,
            "parameter_count": count,
            "kind": "fresh_architecture_native",
            "origin_stage": 44,
            "step": 0,
            "seed": 1337,
            "device": "cpu",
            "historical_weights_loaded": False,
            "historical_optimizer_loaded": False,
            "cross_width_tensor_identity_claimed": False,
            "optimizer_state_entries_at_initialization": 0,
        }.items():
            require(init[name] == value, f"{key}: initialization {name}")
        initial = branch["initial_streams"]
        require(
            initial == report["branches"][index]["initial_streams"],
            f"{key}: initial streams",
        )
        require(initial["model_sha256"] == native_hash, f"{key}: initial model stream")
        require(
            initial["training_generator_sha256"] == batch_hashes[0],
            f"{key}: initial sampler",
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
        )
        require(
            initial["optimizer_sha256"] == stage.state_hash(optimizer.state_dict()),
            f"{key}: native empty AdamW state",
        )
        history = branch["training"]["history"]
        require(
            [x["step"] for x in history] == list(range(0, 50001, 500)),
            f"{key}: eval cadence",
        )
        for record in history:
            require(
                all(math.isfinite(x) for x in record["losses"].values()),
                f"{key}: finite history",
            )
        best = min(history, key=lambda x: x["losses"]["val"])
        require(
            branch["training"]["initial"] == history[0]["losses"],
            f"{key}: initial losses",
        )
        require(
            branch["training"]["final"] == history[-1]["losses"], f"{key}: final losses"
        )
        require(
            branch["best_step"] == best["step"]
            and branch["best_losses"] == best["losses"],
            f"{key}: strict minimum selection",
        )
        info = {"parameter_count": count, "native_initialization_sha256": native_hash}
        for kind in ("best", "final"):
            path = Path(branch["spec"][f"{kind}_path"])
            digest = file_sha256(path)
            require(digest == branch[f"{kind}_sha256"], f"{key} {kind}: file SHA256")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            embedded_digest = payload.pop("payload_sha256")
            require(
                stage.state_hash(payload) == embedded_digest,
                f"{key} {kind}: payload integrity",
            )
            payload["payload_sha256"] = embedded_digest
            record = best if kind == "best" else history[-1]
            step = record["step"]
            require(
                payload["step"] == step and payload["checkpoint_kind"] == kind,
                f"{key} {kind}: step and kind",
            )
            require(
                payload["snapshot_losses"]
                == {
                    "train": record["losses"]["train"],
                    "validation": record["losses"]["val"],
                },
                f"{key} {kind}: snapshot losses",
            )
            require(
                payload["best_step"] == best["step"]
                and payload["best_val_loss"] == best["losses"]["val"]
                and payload["best_training_loss"] == best["losses"]["train"],
                f"{key} {kind}: best identity",
            )
            spec = stage.BranchSpec(
                key,
                width,
                Path(branch["spec"]["best_path"]),
                Path(branch["spec"]["final_path"]),
            )
            stage.validate_branch_checkpoint(path, spec, data, config, kind)
            require(
                payload["experiment"]["canonical_protocol"] is True
                and payload["experiment"]["from_scratch"] is True,
                f"{key} {kind}: native protocol",
            )
            require(
                payload["initialization"] == init
                and payload["initial_streams"] == initial,
                f"{key} {kind}: initialization provenance",
            )
            cfg = payload["training_config"]
            for name, value in {
                "training_batch_seed": 1338,
                "training_rng_seed": 1340,
                "fixed_panel_seed": 1339,
                "targets_per_update": 2048,
                "total_target_presentations": step * 2048,
                "target_total_presentations": 102400000,
            }.items():
                require(cfg[name] == value, f"{key} {kind}: {name}")
            require(
                payload["optimizer_restart_step"] == 0
                and payload["optimizer_provenance_known"] is True,
                f"{key} {kind}: optimizer provenance",
            )
            (group,) = payload["optimizer_state_dict"]["param_groups"]
            states = payload["optimizer_state_dict"]["state"]
            require(
                len(group["params"]) == len(list(model.parameters())),
                f"{key} {kind}: parameter slots",
            )
            if step:
                require(
                    set(states) == set(group["params"]),
                    f"{key} {kind}: AdamW slot coverage",
                )
                for parameter_id, parameter in zip(
                    group["params"], model.parameters(), strict=True
                ):
                    state = states[parameter_id]
                    require(
                        state["step"].item() == step,
                        f"{key} {kind}: AdamW update counter",
                    )
                    for moment in ("exp_avg", "exp_avg_sq"):
                        require(
                            state[moment].shape == parameter.shape
                            and bool(torch.isfinite(state[moment]).all()),
                            f"{key} {kind}: {moment}",
                        )
            require(
                payload["model_state_dict"].keys() == model.state_dict().keys(),
                f"{key} {kind}: model tensor key coverage",
            )
            for name, tensor in payload["model_state_dict"].items():
                require(
                    tensor.shape == model.state_dict()[name].shape
                    and bool(torch.isfinite(tensor).all()),
                    f"{key} {kind}: tensor {name}",
                )
            actual_streams = {
                "model_sha256": stage._STAGE_41.tensor_mapping_sha256(
                    payload["model_state_dict"]
                ),
                "optimizer_sha256": stage.state_hash(payload["optimizer_state_dict"]),
                "training_generator_sha256": stage._STAGE_41.tensor_sha256(
                    payload["training_generator_state"]
                ),
                "torch_rng_sha256": stage.state_hash(payload["rng_state"]),
            }
            require(
                actual_streams["training_generator_sha256"] == batch_hashes[step],
                f"{key} {kind}: independently replayed batch stream",
            )
            require(
                payload["historical_lineage"] == report["historical_lineage"],
                f"{key} {kind}: historical metadata",
            )
            if kind == "final":
                require(
                    actual_streams == branch["final_streams"],
                    f"{key}: final stream round trip",
                )
            require(
                file_sha256(path) == digest,
                f"{key} {kind}: artifact changed during audit",
            )
            info[kind] = {
                "path": str(path.resolve()),
                "sha256": digest,
                "step": step,
                "losses": payload["snapshot_losses"],
                "adamw_slots": len(states),
                "adamw_step": step,
                "learning_rate": group["lr"],
            }
            checkpoints[(key, kind)] = info[kind]
            del payload
        findings[key] = info
        del model, optimizer

    for name in ("training_generator_sha256", "torch_rng_sha256"):
        require(
            report["control"]["initial_streams"][name]
            == report["treatment"]["initial_streams"][name],
            f"paired initial {name}",
        )
    require(
        report["control"]["final_streams"]["training_generator_sha256"]
        == report["treatment"]["final_streams"]["training_generator_sha256"],
        "paired terminal sampler",
    )
    pair = report["fresh_50k_branches"]
    paired_stats = audit_pair(pair, seed=1358)
    for role, branch in (("baseline", "control"), ("candidate", "treatment")):
        endpoint = checkpoints[(branch, "final")]
        require(
            pair[role]["checkpoint_sha256"] == endpoint["sha256"]
            and pair[role]["checkpoint_step"] == 50000
            and Path(pair[role]["checkpoint_path"]).resolve() == Path(endpoint["path"]),
            f"fresh {role}: terminal checkpoint identity",
        )
        close(
            pair[role]["fixed_panel_loss"],
            endpoint["losses"]["validation"],
            f"fresh {role}: fixed loss",
        )
    fresh_generator = torch.Generator().manual_seed(1358)
    for _ in range(500):
        torch.randint(0, len(data.val_data) - 512, (4,), generator=fresh_generator)
    require(
        pair["final_generator_sha256"]
        == stage._STAGE_41.tensor_sha256(fresh_generator.get_state()),
        "fresh panel sampler replay",
    )
    close(
        report["parameter_count_ratio"],
        INITIALIZATIONS[256][0] / INITIALIZATIONS[192][0],
        "parameter ratio",
    )
    for kind, field in (
        ("final", "final_fixed_panel_delta_treatment_minus_control"),
        ("best", "selected_best_fixed_panel_delta_treatment_minus_control"),
    ):
        close(
            report[field],
            checkpoints[("treatment", kind)]["losses"]["validation"]
            - checkpoints[("control", kind)]["losses"]["validation"],
            field,
        )
    low, high = paired_stats["independent_student_t_95_interval"]
    expected_outcome = (
        "C256 improves validation at matched 50k updates on the fresh paired panel"
        if high < 0
        else "C192 wins at matched 50k updates under the established schedule"
        if low > 0
        else "unresolved within paired panel uncertainty; this does not prove equivalence"
    )
    require(
        report["outcome"] == expected_outcome, "outcome disagrees with paired interval"
    )
    historical = []
    for filename, expected in HISTORICAL_HASHES.items():
        path = PROJECT / "checkpoints" / filename
        actual = file_sha256(path) if path.is_file() else None
        if actual is not None:
            require(actual == expected, f"historical artifact changed: {filename}")
        historical.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "verified" if actual is not None else "absent; not verified",
            }
        )
    historical_comparison = {"status": "Stage 43 C192 endpoint absent; not compared"}
    if STAGE_43_C192_FINAL.is_file():
        historical_payload = torch.load(
            STAGE_43_C192_FINAL, map_location="cpu", weights_only=True
        )
        historical_streams = {
            "model_sha256": stage._STAGE_41.tensor_mapping_sha256(
                historical_payload["model_state_dict"]
            ),
            "optimizer_sha256": stage.state_hash(
                historical_payload["optimizer_state_dict"]
            ),
            "training_generator_sha256": stage._STAGE_41.tensor_sha256(
                historical_payload["training_generator_state"]
            ),
            "torch_rng_sha256": stage.state_hash(historical_payload["rng_state"]),
        }
        historical_comparison = {
            "status": "compared; diagnostic only",
            "equal_streams": {
                name: value == report["control"]["final_streams"][name]
                for name, value in historical_streams.items()
            },
            "stage_43_streams": historical_streams,
            "stage_44_streams": report["control"]["final_streams"],
            "stage_43_snapshot_losses": historical_payload.get("snapshot_losses"),
            "stage_44_snapshot_losses": checkpoints[("control", "final")]["losses"],
            "interpretation": "The fresh C192 control re-derives the Stage 43 C192 "
            "trajectory; historical agreement is not required for the paired width "
            "experiment.",
        }
        del historical_payload
    status_path = report_path.with_name("stage_44_run_status.json")
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8-sig"))
        require(
            status["script_sha256"] == file_sha256(PROJECT / "44_width_scaling.py"),
            "runner source differs from launch manifest",
        )
    require(file_sha256(report_path) == report_hash, "report changed during audit")
    return {
        "status": "passed",
        "report_path": str(report_path.resolve()),
        "report_sha256": report_hash,
        "branches": findings,
        "fresh_paired_statistics": paired_stats,
        "training_sampler_replayed_updates": 50000,
        "historical_artifacts": historical,
        "historical_c192_endpoint_comparison": historical_comparison,
        "outcome": expected_outcome,
        "limits": "Stored batch losses were recomputed statistically; model forward losses were not rerun. "
        "Shared initial Torch RNG hashes are checked, without asserting cross-width dropout-mask identity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=PROJECT / "checkpoints" / "stage_44_report.json"
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--statistics-only",
        action="store_true",
        help="audit every 500-batch paired panel in the report without importing Torch",
    )
    args = parser.parse_args()
    if args.statistics_only:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        pairs = {
            key: audit_pair(value)
            for key, value in report.items()
            if isinstance(value, dict)
            and {"baseline", "candidate", "delta", "eval_iters"} <= value.keys()
        }
        require(bool(pairs), "no paired panels found")
        result = {"status": "passed", "statistics_only": True, "panels": pairs}
    else:
        result = audit_run(args.report, args.data)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ArithmeticError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        raise SystemExit(1) from error
