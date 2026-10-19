"""Read-only CPU audit of a completed canonical Stage 49 run.

The runner's model factory, checkpoint validators, and serialization hash
primitives are reused, but protocol constants, the sampler, paired statistics,
the capacity ladder, and every recorded hash are re-derived here. No checkpoint
is modified, no accelerator is touched, and no model forward pass is run.
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
# Architecture-native initialization fingerprints for the runtime that produced
# the recorded artifacts (torch 2.14.0+cu130). Residual dropout neither adds
# parameters nor consumes initialization RNG, so one fingerprint per width
# covers every branch at that width.
C256_NATIVE_INITIALIZATION = (
    4850274,
    "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
)
C320_NATIVE_INITIALIZATION = (
    7537378,
    "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
)
# One sampler, one seed, one batch shape: every branch in the four-way champion
# panel consumed the same training stream, so each endpoint's terminal
# batch-generator state must equal this replay.
SHARED_TRAINING_GENERATOR_SHA256 = (
    "986ffc5650abe72ba270a01cc94ab5a247a946e1f4f040d1eef1508f8a7c86b9"
)
SHARED_INITIAL_GENERATOR_SHA256 = (
    "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced"
)
# The global Torch RNG advances by a fixed number of draws per update in this
# architecture, so every branch of the lineage ends on the same training-device
# RNG state. The full capture is compared only through its training-device
# projection: it also stores every other visible accelerator, and the host's GPU
# set changed between these runs.
SHARED_TERMINAL_TRAINING_RNG_SHA256 = (
    "0b8d137f0468f1a12db1c880b3d7c74f8dc0039927742b7d71cc9e107546ee97"
)
LINEAGE_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
TRAINING_BATCH_SEED = 1338
TRAINING_RNG_SEED = 1340
INITIALIZATION_SEED = 1337
STAGE_45_CHAMPION_SHA256 = (
    "e58ce4532a3ce0623ea622b91ea7ee04232c8d1f12d8b2c53591c7efec299e9f"
)
# Stage 48's fresh C256 p=0.15 treatment is the read-only reference this
# stage's p=0.15 control must reproduce exactly.
STAGE_48_TREATMENT_SHA256 = (
    "04952aeeab517f2a31ba4209737d4f8d495d40fb84aca8c9b5f3b58bb44f98b4"
)
# The prior rows of the capacity-versus-regularization ladder, as measured by
# the stages that ran them. Stage 49 transcribes these rather than retraining
# them.
PRIOR_DROPOUT_STEPS = (
    (48, 256, -0.0015666978359222413, 1364),
    (46, 320, -0.0050533, 1360),
    (47, 320, 0.0000117, 1362),
)
CHAMPION_KEYS = ("c256_p015", "c256_p020", "c320_p015", "c320_p020")
# Every artifact of this lineage, including the stored C320 endpoints this
# stage reads and the Stage 48 endpoints whose treatment this stage's control
# must reproduce.
HISTORICAL_HASHES = {
    "stage_41_complete_shakespeare_step_44000_final_checkpoint.pt": "666c9e72345399cf63b412128a1427d1c778d58794473c0e8503516d2235db36",
    "stage_42_control_3e_5_step_50000_final_checkpoint.pt": "57024bd7bb9e698ba3c16c23533aa5e5d52562eebe87841ac25a992def2b9baf",
    "stage_44_treatment_c256_step_50000_final_checkpoint.pt": "c9e0092cb1986885f8cef948b703bb4fbf18ae0714e4e3d062fecc337081b247",
    "stage_45_control_c256_step_50000_final_checkpoint.pt": STAGE_45_CHAMPION_SHA256,
    "stage_45_control_c256_best_checkpoint.pt": "1c3668c88a82942fed3ce25b641e213f566ee4b500a3d0766584e158b9f51d65",
    "stage_45_treatment_c320_step_50000_final_checkpoint.pt": "a6f908c18d02ea3d7a9060d4fb7886320bff41eea06a0e97e2196ea57f34984b",
    "stage_46_treatment_c320_p015_step_50000_final_checkpoint.pt": "e47016371b6a588448dbaa0bad6d05ff8564b7b21c6f6da81f20bbb56e916197",
    "stage_47_treatment_c320_p020_step_50000_final_checkpoint.pt": "7289527c6078b92f52abc98f39c3871c30aed7fa56a2d8a2901d5f4f035b170f",
    "stage_48_control_c256_p010_step_50000_final_checkpoint.pt": "8c817fd5ea8a557e60f693a577cad68bbd9d483402d8e90fe343b23590c8090e",
    "stage_48_treatment_c256_p015_step_50000_final_checkpoint.pt": STAGE_48_TREATMENT_SHA256,
    "stage_48_treatment_c256_p015_best_checkpoint.pt": "62d6b5ee7d9009cafd48561e5622c9a1e7b3254e3d26fbfc57c280662ce723ab",
}


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


def fresh_endpoints(fresh: dict[str, Any]) -> dict[str, Any]:
    """Wrap a paired panel's two endpoints under the names its delta refers to."""

    return {"baseline": fresh["baseline"], "candidate": fresh["candidate"]}


def audit_endpoint(endpoint: dict[str, Any], label: str) -> dict[str, Any]:
    """Re-derive one endpoint's mean and standard error from its stored losses."""

    require(
        endpoint["checkpoint_step"] == 50000,
        f"{label}: endpoint must be the terminal step",
    )
    losses = endpoint["batch_losses"]
    require(len(losses) == 500, f"{label}: batch count changed")
    require(all(math.isfinite(value) for value in losses), f"{label}: nonfinite loss")
    mean, se = mean_se(losses)
    close(endpoint["mean_loss"], mean, f"{label}: mean")
    close(endpoint["standard_error"], se, f"{label}: SE")
    require(
        math.isfinite(endpoint["fixed_panel_loss"]),
        f"{label}: fixed-panel loss is not finite",
    )
    path = Path(endpoint["checkpoint_path"])
    require(path.is_file(), f"{label}: checkpoint missing")
    require(
        file_sha256(path) == endpoint["checkpoint_sha256"],
        f"{label}: checkpoint hash changed",
    )
    return {
        "mean_loss": mean,
        "standard_error": se,
        "fixed_panel_loss": endpoint["fixed_panel_loss"],
        "checkpoint_sha256": endpoint["checkpoint_sha256"],
    }


def audit_delta(
    reported: dict[str, Any],
    endpoints: dict[str, Any],
    label: str,
    critical: float,
) -> dict[str, Any]:
    """Re-derive one pairwise delta from the two endpoints it compares.

    `reported` is the stored delta record itself, so both report shapes work: a
    paired panel stores its delta under `delta`, while the champion panel stores
    one delta per unordered pair under `comparisons`.
    """

    names = {entry["name"]: entry for entry in endpoints.values()}
    if reported["candidate"] not in names or reported["baseline"] not in names:
        raise ValueError(f"{label}: delta names do not match the panel endpoints")
    candidate = names[reported["candidate"]]
    baseline = names[reported["baseline"]]
    if len(candidate["batch_losses"]) != len(baseline["batch_losses"]):
        raise ValueError(f"{label}: paired endpoints have different batch counts")
    differences = [
        c - b
        for b, c in zip(
            baseline["batch_losses"], candidate["batch_losses"], strict=True
        )
    ]
    mean, se = mean_se(differences)
    low, high = mean - critical * se, mean + critical * se
    close(reported["mean_delta"], mean, f"{label}: mean", atol=1e-9)
    close(reported["standard_error"], se, f"{label}: SE", atol=1e-9)
    close(reported["critical_value"], critical, f"{label}: t critical", atol=1e-9)
    close(reported["confidence_low"], low, f"{label}: lower", atol=1e-10)
    close(reported["confidence_high"], high, f"{label}: upper", atol=1e-10)
    # The interval must also be self-consistent with its own recorded summary.
    margin = reported["critical_value"] * reported["standard_error"]
    close(reported["confidence_low"], reported["mean_delta"] - margin, f"{label}: refit lower")
    close(reported["confidence_high"], reported["mean_delta"] + margin, f"{label}: refit upper")
    require(
        reported["confidence_low"] <= reported["mean_delta"] <= reported["confidence_high"],
        f"{label}: delta outside its own interval",
    )
    return {
        "candidate": reported["candidate"],
        "baseline": reported["baseline"],
        "mean_delta": mean,
        "standard_error": se,
        "interval": [low, high],
        "candidate_lower_loss_batches": sum(value < 0 for value in differences),
        "sign": (
            "candidate_better"
            if high < 0
            else "baseline_better"
            if low > 0
            else "unresolved"
        ),
    }


def audit_champion_panel(
    panel: dict[str, Any], expected_seed: int
) -> dict[str, Any]:
    """Re-derive every champion-shootout statistic and its ranking."""

    require(panel["seed"] == expected_seed, "champion panel seed changed")
    require(panel["eval_iters"] == 500, "champion panel must hold 500 batches")
    require(panel["shared_panel_per_batch"] is True, "champion panel is not shared")
    require(panel["dropout_enabled"] is False, "champion panel enabled dropout")
    require(
        panel["shared_training_batch_stream_verified"] is True,
        "champion panel did not verify its shared training stream",
    )
    require(
        panel["shared_training_generator_sha256"] == SHARED_TRAINING_GENERATOR_SHA256,
        "champion panel recorded a different shared training stream",
    )
    projections = panel.get("rng_projections")
    if projections is not None:
        require(
            {value["training_device_name"] for value in projections.values()}
            == {LINEAGE_DEVICE_NAME},
            "champion panel was not run entirely on the recorded lineage device",
        )
        require(
            {value["terminal_training_rng_sha256"] for value in projections.values()}
            == {SHARED_TERMINAL_TRAINING_RNG_SHA256},
            "champion panel endpoints disagree on the training-device RNG state",
        )
    require(
        sorted(panel["endpoints"]) == sorted(CHAMPION_KEYS),
        "champion endpoint set changed",
    )
    endpoints = {}
    for key, endpoint in panel["endpoints"].items():
        endpoints[key] = audit_endpoint(endpoint, f"endpoint {key}")
    critical = student_t_975(499)
    pairs = sorted(
        f"{candidate}_minus_{baseline}"
        for index, candidate in enumerate(CHAMPION_KEYS)
        for baseline in CHAMPION_KEYS[index + 1 :]
    )
    forward = sorted(panel["comparisons"])
    require(
        set(forward) == set(pairs),
        "champion comparisons must cover every unordered pair exactly once",
    )
    comparisons = {
        key: audit_delta(
            panel["comparisons"][key], panel["endpoints"], key, critical
        )
        for key in forward
    }
    # Every comparison that is not stored in the requested direction must be
    # exactly the negation of its mirror image.
    for candidate in CHAMPION_KEYS:
        for baseline in CHAMPION_KEYS:
            if candidate >= baseline:
                continue
            mirror = panel["comparisons"].get(f"{baseline}_minus_{candidate}")
            if mirror is None:
                continue
            direct = panel["comparisons"][f"{candidate}_minus_{baseline}"]
            close(mirror["mean_delta"], -direct["mean_delta"], "mirror mean")
            close(mirror["confidence_low"], -direct["confidence_high"], "mirror lower")
            close(mirror["confidence_high"], -direct["confidence_low"], "mirror upper")
    ranked = sorted(
        CHAMPION_KEYS, key=lambda key: panel["endpoints"][key]["mean_loss"]
    )
    require(
        [entry["key"] for entry in panel["ranking"]] == ranked,
        "champion ranking is not ordered by mean validation loss",
    )
    for position, entry in enumerate(panel["ranking"], start=1):
        require(entry["rank"] == position, "champion rank numbering changed")
        close(
            entry["mean_validation_loss"],
            panel["endpoints"][entry["key"]]["mean_loss"],
            f"ranking {entry['key']}: mean",
        )
        require(
            entry["checkpoint_sha256"]
            == panel["endpoints"][entry["key"]]["checkpoint_sha256"],
            f"ranking {entry['key']}: checkpoint hash",
        )
    champion = ranked[0]
    defeated = []
    for key in ranked[1:]:
        delta = panel["comparisons"].get(f"{key}_minus_{champion}") or panel[
            "comparisons"
        ][f"{champion}_minus_{key}"]
        if f"{key}_minus_{champion}" in panel["comparisons"]:
            resolved = delta["confidence_low"] > 0
        else:
            resolved = delta["confidence_high"] < 0
        if resolved:
            defeated.append(key)
    outcome = panel.get("champion_outcome")
    if outcome is not None:
        if len(defeated) == len(ranked) - 1:
            require("resolved champion" in outcome, "outcome ignores a resolved panel")
        else:
            require(
                "does not resolve" in outcome and "leads the shared fresh panel" in outcome,
                "outcome ignores an unresolved peer",
            )
    return {
        "panel_seed": panel["seed"],
        "student_t_critical_value": critical,
        "endpoints": endpoints,
        "comparisons": comparisons,
        "ranking": ranked,
        "resolved_against": defeated,
        "unresolved_against": [key for key in ranked[1:] if key not in defeated],
        "scope": "panel-sampling uncertainty conditional on stored endpoints",
    }


def load_runtime() -> tuple[Any, Any]:
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location(
        "stage_49_for_independent_audit", PROJECT / "49_dropout_strength.py"
    )
    require(spec is not None and spec.loader is not None, "runner import unavailable")
    stage = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage
    spec.loader.exec_module(stage)
    return torch, stage


def audit_sampler(torch: Any, stage: Any, data: Any) -> str:
    """Replay the 50,000-update batch stream from its seed and batch shape.

    The sampler draws batch_size//2 panel starts per update, so the terminal
    generator state depends only on the seed and the corpus. Reproducing the
    recorded stream here is what justifies treating the four champion endpoints
    as trained on one batch stream.
    """

    generator = torch.Generator().manual_seed(TRAINING_BATCH_SEED)
    initial = stage._STAGE_41.tensor_sha256(generator.get_state())
    require(
        initial == SHARED_INITIAL_GENERATOR_SHA256,
        "replayed batch stream does not start at the recorded initial state",
    )
    high = len(data.train_data) - 512
    for _ in range(50000):
        torch.randint(0, high, (4,), generator=generator)
    terminal = stage._STAGE_41.tensor_sha256(generator.get_state())
    require(
        terminal == SHARED_TRAINING_GENERATOR_SHA256,
        "replayed batch stream does not reach the recorded terminal state",
    )
    return terminal


def audit_endpoint_checkpoints(torch: Any, stage: Any, data: Any) -> dict[str, Any]:
    """Re-authenticate all four champion checkpoints through the runner itself."""

    directory = stage.DEFAULT_OUTPUT_DIRECTORY
    config = stage.canonical_config()
    endpoints = {}
    for spec in stage.branch_specs(directory):
        path = spec.final_path
        require(path.is_file(), f"fresh endpoint missing: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        recorded = payload.pop("payload_sha256", None)
        require(
            recorded == stage.state_hash(payload),
            f"{path.name}: payload integrity mismatch",
        )
        payload["payload_sha256"] = recorded
        stage.validate_branch_checkpoint(path, spec, data, config, "final")
        key = f"c{spec.width}_{stage.dropout_label(spec.dropout)}"
        endpoints[key] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256(path),
            "payload_sha256": recorded,
            "initialization_sha256": payload["initialization"]["state_sha256"],
            "parameter_count": payload["initialization"]["parameter_count"],
            "terminal_generator_sha256": stage._STAGE_41.tensor_sha256(
                payload["training_generator_state"]
            ),
            "initial_streams": dict(payload["initial_streams"]),
            "origin_stage": payload["initialization"]["origin_stage"],
            "best_step": payload["best_step"],
            "rng_projection": stage.terminal_rng_normalization(payload),
            "recorded_training_device_index": payload["runtime"].get("device_index"),
            "cuda_entries_at_capture": len(payload["rng_state"].get("cuda", ())),
        }
    for ref in stage.C320_CHAMPION_REFS:
        endpoint = stage.load_c320_champion(ref, data, directory)
        payload = endpoint.payload
        recorded = payload["payload_sha256"]
        stripped = {key: value for key, value in payload.items() if key != "payload_sha256"}
        require(
            recorded == stage.state_hash(stripped),
            f"{endpoint.path.name}: payload integrity mismatch",
        )
        require(
            file_sha256(endpoint.path) == ref["file_sha256"],
            f"{endpoint.path.name}: file hash changed",
        )
        endpoints[ref["key"]] = {
            "path": str(endpoint.path.resolve()),
            "file_sha256": file_sha256(endpoint.path),
            "payload_sha256": recorded,
            "initialization_sha256": payload["initialization"]["state_sha256"],
            "parameter_count": payload["initialization"]["parameter_count"],
            "terminal_generator_sha256": stage._STAGE_41.tensor_sha256(
                payload["training_generator_state"]
            ),
            "initial_streams": dict(payload["initial_streams"]),
            "origin_stage": payload["initialization"]["origin_stage"],
            "best_step": payload["best_step"],
            "rng_projection": stage.terminal_rng_normalization(payload),
            "recorded_training_device_index": payload["runtime"].get("device_index"),
            "cuda_entries_at_capture": len(payload["rng_state"].get("cuda", ())),
        }
    require(
        sorted(endpoints) == sorted(CHAMPION_KEYS),
        "re-authenticated endpoint set changed",
    )
    # One batch stream, two initializations: the width groups must differ in
    # initial weights and agree on everything the sampler controls.
    require(
        len({entry["terminal_generator_sha256"] for entry in endpoints.values()}) == 1,
        "champion endpoints did not share one terminal batch stream",
    )
    require(
        {entry["terminal_generator_sha256"] for entry in endpoints.values()}
        == {SHARED_TRAINING_GENERATOR_SHA256},
        "champion endpoints did not end on the recorded shared batch stream",
    )
    require(
        len(
            {
                entry["initial_streams"]["training_generator_sha256"]
                for entry in endpoints.values()
            }
        )
        == 1,
        "champion endpoints started from different batch streams",
    )
    require(
        {
            entry["initial_streams"]["training_generator_sha256"]
            for entry in endpoints.values()
        }
        == {SHARED_INITIAL_GENERATOR_SHA256},
        "champion endpoints did not start at the recorded batch stream",
    )
    # Training-device RNG identity is compared through its projection, because
    # the full capture also stores every other accelerator the host exposed.
    projections = {
        key: value["rng_projection"] for key, value in endpoints.items()
    }
    require(
        {value["training_device_name"] for value in projections.values()}
        == {LINEAGE_DEVICE_NAME},
        "champion endpoints were not all trained on the recorded device",
    )
    require(
        {value["terminal_training_rng_sha256"] for value in projections.values()}
        == {SHARED_TERMINAL_TRAINING_RNG_SHA256},
        "champion endpoints do not share the recorded training-device RNG state",
    )
    for key, entry in endpoints.items():
        if entry["origin_stage"] == 49:
            require(
                entry["recorded_training_device_index"]
                == entry["rng_projection"]["training_device_index"],
                f"{key}: recorded training device index disagrees with its projection",
            )
            require(
                entry["cuda_entries_at_capture"]
                == entry["rng_projection"]["cuda_entry_count"],
                f"{key}: recorded accelerator count disagrees with its capture",
            )
    for key, expected in (
        ("c256_p015", C256_NATIVE_INITIALIZATION),
        ("c256_p020", C256_NATIVE_INITIALIZATION),
        ("c320_p015", C320_NATIVE_INITIALIZATION),
        ("c320_p020", C320_NATIVE_INITIALIZATION),
    ):
        entry = endpoints[key]
        require(
            (entry["parameter_count"], entry["initialization_sha256"]) == expected,
            f"{key}: native initialization fingerprint changed",
        )
    for path, digest in historical_paths():
        require(path.is_file(), f"historical artifact missing: {path.name}")
        require(file_sha256(path) == digest, f"historical artifact changed: {path.name}")
    return endpoints


def historical_paths() -> list[tuple[Path, str]]:
    directory = PROJECT / "checkpoints"
    return [
        (directory / name, digest) for name, digest in HISTORICAL_HASHES.items()
    ]


def audit_manifest(report_path: Path, report_hash: str) -> dict[str, Any]:
    """Check the launch manifest when present; never require it."""

    manifest_path = report_path.with_name("stage_49_run_status.json")
    if not manifest_path.is_file():
        return {"status": "skipped", "reason": "launch manifest not present"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(manifest["stage"] == 49, "manifest stage changed")
    require(
        manifest["script_sha256"] == file_sha256(PROJECT / "49_dropout_strength.py"),
        "runner source differs from launch manifest",
    )
    require(file_sha256(report_path) == report_hash, "report changed during audit")
    manifest["report_sha256"] = report_hash
    log_path = PROJECT / manifest["log"]
    require(log_path.is_file(), "launch log missing")
    require(
        file_sha256(log_path) == manifest["log_sha256"], "launch log changed"
    )
    champion_path = PROJECT / manifest["champion_report"]
    if champion_path.is_file():
        require(
            file_sha256(champion_path) == manifest["champion_report_sha256"],
            "champion report changed",
        )
    return {"status": "passed", "manifest": str(manifest_path.resolve())}


def audit_ladder(report: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Check the capacity-versus-regularization ladder against its sources."""

    ladder = report["capacity_regularization_ladder"]
    require(len(ladder) == 4, "ladder must carry one row per measured dropout step")
    measured = ladder[0]
    require(measured["source_stage"] == 49 and measured["width"] == 256, "ladder row 1")
    close(measured["mean_delta"], fresh["delta"]["mean_delta"], "ladder measured delta")
    require(
        measured["fresh_panel_seed"] == fresh["seed"], "ladder measured panel seed"
    )
    for index, (stage, width, delta, seed) in enumerate(PRIOR_DROPOUT_STEPS, start=1):
        row = ladder[index]
        require(
            (row["source_stage"], row["width"]) == (stage, width),
            f"ladder row {index + 1} provenance",
        )
        close(row["mean_delta"], delta, f"ladder row {index + 1} delta")
        require(row["fresh_panel_seed"] == seed, f"ladder row {index + 1} seed")
        require(
            row["confidence_low"] <= row["mean_delta"] <= row["confidence_high"],
            f"ladder row {index + 1} delta outside its interval",
        )
    return {
        "rows": len(ladder),
        "measured_width": measured["width"],
        "measured_delta": measured["mean_delta"],
        "prior_steps": [row["dropout_step"] for row in ladder[1:]],
    }


def audit_run(report_path: Path, data_path: Path | None) -> dict[str, Any]:
    require(report_path.is_file(), f"completed report not present: {report_path}")
    report_hash = file_sha256(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(
        report["stage"] == 49 and report["training_executed"] is True,
        "report must describe a completed Stage 49 training run",
    )
    require(
        report["protocol"] == "paired_architecture_native_regularization", "protocol"
    )
    require(report["updates_per_branch"] == 50000, "update budget")
    require(report["target_presentations_per_branch"] == 102400000, "token budget")
    require(report["learning_rate_schedule"] == SCHEDULE, "LR schedule")
    require(report["paired_batch_streams_verified"] is True, "paired verification flag")
    require(report["parameter_count"] == C256_NATIVE_INITIALIZATION[0], "parameter count")
    require(
        report["runtime"]["lineage_device_required"] == "NVIDIA GeForce RTX 3090",
        "lineage device requirement",
    )
    require(
        report["control_initialization_equivalence"]["identical"] is True
        and report["control_initialization_equivalence"]["state_sha256"]
        == C256_NATIVE_INITIALIZATION[1]
        and report["control_initialization_equivalence"]["stage_45_factory_identical"]
        is True,
        "control initialization equivalence",
    )
    torch, stage = load_runtime()
    data = stage.CharacterData.from_file(
        data_path or stage.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
    )
    require(
        stage.fingerprint_data(data) == DATA_FINGERPRINT, "current corpus fingerprint"
    )
    require(report["data_fingerprint"] == DATA_FINGERPRINT, "report corpus fingerprint")
    require(data.vocabulary.size == 98, "vocabulary size")

    sampler_terminal = audit_sampler(torch, stage, data)
    for key in ("control", "treatment"):
        branch = report[key]
        require(
            branch["spec"]["key"] == key
            and branch["spec"]["width"] == 256
            and branch["spec"]["dropout"] == (0.15 if key == "control" else 0.20),
            f"{key}: branch specification",
        )
        history = branch["training"]["history"]
        require(
            [record["step"] for record in history] == list(range(0, 50001, 500)),
            f"{key}: evaluation cadence changed",
        )
        require(
            branch["final_sha256"] == file_sha256(Path(branch["spec"]["final_path"])),
            f"{key}: final endpoint hash changed",
        )
        require(
            branch["best_sha256"] == file_sha256(Path(branch["spec"]["best_path"])),
            f"{key}: best endpoint hash changed",
        )
        require(
            branch["initialization"]["residual_dropout"]
            == branch["spec"]["dropout"],
            f"{key}: initialization dropout",
        )
        require(
            branch["final_streams"]["training_generator_sha256"] == sampler_terminal,
            f"{key}: terminal batch stream disagrees with the replayed sampler",
        )
        require(
            branch["initialization"]["state_sha256"] == C256_NATIVE_INITIALIZATION[1],
            f"{key}: native initialization changed",
        )
        require(
            branch["initialization"]["parameter_count"] == C256_NATIVE_INITIALIZATION[0],
            f"{key}: parameter count changed",
        )
    control, treatment = report["control"], report["treatment"]
    require(
        control["initial_streams"]["model_sha256"]
        == treatment["initial_streams"]["model_sha256"],
        "fresh branches started from different weights",
    )
    for key in ("training_generator_sha256", "torch_rng_sha256"):
        require(
            control["initial_streams"][key] == treatment["initial_streams"][key],
            f"fresh branches started with different {key}",
        )
    require(
        control["final_streams"]["model_sha256"]
        != treatment["final_streams"]["model_sha256"],
        "fresh branches ended with identical weights",
    )
    require(
        report["final_fixed_panel_delta_treatment_minus_control"]
        == treatment["training"]["final"]["val"] - control["training"]["final"]["val"],
        "final fixed-panel delta disagrees with its branch endpoints",
    )
    require(
        report["selected_best_fixed_panel_delta_treatment_minus_control"]
        == treatment["best_losses"]["val"] - control["best_losses"]["val"],
        "selected-best delta disagrees with its branch endpoints",
    )

    fresh = report["fresh_50k_branches"]
    require(fresh["seed"] == 1366, "fresh paired panel must use an unused seed")
    fresh_audit = {
        "baseline": audit_endpoint(fresh["baseline"], "fresh baseline"),
        "candidate": audit_endpoint(fresh["candidate"], "fresh candidate"),
    }
    fresh_delta = audit_delta(
        fresh["delta"], fresh_endpoints(fresh), "fresh delta", student_t_975(499)
    )
    require(fresh_delta["candidate"] == fresh["candidate"]["name"], "fresh candidate label")
    require(fresh_delta["baseline"] == fresh["baseline"]["name"], "fresh baseline label")

    expected_outcome = (
        "p=0.20 improves C256 validation at matched 50k updates on the fresh paired panel"
        if fresh_delta["sign"] == "candidate_better"
        else "p=0.15 beats p=0.20 at C256 on the fresh paired panel"
        if fresh_delta["sign"] == "baseline_better"
        else "unresolved within paired panel uncertainty; this does not prove equivalence"
    )
    require(report["outcome"] == expected_outcome, "outcome disagrees with its interval")

    champion = audit_champion_panel(report["champion_shootout"], 1367)
    endpoints = audit_endpoint_checkpoints(torch, stage, data)
    for key, entry in endpoints.items():
        recorded = report["champion_shootout"]["endpoints"][key]
        require(
            recorded["checkpoint_sha256"] == entry["file_sha256"],
            f"{key}: report hash disagrees with the stored endpoint",
        )
        require(
            recorded["checkpoint_step"] == 50000,
            f"{key}: report endpoint is not terminal",
        )
    reproduction = report["stage_48_control_reproduction"]
    reference = stage.load_stage_48_treatment(data, stage.DEFAULT_OUTPUT_DIRECTORY)
    require(
        file_sha256(reference.path) == STAGE_48_TREATMENT_SHA256,
        "Stage 48 C256 p=0.15 treatment file hash changed",
    )
    require(
        reproduction["identical"] is True
        and reproduction["reference_sha256"] == STAGE_48_TREATMENT_SHA256
        and Path(reproduction["reference_path"]).resolve() == reference.path.resolve(),
        "Stage 48 control reproduction claim is not authenticated",
    )
    independent = {
        "model_sha256": (
            control["final_streams"]["model_sha256"],
            stage._STAGE_41.tensor_mapping_sha256(
                reference.payload["model_state_dict"]
            ),
        ),
        "optimizer_sha256": (
            control["final_streams"]["optimizer_sha256"],
            stage.state_hash(reference.payload["optimizer_state_dict"]),
        ),
        "training_generator_sha256": (
            control["final_streams"]["training_generator_sha256"],
            stage._STAGE_41.tensor_sha256(
                reference.payload["training_generator_state"]
            ),
        ),
    }
    for key, (trained, stored) in independent.items():
        require(
            trained == stored,
            f"trained C256 p=0.15 control is not the Stage 48 treatment in {key}",
        )
        recorded = reproduction["state_sha256"][key]
        require(
            list(recorded) == [trained, stored],
            f"control reproduction record disagrees with the stored treatment in {key}",
        )
    control_spec = next(
        spec
        for spec in stage.branch_specs(PROJECT / "checkpoints")
        if spec.key == "control"
    )
    require(
        Path(control["spec"]["final_path"]).resolve() == control_spec.final_path.resolve(),
        "trained control path disagrees with the canonical control spec",
    )
    control_payload = stage.validate_branch_checkpoint(
        control_spec.final_path,
        control_spec,
        data,
        stage.canonical_config(),
        "final",
    )
    ours = stage.terminal_rng_normalization(control_payload)
    theirs = stage.terminal_rng_normalization(reference.payload)
    require(
        ours["training_device_name"] == theirs["training_device_name"],
        "control reproduction trained on a different device than the treatment",
    )
    for key, (trained, stored) in (
        (
            "training_device_rng_sha256",
            (
                ours["terminal_training_rng_sha256"],
                theirs["terminal_training_rng_sha256"],
            ),
        ),
        (
            "cpu_rng_sha256",
            (
                stage.state_hash(control_payload["rng_state"]["cpu"]),
                stage.state_hash(reference.payload["rng_state"]["cpu"]),
            ),
        ),
    ):
        require(
            trained == stored,
            f"control reproduction record disagrees with the stored treatment in {key}",
        )
        require(
            list(reproduction["state_sha256"].get(key, ())) == [trained, stored],
            f"control reproduction did not record {key}",
        )
    require(
        set(reproduction["fields_checked"])
        == {
            "step",
            "snapshot_losses",
            "best_step",
            "best_val_loss",
            "best_training_loss",
            "architecture",
            "data_fingerprint",
            "model_sha256",
            "optimizer_sha256",
            "training_generator_sha256",
            "cpu_rng_sha256",
            "training_device_rng_sha256",
            "training_device_name",
        }
        and len(reproduction["fields_checked"]) == 13,
        "control reproduction did not check every field it claims to",
    )
    require(
        control["best_step"] == reference.payload["best_step"],
        "trained control selected a different best step than the Stage 48 treatment",
    )
    require(
        reproduction["rng_capture"]["training_device_name"] == LINEAGE_DEVICE_NAME,
        "control reproduction did not train on the recorded lineage device",
    )
    ladder = audit_ladder(report, fresh)
    manifest = audit_manifest(report_path, report_hash)
    require(file_sha256(report_path) == report_hash, "report changed during audit")
    return {
        "status": "passed",
        "report_path": str(report_path.resolve()),
        "report_sha256": report_hash,
        "sampler_replay_terminal_sha256": sampler_terminal,
        "fresh_paired_panel_endpoints": fresh_audit,
        "fresh_paired_panel": fresh_delta,
        "champion_panel": champion,
        "endpoint_authentication": endpoints,
        "capacity_regularization_ladder": ladder,
        "stage_48_control_reproduction": {
            "identical": reproduction["identical"],
            "fields_checked": len(reproduction["fields_checked"]),
            "reference_sha256": reproduction["reference_sha256"],
        },
        "launch_manifest": manifest,
        "outcome": expected_outcome,
        "limits": "Stored batch losses were recomputed statistically; model forward "
        "losses were not rerun. Terminal batch streams are replayed from the seed "
        "and corpus, and each checkpoint payload is re-hashed from disk.",
    }


def audit_champion_only(path: Path) -> dict[str, Any]:
    """Audit the standalone four-way panel without touching the runner."""

    require(path.is_file(), f"champion report not present: {path}")
    panel = json.loads(path.read_text(encoding="utf-8"))
    audit = audit_champion_panel(panel, 1367)
    require(
        panel["uncertainty_scope"].startswith("Student-t 95% panel-sampling interval"),
        "champion report must state its uncertainty scope",
    )
    return {
        "status": "passed",
        "champion_report_path": str(path.resolve()),
        "champion_report_sha256": file_sha256(path),
        "champion_panel": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=PROJECT / "checkpoints" / "stage_49_report.json"
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--champion-report",
        type=Path,
        default=PROJECT / "checkpoints" / "stage_49_champion_shootout.json",
        help="standalone four-way panel written by --champion-panel-only",
    )
    parser.add_argument(
        "--statistics-only",
        action="store_true",
        help="audit the report's 500-batch paired panels without importing Torch",
    )
    parser.add_argument(
        "--champion-only",
        action="store_true",
        help="audit only the standalone four-way panel report",
    )
    args = parser.parse_args()
    if args.champion_only:
        result = audit_champion_only(args.champion_report)
    elif args.statistics_only:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        panels = {
            "fresh_50k_branches": report["fresh_50k_branches"],
            "champion_shootout": report["champion_shootout"],
        }
        critical = student_t_975(499)
        result = {
            "status": "passed",
            "statistics_only": True,
            "fresh_paired_panel": audit_delta(
                panels["fresh_50k_branches"]["delta"],
                fresh_endpoints(panels["fresh_50k_branches"]),
                "fresh delta",
                critical,
            ),
            "champion_panel": audit_champion_panel(panels["champion_shootout"], 1367),
        }
    else:
        result = audit_run(args.report, args.data)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ArithmeticError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        raise SystemExit(1) from error