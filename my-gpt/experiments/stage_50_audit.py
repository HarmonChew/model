"""Read-only CPU audit of a completed canonical Stage 50 run.

The runner's model factory, checkpoint validators, and serialization hash
primitives are reused, but protocol constants, the sampler, paired statistics,
the shared-initialization prefix, the parameter-budget ladder, and every
recorded hash are re-derived here. No checkpoint is modified, no accelerator is
touched, and no model forward pass is run.
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
# the recorded artifacts (torch 2.14.0+cu130). Residual dropout adds no
# parameters and consumes no initialization RNG, so one fingerprint covers both
# probabilities at a width; depth changes the construction, so each depth has its
# own. The C320 entry is the stored peer's own recorded count and hash.
C256_L6_NATIVE_INITIALIZATION = (
    4850274,
    "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
)
C256_L8_NATIVE_INITIALIZATION = (
    6428258,
    "6bef5bb6fa54e908ca06f64d299b1c3b35db7cf8676a8ade7d33c2b9cdb2f8c3",
)
C320_NATIVE_INITIALIZATION = (
    7537378,
    "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
)
# The recorded per-block cost of one C256 layer, which is what makes the depth
# step a clean parameter-budget question.
C256_BLOCK_PARAMETERS = 788_992
# Stage 50 records these two depths and no others, and the treatment must be the
# control plus exactly the two added blocks.
CONTROL_DEPTH = 6
TREATMENT_DEPTH = 8
CONTROL_TOPOLOGY = (23, 23, 23, 23, None, None)
TREATMENT_TOPOLOGY = (23, 23, 23, 23, 23, 23, None, None)
PRE_GLOBAL_RECEPTIVE_FIELD = {CONTROL_DEPTH: 89, TREATMENT_DEPTH: 133}
FIRST_GLOBAL_REACHABILITY_LAYER = {CONTROL_DEPTH: 5, TREATMENT_DEPTH: 7}
# One state-dict entry per head per layer is an attention-mask buffer; the
# shared prefix is compared on the learned entries and on the unchanged masks.
MASK_SUFFIX = ".tril"
# The two layers whose attention policy differs between the depths, so their mask
# buffers are the only shared entries expected to differ alongside the head.
CHANGED_MASK_LAYERS = (4, 5)
EXPECTED_DIFFERING = tuple(
    f"blocks.{layer}.sa.heads.{head}{MASK_SUFFIX}"
    for layer in CHANGED_MASK_LAYERS
    for head in range(4)
) + ("lm_head.weight", "lm_head.bias")
CONTROL_STATE_KEYS = 162
TREATMENT_STATE_KEYS = 214
ADDED_STATE_KEYS = 52
IDENTICAL_STATE_KEYS = 152
IDENTICAL_STATE_SHA256 = "24d2260f37d3af57a2f6c323524bc448ea6631bccc50079ebaf29b2c7ad05234"
# One sampler, one seed, one batch shape: every endpoint in the budget panel
# consumed the same training stream, so each terminal batch-generator state must
# equal this replay.
SHARED_TRAINING_GENERATOR_SHA256 = (
    "986ffc5650abe72ba270a01cc94ab5a247a946e1f4f040d1eef1508f8a7c86b9"
)
SHARED_INITIAL_GENERATOR_SHA256 = (
    "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced"
)
# The dropout schedule advances the accelerator RNG by a fixed quantum per
# dropout site, which is why two widths and two probabilities recorded the same
# terminal device state in Stages 46-49. Depth changes the number of sites, so
# the L=6 endpoints share this state and the L=8 treatment must not.
SHARED_L6_TERMINAL_TRAINING_RNG_SHA256 = (
    "0b8d137f0468f1a12db1c880b3d7c74f8dc0039927742b7d71cc9e107546ee97"
)
LINEAGE_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
TRAINING_BATCH_SEED = 1338
TRAINING_RNG_SEED = 1340
INITIALIZATION_SEED = 1337
# The stored endpoints this stage's control must reproduce and be scored against.
STAGE_49_CONTROL_SHA256 = (
    "8aaf70ae60e546168a976ecc640a6b1dac3044d135ccf7f9d822dfa5598affec"
)
STAGE_49_CONTROL_BEST_SHA256 = (
    "a73a9912e150a2ba56aa29568f1abef3afd466dd8bb36aeccf973e9effa51c18"
)
STAGE_46_C320_P015_SHA256 = (
    "e47016371b6a588448dbaa0bad6d05ff8564b7b21c6f6da81f20bbb56e916197"
)
STAGE_48_TREATMENT_SHA256 = (
    "04952aeeab517f2a31ba4209737d4f8d495d40fb84aca8c9b5f3b58bb44f98b4"
)
STAGE_45_CHAMPION_SHA256 = (
    "e58ce4532a3ce0623ea622b91ea7ee04232c8d1f12d8b2c53591c7efec299e9f"
)
PANEL_KEYS = ("c256_l6", "c256_l8", "c320_l6")
FOREIGN_PANEL_KEYS = ("c256_l6", "c256_l8", "c256_l6_lineage", "c320_l6")
CANONICAL_DEVICE_MODE = "canonical_lineage_device"
FOREIGN_DEVICE_MODE = "foreign_device"
FRESH_PANEL_SEED = 1368
BUDGET_PANEL_SEED = 1369
# The original depth result, transcribed from the Stage 24 report rather than
# recomputed: a different corpus size, width, depth, and budget.
PRIOR_DEPTH_REFERENCE = (24, -0.011038, 0.000647, 816_705, 1_212_481)
# Every artifact of this lineage, including the stored endpoints this stage reads
# and the Stages 43-49 endpoints those were already compared with.
HISTORICAL_HASHES = {
    "stage_41_complete_shakespeare_step_44000_final_checkpoint.pt": "666c9e72345399cf63b412128a1427d1c778d58794473c0e8503516d2235db36",
    "stage_42_control_3e_5_step_50000_final_checkpoint.pt": "57024bd7bb9e698ba3c16c23533aa5e5d52562eebe87841ac25a992def2b9baf",
    "stage_44_treatment_c256_step_50000_final_checkpoint.pt": "c9e0092cb1986885f8cef948b703bb4fbf18ae0714e4e3d062fecc337081b247",
    "stage_45_control_c256_step_50000_final_checkpoint.pt": STAGE_45_CHAMPION_SHA256,
    "stage_45_control_c256_best_checkpoint.pt": "1c3668c88a82942fed3ce25b641e213f566ee4b500a3d0766584e158b9f51d65",
    "stage_45_treatment_c320_step_50000_final_checkpoint.pt": "a6f908c18d02ea3d7a9060d4fb7886320bff41eea06a0e97e2196ea57f34984b",
    "stage_46_treatment_c320_p015_step_50000_final_checkpoint.pt": STAGE_46_C320_P015_SHA256,
    "stage_47_treatment_c320_p020_step_50000_final_checkpoint.pt": "7289527c6078b92f52abc98f39c3871c30aed7fa56a2d8a2901d5f4f035b170f",
    "stage_48_control_c256_p010_step_50000_final_checkpoint.pt": "8c817fd5ea8a557e60f693a577cad68bbd9d483402d8e90fe343b23590c8090e",
    "stage_48_treatment_c256_p015_step_50000_final_checkpoint.pt": STAGE_48_TREATMENT_SHA256,
    "stage_48_treatment_c256_p015_best_checkpoint.pt": "62d6b5ee7d9009cafd48561e5622c9a1e7b3254e3d26fbfc57c280662ce723ab",
    "stage_49_control_c256_p015_step_50000_final_checkpoint.pt": STAGE_49_CONTROL_SHA256,
    "stage_49_control_c256_p015_best_checkpoint.pt": STAGE_49_CONTROL_BEST_SHA256,
    "stage_49_treatment_c256_p020_step_50000_final_checkpoint.pt": "cff587a1b79b03e946702947cd2fe6811b08c44c60488c02a43c97bc7fac5105",
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
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for iteration in range(1, 300):
        m2 = 2 * iteration
        numerator = iteration * (b - iteration) * x / (
            (a + m2 - 1) * (a + m2)
        )
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + iteration) * (a + b + iteration) * x / (
            (a + m2) * (a + m2 + 1)
        )
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * beta_fraction(a, b, x) / a
    return 1 - front * beta_fraction(b, a, 1 - x) / b


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
    paired panel stores its delta under `delta`, while the budget panel stores one
    delta per unordered pair under `comparisons`.
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
    close(
        reported["confidence_low"],
        reported["mean_delta"] - margin,
        f"{label}: refit lower",
    )
    close(
        reported["confidence_high"],
        reported["mean_delta"] + margin,
        f"{label}: refit upper",
    )
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


def panel_endpoint_architecture(
    entry: dict[str, Any], label: str
) -> dict[str, Any]:
    """Check one budget endpoint's declared architecture against its depth."""

    depth = entry["depth"]
    width = entry["width"]
    parameters = entry["parameter_count"]
    require(depth in (CONTROL_DEPTH, TREATMENT_DEPTH), f"{label}: depth changed")
    require(width in (256, 320), f"{label}: width changed")
    require(
        entry["parameter_ratio_vs_control"] > 0,
        f"{label}: parameter ratio is not positive",
    )
    if width == 256 and depth == CONTROL_DEPTH:
        require(
            parameters == C256_L6_NATIVE_INITIALIZATION[0],
            f"{label}: the L=6 C256 budget is not the recorded one",
        )
    elif width == 256 and depth == TREATMENT_DEPTH:
        require(
            parameters == C256_L8_NATIVE_INITIALIZATION[0],
            f"{label}: the L=8 C256 budget is not the recorded one",
        )
    elif width == 320:
        require(
            parameters == C320_NATIVE_INITIALIZATION[0],
            f"{label}: the C320 budget is not the recorded one",
        )
        require(depth == CONTROL_DEPTH, f"{label}: C320 is only recorded at L=6")
    return {"width": width, "depth": depth, "parameters": parameters}


def audit_budget_panel(
    panel: dict[str, Any], expected_seed: int, outcome: str | None = None
) -> dict[str, Any]:
    """Re-derive every budget-shootout statistic, its ranking, and its RNG story.

    `outcome` is the prose the panel's own conclusion is stated in. A stage
    report keeps it beside the panel rather than inside it, so it is passed in
    there; a standalone panel file carries it itself.
    """

    require(panel["seed"] == expected_seed, "budget panel seed changed")
    require(panel["eval_iters"] == 500, "budget panel must hold 500 batches")
    require(panel["shared_panel_per_batch"] is True, "budget panel is not shared")
    require(panel["dropout_enabled"] is False, "budget panel enabled dropout")
    require(
        panel["shared_training_batch_stream_verified"] is True,
        "budget panel did not verify its shared training stream",
    )
    require(
        panel["shared_training_generator_sha256"] == SHARED_TRAINING_GENERATOR_SHA256,
        "budget panel recorded a different shared training stream",
    )
    keys = panel["panel_keys"]
    require(
        keys.get("control") == "c256_l6"
        and keys.get("treatment") == "c256_l8"
        and keys.get("budget_peer") == "c320_l6",
        "budget panel keys changed",
    )
    stored_l6 = keys.get("stored_l6_peer")
    expected_keys = FOREIGN_PANEL_KEYS if stored_l6 else PANEL_KEYS
    require(
        stored_l6 in (None, "c256_l6_lineage"), "unexpected stored L=6 peer key"
    )
    require(
        sorted(panel["endpoints"]) == sorted(expected_keys),
        "budget endpoint set changed",
    )
    architectures = {
        key: panel_endpoint_architecture(entry, f"endpoint {key}")
        for key, entry in panel["endpoints"].items()
    }
    # Two endpoints of one weight budget: the depth step must be exactly the two
    # added blocks, and the wider spend must cost strictly more.
    control_parameters = architectures["c256_l6"]["parameters"]
    treatment_parameters = architectures["c256_l8"]["parameters"]
    require(
        treatment_parameters - control_parameters
        == (TREATMENT_DEPTH - CONTROL_DEPTH) * C256_BLOCK_PARAMETERS,
        "the depth step is not exactly two C256 blocks",
    )
    require(
        architectures["c320_l6"]["parameters"] > treatment_parameters,
        "the width peer must cost more than the depth treatment",
    )
    if stored_l6:
        # The device-shift row compares one architecture against itself, so the
        # two L=6 endpoints must be the same budget.
        require(
            architectures[stored_l6]["parameters"] == control_parameters,
            "the stored L=6 peer is not the same architecture as the control",
        )
    for key, entry in panel["endpoints"].items():
        close(
            entry["parameter_ratio_vs_control"],
            entry["parameter_count"] / control_parameters,
            f"endpoint {key}: parameter ratio",
        )
    projections = panel["rng_projections"]
    device_names = panel["device_names"]
    require(
        set(device_names) == set(expected_keys), "panel device names changed"
    )
    require(
        all(
            projections[key]["training_device_name"] == device_names[key]
            for key in device_names
        ),
        "panel device names disagree with its RNG projections",
    )
    fresh_device = panel["fresh_device_name"]
    require(
        (fresh_device, device_names["c256_l6"], device_names["c256_l8"])
        == (fresh_device, fresh_device, fresh_device),
        "the two fresh branches were not trained on the panel's fresh device",
    )
    confounded = sorted(panel["device_confounded_endpoints"])
    require(
        panel["device_confounded"] is bool(confounded),
        "the device-confound flag disagrees with its endpoint list",
    )
    require(
        confounded
        == sorted(
            key
            for key in expected_keys
            if key not in ("c256_l6", "c256_l8") and device_names[key] != fresh_device
        ),
        "the device-confound list disagrees with the recorded device names",
    )
    if confounded:
        require(
            fresh_device != LINEAGE_DEVICE_NAME,
            "a confounded panel must not have trained its branches on the lineage device",
        )
        require(
            all(device_names[key] == LINEAGE_DEVICE_NAME for key in confounded),
            "a device-confounded stored peer was not trained on the lineage device",
        )
    else:
        require(
            device_names["c320_l6"] == fresh_device,
            "an unconfounded panel must train every endpoint on one device",
        )
    if "c320_l6" not in confounded:
        require(
            projections["c256_l6"]["terminal_training_rng_sha256"]
            == SHARED_L6_TERMINAL_TRAINING_RNG_SHA256,
            "the L=6 control did not end on the recorded training-device RNG state",
        )
        require(
            projections["c320_l6"]["terminal_training_rng_sha256"]
            == SHARED_L6_TERMINAL_TRAINING_RNG_SHA256,
            "the stored C320 peer disagrees with the recorded L=6 draw schedule",
        )
        require(
            projections["c256_l8"]["terminal_training_rng_sha256"]
            != SHARED_L6_TERMINAL_TRAINING_RNG_SHA256,
            "the deeper treatment did not consume a different dropout schedule",
        )
    else:
        require(
            projections["c256_l6"]["terminal_training_rng_sha256"]
            != projections["c256_l8"]["terminal_training_rng_sha256"],
            "the deeper treatment did not consume a different dropout schedule",
        )
    if stored_l6:
        # The shift is a device effect only if the two L=6 endpoints started from
        # the same weights. Initialization happens on the CPU, so that is a
        # checkable claim rather than an assumption.
        require(
            panel["endpoints"][stored_l6]["initial_streams"]["model_sha256"]
            == panel["endpoints"]["c256_l6"]["initial_streams"]["model_sha256"],
            "the device-shift row compares different initializations",
        )
        require(
            panel["endpoints"][stored_l6]["initialization_sha256"]
            == C256_L6_NATIVE_INITIALIZATION[1]
            and panel["endpoints"]["c256_l6"]["initialization_sha256"]
            == C256_L6_NATIVE_INITIALIZATION[1],
            "the device-shift row does not carry the recorded native initialization",
        )
        require(
            panel["endpoints"][stored_l6]["checkpoint_sha256"]
            != panel["endpoints"]["c256_l6"]["checkpoint_sha256"],
            "the device-shift row compares one checkpoint with itself",
        )
    endpoints = {}
    for key, endpoint in panel["endpoints"].items():
        endpoints[key] = audit_endpoint(endpoint, f"endpoint {key}")
    critical = student_t_975(499)
    expected_pairs = {
        frozenset((candidate, baseline))
        for index, candidate in enumerate(expected_keys)
        for baseline in expected_keys[index + 1 :]
    }
    forward = sorted(panel["comparisons"])
    stored_pairs = set()
    for key in forward:
        candidate, separator, baseline = key.partition("_minus_")
        require(
            separator == "_minus_" and candidate in expected_keys and baseline in expected_keys,
            f"unrecognised comparison key {key!r}",
        )
        require(candidate != baseline, f"self-comparison {key!r}")
        stored_pairs.add(frozenset((candidate, baseline)))
    # The runner stores each unordered pair once, in whatever order it enumerated
    # the endpoints, so coverage is checked as a set of pairs rather than as a set
    # of strings.
    require(
        stored_pairs == expected_pairs and len(stored_pairs) == len(forward),
        "budget comparisons must cover every unordered pair exactly once",
    )
    comparisons = {
        key: audit_delta(panel["comparisons"][key], panel["endpoints"], key, critical)
        for key in forward
    }
    for key, stored in panel["comparisons"].items():
        candidate, _, baseline = key.partition("_minus_")
        mirror = panel["comparisons"].get(f"{baseline}_minus_{candidate}")
        if mirror is None:
            # The runner stores each unordered pair once, so a missing mirror is
            # the expected shape rather than a defect.
            continue
        close(mirror["mean_delta"], -stored["mean_delta"], "mirror mean")
        close(mirror["confidence_low"], -stored["confidence_high"], "mirror lower")
        close(mirror["confidence_high"], -stored["confidence_low"], "mirror upper")
    ranked = sorted(expected_keys, key=lambda key: panel["endpoints"][key]["mean_loss"])
    require(
        [entry["key"] for entry in panel["ranking"]] == ranked,
        "budget ranking is not ordered by mean validation loss",
    )
    for position, entry in enumerate(panel["ranking"], start=1):
        require(entry["rank"] == position, "budget rank numbering changed")
        close(
            entry["mean_validation_loss"],
            panel["endpoints"][entry["key"]]["mean_loss"],
            f"ranking {entry['key']}: mean",
        )
        require(
            entry["parameter_count"]
            == panel["endpoints"][entry["key"]]["parameter_count"],
            f"ranking {entry['key']}: parameter count",
        )
        require(
            entry["checkpoint_sha256"]
            == panel["endpoints"][entry["key"]]["checkpoint_sha256"],
            f"ranking {entry['key']}: checkpoint hash",
        )
    champion = ranked[0]
    defeated = []
    for key in ranked[1:]:
        stored = panel["comparisons"].get(f"{key}_minus_{champion}")
        if stored is not None:
            resolved = stored["confidence_low"] > 0
        else:
            stored = panel["comparisons"][f"{champion}_minus_{key}"]
            resolved = stored["confidence_high"] < 0
        if resolved:
            defeated.append(key)
    if outcome is None:
        outcome = panel.get("budget_outcome")
    if outcome is not None:
        if len(defeated) == len(ranked) - 1:
            require("resolved champion" in outcome, "outcome ignores a resolved panel")
        else:
            require(
                "does not resolve" in outcome
                and "leads the shared fresh panel" in outcome,
                "outcome ignores an unresolved peer",
            )
        # The substitution clause is a separate claim about the same budget, and
        # it must name the direction its own interval supports.
        depth = panel["comparisons"].get("c256_l8_minus_c320_l6") or panel[
            "comparisons"
        ]["c320_l6_minus_c256_l8"]
        mirrored = "c256_l8_minus_c320_l6" not in panel["comparisons"]
        low = -depth["confidence_high"] if mirrored else depth["confidence_low"]
        high = -depth["confidence_low"] if mirrored else depth["confidence_high"]
        if high < 0:
            require("depth is the better spend" in outcome, "outcome missed a depth win")
        elif low > 0:
            require("width is the better spend" in outcome, "outcome missed a width win")
        else:
            require(
                "neither allocation is preferred" in outcome,
                "outcome missed an unresolved spend",
            )
        # The device clause is the third claim the panel makes; its own interval
        # is re-derived by audit_delta above like every other pair.
        if stored_l6:
            if confounded:
                require(
                    "device-confounded" in outcome, "a confounded panel must say so"
                )
            require(
                "moving the same L=6 architecture" in outcome,
                "outcome omitted the measured device shift",
            )
        else:
            require(
                "no device shift was measured" in outcome,
                "a single-device panel must not claim a device shift",
            )
    device_shift = None
    if stored_l6:
        forward = f"c256_l6_minus_{stored_l6}"
        if forward in panel["comparisons"]:
            record = panel["comparisons"][forward]
            device_shift = {
                "mean_delta": record["mean_delta"],
                "confidence_low": record["confidence_low"],
                "confidence_high": record["confidence_high"],
            }
        else:
            record = panel["comparisons"][f"{stored_l6}_minus_c256_l6"]
            device_shift = {
                "mean_delta": -record["mean_delta"],
                "confidence_low": -record["confidence_high"],
                "confidence_high": -record["confidence_low"],
            }
    return {
        "panel_seed": panel["seed"],
        "student_t_critical_value": critical,
        "endpoint_architectures": architectures,
        "endpoints": endpoints,
        "comparisons": comparisons,
        "ranking": ranked,
        "resolved_against": defeated,
        "unresolved_against": [key for key in ranked[1:] if key not in defeated],
        "depth_parameter_delta": treatment_parameters - control_parameters,
        "fresh_device_name": fresh_device,
        "device_names": device_names,
        "device_confounded": bool(confounded),
        "device_confounded_endpoints": confounded,
        "device_shift": device_shift,
        "outcome": outcome,
        "budget_ratio_depth_over_width": (
            treatment_parameters / architectures["c320_l6"]["parameters"]
        ),
        "scope": "panel-sampling uncertainty conditional on stored endpoints",
    }


def load_runtime() -> tuple[Any, Any]:
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(PROJECT))
    spec = importlib.util.spec_from_file_location(
        "stage_50_for_independent_audit", PROJECT / "50_depth_revisit.py"
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
    recorded stream here is what justifies treating the three budget endpoints as
    trained on one batch stream.
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


def audit_shared_prefix(
    torch: Any, stage: Any, data: Any, report: dict[str, Any]
) -> dict[str, Any]:
    """Re-derive the depth step's shared initialization prefix from the seed.

    The claim is structural rather than statistical: because the native
    construction draws in block order, the deeper model's first six blocks and
    both embedding tables are byte-identical to the shallower model's, and the
    only shared entries that may differ are the language head and the mask
    buffers of the layers whose attention policy changed.
    """

    config = stage.canonical_config()
    specs = stage.branch_specs(PROJECT / "checkpoints")
    stage.validate_branch_specs(specs)
    control_spec, treatment_spec = specs
    recorded_branches = {
        entry["branch_key"]: entry for entry in report["branches"]
    }
    states = []
    for spec in specs:
        stage._STAGE_41.seed_everything(config.seed)
        model = stage.build_model(
            data,
            config,
            stage.spec_for(spec, config.block_size),
            4,
            torch.device("cpu"),
        )
        states.append(
            {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            }
        )
        recorded = recorded_branches[spec.key]["initialization"]
        require(
            (recorded["parameter_count"], recorded["state_sha256"])
            == stage.locked_initialization_fingerprint(spec),
            f"{spec.key}: recorded native fingerprint changed",
        )
        require(
            stage._STAGE_41.tensor_mapping_sha256(model.state_dict())
            == recorded["state_sha256"],
            f"{spec.key}: native initialization does not reproduce from its seed",
        )
        require(
            recorded["block_parameter_count"] == C256_BLOCK_PARAMETERS,
            f"{spec.key}: per-block parameter count changed",
        )
        del model
    control_state, treatment_state = states
    require(
        (len(control_state), len(treatment_state))
        == (CONTROL_STATE_KEYS, TREATMENT_STATE_KEYS),
        "native state-dict key counts changed",
    )
    shared = set(control_state) & set(treatment_state)
    added = sorted(set(treatment_state) - set(control_state))
    require(
        not set(control_state) - set(treatment_state),
        "the treatment dropped control initialization entries",
    )
    require(
        len(added) == ADDED_STATE_KEYS,
        "the treatment added the wrong number of initialization entries",
    )
    require(
        all(
            name.startswith(("blocks.6.", "blocks.7."))
            for name in added
        ),
        "the treatment added initialization entries outside the new blocks",
    )
    differing = sorted(
        name for name in shared if not torch.equal(control_state[name], treatment_state[name])
    )
    require(
        differing == sorted(EXPECTED_DIFFERING),
        "the shared initialization prefix changed outside its licensed exceptions",
    )
    identical = tuple(sorted(shared - set(EXPECTED_DIFFERING)))
    require(
        len(identical) == IDENTICAL_STATE_KEYS,
        "the number of byte-identical shared entries changed",
    )
    digests = (
        stage._STAGE_41.tensor_mapping_sha256(
            {name: control_state[name] for name in identical}
        ),
        stage._STAGE_41.tensor_mapping_sha256(
            {name: treatment_state[name] for name in identical}
        ),
    )
    require(
        digests[0] == digests[1] == IDENTICAL_STATE_SHA256,
        "the byte-identical shared entries do not reproduce their recorded hash",
    )
    recorded_shared = report["shared_initialization"]
    require(
        recorded_shared["control_identical_keys_sha256"] == digests[0]
        and recorded_shared["treatment_identical_keys_sha256"] == digests[1]
        and recorded_shared["identical_key_count"] == IDENTICAL_STATE_KEYS
        and recorded_shared["added_key_count"] == ADDED_STATE_KEYS
        and recorded_shared["treatment_is_a_key_superset"] is True
        and tuple(recorded_shared["changed_mask_layers"]) == CHANGED_MASK_LAYERS,
        "the report's shared-initialization record disagrees with the rebuild",
    )
    # Every trained endpoint must carry the initialization those rebuilds prove.
    for spec in specs:
        payload = stage.validate_branch_checkpoint(
            spec.final_path, spec, data, config, "final"
        )
        require(
            payload["initialization"]["state_sha256"]
            == stage.locked_initialization_fingerprint(spec)[1],
            f"{spec.key}: trained endpoint lost its native initialization",
        )
    for depth, topology in (
        (CONTROL_DEPTH, CONTROL_TOPOLOGY),
        (TREATMENT_DEPTH, TREATMENT_TOPOLOGY),
    ):
        audit = stage.reachability_audit(config.block_size, topology)
        require(
            tuple(audit.layer_contexts) == topology,
            f"L={depth}: topology changed",
        )
        require(
            audit.widths_by_layer[sum(1 for c in topology if c is not None) - 1]
            == PRE_GLOBAL_RECEPTIVE_FIELD[depth],
            f"L={depth}: pre-global receptive field changed",
        )
        require(
            audit.first_global_reachability_layer
            == FIRST_GLOBAL_REACHABILITY_LAYER[depth],
            f"L={depth}: first global reachability layer changed",
        )
    return {
        "control_state_keys": len(control_state),
        "treatment_state_keys": len(treatment_state),
        "added_state_keys": len(added),
        "identical_state_keys": len(identical),
        "identical_state_sha256": digests[0],
        "differing_state_entries": differing,
        "added_block_layers": [CONTROL_DEPTH, CONTROL_DEPTH + 1],
        "pre_global_receptive_field": dict(PRE_GLOBAL_RECEPTIVE_FIELD),
    }


def audit_endpoint_checkpoints(
    torch: Any, stage: Any, data: Any, mode: str
) -> dict[str, Any]:
    """Re-authenticate every budget endpoint through the runner itself."""

    if mode not in (CANONICAL_DEVICE_MODE, FOREIGN_DEVICE_MODE):
        raise ValueError(f"unknown device mode: {mode!r}")
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
        require(
            payload["architecture"]["n_layer"] == spec.depth
            and tuple(payload["architecture"]["layer_contexts"]) == spec.layer_contexts
            and payload["architecture"]["topology"]
            == stage.topology_label(spec.layer_contexts)
            and payload["architecture"]["pre_global_receptive_field"]
            == PRE_GLOBAL_RECEPTIVE_FIELD[spec.depth],
            f"{path.name}: depth architecture record changed",
        )
        endpoints[f"c{spec.width}_l{spec.depth}"] = {
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
    peer = stage.load_c320_budget_peer(data, directory)
    payload = peer.payload
    recorded = payload["payload_sha256"]
    stripped = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    require(
        recorded == stage.state_hash(stripped),
        f"{peer.path.name}: payload integrity mismatch",
    )
    require(
        file_sha256(peer.path) == STAGE_46_C320_P015_SHA256,
        f"{peer.path.name}: file hash changed",
    )
    endpoints[peer.key] = {
        "path": str(peer.path.resolve()),
        "file_sha256": file_sha256(peer.path),
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
    if mode == FOREIGN_DEVICE_MODE:
        stored = stage.load_stage_49_control(data, directory)
        stored_payload = stored.payload
        stored_recorded = stored_payload["payload_sha256"]
        stripped = {
            key: value
            for key, value in stored_payload.items()
            if key != "payload_sha256"
        }
        require(
            stored_recorded == stage.state_hash(stripped),
            f"{stored.path.name}: payload integrity mismatch",
        )
        require(
            file_sha256(stored.path) == STAGE_49_CONTROL_SHA256,
            f"{stored.path.name}: file hash changed",
        )
        require(
            stored.key == "c256_l6_lineage",
            "the stored control's panel key changed",
        )
        endpoints[stored.key] = {
            "path": str(stored.path.resolve()),
            "file_sha256": file_sha256(stored.path),
            "payload_sha256": stored_recorded,
            "initialization_sha256": stored_payload["initialization"]["state_sha256"],
            "parameter_count": stored_payload["initialization"]["parameter_count"],
            "terminal_generator_sha256": stage._STAGE_41.tensor_sha256(
                stored_payload["training_generator_state"]
            ),
            "initial_streams": dict(stored_payload["initial_streams"]),
            "origin_stage": stored_payload["initialization"]["origin_stage"],
            "best_step": stored_payload["best_step"],
            "rng_projection": stage.terminal_rng_normalization(stored_payload),
            "recorded_training_device_index": stored_payload["runtime"].get(
                "device_index"
            ),
            "cuda_entries_at_capture": len(
                stored_payload["rng_state"].get("cuda", ())
            ),
        }
    expected = FOREIGN_PANEL_KEYS if mode == FOREIGN_DEVICE_MODE else PANEL_KEYS
    require(
        sorted(endpoints) == sorted(expected),
        "re-authenticated endpoint set changed",
    )
    # One batch stream, two architectures: the depth groups must differ in
    # initial weights and agree on everything the sampler controls. The sampler
    # is a CPU generator, so this holds across accelerators too.
    require(
        {entry["terminal_generator_sha256"] for entry in endpoints.values()}
        == {SHARED_TRAINING_GENERATOR_SHA256},
        "budget endpoints did not end on the recorded shared batch stream",
    )
    require(
        {
            entry["initial_streams"]["training_generator_sha256"]
            for entry in endpoints.values()
        }
        == {SHARED_INITIAL_GENERATOR_SHA256},
        "budget endpoints did not start at the recorded batch stream",
    )
    expected_fingerprints = {
        "c256_l6": C256_L6_NATIVE_INITIALIZATION,
        "c256_l8": C256_L8_NATIVE_INITIALIZATION,
        "c320_l6": C320_NATIVE_INITIALIZATION,
    }
    expected_origins = {"c256_l6": 50, "c256_l8": 50, "c320_l6": 46}
    if mode == FOREIGN_DEVICE_MODE:
        # Initialization is a CPU-side draw, so the stored lineage control and a
        # foreign control must carry the same native state. That is what makes
        # the device shift a device effect rather than a different starting point.
        expected_fingerprints["c256_l6_lineage"] = C256_L6_NATIVE_INITIALIZATION
        expected_origins["c256_l6_lineage"] = 49
    for key, expected_fingerprint in expected_fingerprints.items():
        entry = endpoints[key]
        require(
            (entry["parameter_count"], entry["initialization_sha256"])
            == expected_fingerprint,
            f"{key}: native initialization fingerprint changed",
        )
        require(
            entry["origin_stage"] == expected_origins[key],
            f"{key}: initialization origin changed",
        )
    projections = {key: value["rng_projection"] for key, value in endpoints.items()}
    fresh_device = projections["c256_l6"]["training_device_name"]
    require(
        projections["c256_l8"]["training_device_name"] == fresh_device,
        "the two fresh branches were not trained on one device",
    )
    if mode == CANONICAL_DEVICE_MODE:
        require(
            {value["training_device_name"] for value in projections.values()}
            == {LINEAGE_DEVICE_NAME},
            "budget endpoints were not all trained on the recorded device",
        )
        require(
            projections["c256_l6"]["terminal_training_rng_sha256"]
            == SHARED_L6_TERMINAL_TRAINING_RNG_SHA256
            and projections["c320_l6"]["terminal_training_rng_sha256"]
            == SHARED_L6_TERMINAL_TRAINING_RNG_SHA256,
            "the two L=6 endpoints do not share the recorded draw schedule",
        )
        require(
            projections["c256_l8"]["terminal_training_rng_sha256"]
            != SHARED_L6_TERMINAL_TRAINING_RNG_SHA256,
            "the L=8 treatment shares the L=6 draw schedule",
        )
    else:
        require(
            fresh_device != LINEAGE_DEVICE_NAME,
            "a foreign run must not have trained on the lineage device",
        )
        require(
            all(
                projections[key]["training_device_name"] == LINEAGE_DEVICE_NAME
                for key in ("c320_l6", "c256_l6_lineage")
            ),
            "the stored peers were not trained on the lineage device",
        )
        require(
            projections["c256_l8"]["terminal_training_rng_sha256"]
            != projections["c256_l6"]["terminal_training_rng_sha256"],
            "the deeper treatment did not consume a different dropout schedule",
        )
    for key, entry in endpoints.items():
        if entry["origin_stage"] == 50:
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
    for path, digest in historical_paths():
        require(path.is_file(), f"historical artifact missing: {path.name}")
        require(file_sha256(path) == digest, f"historical artifact changed: {path.name}")
    return endpoints


def historical_paths() -> list[tuple[Path, str]]:
    directory = PROJECT / "checkpoints"
    return [(directory / name, digest) for name, digest in HISTORICAL_HASHES.items()]


def audit_manifest(report_path: Path, report_hash: str) -> dict[str, Any]:
    """Check the launch manifest when present; never require it."""

    manifest_path = report_path.with_name("stage_50_run_status.json")
    if not manifest_path.is_file():
        return {"status": "skipped", "reason": "launch manifest not present"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(manifest["stage"] == 50, "manifest stage changed")
    require(
        manifest["script_sha256"] == file_sha256(PROJECT / "50_depth_revisit.py"),
        "runner source differs from launch manifest",
    )
    require(file_sha256(report_path) == report_hash, "report changed during audit")
    manifest["report_sha256"] = report_hash
    log_path = PROJECT / manifest["log"]
    require(log_path.is_file(), "launch log missing")
    require(file_sha256(log_path) == manifest["log_sha256"], "launch log changed")
    panel_path = PROJECT / manifest["budget_report"]
    if panel_path.is_file():
        require(
            file_sha256(panel_path) == manifest["budget_report_sha256"],
            "budget report changed",
        )
    return {"status": "passed", "manifest": str(manifest_path.resolve())}


def audit_ladder(report: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Check the parameter-budget ladder against its sources."""

    ladder = report["depth_budget_ladder"]
    measured = ladder["measured_depth_step"]
    require(
        measured["source_stage"] == 50 and measured["width"] == 256,
        "ladder measured-step provenance",
    )
    close(measured["mean_delta"], fresh["delta"]["mean_delta"], "ladder measured delta")
    require(
        measured["fresh_panel_seed"] == fresh["seed"], "ladder measured panel seed"
    )
    require(
        measured["parameter_delta"]
        == (TREATMENT_DEPTH - CONTROL_DEPTH) * C256_BLOCK_PARAMETERS,
        "ladder measured parameter delta",
    )
    close(
        measured["parameter_ratio"],
        C256_L8_NATIVE_INITIALIZATION[0] / C256_L6_NATIVE_INITIALIZATION[0],
        "ladder measured parameter ratio",
    )
    rows = ladder["rows"]
    panel = report["budget_shootout"]
    panel_keys = panel["panel_keys"]
    stored_l6 = panel_keys.get("stored_l6_peer")
    require(
        len(rows) == sum(1 for key in panel_keys.values() if key is not None),
        "ladder must carry one row per panel endpoint",
    )
    require(
        [row["key"] for row in rows]
        == sorted(
            panel_keys[key]
            for key in ("control", "treatment", "budget_peer", "stored_l6_peer")
            if panel_keys.get(key) is not None
        ),
        "ladder rows do not name the panel's endpoints",
    )
    require(
        [row["parameters"] for row in rows]
        == sorted(row["parameters"] for row in rows),
        "ladder rows are not ordered by parameter count",
    )
    expected_stage = {
        panel_keys["control"]: 50,
        panel_keys["treatment"]: 50,
        panel_keys["budget_peer"]: 46,
        stored_l6: 49 if stored_l6 else None,
    }
    for key in panel_keys.values():
        if key is None:
            continue
        row = next(row for row in rows if row["key"] == key)
        endpoint = panel["endpoints"][key]
        require(
            (row["width"], row["depth"], row["parameters"])
            == (endpoint["width"], endpoint["depth"], endpoint["parameter_count"]),
            f"ladder row {key}: architecture",
        )
        close(row["mean_validation_loss"], endpoint["mean_loss"], f"ladder row {key}")
        require(
            row["fresh_panel_seed"] == panel["seed"], f"ladder row {key}: panel seed"
        )
        require(
            row["source_stage"] == expected_stage[key],
            f"ladder row {key}: source stage",
        )
    substitution = ladder["budget_substitution"]
    require(
        substitution is not None, "ladder must score the depth-versus-width spend"
    )
    direct = panel["comparisons"].get("c256_l8_minus_c320_l6")
    if direct is None:
        direct = {
            "mean_delta": -panel["comparisons"]["c320_l6_minus_c256_l8"]["mean_delta"],
            "confidence_low": -panel["comparisons"]["c320_l6_minus_c256_l8"][
                "confidence_high"
            ],
            "confidence_high": -panel["comparisons"]["c320_l6_minus_c256_l8"][
                "confidence_low"
            ],
        }
    close(substitution["mean_delta"], direct["mean_delta"], "ladder substitution mean")
    close(
        substitution["confidence_low"],
        direct["confidence_low"],
        "ladder substitution lower",
    )
    close(
        substitution["confidence_high"],
        direct["confidence_high"],
        "ladder substitution upper",
    )
    prior = ladder["prior_depth_reference"]
    stage_number, delta, error, control, treatment = PRIOR_DEPTH_REFERENCE
    require(
        (prior["source_stage"], prior["mean_delta"], prior["standard_error"])
        == (stage_number, delta, error),
        "the transcribed prior depth reference changed",
    )
    require(
        (prior["control_parameter_count"], prior["treatment_parameter_count"])
        == (control, treatment),
        "the transcribed prior depth reference changed its parameter counts",
    )
    require(
        prior["confidence_low"] <= prior["mean_delta"] <= prior["confidence_high"],
        "prior depth reference delta outside its interval",
    )
    # The device-shift row travels with the ladder, and it must be the same
    # measurement the panel reports rather than a second, independent claim.
    shift = ladder["device_shift"]
    if stored_l6 is None:
        require(shift is None, "a single-device ladder must not carry a device shift")
        require(
            ladder["device_confounded_endpoints"] == [],
            "a single-device ladder must not report confounded endpoints",
        )
    else:
        require(shift is not None, "the ladder dropped its measured device shift")
        forward = f"c256_l6_minus_{stored_l6}"
        if forward in panel["comparisons"]:
            expected_shift = panel["comparisons"][forward]["mean_delta"]
        else:
            expected_shift = -panel["comparisons"][
                f"{stored_l6}_minus_c256_l6"
            ]["mean_delta"]
        close(shift["mean_delta"], expected_shift, "ladder device shift")
        require(
            ladder["device_confounded_endpoints"]
            == sorted(panel["device_confounded_endpoints"]),
            "ladder and panel disagree about which endpoints are confounded",
        )
    return {
        "rows": len(rows),
        "measured_depth_step": measured["mean_delta"],
        "budget_substitution": substitution["mean_delta"],
        "device_shift": shift,
        "prior_reference": (stage_number, delta),
    }


def audit_run(report_path: Path, data_path: Path | None) -> dict[str, Any]:
    require(report_path.is_file(), f"completed report not present: {report_path}")
    report_hash = file_sha256(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(
        report["stage"] == 50 and report["training_executed"] is True,
        "report must describe a completed Stage 50 training run",
    )
    require(report["protocol"] == "paired_architecture_native_depth", "protocol")
    require(report["updates_per_branch"] == 50000, "update budget")
    require(report["target_presentations_per_branch"] == 102400000, "token budget")
    require(report["learning_rate_schedule"] == SCHEDULE, "LR schedule")
    require(report["paired_batch_streams_verified"] is True, "paired verification flag")
    require(
        report["parameter_counts"]
        == {
            "control": C256_L6_NATIVE_INITIALIZATION[0],
            "treatment": C256_L8_NATIVE_INITIALIZATION[0],
            "added": C256_L8_NATIVE_INITIALIZATION[0]
            - C256_L6_NATIVE_INITIALIZATION[0],
            "added_fraction": C256_L8_NATIVE_INITIALIZATION[0]
            / C256_L6_NATIVE_INITIALIZATION[0]
            - 1.0,
            "stored_c320_peer": C320_NATIVE_INITIALIZATION[0],
        },
        "parameter counts changed",
    )
    require(
        report["runtime"]["lineage_device_required"] == LINEAGE_DEVICE_NAME,
        "lineage device requirement",
    )
    require(
        report["control_initialization_equivalence"]["identical"] is True
        and report["control_initialization_equivalence"]["state_sha256"]
        == C256_L6_NATIVE_INITIALIZATION[1]
        and report["control_initialization_equivalence"]["stage_46_factory_identical"]
        is True
        and report["control_initialization_equivalence"]["stage_30_factory_identical"]
        is True,
        "control initialization equivalence",
    )
    require(
        report["treatment_initialization_equivalence"]["identical"] is True
        and report["treatment_initialization_equivalence"]["state_sha256"]
        == C256_L8_NATIVE_INITIALIZATION[1]
        and report["treatment_initialization_equivalence"][
            "stage_30_factory_identical"
        ]
        is True,
        "treatment initialization equivalence",
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
    for key, depth in (("control", CONTROL_DEPTH), ("treatment", TREATMENT_DEPTH)):
        branch = report[key]
        topology = CONTROL_TOPOLOGY if depth == CONTROL_DEPTH else TREATMENT_TOPOLOGY
        require(
            branch["spec"]["key"] == key
            and branch["spec"]["width"] == 256
            and branch["spec"]["depth"] == depth
            and tuple(branch["spec"]["layer_contexts"]) == topology
            and branch["spec"]["dropout"] == 0.15,
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
        expected = (
            C256_L6_NATIVE_INITIALIZATION
            if depth == CONTROL_DEPTH
            else C256_L8_NATIVE_INITIALIZATION
        )
        require(
            branch["initialization"]["depth"] == depth
            and tuple(branch["initialization"]["layer_contexts"]) == topology
            and branch["initialization"]["residual_dropout"] == 0.15,
            f"{key}: initialization architecture",
        )
        require(
            (
                branch["initialization"]["parameter_count"],
                branch["initialization"]["state_sha256"],
            )
            == expected,
            f"{key}: native initialization changed",
        )
        require(
            branch["final_streams"]["training_generator_sha256"] == sampler_terminal,
            f"{key}: terminal batch stream disagrees with the replayed sampler",
        )
    control, treatment = report["control"], report["treatment"]
    require(
        control["initial_streams"]["model_sha256"]
        != treatment["initial_streams"]["model_sha256"],
        "fresh depth branches started from identical weights",
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
        control["final_streams"]["training_generator_sha256"]
        == treatment["final_streams"]["training_generator_sha256"],
        "fresh branches consumed different batches",
    )
    require(
        control["final_streams"]["torch_rng_sha256"]
        != treatment["final_streams"]["torch_rng_sha256"],
        "the deeper branch did not consume a different dropout schedule",
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
    require(fresh["seed"] == FRESH_PANEL_SEED, "fresh paired panel must use its own seed")
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
        "L=8 improves C256 validation at matched 50k updates on the fresh paired panel"
        if fresh_delta["sign"] == "candidate_better"
        else "L=6 beats L=8 at C256 on the fresh paired panel"
        if fresh_delta["sign"] == "baseline_better"
        else "unresolved within paired panel uncertainty; this does not prove equivalence"
    )
    require(report["outcome"] == expected_outcome, "outcome disagrees with its interval")

    panel = audit_budget_panel(
        report["budget_shootout"], BUDGET_PANEL_SEED, report["budget_outcome"]
    )
    endpoints = audit_endpoint_checkpoints(torch, stage, data, report["device_mode"])
    for key, entry in endpoints.items():
        recorded = report["budget_shootout"]["endpoints"][key]
        require(
            recorded["checkpoint_sha256"] == entry["file_sha256"],
            f"{key}: report hash disagrees with the stored endpoint",
        )
        require(
            recorded["checkpoint_step"] == 50000,
            f"{key}: report endpoint is not terminal",
        )
    shared = audit_shared_prefix(torch, stage, data, report)
    reproduction = report["stage_49_control_reproduction"]
    reference = stage.load_stage_49_control(data, stage.DEFAULT_OUTPUT_DIRECTORY)
    recorded_peers = report["stored_peers"]
    require(
        recorded_peers["stage_49_control"]["file_sha256"] == STAGE_49_CONTROL_SHA256
        and Path(recorded_peers["stage_49_control"]["path"]).resolve()
        == reference.path.resolve()
        and recorded_peers["stage_49_control"]["parameter_count"]
        == C256_L6_NATIVE_INITIALIZATION[0]
        and recorded_peers["stage_49_control"]["best_step"]
        == reference.payload["best_step"]
        and recorded_peers["stage_49_control"]["depth"] == CONTROL_DEPTH
        and recorded_peers["c320_budget_peer"]["file_sha256"]
        == STAGE_46_C320_P015_SHA256
        and recorded_peers["c320_budget_peer"]["parameter_count"]
        == C320_NATIVE_INITIALIZATION[0]
        and recorded_peers["c320_budget_peer"]["width"] == 320
        and recorded_peers["c320_budget_peer"]["depth"] == CONTROL_DEPTH,
        "preflight stored-peer record changed",
    )
    require(
        file_sha256(reference.path) == STAGE_49_CONTROL_SHA256,
        "Stage 49 C256 p=0.15 champion file hash changed",
    )
    mode = report["device_mode"]
    require(
        mode in (CANONICAL_DEVICE_MODE, FOREIGN_DEVICE_MODE), "device mode changed"
    )
    require(
        report["runtime"]["foreign_device_authorized"] is (mode == FOREIGN_DEVICE_MODE),
        "the device mode disagrees with the recorded authorization",
    )
    if mode == CANONICAL_DEVICE_MODE:
        require(
            reproduction.get("identical") is True
            and reproduction["reference_sha256"] == STAGE_49_CONTROL_SHA256
            and Path(reproduction["reference_path"]).resolve()
            == reference.path.resolve(),
            "Stage 49 control reproduction claim is not authenticated",
        )
    else:
        # Off the lineage device the stage must not claim reproduction, and the
        # weaker fixed-panel readout it does claim has to be re-derivable.
        require(
            reproduction["attempted"] is False
            and reproduction["device_mode"] == FOREIGN_DEVICE_MODE
            and reproduction["reference_sha256"] == STAGE_49_CONTROL_SHA256
            and Path(reproduction["reference_path"]).resolve()
            == reference.path.resolve(),
            "a foreign-device run must not claim control reproduction",
        )
        fixed = reproduction["fixed_panel"]
        close(
            fixed["trained_validation"],
            control["training"]["final"]["val"],
            "foreign control fixed-panel validation",
        )
        close(
            fixed["trained_train"],
            control["training"]["final"]["train"],
            "foreign control fixed-panel train",
        )
        close(
            fixed["reference_validation"],
            reference.payload["snapshot_losses"]["validation"],
            "stored champion fixed-panel validation",
        )
        close(
            fixed["reference_train"],
            reference.payload["snapshot_losses"]["train"],
            "stored champion fixed-panel train",
        )
        close(
            fixed["delta_validation_foreign_minus_lineage"],
            fixed["trained_validation"] - fixed["reference_validation"],
            "foreign control fixed-panel delta",
        )
        require(
            reproduction["training_device_name"] != LINEAGE_DEVICE_NAME
            and reproduction["reference_training_device_name"] == LINEAGE_DEVICE_NAME,
            "a foreign-device run must record both devices",
        )
        require(
            control["final_streams"]["model_sha256"]
            != stage._STAGE_41.tensor_mapping_sha256(
                reference.payload["model_state_dict"]
            ),
            "a foreign-device control cannot hold the lineage champion's weights",
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
    if mode == CANONICAL_DEVICE_MODE:
        for key, (trained, stored) in independent.items():
            require(
                trained == stored,
                f"trained L=6 control is not the Stage 49 champion in {key}",
            )
            recorded = reproduction["state_sha256"][key]
            require(
                list(recorded) == [trained, stored],
                f"control reproduction record disagrees with the stored champion in {key}",
            )
    else:
        # The sampler is device-independent, so the batch stream must still agree
        # even where the weights cannot.
        require(
            independent["training_generator_sha256"][0]
            == independent["training_generator_sha256"][1],
            "a foreign-device control did not consume the recorded batch stream",
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
    if mode == CANONICAL_DEVICE_MODE:
        require(
            ours["training_device_name"] == theirs["training_device_name"],
            "control reproduction trained on a different device than the champion",
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
                f"control reproduction record disagrees with the stored champion in {key}",
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
            tuple(reproduction["architecture_fields_checked"])
            == stage.SHARED_ARCHITECTURE_FIELDS,
            "control reproduction compared a different architecture field set",
        )
        require(
            control["best_step"] == reference.payload["best_step"],
            "trained control selected a different best step than the Stage 49 champion",
        )
    else:
        require(
            ours["training_device_name"] != theirs["training_device_name"],
            "a foreign-device control must record a different training device",
        )
        require(
            reproduction["training_device_name"] == ours["training_device_name"]
            and reproduction["reference_training_device_name"]
            == theirs["training_device_name"],
            "foreign-device control record names the wrong devices",
        )
        require(
            reproduction["reference_best_step"] == reference.payload["best_step"]
            and reproduction["same_selected_best_step"]
            is (reproduction["best_step"] == reference.payload["best_step"])
            and reproduction["parameter_delta"] == 0,
            "foreign-device control record is internally inconsistent",
        )
    if mode == CANONICAL_DEVICE_MODE:
        require(
            reproduction["rng_capture"]["training_device_name"] == LINEAGE_DEVICE_NAME,
            "control reproduction did not train on the recorded lineage device",
        )
    else:
        require(
            "rng_capture" not in reproduction,
            "a foreign-device run must not record a reproduction RNG capture",
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
        "budget_panel": panel,
        "endpoint_authentication": endpoints,
        "shared_initialization": shared,
        "depth_budget_ladder": ladder,
        "device_mode": mode,
        "device_shift": panel["device_shift"],
        "stage_49_control_reproduction": (
            {
                "identical": reproduction["identical"],
                "fields_checked": len(reproduction["fields_checked"]),
                "reference_sha256": reproduction["reference_sha256"],
            }
            if mode == CANONICAL_DEVICE_MODE
            else {
                "attempted": False,
                "training_device_name": reproduction["training_device_name"],
                "reference_training_device_name": reproduction[
                    "reference_training_device_name"
                ],
                "reference_sha256": reproduction["reference_sha256"],
            }
        ),
        "launch_manifest": manifest,
        "outcome": expected_outcome,
        "limits": "Stored batch losses were recomputed statistically; model forward "
        "losses were not rerun. Terminal batch streams are replayed from the seed "
        "and corpus, native initializations are rebuilt on the CPU, and each "
        "checkpoint payload is re-hashed from disk.",
    }


def audit_budget_only(path: Path) -> dict[str, Any]:
    """Audit the standalone budget panel without touching the runner."""

    require(path.is_file(), f"budget report not present: {path}")
    panel = json.loads(path.read_text(encoding="utf-8"))
    audit = audit_budget_panel(panel, BUDGET_PANEL_SEED)
    require(
        panel["uncertainty_scope"].startswith("Student-t 95% panel-sampling interval"),
        "budget report must state its uncertainty scope",
    )
    return {
        "status": "passed",
        "budget_report_path": str(path.resolve()),
        "budget_report_sha256": file_sha256(path),
        "budget_panel": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=PROJECT / "checkpoints" / "stage_50_report.json"
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--budget-report",
        type=Path,
        default=PROJECT / "checkpoints" / "stage_50_budget_shootout.json",
        help="standalone budget panel written by --budget-panel-only",
    )
    parser.add_argument(
        "--statistics-only",
        action="store_true",
        help="audit the report's 500-batch paired panels without importing Torch",
    )
    parser.add_argument(
        "--budget-only",
        action="store_true",
        help="audit only the standalone budget panel report",
    )
    args = parser.parse_args()
    if args.budget_only:
        result = audit_budget_only(args.budget_report)
    elif args.statistics_only:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        panels = {
            "fresh_50k_branches": report["fresh_50k_branches"],
            "budget_shootout": report["budget_shootout"],
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
            "budget_panel": audit_budget_panel(panels["budget_shootout"], BUDGET_PANEL_SEED),
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
