from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

from data_utils import CharacterData, CharacterVocabulary
from training import LossEstimate


SPEC = importlib.util.spec_from_file_location(
    "stage_49_under_test", Path(__file__).with_name("49_dropout_strength.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage49Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, width: int = 16, **overrides):
        vocabulary = CharacterVocabulary.from_text("abcde")
        tokens = torch.arange(2400) % vocabulary.size
        data = CharacterData(vocabulary, tokens[:1800], tokens[1800:], 2400)
        settings = dict(
            batch_size=2,
            block_size=24,
            n_embd=width,
            max_iters=3,
            eval_iters=1,
            eval_interval=1,
        )
        settings.update(overrides)
        return data, replace(STAGE.canonical_config(), **settings), torch.device("cpu")

    def spec(self, directory: Path, key: str, width: int, dropout: float):
        return STAGE.BranchSpec(
            key,
            width,
            dropout,
            directory / f"{key}_best.pt",
            directory / f"{key}_final.pt",
        )

    def run_pair(self, directory: Path, width: int = 16):
        data, config, device = self.fixture(width)
        control_spec = self.spec(
            directory, "control", config.n_embd, STAGE.DEFAULT_CONTROL_DROPOUT
        )
        treatment_spec = self.spec(
            directory, "treatment", config.n_embd, STAGE.DEFAULT_TREATMENT_DROPOUT
        )
        with contextlib.redirect_stdout(io.StringIO()):
            control = STAGE.run_branch(control_spec, data, config, device)
            treatment = STAGE.run_branch(treatment_spec, data, config, device)
        return control, treatment, data, config, device

    def test_protocol_locks_every_setting_except_dropout(self):
        control = STAGE.canonical_config(STAGE.DEFAULT_CONTROL_DROPOUT)
        treatment = STAGE.canonical_config(STAGE.DEFAULT_TREATMENT_DROPOUT)
        self.assertEqual(asdict(control), asdict(treatment))
        self.assertEqual(
            (control.batch_size, control.block_size), (8, 256)
        )
        self.assertEqual(
            (control.n_embd, STAGE.N_HEAD, STAGE.N_LAYER), (256, 4, 6)
        )
        self.assertEqual((control.max_iters, control.learning_rate), (50000, 1e-3))
        self.assertEqual((control.eval_interval, control.eval_iters), (500, 100))
        self.assertEqual(
            control.batch_size * control.block_size * control.max_iters, 102_400_000
        )
        self.assertEqual(
            (STAGE.DEFAULT_WIDTH, STAGE.DEFAULT_CONTROL_DROPOUT),
            (256, 0.15),
        )
        self.assertEqual(STAGE.DEFAULT_TREATMENT_DROPOUT, 0.20)
        schedule = STAGE.make_schedule()
        for step, expected in (
            (0, 1e-3),
            (9999, 1e-3),
            (10000, 3e-4),
            (13000, 1e-4),
            (17999, 1e-4),
            (18000, 3e-5),
            (49999, 3e-5),
        ):
            self.assertEqual(schedule.for_update(step), expected)
        # Stage 48's p=0.10 is no longer a canonical setting: this stage steps up
        # from the p=0.15 it resolved, so the old value must now be rejected.
        for dropout in (0.0, 0.05, 0.10, 0.125, 0.25, 0.30):
            with self.assertRaises(ValueError):
                STAGE.canonical_config(dropout)

    def test_branches_differ_only_in_their_residual_dropout(self):
        data, config, _ = self.fixture()
        metadata = {
            dropout: STAGE.dropout_architecture_metadata(data, config, dropout)
            for dropout in (STAGE.DEFAULT_CONTROL_DROPOUT, STAGE.DEFAULT_TREATMENT_DROPOUT)
        }
        control, treatment = (
            metadata[STAGE.DEFAULT_CONTROL_DROPOUT],
            metadata[STAGE.DEFAULT_TREATMENT_DROPOUT],
        )
        self.assertEqual(
            {key: value for key, value in control.items() if key != "residual_dropout"},
            {key: value for key, value in treatment.items() if key != "residual_dropout"},
        )
        self.assertEqual(
            (control["residual_dropout"], treatment["residual_dropout"]), (0.15, 0.20)
        )
        canonical = replace(
            config,
            n_embd=STAGE.DEFAULT_WIDTH,
            block_size=256,
            batch_size=8,
        )
        canonical_metadata = STAGE.dropout_architecture_metadata(
            data, canonical, STAGE.DEFAULT_CONTROL_DROPOUT
        )
        self.assertEqual(
            (
                canonical_metadata["n_embd"],
                canonical_metadata["n_head"],
                canonical_metadata["n_layer"],
                canonical_metadata["feed_forward_width"],
                canonical_metadata["topology"],
                canonical_metadata["head_size"],
                canonical_metadata["block_size"],
            ),
            (256, 4, 6, 1024, "LLLLGG", 64, 256),
        )
        counts = set()
        for dropout in (0.15, 0.20):
            model = STAGE.build_model(data, config, STAGE.N_HEAD, STAGE.N_LAYER, dropout)
            counts.add(sum(parameter.numel() for parameter in model.parameters()))
            del model
        self.assertEqual(len(counts), 1)

    def test_native_initialization_is_reproducible_and_dropout_invariant(self):
        data, config, device = self.fixture()
        torch.manual_seed(13)
        control, _, generator, first = STAGE.initialize_branch(
            data, config, STAGE.DEFAULT_CONTROL_DROPOUT, device
        )
        torch.manual_seed(99)
        treatment, _, other_generator, second = STAGE.initialize_branch(
            data, config, STAGE.DEFAULT_TREATMENT_DROPOUT, device
        )
        self.assertEqual(first["kind"], "fresh_architecture_native")
        self.assertEqual((first["origin_stage"], first["step"]), (49, 0))
        self.assertEqual(first["residual_dropout"], STAGE.DEFAULT_CONTROL_DROPOUT)
        self.assertEqual(second["residual_dropout"], STAGE.DEFAULT_TREATMENT_DROPOUT)
        for info in (first, second):
            self.assertFalse(info["historical_weights_loaded"])
            self.assertFalse(info["historical_optimizer_loaded"])
            self.assertEqual(info["optimizer_state_entries_at_initialization"], 0)
        self.assertEqual(first["state_sha256"], second["state_sha256"])
        self.assertEqual(first["parameter_count"], second["parameter_count"])
        for key, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, treatment.state_dict()[key]), key)
        self.assertTrue(
            torch.equal(generator.get_state(), other_generator.get_state())
        )

    def test_native_initialization_is_only_defined_at_the_canonical_width(self):
        data, config, device = self.fixture()
        _, _, _, initialization = STAGE.initialize_branch(
            data, config, STAGE.DEFAULT_CONTROL_DROPOUT, device
        )
        with self.assertRaisesRegex(ValueError, "C256"):
            STAGE.verify_native_initialization(data, config, initialization)
        canonical = replace(config, n_embd=STAGE.DEFAULT_WIDTH)
        _, _, _, reference = STAGE._STAGE_46.initialize_branch(
            data, canonical, STAGE.DEFAULT_CONTROL_DROPOUT, torch.device("cpu")
        )
        self.assertEqual(
            STAGE._STAGE_46.verify_control_factory_equivalence(data, canonical),
            reference["state_sha256"],
        )
        # The fixture vocabulary is not Complete Shakespeare's, so the recorded
        # constant is exercised by pinning the constant itself.
        with patch.object(
            STAGE,
            "C256_NATIVE_INITIALIZATION_SHA256",
            reference["state_sha256"],
        ):
            self.assertEqual(
                STAGE.verify_native_initialization(data, canonical, reference),
                reference["state_sha256"],
            )
        with patch.object(
            STAGE, "C256_NATIVE_INITIALIZATION_SHA256", "0" * 64
        ):
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                STAGE.verify_native_initialization(data, canonical, reference)
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            STAGE.verify_native_initialization(
                data, canonical, {**reference, "state_sha256": "0" * 64}
            )
        self.assertEqual(
            STAGE.C256_NATIVE_INITIALIZATION_SHA256,
            "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        )

    def test_real_pair_starts_identical_and_saves_resumable_endpoints(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, _ = self.run_pair(root)
            STAGE.verify_paired_branches(control, treatment)
            self.assertEqual(
                tuple(record.step for record in control.training.history), (0, 1, 2, 3)
            )
            self.assertEqual(
                control.initial_streams["model_sha256"],
                treatment.initial_streams["model_sha256"],
            )
            self.assertEqual(
                control.final_streams["training_generator_sha256"],
                treatment.final_streams["training_generator_sha256"],
            )
            self.assertNotEqual(
                control.final_streams["model_sha256"],
                treatment.final_streams["model_sha256"],
            )
            for report in (control, treatment):
                final = STAGE.validate_branch_checkpoint(
                    report.spec.final_path, report.spec, data, config, kind="final"
                )
                self.assertEqual(final["step"], 3)
                self.assertEqual(
                    final["architecture"]["residual_dropout"], report.spec.dropout
                )
                self.assertEqual(
                    final["experiment"]["branch_residual_dropout"], report.spec.dropout
                )
                self.assertEqual(
                    {
                        int(state["step"])
                        for state in final["optimizer_state_dict"]["state"].values()
                    },
                    {3},
                )
                self.assertEqual(STAGE.file_hash(report.spec.final_path), report.final_sha256)
                self.assertEqual(STAGE.file_hash(report.spec.best_path), report.best_sha256)

    def test_checkpoint_rejects_structural_tampering_independently_of_its_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, _, data, config, _ = self.run_pair(root)
            original = torch.load(control.spec.final_path, weights_only=True)
            # Each of these fields is validated structurally, so the tamper must
            # be rejected even when the payload integrity hash is recomputed to
            # match the tampered content.
            structural = (
                lambda payload: payload.update(step=2),
                lambda payload: payload.update(best_step=4),
                lambda payload: payload.update(checkpoint_kind="best"),
                lambda payload: payload["optimizer_state_dict"]["param_groups"][
                    0
                ].update(lr=3e-5),
                lambda payload: payload["optimizer_state_dict"]["param_groups"][
                    0
                ].update(betas=(0.8, 0.99)),
                lambda payload: payload["optimizer_state_dict"]["param_groups"][
                    0
                ].update(weight_decay=0.0),
                lambda payload: payload["architecture"].update(residual_dropout=0.20),
                lambda payload: payload["experiment"].update(
                    branch_residual_dropout=0.20
                ),
                lambda payload: payload["experiment"].update(stage=48),
                lambda payload: payload["initialization"].update(residual_dropout=0.20),
                lambda payload: payload["initialization"].update(origin_stage=48),
                lambda payload: payload["initialization"].update(
                    historical_weights_loaded=True
                ),
                lambda payload: payload.update(optimizer_restart_step=1),
                lambda payload: payload.update(training_config={
                    **payload["training_config"], "eval_iters": 99
                }),
            )
            for mutate in structural:
                payload = copy.deepcopy(original)
                mutate(payload)
                payload["payload_sha256"] = STAGE.state_hash(
                    {key: value for key, value in payload.items() if key != "payload_sha256"}
                )
                path = root / "tampered.pt"
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    STAGE.validate_branch_checkpoint(
                        path, control.spec, data, config, kind="final"
                    )

    def test_checkpoint_rejects_content_tampering_through_its_integrity_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, _, data, config, _ = self.run_pair(root)
            original = torch.load(control.spec.final_path, weights_only=True)
            # These fields carry no independent structural check, so the stored
            # payload hash is the defense that must reject them.
            content = (
                lambda payload: payload["snapshot_losses"].update(train=9.0),
                lambda payload: payload.update(best_val_loss=9.0),
                lambda payload: payload.update(
                    training_generator_state=torch.Generator()
                    .manual_seed(999)
                    .get_state()
                ),
                lambda payload: next(iter(payload["model_state_dict"].values())).add_(1),
                lambda payload: payload.update(data_fingerprint="0" * 64),
            )
            for mutate in content:
                payload = copy.deepcopy(original)
                mutate(payload)
                path = root / "stale.pt"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "integrity"):
                    STAGE.validate_branch_checkpoint(
                        path, control.spec, data, config, kind="final"
                    )

    def test_paired_comparison_requires_the_exact_dropout_step_at_fixed_width(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            with contextlib.redirect_stdout(io.StringIO()):
                control = STAGE.run_branch(
                    self.spec(root, "control", config.n_embd, 0.15), data, config, device
                )
                wrong_step = STAGE.run_branch(
                    self.spec(root, "treatment", config.n_embd, 0.25), data, config, device
                )
            with self.assertRaisesRegex(RuntimeError, "0.20"):
                STAGE.verify_paired_branches(control, wrong_step)
            treatment = replace(wrong_step, spec=replace(wrong_step.spec, dropout=0.20))
            with self.assertRaisesRegex(RuntimeError, "hold width fixed"):
                STAGE.verify_paired_branches(
                    control, replace(treatment, spec=replace(treatment.spec, width=20))
                )
            with self.assertRaisesRegex(RuntimeError, "parameter count"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        initialization={
                            **treatment.initialization,
                            "parameter_count": control.initialization["parameter_count"]
                            + 1,
                        },
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "initial weights"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        initial_streams={**treatment.initial_streams, "model_sha256": "x"},
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "training batches"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        final_streams={
                            **treatment.final_streams,
                            "training_generator_sha256": "x",
                        },
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "budgets"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        training=replace(
                            treatment.training,
                            history=treatment.training.history[:-1],
                        ),
                    ),
                )

    def test_fresh_panel_is_shared_repeatable_and_seed_disjoint_from_earlier_stages(self):
        with tempfile.TemporaryDirectory() as raw:
            control, treatment, data, config, device = self.run_pair(Path(raw))
            torch.manual_seed(991)
            before = torch.get_rng_state().clone()
            first = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1366, eval_iters=5
            )
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            second = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1366, eval_iters=5
            )
            self.assertEqual(asdict(first), asdict(second))
            self.assertEqual(len(first.baseline.batch_losses), 5)
            self.assertEqual(len(first.candidate.batch_losses), 5)
            differences = [
                candidate - baseline
                for baseline, candidate in zip(
                    first.baseline.batch_losses, first.candidate.batch_losses
                )
            ]
            self.assertAlmostEqual(first.delta.mean_delta, sum(differences) / 5)
            # Every seed Stages 43-48 consumed, plus the fixed panel, is refused
            # by both panel constructors.
            for seed in (
                STAGE.DEFAULT_FIXED_PANEL_SEED,
                1354,
                1355,
                1356,
                1357,
                1358,
                1359,
                1360,
                1361,
                1362,
                1363,
                1364,
                1365,
            ):
                for evaluate in (
                    lambda s=seed: STAGE.evaluate_checkpoint_pair(
                        control, treatment, data, config, device, seed=s, eval_iters=5
                    ),
                    lambda s=seed: STAGE.evaluate_champion_shootout(
                        data, config, device, Path(raw), refs=(), specs=(),
                        expected_generator_sha256=None, seed=s, eval_iters=5
                    ),
                ):
                    with self.assertRaises(ValueError):
                        evaluate()
            for count in (0, 1):
                with self.assertRaises(ValueError):
                    STAGE.evaluate_checkpoint_pair(
                        control, treatment, data, config, device, seed=1366, eval_iters=count
                    )

    def test_champion_panel_ranks_pairs_and_mirrors_deltas(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, device = self.run_pair(root)
            specs = (control.spec, treatment.spec)
            report = STAGE.evaluate_champion_shootout(
                data,
                config,
                device,
                root,
                refs=(),
                specs=specs,
                expected_generator_sha256=None,
                seed=1367,
                eval_iters=5,
            )
            self.assertEqual(sorted(report["endpoints"]), ["c16_p015", "c16_p020"])
            self.assertEqual(len(report["comparisons"]), 1)
            self.assertTrue(report["shared_training_batch_stream_verified"])
            self.assertFalse(report["dropout_enabled"])
            self.assertEqual(
                [entry["rank"] for entry in report["ranking"]], [1, 2]
            )
            losses = [entry["mean_validation_loss"] for entry in report["ranking"]]
            self.assertEqual(losses, sorted(losses))
            self.assertEqual(len(report["final_generator_sha256"]), 64)
            champion = report["ranking"][0]["key"]
            runner_up = report["ranking"][1]["key"]
            stored = report["comparisons"][f"{champion}_minus_{runner_up}"]
            mirror = STAGE.compared_delta(report["comparisons"], runner_up, champion)
            self.assertAlmostEqual(mirror["mean_delta"], -stored["mean_delta"], places=15)
            self.assertAlmostEqual(
                mirror["confidence_low"], -stored["confidence_high"], places=15
            )
            self.assertAlmostEqual(
                mirror["confidence_high"], -stored["confidence_low"], places=15
            )
            self.assertEqual(mirror["standard_error"], stored["standard_error"])
            self.assertEqual((mirror["candidate"], mirror["baseline"]), (runner_up, champion))
            self.assertIn(report["ranking"][0]["label"], STAGE.champion_outcome(report))
            with self.assertRaises(KeyError):
                STAGE.compared_delta(report["comparisons"], "c16_p015", "absent")
            with self.assertRaisesRegex(ValueError, "lineage stream"):
                STAGE.evaluate_champion_shootout(
                    data,
                    config,
                    device,
                    root,
                    refs=(),
                    specs=specs,
                    expected_generator_sha256="0" * 64,
                    seed=1367,
                    eval_iters=5,
                )

    def test_champion_panel_rejects_a_mismatched_width_or_batch_stream(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            control = self.spec(root, "control", 16, 0.15)
            treatment = self.spec(root, "treatment", 16, 0.20)
            with contextlib.redirect_stdout(io.StringIO()):
                STAGE.run_branch(control, data, config, device)
                with patch.object(STAGE, "DEFAULT_TRAINING_BATCH_SEED", 4242):
                    STAGE.run_branch(treatment, data, config, device)
            with self.assertRaisesRegex(ValueError, "must match the evaluated width"):
                STAGE.champion_plan(
                    data,
                    replace(config, n_embd=20),
                    root,
                    (),
                    (control, treatment),
                )
            with self.assertRaisesRegex(RuntimeError, "training_generator_sha256"):
                STAGE.evaluate_champion_shootout(
                    data,
                    config,
                    device,
                    root,
                    refs=(),
                    specs=(control, treatment),
                    expected_generator_sha256=None,
                    seed=1367,
                    eval_iters=2,
                )

    def test_outcome_classification_reports_resolution_direction(self):
        def paired(mean: float, low: float, high: float):
            return types.SimpleNamespace(
                delta=types.SimpleNamespace(
                    mean_delta=mean, confidence_low=low, confidence_high=high
                )
            )

        # The treatment is the stronger regularizer, so a negative delta (higher
        # dropout validating better) is the improving direction.
        self.assertIn("p=0.20 improves", STAGE.paired_outcome(paired(-0.01, -0.02, -0.001)))
        self.assertIn("p=0.15 beats", STAGE.paired_outcome(paired(0.01, 0.001, 0.02)))
        self.assertIn("unresolved", STAGE.paired_outcome(paired(0.0, -0.01, 0.01)))
        # The direction this stage actually measured must classify as the weaker
        # regularizer winning, not as an improvement for the stronger one.
        self.assertNotIn(
            "improves", STAGE.paired_outcome(paired(0.01, 0.001, 0.02))
        )

        def delta(low: float, high: float) -> dict:
            return {
                "candidate": "candidate",
                "baseline": "baseline",
                "mean_delta": (low + high) / 2,
                "standard_error": (high - low) / 4,
                "confidence_low": low,
                "confidence_high": high,
                "critical_value": 2.0,
            }

        def panel(champion_delta: tuple[float, float], peer_delta: tuple[float, float]) -> dict:
            return {
                "comparisons": {
                    "c256_p015_minus_c256_p020": delta(*champion_delta),
                    "c256_p015_minus_c320_p015": delta(*peer_delta),
                    "c256_p015_minus_c320_p020": delta(*peer_delta),
                },
                "ranking": [
                    {"key": "c256_p015", "label": "C256 p=0.15"},
                    {"key": "c256_p020", "label": "C256 p=0.2"},
                    {"key": "c320_p015", "label": "C320 p=0.15"},
                    {"key": "c320_p020", "label": "C320 p=0.2"},
                ],
            }

        # Every peer interval lies entirely below the champion: resolved.
        resolved = STAGE.champion_outcome(
            panel((-0.03, -0.01), (-0.02, -0.005))
        )
        self.assertIn("resolved champion", resolved)
        self.assertIn("beats all 3 peers", resolved)
        self.assertIn("C256 p=0.15", resolved)
        # The C320 peers straddle zero: the champion leads but is not resolved
        # against them, and both are named.
        unresolved = STAGE.champion_outcome(panel((-0.03, -0.01), (-0.01, 0.01)))
        self.assertIn("does not resolve", unresolved)
        self.assertIn("C320 p=0.15", unresolved)
        self.assertIn("C320 p=0.2", unresolved)
        self.assertNotIn("C256 p=0.2", unresolved)

    def test_regularization_ladder_carries_the_measured_step_and_prior_steps(self):
        with tempfile.TemporaryDirectory() as raw:
            control, treatment, data, config, device = self.run_pair(Path(raw))
            measured = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1366, eval_iters=4
            )
            ladder = STAGE.capacity_regularization_ladder(measured)
            self.assertEqual(len(ladder), 4)
            self.assertEqual(
                [row["source_stage"] for row in ladder], [49, 48, 46, 47]
            )
            self.assertEqual([row["width"] for row in ladder], [256, 256, 320, 320])
            self.assertEqual(
                [row["dropout_step"] for row in ladder],
                [
                    "p=0.15 -> p=0.2",
                    "p=0.1 -> p=0.15",
                    "p=0.1 -> p=0.15",
                    "p=0.15 -> p=0.2",
                ],
            )
            self.assertEqual(ladder[0]["mean_delta"], measured.delta.mean_delta)
            self.assertEqual(ladder[0]["fresh_panel_seed"], 1366)
            # Stage 48's measured C256 first step is now a transcribed prior row,
            # and it must be carried with its own interval, not recomputed.
            self.assertEqual(ladder[1]["mean_delta"], -0.0015666978359222413)
            self.assertEqual(ladder[1]["standard_error"], 0.0006895126702513343)
            self.assertEqual(ladder[1]["fresh_panel_seed"], 1364)
            self.assertEqual(ladder[2]["mean_delta"], -0.0050533)
            self.assertEqual(ladder[2]["source_stage"], 46)
            self.assertEqual(ladder[2]["fresh_panel_seed"], 1360)
            self.assertEqual(ladder[3]["mean_delta"], 0.0000117)
            self.assertEqual(ladder[3]["fresh_panel_seed"], 1362)
            for row in ladder:
                self.assertLess(row["confidence_low"], row["mean_delta"])
                self.assertLess(row["mean_delta"], row["confidence_high"])
            ladder[1]["mean_delta"] = 999.0
            self.assertEqual(STAGE.PRIOR_DROPOUT_STEPS[0]["mean_delta"], -0.0015666978359222413)
            self.assertEqual(len(STAGE.PRIOR_DROPOUT_STEPS), 3)

    def test_branch_specs_and_artifact_paths_name_and_protect_stage_49_outputs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            specs = STAGE.branch_specs(root)
            STAGE.validate_branch_specs(specs)
            self.assertEqual(
                [(spec.key, spec.width, spec.dropout) for spec in specs],
                [("control", 256, 0.15), ("treatment", 256, 0.20)],
            )
            self.assertEqual(
                [spec.best_path.name for spec in specs],
                [
                    "stage_49_control_c256_p015_best_checkpoint.pt",
                    "stage_49_treatment_c256_p020_best_checkpoint.pt",
                ],
            )
            self.assertEqual(
                [spec.final_path.name for spec in specs],
                [
                    "stage_49_control_c256_p015_step_50000_final_checkpoint.pt",
                    "stage_49_treatment_c256_p020_step_50000_final_checkpoint.pt",
                ],
            )
            for changed in (
                (specs[1], specs[0]),
                (replace(specs[0], dropout=0.10), specs[1]),
                (specs[0], replace(specs[1], dropout=0.15)),
                (specs[0], replace(specs[1], width=320)),
            ):
                with self.assertRaises(ValueError):
                    STAGE.validate_branch_specs(changed)
            for dropout in (-0.1, 1.0, float("nan")):
                with self.assertRaises(ValueError):
                    replace(specs[0], dropout=dropout)
            with self.assertRaises(ValueError):
                replace(specs[0], key="other")
            with self.assertRaises(ValueError):
                replace(specs[0], width=True)
            outputs = (
                specs[0].best_path,
                specs[0].final_path,
                specs[1].best_path,
                specs[1].final_path,
            )
            STAGE.validate_artifact_paths((root / "corpus.txt",), outputs)
            with self.assertRaises(ValueError):
                STAGE.validate_artifact_paths((specs[0].best_path,), outputs)
            with self.assertRaises(ValueError):
                STAGE.validate_artifact_paths((), (specs[0].best_path, specs[0].best_path))
            temporary = specs[0].best_path.with_name(f".{specs[0].best_path.name}.tmp")
            temporary.write_bytes(b"partial")
            with self.assertRaises(FileExistsError):
                STAGE.validate_artifact_paths((), outputs)
            temporary.unlink()
            specs[0].best_path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                STAGE.validate_artifact_paths((), outputs)

    def test_protocol_rejects_a_changed_corpus_or_an_additional_variable(self):
        data, _, _ = self.fixture()
        configs = (
            STAGE.canonical_config(STAGE.DEFAULT_CONTROL_DROPOUT),
            STAGE.canonical_config(STAGE.DEFAULT_TREATMENT_DROPOUT),
        )
        with self.assertRaisesRegex(ValueError, "Shakespeare"):
            STAGE.validate_protocol(data, configs)
        with patch.object(
            STAGE,
            "fingerprint_data",
            return_value=STAGE._STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT,
        ):
            STAGE.validate_protocol(data, configs)
            for changed in (
                replace(configs[1], seed=configs[1].seed + 1),
                replace(configs[1], batch_size=16),
                replace(configs[1], n_embd=320),
                replace(configs[1], max_iters=49999),
                replace(configs[1], learning_rate=3e-5),
                replace(configs[1], eval_iters=99),
            ):
                with self.assertRaisesRegex(ValueError, "locks"):
                    STAGE.validate_protocol(data, (configs[0], changed))

    def test_stage_48_reference_is_authenticated_before_it_is_trusted(self):
        directory = STAGE.DEFAULT_OUTPUT_DIRECTORY
        data = STAGE.CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        specs = STAGE._STAGE_48.branch_specs(directory)
        reference_spec = next(spec for spec in specs if spec.key == "treatment")
        # The stage this one steps up from is pinned by name and by content, so a
        # rerun that changed either would invalidate the ladder row and the
        # control reproduction rather than silently comparing different models.
        self.assertEqual(
            reference_spec.final_path.name,
            STAGE.DEFAULT_STAGE_48_TREATMENT_FILENAME,
        )
        self.assertEqual(
            STAGE.file_hash(reference_spec.final_path),
            STAGE.DEFAULT_STAGE_48_TREATMENT_SHA256,
        )
        self.assertEqual(
            STAGE.file_hash(reference_spec.best_path),
            STAGE.DEFAULT_STAGE_48_TREATMENT_BEST_SHA256,
        )
        endpoint = STAGE.load_stage_48_treatment(data, directory)
        self.assertEqual(
            (endpoint.key, endpoint.width, endpoint.dropout), ("c256_stage_48", 256, 0.15)
        )
        self.assertEqual(
            (endpoint.payload["step"], endpoint.payload["best_step"]), (50000, 48500)
        )
        with patch.object(STAGE, "DEFAULT_STAGE_48_TREATMENT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "hash changed"):
                STAGE.load_stage_48_treatment(data, directory)
        with patch.object(STAGE, "DEFAULT_STAGE_48_TREATMENT_BEST_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "best checkpoint hash changed"):
                STAGE.load_stage_48_treatment(data, directory)
        with patch.object(
            STAGE,
            "DEFAULT_STAGE_48_TREATMENT_FIXED_PANEL_VALIDATION_LOSS",
            1.0,
        ):
            with self.assertRaisesRegex(ValueError, "fixed-panel record changed"):
                STAGE.load_stage_48_treatment(data, directory)
        with patch.object(STAGE, "DEFAULT_STAGE_48_TREATMENT_FILENAME", "other.pt"):
            with self.assertRaisesRegex(ValueError, "changed its name"):
                STAGE.load_stage_48_treatment(data, directory)

    def test_lineage_device_resolution_prefers_the_recorded_accelerator(self):
        self.assertEqual(STAGE.DEFAULT_LINEAGE_DEVICE_NAME, "NVIDIA GeForce RTX 3090")
        self.assertEqual(STAGE.resolve_lineage_device("cpu"), torch.device("cpu"))
        STAGE.verify_lineage_device(torch.device("cpu"))
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this host")
        available = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        resolved = STAGE.resolve_lineage_device("cuda")
        if STAGE.DEFAULT_LINEAGE_DEVICE_NAME not in available:
            with self.assertRaisesRegex(RuntimeError, "no device named"):
                STAGE.resolve_lineage_device("cuda")
            return
        self.assertEqual(resolved.type, "cuda")
        self.assertEqual(
            torch.cuda.get_device_name(resolved), STAGE.DEFAULT_LINEAGE_DEVICE_NAME
        )
        STAGE.verify_lineage_device(resolved)
        recorded = available.index(STAGE.DEFAULT_LINEAGE_DEVICE_NAME)
        self.assertEqual(resolved.index, recorded)
        other = next(
            (
                index
                for index, name in enumerate(available)
                if name != STAGE.DEFAULT_LINEAGE_DEVICE_NAME
            ),
            None,
        )
        if other is not None:
            self.assertEqual(
                STAGE.resolve_lineage_device(f"cuda:{other}"), torch.device("cuda", other)
            )
            with self.assertRaisesRegex(RuntimeError, "requires"):
                STAGE.verify_lineage_device(torch.device("cuda", other))
        with self.assertRaisesRegex(ValueError, "not visible"):
            STAGE.resolve_lineage_device("cuda:99")
        with self.assertRaisesRegex(ValueError, "malformed"):
            STAGE.resolve_lineage_device("cuda:abc")

    def test_no_improvement_retains_the_native_initialization(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            spec = self.spec(root, "control", 16, STAGE.DEFAULT_CONTROL_DROPOUT)
            initialized, _, _, _ = STAGE.initialize_branch(
                data, config, STAGE.DEFAULT_CONTROL_DROPOUT, device
            )
            initial_weights = {
                key: value.clone() for key, value in initialized.state_dict().items()
            }
            with (
                patch.object(
                    STAGE,
                    "evaluate_on_fixed_batches",
                    side_effect=[LossEstimate(1, 1)] + [LossEstimate(2, 2)] * 3,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = STAGE.run_branch(spec, data, config, device)
            best = torch.load(spec.best_path, weights_only=True)
            final = torch.load(spec.final_path, weights_only=True)
            self.assertEqual((report.best_step, best["step"], final["step"]), (0, 0, 3))
            self.assertFalse(best["optimizer_state_dict"]["state"])
            for key, value in initial_weights.items():
                self.assertTrue(torch.equal(value, best["model_state_dict"][key]), key)


if __name__ == "__main__":
    unittest.main()
