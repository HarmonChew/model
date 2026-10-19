from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

from data_utils import CharacterData, CharacterVocabulary
from training import LossEstimate


SPEC = importlib.util.spec_from_file_location(
    "stage_47_under_test", Path(__file__).with_name("47_dropout_strength.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage47Tests(unittest.TestCase):
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

    def run_pair(self, directory: Path):
        data, config, device = self.fixture()
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

    def test_protocol_changes_only_dropout_and_preserves_the_established_schedule(self):
        control = STAGE.canonical_config(0.15)
        treatment = STAGE.canonical_config(0.20)
        self.assertEqual(asdict(control), asdict(treatment))
        self.assertEqual((control.batch_size, control.block_size), (8, 256))
        self.assertEqual((control.n_embd, STAGE.N_HEAD, STAGE.N_LAYER), (320, 4, 6))
        self.assertEqual(control.max_iters, 50000)
        self.assertEqual((control.eval_interval, control.eval_iters), (500, 100))
        self.assertEqual(
            control.batch_size * control.block_size * control.max_iters, 102400000
        )
        self.assertEqual(control.learning_rate, 1e-3)
        self.assertEqual((STAGE.DEFAULT_WIDTH, STAGE.DEFAULT_TREATMENT_DROPOUT), (320, 0.20))
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
        for dropout in (0.0, 0.05, 0.10, 0.125, 0.25):
            with self.assertRaises(ValueError):
                STAGE.canonical_config(dropout)

    def test_native_initialization_is_reproducible_and_dropout_invariant(self):
        data, config, device = self.fixture()
        torch.manual_seed(13)
        first, first_optimizer, first_batches, first_info = STAGE.initialize_branch(
            data, config, STAGE.DEFAULT_CONTROL_DROPOUT, device
        )
        torch.manual_seed(99)
        second, second_optimizer, second_batches, second_info = STAGE.initialize_branch(
            data, config, STAGE.DEFAULT_CONTROL_DROPOUT, device
        )
        self.assertEqual(first_info, second_info)
        self.assertEqual(first_info["kind"], "fresh_architecture_native")
        self.assertEqual((first_info["origin_stage"], first_info["step"]), (47, 0))
        self.assertEqual(first_info["residual_dropout"], STAGE.DEFAULT_CONTROL_DROPOUT)
        self.assertFalse(first_info["historical_weights_loaded"])
        self.assertFalse(first_info["historical_optimizer_loaded"])
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        self.assertEqual(first_optimizer.state_dict(), second_optimizer.state_dict())
        self.assertEqual(len(first_optimizer.state), 0)
        self.assertIsInstance(first_optimizer, torch.optim.AdamW)
        self.assertEqual(first_optimizer.param_groups[0]["lr"], 1e-3)
        self.assertTrue(
            torch.equal(first_batches.get_state(), second_batches.get_state())
        )

        # Residual dropout adds no parameters and consumes no initialization
        # RNG, so a stronger probability must not move a single initial weight.
        higher, higher_optimizer, higher_batches, higher_info = STAGE.initialize_branch(
            data, config, STAGE.DEFAULT_TREATMENT_DROPOUT, device
        )
        self.assertEqual(higher_info["state_sha256"], first_info["state_sha256"])
        self.assertEqual(
            higher_info["parameter_count"], first_info["parameter_count"]
        )
        self.assertEqual(len(higher_optimizer.state), 0)
        self.assertEqual(higher_info["residual_dropout"], STAGE.DEFAULT_TREATMENT_DROPOUT)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, higher.state_dict()[key]), key)
        self.assertTrue(
            torch.equal(first_batches.get_state(), higher_batches.get_state())
        )
        self.assertEqual(
            [type(module).__name__ for module in higher.modules()].count("Dropout"),
            [type(module).__name__ for module in first.modules()].count("Dropout"),
        )
        for module in higher.modules():
            if isinstance(module, torch.nn.Dropout) and module.p != 0:
                self.assertEqual(module.p, STAGE.DEFAULT_TREATMENT_DROPOUT)

    def test_native_initialization_rebuilds_the_stage_46_weights(self):
        data, config, device = self.fixture()
        # Stage 47 introduces no architecture of its own: it re-issues the
        # Stage 46 factory, whose exposed-dropout construction was already
        # shown to reproduce Stage 45's p=0.10 model.
        self.assertIs(STAGE.build_model, STAGE._STAGE_46.build_model)
        self.assertEqual(
            STAGE.canonical_config(STAGE.DEFAULT_CONTROL_DROPOUT),
            STAGE._STAGE_46.canonical_config(STAGE.DEFAULT_CONTROL_DROPOUT),
        )
        self.assertEqual(
            STAGE.STAGE_46_NATIVE_INITIALIZATION_SHA256,
            "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
        )
        _, _, _, initialization = STAGE.initialize_branch(
            data, config, STAGE.DEFAULT_CONTROL_DROPOUT, device
        )
        _, _, _, reference = STAGE._STAGE_46.initialize_branch(
            data, config, STAGE.DEFAULT_CONTROL_DROPOUT, device
        )
        self.assertEqual(
            initialization["state_sha256"], reference["state_sha256"]
        )
        self.assertEqual(
            STAGE.verify_native_initialization(data, config, initialization),
            initialization["state_sha256"],
        )
        # The recorded-hash branch is only reachable at the canonical width; it
        # must reject a control whose native weights moved.
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            STAGE.verify_native_initialization(
                data,
                config,
                {
                    **initialization,
                    "state_sha256": "0" * 64,
                },
            )
        # At the canonical width the recorded hash is also required, so a
        # stale constant must be rejected even when both factories agree.
        canonical = replace(config, n_embd=STAGE.DEFAULT_WIDTH)
        _, _, _, canonical_reference = STAGE._STAGE_46.initialize_branch(
            data, canonical, STAGE.DEFAULT_CONTROL_DROPOUT, device
        )
        # The fixture vocabulary differs from Complete Shakespeare's, so the
        # recorded constant is exercised by pinning the constant itself.
        with patch.object(
            STAGE,
            "STAGE_46_NATIVE_INITIALIZATION_SHA256",
            canonical_reference["state_sha256"],
        ):
            self.assertEqual(
                STAGE.verify_native_initialization(data, canonical, canonical_reference),
                canonical_reference["state_sha256"],
            )
        with patch.object(
            STAGE, "STAGE_46_NATIVE_INITIALIZATION_SHA256", "0" * 64
        ):
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                STAGE.verify_native_initialization(data, canonical, canonical_reference)

    def test_real_dropout_pair_starts_at_zero_and_saves_resumable_endpoints(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original_load = torch.load
            loaded_paths = []

            def record_load(path, *args, **kwargs):
                loaded_paths.append(Path(path).resolve())
                return original_load(path, *args, **kwargs)

            with patch.object(torch, "load", side_effect=record_load):
                control, treatment, data, config, device = self.run_pair(root)
            self.assertTrue(loaded_paths)
            self.assertTrue(all(path.parent == root.resolve() for path in loaded_paths))
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
                    report.spec.final_path,
                    report.spec,
                    data,
                    config,
                    kind="final",
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
                self.assertEqual(
                    STAGE.file_hash(report.spec.final_path), report.final_sha256
                )
                self.assertEqual(
                    STAGE.file_hash(report.spec.best_path), report.best_sha256
                )

            changed = {
                **treatment.final_streams,
                "training_generator_sha256": "changed",
            }
            with self.assertRaises(RuntimeError):
                STAGE.verify_paired_branches(
                    control, replace(treatment, final_streams=changed)
                )
            with self.assertRaises(RuntimeError):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        initialization={
                            **treatment.initialization,
                            "parameter_count": treatment.initialization[
                                "parameter_count"
                            ]
                            + 1,
                        },
                    ),
                )
            with self.assertRaises(RuntimeError):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        initial_streams={
                            **treatment.initial_streams,
                            "model_sha256": "changed",
                        },
                    ),
                )

    def test_paired_comparison_requires_the_exact_dropout_step_at_fixed_width(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            with contextlib.redirect_stdout(io.StringIO()):
                control = STAGE.run_branch(
                    self.spec(
                        root,
                        "control",
                        config.n_embd,
                        STAGE.DEFAULT_CONTROL_DROPOUT,
                    ),
                    data,
                    config,
                    device,
                )
                treatment = STAGE.run_branch(
                    self.spec(root, "treatment", config.n_embd, 0.25),
                    data,
                    config,
                    device,
                )
            with self.assertRaisesRegex(RuntimeError, "0.20"):
                STAGE.verify_paired_branches(control, treatment)
            with self.assertRaisesRegex(RuntimeError, "hold width fixed"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        spec=replace(
                            treatment.spec,
                            width=config.n_embd + 4,
                            dropout=STAGE.DEFAULT_TREATMENT_DROPOUT,
                        ),
                    ),
                )

    def test_canonical_branch_specs_lock_width_and_both_probabilities(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            specs = STAGE.branch_specs(root)
            STAGE.validate_branch_specs(specs)
            self.assertEqual(
                [(spec.key, spec.width, spec.dropout) for spec in specs],
                [
                    ("control", 320, 0.15),
                    ("treatment", 320, 0.20),
                ],
            )
            for spec in specs:
                self.assertIn("p015" if spec.dropout == 0.15 else "p020", spec.best_path.name)
                self.assertIn("step_50000_final", spec.final_path.name)
            for changed in (
                (specs[1], specs[0]),
                (replace(specs[0], dropout=0.20), specs[1]),
                (specs[0], replace(specs[1], dropout=0.125)),
                (specs[0], replace(specs[1], width=256)),
            ):
                with self.assertRaises(ValueError):
                    STAGE.validate_branch_specs(changed)
            for dropout in (-0.1, 1.0, float("nan")):
                with self.assertRaises(ValueError):
                    replace(specs[0], dropout=dropout)
            with self.assertRaises(ValueError):
                replace(specs[0], key="other")

    def test_evaluation_cadence_does_not_change_training(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            frequent_spec = self.spec(root, "control", 16, STAGE.DEFAULT_CONTROL_DROPOUT)
            sparse_spec = STAGE.BranchSpec(
                "control",
                16,
                STAGE.DEFAULT_CONTROL_DROPOUT,
                root / "sparse_best.pt",
                root / "sparse_final.pt",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                frequent = STAGE.run_branch(frequent_spec, data, config, device)
                sparse = STAGE.run_branch(
                    sparse_spec,
                    data,
                    replace(config, eval_interval=2, eval_iters=3),
                    device,
                )
            self.assertEqual(frequent.final_streams, sparse.final_streams)
            self.assertEqual(frequent.initial_streams, sparse.initial_streams)

    def test_fresh_pair_is_repeatable_shared_and_preserves_rng(self):
        with tempfile.TemporaryDirectory() as raw:
            control, treatment, data, config, device = self.run_pair(Path(raw))
            torch.manual_seed(991)
            before = torch.get_rng_state().clone()
            first = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1362, eval_iters=5
            )
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            second = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1362, eval_iters=5
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
            ):
                with self.assertRaises(ValueError):
                    STAGE.evaluate_checkpoint_pair(
                        control,
                        treatment,
                        data,
                        config,
                        device,
                        seed=seed,
                        eval_iters=5,
                    )
            for count in (0, 1):
                with self.assertRaises(ValueError):
                    STAGE.evaluate_checkpoint_pair(
                        control,
                        treatment,
                        data,
                        config,
                        device,
                        seed=1362,
                        eval_iters=count,
                    )

    def test_frontier_comparison_pairs_the_stronger_c320_against_a_narrower_endpoint(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, device = self.run_pair(root)
            # A narrower stored endpoint stands in for the C256 frontier: it is a
            # genuine Stage 45 checkpoint, produced and validated by Stage 45.
            frontier_config = replace(config, n_embd=config.n_embd - 4)
            frontier_key = "control"
            frontier_spec = STAGE._STAGE_45.BranchSpec(
                frontier_key,
                frontier_config.n_embd,
                root / "frontier_best.pt",
                root / "frontier_final.pt",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                STAGE._STAGE_45.run_branch(
                    frontier_spec, data, frontier_config, device
                )
            frontier_payload = STAGE._STAGE_45.validate_branch_checkpoint(
                frontier_spec.final_path,
                frontier_spec,
                data,
                frontier_config,
                "final",
            )
            STAGE.verify_paired_branches(control, treatment)
            report = STAGE.evaluate_frontier_comparison(
                control.spec,
                treatment.spec,
                STAGE.validate_branch_checkpoint(
                    control.spec.final_path, control.spec, data, config, "final"
                ),
                STAGE.validate_branch_checkpoint(
                    treatment.spec.final_path, treatment.spec, data, config, "final"
                ),
                data,
                config,
                device,
                STAGE.FrontierEndpoint(
                    frontier_spec.final_path, frontier_payload, frontier_config
                ),
                seed=1363,
                eval_iters=4,
            )
            self.assertEqual(report["seed"], 1363)
            self.assertEqual(report["eval_iters"], 4)
            self.assertTrue(report["shared_panel_per_batch"])
            self.assertFalse(report["dropout_enabled"])
            self.assertEqual(len(report["endpoints"]["frontier_c256_p010"]["batch_losses"]), 4)
            self.assertEqual(
                report["frontier_checkpoint_sha256"],
                STAGE.file_hash(frontier_spec.final_path),
            )
            comparisons = report["comparisons"]
            self.assertEqual(
                (
                    comparisons["treatment_minus_frontier"]["candidate"],
                    comparisons["treatment_minus_frontier"]["baseline"],
                ),
                (
                    report["endpoints"]["treatment_c320_p020"]["name"],
                    report["endpoints"]["frontier_c256_p010"]["name"],
                ),
            )
            # Every interval must be exactly the per-batch difference of the
            # three shared endpoints, recomputed here without the runner.
            for key, candidate, baseline in (
                (
                    "treatment_minus_frontier",
                    "treatment_c320_p020",
                    "frontier_c256_p010",
                ),
                (
                    "control_minus_frontier",
                    "control_c320_p015",
                    "frontier_c256_p010",
                ),
                (
                    "treatment_minus_control",
                    "treatment_c320_p020",
                    "control_c320_p015",
                ),
            ):
                differences = [
                    high - low
                    for high, low in zip(
                        report["endpoints"][candidate]["batch_losses"],
                        report["endpoints"][baseline]["batch_losses"],
                        strict=True,
                    )
                ]
                delta = comparisons[key]
                self.assertAlmostEqual(
                    delta["mean_delta"], sum(differences) / len(differences), places=12
                )
                self.assertAlmostEqual(
                    delta["mean_delta"],
                    report["endpoints"][candidate]["mean_loss"]
                    - report["endpoints"][baseline]["mean_loss"],
                    places=12,
                )
                self.assertLess(delta["confidence_low"], delta["mean_delta"])
                self.assertGreater(delta["confidence_high"], delta["mean_delta"])
            self.assertEqual(
                STAGE.frontier_outcome(
                    {
                        "comparisons": {
                            "treatment_minus_frontier": {
                                "confidence_high": -1e-3,
                                "confidence_low": -2e-3,
                            }
                        }
                    }
                ),
                "C320 p=0.20 overtakes C256 at matched 50k updates",
            )
            self.assertEqual(
                STAGE.frontier_outcome(
                    {
                        "comparisons": {
                            "treatment_minus_frontier": {
                                "confidence_high": 2e-3,
                                "confidence_low": 1e-3,
                            }
                        }
                    }
                ),
                "C256 still wins against C320 p=0.20 at matched 50k updates",
            )
            self.assertEqual(
                STAGE.frontier_outcome(
                    {
                        "comparisons": {
                            "treatment_minus_frontier": {
                                "confidence_high": 2e-3,
                                "confidence_low": -2e-3,
                            }
                        }
                    }
                ),
                "C320 p=0.20 is unresolved against C256 on the shared panel",
            )
            with self.assertRaisesRegex(ValueError, "unused"):
                STAGE.evaluate_frontier_comparison(
                    control.spec,
                    treatment.spec,
                    STAGE.validate_branch_checkpoint(
                        control.spec.final_path, control.spec, data, config, "final"
                    ),
                    STAGE.validate_branch_checkpoint(
                        treatment.spec.final_path,
                        treatment.spec,
                        data,
                        config,
                        "final",
                    ),
                    data,
                    config,
                    device,
                    STAGE.FrontierEndpoint(
                        frontier_spec.final_path, frontier_payload, frontier_config
                    ),
                    seed=STAGE.DEFAULT_FRESH_VALIDATION_SEED,
                    eval_iters=4,
                )
            with self.assertRaisesRegex(ValueError, "differ in width"):
                STAGE.evaluate_frontier_comparison(
                    control.spec,
                    treatment.spec,
                    STAGE.validate_branch_checkpoint(
                        control.spec.final_path, control.spec, data, config, "final"
                    ),
                    STAGE.validate_branch_checkpoint(
                        treatment.spec.final_path,
                        treatment.spec,
                        data,
                        config,
                        "final",
                    ),
                    data,
                    config,
                    device,
                    STAGE.FrontierEndpoint(
                        frontier_spec.final_path, frontier_payload, config
                    ),
                    seed=1363,
                    eval_iters=4,
                )

    def test_canonical_protocol_rejects_changed_data_and_additional_variables(self):
        data, _, _ = self.fixture()
        configs = (STAGE.canonical_config(0.15), STAGE.canonical_config(0.20))
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
                replace(configs[1], n_embd=256),
                replace(configs[1], max_iters=49999),
                replace(configs[1], learning_rate=3e-5),
                replace(configs[1], eval_iters=99),
            ):
                with self.assertRaisesRegex(ValueError, "locks"):
                    STAGE.validate_protocol(data, (configs[0], changed))

    def test_artifact_paths_protect_inputs_existing_files_and_temporary_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment = STAGE.branch_specs(root)
            outputs = (
                control.best_path,
                control.final_path,
                treatment.best_path,
                treatment.final_path,
            )
            STAGE.validate_artifact_paths((root / "corpus.txt",), outputs)
            with self.assertRaises(ValueError):
                STAGE.validate_artifact_paths((control.best_path,), outputs)
            with self.assertRaises(ValueError):
                STAGE.validate_artifact_paths(
                    (), (control.best_path, control.best_path)
                )
            temporary = control.best_path.with_name(f".{control.best_path.name}.tmp")
            temporary.write_bytes(b"partial checkpoint")
            with self.assertRaises(FileExistsError):
                STAGE.validate_artifact_paths((), outputs)
            temporary.unlink()
            control.best_path.write_bytes(b"existing checkpoint")
            with self.assertRaises(FileExistsError):
                STAGE.validate_artifact_paths((), outputs)
            data, config, device = self.fixture()
            small_spec = replace(control, width=16)
            with self.assertRaises(FileExistsError):
                STAGE.run_branch(small_spec, data, config, device)
            self.assertEqual(control.best_path.read_bytes(), b"existing checkpoint")

    def test_no_improvement_retains_native_initialization_and_terminal_checkpoint(self):
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

    def test_checkpoint_rejects_step_learning_rate_dropout_and_stream_tampering(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, _, data, config, _ = self.run_pair(root)
            original = torch.load(control.spec.final_path, weights_only=True)
            mutations = (
                lambda payload: payload.update(step=2),
                lambda payload: payload["optimizer_state_dict"]["param_groups"][
                    0
                ].update(lr=3e-5),
                lambda payload: payload["architecture"].update(residual_dropout=0.20),
                lambda payload: payload["experiment"].update(
                    branch_residual_dropout=0.20
                ),
                lambda payload: payload["initialization"].update(residual_dropout=0.20),
                lambda payload: payload.update(
                    training_generator_state=torch.Generator()
                    .manual_seed(999)
                    .get_state()
                ),
                lambda payload: next(iter(payload["model_state_dict"].values())).add_(
                    1
                ),
            )
            for mutate in mutations:
                payload = copy.deepcopy(original)
                mutate(payload)
                path = root / "tampered.pt"
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    STAGE.validate_branch_checkpoint(
                        path, control.spec, data, config, kind="final"
                    )


if __name__ == "__main__":
    unittest.main()
