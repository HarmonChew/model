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
    "stage_43_under_test", Path(__file__).with_name("43_capacity_scaling.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage43Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, width: int = 8, **overrides):
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

    def spec(self, directory: Path, key: str, width: int):
        return STAGE.BranchSpec(
            key, width, directory / f"{key}_best.pt", directory / f"{key}_final.pt"
        )

    def run_pair(self, directory: Path):
        data, config, device = self.fixture()
        wider = replace(config, n_embd=12)
        control_spec = self.spec(directory, "control", config.n_embd)
        treatment_spec = self.spec(directory, "treatment", wider.n_embd)
        with contextlib.redirect_stdout(io.StringIO()):
            control = STAGE.run_branch(control_spec, data, config, device)
            treatment = STAGE.run_branch(treatment_spec, data, wider, device)
        return control, treatment, data, config, device

    def test_protocol_changes_only_width_and_preserves_the_established_schedule(self):
        control = STAGE.canonical_config(128)
        treatment = STAGE.canonical_config(192)
        self.assertEqual(
            {
                key
                for key in asdict(control)
                if asdict(control)[key] != asdict(treatment)[key]
            },
            {"n_embd"},
        )
        self.assertEqual((control.batch_size, control.block_size), (8, 256))
        self.assertEqual((STAGE.N_HEAD, STAGE.N_LAYER), (4, 6))
        self.assertEqual(control.max_iters, 50000)
        self.assertEqual((control.eval_interval, control.eval_iters), (500, 100))
        self.assertEqual(
            control.batch_size * control.block_size * control.max_iters, 102400000
        )
        self.assertEqual(control.learning_rate, 1e-3)
        schedule = STAGE.make_schedule()
        for step, expected in (
            (0, 1e-3),
            (9999, 1e-3),
            (10000, 3e-4),
            (12999, 3e-4),
            (13000, 1e-4),
            (17999, 1e-4),
            (18000, 3e-5),
            (43999, 3e-5),
            (49999, 3e-5),
        ):
            self.assertEqual(schedule.for_update(step), expected)
        for width in (0, 127, 256):
            with self.assertRaises(ValueError):
                STAGE.canonical_config(width)

    def test_native_initialization_is_reproducible_and_adamw_starts_empty(self):
        data, config, device = self.fixture()
        torch.manual_seed(13)
        first, first_optimizer, first_batches, first_info = STAGE.initialize_branch(
            data, config, device
        )
        torch.manual_seed(99)
        second, second_optimizer, second_batches, second_info = STAGE.initialize_branch(
            data, config, device
        )
        self.assertEqual(first_info, second_info)
        self.assertEqual(first_info["kind"], "fresh_architecture_native")
        self.assertEqual((first_info["origin_stage"], first_info["step"]), (43, 0))
        self.assertFalse(first_info["cross_width_tensor_identity_claimed"])
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

        wider, wider_optimizer, wider_batches, _ = STAGE.initialize_branch(
            data, replace(config, n_embd=12), device
        )
        self.assertEqual(len(wider_optimizer.state), 0)
        self.assertGreater(
            sum(parameter.numel() for parameter in wider.parameters()),
            sum(parameter.numel() for parameter in first.parameters()),
        )
        self.assertNotEqual(
            {key: tuple(value.shape) for key, value in first.state_dict().items()},
            {key: tuple(value.shape) for key, value in wider.state_dict().items()},
        )
        for _ in range(3):
            a = STAGE.get_batch(data, "train", config, first_batches, device)
            b = STAGE.get_batch(
                data, "train", replace(config, n_embd=12), wider_batches, device
            )
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_real_width_pair_starts_at_zero_and_saves_resumable_endpoints(self):
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
                control.final_streams["training_generator_sha256"],
                treatment.final_streams["training_generator_sha256"],
            )
            for report in (control, treatment):
                branch_config = replace(config, n_embd=report.spec.width)
                final = STAGE.validate_branch_checkpoint(
                    report.spec.final_path,
                    report.spec,
                    data,
                    branch_config,
                    kind="final",
                )
                self.assertEqual(final["step"], 3)
                self.assertEqual(final["architecture"]["n_embd"], report.spec.width)
                self.assertEqual(
                    {
                        int(state["step"])
                        for state in final["optimizer_state_dict"]["state"].values()
                    },
                    {3},
                )
                self.assertTrue(final["optimizer_state_dict"]["state"])
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

    def test_evaluation_cadence_does_not_change_training(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            frequent_spec = STAGE.BranchSpec(
                "control", 8, root / "frequent_best.pt", root / "frequent_final.pt"
            )
            sparse_spec = STAGE.BranchSpec(
                "control", 8, root / "sparse_best.pt", root / "sparse_final.pt"
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
                control, treatment, data, config, device, seed=1357, eval_iters=5
            )
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            second = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1357, eval_iters=5
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
            for seed in (STAGE.DEFAULT_FIXED_PANEL_SEED, 1354, 1355, 1356):
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
                        seed=1357,
                        eval_iters=count,
                    )

    def test_canonical_protocol_rejects_changed_data_and_additional_variables(self):
        data, _, _ = self.fixture()
        configs = (STAGE.canonical_config(128), STAGE.canonical_config(192))
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
            self.assertEqual((control.width, treatment.width), (128, 192))
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
            small_spec = replace(control, width=8)
            with self.assertRaises(FileExistsError):
                STAGE.run_branch(small_spec, data, config, device)
            self.assertEqual(control.best_path.read_bytes(), b"existing checkpoint")

    def test_no_improvement_retains_native_initialization_and_terminal_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            spec = self.spec(root, "control", 8)
            initialized, _, _, _ = STAGE.initialize_branch(data, config, device)
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

    def test_checkpoint_rejects_structural_tampering_independently_of_its_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, _, data, config, _ = self.run_pair(root)
            original = torch.load(control.spec.final_path, weights_only=True)
            # These fields are validated structurally, so the tamper must be
            # rejected even when the payload integrity hash is recomputed to
            # match the tampered content.
            structural = (
                (lambda payload: payload.update(step=2), "steps are invalid"),
                (
                    lambda payload: payload["optimizer_state_dict"]["param_groups"][
                        0
                    ].update(lr=3e-5),
                    "AdamW settings changed",
                ),
                (
                    lambda payload: payload["architecture"].update(n_embd=12),
                    "architecture changed",
                ),
            )
            for mutate, message in structural:
                payload = copy.deepcopy(original)
                mutate(payload)
                payload["payload_sha256"] = STAGE.state_hash(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "payload_sha256"
                    }
                )
                path = root / "tampered.pt"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, message):
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
                lambda payload: payload.update(
                    training_generator_state=torch.Generator()
                    .manual_seed(999)
                    .get_state()
                ),
                lambda payload: next(iter(payload["model_state_dict"].values())).add_(
                    1
                ),
            )
            for mutate in content:
                payload = copy.deepcopy(original)
                mutate(payload)
                path = root / "stale.pt"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "payload integrity mismatch"):
                    STAGE.validate_branch_checkpoint(
                        path, control.spec, data, config, kind="final"
                    )


if __name__ == "__main__":
    unittest.main()
