from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary
from training import LossEstimate


SPEC = importlib.util.spec_from_file_location(
    "stage_42_under_test", Path(__file__).with_name("42_final_learning_rate_fork.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage42Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def make_source(self, directory: Path):
        vocabulary = CharacterVocabulary.from_text("abcde")
        tokens = torch.arange(2400) % vocabulary.size
        data = CharacterData(vocabulary, tokens[:1800], tokens[1800:], 2400)
        config = TrainingConfig(
            batch_size=2,
            block_size=24,
            n_embd=8,
            learning_rate=3e-5,
            max_iters=44003,
            eval_interval=1,
            eval_iters=1,
        )
        device = torch.device("cpu")
        torch.manual_seed(421)
        model = STAGE.build_model(data, config, STAGE.N_HEAD, STAGE.N_LAYER, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        generator = torch.Generator().manual_seed(STAGE.DEFAULT_TRAINING_BATCH_SEED)
        inputs, targets = STAGE.get_batch(data, "train", config, generator, device)
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        # Synthetic moment history for a cheap real-model continuation test.
        # Production load_sources accepts only the authenticated 44k artifact.
        for state in optimizer.state.values():
            state["step"].fill_(44000)
        losses = STAGE.evaluate_on_fixed_batches(
            model, data, config, device, fixed_panel_seed=STAGE.DEFAULT_FIXED_PANEL_SEED
        )
        payload = {
            "step": 44000,
            "checkpoint_kind": "final",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_generator_state": generator.get_state(),
            "rng_state": STAGE.capture_rng_state(device),
            "snapshot_losses": {"train": losses.train, "validation": losses.val},
            "best_step": 44000,
            "best_val_loss": losses.val,
            "best_training_loss": losses.train,
            "architecture": {"n_embd": 8, "block_size": 24, "n_head": 4, "n_layer": 6},
            "corpus": {"train_characters": len(data.train_data)},
            "data_fingerprint": STAGE._STAGE_41.fingerprint_data(data),
            "initialization": {"origin_stage": 37},
            "training_config": {
                **asdict(config),
                "training_batch_seed": STAGE.DEFAULT_TRAINING_BATCH_SEED,
                "fixed_panel_seed": STAGE.DEFAULT_FIXED_PANEL_SEED,
                "training_rng_seed": STAGE.DEFAULT_TRAINING_RNG_SEED,
            },
            "evaluation_protocol": {},
        }
        path = directory / "synthetic_44k.pt"
        torch.save(payload, path)
        runtime_type, runtime_name = STAGE._STAGE_41._runtime_identity(device)
        source = STAGE.ForkSource(
            path,
            STAGE.file_hash(path),
            torch.load(path, weights_only=True),
            runtime_type,
            runtime_name,
            str(torch.__version__),
        )
        return source, data, config, device

    def test_locked_protocol_and_complete_schedule_history(self) -> None:
        config = STAGE.canonical_config()
        self.assertEqual(
            (config.batch_size, config.block_size, config.n_embd), (8, 256, 128)
        )
        self.assertEqual((STAGE.N_HEAD, STAGE.N_LAYER), (4, 6))
        self.assertEqual(config.max_iters, 50000)
        self.assertEqual((config.eval_interval, config.eval_iters), (500, 100))
        self.assertEqual(STAGE.DEFAULT_FRESH_VALIDATION_ITERS, 500)
        self.assertEqual(config.batch_size * config.block_size * 6000, 12288000)
        self.assertEqual(
            len(
                {
                    STAGE.DEFAULT_HISTORY_VALIDATION_SEED,
                    STAGE.DEFAULT_BRANCH_VALIDATION_SEED,
                    STAGE.DEFAULT_FIXED_PANEL_SEED,
                    STAGE._STAGE_40.DEFAULT_FRESH_VALIDATION_SEED,
                }
            ),
            4,
        )
        for rate in (3e-5, 1e-5):
            schedule = STAGE.FinalLearningRateSchedule(rate)
            for step, expected in (
                (0, 1e-3),
                (10000, 3e-4),
                (13000, 1e-4),
                (18000, 3e-5),
                (32000, 3e-5),
                (38000, 3e-5),
                (43999, 3e-5),
                (44000, rate),
                (49999, rate),
            ):
                self.assertEqual(schedule.for_update(step), expected)
            segments = schedule.as_metadata(target_step=50000)
            self.assertEqual(
                segments[:-1],
                STAGE._STAGE_41.FinalComputeExtensionSchedule().as_metadata(
                    target_step=44000
                ),
            )
            self.assertEqual(
                segments[-1],
                {
                    "start_update": 44000,
                    "end_update_exclusive": 50000,
                    "learning_rate": rate,
                },
            )
        for invalid in (0, 1e-4, float("nan")):
            with self.assertRaises(ValueError):
                STAGE.FinalLearningRateSchedule(invalid)
        with patch.object(sys, "argv", [str(SPEC.origin)]):
            args = STAGE.parse_args()
        self.assertEqual(
            args.source_checkpoint, STAGE._STAGE_41.DEFAULT_FINAL_CHECKPOINT_PATH
        )
        self.assertFalse(args.validate_only)

    def test_paths_reject_collisions_and_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            a, b = STAGE.branch_specs(root)
            outputs = (a.best_path, a.final_path, b.best_path, b.final_path)
            STAGE.validate_artifact_paths((root / "source.pt",), outputs)
            with self.assertRaises(ValueError):
                STAGE.validate_artifact_paths((a.best_path,), outputs)
            a.best_path.touch()
            with self.assertRaises(FileExistsError):
                STAGE.validate_artifact_paths((root / "source.pt",), outputs)

    def test_source_authentication_rejects_synthetic_or_changed_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, data, config, device = self.make_source(Path(raw))
            with self.assertRaisesRegex(ValueError, "locks architecture"):
                STAGE.load_sources(source.path, source.path, data, config, device)
            with self.assertRaisesRegex(ValueError, "authenticated Stage 41"):
                STAGE.load_sources(
                    source.path, source.path, data, STAGE.canonical_config(), device
                )
            source.path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                STAGE.reload_source(source)

    def test_real_dropout_fork_matches_batches_rng_and_checkpoint_round_trips(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, data, config, device = self.make_source(root)
            a, b = STAGE.branch_specs(root)
            before = STAGE.file_hash(source.path)
            with contextlib.redirect_stdout(io.StringIO()):
                control = STAGE.run_branch(source, a, data, config, device, None)
                treatment = STAGE.run_branch(source, b, data, config, device, None)
            STAGE.verify_paired_branches(control, treatment)
            self.assertEqual(before, STAGE.file_hash(source.path))
            self.assertNotEqual(
                control.final_state["model_sha256"],
                treatment.final_state["model_sha256"],
            )
            self.assertNotEqual(
                control.final_state["optimizer_sha256"],
                treatment.final_state["optimizer_sha256"],
            )
            self.assertEqual(
                tuple(r.step for r in control.training.history),
                (44000, 44001, 44002, 44003),
            )
            for spec in (a, b):
                final = torch.load(spec.final_path, weights_only=True)
                self.assertEqual(
                    {
                        int(state["step"])
                        for state in final["optimizer_state_dict"]["state"].values()
                    },
                    {44003},
                )
                self.assertEqual(len(final["optimizer_state_dict"]["state"]), 138)
                self.assertEqual(
                    final["training_config"]["branch_learning_rate"], spec.learning_rate
                )
            broken_state = {**treatment.final_state, "torch_rng_sha256": "different"}
            with self.assertRaisesRegex(RuntimeError, "different"):
                STAGE.verify_paired_branches(
                    control, replace(treatment, final_state=broken_state)
                )

    def test_no_improvement_retains_real_44k_fallback_and_unconditional_final(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, data, config, device = self.make_source(Path(raw))
            spec = STAGE.branch_specs(Path(raw))[1]
            values = [source.losses] + [
                LossEstimate(source.losses.train, source.losses.val + 0.01)
            ] * 3
            with (
                patch.object(STAGE, "evaluate_on_fixed_batches", side_effect=values),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = STAGE.run_branch(source, spec, data, config, device, None)
            self.assertEqual(report.best_step, 44000)
            best = torch.load(spec.best_path, weights_only=True)
            final = torch.load(spec.final_path, weights_only=True)
            self.assertEqual((best["step"], final["step"]), (44000, 44003))
            self.assertEqual(
                STAGE.state_hash(best["model_state_dict"]),
                STAGE.state_hash(source.checkpoint["model_state_dict"]),
            )
            self.assertEqual(
                STAGE._STAGE_40.optimizer_lineage_sha256(best["optimizer_state_dict"]),
                STAGE._STAGE_40.optimizer_lineage_sha256(
                    source.checkpoint["optimizer_state_dict"]
                ),
            )

    def test_failed_source_loss_audit_writes_no_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source, data, config, device = self.make_source(Path(raw))
            spec = STAGE.branch_specs(Path(raw))[0]
            with patch.object(
                STAGE, "evaluate_on_fixed_batches", return_value=LossEstimate(9.0, 9.0)
            ):
                with self.assertRaisesRegex(RuntimeError, "audit failed"):
                    STAGE.run_branch(source, spec, data, config, device, None)
            self.assertFalse(spec.best_path.exists())
            self.assertFalse(spec.final_path.exists())

    def test_fresh_pair_is_shared_rng_isolated_and_metadata_weights_only_safe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, data, config, device = self.make_source(root)
            rng_before = STAGE.state_hash(STAGE.capture_rng_state(device))
            report = STAGE.evaluate_checkpoint_pair(
                source.path,
                source.path,
                source.sha256,
                source.sha256,
                "baseline",
                "identical candidate",
                data,
                config,
                device,
                seed=STAGE.DEFAULT_HISTORY_VALIDATION_SEED,
                eval_iters=5,
            )
            self.assertEqual(
                rng_before, STAGE.state_hash(STAGE.capture_rng_state(device))
            )
            self.assertEqual(
                report.baseline.batch_losses, report.candidate.batch_losses
            )
            self.assertEqual(
                (
                    report.delta.mean_delta,
                    report.delta.standard_error,
                    report.delta.confidence_low,
                    report.delta.confidence_high,
                ),
                (0, 0, 0, 0),
            )
            # Include actual fresh metadata in a resumable checkpoint. Path
            # objects must not force unsafe torch.load(weights_only=False).
            spec = STAGE.branch_specs(root)[0]
            with contextlib.redirect_stdout(io.StringIO()):
                STAGE.run_branch(source, spec, data, config, device, report)
            payload = torch.load(spec.final_path, weights_only=True)
            self.assertIsInstance(
                payload["fresh_validation_38k_vs_44k"]["baseline"]["checkpoint_path"],
                str,
            )
            STAGE.write_report(root / "report.json", asdict(report))
            parsed = json.loads((root / "report.json").read_text())
            self.assertEqual(parsed["eval_iters"], 5)
            self.assertEqual(parsed["delta"]["mean_delta"], 0)
            for seed in (
                STAGE.DEFAULT_FIXED_PANEL_SEED,
                STAGE._STAGE_40.DEFAULT_FRESH_VALIDATION_SEED,
            ):
                with self.assertRaises(ValueError):
                    STAGE.evaluate_checkpoint_pair(
                        source.path,
                        source.path,
                        source.sha256,
                        source.sha256,
                        "a",
                        "b",
                        data,
                        config,
                        device,
                        seed=seed,
                        eval_iters=5,
                    )

    def test_checkpoint_rejects_lr_step_and_stream_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, data, config, device = self.make_source(root)
            spec = STAGE.branch_specs(root)[0]
            with contextlib.redirect_stdout(io.StringIO()):
                STAGE.run_branch(source, spec, data, config, device, None)
            original = torch.load(spec.final_path, weights_only=True)
            mutations = (
                lambda p: p["optimizer_state_dict"]["param_groups"][0].update(lr=1e-5),
                lambda p: p.update(step=44002),
                lambda p: p.update(
                    training_generator_state=torch.Generator()
                    .manual_seed(987)
                    .get_state()
                ),
            )
            for mutate in mutations:
                payload = copy.deepcopy(original)
                mutate(payload)
                torch.save(payload, root / "tampered.pt")
                with self.assertRaises(ValueError):
                    STAGE.validate_branch_checkpoint(
                        root / "tampered.pt", source, spec, config, kind="final"
                    )


if __name__ == "__main__":
    unittest.main()
