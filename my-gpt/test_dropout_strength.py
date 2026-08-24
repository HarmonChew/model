import contextlib
import copy
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("16_dropout_strength.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_16_dropout_strength",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_16 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_16
SPEC.loader.exec_module(STAGE_16)


class Stage16DefaultsTests(unittest.TestCase):
    def test_defaults_define_one_half_strength_treatment(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_16.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 64)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.initial_learning_rate, 1e-3)
        self.assertEqual(args.first_decay_step, 10_000)
        self.assertEqual(args.middle_learning_rate, 3e-4)
        self.assertEqual(args.second_decay_step, 13_000)
        self.assertEqual(args.final_learning_rate, 1e-4)
        self.assertEqual(args.dropout, 0.05)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertIsNone(args.precise_eval_seed)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_15_control_best_checkpoint.pt",
        )
        self.assertEqual(
            args.high_dropout_checkpoint_path.name,
            "stage_15_dropout_best_checkpoint.pt",
        )
        self.assertEqual(
            args.checkpoint_path.name,
            "stage_16_dropout_best_checkpoint.pt",
        )

    def test_default_paths_pin_the_exact_stage_15_artifacts(self) -> None:
        control_hash, control_step = STAGE_16._pin_for_default_path(
            STAGE_16.DEFAULT_CONTROL_CHECKPOINT_PATH,
            STAGE_16.DEFAULT_CONTROL_CHECKPOINT_PATH,
            STAGE_16.DEFAULT_CONTROL_CHECKPOINT_SHA256,
        )
        high_hash, high_step = STAGE_16._pin_for_default_path(
            STAGE_16.DEFAULT_HIGH_DROPOUT_CHECKPOINT_PATH,
            STAGE_16.DEFAULT_HIGH_DROPOUT_CHECKPOINT_PATH,
            STAGE_16.DEFAULT_HIGH_DROPOUT_CHECKPOINT_SHA256,
        )

        self.assertEqual(
            control_hash,
            "4d05d242344f6020a56d93c21507e8b0bc7a73a23d51d65e673982192b180168",
        )
        self.assertEqual(
            high_hash,
            "8b3d781476cfbdbe304a54ff6883264697f16d2291ce724997ead29cc28ecfd6",
        )
        self.assertEqual(control_step, 17_000)
        self.assertEqual(high_step, 17_000)


class Stage16ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        text = "abcd\n" * 100
        vocabulary = CharacterVocabulary.from_text(text)
        token_ids = torch.tensor(vocabulary.encode(text), dtype=torch.long)
        self.data = CharacterData(
            vocabulary=vocabulary,
            train_data=token_ids[:400],
            val_data=token_ids[400:],
            num_characters=len(text),
        )
        self.device = torch.device("cpu")
        self.n_head = 2
        self.n_layer = 1
        self.schedule = STAGE_16.LearningRateSchedule(
            initial_learning_rate=2e-2,
            first_decay_step=2,
            middle_learning_rate=7e-3,
            second_decay_step=4,
            final_learning_rate=2e-3,
        )
        self.config = TrainingConfig(
            batch_size=2,
            block_size=4,
            n_embd=8,
            learning_rate=self.schedule.initial_learning_rate,
            max_iters=6,
            eval_interval=2,
            eval_iters=1,
            seed=101,
        )
        self.training_batch_seed = 202
        self.training_rng_seed = 303
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.control_path = self.directory / "stage15_control.pt"
        self.high_path = self.directory / "stage15_dropout.pt"
        self.treatment_path = self.directory / "stage16_dropout.pt"

    def make_treatment_spec(self, dropout: float = 0.05) -> object:
        return STAGE_16.BranchSpec(
            name="dropout_p005",
            dropout=dropout,
            checkpoint_path=self.treatment_path,
        )

    def create_stage_15_references(self) -> tuple[object, object, str]:
        initial_state, initial_hash, _ = STAGE_16.create_shared_initial_state(
            self.data,
            self.config,
            self.n_head,
            self.n_layer,
        )
        reports = []
        for name, dropout, path in (
            ("control", 0.0, self.control_path),
            ("dropout", 0.1, self.high_path),
        ):
            spec = STAGE_16.BranchSpec(
                name=name,
                dropout=dropout,
                checkpoint_path=path,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                report, _ = STAGE_16._STAGE_15.run_branch(
                    self.data,
                    self.config,
                    self.n_head,
                    self.n_layer,
                    self.device,
                    0,
                    self.schedule,
                    initial_state,
                    initial_hash,
                    spec,
                    initialization_seed=self.config.seed,
                    training_batch_seed=self.training_batch_seed,
                    training_rng_seed=self.training_rng_seed,
                )
            reports.append(report)
        return reports[0], reports[1], initial_hash

    def run_tiny_experiment(self) -> object:
        self.create_stage_15_references()
        with contextlib.redirect_stdout(io.StringIO()):
            return STAGE_16.run_experiment(
                self.data,
                self.config,
                self.n_head,
                self.n_layer,
                self.device,
                0,
                self.schedule,
                self.make_treatment_spec(),
                self.control_path,
                self.high_path,
                training_batch_seed=self.training_batch_seed,
                training_rng_seed=self.training_rng_seed,
            )

    def test_distinct_path_validation_protects_reference_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use different paths"):
            STAGE_16.validate_distinct_paths(
                self.control_path,
                self.control_path,
                self.high_path,
            )

        candidate_named_as_control_temp = self.control_path.with_name(
            f".{self.control_path.name}.tmp"
        )
        with self.assertRaisesRegex(ValueError, "must use different paths"):
            STAGE_16.validate_distinct_paths(
                candidate_named_as_control_temp,
                self.control_path,
                self.high_path,
            )

    def test_treatment_probability_is_pinned_to_point_zero_five(self) -> None:
        self.create_stage_15_references()
        with self.assertRaisesRegex(ValueError, "exactly 0.05"):
            STAGE_16.run_experiment(
                self.data,
                self.config,
                self.n_head,
                self.n_layer,
                self.device,
                0,
                self.schedule,
                self.make_treatment_spec(0.04),
                self.control_path,
                self.high_path,
                training_batch_seed=self.training_batch_seed,
                training_rng_seed=self.training_rng_seed,
            )

    def test_reference_validator_rejects_the_wrong_expected_hash(self) -> None:
        _, _, initial_hash = self.create_stage_15_references()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            STAGE_16.validate_stage_15_reference(
                self.control_path,
                branch_name="control",
                display_name="p=0",
                dropout=0.0,
                data=self.data,
                config=self.config,
                n_head=self.n_head,
                n_layer=self.n_layer,
                schedule=self.schedule,
                initial_state_sha256=initial_hash,
                training_batch_seed=self.training_batch_seed,
                training_rng_seed=self.training_rng_seed,
                expected_checkpoint_sha256="0" * 64,
            )

    def test_treatment_only_run_preserves_references_and_writes_provenance(
        self,
    ) -> None:
        control_report, high_report, initial_hash = self.create_stage_15_references()
        control_hash_before = STAGE_16.checkpoint_sha256(self.control_path)
        high_hash_before = STAGE_16.checkpoint_sha256(self.high_path)

        with contextlib.redirect_stdout(io.StringIO()):
            report = STAGE_16.run_experiment(
                self.data,
                self.config,
                self.n_head,
                self.n_layer,
                self.device,
                0,
                self.schedule,
                self.make_treatment_spec(),
                self.control_path,
                self.high_path,
                training_batch_seed=self.training_batch_seed,
                training_rng_seed=self.training_rng_seed,
                expected_control_sha256=control_hash_before,
                expected_control_step=control_report.best_step,
                expected_high_dropout_sha256=high_hash_before,
                expected_high_dropout_step=high_report.best_step,
            )

        self.assertTrue(self.treatment_path.is_file())
        self.assertEqual(
            STAGE_16.checkpoint_sha256(self.control_path),
            control_hash_before,
        )
        self.assertEqual(
            STAGE_16.checkpoint_sha256(self.high_path),
            high_hash_before,
        )
        self.assertEqual(report.initial_state_sha256, initial_hash)
        self.assertEqual(report.control.checkpoint_sha256, control_hash_before)
        self.assertEqual(report.high_dropout.checkpoint_sha256, high_hash_before)
        self.assertEqual(report.treatment.dropout, 0.05)
        self.assertEqual(
            report.treatment.final_batch_generator_sha256,
            STAGE_16.expected_training_generator_sha256(
                self.data,
                self.config,
                training_batch_seed=self.training_batch_seed,
            ),
        )

        payload = STAGE_16.validate_stage_16_checkpoint(
            self.treatment_path,
            self.make_treatment_spec(),
            self.data,
            self.config,
            self.n_head,
            self.n_layer,
            self.schedule,
            initial_state_sha256=initial_hash,
            training_batch_seed=self.training_batch_seed,
            training_rng_seed=self.training_rng_seed,
            control_checkpoint_sha256=control_hash_before,
            high_dropout_checkpoint_sha256=high_hash_before,
        )
        self.assertEqual(payload["experiment"]["stage"], 16)
        self.assertTrue(payload["experiment"]["reused_control"])
        self.assertEqual(
            payload["experiment"]["control_checkpoint_sha256"],
            control_hash_before,
        )
        self.assertEqual(
            payload["experiment"]["high_dropout_reference_checkpoint_sha256"],
            high_hash_before,
        )
        self.assertEqual(payload["architecture"]["residual_dropout"], 0.05)
        self.assertEqual(
            payload["training_config"]["learning_rate_schedule"],
            self.schedule.as_metadata(max_iters=self.config.max_iters),
        )

        tampered = copy.deepcopy(payload)
        tampered["experiment"]["control_checkpoint_sha256"] = "wrong"
        tampered_path = self.directory / "tampered.pt"
        torch.save(tampered, tampered_path)
        tampered_spec = STAGE_16.BranchSpec(
            name="dropout_p005",
            dropout=0.05,
            checkpoint_path=tampered_path,
        )
        with self.assertRaisesRegex(
            ValueError,
            "control_checkpoint_sha256",
        ):
            STAGE_16.validate_stage_16_checkpoint(
                tampered_path,
                tampered_spec,
                self.data,
                self.config,
                self.n_head,
                self.n_layer,
                self.schedule,
                initial_state_sha256=initial_hash,
                training_batch_seed=self.training_batch_seed,
                training_rng_seed=self.training_rng_seed,
                control_checkpoint_sha256=control_hash_before,
                high_dropout_checkpoint_sha256=high_hash_before,
            )

    def test_fresh_validation_is_paired_across_all_three_doses(self) -> None:
        experiment = self.run_tiny_experiment()
        report = STAGE_16.run_precise_validation(
            self.data,
            self.config,
            self.n_head,
            self.n_layer,
            self.device,
            self.schedule,
            self.make_treatment_spec(),
            experiment,
            eval_iters=5,
            seed=404,
        )

        self.assertEqual(
            [result.name for result in report.results],
            [
                "p=0",
                "p=0.05",
                "p=0.10",
            ],
        )
        self.assertTrue(all(len(result.batch_losses) == 5 for result in report.results))
        self.assertEqual(
            [(delta.candidate, delta.baseline) for delta in report.deltas],
            [
                ("p=0.05", "p=0"),
                ("p=0.10", "p=0"),
                ("p=0.10", "p=0.05"),
            ],
        )
        expected_differences = tuple(
            candidate - control
            for candidate, control in zip(
                report.results[1].batch_losses,
                report.results[0].batch_losses,
                strict=True,
            )
        )
        expected_mean = sum(expected_differences) / len(expected_differences)
        self.assertAlmostEqual(report.deltas[0].mean_delta, expected_mean)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            STAGE_16.print_precise_validation(report)
        rendered = output.getvalue()
        self.assertIn("shared new-panel seed=404", rendered)
        self.assertIn("Dropout is disabled for every measurement.", rendered)
        self.assertIn("delta p=0.05 - p=0", rendered)
        self.assertIn("negative favors p=0.05", rendered)

    @unittest.skipUnless(
        STAGE_16.DEFAULT_CONTROL_CHECKPOINT_PATH.is_file()
        and STAGE_16.DEFAULT_HIGH_DROPOUT_CHECKPOINT_PATH.is_file(),
        "canonical Stage 15 checkpoints are not present",
    )
    def test_canonical_stage_15_inputs_match_the_recorded_protocol(self) -> None:
        data = CharacterData.from_file(STAGE_16.DEFAULT_DATA_PATH)
        schedule = STAGE_16.LearningRateSchedule()
        config = TrainingConfig(
            batch_size=32,
            block_size=64,
            n_embd=64,
            learning_rate=schedule.initial_learning_rate,
            max_iters=18_000,
            eval_interval=500,
            eval_iters=100,
            seed=1_337,
        )
        _, initial_hash, parameter_count = STAGE_16.create_shared_initial_state(
            data, config, 4, 4
        )
        control = STAGE_16.validate_stage_15_reference(
            STAGE_16.DEFAULT_CONTROL_CHECKPOINT_PATH,
            branch_name="control",
            display_name="p=0",
            dropout=0.0,
            data=data,
            config=config,
            n_head=4,
            n_layer=4,
            schedule=schedule,
            initial_state_sha256=initial_hash,
            training_batch_seed=1_338,
            training_rng_seed=1_340,
            expected_checkpoint_sha256=(STAGE_16.DEFAULT_CONTROL_CHECKPOINT_SHA256),
            expected_step=17_000,
        )
        high = STAGE_16.validate_stage_15_reference(
            STAGE_16.DEFAULT_HIGH_DROPOUT_CHECKPOINT_PATH,
            branch_name="dropout",
            display_name="p=0.10",
            dropout=0.1,
            data=data,
            config=config,
            n_head=4,
            n_layer=4,
            schedule=schedule,
            initial_state_sha256=initial_hash,
            training_batch_seed=1_338,
            training_rng_seed=1_340,
            expected_checkpoint_sha256=(
                STAGE_16.DEFAULT_HIGH_DROPOUT_CHECKPOINT_SHA256
            ),
            expected_step=17_000,
        )

        self.assertEqual(
            initial_hash,
            "88dae91952fe315e838766caa8df2e8624fdcd9dbf0c0ab870cfeeca5ea4bb88",
        )
        self.assertEqual(parameter_count, 211_777)
        self.assertAlmostEqual(control.fixed_panel_loss, 1.5688183736801147)
        self.assertAlmostEqual(high.fixed_panel_loss, 1.5842843401432036)


if __name__ == "__main__":
    unittest.main()
