import importlib.util
import math
import sys
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary
from training import EvaluationRecord, LossEstimate, TrainingResult


MODULE_PATH = Path(__file__).with_name("12_learning_rate_drop.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_12_learning_rate_drop",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_12 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_12
SPEC.loader.exec_module(STAGE_12)


class LearningRateDropTests(unittest.TestCase):
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
        self.source_step = 2
        self.source_val_loss = 1.75
        self.source_config = TrainingConfig(
            batch_size=2,
            block_size=4,
            n_embd=8,
            learning_rate=1e-2,
            max_iters=4,
            eval_interval=1,
            eval_iters=2,
            seed=17,
        )
        self.device = torch.device("cpu")
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.source_checkpoint = self.directory / "source.pt"
        self._write_source_checkpoint()

    def make_model(self) -> torch.nn.Module:
        return STAGE_12.GPTLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.source_config.block_size,
            n_embd=self.source_config.n_embd,
            n_head=2,
            n_layer=1,
        )

    def _write_source_checkpoint(self) -> None:
        torch.manual_seed(self.source_config.seed)
        model = self.make_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.source_config.learning_rate,
        )
        training_generator = torch.Generator().manual_seed(
            self.source_config.seed + 1
        )

        for _ in range(self.source_step):
            inputs, targets = self.data.get_batch(
                "train",
                batch_size=self.source_config.batch_size,
                block_size=self.source_config.block_size,
                device=self.device,
                generator=training_generator,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        tracker = STAGE_12.BestValidationCheckpoint(
            model=model,
            optimizer=optimizer,
            training_generator=training_generator,
            path=self.source_checkpoint,
            device=self.device,
            config=self.source_config,
            n_head=2,
            n_layer=1,
            data_fingerprint=STAGE_12.fingerprint_data(self.data),
        )
        saved = tracker.consider(
            EvaluationRecord(
                step=self.source_step,
                losses=LossEstimate(train=1.5, val=self.source_val_loss),
            )
        )
        self.assertTrue(saved)

    def load_source_payload(self) -> dict[object, object]:
        payload = torch.load(
            self.source_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        self.assertIsInstance(payload, dict)
        return payload

    def assert_nested_equal(self, actual: object, expected: object) -> None:
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return

        if isinstance(expected, Mapping):
            self.assertIsInstance(actual, Mapping)
            assert isinstance(actual, Mapping)
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
            return

        if isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected))
            assert isinstance(actual, (list, tuple))
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(
                actual,
                expected,
                strict=True,
            ):
                self.assert_nested_equal(actual_item, expected_item)
            return

        self.assertEqual(actual, expected)

    def assert_optimizer_equal_except_lr(
        self,
        actual: Mapping[str, object],
        expected: Mapping[str, object],
    ) -> None:
        self.assert_nested_equal(actual["state"], expected["state"])
        actual_groups = actual["param_groups"]
        expected_groups = expected["param_groups"]
        self.assertIsInstance(actual_groups, list)
        self.assertIsInstance(expected_groups, list)
        assert isinstance(actual_groups, list)
        assert isinstance(expected_groups, list)
        self.assertEqual(len(actual_groups), len(expected_groups))

        for actual_group, expected_group in zip(
            actual_groups,
            expected_groups,
            strict=True,
        ):
            self.assertIsInstance(actual_group, Mapping)
            self.assertIsInstance(expected_group, Mapping)
            assert isinstance(actual_group, Mapping)
            assert isinstance(expected_group, Mapping)
            self.assertEqual(set(actual_group), set(expected_group))
            for key in expected_group:
                if key != "lr":
                    self.assert_nested_equal(
                        actual_group[key],
                        expected_group[key],
                    )

    def make_branch_spec(self, name: str, learning_rate: float) -> object:
        return STAGE_12.BranchSpec(
            name=name,
            learning_rate=learning_rate,
            checkpoint_path=self.directory / f"{name}.pt",
        )

    def load_branch(self, name: str, learning_rate: float) -> object:
        return STAGE_12.load_branch(
            self.data,
            self.source_config,
            2,
            1,
            self.device,
            self.source_checkpoint,
            self.make_branch_spec(name, learning_rate),
            expected_source_step=self.source_step,
        )

    def test_defaults_define_the_controlled_10k_to_15k_fork(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_12.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 64)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.source_step, 10_000)
        self.assertEqual(args.max_iters, 15_000)
        self.assertEqual(args.source_learning_rate, 1e-3)
        self.assertEqual(args.control_learning_rate, 1e-3)
        self.assertEqual(args.reduced_learning_rate, 3e-4)
        self.assertEqual(args.eval_interval, 500)
        self.assertEqual(args.eval_iters, 100)
        self.assertEqual(
            args.source_checkpoint,
            STAGE_12.DEFAULT_SOURCE_CHECKPOINT_PATH,
        )
        self.assertEqual(
            args.control_checkpoint_path,
            STAGE_12.DEFAULT_CONTROL_CHECKPOINT_PATH,
        )
        self.assertEqual(
            args.lr_drop_checkpoint_path,
            STAGE_12.DEFAULT_LR_DROP_CHECKPOINT_PATH,
        )
        self.assertEqual(
            len(
                {
                    args.source_checkpoint.resolve(),
                    args.control_checkpoint_path.resolve(),
                    args.lr_drop_checkpoint_path.resolve(),
                }
            ),
            3,
        )

    def test_source_and_branch_outputs_must_use_distinct_paths(self) -> None:
        source = self.directory / "source.pt"
        control = self.directory / "control.pt"
        lr_drop = self.directory / "lr_drop.pt"
        STAGE_12.validate_distinct_paths(source, control, lr_drop)

        collisions = (
            (source, source, lr_drop),
            (source, control, source),
            (source, control, control),
            (
                control.with_name(f".{control.name}.tmp"),
                control,
                lr_drop,
            ),
            (
                source,
                control,
                control.with_name(f".{control.name}.tmp"),
            ),
        )
        for paths in collisions:
            with self.subTest(paths=paths):
                with self.assertRaisesRegex(ValueError, "different paths"):
                    STAGE_12.validate_distinct_paths(*paths)

    def test_source_contract_accepts_only_an_exact_clean_resume(self) -> None:
        valid = STAGE_12.validate_source_checkpoint_contract(
            self.source_checkpoint,
            expected_step=self.source_step,
        )
        self.assertEqual(valid["step"], self.source_step)
        self.assertEqual(valid["best_step"], self.source_step)

        def clone_source() -> dict[object, object]:
            return STAGE_12._STAGE_11.clone_to_cpu(valid)

        wrong_step = clone_source()
        wrong_step["step"] = self.source_step - 1
        wrong_best_step = clone_source()
        wrong_best_step["best_step"] = self.source_step - 1
        missing_generator = clone_source()
        del missing_generator["training_generator_state"]
        missing_cpu_rng = clone_source()
        missing_cpu_rng["rng_state"] = {}
        unknown_provenance = clone_source()
        unknown_provenance["optimizer_provenance_known"] = False
        restarted_optimizer = clone_source()
        restarted_optimizer["optimizer_restart_step"] = 1
        empty_moments = clone_source()
        optimizer_state = empty_moments["optimizer_state_dict"]
        assert isinstance(optimizer_state, dict)
        optimizer_state["state"] = {}

        cases = (
            ("wrong_step", wrong_step, "must be step"),
            ("wrong_best_step", wrong_best_step, "best_step"),
            ("missing_generator", missing_generator, "missing"),
            ("missing_cpu_rng", missing_cpu_rng, "CPU RNG state"),
            ("unknown_provenance", unknown_provenance, "known optimizer"),
            ("restarted_optimizer", restarted_optimizer, "uninterrupted"),
            ("empty_moments", empty_moments, "moment state"),
        )
        for name, payload, message in cases:
            with self.subTest(name=name):
                path = self.directory / f"invalid_{name}.pt"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, message):
                    STAGE_12.validate_source_checkpoint_contract(
                        path,
                        expected_step=self.source_step,
                    )

        with self.assertRaises(FileNotFoundError):
            STAGE_12.validate_source_checkpoint_contract(
                self.directory / "missing.pt",
                expected_step=self.source_step,
            )

    def test_source_learning_rate_metadata_must_match_before_override(
        self,
    ) -> None:
        mismatched_config = replace(
            self.source_config,
            learning_rate=3e-3,
        )
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            STAGE_12.load_branch(
                self.data,
                mismatched_config,
                2,
                1,
                self.device,
                self.source_checkpoint,
                self.make_branch_spec("mismatch", 3e-3),
                expected_source_step=self.source_step,
            )

    def test_each_branch_restores_exact_state_then_changes_only_lr(self) -> None:
        source_payload = self.load_source_payload()
        source_hash = STAGE_12.checkpoint_sha256(self.source_checkpoint)
        control = self.load_branch(
            "control",
            self.source_config.learning_rate,
        )
        control_rng = torch.get_rng_state().clone()
        reduced_lr = 3e-3
        lr_drop = self.load_branch("lr_drop", reduced_lr)
        lr_drop_rng = torch.get_rng_state().clone()

        for branch in (control, lr_drop):
            self.assertEqual(branch.resume.mode, "full")
            self.assertEqual(branch.resume.start_step, self.source_step)
            self.assertTrue(branch.resume.optimizer_restored)
            self.assertTrue(branch.resume.training_generator_restored)
            self.assertIsNone(branch.resume.optimizer_restart_step)
            self.assert_nested_equal(
                branch.model.state_dict(),
                source_payload["model_state_dict"],
            )
            self.assert_nested_equal(
                branch.training_generator.get_state(),
                source_payload["training_generator_state"],
            )

        self.assert_nested_equal(
            control_rng,
            source_payload["rng_state"]["cpu"],
        )
        self.assert_nested_equal(
            lr_drop_rng,
            source_payload["rng_state"]["cpu"],
        )
        self.assert_optimizer_equal_except_lr(
            control.optimizer.state_dict(),
            source_payload["optimizer_state_dict"],
        )
        self.assert_optimizer_equal_except_lr(
            lr_drop.optimizer.state_dict(),
            source_payload["optimizer_state_dict"],
        )
        self.assert_nested_equal(
            control.optimizer.state_dict()["state"],
            lr_drop.optimizer.state_dict()["state"],
        )
        self.assertTrue(
            all(
                group["lr"] == self.source_config.learning_rate
                for group in control.optimizer.param_groups
            )
        )
        self.assertTrue(
            all(
                group["lr"] == reduced_lr
                for group in lr_drop.optimizer.param_groups
            )
        )
        self.assertEqual(control.config, self.source_config)
        self.assertEqual(
            lr_drop.config,
            replace(self.source_config, learning_rate=reduced_lr),
        )

        lr_drop_parameter = next(lr_drop.model.parameters())
        lr_drop_before = lr_drop_parameter.detach().clone()
        with torch.no_grad():
            next(control.model.parameters()).add_(1.0)
        torch.testing.assert_close(
            lr_drop_parameter,
            lr_drop_before,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            STAGE_12.checkpoint_sha256(self.source_checkpoint),
            source_hash,
        )

    def test_lr_override_updates_every_group_without_touching_state(
        self,
    ) -> None:
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = torch.optim.AdamW(
            (
                {"params": [first], "lr": 1e-2, "weight_decay": 0.01},
                {"params": [second], "lr": 2e-2, "weight_decay": 0.02},
            )
        )
        (first.square() + second.square()).backward()
        optimizer.step()
        before = STAGE_12._STAGE_11.clone_to_cpu(optimizer.state_dict())

        STAGE_12.override_optimizer_learning_rate(optimizer, 3e-4)
        after = optimizer.state_dict()

        self.assertEqual(
            [group["lr"] for group in after["param_groups"]],
            [3e-4, 3e-4],
        )
        self.assert_optimizer_equal_except_lr(after, before)

    def test_fixed_evaluation_and_training_streams_are_paired(self) -> None:
        control = self.load_branch(
            "control",
            self.source_config.learning_rate,
        )
        lr_drop = self.load_branch("lr_drop", 3e-3)
        control.model.train()
        control_generator_before = (
            control.training_generator.get_state().clone()
        )
        global_rng_before = torch.get_rng_state().clone()

        first_control_loss = STAGE_12.evaluate_on_fixed_batches(
            control.model,
            self.data,
            control.config,
            self.device,
        )
        second_control_loss = STAGE_12.evaluate_on_fixed_batches(
            control.model,
            self.data,
            control.config,
            self.device,
        )
        initial_drop_loss = STAGE_12.evaluate_on_fixed_batches(
            lr_drop.model,
            self.data,
            lr_drop.config,
            self.device,
        )

        self.assertEqual(first_control_loss, second_control_loss)
        self.assertEqual(first_control_loss, initial_drop_loss)
        self.assertTrue(control.model.training)
        self.assert_nested_equal(
            control.training_generator.get_state(),
            control_generator_before,
        )
        self.assert_nested_equal(torch.get_rng_state(), global_rng_before)

        control_probe = torch.Generator()
        control_probe.set_state(control.training_generator.get_state())
        lr_drop_probe = torch.Generator()
        lr_drop_probe.set_state(lr_drop.training_generator.get_state())
        for _ in range(self.source_config.max_iters - self.source_step):
            control_batch = self.data.get_batch(
                "train",
                batch_size=control.config.batch_size,
                block_size=control.config.block_size,
                device=self.device,
                generator=control_probe,
            )
            lr_drop_batch = self.data.get_batch(
                "train",
                batch_size=lr_drop.config.batch_size,
                block_size=lr_drop.config.block_size,
                device=self.device,
                generator=lr_drop_probe,
            )
            self.assert_nested_equal(control_batch, lr_drop_batch)

        control_records: list[EvaluationRecord] = []
        lr_drop_records: list[EvaluationRecord] = []
        control_result = STAGE_12.train_until(
            control.model,
            control.optimizer,
            self.data,
            control.config,
            self.device,
            control.training_generator,
            start_step=self.source_step,
            on_evaluation=control_records.append,
        )
        lr_drop_result = STAGE_12.train_until(
            lr_drop.model,
            lr_drop.optimizer,
            self.data,
            lr_drop.config,
            self.device,
            lr_drop.training_generator,
            start_step=self.source_step,
            on_evaluation=lr_drop_records.append,
        )

        expected_steps = [2, 3, 4]
        self.assertEqual(
            [record.step for record in control_result.history],
            expected_steps,
        )
        self.assertEqual(
            [record.step for record in lr_drop_result.history],
            expected_steps,
        )
        self.assertEqual(control_records, list(control_result.history))
        self.assertEqual(lr_drop_records, list(lr_drop_result.history))
        self.assertEqual(control_result.initial, lr_drop_result.initial)
        self.assert_nested_equal(
            control.training_generator.get_state(),
            lr_drop.training_generator.get_state(),
        )
        self.assertTrue(
            any(
                not torch.equal(control_value, lr_drop_value)
                for control_value, lr_drop_value in zip(
                    control.model.state_dict().values(),
                    lr_drop.model.state_dict().values(),
                    strict=True,
                )
            )
        )

    def test_no_improvement_still_writes_branch_checkpoint_metadata(
        self,
    ) -> None:
        source_payload = self.load_source_payload()
        source_hash = STAGE_12.checkpoint_sha256(self.source_checkpoint)

        def no_improvement_train_until(
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            data: CharacterData,
            config: TrainingConfig,
            device: torch.device,
            training_generator: torch.Generator,
            *,
            start_step: int,
            on_evaluation: object,
        ) -> TrainingResult:
            del model, optimizer, data, config, device, training_generator
            record = EvaluationRecord(
                step=start_step,
                losses=LossEstimate(
                    train=1.6,
                    val=self.source_val_loss + 0.1,
                ),
            )
            on_evaluation(record)
            return TrainingResult(
                initial=record.losses,
                final=record.losses,
                history=(record,),
            )

        specs = (
            self.make_branch_spec(
                "control",
                self.source_config.learning_rate,
            ),
            self.make_branch_spec("lr_drop", 3e-3),
        )
        with (
            mock.patch.object(
                STAGE_12,
                "train_until",
                side_effect=no_improvement_train_until,
            ),
            mock.patch.object(
                STAGE_12,
                "generate_from_final_model",
                return_value="",
            ),
            mock.patch("builtins.print"),
        ):
            for spec in specs:
                with self.subTest(branch=spec.name):
                    report, _ = STAGE_12.run_branch(
                        self.data,
                        self.source_config,
                        2,
                        1,
                        self.device,
                        0,
                        self.source_checkpoint,
                        source_hash,
                        spec,
                        expected_source_step=self.source_step,
                    )
                    payload = torch.load(
                        spec.checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )

                    self.assertEqual(report.best_step, self.source_step)
                    self.assertEqual(report.best_val_loss, self.source_val_loss)
                    self.assertEqual(payload["step"], self.source_step)
                    self.assertEqual(payload["best_step"], self.source_step)
                    self.assertEqual(
                        payload["best_val_loss"],
                        self.source_val_loss,
                    )
                    self.assertEqual(payload["checkpoint_kind"], "best")
                    self.assertEqual(
                        payload["training_config"]["learning_rate"],
                        spec.learning_rate,
                    )
                    self.assertTrue(
                        all(
                            group["lr"] == spec.learning_rate
                            for group in payload["optimizer_state_dict"][
                                "param_groups"
                            ]
                        )
                    )
                    self.assert_nested_equal(
                        payload["model_state_dict"],
                        source_payload["model_state_dict"],
                    )
                    self.assert_optimizer_equal_except_lr(
                        payload["optimizer_state_dict"],
                        source_payload["optimizer_state_dict"],
                    )
                    self.assert_nested_equal(
                        payload["training_generator_state"],
                        source_payload["training_generator_state"],
                    )

                    experiment = payload["experiment"]
                    self.assertEqual(experiment["stage"], 12)
                    self.assertEqual(experiment["branch"], spec.name)
                    self.assertEqual(
                        experiment["source_checkpoint_sha256"],
                        source_hash,
                    )
                    self.assertEqual(
                        experiment["source_step"],
                        self.source_step,
                    )
                    self.assertEqual(
                        experiment["source_learning_rate"],
                        self.source_config.learning_rate,
                    )
                    self.assertEqual(
                        experiment["branch_learning_rate"],
                        spec.learning_rate,
                    )
                    self.assertEqual(
                        experiment["learning_rate_changed"],
                        spec.learning_rate
                        != self.source_config.learning_rate,
                    )
                    self.assertFalse(
                        spec.checkpoint_path.with_name(
                            f".{spec.checkpoint_path.name}.tmp"
                        ).exists()
                    )

        self.assertEqual(
            STAGE_12.checkpoint_sha256(self.source_checkpoint),
            source_hash,
        )


if __name__ == "__main__":
    unittest.main()
