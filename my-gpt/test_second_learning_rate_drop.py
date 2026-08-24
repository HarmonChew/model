import importlib.util
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


MODULE_PATH = Path(__file__).with_name("13_second_learning_rate_drop.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_13_second_learning_rate_drop",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_13 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_13
SPEC.loader.exec_module(STAGE_13)


class SecondLearningRateDropTests(unittest.TestCase):
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
        self.source_learning_rate = 3e-3
        self.reduced_learning_rate = 1e-3
        self.source_config = TrainingConfig(
            batch_size=2,
            block_size=4,
            n_embd=8,
            learning_rate=self.source_learning_rate,
            max_iters=4,
            eval_interval=1,
            eval_iters=2,
            seed=17,
        )
        self.device = torch.device("cpu")
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.source_checkpoint = self.directory / "stage_12_lr_drop.pt"
        self._write_stage_12_source_checkpoint()

    def make_model(self) -> torch.nn.Module:
        return STAGE_13.GPTLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.source_config.block_size,
            n_embd=self.source_config.n_embd,
            n_head=2,
            n_layer=1,
        )

    def _write_stage_12_source_checkpoint(self) -> None:
        torch.manual_seed(self.source_config.seed)
        model = self.make_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.source_learning_rate,
        )
        training_generator = torch.Generator().manual_seed(self.source_config.seed + 1)

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

        payload = {
            "checkpoint_version": 1,
            "checkpoint_kind": "best",
            "step": self.source_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": self.source_val_loss,
            "best_step": self.source_step,
            "training_generator_state": training_generator.get_state(),
            "rng_state": {"cpu": torch.get_rng_state()},
            "architecture": {
                "block_size": self.source_config.block_size,
                "n_embd": self.source_config.n_embd,
                "n_head": 2,
                "n_layer": 1,
            },
            "training_config": {
                "batch_size": self.source_config.batch_size,
                "learning_rate": self.source_learning_rate,
                "eval_interval": self.source_config.eval_interval,
                "eval_iters": self.source_config.eval_iters,
                "seed": self.source_config.seed,
            },
            "data_fingerprint": STAGE_13.fingerprint_data(self.data),
            "optimizer_restart_step": None,
            "optimizer_provenance_known": True,
            "experiment": {
                "stage": 12,
                "branch": "lr_drop",
                "source_checkpoint_sha256": "a" * 64,
                "source_step": 1,
                "source_learning_rate": 1e-2,
                "branch_learning_rate": self.source_learning_rate,
                "learning_rate_changed": True,
            },
        }
        torch.save(payload, self.source_checkpoint)

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
        return STAGE_13.BranchSpec(
            name=name,
            learning_rate=learning_rate,
            checkpoint_path=self.directory / f"{name}.pt",
        )

    def load_branch(self, name: str, learning_rate: float) -> object:
        return STAGE_13.load_branch(
            self.data,
            self.source_config,
            2,
            1,
            self.device,
            self.source_checkpoint,
            self.make_branch_spec(name, learning_rate),
            expected_source_step=self.source_step,
        )

    def test_defaults_define_the_controlled_13k_to_18k_fork(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_13.parse_args()

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.block_size, 64)
        self.assertEqual(args.n_embd, 64)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.source_step, 13_000)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.source_learning_rate, 3e-4)
        self.assertEqual(args.control_learning_rate, 3e-4)
        self.assertEqual(args.reduced_learning_rate, 1e-4)
        self.assertEqual(args.eval_interval, 500)
        self.assertEqual(args.eval_iters, 100)
        self.assertEqual(
            args.source_checkpoint.name,
            "stage_12_lr_drop_best_checkpoint.pt",
        )
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_13_control_best_checkpoint.pt",
        )
        self.assertEqual(
            args.lr_drop_checkpoint_path.name,
            "stage_13_lr_drop_best_checkpoint.pt",
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

    def test_source_contract_requires_stage_12_lr_drop_provenance(
        self,
    ) -> None:
        valid = STAGE_13.validate_source_checkpoint_contract(
            self.source_checkpoint,
            expected_step=self.source_step,
            expected_learning_rate=self.source_learning_rate,
        )
        self.assertEqual(valid["step"], self.source_step)
        self.assertEqual(valid["best_step"], self.source_step)

        mutated_payloads: list[tuple[str, dict[object, object], str]] = []
        for name, key, value, message in (
            ("wrong_stage", "stage", 11, "stage|Stage 12"),
            ("wrong_branch", "branch", "control", "branch|lr_drop"),
            (
                "wrong_branch_lr",
                "branch_learning_rate",
                9e-4,
                "learning.rate",
            ),
            (
                "not_changed",
                "learning_rate_changed",
                False,
                "change|LR",
            ),
        ):
            payload = self.load_source_payload()
            experiment = payload["experiment"]
            assert isinstance(experiment, dict)
            experiment[key] = value
            mutated_payloads.append((name, payload, message))

        missing_experiment = self.load_source_payload()
        del missing_experiment["experiment"]
        cases = (
            *mutated_payloads,
            ("missing_experiment", missing_experiment, "experiment"),
        )

        for name, payload, message in cases:
            with self.subTest(name=name):
                path = self.directory / f"invalid_{name}.pt"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, message):
                    STAGE_13.validate_source_checkpoint_contract(
                        path,
                        expected_step=self.source_step,
                        expected_learning_rate=self.source_learning_rate,
                    )

        with self.assertRaisesRegex(ValueError, "learning.rate"):
            STAGE_13.validate_source_checkpoint_contract(
                self.source_checkpoint,
                expected_step=self.source_step,
                expected_learning_rate=9e-4,
            )

    def test_each_branch_restores_exact_state_then_changes_only_lr(self) -> None:
        source_payload = self.load_source_payload()
        source_hash = STAGE_13.checkpoint_sha256(self.source_checkpoint)
        control = self.load_branch("control", self.source_learning_rate)
        control_rng = torch.get_rng_state().clone()
        lr_drop = self.load_branch("lr_drop", self.reduced_learning_rate)
        lr_drop_rng = torch.get_rng_state().clone()

        for branch in (control, lr_drop):
            self.assertEqual(branch.resume.mode, "full")
            self.assertEqual(branch.resume.start_step, self.source_step)
            self.assertTrue(branch.resume.optimizer_restored)
            self.assertTrue(branch.resume.training_generator_restored)
            self.assertTrue(branch.resume.optimizer_provenance_known)
            self.assertIsNone(branch.resume.optimizer_restart_step)
            self.assert_nested_equal(
                branch.model.state_dict(),
                source_payload["model_state_dict"],
            )
            self.assert_nested_equal(
                branch.training_generator.get_state(),
                source_payload["training_generator_state"],
            )
            self.assert_optimizer_equal_except_lr(
                branch.optimizer.state_dict(),
                source_payload["optimizer_state_dict"],
            )

        self.assert_nested_equal(control_rng, source_payload["rng_state"]["cpu"])
        self.assert_nested_equal(lr_drop_rng, source_payload["rng_state"]["cpu"])
        self.assert_nested_equal(
            control.optimizer.state_dict()["state"],
            lr_drop.optimizer.state_dict()["state"],
        )
        self.assertEqual(
            {group["lr"] for group in control.optimizer.param_groups},
            {self.source_learning_rate},
        )
        self.assertEqual(
            {group["lr"] for group in lr_drop.optimizer.param_groups},
            {self.reduced_learning_rate},
        )
        self.assertEqual(control.config, self.source_config)
        self.assertEqual(
            lr_drop.config,
            replace(
                self.source_config,
                learning_rate=self.reduced_learning_rate,
            ),
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
            STAGE_13.checkpoint_sha256(self.source_checkpoint),
            source_hash,
        )

    def test_fixed_evaluation_and_training_streams_are_paired(self) -> None:
        control = self.load_branch("control", self.source_learning_rate)
        lr_drop = self.load_branch("lr_drop", self.reduced_learning_rate)
        control.model.train()
        control_generator_before = control.training_generator.get_state().clone()
        global_rng_before = torch.get_rng_state().clone()

        first_control_loss = STAGE_13.evaluate_on_fixed_batches(
            control.model,
            self.data,
            control.config,
            self.device,
        )
        second_control_loss = STAGE_13.evaluate_on_fixed_batches(
            control.model,
            self.data,
            control.config,
            self.device,
        )
        initial_drop_loss = STAGE_13.evaluate_on_fixed_batches(
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

        control_result = STAGE_13.train_until(
            control.model,
            control.optimizer,
            self.data,
            control.config,
            self.device,
            control.training_generator,
            start_step=self.source_step,
            on_evaluation=lambda record: None,
        )
        lr_drop_result = STAGE_13.train_until(
            lr_drop.model,
            lr_drop.optimizer,
            self.data,
            lr_drop.config,
            self.device,
            lr_drop.training_generator,
            start_step=self.source_step,
            on_evaluation=lambda record: None,
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

    def test_no_improvement_writes_stage_13_branch_checkpoints(self) -> None:
        source_payload = self.load_source_payload()
        source_hash = STAGE_13.checkpoint_sha256(self.source_checkpoint)

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

        control_spec = self.make_branch_spec(
            "control",
            self.source_learning_rate,
        )
        lr_drop_spec = self.make_branch_spec(
            "lr_drop",
            self.reduced_learning_rate,
        )
        with (
            mock.patch.object(
                STAGE_13,
                "train_until",
                side_effect=no_improvement_train_until,
            ),
            mock.patch.object(
                STAGE_13,
                "generate_from_final_model",
                return_value="",
            ),
            mock.patch("builtins.print"),
        ):
            report = STAGE_13.run_experiment(
                self.data,
                self.source_config,
                2,
                1,
                self.device,
                0,
                self.source_checkpoint,
                control_spec,
                lr_drop_spec,
                expected_source_step=self.source_step,
            )

        self.assertEqual(report.source_step, self.source_step)
        self.assertEqual(report.source_val_loss, self.source_val_loss)
        for spec, branch_report in (
            (control_spec, report.control),
            (lr_drop_spec, report.lr_drop),
        ):
            with self.subTest(branch=spec.name):
                payload = torch.load(
                    spec.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(branch_report.best_step, self.source_step)
                self.assertEqual(
                    branch_report.best_val_loss,
                    self.source_val_loss,
                )
                self.assertEqual(payload["step"], self.source_step)
                self.assertEqual(payload["best_step"], self.source_step)
                self.assertEqual(payload["best_val_loss"], self.source_val_loss)
                self.assertEqual(payload["checkpoint_kind"], "best")
                self.assertEqual(
                    payload["training_config"]["learning_rate"],
                    spec.learning_rate,
                )
                self.assertEqual(
                    {
                        group["lr"]
                        for group in payload["optimizer_state_dict"]["param_groups"]
                    },
                    {spec.learning_rate},
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
                self.assertEqual(experiment["stage"], 13)
                self.assertEqual(experiment["source_stage"], 12)
                self.assertEqual(experiment["source_branch"], "lr_drop")
                self.assertEqual(experiment["branch"], spec.name)
                self.assertEqual(
                    experiment["source_checkpoint_sha256"],
                    source_hash,
                )
                self.assertEqual(experiment["source_step"], self.source_step)
                self.assertEqual(
                    experiment["source_learning_rate"],
                    self.source_learning_rate,
                )
                self.assertEqual(
                    experiment["branch_learning_rate"],
                    spec.learning_rate,
                )
                self.assertEqual(
                    experiment["learning_rate_changed"],
                    spec.learning_rate != self.source_learning_rate,
                )
                self.assertFalse(
                    spec.checkpoint_path.with_name(
                        f".{spec.checkpoint_path.name}.tmp"
                    ).exists()
                )

        self.assertEqual(
            STAGE_13.checkpoint_sha256(self.source_checkpoint),
            source_hash,
        )


if __name__ == "__main__":
    unittest.main()
