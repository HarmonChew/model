import copy
import importlib.util
import math
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary
from training import EvaluationRecord, LossEstimate, TrainingResult


MODULE_PATH = Path(__file__).with_name("14_dropout.py")
SPEC = importlib.util.spec_from_file_location("stage_14_dropout", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE_14 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_14
SPEC.loader.exec_module(STAGE_14)


class NestedEqualityMixin:
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


class DropoutArchitectureTests(NestedEqualityMixin, unittest.TestCase):
    vocab_size = 31
    block_size = 8
    n_embd = 16
    n_head = 4
    n_layer = 3

    def make_model(self, dropout: float) -> nn.Module:
        return STAGE_14.GPTLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
            dropout=dropout,
        )

    def make_stage_13_model(self) -> nn.Module:
        return STAGE_14._STAGE_13.GPTLanguageModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
        )

    def test_defaults_define_the_controlled_17k_to_22k_fork(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_14.parse_args()

        self.assertEqual(args.source_step, 17_000)
        self.assertEqual(args.max_iters, 22_000)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.control_dropout, 0.0)
        self.assertEqual(args.dropout, 0.1)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(
            args.source_checkpoint.name,
            "stage_13_lr_drop_best_checkpoint.pt",
        )

    def test_exactly_two_dropout_modules_per_residual_block(self) -> None:
        probability = 0.1
        model = self.make_model(probability)

        expected_names = {
            name
            for block_index in range(self.n_layer)
            for name in (
                f"blocks.{block_index}.sa.dropout",
                f"blocks.{block_index}.ffwd.net.3",
            )
        }
        actual_names = {
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Dropout)
        }
        self.assertEqual(actual_names, expected_names)

        for block in model.blocks:
            self.assertIsInstance(block.sa.dropout, nn.Dropout)
            self.assertEqual(block.sa.dropout.p, probability)
            self.assertIsInstance(block.sa.proj, nn.Linear)
            self.assertIs(list(block.sa.children())[-1], block.sa.dropout)

            self.assertEqual(len(block.ffwd.net), 4)
            self.assertIsInstance(block.ffwd.net[0], nn.Linear)
            self.assertIsInstance(block.ffwd.net[1], nn.GELU)
            self.assertIsInstance(block.ffwd.net[2], nn.Linear)
            self.assertIsInstance(block.ffwd.net[3], nn.Dropout)
            self.assertEqual(block.ffwd.net[3].p, probability)

    def test_attention_dropout_receives_the_projected_head_output(self) -> None:
        attention = STAGE_14.MultiHeadAttention(
            n_embd=self.n_embd,
            n_head=self.n_head,
            block_size=self.block_size,
            dropout=0.25,
        )
        projected: list[torch.Tensor] = []
        dropout_inputs: list[torch.Tensor] = []

        projection_handle = attention.proj.register_forward_hook(
            lambda _module, _inputs, output: projected.append(output.detach().clone())
        )
        dropout_handle = attention.dropout.register_forward_pre_hook(
            lambda _module, inputs: dropout_inputs.append(inputs[0].detach().clone())
        )
        self.addCleanup(projection_handle.remove)
        self.addCleanup(dropout_handle.remove)

        attention(torch.randn(2, self.block_size, self.n_embd))

        self.assertEqual(len(projected), 1)
        self.assertEqual(len(dropout_inputs), 1)
        torch.testing.assert_close(
            dropout_inputs[0],
            projected[0],
            rtol=0,
            atol=0,
        )

    def test_p_zero_state_and_outputs_exactly_match_stage_13(self) -> None:
        seed = 2026
        torch.manual_seed(seed)
        stage_13 = self.make_stage_13_model()
        torch.manual_seed(seed)
        stage_14 = self.make_model(0.0)

        self.assertEqual(
            tuple(stage_14.state_dict()),
            tuple(stage_13.state_dict()),
        )
        self.assert_nested_equal(stage_14.state_dict(), stage_13.state_dict())
        self.assertEqual(
            tuple(name for name, _ in stage_14.named_parameters()),
            tuple(name for name, _ in stage_13.named_parameters()),
        )

        inputs = (
            torch.arange(3 * self.block_size)
            .reshape(3, self.block_size)
            .remainder(self.vocab_size)
        )
        targets = (inputs + 1).remainder(self.vocab_size)
        stage_13.train()
        stage_14.train()
        reference_logits, reference_loss = stage_13(inputs, targets)
        actual_logits, actual_loss = stage_14(inputs, targets)

        torch.testing.assert_close(actual_logits, reference_logits, rtol=0, atol=0)
        assert actual_loss is not None and reference_loss is not None
        torch.testing.assert_close(actual_loss, reference_loss, rtol=0, atol=0)

    def test_training_is_stochastic_but_eval_matches_p_zero(self) -> None:
        torch.manual_seed(91)
        stochastic = self.make_model(0.5)
        no_dropout = self.make_model(0.0)
        no_dropout.load_state_dict(stochastic.state_dict(), strict=True)
        inputs = torch.randint(
            self.vocab_size,
            (4, self.block_size),
            generator=torch.Generator().manual_seed(7),
        )

        stochastic.train()
        first_train, _ = stochastic(inputs)
        second_train, _ = stochastic(inputs)
        self.assertFalse(torch.equal(first_train, second_train))

        stochastic.eval()
        no_dropout.eval()
        first_eval, _ = stochastic(inputs)
        second_eval, _ = stochastic(inputs)
        no_dropout_eval, _ = no_dropout(inputs)
        torch.testing.assert_close(first_eval, second_eval, rtol=0, atol=0)
        torch.testing.assert_close(first_eval, no_dropout_eval, rtol=0, atol=0)

    def test_default_parameter_count_is_unchanged(self) -> None:
        constructor = {
            "vocab_size": 65,
            "block_size": 64,
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 4,
        }
        stage_13 = STAGE_14._STAGE_13.GPTLanguageModel(**constructor)
        control = STAGE_14.GPTLanguageModel(**constructor, dropout=0.0)
        dropout = STAGE_14.GPTLanguageModel(**constructor, dropout=0.1)

        counts = {
            sum(parameter.numel() for parameter in model.parameters())
            for model in (stage_13, control, dropout)
        }
        self.assertEqual(counts, {211_777})
        self.assertEqual(tuple(control.state_dict()), tuple(dropout.state_dict()))
        self.assertEqual(tuple(control.state_dict()), tuple(stage_13.state_dict()))

    def test_invalid_probabilities_are_rejected_everywhere(self) -> None:
        factories = {
            "attention": lambda probability: STAGE_14.MultiHeadAttention(
                n_embd=8,
                n_head=2,
                block_size=4,
                dropout=probability,
            ),
            "feed_forward": lambda probability: STAGE_14.FeedForward(
                n_embd=8,
                dropout=probability,
            ),
            "block": lambda probability: STAGE_14.Block(
                n_embd=8,
                n_head=2,
                block_size=4,
                dropout=probability,
            ),
            "model": lambda probability: STAGE_14.GPTLanguageModel(
                vocab_size=5,
                block_size=4,
                n_embd=8,
                n_head=2,
                n_layer=1,
                dropout=probability,
            ),
            "branch": lambda probability: STAGE_14.BranchSpec(
                name="branch",
                dropout=probability,
                checkpoint_path=Path("branch.pt"),
            ),
        }
        invalid = (-0.01, 1.0, 2.0, math.inf, -math.inf, math.nan)

        for factory_name, factory in factories.items():
            for probability in invalid:
                with self.subTest(factory=factory_name, probability=probability):
                    with self.assertRaises(ValueError):
                        factory(probability)

        self.assertEqual(STAGE_14._validate_dropout(0.0), 0.0)
        self.assertLess(STAGE_14._validate_dropout(0.999), 1.0)

    @unittest.skipUnless(
        STAGE_14.DEFAULT_SOURCE_CHECKPOINT_PATH.is_file(),
        "real Stage 13 checkpoint is not available",
    )
    def test_real_stage_13_checkpoint_loads_strictly_with_dropout(self) -> None:
        validated = STAGE_14.validate_source_checkpoint_contract(
            STAGE_14.DEFAULT_SOURCE_CHECKPOINT_PATH,
            expected_step=STAGE_14.DEFAULT_SOURCE_STEP,
            expected_learning_rate=STAGE_14.DEFAULT_LEARNING_RATE,
        )
        self.assertEqual(validated["step"], STAGE_14.DEFAULT_SOURCE_STEP)
        checkpoint = torch.load(
            STAGE_14.DEFAULT_SOURCE_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint["model_state_dict"]

        for probability in (0.0, 0.1):
            with self.subTest(dropout=probability):
                model = STAGE_14.GPTLanguageModel(
                    vocab_size=65,
                    block_size=64,
                    n_embd=64,
                    n_head=4,
                    n_layer=4,
                    dropout=probability,
                )
                model.load_state_dict(state, strict=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    211_777,
                )
                self.assertTrue(optimizer.state)


class Stage14CheckpointFixture(NestedEqualityMixin):
    def setUp(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz\n" * 40
        vocabulary = CharacterVocabulary.from_text(text)
        token_ids = torch.tensor(vocabulary.encode(text), dtype=torch.long)
        self.data = CharacterData(
            vocabulary=vocabulary,
            train_data=token_ids[:800],
            val_data=token_ids[800:],
            num_characters=len(text),
        )
        self.source_step = 2
        self.source_val_loss = 1.75
        self.learning_rate = 1e-3
        self.config = TrainingConfig(
            batch_size=2,
            block_size=4,
            n_embd=8,
            learning_rate=self.learning_rate,
            max_iters=4,
            eval_interval=1,
            eval_iters=2,
            seed=17,
        )
        self.device = torch.device("cpu")
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.source_checkpoint = self.directory / "stage_13_lr_drop.pt"
        self._write_stage_13_source_checkpoint()

    def make_stage_13_model(self) -> nn.Module:
        return STAGE_14._STAGE_13.GPTLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
            n_head=2,
            n_layer=1,
        )

    def _write_stage_13_source_checkpoint(self) -> None:
        torch.manual_seed(self.config.seed)
        model = self.make_stage_13_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
        )
        training_generator = torch.Generator().manual_seed(self.config.seed + 1)

        for _ in range(self.source_step):
            inputs, targets = self.data.get_batch(
                "train",
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
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
                "block_size": self.config.block_size,
                "n_embd": self.config.n_embd,
                "n_head": 2,
                "n_layer": 1,
            },
            "training_config": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.learning_rate,
                "eval_interval": self.config.eval_interval,
                "eval_iters": self.config.eval_iters,
                "seed": self.config.seed,
            },
            "data_fingerprint": STAGE_14.fingerprint_data(self.data),
            "optimizer_restart_step": None,
            "optimizer_provenance_known": True,
            "experiment": {
                "stage": 13,
                "branch": "lr_drop",
                "source_stage": 12,
                "source_branch": "lr_drop",
                "source_checkpoint_sha256": "a" * 64,
                "source_step": 1,
                "source_learning_rate": 3e-3,
                "branch_learning_rate": self.learning_rate,
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

    def make_spec(self, name: str, dropout: float) -> object:
        return STAGE_14.BranchSpec(
            name=name,
            dropout=dropout,
            checkpoint_path=self.directory / f"{name}.pt",
        )


class Stage14CheckpointTests(Stage14CheckpointFixture, unittest.TestCase):
    def test_precise_inputs_cannot_overlap_branch_outputs(self) -> None:
        control_output = self.directory / "control.pt"
        dropout_output = self.directory / "dropout.pt"
        with self.assertRaisesRegex(ValueError, "must use different paths"):
            STAGE_14.validate_distinct_paths(
                self.source_checkpoint,
                control_output,
                dropout_output,
                additional_input_paths=(control_output,),
            )

    def test_source_contract_requires_stage_13_lr_drop_provenance(self) -> None:
        valid = STAGE_14.validate_source_checkpoint_contract(
            self.source_checkpoint,
            expected_step=self.source_step,
            expected_learning_rate=self.learning_rate,
        )
        self.assertEqual(valid["step"], self.source_step)

        provenance_cases = (
            ("stage", 12),
            ("branch", "control"),
            ("source_stage", 11),
            ("source_branch", "control"),
            ("branch_learning_rate", 9e-4),
            ("learning_rate_changed", False),
        )
        for key, invalid_value in provenance_cases:
            with self.subTest(key=key):
                payload = copy.deepcopy(self.load_source_payload())
                payload["experiment"][key] = invalid_value
                path = self.directory / f"wrong_{key}.pt"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, key):
                    STAGE_14.validate_source_checkpoint_contract(
                        path,
                        expected_step=self.source_step,
                        expected_learning_rate=self.learning_rate,
                    )

    def test_synthetic_stage_13_checkpoint_restores_both_branches_strictly(
        self,
    ) -> None:
        source = self.load_source_payload()
        original_rng = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, original_rng)

        for name, probability in (("control", 0.0), ("dropout", 0.1)):
            with self.subTest(branch=name):
                branch = STAGE_14.load_branch(
                    self.data,
                    self.config,
                    2,
                    1,
                    self.device,
                    self.source_checkpoint,
                    self.make_spec(name, probability),
                    expected_source_step=self.source_step,
                )
                self.assert_nested_equal(
                    branch.model.state_dict(),
                    source["model_state_dict"],
                )
                self.assert_nested_equal(
                    branch.optimizer.state_dict(),
                    source["optimizer_state_dict"],
                )
                self.assert_nested_equal(
                    branch.training_generator.get_state(),
                    source["training_generator_state"],
                )
                self.assertEqual(branch.dropout, probability)
                dropout_modules = [
                    module
                    for module in branch.model.modules()
                    if isinstance(module, nn.Dropout)
                ]
                self.assertEqual(len(dropout_modules), 2)
                self.assertEqual(
                    {module.p for module in dropout_modules}, {probability}
                )

    def test_no_improvement_writes_full_dropout_provenance(self) -> None:
        source = self.load_source_payload()
        source_hash = STAGE_14.checkpoint_sha256(self.source_checkpoint)

        def no_improvement_train_until(
            model: nn.Module,
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
                    val=self.source_val_loss,
                ),
            )
            on_evaluation(record)
            return TrainingResult(
                initial=record.losses,
                final=record.losses,
                history=(record,),
            )

        control_spec = self.make_spec("control", 0.0)
        dropout_spec = self.make_spec("dropout", 0.1)
        with (
            mock.patch.object(
                STAGE_14,
                "train_until",
                side_effect=no_improvement_train_until,
            ),
            mock.patch.object(
                STAGE_14,
                "generate_from_final_model",
                return_value="",
            ),
            mock.patch("builtins.print"),
        ):
            report = STAGE_14.run_experiment(
                self.data,
                self.config,
                2,
                1,
                self.device,
                0,
                self.source_checkpoint,
                control_spec,
                dropout_spec,
                expected_source_step=self.source_step,
            )

        self.assertEqual(report.source_checkpoint_sha256, source_hash)
        for spec, branch_report in (
            (control_spec, report.control),
            (dropout_spec, report.dropout),
        ):
            with self.subTest(branch=spec.name):
                payload = torch.load(
                    spec.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(branch_report.best_step, self.source_step)
                self.assertEqual(branch_report.best_val_loss, self.source_val_loss)
                self.assertEqual(payload["step"], self.source_step)
                self.assertEqual(payload["best_step"], self.source_step)
                self.assertEqual(payload["best_val_loss"], self.source_val_loss)
                self.assert_nested_equal(
                    payload["model_state_dict"],
                    source["model_state_dict"],
                )
                self.assert_nested_equal(
                    payload["optimizer_state_dict"],
                    source["optimizer_state_dict"],
                )

                self.assertEqual(
                    payload["architecture"]["residual_dropout"],
                    spec.dropout,
                )
                self.assertEqual(
                    payload["architecture"]["dropout_placement"],
                    STAGE_14.DROPOUT_PLACEMENT,
                )
                self.assertEqual(
                    payload["training_config"]["residual_dropout"],
                    spec.dropout,
                )
                experiment = payload["experiment"]
                self.assertEqual(experiment["stage"], 14)
                self.assertEqual(experiment["branch"], spec.name)
                self.assertEqual(experiment["source_stage"], 13)
                self.assertEqual(experiment["source_branch"], "lr_drop")
                self.assertEqual(
                    experiment["source_checkpoint_sha256"],
                    source_hash,
                )
                self.assertEqual(experiment["source_step"], self.source_step)
                self.assertEqual(
                    experiment["source_learning_rate"],
                    self.learning_rate,
                )
                self.assertEqual(
                    experiment["branch_learning_rate"],
                    self.learning_rate,
                )
                self.assertFalse(experiment["learning_rate_changed"])
                self.assertEqual(experiment["source_residual_dropout"], 0.0)
                self.assertEqual(
                    experiment["branch_residual_dropout"],
                    spec.dropout,
                )
                self.assertEqual(
                    experiment["dropout_changed"],
                    spec.dropout != 0.0,
                )
                self.assertEqual(
                    experiment["dropout_placement"],
                    STAGE_14.DROPOUT_PLACEMENT,
                )
                self.assertFalse(
                    spec.checkpoint_path.with_name(
                        f".{spec.checkpoint_path.name}.tmp"
                    ).exists()
                )

        self.assertEqual(
            STAGE_14.checkpoint_sha256(self.source_checkpoint),
            source_hash,
        )


class PreciseValidationTests(Stage14CheckpointFixture, unittest.TestCase):
    def test_precise_evaluator_uses_eval_no_grad_and_preserves_rng_mode(
        self,
    ) -> None:
        model = STAGE_14.GPTLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
            n_head=2,
            n_layer=1,
            dropout=0.75,
        )
        observed_modes: list[bool] = []
        observed_grad_modes: list[bool] = []

        def observe(module: nn.Module, _inputs: tuple[object, ...]) -> None:
            observed_modes.append(module.training)
            observed_grad_modes.append(torch.is_grad_enabled())

        handle = model.register_forward_pre_hook(observe)
        self.addCleanup(handle.remove)
        model.train()
        rng_before = torch.get_rng_state().clone()
        first = STAGE_14.estimate_validation_precise(
            model,
            self.data,
            self.config,
            self.device,
            eval_iters=5,
            seed=404,
        )
        second = STAGE_14.estimate_validation_precise(
            model,
            self.data,
            self.config,
            self.device,
            eval_iters=5,
            seed=404,
        )

        self.assertEqual(first, second)
        self.assertEqual(observed_modes, [False] * 10)
        self.assertEqual(observed_grad_modes, [False] * 10)
        self.assertTrue(model.training)
        self.assert_nested_equal(torch.get_rng_state(), rng_before)

        model.eval()
        STAGE_14.estimate_validation_precise(
            model,
            self.data,
            self.config,
            self.device,
            eval_iters=1,
            seed=405,
        )
        self.assertFalse(model.training)

    def test_precise_comparison_pairs_identical_checkpoints(self) -> None:
        twin_checkpoint = self.directory / "stage_13_twin.pt"
        torch.save(self.load_source_payload(), twin_checkpoint)
        report = STAGE_14.run_precise_validation(
            self.data,
            self.config,
            2,
            1,
            self.device,
            (
                STAGE_14.PreciseCheckpointSpec(
                    name="baseline",
                    checkpoint_path=self.source_checkpoint,
                ),
                STAGE_14.PreciseCheckpointSpec(
                    name="twin",
                    checkpoint_path=twin_checkpoint,
                ),
            ),
            eval_iters=7,
            seed=909,
        )

        self.assertEqual(report.eval_iters, 7)
        self.assertEqual(report.seed, 909)
        self.assertEqual(len(report.results), 2)
        self.assertEqual(
            report.results[0].batch_losses,
            report.results[1].batch_losses,
        )
        self.assertEqual(report.results[0].mean_loss, report.results[1].mean_loss)
        self.assertEqual(len(report.adjacent_deltas), 1)
        delta = report.adjacent_deltas[0]
        self.assertEqual(delta.baseline, "baseline")
        self.assertEqual(delta.candidate, "twin")
        self.assertEqual(delta.mean_delta, 0.0)
        self.assertEqual(delta.standard_error, 0.0)
        self.assertEqual(delta.confidence_low, 0.0)
        self.assertEqual(delta.confidence_high, 0.0)

    def test_precise_evaluator_rejects_invalid_sampling_arguments(self) -> None:
        model = STAGE_14.GPTLanguageModel(
            vocab_size=self.data.vocabulary.size,
            block_size=self.config.block_size,
            n_embd=self.config.n_embd,
            n_head=2,
            n_layer=1,
            dropout=0.1,
        )
        with self.assertRaisesRegex(ValueError, "eval_iters"):
            STAGE_14.estimate_validation_precise(
                model,
                self.data,
                self.config,
                self.device,
                eval_iters=0,
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            STAGE_14.estimate_validation_precise(
                model,
                self.data,
                self.config,
                self.device,
                eval_iters=1,
                seed=-1,
            )


if __name__ == "__main__":
    unittest.main()
