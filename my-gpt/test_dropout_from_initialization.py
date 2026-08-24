import copy
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary
from training import EvaluationRecord, LossEstimate


MODULE_PATH = Path(__file__).with_name("15_dropout_from_initialization.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_15_dropout_from_initialization",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_15 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_15
SPEC.loader.exec_module(STAGE_15)


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


class LearningRateScheduleTests(unittest.TestCase):
    def test_defaults_define_the_from_initialization_experiment(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_15.parse_args()

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
        self.assertEqual(args.control_dropout, 0.0)
        self.assertEqual(args.dropout, 0.1)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertIsNone(args.training_batch_seed)
        self.assertIsNone(args.training_rng_seed)
        self.assertEqual(
            args.control_checkpoint_path.name,
            "stage_15_control_best_checkpoint.pt",
        )
        self.assertEqual(
            args.dropout_checkpoint_path.name,
            "stage_15_dropout_best_checkpoint.pt",
        )

    def test_default_schedule_uses_exact_zero_based_boundaries(self) -> None:
        schedule = STAGE_15.LearningRateSchedule()
        expected = {
            0: 1e-3,
            9_999: 1e-3,
            10_000: 3e-4,
            12_999: 3e-4,
            13_000: 1e-4,
            17_999: 1e-4,
            18_000: 1e-4,
        }
        for update_index, learning_rate in expected.items():
            with self.subTest(update_index=update_index):
                self.assertEqual(
                    schedule.for_update(update_index),
                    learning_rate,
                )

        self.assertEqual(
            schedule.as_metadata(max_iters=18_000),
            [
                {
                    "start_update": 0,
                    "end_update_exclusive": 10_000,
                    "learning_rate": 1e-3,
                },
                {
                    "start_update": 10_000,
                    "end_update_exclusive": 13_000,
                    "learning_rate": 3e-4,
                },
                {
                    "start_update": 13_000,
                    "end_update_exclusive": 18_000,
                    "learning_rate": 1e-4,
                },
            ],
        )
        with self.assertRaisesRegex(ValueError, "update index"):
            schedule.for_update(-1)


class Stage15ProtocolTests(NestedEqualityMixin, unittest.TestCase):
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
        self.schedule = STAGE_15.LearningRateSchedule(
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
            eval_interval=6,
            eval_iters=1,
            seed=101,
        )
        self.training_batch_seed = 202
        self.training_rng_seed = 303
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)

    def make_spec(self, name: str, dropout: float) -> object:
        return STAGE_15.BranchSpec(
            name=name,
            dropout=dropout,
            checkpoint_path=self.directory / f"{name}.pt",
        )

    def create_initial_state(
        self,
    ) -> tuple[dict[str, torch.Tensor], str, int]:
        return STAGE_15.create_shared_initial_state(
            self.data,
            self.config,
            self.n_head,
            self.n_layer,
        )

    def prepare(
        self,
        initial_state: Mapping[str, torch.Tensor],
        name: str,
        dropout: float,
    ) -> object:
        return STAGE_15.prepare_branch(
            self.data,
            self.config,
            self.n_head,
            self.n_layer,
            self.device,
            self.schedule,
            initial_state,
            self.make_spec(name, dropout),
            training_batch_seed=self.training_batch_seed,
        )

    def test_lr_is_set_before_every_zero_based_optimizer_update(self) -> None:
        initial_state, _, _ = self.create_initial_state()
        branch = self.prepare(initial_state, "control", 0.0)
        events: list[tuple[str, int, float]] = []
        step_learning_rates: list[float] = []
        original_step = branch.optimizer.step

        def observe_update(update_index: int, learning_rate: float) -> None:
            active_rates = {
                float(group["lr"]) for group in branch.optimizer.param_groups
            }
            self.assertEqual(active_rates, {learning_rate})
            events.append(("schedule", update_index, learning_rate))

        def recorded_step(*args: object, **kwargs: object) -> object:
            learning_rates = {
                float(group["lr"]) for group in branch.optimizer.param_groups
            }
            self.assertEqual(len(learning_rates), 1)
            learning_rate = learning_rates.pop()
            update_index = len(step_learning_rates)
            step_learning_rates.append(learning_rate)
            events.append(("step", update_index, learning_rate))
            return original_step(*args, **kwargs)

        with mock.patch.object(
            branch.optimizer,
            "step",
            side_effect=recorded_step,
        ):
            result = STAGE_15.train_with_schedule(
                branch.model,
                branch.optimizer,
                self.data,
                self.config,
                self.device,
                branch.training_generator,
                self.schedule,
                on_evaluation=lambda _record: None,
                on_update=observe_update,
            )

        expected_rates = [2e-2, 2e-2, 7e-3, 7e-3, 2e-3, 2e-3]
        self.assertEqual(step_learning_rates, expected_rates)
        self.assertEqual(
            events,
            [
                event
                for update_index, learning_rate in enumerate(expected_rates)
                for event in (
                    ("schedule", update_index, learning_rate),
                    ("step", update_index, learning_rate),
                )
            ],
        )
        self.assertEqual(
            [record.step for record in result.history],
            [0, 2, 4, 6],
        )

    def test_branches_clone_one_identical_but_independent_initialization(
        self,
    ) -> None:
        initial_state, initial_hash, parameter_count = self.create_initial_state()
        control = self.prepare(initial_state, "control", 0.0)
        dropout = self.prepare(initial_state, "dropout", 0.5)

        self.assertEqual(
            STAGE_15.tensor_mapping_sha256(control.model.state_dict()),
            initial_hash,
        )
        self.assertEqual(
            STAGE_15.tensor_mapping_sha256(dropout.model.state_dict()),
            initial_hash,
        )
        self.assert_nested_equal(
            control.model.state_dict(),
            dropout.model.state_dict(),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in control.model.parameters()),
            parameter_count,
        )

        control_parameters = dict(control.model.named_parameters())
        dropout_parameters = dict(dropout.model.named_parameters())
        self.assertEqual(tuple(control_parameters), tuple(dropout_parameters))
        for name in control_parameters:
            with self.subTest(parameter=name):
                control_parameter = control_parameters[name]
                dropout_parameter = dropout_parameters[name]
                self.assertIsNot(control_parameter, dropout_parameter)
                self.assertNotEqual(
                    control_parameter.data_ptr(),
                    dropout_parameter.data_ptr(),
                )
                self.assertNotEqual(
                    control_parameter.data_ptr(),
                    initial_state[name].data_ptr(),
                )

        control_optimizer_parameters = {
            id(parameter)
            for group in control.optimizer.param_groups
            for parameter in group["params"]
        }
        dropout_optimizer_parameters = {
            id(parameter)
            for group in dropout.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertIsNot(control.optimizer, dropout.optimizer)
        self.assertTrue(
            control_optimizer_parameters.isdisjoint(
                dropout_optimizer_parameters,
            )
        )
        self.assertIsNot(control.training_generator, dropout.training_generator)
        self.assert_nested_equal(
            control.training_generator.get_state(),
            dropout.training_generator.get_state(),
        )

        dropout_parameter = next(dropout.model.parameters())
        dropout_before = dropout_parameter.detach().clone()
        with torch.no_grad():
            next(control.model.parameters()).add_(1.0)
        torch.testing.assert_close(
            dropout_parameter,
            dropout_before,
            rtol=0,
            atol=0,
        )

    def test_real_dropout_training_keeps_the_batch_stream_paired(self) -> None:
        initial_state, _, _ = self.create_initial_state()
        control = self.prepare(initial_state, "control", 0.0)
        dropout = self.prepare(initial_state, "dropout", 0.5)
        original_get_batch = CharacterData.get_batch

        def train_and_record(branch: object) -> tuple[object, ...]:
            batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            training_generator = branch.training_generator

            def recording_get_batch(
                data: CharacterData,
                split: str,
                *,
                batch_size: int,
                block_size: int,
                device: torch.device,
                generator: torch.Generator | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                batch = original_get_batch(
                    data,
                    split,
                    batch_size=batch_size,
                    block_size=block_size,
                    device=device,
                    generator=generator,
                )
                if split == "train" and generator is training_generator:
                    batches.append(
                        tuple(tensor.detach().cpu().clone() for tensor in batch)
                    )
                return batch

            STAGE_15.seed_everything(self.training_rng_seed)
            with mock.patch.object(
                CharacterData,
                "get_batch",
                new=recording_get_batch,
            ):
                result = STAGE_15.train_with_schedule(
                    branch.model,
                    branch.optimizer,
                    self.data,
                    self.config,
                    self.device,
                    branch.training_generator,
                    self.schedule,
                    on_evaluation=lambda _record: None,
                )
            return (
                result,
                tuple(batches),
                branch.training_generator.get_state().clone(),
                torch.get_rng_state().clone(),
            )

        control_result, control_batches, control_batch_state, control_rng = (
            train_and_record(control)
        )
        dropout_result, dropout_batches, dropout_batch_state, dropout_rng = (
            train_and_record(dropout)
        )

        self.assertEqual(len(control_batches), self.config.max_iters)
        self.assert_nested_equal(control_batches, dropout_batches)
        self.assert_nested_equal(control_batch_state, dropout_batch_state)
        self.assertEqual(control_result.initial, dropout_result.initial)
        self.assertFalse(torch.equal(control_rng, dropout_rng))
        self.assertTrue(
            any(
                not torch.equal(control_value, dropout_value)
                for control_value, dropout_value in zip(
                    control.model.state_dict().values(),
                    dropout.model.state_dict().values(),
                    strict=True,
                )
            )
        )

    def test_fixed_evaluation_disables_dropout_and_preserves_rng_and_mode(
        self,
    ) -> None:
        initial_state, _, _ = self.create_initial_state()
        control = self.prepare(initial_state, "control", 0.0)
        dropout = self.prepare(initial_state, "dropout", 0.75)
        observed_modes: list[bool] = []
        observed_grad_modes: list[bool] = []

        def observe(module: torch.nn.Module, _inputs: tuple[object, ...]) -> None:
            observed_modes.append(module.training)
            observed_grad_modes.append(torch.is_grad_enabled())

        handle = dropout.model.register_forward_pre_hook(observe)
        self.addCleanup(handle.remove)
        dropout.model.train()
        global_rng_before = torch.get_rng_state().clone()
        training_rng_before = dropout.training_generator.get_state().clone()

        first = STAGE_15.evaluate_on_fixed_batches(
            dropout.model,
            self.data,
            self.config,
            self.device,
        )
        second = STAGE_15.evaluate_on_fixed_batches(
            dropout.model,
            self.data,
            self.config,
            self.device,
        )
        control_loss = STAGE_15.evaluate_on_fixed_batches(
            control.model,
            self.data,
            self.config,
            self.device,
        )

        self.assertEqual(first, second)
        self.assertEqual(first, control_loss)
        self.assertTrue(dropout.model.training)
        self.assertEqual(observed_modes, [False] * 4)
        self.assertEqual(observed_grad_modes, [False] * 4)
        self.assert_nested_equal(torch.get_rng_state(), global_rng_before)
        self.assert_nested_equal(
            dropout.training_generator.get_state(),
            training_rng_before,
        )

        dropout.model.eval()
        STAGE_15.evaluate_on_fixed_batches(
            dropout.model,
            self.data,
            self.config,
            self.device,
        )
        self.assertFalse(dropout.model.training)

    def test_checkpoint_callback_writes_full_provenance_atomically(self) -> None:
        initial_state, initial_hash, _ = self.create_initial_state()
        spec = self.make_spec("dropout", 0.5)
        branch = self.prepare(initial_state, spec.name, spec.dropout)
        checkpoint = STAGE_15.BranchValidationCheckpoint(
            model=branch.model,
            optimizer=branch.optimizer,
            training_generator=branch.training_generator,
            path=spec.checkpoint_path,
            device=self.device,
            config=self.config,
            n_head=self.n_head,
            n_layer=self.n_layer,
            data_fingerprint=STAGE_15.fingerprint_data(self.data),
            optimizer_restart_step=None,
            optimizer_provenance_known=True,
            branch_name=spec.name,
            branch_dropout=spec.dropout,
            initial_state_sha256=initial_hash,
            initialization_seed=self.config.seed,
            training_batch_seed=self.training_batch_seed,
            training_rng_seed=self.training_rng_seed,
            schedule=self.schedule,
            target_step=self.config.max_iters,
        )
        callback = STAGE_15.make_evaluation_callback(
            checkpoint,
            self.schedule,
        )
        record = EvaluationRecord(
            step=0,
            losses=LossEstimate(train=1.5, val=1.75),
        )

        with mock.patch("builtins.print"):
            callback(record)

        temporary_path = spec.checkpoint_path.with_name(
            f".{spec.checkpoint_path.name}.tmp"
        )
        self.assertTrue(spec.checkpoint_path.is_file())
        self.assertFalse(temporary_path.exists())
        step_zero_payload = torch.load(
            spec.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(step_zero_payload["step"], 0)
        self.assertEqual(
            step_zero_payload["training_config"]["learning_rate"],
            self.schedule.for_update(max(0, 0 - 1)),
        )

        boundary_step = self.schedule.first_decay_step
        boundary_record = EvaluationRecord(
            step=boundary_step,
            losses=LossEstimate(train=1.4, val=1.7),
        )
        with mock.patch("builtins.print"):
            callback(boundary_record)

        self.assertFalse(temporary_path.exists())
        payload = torch.load(
            spec.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        expected_schedule = self.schedule.as_metadata(max_iters=self.config.max_iters)
        self.assertEqual(payload["checkpoint_kind"], "best")
        self.assertEqual(payload["step"], boundary_step)
        self.assertEqual(payload["best_step"], boundary_step)
        self.assertEqual(payload["best_val_loss"], 1.7)
        self.assertEqual(
            payload["architecture"]["residual_dropout"],
            spec.dropout,
        )
        self.assertEqual(
            payload["architecture"]["dropout_placement"],
            STAGE_15.DROPOUT_PLACEMENT,
        )
        self.assertEqual(
            payload["initialization"],
            {
                "kind": "shared_random_initialization",
                "seed": self.config.seed,
                "state_sha256": initial_hash,
            },
        )
        self.assertEqual(
            payload["training_config"]["learning_rate_schedule"],
            expected_schedule,
        )
        self.assertEqual(
            payload["training_config"]["learning_rate"],
            self.schedule.for_update(max(0, boundary_step - 1)),
        )
        self.assertEqual(
            payload["training_config"]["training_batch_seed"],
            self.training_batch_seed,
        )
        self.assertEqual(
            payload["training_config"]["training_rng_seed"],
            self.training_rng_seed,
        )

        experiment = payload["experiment"]
        self.assertEqual(experiment["stage"], 15)
        self.assertEqual(experiment["branch"], spec.name)
        self.assertTrue(experiment["from_scratch"])
        self.assertEqual(experiment["comparison_variable"], "residual_dropout")
        self.assertTrue(experiment["identical_initialization"])
        self.assertEqual(experiment["initial_state_sha256"], initial_hash)
        self.assertTrue(experiment["identical_training_batches"])
        self.assertEqual(
            experiment["learning_rate_schedule"],
            expected_schedule,
        )

        validated = STAGE_15.validate_stage_15_checkpoint(
            spec.checkpoint_path,
            spec,
            self.data,
            self.config,
            self.n_head,
            self.n_layer,
            self.schedule,
            initial_state_sha256=initial_hash,
            training_batch_seed=self.training_batch_seed,
            training_rng_seed=self.training_rng_seed,
        )
        self.assertEqual(validated["experiment"], experiment)

        wrong_learning_rate = self.schedule.for_update(boundary_step)
        STAGE_15.set_optimizer_learning_rate(
            branch.optimizer,
            wrong_learning_rate,
        )
        with self.assertRaisesRegex(RuntimeError, "completed step 2"):
            checkpoint._payload(boundary_step)

        mutually_consistent_but_wrong = copy.deepcopy(payload)
        mutually_consistent_but_wrong["training_config"]["learning_rate"] = (
            wrong_learning_rate
        )
        for group in mutually_consistent_but_wrong["optimizer_state_dict"][
            "param_groups"
        ]:
            group["lr"] = wrong_learning_rate
        wrong_path = self.directory / "schedule_wrong.pt"
        torch.save(mutually_consistent_but_wrong, wrong_path)
        with self.assertRaisesRegex(ValueError, "completed step 2"):
            STAGE_15.validate_stage_15_checkpoint(
                wrong_path,
                spec,
                self.data,
                self.config,
                self.n_head,
                self.n_layer,
                self.schedule,
                initial_state_sha256=initial_hash,
                training_batch_seed=self.training_batch_seed,
                training_rng_seed=self.training_rng_seed,
            )

    def test_paired_delta_and_printing_use_dropout_minus_control(self) -> None:
        control = STAGE_15.PreciseValidationResult(
            name="control",
            checkpoint_path=Path("control.pt"),
            checkpoint_sha256="control-hash",
            checkpoint_step=6,
            fixed_panel_loss=2.0,
            mean_loss=37.0,
            standard_error=30.0,
            batch_losses=(1.0, 10.0, 100.0),
        )
        dropout = STAGE_15.PreciseValidationResult(
            name="dropout",
            checkpoint_path=Path("dropout.pt"),
            checkpoint_sha256="dropout-hash",
            checkpoint_step=6,
            fixed_panel_loss=1.5,
            mean_loss=36.5,
            standard_error=30.0,
            batch_losses=(0.5, 9.5, 99.5),
        )
        delta = STAGE_15._STAGE_14._paired_delta(dropout, control)

        self.assertEqual(delta.candidate, "dropout")
        self.assertEqual(delta.baseline, "control")
        self.assertEqual(delta.mean_delta, -0.5)
        self.assertEqual(delta.standard_error, 0.0)
        self.assertEqual(delta.confidence_low, -0.5)
        self.assertEqual(delta.confidence_high, -0.5)

        report = STAGE_15.PreciseValidationReport(
            eval_iters=3,
            seed=404,
            results=(control, dropout),
            adjacent_deltas=(delta,),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            STAGE_15.print_precise_validation(report)
        rendered = output.getvalue()

        self.assertIn("shared panel seed=404", rendered)
        self.assertIn("Dropout is disabled for every measurement.", rendered)
        self.assertIn("delta dropout - control", rendered)
        self.assertIn("-0.500000 +/- 0.000000 SE", rendered)
        self.assertIn("95% CI [-0.500000, -0.500000]", rendered)
        self.assertIn("negative favors dropout", rendered)


if __name__ == "__main__":
    unittest.main()
