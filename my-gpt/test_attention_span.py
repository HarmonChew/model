from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("18_attention_span.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_18_attention_span",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_18 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_18
SPEC.loader.exec_module(STAGE_18)


def make_character_data(
    *,
    vocabulary_size: int = 5,
    train_tokens: int = 400,
    val_tokens: int = 160,
) -> CharacterData:
    chars = "".join(chr(32 + index) for index in range(vocabulary_size))
    vocabulary = CharacterVocabulary.from_text(chars)
    token_ids = torch.arange(train_tokens + val_tokens) % vocabulary_size
    return CharacterData(
        vocabulary=vocabulary,
        train_data=token_ids[:train_tokens].to(torch.long),
        val_data=token_ids[train_tokens:].to(torch.long),
        num_characters=train_tokens + val_tokens,
    )


def make_specs(
    directory: Path,
    *,
    local_context: int = 4,
) -> tuple[STAGE_18.AttentionSpec, STAGE_18.AttentionSpec]:
    return (
        STAGE_18.AttentionSpec(
            f"local-{local_context}",
            local_context,
            directory / "local.pt",
        ),
        STAGE_18.AttentionSpec(
            f"full-{2 * local_context}",
            None,
            directory / "full.pt",
        ),
    )


class Stage18DefaultsAndMaskTests(unittest.TestCase):
    def test_defaults_define_one_geometry_and_one_mask_treatment(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_18.parse_args()

        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.block_size, 128)
        self.assertEqual(args.local_context, 64)
        self.assertEqual(args.n_embd, 64)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.initial_learning_rate, 1e-3)
        self.assertEqual(args.first_decay_step, 10_000)
        self.assertEqual(args.middle_learning_rate, 3e-4)
        self.assertEqual(args.second_decay_step, 13_000)
        self.assertEqual(args.final_learning_rate, 1e-4)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(
            args.local_checkpoint_path.name,
            "stage_18_local_64_best_checkpoint.pt",
        )
        self.assertEqual(
            args.full_checkpoint_path.name,
            "stage_18_full_128_best_checkpoint.pt",
        )

        with tempfile.TemporaryDirectory() as directory_name:
            local, full = make_specs(Path(directory_name), local_context=64)
            config = TrainingConfig(
                batch_size=args.batch_size,
                block_size=args.block_size,
                n_embd=args.n_embd,
                learning_rate=args.initial_learning_rate,
                max_iters=args.max_iters,
                eval_interval=500,
                eval_iters=1,
                seed=1_337,
            )
            targets, bands = STAGE_18.validate_attention_protocol(
                config,
                local,
                full,
            )
        self.assertEqual(targets, 2_048)
        self.assertEqual(targets * args.max_iters, 36_864_000)
        self.assertEqual(
            bands,
            ((0, 32), (32, 64), (64, 96), (96, 128)),
        )
        self.assertEqual(STAGE_18.DEFAULT_DROPOUT, 0.0)

    def test_local_mask_truth_table_boundaries_and_validation(self) -> None:
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        actual = STAGE_18.build_attention_mask(6, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(tuple(actual.shape), (6, 6))
        self.assertEqual(actual.sum(dim=1).tolist(), [1, 2, 3, 3, 3, 3])
        torch.testing.assert_close(
            STAGE_18.build_attention_mask(6, 1),
            torch.eye(6, dtype=torch.bool),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            STAGE_18.build_attention_mask(6, 6),
            STAGE_18.build_attention_mask(6, None),
            rtol=0,
            atol=0,
        )

        canonical = STAGE_18.build_attention_mask(128, 64)
        self.assertFalse(canonical[64, 0].item())
        self.assertTrue(canonical[64, 1].item())
        self.assertTrue(canonical[64, 64].item())
        self.assertFalse(canonical[64, 65].item())
        self.assertEqual(
            STAGE_18.theoretical_receptive_field_width(4, 64),
            253,
        )
        self.assertAlmostEqual(
            STAGE_18._paired_t_95_critical_value(2),
            12.706204736,
        )
        self.assertAlmostEqual(
            STAGE_18._paired_t_95_critical_value(500),
            1.964729,
            places=5,
        )

        for invalid in (0, -1, 7, True, 2.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    STAGE_18.build_attention_mask(6, invalid)
        for invalid_block in (0, -1, True, 2.5):
            with self.subTest(invalid_block=invalid_block):
                with self.assertRaises(ValueError):
                    STAGE_18.build_attention_mask(invalid_block, None)
        for n_layer, span in ((0, 64), (4, 0), (True, 64), (4, 2.5)):
            with self.subTest(n_layer=n_layer, span=span):
                with self.assertRaises(ValueError):
                    STAGE_18.theoretical_receptive_field_width(n_layer, span)
        for sample_count in (0, 1, True, 2.5):
            with self.subTest(sample_count=sample_count):
                with self.assertRaises(ValueError):
                    STAGE_18._paired_t_95_critical_value(sample_count)

    def test_attention_weights_follow_the_mask_in_every_head(self) -> None:
        data = make_character_data()
        config = TrainingConfig(
            batch_size=1,
            block_size=8,
            n_embd=8,
            learning_rate=1e-3,
            max_iters=3,
            eval_interval=1,
            eval_iters=1,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            local, full = make_specs(Path(directory_name))
            local_model = STAGE_18._build_model(data, config, 2, 2, local)
            full_model = STAGE_18._build_model(data, config, 2, 2, full)

        expected_local = STAGE_18.build_attention_mask(8, 4)
        expected_full = STAGE_18.build_attention_mask(8, None)
        for model, expected in (
            (local_model, expected_local),
            (full_model, expected_full),
        ):
            for block in model.blocks:
                for head in block.sa.heads:
                    torch.testing.assert_close(
                        head.tril.bool(),
                        expected,
                        rtol=0,
                        atol=0,
                    )
                    torch.nn.init.zeros_(head.key.weight)
                    torch.nn.init.zeros_(head.query.weight)

        inputs = torch.zeros((1, 8), dtype=torch.long)
        local_weights = local_model.get_attention_weights(inputs)
        full_weights = full_model.get_attention_weights(inputs)
        torch.testing.assert_close(
            local_weights[0, 0, 4],
            torch.tensor([0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            full_weights[0, 0, 4],
            torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]),
        )
        self.assertTrue(torch.isfinite(local_weights).all().item())
        self.assertTrue(torch.isfinite(full_weights).all().item())

    def test_protocol_rejects_geometry_and_output_aliases(self) -> None:
        config = TrainingConfig(
            batch_size=2,
            block_size=8,
            n_embd=8,
            learning_rate=1e-3,
            max_iters=3,
            eval_interval=1,
            eval_iters=1,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            local, full = make_specs(directory)
            STAGE_18.validate_attention_protocol(config, local, full)

            with self.assertRaises(ValueError):
                STAGE_18.validate_attention_protocol(
                    config,
                    STAGE_18.AttentionSpec("local", 3, directory / "a.pt"),
                    full,
                )
            with self.assertRaises(ValueError):
                STAGE_18.validate_attention_protocol(
                    config,
                    local,
                    STAGE_18.AttentionSpec("full", 8, directory / "b.pt"),
                )
            with self.assertRaises(ValueError):
                STAGE_18.validate_attention_protocol(
                    config,
                    STAGE_18.AttentionSpec("same", 4, directory / "a.pt"),
                    STAGE_18.AttentionSpec("same", None, directory / "b.pt"),
                )
            with self.assertRaises(ValueError):
                STAGE_18.validate_attention_protocol(
                    config,
                    STAGE_18.AttentionSpec("local", 4, directory / "same.pt"),
                    STAGE_18.AttentionSpec("full", None, directory / "same.pt"),
                )
            with self.assertRaises(ValueError):
                STAGE_18.validate_attention_protocol(
                    config,
                    STAGE_18.AttentionSpec(
                        "local",
                        4,
                        directory / ".full.pt.tmp",
                    ),
                    STAGE_18.AttentionSpec("full", None, directory / "full.pt"),
                )


class Stage18InitializationAndLossTests(unittest.TestCase):
    def test_initialization_is_identical_except_for_mask_buffers(self) -> None:
        data = make_character_data(
            vocabulary_size=65,
            train_tokens=500,
            val_tokens=200,
        )
        config = TrainingConfig(
            batch_size=16,
            block_size=128,
            n_embd=64,
            learning_rate=1e-3,
            max_iters=18_000,
            eval_interval=500,
            eval_iters=1,
            seed=1_337,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            local, full = make_specs(Path(directory_name), local_context=64)
            initialization = STAGE_18.create_paired_initial_states(
                data,
                config,
                4,
                4,
                local,
                full,
            )
            local_model = STAGE_18._build_model(data, config, 4, 4, local)
            full_model = STAGE_18._build_model(data, config, 4, 4, full)
            local_model.load_state_dict(initialization.local, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(initialization.parameter_count, 215_873)
        self.assertNotEqual(
            initialization.local_state_sha256,
            initialization.full_state_sha256,
        )
        self.assertNotEqual(
            initialization.local_mask_sha256,
            initialization.full_mask_sha256,
        )
        self.assertTrue(initialization.differing_state_names)
        self.assertTrue(
            all(name.endswith(".tril") for name in initialization.differing_state_names)
        )
        self.assertEqual(len(initialization.differing_state_names), 16)

        for name, local_tensor in initialization.local.items():
            full_tensor = initialization.full[name]
            with self.subTest(state=name):
                self.assertNotEqual(local_tensor.data_ptr(), full_tensor.data_ptr())
                if not name.endswith(".tril"):
                    torch.testing.assert_close(
                        local_tensor,
                        full_tensor,
                        rtol=0,
                        atol=0,
                    )
        torch.testing.assert_close(
            initialization.local["position_embedding_table.weight"],
            initialization.full["position_embedding_table.weight"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            STAGE_18.tensor_mapping_sha256(
                STAGE_18._clone_parameters(local_model)
            ),
            initialization.parameter_state_sha256,
        )
        self.assertEqual(
            STAGE_18.tensor_mapping_sha256(
                STAGE_18._clone_parameters(full_model)
            ),
            initialization.parameter_state_sha256,
        )
        difference = STAGE_18.verify_initial_first_half_equivalence(
            local_model,
            full_model,
            data,
            config,
            torch.device("cpu"),
            seed=1_343,
        )
        self.assertLessEqual(difference, 1e-6)
        self.assertEqual(
            STAGE_18.inspect_attention_shape(
                local_model,
                config,
                local,
                4,
                torch.device("cpu"),
            ),
            (16, 4, 128, 128),
        )
        self.assertEqual(
            STAGE_18.inspect_attention_shape(
                full_model,
                config,
                full,
                4,
                torch.device("cpu"),
            ),
            (16, 4, 128, 128),
        )
        self.assertEqual(math.prod((16, 4, 128, 128)), 1_048_576)

    def test_first_half_objective_has_identical_initial_gradients(self) -> None:
        data = make_character_data()
        config = TrainingConfig(
            batch_size=2,
            block_size=8,
            n_embd=8,
            learning_rate=1e-3,
            max_iters=3,
            eval_interval=1,
            eval_iters=1,
            seed=23,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            local, full = make_specs(Path(directory_name))
            initialization = STAGE_18.create_paired_initial_states(
                data, config, 2, 2, local, full
            )
            models = (
                STAGE_18._build_model(data, config, 2, 2, local),
                STAGE_18._build_model(data, config, 2, 2, full),
            )
            models[0].load_state_dict(initialization.local, strict=True)
            models[1].load_state_dict(initialization.full, strict=True)

        generator = torch.Generator().manual_seed(41)
        inputs, targets = data.get_batch(
            "train",
            batch_size=2,
            block_size=8,
            device=torch.device("cpu"),
            generator=generator,
        )
        prefix_logits: list[torch.Tensor] = []
        for model in models:
            logits, _ = model(inputs)
            prefix_logits.append(logits[:, :4].detach())
            loss = F.cross_entropy(
                logits[:, :4].reshape(-1, logits.shape[-1]),
                targets[:, :4].reshape(-1),
            )
            loss.backward()
        torch.testing.assert_close(
            prefix_logits[0], prefix_logits[1], rtol=0, atol=0
        )
        for (local_name, local_parameter), (full_name, full_parameter) in zip(
            models[0].named_parameters(),
            models[1].named_parameters(),
            strict=True,
        ):
            self.assertEqual(local_name, full_name)
            assert local_parameter.grad is not None
            assert full_parameter.grad is not None
            torch.testing.assert_close(
                local_parameter.grad,
                full_parameter.grad,
                rtol=0,
                atol=0,
            )

    def test_position_losses_match_each_exact_slice_and_decompose(self) -> None:
        generator = torch.Generator().manual_seed(73)
        logits = torch.randn((2, 8, 5), generator=generator)
        logits[:, 2:4] *= 2
        logits[:, 4:6] *= 3
        logits[:, 6:8] *= 4
        targets = torch.randint(0, 5, (2, 8), generator=generator)
        bands = ((0, 2), (2, 4), (4, 6), (6, 8))
        actual = STAGE_18.position_cross_entropy(logits, targets, bands)

        token_losses = F.cross_entropy(
            logits.reshape(-1, 5),
            targets.reshape(-1),
            reduction="none",
        ).reshape(2, 8)
        expected = (
            token_losses.mean(),
            token_losses[:, :4].mean(),
            token_losses[:, 4:].mean(),
            *(token_losses[:, start:end].mean() for start, end in bands),
        )
        for measured, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(measured, wanted)
        torch.testing.assert_close(actual[0], sum(actual[3:]) / 4)
        torch.testing.assert_close(actual[1], (actual[3] + actual[4]) / 2)
        torch.testing.assert_close(actual[2], (actual[5] + actual[6]) / 2)

        invalid_bands = (
            (),
            ((0, 2), (3, 8)),
            ((0, 4), (3, 8)),
            ((1, 8),),
            ((0, 9),),
            ((0, 7),),
        )
        for invalid in invalid_bands:
            with self.subTest(bands=invalid):
                with self.assertRaises(ValueError):
                    STAGE_18.position_cross_entropy(logits, targets, invalid)
        with self.assertRaises(ValueError):
            STAGE_18.position_cross_entropy(logits[:, :-1], targets[:, :-1], ((0, 7),))
        with self.assertRaises(ValueError):
            STAGE_18.position_cross_entropy(logits, targets[:, :-1], bands)


class Stage18TinyWorkflowTests(unittest.TestCase):
    def test_training_checkpoints_and_precise_validation_are_paired(self) -> None:
        data = make_character_data()
        device = torch.device("cpu")
        schedule = STAGE_18.LearningRateSchedule(
            initial_learning_rate=2e-2,
            first_decay_step=2,
            middle_learning_rate=7e-3,
            second_decay_step=4,
            final_learning_rate=2e-3,
        )
        config = TrainingConfig(
            batch_size=2,
            block_size=8,
            n_embd=8,
            learning_rate=schedule.initial_learning_rate,
            max_iters=6,
            eval_interval=6,
            eval_iters=1,
            seed=101,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            local, full = make_specs(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                report = STAGE_18.run_experiment(
                    data,
                    config,
                    local,
                    full,
                    2,
                    1,
                    device,
                    0,
                    schedule,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    benchmark=None,
                )

            self.assertEqual(report.targets_per_update, 16)
            self.assertEqual(
                report.bands,
                ((0, 2), (2, 4), (4, 6), (6, 8)),
            )
            self.assertEqual(report.local.parameter_count, report.full.parameter_count)
            self.assertEqual(report.local.parameter_count, 1_013)
            self.assertEqual(report.local.attention_shape, (2, 2, 8, 8))
            self.assertEqual(report.full.attention_shape, (2, 2, 8, 8))
            self.assertEqual(report.initial_first_half_max_difference, 0.0)
            self.assertEqual(
                report.local.training.initial.val.first_half,
                report.full.training.initial.val.first_half,
            )
            self.assertEqual(
                report.local.training.initial.val.bands[:2],
                report.full.training.initial.val.bands[:2],
            )
            self.assertEqual(
                report.local.final_batch_generator_sha256,
                report.full.final_batch_generator_sha256,
            )
            self.assertEqual(
                tuple(record.step for record in report.local.training.history),
                (0, 2, 4, 6),
            )
            self.assertEqual(
                tuple(record.step for record in report.full.training.history),
                (0, 2, 4, 6),
            )

            for spec, initial_hash in (
                (local, report.initialization.local_state_sha256),
                (full, report.initialization.full_state_sha256),
            ):
                checkpoint = STAGE_18.validate_stage_18_checkpoint(
                    spec.checkpoint_path,
                    data,
                    config,
                    spec,
                    local,
                    full,
                    2,
                    1,
                    schedule,
                    report.bands,
                    initialization_seed=101,
                    branch_initial_state_sha256=initial_hash,
                    parameter_state_sha256=(
                        report.initialization.parameter_state_sha256
                    ),
                    local_mask_sha256=report.initialization.local_mask_sha256,
                    full_mask_sha256=report.initialization.full_mask_sha256,
                    training_batch_seed=202,
                    training_rng_seed=303,
                )
                experiment = checkpoint["experiment"]
                architecture = checkpoint["architecture"]
                self.assertEqual(
                    architecture["attention_mask"],
                    STAGE_18.attention_mask_kind(spec),
                )
                self.assertEqual(
                    architecture["attention_mask_scope"],
                    "per_layer",
                )
                self.assertEqual(experiment["stage"], 18)
                self.assertEqual(
                    experiment["comparison_variable"],
                    "maximum_attention_span",
                )
                self.assertTrue(experiment["identical_parameter_count"])
                self.assertTrue(experiment["identical_learned_initialization"])
                self.assertTrue(experiment["identical_training_batches"])
                self.assertTrue(experiment["identical_position_embeddings"])
                self.assertTrue(experiment["identical_attention_shapes"])
                self.assertEqual(
                    experiment["branch_direct_attention_span"],
                    spec.max_context or config.block_size,
                )
                expected_uncapped = (
                    STAGE_18.theoretical_receptive_field_width(
                        1,
                        spec.max_context,
                    )
                    if spec.max_context is not None
                    else None
                )
                self.assertEqual(
                    experiment["uncapped_local_receptive_field_width"],
                    expected_uncapped,
                )
                self.assertEqual(
                    experiment["effective_receptive_field_width"],
                    STAGE_18.effective_receptive_field_width(
                        config.block_size,
                        1,
                        spec,
                    ),
                )

            precise = STAGE_18.run_precise_validation(
                data,
                config,
                local,
                full,
                2,
                1,
                device,
                schedule,
                report.initialization,
                report.bands,
                initialization_seed=101,
                training_batch_seed=202,
                training_rng_seed=303,
                eval_iters=3,
                seed=404,
            )
            self.assertEqual(len(precise.results), 2)
            self.assertEqual(len(precise.results[0].batch_losses), 3)
            self.assertEqual(len(precise.results[1].batch_losses), 3)
            expected_differences = tuple(
                STAGE_18._subtract_position_losses(candidate, baseline)
                for candidate, baseline in zip(
                    precise.results[1].batch_losses,
                    precise.results[0].batch_losses,
                    strict=True,
                )
            )
            expected_statistics = STAGE_18._position_statistics(
                expected_differences
            )
            self.assertEqual(
                precise.delta.candidate,
                full.name,
            )
            self.assertEqual(
                precise.delta.baseline,
                local.name,
            )
            self.assertEqual(
                precise.delta.mean_delta,
                expected_statistics.mean,
            )
            self.assertEqual(
                precise.delta.standard_error,
                expected_statistics.standard_error,
            )
            self.assertEqual(
                precise.delta.critical_value,
                STAGE_18._paired_t_95_critical_value(3),
            )

            # Same-shaped persistent masks make a swapped/corrupt state load
            # possible, so semantic mask validation must reject it explicitly.
            corrupt = torch.load(
                local.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            for name in tuple(corrupt["model_state_dict"]):
                if name.endswith(".tril"):
                    corrupt["model_state_dict"][name] = (
                        STAGE_18.build_attention_mask(8, None).float()
                    )
            torch.save(corrupt, local.checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "incorrect mask"):
                STAGE_18.validate_stage_18_checkpoint(
                    local.checkpoint_path,
                    data,
                    config,
                    local,
                    local,
                    full,
                    2,
                    1,
                    schedule,
                    report.bands,
                    initialization_seed=101,
                    branch_initial_state_sha256=(
                        report.initialization.local_state_sha256
                    ),
                    parameter_state_sha256=(
                        report.initialization.parameter_state_sha256
                    ),
                    local_mask_sha256=report.initialization.local_mask_sha256,
                    full_mask_sha256=report.initialization.full_mask_sha256,
                    training_batch_seed=202,
                    training_rng_seed=303,
                )

    def test_training_batch_traces_match_exactly(self) -> None:
        data = make_character_data()
        device = torch.device("cpu")
        schedule = STAGE_18.LearningRateSchedule(
            initial_learning_rate=2e-2,
            first_decay_step=1,
            middle_learning_rate=7e-3,
            second_decay_step=2,
            final_learning_rate=2e-3,
        )
        config = TrainingConfig(
            batch_size=2,
            block_size=8,
            n_embd=8,
            learning_rate=schedule.initial_learning_rate,
            max_iters=3,
            eval_interval=3,
            eval_iters=1,
            seed=11,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            local, full = make_specs(Path(directory_name))
            initialization = STAGE_18.create_paired_initial_states(
                data, config, 2, 1, local, full
            )
            traces: list[list[tuple[int, float, torch.Tensor, torch.Tensor]]] = []
            final_states: list[torch.Tensor] = []
            for spec, state in (
                (local, initialization.local),
                (full, initialization.full),
            ):
                model = STAGE_18._build_model(data, config, 2, 1, spec)
                model.load_state_dict(state, strict=True)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=schedule.initial_learning_rate
                )
                generator = torch.Generator().manual_seed(29)
                trace: list[tuple[int, float, torch.Tensor, torch.Tensor]] = []

                def capture(
                    step: int,
                    learning_rate: float,
                    inputs: torch.Tensor,
                    targets: torch.Tensor,
                ) -> None:
                    trace.append(
                        (
                            step,
                            learning_rate,
                            inputs.detach().clone(),
                            targets.detach().clone(),
                        )
                    )

                STAGE_18.seed_everything(31)
                STAGE_18.train_with_schedule(
                    model,
                    optimizer,
                    data,
                    config,
                    device,
                    generator,
                    schedule,
                    STAGE_18.make_position_bands(8),
                    on_evaluation=lambda _record: None,
                    on_update=capture,
                )
                traces.append(trace)
                final_states.append(generator.get_state().clone())

        self.assertEqual(len(traces[0]), 3)
        self.assertEqual(
            [row[1] for row in traces[0]],
            [2e-2, 7e-3, 2e-3],
        )
        for local_row, full_row in zip(traces[0], traces[1], strict=True):
            self.assertEqual(local_row[:2], full_row[:2])
            torch.testing.assert_close(local_row[2], full_row[2], rtol=0, atol=0)
            torch.testing.assert_close(local_row[3], full_row[3], rtol=0, atol=0)
        torch.testing.assert_close(final_states[0], final_states[1], rtol=0, atol=0)
        torch.testing.assert_close(
            final_states[0],
            STAGE_18.expected_training_generator_state(
                data,
                config,
                training_batch_seed=29,
                completed_steps=3,
            ),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
