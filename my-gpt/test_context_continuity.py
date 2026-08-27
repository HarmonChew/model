from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("28_context_continuity.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_28_context_continuity",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_28 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_28
SPEC.loader.exec_module(STAGE_28)


def make_character_data(
    *,
    vocabulary_size: int = 5,
    train_tokens: int = 500,
    val_tokens: int = 200,
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


def make_tiny_protocol(
    directory: Path,
) -> tuple[
    TrainingConfig,
    object,
    object,
    object,
    Path,
]:
    schedule = STAGE_28.LearningRateSchedule(
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
        eval_interval=1,
        eval_iters=1,
        seed=101,
    )
    bounded = STAGE_28.AttentionSpec(
        "bounded-W3-at-T8 control",
        3,
        directory / "stage28_bounded_best.pt",
    )
    full = STAGE_28.AttentionSpec(
        "full-T8 reference",
        None,
        directory / "stage26_full.pt",
    )
    anchor = directory / "stage28_bounded_anchor.pt"
    return config, schedule, bounded, full, anchor


def create_tiny_stage_26_reference(
    data: CharacterData,
    config: TrainingConfig,
    schedule: object,
    full_path: Path,
) -> object:
    control_config, treatment_config = STAGE_28._stage_26_configs(config)
    control = STAGE_28._STAGE_26.ContextSpec(
        "T=4 control",
        control_config.batch_size,
        control_config.block_size,
        full_path.with_name("stage26_control.pt"),
    )
    treatment = STAGE_28._STAGE_26.ContextSpec(
        "T=8 treatment",
        treatment_config.batch_size,
        treatment_config.block_size,
        full_path,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return STAGE_28._STAGE_26.run_experiment(
            data,
            control_config,
            treatment_config,
            control,
            treatment,
            2,
            1,
            torch.device("cpu"),
            0,
            schedule,
            training_batch_seed=202,
            training_rng_seed=303,
            benchmark=None,
        )


def boolean_reachability(mask: torch.Tensor, n_layer: int) -> torch.Tensor:
    reachability = mask.clone()
    for _ in range(n_layer - 1):
        reachability = (reachability.to(torch.int32) @ mask.to(torch.int32)) > 0
    return reachability


class Stage28MaskAndReachabilityTests(unittest.TestCase):
    def test_mask_is_causal_continuous_and_has_exact_edge_counts(self) -> None:
        bounded = STAGE_28.build_attention_mask(8, 3)
        full = STAGE_28.build_attention_mask(8, None)
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(bounded, expected, rtol=0, atol=0)
        self.assertEqual(int(bounded.sum()), 21)
        self.assertEqual(int(full.sum()), 36)
        self.assertEqual(int((full & ~bounded).sum()), 15)
        self.assertTrue(torch.equal(bounded[:3], full[:3]))

        # Every query after position zero retains its immediately preceding
        # token. In particular, there is no Stage-27-style reset at T/2.
        self.assertTrue(bounded.diagonal().all().item())
        self.assertTrue(bounded.diagonal(offset=-1).all().item())
        self.assertTrue(bounded[4, 3].item())
        self.assertTrue(bounded[4, 2].item())
        self.assertFalse(bounded[4, 1].item())

        canonical_bounded = STAGE_28.build_attention_mask(128, 12)
        canonical_full = STAGE_28.build_attention_mask(128, None)
        self.assertEqual(int(canonical_bounded.sum()), 1_470)
        self.assertEqual(int(canonical_full.sum()), 8_256)
        self.assertEqual(
            int((canonical_full & ~canonical_bounded).sum()),
            6_786,
        )
        self.assertTrue(torch.equal(canonical_bounded[:12], canonical_full[:12]))
        self.assertTrue(canonical_bounded.diagonal(offset=-1).all().item())
        self.assertEqual(
            STAGE_28.tensor_sha256(canonical_bounded),
            "7ec1b36fc194b2f73d2d1ff723fbb8f523cab5160f7e03ebe100a7255ffdd1d4",
        )
        self.assertEqual(
            STAGE_28.tensor_sha256(canonical_bounded.float()),
            "0509f55a67587266e0d8670daec81e1b22b923160f6a89590e46934c04ba87a8",
        )

    def test_six_layer_reachability_is_exactly_67_without_a_reset(self) -> None:
        mask = STAGE_28.build_attention_mask(128, 12)
        reachability = boolean_reachability(mask, 6)
        self.assertEqual(
            torch.nonzero(reachability[127], as_tuple=False).flatten().tolist(),
            list(range(61, 128)),
        )
        self.assertTrue(reachability[66, 0].item())
        self.assertFalse(reachability[67, 0].item())
        self.assertTrue(reachability[67, 1].item())

        audit = STAGE_28.audit_bounded_reachability(mask, n_layer=6)
        self.assertEqual(audit.theoretical_receptive_field_width, 67)
        self.assertEqual(audit.effective_receptive_field_width, 67)
        self.assertEqual(audit.final_earliest_source_by_query[-1], 61)
        self.assertEqual(audit.first_query_with_unavailable_source, 67)
        self.assertTrue(audit.no_context_reset)
        self.assertTrue(audit.causal)
        self.assertTrue(audit.contiguous)

        with tempfile.TemporaryDirectory() as directory_name:
            bounded = STAGE_28.AttentionSpec(
                "bounded",
                12,
                Path(directory_name) / "bounded.pt",
            )
            full = STAGE_28.AttentionSpec(
                "full",
                None,
                Path(directory_name) / "full.pt",
            )
            self.assertEqual(
                STAGE_28.effective_receptive_field_width(128, 6, bounded),
                67,
            )
            self.assertEqual(
                STAGE_28.effective_receptive_field_width(128, 6, full),
                128,
            )

    def test_protocol_pins_the_canonical_geometry_bands_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = TrainingConfig(
                batch_size=16,
                block_size=128,
                n_embd=128,
                learning_rate=1e-3,
                max_iters=18_000,
                eval_interval=500,
                eval_iters=100,
                seed=1_337,
            )
            bounded = STAGE_28.AttentionSpec(
                "bounded-W12-at-T128",
                12,
                directory / "bounded.pt",
            )
            full = STAGE_28.AttentionSpec(
                "full-T128",
                None,
                directory / "full.pt",
            )
            anchor = directory / "anchor.pt"
            targets, bands = STAGE_28.validate_attention_protocol(
                config,
                bounded,
                full,
                4,
                6,
                anchor_checkpoint_path=anchor,
            )

            self.assertEqual(targets, 2_048)
            self.assertEqual(
                bands,
                ((0, 12), (12, 32), (32, 64), (64, 67), (67, 128)),
            )
            self.assertEqual(
                STAGE_28.make_position_bands(8, 3, 1),
                ((0, 2), (2, 3), (3, 8)),
            )

            aliased = STAGE_28.AttentionSpec(
                "full-T128",
                None,
                bounded.checkpoint_path,
            )
            with self.assertRaises(ValueError):
                STAGE_28.validate_attention_protocol(
                    config,
                    bounded,
                    aliased,
                    4,
                    6,
                )
            with self.assertRaises(ValueError):
                STAGE_28.validate_attention_protocol(
                    config,
                    bounded,
                    full,
                    4,
                    6,
                    anchor_checkpoint_path=bounded.checkpoint_path,
                )


class Stage28InitializationTests(unittest.TestCase):
    def test_initial_logit_equivalence_ends_after_query_row_11(self) -> None:
        data = make_character_data()
        config = TrainingConfig(
            batch_size=2,
            block_size=16,
            n_embd=8,
            learning_rate=1e-3,
            max_iters=1,
            eval_interval=1,
            eval_iters=1,
            seed=103,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            bounded = STAGE_28.AttentionSpec(
                "bounded-W12-at-T16",
                12,
                directory / "bounded.pt",
            )
            full = STAGE_28.AttentionSpec(
                "full-T16",
                None,
                directory / "full.pt",
            )
            initialization = STAGE_28.create_paired_initial_states(
                data,
                config,
                bounded,
                full,
                2,
                1,
            )
            bounded_model = STAGE_28.build_model(
                data,
                config,
                bounded,
                2,
                1,
            )
            full_model = STAGE_28.build_model(data, config, full, 2, 1)
            bounded_model.load_state_dict(initialization.local, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(
            STAGE_28.verify_initial_shared_prefix_equivalence(
                bounded_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=109,
                prefix_length=12,
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "shared-prefix logits differ"):
            STAGE_28.verify_initial_shared_prefix_equivalence(
                bounded_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=109,
                prefix_length=13,
            )

    def test_initial_states_match_except_masks_and_prefix_logits_are_exact(
        self,
    ) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            config, _, bounded, full, _ = make_tiny_protocol(Path(directory_name))
            initialization = STAGE_28.create_paired_initial_states(
                data,
                config,
                bounded,
                full,
                2,
                2,
            )
            bounded_model = STAGE_28.build_model(
                data,
                config,
                bounded,
                2,
                2,
            )
            full_model = STAGE_28.build_model(data, config, full, 2, 2)
            bounded_model.load_state_dict(initialization.local, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(initialization.parameter_count, 1_861)
        STAGE_28.verify_model_mask(bounded_model, bounded)
        STAGE_28.verify_model_mask(full_model, full)
        self.assertEqual(
            sum(name.endswith(".tril") for name in bounded_model.state_dict()),
            4,
        )
        self.assertEqual(len(initialization.differing_state_names), 4)
        self.assertTrue(
            all(name.endswith(".tril") for name in initialization.differing_state_names)
        )
        parameter_names = dict(bounded_model.named_parameters())
        for name, bounded_tensor in initialization.local.items():
            full_tensor = initialization.full[name]
            self.assertNotEqual(
                bounded_tensor.data_ptr(),
                full_tensor.data_ptr(),
            )
            if name in parameter_names:
                torch.testing.assert_close(
                    bounded_tensor,
                    full_tensor,
                    rtol=0,
                    atol=0,
                )

        self.assertEqual(
            STAGE_28.verify_initial_shared_prefix_equivalence(
                bounded_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=107,
                prefix_length=3,
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "shared-prefix logits differ"):
            STAGE_28.verify_initial_shared_prefix_equivalence(
                bounded_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=107,
                prefix_length=4,
            )

    def test_canonical_initialization_has_24_mask_only_differences(self) -> None:
        data = CharacterData.from_file(Path(__file__).parent / "data" / "input.txt")
        config = TrainingConfig(
            batch_size=16,
            block_size=128,
            n_embd=128,
            learning_rate=1e-3,
            max_iters=18_000,
            eval_interval=500,
            eval_iters=100,
            seed=1_337,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            bounded = STAGE_28.AttentionSpec(
                "bounded",
                12,
                directory / "bounded.pt",
            )
            full = STAGE_28.AttentionSpec(
                "full",
                None,
                directory / "full.pt",
            )
            initialization = STAGE_28.create_paired_initial_states(
                data,
                config,
                bounded,
                full,
                4,
                6,
            )

        self.assertEqual(initialization.parameter_count, 1_220_673)
        self.assertEqual(len(initialization.differing_state_names), 24)
        self.assertTrue(
            all(name.endswith(".tril") for name in initialization.differing_state_names)
        )
        self.assertEqual(
            initialization.local_mask_sha256,
            "7ec1b36fc194b2f73d2d1ff723fbb8f523cab5160f7e03ebe100a7255ffdd1d4",
        )
        self.assertEqual(
            initialization.local_float_mask_sha256,
            "0509f55a67587266e0d8670daec81e1b22b923160f6a89590e46934c04ba87a8",
        )
        self.assertEqual(
            initialization.parameter_state_sha256,
            STAGE_28.DEFAULT_PARAMETER_INITIALIZATION_SHA256,
        )
        self.assertEqual(
            initialization.local_state_sha256,
            STAGE_28.DEFAULT_LOCAL_INITIAL_STATE_SHA256,
        )
        self.assertEqual(
            initialization.full_state_sha256,
            STAGE_28.DEFAULT_FULL_INITIAL_STATE_SHA256,
        )


class Stage28TinyWorkflowTests(unittest.TestCase):
    def test_matched_stream_checkpoint_metadata_and_fresh_paired_evaluator(
        self,
    ) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, schedule, bounded, full, anchor = make_tiny_protocol(directory)
            stage_26 = create_tiny_stage_26_reference(
                data,
                config,
                schedule,
                full.checkpoint_path,
            )
            source_bytes = full.checkpoint_path.read_bytes()
            source_hash = STAGE_28.checkpoint_sha256(full.checkpoint_path)

            updates: list[tuple[int, torch.Tensor, torch.Tensor]] = []

            def capture_update(
                step: int,
                _learning_rate: float,
                inputs: torch.Tensor,
                targets: torch.Tensor,
            ) -> None:
                updates.append(
                    (
                        step,
                        inputs.detach().cpu().clone(),
                        targets.detach().cpu().clone(),
                    )
                )

            with contextlib.redirect_stdout(io.StringIO()):
                experiment = STAGE_28.run_experiment(
                    data,
                    config,
                    bounded,
                    full,
                    2,
                    1,
                    torch.device("cpu"),
                    0,
                    schedule,
                    anchor,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    full_reference_expected_sha256=source_hash,
                    full_reference_expected_step=stage_26.treatment.best_step,
                    benchmark=None,
                    on_update=capture_update,
                )

            self.assertEqual(full.checkpoint_path.read_bytes(), source_bytes)
            self.assertEqual(len(updates), config.max_iters)
            expected_generator = torch.Generator().manual_seed(202)
            for expected_step, (step, actual_inputs, actual_targets) in enumerate(
                updates
            ):
                expected_inputs, expected_targets = data.get_batch(
                    "train",
                    batch_size=config.batch_size,
                    block_size=config.block_size,
                    device=torch.device("cpu"),
                    generator=expected_generator,
                )
                self.assertEqual(step, expected_step)
                torch.testing.assert_close(
                    actual_inputs,
                    expected_inputs,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual_targets,
                    expected_targets,
                    rtol=0,
                    atol=0,
                )

            self.assertEqual(experiment.full.checkpoint_sha256, source_hash)
            self.assertEqual(
                experiment.local_checkpoint_sha256,
                STAGE_28.checkpoint_sha256(bounded.checkpoint_path),
            )
            self.assertEqual(
                experiment.local_anchor_checkpoint_sha256,
                STAGE_28.checkpoint_sha256(anchor),
            )
            self.assertEqual(
                experiment.full.checkpoint_step,
                stage_26.treatment.best_step,
            )
            self.assertEqual(experiment.initial_shared_prefix_length, 3)
            self.assertEqual(experiment.initial_shared_prefix_max_difference, 0.0)
            self.assertEqual(
                experiment.reachability.effective_receptive_field_width,
                3,
            )

            best = STAGE_28.validate_stage_28_checkpoint(
                bounded.checkpoint_path,
                data,
                config,
                bounded,
                full,
                2,
                1,
                schedule,
                experiment.bands,
                experiment.initialization,
                experiment.full,
                training_batch_seed=202,
                training_rng_seed=303,
                checkpoint_kind="best",
            )
            anchor_payload = STAGE_28.validate_stage_28_checkpoint(
                anchor,
                data,
                config,
                bounded,
                full,
                2,
                1,
                schedule,
                experiment.bands,
                experiment.initialization,
                experiment.full,
                training_batch_seed=202,
                training_rng_seed=303,
                checkpoint_kind="anchor",
            )
            metadata = best["experiment"]
            self.assertEqual(metadata["stage"], 28)
            self.assertEqual(
                metadata["comparison_variable"],
                STAGE_28.COMPARISON_VARIABLE,
            )
            self.assertEqual(metadata["max_context_per_layer"], 3)
            self.assertEqual(metadata["effective_receptive_field_width"], 3)
            self.assertFalse(metadata["context_reset"])
            self.assertTrue(metadata["continuous_local_history"])
            self.assertTrue(metadata["multilayer_relay_accounted_for"])
            self.assertTrue(metadata["full_reference_reused_read_only"])
            evaluation_protocol = best["evaluation_protocol"]
            self.assertFalse(evaluation_protocol["same_weight_mask_ablations"])
            self.assertFalse(evaluation_protocol["crossed_mask_variants"])
            self.assertTrue(evaluation_protocol["fresh_validation_natural_masks_only"])
            self.assertEqual(
                anchor_payload["step"],
                stage_26.treatment.best_step,
            )

            persisted_bytes = (
                bounded.checkpoint_path.read_bytes(),
                anchor.read_bytes(),
                full.checkpoint_path.read_bytes(),
            )
            precise = STAGE_28.run_paired_validation(
                data,
                config,
                bounded,
                full,
                2,
                1,
                torch.device("cpu"),
                schedule,
                experiment,
                anchor,
                eval_iters=2,
                seed=404,
            )
            self.assertEqual(precise.eval_iters, 2)
            self.assertEqual(precise.seed, 404)
            for endpoint in (precise.anchor, precise.selected_best):
                self.assertEqual(
                    tuple(field.name for field in fields(endpoint)),
                    ("name", "full_natural", "local_natural", "natural_delta"),
                )
                self.assertEqual(len(endpoint.full_natural.batch_losses), 2)
                self.assertEqual(len(endpoint.local_natural.batch_losses), 2)
                self.assertEqual(
                    endpoint.natural_delta.candidate,
                    endpoint.full_natural.name,
                )
                self.assertEqual(
                    endpoint.natural_delta.baseline,
                    endpoint.local_natural.name,
                )
            self.assertEqual(
                (
                    bounded.checkpoint_path.read_bytes(),
                    anchor.read_bytes(),
                    full.checkpoint_path.read_bytes(),
                ),
                persisted_bytes,
            )

            corrupt_path = directory / "corrupt_bounded.pt"
            corrupt = dict(best)
            corrupt_state = dict(corrupt["model_state_dict"])
            mask_name = next(name for name in corrupt_state if name.endswith(".tril"))
            corrupt_mask = corrupt_state[mask_name].clone()
            corrupt_mask[4, 0] = 1
            corrupt_state[mask_name] = corrupt_mask
            corrupt["model_state_dict"] = corrupt_state
            torch.save(corrupt, corrupt_path)
            with self.assertRaisesRegex(RuntimeError, "incorrect attention-mask"):
                STAGE_28.validate_stage_28_checkpoint(
                    corrupt_path,
                    data,
                    config,
                    bounded,
                    full,
                    2,
                    1,
                    schedule,
                    experiment.bands,
                    experiment.initialization,
                    experiment.full,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    checkpoint_kind="best",
                )


if __name__ == "__main__":
    unittest.main()
