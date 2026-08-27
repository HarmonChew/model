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


MODULE_PATH = Path(__file__).with_name("29_global_receptive_field.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_29_global_receptive_field",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_29 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_29
SPEC.loader.exec_module(STAGE_29)


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
    schedule = STAGE_29.LearningRateSchedule(
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
    local = STAGE_29.AttentionSpec(
        "global-RF-W5-at-T8 control",
        5,
        directory / "stage29_local_best.pt",
    )
    full = STAGE_29.AttentionSpec(
        "full-T8 reference",
        None,
        directory / "stage26_full.pt",
    )
    anchor = directory / "stage29_local_anchor.pt"
    return config, schedule, local, full, anchor


def create_tiny_stage_26_reference(
    data: CharacterData,
    config: TrainingConfig,
    schedule: object,
    full_path: Path,
) -> object:
    control_config, treatment_config = STAGE_29._stage_26_configs(config)
    control = STAGE_29._STAGE_26.ContextSpec(
        "T=4 control",
        control_config.batch_size,
        control_config.block_size,
        full_path.with_name("stage26_control.pt"),
    )
    treatment = STAGE_29._STAGE_26.ContextSpec(
        "T=8 treatment",
        treatment_config.batch_size,
        treatment_config.block_size,
        full_path,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return STAGE_29._STAGE_26.run_experiment(
            data,
            control_config,
            treatment_config,
            control,
            treatment,
            2,
            2,
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


class Stage29MaskAndReachabilityTests(unittest.TestCase):
    def test_mask_is_causal_continuous_and_has_exact_edge_counts(self) -> None:
        local = STAGE_29.build_attention_mask(8, 5)
        full = STAGE_29.build_attention_mask(8, None)
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(local, expected, rtol=0, atol=0)
        self.assertEqual(int(local.sum()), 30)
        self.assertEqual(int(full.sum()), 36)
        self.assertEqual(int((full & ~local).sum()), 6)
        self.assertTrue(torch.equal(local[:5], full[:5]))

        # Every query after position zero retains its immediately preceding
        # token. In particular, there is no Stage-27-style reset at T/2.
        self.assertTrue(local.diagonal().all().item())
        self.assertTrue(local.diagonal(offset=-1).all().item())
        self.assertTrue(local[5, 1].item())
        self.assertFalse(local[5, 0].item())

        canonical_local = STAGE_29.build_attention_mask(128, 23)
        canonical_full = STAGE_29.build_attention_mask(128, None)
        self.assertEqual(int(canonical_local.sum()), 2_691)
        self.assertEqual(int(canonical_full.sum()), 8_256)
        self.assertEqual(
            int((canonical_full & ~canonical_local).sum()),
            5_565,
        )
        self.assertTrue(torch.equal(canonical_local[:23], canonical_full[:23]))
        self.assertFalse(torch.equal(canonical_local[:24], canonical_full[:24]))
        self.assertTrue(canonical_local.diagonal(offset=-1).all().item())
        self.assertEqual(
            STAGE_29.tensor_sha256(canonical_local),
            "bab7c894395eb65e7ac2d5b7d2eaa0f8687e991914182ec2ad36858d0f16544b",
        )
        self.assertEqual(
            STAGE_29.tensor_sha256(canonical_local.float()),
            "d0ec90278c3d187be126d359bdb43d62e763eb478dc5c3d0fa07ca30f8603089",
        )

    def test_six_layer_reachability_is_global_without_a_reset(self) -> None:
        mask = STAGE_29.build_attention_mask(128, 23)
        reachability = boolean_reachability(mask, 6)
        torch.testing.assert_close(
            reachability,
            STAGE_29.build_attention_mask(128, None),
            rtol=0,
            atol=0,
        )

        audit = STAGE_29.audit_bounded_reachability(mask, n_layer=6)
        self.assertEqual(audit.theoretical_receptive_field_width, 133)
        self.assertEqual(audit.effective_receptive_field_width, 128)
        self.assertEqual(audit.widths_by_layer, (23, 45, 67, 89, 111, 133))
        self.assertEqual(audit.final_earliest_source_by_query, (0,) * 128)
        self.assertEqual(audit.final_width_by_query, tuple(range(1, 129)))
        self.assertIsNone(audit.first_query_with_unavailable_source)
        self.assertTrue(audit.all_causal_sources_reachable)
        self.assertTrue(audit.no_context_reset)
        self.assertTrue(audit.causal)
        self.assertTrue(audit.contiguous)
        STAGE_29.verify_global_reachability(audit, 128)

        w22_audit = STAGE_29.audit_bounded_reachability(
            STAGE_29.build_attention_mask(128, 22),
            n_layer=6,
        )
        self.assertEqual(w22_audit.theoretical_receptive_field_width, 127)
        self.assertEqual(w22_audit.first_query_with_unavailable_source, 127)
        with self.assertRaises(RuntimeError):
            STAGE_29.verify_global_reachability(w22_audit, 128)

        with tempfile.TemporaryDirectory() as directory_name:
            local = STAGE_29.AttentionSpec(
                "local",
                23,
                Path(directory_name) / "local.pt",
            )
            full = STAGE_29.AttentionSpec(
                "full",
                None,
                Path(directory_name) / "full.pt",
            )
            self.assertEqual(
                STAGE_29.effective_receptive_field_width(128, 6, local),
                128,
            )
            self.assertEqual(
                STAGE_29.effective_receptive_field_width(128, 6, full),
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
            local = STAGE_29.AttentionSpec(
                "global-RF-W23-at-T128",
                23,
                directory / "local.pt",
            )
            full = STAGE_29.AttentionSpec(
                "full-T128",
                None,
                directory / "full.pt",
            )
            anchor = directory / "anchor.pt"
            targets, bands = STAGE_29.validate_attention_protocol(
                config,
                local,
                full,
                4,
                6,
                anchor_checkpoint_path=anchor,
            )

            self.assertEqual(targets, 2_048)
            self.assertEqual(
                bands,
                (
                    (0, 23),
                    (23, 45),
                    (45, 67),
                    (67, 89),
                    (89, 111),
                    (111, 128),
                ),
            )
            self.assertEqual(
                STAGE_29.hop_depths_for_bands(bands, 23),
                (1, 2, 3, 4, 5, 6),
            )
            self.assertEqual(
                STAGE_29.make_position_bands(8, 5, 2),
                ((0, 5), (5, 8)),
            )
            self.assertEqual(
                STAGE_29.minimum_local_context_for_global_receptive_field(128, 6),
                23,
            )
            self.assertEqual(STAGE_29.DEFAULT_PRECISE_EVAL_SEED, 1_346)

            aliased = STAGE_29.AttentionSpec(
                "full-T128",
                None,
                local.checkpoint_path,
            )
            with self.assertRaises(ValueError):
                STAGE_29.validate_attention_protocol(
                    config,
                    local,
                    aliased,
                    4,
                    6,
                )
            with self.assertRaises(ValueError):
                STAGE_29.validate_attention_protocol(
                    config,
                    local,
                    full,
                    4,
                    6,
                    anchor_checkpoint_path=local.checkpoint_path,
                )

            for invalid_window in (22, 24, 128):
                invalid = STAGE_29.AttentionSpec(
                    f"invalid-W{invalid_window}",
                    invalid_window,
                    directory / f"invalid-{invalid_window}.pt",
                )
                with self.assertRaises(ValueError):
                    STAGE_29.validate_attention_protocol(
                        config,
                        invalid,
                        full,
                        4,
                        6,
                    )


class Stage29RangeDiagnosticTests(unittest.TestCase):
    def test_multi_band_range_is_token_weighted_and_requires_exact_coverage(
        self,
    ) -> None:
        bands = ((0, 2), (2, 5), (5, 9))
        losses = STAGE_29.PositionLosses(
            overall=0.0,
            first_half=0.0,
            second_half=0.0,
            bands=(100.0, 10.0, 4.0),
        )
        self.assertAlmostEqual(
            STAGE_29.position_range_mean(losses, bands, 2, 9),
            (3 * 10.0 + 4 * 4.0) / 7,
        )
        with self.assertRaises(ValueError):
            STAGE_29.position_range_mean(losses, bands, 3, 9)

    def test_paired_range_delta_uses_per_panel_differences(self) -> None:
        bands = ((0, 2), (2, 5), (5, 9))

        def losses(values: tuple[float, ...]) -> object:
            return STAGE_29.PositionLosses(
                overall=0.0,
                first_half=0.0,
                second_half=0.0,
                bands=values,
            )

        def result(name: str, values: tuple[object, ...]) -> object:
            return STAGE_29.PreciseBranchResult(
                name=name,
                checkpoint_path=Path(f"{name}.pt"),
                checkpoint_sha256=name,
                checkpoint_step=1,
                fixed_panel_loss=0.0,
                statistics=None,
                batch_losses=values,
            )

        candidate = result(
            "candidate",
            (
                losses((0.0, 4.0, 1.0)),
                losses((0.0, 2.0, 5.0)),
            ),
        )
        baseline = result(
            "baseline",
            (
                losses((0.0, 1.0, 1.0)),
                losses((0.0, 2.0, 1.0)),
            ),
        )
        delta = STAGE_29.paired_position_range_delta(
            candidate,
            baseline,
            bands,
            2,
            9,
        )
        self.assertAlmostEqual(delta.mean_delta, 25 / 14)
        self.assertAlmostEqual(delta.standard_error, 0.5)
        self.assertAlmostEqual(delta.critical_value, 12.706204736)
        self.assertAlmostEqual(
            delta.confidence_low,
            25 / 14 - 12.706204736 * 0.5,
        )
        self.assertAlmostEqual(
            delta.confidence_high,
            25 / 14 + 12.706204736 * 0.5,
        )


class Stage29InitializationTests(unittest.TestCase):
    def test_initial_logit_equivalence_ends_at_the_local_window(self) -> None:
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
            local = STAGE_29.AttentionSpec(
                "global-RF-W9-at-T16",
                9,
                directory / "local.pt",
            )
            full = STAGE_29.AttentionSpec(
                "full-T16",
                None,
                directory / "full.pt",
            )
            initialization = STAGE_29.create_paired_initial_states(
                data,
                config,
                local,
                full,
                2,
                2,
            )
            local_model = STAGE_29.build_model(
                data,
                config,
                local,
                2,
                2,
            )
            full_model = STAGE_29.build_model(data, config, full, 2, 2)
            local_model.load_state_dict(initialization.local, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(
            STAGE_29.verify_initial_shared_prefix_equivalence(
                local_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=109,
                prefix_length=9,
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "shared-prefix logits differ"):
            STAGE_29.verify_initial_shared_prefix_equivalence(
                local_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=109,
                prefix_length=10,
            )

    def test_initial_states_match_except_masks_and_prefix_logits_are_exact(
        self,
    ) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            config, _, bounded, full, _ = make_tiny_protocol(Path(directory_name))
            initialization = STAGE_29.create_paired_initial_states(
                data,
                config,
                bounded,
                full,
                2,
                2,
            )
            bounded_model = STAGE_29.build_model(
                data,
                config,
                bounded,
                2,
                2,
            )
            full_model = STAGE_29.build_model(data, config, full, 2, 2)
            bounded_model.load_state_dict(initialization.local, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(initialization.parameter_count, 1_861)
        STAGE_29.verify_model_mask(bounded_model, bounded)
        STAGE_29.verify_model_mask(full_model, full)
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
            STAGE_29.verify_initial_shared_prefix_equivalence(
                bounded_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=107,
                prefix_length=5,
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "shared-prefix logits differ"):
            STAGE_29.verify_initial_shared_prefix_equivalence(
                bounded_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=107,
                prefix_length=6,
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
            bounded = STAGE_29.AttentionSpec(
                "global-RF local",
                23,
                directory / "local.pt",
            )
            full = STAGE_29.AttentionSpec(
                "full",
                None,
                directory / "full.pt",
            )
            initialization = STAGE_29.create_paired_initial_states(
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
            "bab7c894395eb65e7ac2d5b7d2eaa0f8687e991914182ec2ad36858d0f16544b",
        )
        self.assertEqual(
            initialization.local_float_mask_sha256,
            "d0ec90278c3d187be126d359bdb43d62e763eb478dc5c3d0fa07ca30f8603089",
        )
        self.assertEqual(
            initialization.parameter_state_sha256,
            STAGE_29.DEFAULT_PARAMETER_INITIALIZATION_SHA256,
        )
        self.assertEqual(
            initialization.local_state_sha256,
            STAGE_29.DEFAULT_LOCAL_INITIAL_STATE_SHA256,
        )
        self.assertEqual(
            initialization.full_state_sha256,
            STAGE_29.DEFAULT_FULL_INITIAL_STATE_SHA256,
        )


class Stage29TinyWorkflowTests(unittest.TestCase):
    def test_matched_stream_checkpoint_metadata_and_held_out_paired_evaluator(
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
            source_hash = STAGE_29.checkpoint_sha256(full.checkpoint_path)

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
                experiment = STAGE_29.run_experiment(
                    data,
                    config,
                    bounded,
                    full,
                    2,
                    2,
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
                STAGE_29.checkpoint_sha256(bounded.checkpoint_path),
            )
            self.assertEqual(
                experiment.local_anchor_checkpoint_sha256,
                STAGE_29.checkpoint_sha256(anchor),
            )
            self.assertEqual(
                experiment.full.checkpoint_step,
                stage_26.treatment.best_step,
            )
            self.assertEqual(experiment.initial_shared_prefix_length, 5)
            self.assertEqual(experiment.initial_shared_prefix_max_difference, 0.0)
            self.assertEqual(
                experiment.reachability.effective_receptive_field_width,
                8,
            )
            self.assertEqual(
                experiment.reachability.theoretical_receptive_field_width,
                9,
            )
            self.assertTrue(experiment.reachability.all_causal_sources_reachable)
            self.assertEqual(experiment.bands, ((0, 5), (5, 8)))

            best = STAGE_29.validate_stage_29_checkpoint(
                bounded.checkpoint_path,
                data,
                config,
                bounded,
                full,
                2,
                2,
                schedule,
                experiment.bands,
                experiment.initialization,
                experiment.full,
                training_batch_seed=202,
                training_rng_seed=303,
                checkpoint_kind="best",
            )
            anchor_payload = STAGE_29.validate_stage_29_checkpoint(
                anchor,
                data,
                config,
                bounded,
                full,
                2,
                2,
                schedule,
                experiment.bands,
                experiment.initialization,
                experiment.full,
                training_batch_seed=202,
                training_rng_seed=303,
                checkpoint_kind="anchor",
            )
            metadata = best["experiment"]
            self.assertEqual(metadata["stage"], 29)
            self.assertEqual(
                metadata["comparison_variable"],
                STAGE_29.COMPARISON_VARIABLE,
            )
            self.assertEqual(metadata["max_context_per_layer"], 5)
            self.assertEqual(metadata["minimum_global_receptive_field_window"], 5)
            self.assertEqual(metadata["effective_receptive_field_width"], 8)
            self.assertEqual(metadata["uncapped_receptive_field_width"], 9)
            self.assertIsNone(metadata["first_query_with_unavailable_source"])
            self.assertTrue(
                metadata["all_causal_source_origins_reachable_at_final_layer"]
            )
            self.assertTrue(metadata["global_end_to_end_receptive_field"])
            self.assertFalse(metadata["direct_global_attention"])
            self.assertEqual(metadata["first_query_with_shorter_full_path"], 5)
            self.assertEqual(metadata["hop_depths_by_position_band"], (1, 2))
            self.assertEqual(metadata["shorter_path_position_range"], (5, 8))
            self.assertFalse(
                metadata[
                    "intervention_changes_reachable_source_origins_at_final_layer"
                ]
            )
            self.assertFalse(metadata["context_reset"])
            self.assertTrue(metadata["continuous_local_history"])
            self.assertTrue(metadata["multilayer_relay_accounted_for"])
            self.assertTrue(metadata["full_reference_reused_read_only"])
            evaluation_protocol = best["evaluation_protocol"]
            self.assertFalse(evaluation_protocol["same_weight_mask_ablations"])
            self.assertFalse(evaluation_protocol["crossed_mask_variants"])
            self.assertTrue(
                evaluation_protocol["held_out_validation_natural_masks_only"]
            )
            self.assertEqual(
                evaluation_protocol["predeclared_mechanistic_diagnostic"],
                "positions_5_through_7_multi_hop_loss",
            )
            self.assertEqual(
                evaluation_protocol["mechanistic_diagnostic_position_range"],
                (5, 8),
            )
            self.assertNotIn("primary_extra_history_diagnostic", evaluation_protocol)
            self.assertNotIn("extra_history_position_band", metadata)
            self.assertEqual(
                anchor_payload["step"],
                stage_26.treatment.best_step,
            )

            missing_key_path = directory / "missing_global_reachability_key.pt"
            missing_key = dict(best)
            missing_experiment = dict(missing_key["experiment"])
            missing_experiment.pop("first_query_with_unavailable_source")
            missing_key["experiment"] = missing_experiment
            torch.save(missing_key, missing_key_path)
            with self.assertRaisesRegex(ValueError, "experiment keys differ"):
                STAGE_29.validate_stage_29_checkpoint(
                    missing_key_path,
                    data,
                    config,
                    bounded,
                    full,
                    2,
                    2,
                    schedule,
                    experiment.bands,
                    experiment.initialization,
                    experiment.full,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    checkpoint_kind="best",
                )

            legacy_key_path = directory / "legacy_extra_history_key.pt"
            legacy_key = dict(best)
            legacy_experiment = dict(legacy_key["experiment"])
            legacy_experiment["extra_history_position_band"] = (5, 8)
            legacy_key["experiment"] = legacy_experiment
            torch.save(legacy_key, legacy_key_path)
            with self.assertRaisesRegex(ValueError, "experiment keys differ"):
                STAGE_29.validate_stage_29_checkpoint(
                    legacy_key_path,
                    data,
                    config,
                    bounded,
                    full,
                    2,
                    2,
                    schedule,
                    experiment.bands,
                    experiment.initialization,
                    experiment.full,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    checkpoint_kind="best",
                )

            persisted_bytes = (
                bounded.checkpoint_path.read_bytes(),
                anchor.read_bytes(),
                full.checkpoint_path.read_bytes(),
            )
            precise = STAGE_29.run_paired_validation(
                data,
                config,
                bounded,
                full,
                2,
                2,
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
                    (
                        "name",
                        "full_natural",
                        "local_natural",
                        "natural_delta",
                        "multi_hop_delta",
                    ),
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
                    (endpoint.multi_hop_delta.start, endpoint.multi_hop_delta.end),
                    (5, 8),
                )
                self.assertEqual(
                    endpoint.multi_hop_delta.candidate,
                    endpoint.full_natural.name,
                )
                self.assertEqual(
                    endpoint.multi_hop_delta.baseline,
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
            corrupt_mask[5, 0] = 1
            corrupt_state[mask_name] = corrupt_mask
            corrupt["model_state_dict"] = corrupt_state
            torch.save(corrupt, corrupt_path)
            with self.assertRaisesRegex(RuntimeError, "incorrect attention-mask"):
                STAGE_29.validate_stage_29_checkpoint(
                    corrupt_path,
                    data,
                    config,
                    bounded,
                    full,
                    2,
                    2,
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
