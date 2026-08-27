from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("30_sparse_global_shortcut.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_30_sparse_global_shortcut",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_30 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_30
SPEC.loader.exec_module(STAGE_30)


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


def canonical_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=16,
        block_size=128,
        n_embd=128,
        learning_rate=1e-3,
        max_iters=18_000,
        eval_interval=500,
        eval_iters=100,
        seed=1_337,
    )


def canonical_specs(directory: Path) -> tuple[object, object, object, Path, Path]:
    hybrid = STAGE_30.AttentionSpec(
        "one middle global layer",
        (23, 23, None, 23, 23, 23),
        directory / "hybrid-best.pt",
    )
    window = STAGE_30.UniformAttentionSpec(
        "W=23 Stage 29 reference",
        23,
        directory / "window-best.pt",
    )
    full = STAGE_30.UniformAttentionSpec(
        "Full-128 Stage 26 reference",
        None,
        directory / "full.pt",
    )
    return (
        hybrid,
        window,
        full,
        directory / "hybrid-anchor.pt",
        directory / "window-anchor.pt",
    )


def make_tiny_protocol(
    directory: Path,
) -> tuple[TrainingConfig, object, object, object, object, Path, Path]:
    schedule = STAGE_30.LearningRateSchedule(
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
    hybrid = STAGE_30.AttentionSpec(
        "tiny layer-3 shortcut",
        (3, 3, None, 3, 3, 3),
        directory / "stage30-hybrid-best.pt",
    )
    window = STAGE_30.UniformAttentionSpec(
        "tiny W=3 reference",
        3,
        directory / "stage29-window-best.pt",
    )
    full = STAGE_30.UniformAttentionSpec(
        "tiny Full reference",
        None,
        directory / "stage26-full.pt",
    )
    return (
        config,
        schedule,
        hybrid,
        window,
        full,
        directory / "stage30-hybrid-anchor.pt",
        directory / "stage29-window-anchor.pt",
    )


def create_tiny_references(
    data: CharacterData,
    config: TrainingConfig,
    schedule: object,
    window: object,
    window_anchor: Path,
    full: object,
    directory: Path,
) -> tuple[object, object]:
    control_config, treatment_config = STAGE_30._STAGE_29._stage_26_configs(config)
    control = STAGE_30._STAGE_26.ContextSpec(
        "T=4 control",
        control_config.batch_size,
        control_config.block_size,
        directory / "stage26-control.pt",
    )
    treatment = STAGE_30._STAGE_26.ContextSpec(
        "T=8 treatment",
        treatment_config.batch_size,
        treatment_config.block_size,
        full.checkpoint_path,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        stage_26 = STAGE_30._STAGE_26.run_experiment(
            data,
            control_config,
            treatment_config,
            control,
            treatment,
            2,
            6,
            torch.device("cpu"),
            0,
            schedule,
            training_batch_seed=202,
            training_rng_seed=303,
            benchmark=None,
        )
        stage_29 = STAGE_30._STAGE_29.run_experiment(
            data,
            config,
            window,
            full,
            2,
            6,
            torch.device("cpu"),
            0,
            schedule,
            window_anchor,
            training_batch_seed=202,
            training_rng_seed=303,
            full_reference_expected_sha256=STAGE_30.checkpoint_sha256(
                full.checkpoint_path
            ),
            full_reference_expected_step=stage_26.treatment.best_step,
            benchmark=None,
        )
    return stage_26, stage_29


class Stage30TopologyTests(unittest.TestCase):
    def test_exact_middle_global_topology_and_layerwise_reachability(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            hybrid, _, _, _, _ = canonical_specs(Path(directory_name))
            masks = STAGE_30.build_layer_attention_masks(128, hybrid)

        self.assertEqual(hybrid.layer_contexts, STAGE_30.DEFAULT_LAYER_CONTEXTS)
        self.assertEqual(hybrid.global_layer_index, 2)
        self.assertEqual(hybrid.global_layer_indices, (2,))
        self.assertEqual(hybrid.local_context, 23)
        self.assertEqual(len(masks), 6)
        local = STAGE_30.build_attention_mask(128, 23)
        full = STAGE_30.build_attention_mask(128, None)
        for layer, mask in enumerate(masks):
            expected = full if layer == 2 else local
            torch.testing.assert_close(mask, expected, rtol=0, atol=0)
        self.assertEqual(
            tuple(int(mask.sum()) for mask in masks),
            STAGE_30.DEFAULT_ALLOWED_EDGES_BY_LAYER_PER_HEAD,
        )
        self.assertEqual(
            sum(int(mask.sum()) for mask in masks),
            STAGE_30.DEFAULT_ALLOWED_EDGES_PER_HEAD_ACROSS_DEPTH,
        )
        self.assertEqual(
            6 * int(full.sum()) - sum(int(mask.sum()) for mask in masks),
            STAGE_30.DEFAULT_REMOVED_EDGES_PER_HEAD_ACROSS_DEPTH,
        )

        audit = STAGE_30.audit_layerwise_reachability(masks)
        self.assertEqual(audit.layer_contexts, (23, 23, None, 23, 23, 23))
        self.assertEqual(audit.direct_attention_spans, (23, 23, 128, 23, 23, 23))
        self.assertEqual(audit.widths_by_layer, (23, 45, 128, 128, 128, 128))
        self.assertEqual(audit.first_global_reachability_layer, 3)
        self.assertEqual(
            audit.all_causal_sources_reachable_by_layer,
            (False, False, True, True, True, True),
        )
        self.assertEqual(audit.earliest_source_by_layer[0][-1], 105)
        self.assertEqual(audit.earliest_source_by_layer[1][-1], 83)
        self.assertEqual(audit.earliest_source_by_layer[2], (0,) * 128)
        self.assertEqual(audit.final_earliest_source_by_query, (0,) * 128)
        self.assertEqual(audit.final_width_by_query, tuple(range(1, 129)))
        self.assertIsNone(audit.first_query_with_unavailable_source)
        self.assertTrue(audit.all_causal_sources_reachable)
        self.assertTrue(audit.no_context_reset)
        self.assertTrue(audit.causal)
        self.assertTrue(audit.contiguous)
        STAGE_30.verify_global_reachability(audit, 128)

    def test_spec_and_protocol_pin_the_canonical_geometry(self) -> None:
        for invalid_contexts in (
            (None,),
            (23, 23, 23, 23, 23, 23),
            (23, None, None, 23, 23, 23),
            (23, 23, None, 22, 23, 23),
        ):
            with self.assertRaises(ValueError):
                STAGE_30.AttentionSpec("invalid", invalid_contexts, Path("x.pt"))

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            hybrid, window, full, hybrid_anchor, window_anchor = canonical_specs(
                directory
            )
            targets, bands = STAGE_30.validate_attention_protocol(
                canonical_config(),
                hybrid,
                window,
                full,
                4,
                6,
                hybrid_anchor_checkpoint_path=hybrid_anchor,
                window_anchor_checkpoint_path=window_anchor,
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
            self.assertEqual(STAGE_30.DEFAULT_GLOBAL_LAYER_NUMBER, 3)
            self.assertEqual(STAGE_30.DEFAULT_PRECISE_EVAL_SEED, 1_347)

            aliased_window = STAGE_30.UniformAttentionSpec(
                "aliased W=23",
                23,
                hybrid.checkpoint_path,
            )
            with self.assertRaises(ValueError):
                STAGE_30.validate_attention_protocol(
                    canonical_config(),
                    hybrid,
                    aliased_window,
                    full,
                    4,
                    6,
                )

            duplicate_name = STAGE_30.UniformAttentionSpec(
                hybrid.name,
                23,
                directory / "duplicate-name.pt",
            )
            with self.assertRaisesRegex(ValueError, "branch names"):
                STAGE_30.validate_attention_protocol(
                    canonical_config(),
                    hybrid,
                    duplicate_name,
                    full,
                    4,
                    6,
                )

    def test_model_verifier_rejects_corruption_and_relocated_global_layer(
        self,
    ) -> None:
        data = make_character_data()
        config = TrainingConfig(
            batch_size=2,
            block_size=8,
            n_embd=8,
            learning_rate=1e-3,
            max_iters=1,
            eval_interval=1,
            eval_iters=1,
            seed=101,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            spec = STAGE_30.AttentionSpec(
                "tiny hybrid",
                (3, 3, None, 3, 3, 3),
                Path(directory_name) / "hybrid.pt",
            )

            local_corrupt = STAGE_30.build_model(data, config, spec, 2, 6)
            local_corrupt.blocks[0].sa.heads[0].tril[3, 0] = 1
            with self.assertRaisesRegex(RuntimeError, "layer 0"):
                STAGE_30.verify_model_masks(local_corrupt, spec)

            global_corrupt = STAGE_30.build_model(data, config, spec, 2, 6)
            global_corrupt.blocks[2].sa.heads[0].tril.copy_(
                STAGE_30.build_attention_mask(8, 3).float()
            )
            with self.assertRaisesRegex(RuntimeError, "layer 2"):
                STAGE_30.verify_model_masks(global_corrupt, spec)

            relocated = STAGE_30.build_model(data, config, spec, 2, 6)
            full = STAGE_30.build_attention_mask(8, None).float()
            local = STAGE_30.build_attention_mask(8, 3).float()
            for head in relocated.blocks[2].sa.heads:
                head.tril.copy_(local)
            for head in relocated.blocks[3].sa.heads:
                head.tril.copy_(full)
            # This preserves the aggregate number of full/local masks, so the
            # verifier must inspect layer identity rather than edge counts.
            with self.assertRaisesRegex(RuntimeError, "layer 2"):
                STAGE_30.verify_model_masks(relocated, spec)


class Stage30InitializationTests(unittest.TestCase):
    def test_canonical_initialization_fingerprints_and_mask_only_differences(
        self,
    ) -> None:
        data = CharacterData.from_file(Path(__file__).parent / "data" / "input.txt")
        config = canonical_config()
        with tempfile.TemporaryDirectory() as directory_name:
            hybrid, window, full, _, _ = canonical_specs(Path(directory_name))
            initialization = STAGE_30.create_paired_initial_states(
                data,
                config,
                hybrid,
                window,
                full,
                4,
                6,
            )

            hybrid_model = STAGE_30.build_model(data, config, hybrid, 4, 6)
            full_model = STAGE_30._STAGE_29.build_model(data, config, full, 4, 6)
            hybrid_model.load_state_dict(initialization.hybrid, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(initialization.parameter_count, 1_220_673)
        self.assertEqual(
            initialization.parameter_state_sha256,
            "c29ae7a57c7db105f572b58c17578aea86564c7c9ae8dfb203e3179e61600477",
        )
        self.assertEqual(
            initialization.hybrid_state_sha256,
            "ea57548caadcd0bcc4298471c1e959d1c4e508e028ea19656785e594810db68a",
        )
        self.assertEqual(
            initialization.full_state_sha256,
            "c27f642dfb3ba186430d1f553b14a30fe3ce90c5062e9b2fe0ff448e88b6c8e8",
        )
        self.assertEqual(
            initialization.hybrid_mask_stack_sha256,
            "5aab8cbfff98eed7c3146df3499fcffe5fce5cb3fe4bb183213d8299888d37ef",
        )
        self.assertEqual(
            initialization.hybrid_float_mask_stack_sha256,
            "2005a635e15075b06c1a0ef6eebaab3bed5678bfb68de3b6b0192ad80916308e",
        )
        self.assertEqual(
            initialization.hybrid_mask_buffer_mapping_sha256,
            "5af692a38c69f823f5f9d835a84f1845a87388020f21dbd7863c5a6fa38b6486",
        )
        self.assertEqual(len(initialization.differing_state_names), 20)
        self.assertTrue(
            all(
                name.endswith(".tril") and not name.startswith("blocks.2.")
                for name in initialization.differing_state_names
            )
        )

        hybrid_window_differences = tuple(
            name
            for name in initialization.hybrid
            if not torch.equal(
                initialization.hybrid[name],
                initialization.window[name],
            )
        )
        self.assertEqual(len(hybrid_window_differences), 4)
        self.assertTrue(
            all(
                name.startswith("blocks.2.") and name.endswith(".tril")
                for name in hybrid_window_differences
            )
        )
        parameter_names = dict(hybrid_model.named_parameters())
        for name in parameter_names:
            torch.testing.assert_close(
                initialization.hybrid[name],
                initialization.full[name],
                rtol=0,
                atol=0,
            )
            self.assertNotEqual(
                initialization.hybrid[name].data_ptr(),
                initialization.full[name].data_ptr(),
            )

        self.assertEqual(
            STAGE_30.verify_initial_shared_prefix_equivalence(
                hybrid_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=109,
                prefix_length=23,
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "shared-prefix logits differ"):
            STAGE_30.verify_initial_shared_prefix_equivalence(
                hybrid_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=109,
                prefix_length=24,
            )


class Stage30TinyWorkflowTests(unittest.TestCase):
    def test_three_branch_workflow_metadata_stream_and_corrupt_checkpoint(
        self,
    ) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (
                config,
                schedule,
                hybrid,
                window,
                full,
                hybrid_anchor,
                window_anchor,
            ) = make_tiny_protocol(directory)
            stage_26, stage_29 = create_tiny_references(
                data,
                config,
                schedule,
                window,
                window_anchor,
                full,
                directory,
            )
            reference_paths = (
                full.checkpoint_path,
                window.checkpoint_path,
                window_anchor,
            )
            reference_bytes = tuple(path.read_bytes() for path in reference_paths)
            reference_hashes = tuple(
                STAGE_30.checkpoint_sha256(path) for path in reference_paths
            )

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
                experiment = STAGE_30.run_experiment(
                    data,
                    config,
                    hybrid,
                    window,
                    full,
                    2,
                    6,
                    torch.device("cpu"),
                    0,
                    schedule,
                    hybrid_anchor,
                    window_anchor,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    full_reference_expected_sha256=reference_hashes[0],
                    full_reference_expected_step=stage_26.treatment.best_step,
                    window_best_expected_sha256=reference_hashes[1],
                    window_anchor_expected_sha256=reference_hashes[2],
                    window_best_expected_step=stage_29.local.best_step,
                    window_anchor_expected_step=stage_29.local.anchor_step,
                    benchmark=None,
                    on_update=capture_update,
                )

            self.assertEqual(
                tuple(path.read_bytes() for path in reference_paths),
                reference_bytes,
            )
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

            self.assertEqual(experiment.initial_shared_prefix_length, 3)
            self.assertEqual(experiment.initial_shared_prefix_max_difference, 0.0)
            self.assertEqual(experiment.reachability.widths_by_layer, (3, 5, 8, 8, 8, 8))
            self.assertEqual(experiment.reachability.first_global_reachability_layer, 3)
            self.assertEqual(experiment.bands, ((0, 3), (3, 5), (5, 7), (7, 8)))
            self.assertEqual(experiment.full.checkpoint_step, stage_26.treatment.best_step)
            self.assertEqual(experiment.window.best_step, stage_29.local.best_step)
            self.assertEqual(experiment.window.anchor_step, stage_29.local.anchor_step)
            self.assertEqual(
                experiment.hybrid_checkpoint_sha256,
                STAGE_30.checkpoint_sha256(hybrid.checkpoint_path),
            )
            self.assertEqual(
                experiment.hybrid_anchor_checkpoint_sha256,
                STAGE_30.checkpoint_sha256(hybrid_anchor),
            )

            best = STAGE_30.validate_stage_30_checkpoint(
                hybrid.checkpoint_path,
                data,
                config,
                hybrid,
                window,
                full,
                2,
                6,
                schedule,
                experiment.bands,
                experiment.initialization,
                experiment.full,
                experiment.window,
                training_batch_seed=202,
                training_rng_seed=303,
                checkpoint_kind="best",
            )
            anchor = STAGE_30.validate_stage_30_checkpoint(
                hybrid_anchor,
                data,
                config,
                hybrid,
                window,
                full,
                2,
                6,
                schedule,
                experiment.bands,
                experiment.initialization,
                experiment.full,
                experiment.window,
                training_batch_seed=202,
                training_rng_seed=303,
                checkpoint_kind="anchor",
            )
            metadata = best["experiment"]
            self.assertEqual(metadata["stage"], 30)
            self.assertEqual(metadata["layer_contexts"], (3, 3, None, 3, 3, 3))
            self.assertEqual(metadata["global_layer_indices"], (2,))
            self.assertEqual(metadata["global_layer_number"], 3)
            self.assertEqual(metadata["global_attention_layers"], 1)
            self.assertEqual(metadata["local_attention_layers"], 5)
            self.assertEqual(metadata["first_global_reachability_layer"], 3)
            self.assertEqual(
                metadata["allowed_attention_edges_by_layer_per_head"],
                (21, 21, 36, 21, 21, 21),
            )
            self.assertTrue(metadata["all_references_reused_read_only"])
            self.assertTrue(metadata["only_hybrid_branch_newly_trained"])
            evaluation = best["evaluation_protocol"]
            self.assertEqual(evaluation["primary_comparison"], "full_minus_hybrid")
            self.assertEqual(
                evaluation["secondary_comparisons"],
                ("hybrid_minus_window", "full_minus_window"),
            )
            self.assertEqual(anchor["step"], experiment.full.checkpoint_step)

            persisted_paths = (
                hybrid.checkpoint_path,
                hybrid_anchor,
                *reference_paths,
            )
            persisted_bytes = tuple(path.read_bytes() for path in persisted_paths)
            precise = STAGE_30.run_paired_validation(
                data,
                config,
                hybrid,
                window,
                full,
                2,
                6,
                torch.device("cpu"),
                schedule,
                experiment,
                hybrid_anchor,
                window_anchor,
                eval_iters=2,
                seed=404,
            )
            self.assertEqual(precise.eval_iters, 2)
            self.assertEqual(precise.seed, 404)
            self.assertEqual(precise.bands, experiment.bands)
            for endpoint in (precise.anchor, precise.selected_best):
                self.assertEqual(len(endpoint.full_natural.batch_losses), 2)
                self.assertEqual(len(endpoint.hybrid_natural.batch_losses), 2)
                self.assertEqual(len(endpoint.window_natural.batch_losses), 2)
                self.assertEqual(
                    endpoint.full_minus_hybrid.candidate,
                    endpoint.full_natural.name,
                )
                self.assertEqual(
                    endpoint.full_minus_hybrid.baseline,
                    endpoint.hybrid_natural.name,
                )
                self.assertEqual(
                    endpoint.hybrid_minus_window.candidate,
                    endpoint.hybrid_natural.name,
                )
                self.assertEqual(
                    endpoint.hybrid_minus_window.baseline,
                    endpoint.window_natural.name,
                )
                self.assertEqual(
                    endpoint.full_minus_window.candidate,
                    endpoint.full_natural.name,
                )
                self.assertEqual(
                    endpoint.full_minus_window.baseline,
                    endpoint.window_natural.name,
                )
                for delta in (
                    endpoint.full_minus_hybrid_multi_hop,
                    endpoint.hybrid_minus_window_multi_hop,
                    endpoint.full_minus_window_multi_hop,
                ):
                    self.assertEqual((delta.start, delta.end), (3, 8))

                # The three results share each validation panel, so the direct
                # Full-minus-W23 delta must equal the two deltas through Hybrid.
                self.assertAlmostEqual(
                    endpoint.full_minus_window.mean_delta.overall,
                    endpoint.full_minus_hybrid.mean_delta.overall
                    + endpoint.hybrid_minus_window.mean_delta.overall,
                )
                self.assertAlmostEqual(
                    endpoint.full_minus_window_multi_hop.mean_delta,
                    endpoint.full_minus_hybrid_multi_hop.mean_delta
                    + endpoint.hybrid_minus_window_multi_hop.mean_delta,
                )
                denominator = endpoint.full_minus_window.mean_delta.overall
                expected_recovery = (
                    None
                    if denominator == 0.0
                    else endpoint.hybrid_minus_window.mean_delta.overall
                    / denominator
                )
                if expected_recovery is None:
                    self.assertIsNone(endpoint.recovery_fraction)
                else:
                    self.assertAlmostEqual(
                        endpoint.recovery_fraction,
                        expected_recovery,
                    )
            self.assertEqual(
                tuple(path.read_bytes() for path in persisted_paths),
                persisted_bytes,
            )

            corrupt_path = directory / "corrupt-hybrid.pt"
            corrupt = dict(best)
            corrupt_state = dict(corrupt["model_state_dict"])
            mask_name = next(
                name
                for name in corrupt_state
                if name.startswith("blocks.2.") and name.endswith(".tril")
            )
            corrupt_mask = corrupt_state[mask_name].clone()
            corrupt_mask[3, 0] = 0
            corrupt_state[mask_name] = corrupt_mask
            corrupt["model_state_dict"] = corrupt_state
            torch.save(corrupt, corrupt_path)
            with self.assertRaisesRegex(ValueError, "mask buffer fingerprint"):
                STAGE_30.validate_stage_30_checkpoint(
                    corrupt_path,
                    data,
                    config,
                    hybrid,
                    window,
                    full,
                    2,
                    6,
                    schedule,
                    experiment.bands,
                    experiment.initialization,
                    experiment.full,
                    experiment.window,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    checkpoint_kind="best",
                )

            self.assertEqual(
                tuple(path.read_bytes() for path in reference_paths),
                reference_bytes,
            )


if __name__ == "__main__":
    unittest.main()
