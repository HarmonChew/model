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


MODULE_PATH = Path(__file__).with_name("33_global_processing_depth.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_33_global_processing_depth_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_33 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_33
SPEC.loader.exec_module(STAGE_33)


def make_character_data(
    *,
    vocabulary_size: int = 5,
    train_tokens: int = 600,
    val_tokens: int = 300,
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


def canonical_specs(directory: Path) -> tuple[object, ...]:
    last_three = STAGE_33.AttentionSpec(
        "three final global layers",
        (23, 23, 23, None, None, None),
        directory / "last-three-best.pt",
    )
    last_two = STAGE_33._STAGE_32.AttentionSpec(
        "Stage 32 two final global reference",
        (23, 23, 23, 23, None, None),
        directory / "last-two-best.pt",
    )
    last = STAGE_33._STAGE_31.AttentionSpec(
        "Stage 31 one final global reference",
        (23, 23, 23, 23, 23, None),
        directory / "last-best.pt",
    )
    middle = STAGE_33._STAGE_31.AttentionSpec(
        "Stage 30 middle global reference",
        (23, 23, None, 23, 23, 23),
        directory / "middle-best.pt",
    )
    window = STAGE_33.UniformAttentionSpec(
        "W=23 Stage 29 reference",
        23,
        directory / "window-best.pt",
    )
    full = STAGE_33.UniformAttentionSpec(
        "Full-128 Stage 26 reference",
        None,
        directory / "full.pt",
    )
    return (
        last_three,
        last_two,
        last,
        middle,
        window,
        full,
        directory / "last-three-anchor.pt",
        directory / "last-two-anchor.pt",
        directory / "last-anchor.pt",
        directory / "middle-anchor.pt",
        directory / "window-anchor.pt",
    )


def make_tiny_protocol(directory: Path) -> tuple[object, ...]:
    schedule = STAGE_33.LearningRateSchedule(
        initial_learning_rate=2e-2,
        first_decay_step=1,
        middle_learning_rate=7e-3,
        second_decay_step=2,
        final_learning_rate=2e-3,
    )
    config = TrainingConfig(
        batch_size=2,
        block_size=12,
        n_embd=8,
        learning_rate=schedule.initial_learning_rate,
        max_iters=3,
        eval_interval=1,
        eval_iters=1,
        seed=101,
    )
    last_three = STAGE_33.AttentionSpec(
        "tiny three final global",
        (3, 3, 3, None, None, None),
        directory / "stage33-last-three-best.pt",
    )
    last_two = STAGE_33._STAGE_32.AttentionSpec(
        "tiny two final global",
        (3, 3, 3, 3, None, None),
        directory / "stage32-last-two-best.pt",
    )
    last = STAGE_33._STAGE_31.AttentionSpec(
        "tiny final global",
        (3, 3, 3, 3, 3, None),
        directory / "stage31-last-best.pt",
    )
    middle = STAGE_33._STAGE_31.AttentionSpec(
        "tiny middle global",
        (3, 3, None, 3, 3, 3),
        directory / "stage30-middle-best.pt",
    )
    window = STAGE_33.UniformAttentionSpec(
        "tiny W=3 reference",
        3,
        directory / "stage29-window-best.pt",
    )
    full = STAGE_33.UniformAttentionSpec(
        "tiny Full reference",
        None,
        directory / "stage26-full.pt",
    )
    return (
        config,
        schedule,
        last_three,
        last_two,
        last,
        middle,
        window,
        full,
        directory / "stage33-last-three-anchor.pt",
        directory / "stage32-last-two-anchor.pt",
        directory / "stage31-last-anchor.pt",
        directory / "stage30-middle-anchor.pt",
        directory / "stage29-window-anchor.pt",
    )


def create_tiny_references(
    data: CharacterData,
    config: TrainingConfig,
    schedule: object,
    last_two: object,
    last_two_anchor: Path,
    last: object,
    last_anchor: Path,
    middle: object,
    middle_anchor: Path,
    window: object,
    window_anchor: Path,
    full: object,
    directory: Path,
) -> tuple[object, object, object, object, object]:
    control_config, treatment_config = STAGE_33._STAGE_29._stage_26_configs(config)
    control = STAGE_33._STAGE_26.ContextSpec(
        f"T={control_config.block_size} control",
        control_config.batch_size,
        control_config.block_size,
        directory / "stage26-control.pt",
    )
    treatment = STAGE_33._STAGE_26.ContextSpec(
        f"T={treatment_config.block_size} treatment",
        treatment_config.batch_size,
        treatment_config.block_size,
        full.checkpoint_path,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        stage_26 = STAGE_33._STAGE_26.run_experiment(
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
        stage_29 = STAGE_33._STAGE_29.run_experiment(
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
            full_reference_expected_sha256=STAGE_33.checkpoint_sha256(
                full.checkpoint_path
            ),
            full_reference_expected_step=stage_26.treatment.best_step,
            benchmark=None,
        )
        stage_30 = STAGE_33._STAGE_30.run_experiment(
            data,
            config,
            middle,
            window,
            full,
            2,
            6,
            torch.device("cpu"),
            0,
            schedule,
            middle_anchor,
            window_anchor,
            training_batch_seed=202,
            training_rng_seed=303,
            full_reference_expected_sha256=STAGE_33.checkpoint_sha256(
                full.checkpoint_path
            ),
            full_reference_expected_step=stage_26.treatment.best_step,
            window_best_expected_sha256=STAGE_33.checkpoint_sha256(
                window.checkpoint_path
            ),
            window_anchor_expected_sha256=STAGE_33.checkpoint_sha256(window_anchor),
            window_best_expected_step=stage_29.local.best_step,
            window_anchor_expected_step=stage_29.local.anchor_step,
            benchmark=None,
        )
        stage_31 = STAGE_33._STAGE_31.run_experiment(
            data,
            config,
            last,
            middle,
            window,
            full,
            2,
            6,
            torch.device("cpu"),
            0,
            schedule,
            last_anchor,
            middle_anchor,
            window_anchor,
            training_batch_seed=202,
            training_rng_seed=303,
            full_reference_expected_sha256=STAGE_33.checkpoint_sha256(
                full.checkpoint_path
            ),
            full_reference_expected_step=stage_26.treatment.best_step,
            window_best_expected_sha256=STAGE_33.checkpoint_sha256(
                window.checkpoint_path
            ),
            window_anchor_expected_sha256=STAGE_33.checkpoint_sha256(window_anchor),
            window_best_expected_step=stage_29.local.best_step,
            window_anchor_expected_step=stage_29.local.anchor_step,
            middle_best_expected_sha256=STAGE_33.checkpoint_sha256(
                middle.checkpoint_path
            ),
            middle_anchor_expected_sha256=STAGE_33.checkpoint_sha256(middle_anchor),
            middle_best_expected_step=stage_30.hybrid.best_step,
            middle_anchor_expected_step=stage_30.hybrid.anchor_step,
            benchmark=None,
        )
        stage_32 = STAGE_33._STAGE_32.run_experiment(
            data,
            config,
            last_two,
            last,
            window,
            full,
            2,
            6,
            torch.device("cpu"),
            0,
            schedule,
            last_two_anchor,
            last_anchor,
            window_anchor,
            middle_global=middle,
            middle_global_anchor_checkpoint_path=middle_anchor,
            training_batch_seed=202,
            training_rng_seed=303,
            full_reference_expected_sha256=STAGE_33.checkpoint_sha256(
                full.checkpoint_path
            ),
            full_reference_expected_step=stage_26.treatment.best_step,
            window_best_expected_sha256=STAGE_33.checkpoint_sha256(
                window.checkpoint_path
            ),
            window_anchor_expected_sha256=STAGE_33.checkpoint_sha256(window_anchor),
            window_best_expected_step=stage_29.local.best_step,
            window_anchor_expected_step=stage_29.local.anchor_step,
            last_global_best_expected_sha256=STAGE_33.checkpoint_sha256(
                last.checkpoint_path
            ),
            last_global_anchor_expected_sha256=STAGE_33.checkpoint_sha256(last_anchor),
            last_global_best_expected_step=stage_31.last_global.best_step,
            last_global_anchor_expected_step=stage_31.last_global.anchor_step,
            middle_best_expected_sha256=STAGE_33.checkpoint_sha256(
                middle.checkpoint_path
            ),
            middle_anchor_expected_sha256=STAGE_33.checkpoint_sha256(middle_anchor),
            middle_best_expected_step=stage_30.hybrid.best_step,
            middle_anchor_expected_step=stage_30.hybrid.anchor_step,
            benchmark=None,
        )
    return stage_26, stage_29, stage_30, stage_31, stage_32


class Stage33TopologyTests(unittest.TestCase):
    def test_exact_three_final_global_topology_and_reachability(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            last_three, *_ = canonical_specs(Path(directory_name))
            masks = STAGE_33.build_layer_attention_masks(128, last_three)

        self.assertEqual(last_three.layer_contexts, STAGE_33.DEFAULT_LAYER_CONTEXTS)
        self.assertEqual(
            STAGE_33.DEFAULT_LAYER_CONTEXTS,
            (23, 23, 23, None, None, None),
        )
        self.assertEqual(STAGE_33.DEFAULT_GLOBAL_LAYER_INDICES, (3, 4, 5))
        self.assertEqual(STAGE_33.DEFAULT_GLOBAL_LAYER_NUMBERS, (4, 5, 6))
        self.assertEqual(last_three.global_layer_index, 3)
        self.assertEqual(last_three.global_layer_indices, (3, 4, 5))
        self.assertEqual(last_three.local_context, 23)
        self.assertEqual(len(masks), 6)
        local = STAGE_33.build_attention_mask(128, 23)
        full = STAGE_33.build_attention_mask(128, None)
        for layer, mask in enumerate(masks):
            expected = full if layer in (3, 4, 5) else local
            torch.testing.assert_close(mask, expected, rtol=0, atol=0)
        self.assertEqual(
            tuple(int(mask.sum()) for mask in masks),
            STAGE_33.DEFAULT_ALLOWED_EDGES_BY_LAYER_PER_HEAD,
        )
        self.assertEqual(
            STAGE_33.DEFAULT_ALLOWED_EDGES_BY_LAYER_PER_HEAD,
            (2_691, 2_691, 2_691, 8_256, 8_256, 8_256),
        )
        self.assertEqual(
            sum(int(mask.sum()) for mask in masks),
            STAGE_33.DEFAULT_ALLOWED_EDGES_PER_HEAD_ACROSS_DEPTH,
        )
        self.assertEqual(STAGE_33.DEFAULT_ALLOWED_EDGES_PER_HEAD_ACROSS_DEPTH, 32_841)
        self.assertEqual(
            6 * int(full.sum()) - sum(int(mask.sum()) for mask in masks),
            STAGE_33.DEFAULT_REMOVED_EDGES_PER_HEAD_ACROSS_DEPTH,
        )
        self.assertEqual(STAGE_33.DEFAULT_REMOVED_EDGES_PER_HEAD_ACROSS_DEPTH, 16_695)
        self.assertEqual(STAGE_33.DEFAULT_DIFFERING_MASK_BUFFER_COUNT, 12)
        self.assertEqual(
            STAGE_33.DEFAULT_DIFFERING_FROM_WINDOW_MASK_BUFFER_COUNT,
            12,
        )
        self.assertEqual(
            STAGE_33.DEFAULT_DIFFERING_FROM_LAST_TWO_GLOBAL_MASK_BUFFER_COUNT,
            4,
        )

        audit = STAGE_33.audit_layerwise_reachability(masks)
        self.assertEqual(audit.layer_contexts, (23, 23, 23, None, None, None))
        self.assertEqual(audit.direct_attention_spans, (23, 23, 23, 128, 128, 128))
        self.assertEqual(audit.widths_by_layer, (23, 45, 67, 128, 128, 128))
        self.assertEqual(audit.first_global_reachability_layer, 4)
        self.assertEqual(
            audit.all_causal_sources_reachable_by_layer,
            (False, False, False, True, True, True),
        )
        self.assertEqual(
            tuple(layer[-1] for layer in audit.earliest_source_by_layer),
            (105, 83, 61, 0, 0, 0),
        )
        self.assertEqual(audit.final_earliest_source_by_query, (0,) * 128)
        self.assertEqual(audit.final_width_by_query, tuple(range(1, 129)))
        self.assertIsNone(audit.first_query_with_unavailable_source)
        self.assertTrue(audit.all_causal_sources_reachable)
        self.assertTrue(audit.no_context_reset)
        self.assertTrue(audit.causal)
        self.assertTrue(audit.contiguous)
        STAGE_33.verify_global_reachability(audit, 128)

    def test_spec_protocol_and_path_aliases_pin_the_four_reported_branches(
        self,
    ) -> None:
        for invalid_contexts in (
            (23, 23, 23, 23, None, None),
            (23, 23, 23, 23, 23, 23),
            (23, 23, None, None, None, 23),
            (23, None, 23, None, 23, None),
            (23, 22, 23, None, None, None),
        ):
            with self.assertRaises(ValueError):
                STAGE_33.AttentionSpec("invalid", invalid_contexts, Path("x.pt"))

        self.assertEqual(
            STAGE_33.DEFAULT_HYBRID_CHECKPOINT_PATH,
            STAGE_33.DEFAULT_LAST_THREE_GLOBAL_CHECKPOINT_PATH,
        )
        self.assertEqual(
            STAGE_33.DEFAULT_LOCAL_CHECKPOINT_PATH,
            STAGE_33.DEFAULT_LAST_THREE_GLOBAL_CHECKPOINT_PATH,
        )
        self.assertEqual(
            STAGE_33.DEFAULT_HYBRID_ANCHOR_CHECKPOINT_PATH,
            STAGE_33.DEFAULT_LAST_THREE_GLOBAL_ANCHOR_CHECKPOINT_PATH,
        )
        self.assertEqual(
            STAGE_33.DEFAULT_LOCAL_ANCHOR_CHECKPOINT_PATH,
            STAGE_33.DEFAULT_LAST_THREE_GLOBAL_ANCHOR_CHECKPOINT_PATH,
        )

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (
                last_three,
                last_two,
                last,
                middle,
                window,
                full,
                last_three_anchor,
                last_two_anchor,
                last_anchor,
                middle_anchor,
                window_anchor,
            ) = canonical_specs(directory)
            targets, bands = STAGE_33.validate_attention_protocol(
                canonical_config(),
                last_three,
                last_two,
                window,
                full,
                4,
                6,
                last_three_global_anchor_checkpoint_path=last_three_anchor,
                last_two_global_anchor_checkpoint_path=last_two_anchor,
                window_anchor_checkpoint_path=window_anchor,
                last_global=last,
                last_global_anchor_checkpoint_path=last_anchor,
                middle_global=middle,
                middle_global_anchor_checkpoint_path=middle_anchor,
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
            self.assertEqual(STAGE_33.DEFAULT_PRECISE_EVAL_SEED, 1_350)

            mismatched_last_two = STAGE_33._STAGE_32.AttentionSpec(
                "mismatched Last2Full",
                (22, 22, 22, 22, None, None),
                directory / "mismatched-last-two.pt",
            )
            with self.assertRaisesRegex(ValueError, "share one local window"):
                STAGE_33.validate_attention_protocol(
                    canonical_config(),
                    last_three,
                    mismatched_last_two,
                    window,
                    full,
                )

            aliased_candidate = STAGE_33.AttentionSpec(
                "aliased Last3Full",
                last_three.layer_contexts,
                last_two.checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "distinct"):
                STAGE_33.validate_attention_protocol(
                    canonical_config(),
                    aliased_candidate,
                    last_two,
                    window,
                    full,
                )

            with self.assertRaisesRegex(ValueError, "distinct"):
                STAGE_33.validate_attention_protocol(
                    canonical_config(),
                    last_three,
                    last_two,
                    window,
                    full,
                    last_three_global_anchor_checkpoint_path=full.checkpoint_path,
                )

            temporary_alias = STAGE_33._STAGE_32.AttentionSpec(
                "temporary alias",
                last_two.layer_contexts,
                last_three.checkpoint_path.with_name(
                    f".{last_three.checkpoint_path.name}.tmp"
                ),
            )
            with self.assertRaisesRegex(ValueError, "distinct"):
                STAGE_33.validate_attention_protocol(
                    canonical_config(),
                    last_three,
                    temporary_alias,
                    window,
                    full,
                )

            duplicate_name = STAGE_33.UniformAttentionSpec(
                last_three.name,
                23,
                directory / "duplicate-name.pt",
            )
            with self.assertRaisesRegex(ValueError, "names"):
                STAGE_33.validate_attention_protocol(
                    canonical_config(),
                    last_three,
                    last_two,
                    duplicate_name,
                    full,
                )

            with self.assertRaisesRegex(ValueError, "provided together"):
                STAGE_33.validate_attention_protocol(
                    canonical_config(),
                    last_three,
                    last_two,
                    window,
                    full,
                    last_global=last,
                )

    def test_model_verifier_rejects_corruption_and_relocated_global_layer(
        self,
    ) -> None:
        data = make_character_data()
        config = TrainingConfig(
            batch_size=2,
            block_size=12,
            n_embd=8,
            learning_rate=1e-3,
            max_iters=1,
            eval_interval=1,
            eval_iters=1,
            seed=101,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            spec = STAGE_33.AttentionSpec(
                "tiny three final global",
                (3, 3, 3, None, None, None),
                Path(directory_name) / "last-three.pt",
            )

            local_corrupt = STAGE_33.build_model(data, config, spec, 2, 6)
            local_corrupt.blocks[0].sa.heads[0].tril[3, 0] = 1
            with self.assertRaisesRegex(RuntimeError, "layer 0"):
                STAGE_33.verify_model_masks(local_corrupt, spec)

            global_corrupt = STAGE_33.build_model(data, config, spec, 2, 6)
            global_corrupt.blocks[3].sa.heads[0].tril.copy_(
                STAGE_33.build_attention_mask(12, 3).float()
            )
            with self.assertRaisesRegex(RuntimeError, "layer 3"):
                STAGE_33.verify_model_masks(global_corrupt, spec)

            relocated = STAGE_33.build_model(data, config, spec, 2, 6)
            full = STAGE_33.build_attention_mask(12, None).float()
            local = STAGE_33.build_attention_mask(12, 3).float()
            for head in relocated.blocks[3].sa.heads:
                head.tril.copy_(local)
            for head in relocated.blocks[2].sa.heads:
                head.tril.copy_(full)
            with self.assertRaisesRegex(RuntimeError, "layer 2"):
                STAGE_33.verify_model_masks(relocated, spec)


class Stage33InitializationTests(unittest.TestCase):
    def test_canonical_fingerprints_and_three_mask_only_differences(self) -> None:
        data = CharacterData.from_file(Path(__file__).parent / "data" / "input.txt")
        config = canonical_config()
        with tempfile.TemporaryDirectory() as directory_name:
            last_three, last_two, _, _, window, full, *_ = canonical_specs(
                Path(directory_name)
            )
            initialization = STAGE_33.create_paired_initial_states(
                data,
                config,
                last_three,
                window,
                full,
                4,
                6,
            )
            last_two_initialization = STAGE_33._STAGE_32.create_paired_initial_states(
                data,
                config,
                last_two,
                window,
                full,
                4,
                6,
            )

            last_three_model = STAGE_33.build_model(
                data,
                config,
                last_three,
                4,
                6,
            )
            full_model = STAGE_33._STAGE_29.build_model(
                data,
                config,
                full,
                4,
                6,
            )
            last_three_model.load_state_dict(initialization.hybrid, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(initialization.parameter_count, 1_220_673)
        self.assertEqual(
            initialization.parameter_state_sha256,
            "c22ae640e5d1e70a0b789a30bb5b9f9be833dfc090147151a8a303409e41c53e",
        )
        self.assertEqual(
            initialization.hybrid_state_sha256,
            "c1e2b24d9a0083dbbb841d34009b4526710e45cf1d9b09d02d4722f2d00b98d2",
        )
        self.assertEqual(
            initialization.window_state_sha256,
            "4feb26dae434a6fad7a866895b3e53d7efcec79d3dc31f6af5a412fb758cd4c1",
        )
        self.assertEqual(
            initialization.full_state_sha256,
            "4223e34a42abf674aee39ab130228c5cd8007860f0466906a4d5143cd64f3726",
        )
        self.assertEqual(
            initialization.hybrid_mask_stack_sha256,
            "7d4a87eab60a08228dbe6b1b8f0f999d9e8a74af08b4d3d295512e192194d35b",
        )
        self.assertEqual(
            initialization.hybrid_float_mask_stack_sha256,
            "4291fe296032a817d8b14ac2f2322105d60e90c18fa105e3634226e16b2a0736",
        )
        self.assertEqual(
            initialization.hybrid_mask_buffer_mapping_sha256,
            "10d2d6aa5633701371e31616368914b1de8159a93840f8dce6e668cf9aa8775d",
        )

        differences_from_full = initialization.differing_state_names
        differences_from_window = tuple(
            name
            for name in initialization.hybrid
            if not torch.equal(initialization.hybrid[name], initialization.window[name])
        )
        differences_from_last_two = tuple(
            name
            for name in initialization.hybrid
            if not torch.equal(
                initialization.hybrid[name],
                last_two_initialization.hybrid[name],
            )
        )
        self.assertEqual(len(differences_from_full), 12)
        self.assertTrue(
            all(
                name.startswith(("blocks.0.", "blocks.1.", "blocks.2."))
                and name.endswith(".tril")
                for name in differences_from_full
            )
        )
        self.assertEqual(len(differences_from_window), 12)
        self.assertTrue(
            all(
                name.startswith(("blocks.3.", "blocks.4.", "blocks.5."))
                and name.endswith(".tril")
                for name in differences_from_window
            )
        )
        self.assertEqual(len(differences_from_last_two), 4)
        self.assertTrue(
            all(
                name.startswith("blocks.3.") and name.endswith(".tril")
                for name in differences_from_last_two
            )
        )

        for name in dict(last_three_model.named_parameters()):
            torch.testing.assert_close(
                initialization.hybrid[name],
                initialization.full[name],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                initialization.hybrid[name],
                initialization.window[name],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                initialization.hybrid[name],
                last_two_initialization.hybrid[name],
                rtol=0,
                atol=0,
            )
            self.assertNotEqual(
                initialization.hybrid[name].data_ptr(),
                initialization.full[name].data_ptr(),
            )

        self.assertEqual(
            STAGE_33.verify_initial_shared_prefix_equivalence(
                last_three_model,
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
            STAGE_33.verify_initial_shared_prefix_equivalence(
                last_three_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=109,
                prefix_length=24,
            )


class Stage33TinyWorkflowTests(unittest.TestCase):
    def test_full_provenance_checkpoint_contract_and_paired_directions(self) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (
                config,
                schedule,
                last_three,
                last_two,
                last,
                middle,
                window,
                full,
                last_three_anchor,
                last_two_anchor,
                last_anchor,
                middle_anchor,
                window_anchor,
            ) = make_tiny_protocol(directory)
            stage_26, stage_29, stage_30, stage_31, stage_32 = create_tiny_references(
                data,
                config,
                schedule,
                last_two,
                last_two_anchor,
                last,
                last_anchor,
                middle,
                middle_anchor,
                window,
                window_anchor,
                full,
                directory,
            )
            reference_paths = (
                full.checkpoint_path,
                window.checkpoint_path,
                window_anchor,
                middle.checkpoint_path,
                middle_anchor,
                last.checkpoint_path,
                last_anchor,
                last_two.checkpoint_path,
                last_two_anchor,
            )
            reference_bytes = tuple(path.read_bytes() for path in reference_paths)
            reference_hashes = tuple(
                STAGE_33.checkpoint_sha256(path) for path in reference_paths
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
                experiment = STAGE_33.run_experiment(
                    data,
                    config,
                    last_three,
                    last_two,
                    window,
                    full,
                    2,
                    6,
                    torch.device("cpu"),
                    0,
                    schedule,
                    last_three_anchor,
                    last_two_anchor,
                    window_anchor,
                    last_global=last,
                    last_global_anchor_checkpoint_path=last_anchor,
                    middle_global=middle,
                    middle_global_anchor_checkpoint_path=middle_anchor,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    full_reference_expected_sha256=reference_hashes[0],
                    full_reference_expected_step=stage_26.treatment.best_step,
                    window_best_expected_sha256=reference_hashes[1],
                    window_anchor_expected_sha256=reference_hashes[2],
                    window_best_expected_step=stage_29.local.best_step,
                    window_anchor_expected_step=stage_29.local.anchor_step,
                    last_two_global_best_expected_sha256=reference_hashes[7],
                    last_two_global_anchor_expected_sha256=reference_hashes[8],
                    last_two_global_best_expected_step=(
                        stage_32.last_two_global.best_step
                    ),
                    last_two_global_anchor_expected_step=(
                        stage_32.last_two_global.anchor_step
                    ),
                    last_global_best_expected_sha256=reference_hashes[5],
                    last_global_anchor_expected_sha256=reference_hashes[6],
                    last_global_best_expected_step=stage_31.last_global.best_step,
                    last_global_anchor_expected_step=(stage_31.last_global.anchor_step),
                    middle_best_expected_sha256=reference_hashes[3],
                    middle_anchor_expected_sha256=reference_hashes[4],
                    middle_best_expected_step=stage_30.hybrid.best_step,
                    middle_anchor_expected_step=stage_30.hybrid.anchor_step,
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
            self.assertEqual(
                experiment.reachability.widths_by_layer,
                (3, 5, 7, 12, 12, 12),
            )
            self.assertEqual(
                experiment.reachability.first_global_reachability_layer,
                4,
            )
            self.assertEqual(
                experiment.bands,
                ((0, 3), (3, 5), (5, 7), (7, 9), (9, 11), (11, 12)),
            )
            self.assertEqual(
                experiment.full.checkpoint_step,
                stage_26.treatment.best_step,
            )
            self.assertEqual(experiment.window.best_step, stage_29.local.best_step)
            self.assertEqual(
                experiment.last_two_global.best_step,
                stage_32.last_two_global.best_step,
            )
            self.assertEqual(
                experiment.last_two_global.anchor_step,
                stage_32.last_two_global.anchor_step,
            )
            self.assertEqual(
                experiment.last_three_global_checkpoint_sha256,
                STAGE_33.checkpoint_sha256(last_three.checkpoint_path),
            )
            self.assertEqual(
                experiment.last_three_global_anchor_checkpoint_sha256,
                STAGE_33.checkpoint_sha256(last_three_anchor),
            )

            validation_args = (
                data,
                config,
                last_three,
                last_two,
                window,
                full,
                2,
                6,
                schedule,
                experiment.bands,
                experiment.initialization,
                experiment.full,
                experiment.window,
                experiment.last_two_global,
            )
            best = STAGE_33.validate_stage_33_checkpoint(
                last_three.checkpoint_path,
                *validation_args,
                training_batch_seed=202,
                training_rng_seed=303,
                checkpoint_kind="best",
            )
            anchor = STAGE_33.validate_stage_33_checkpoint(
                last_three_anchor,
                *validation_args,
                training_batch_seed=202,
                training_rng_seed=303,
                checkpoint_kind="anchor",
            )
            architecture = best["architecture"]
            self.assertEqual(
                architecture["layer_contexts"], (3, 3, 3, None, None, None)
            )
            self.assertEqual(architecture["global_layer_indices"], (3, 4, 5))
            self.assertEqual(
                architecture["effective_receptive_field_widths_by_layer"],
                (3, 5, 7, 12, 12, 12),
            )
            metadata = best["experiment"]
            self.assertEqual(metadata["stage"], 33)
            self.assertEqual(metadata["layer_contexts"], (3, 3, 3, None, None, None))
            self.assertEqual(metadata["global_layer_indices"], (3, 4, 5))
            self.assertEqual(metadata["global_layer_numbers"], (4, 5, 6))
            self.assertEqual(
                metadata["global_layer_placement"], "consecutive_final_suffix"
            )
            self.assertEqual(metadata["global_attention_layers"], 3)
            self.assertEqual(metadata["local_attention_layers"], 3)
            self.assertEqual(metadata["local_layers_before_global"], 3)
            self.assertEqual(metadata["local_layers_after_global"], 0)
            self.assertTrue(metadata["no_post_global_local_processing"])
            self.assertEqual(
                metadata["last_two_global_reference_layer_contexts"],
                (3, 3, 3, 3, None, None),
            )
            self.assertEqual(
                metadata["last_two_global_reference_global_layer_indices"],
                (4, 5),
            )
            self.assertEqual(
                metadata["last_three_vs_last_two_differing_mask_buffer_count"],
                2,
            )
            self.assertTrue(
                all(
                    name.startswith("blocks.3.") and name.endswith(".tril")
                    for name in metadata[
                        "last_three_vs_last_two_differing_mask_buffer_names"
                    ]
                )
            )
            self.assertEqual(metadata["first_query_with_topology_difference"], 3)
            self.assertFalse(metadata["intervention_changes_global_layer_placement"])
            self.assertTrue(metadata["intervention_changes_number_of_global_layers"])
            self.assertTrue(metadata["intervention_adds_iterative_global_computation"])
            self.assertTrue(metadata["intervention_adds_third_global_processing_round"])
            self.assertEqual(metadata["first_global_reachability_layer"], 4)
            self.assertEqual(
                metadata["allowed_attention_edges_by_layer_per_head"],
                (33, 33, 33, 78, 78, 78),
            )
            self.assertEqual(metadata["last_two_global_reference_source_stage"], 32)
            self.assertEqual(metadata["last_global_reference_source_stage"], 31)
            self.assertEqual(metadata["middle_reference_source_stage"], 30)
            self.assertTrue(metadata["all_references_reused_read_only"])
            self.assertTrue(metadata["only_last_three_global_branch_newly_trained"])
            evaluation = best["evaluation_protocol"]
            self.assertEqual(
                evaluation["primary_comparison"],
                "last_three_global_minus_last_two_global",
            )
            self.assertEqual(anchor["step"], experiment.full.checkpoint_step)

            matched_references = (
                experiment.full.checkpoint,
                experiment.window.anchor_checkpoint,
                experiment.last_two_global.last_global_reference.middle_reference.anchor_checkpoint,
                experiment.last_two_global.last_global_reference.anchor_checkpoint,
                experiment.last_two_global.anchor_checkpoint,
            )
            for reference_checkpoint in matched_references:
                STAGE_33.assert_matched_training_stream_state(
                    reference_checkpoint,
                    anchor,
                )

            persisted_paths = (
                last_three.checkpoint_path,
                last_three_anchor,
                *reference_paths,
            )
            persisted_bytes = tuple(path.read_bytes() for path in persisted_paths)
            precise = STAGE_33.run_paired_validation(
                data,
                config,
                last_three,
                last_two,
                window,
                full,
                2,
                6,
                torch.device("cpu"),
                schedule,
                experiment,
                last_three_anchor,
                last_two_anchor,
                window_anchor,
                eval_iters=2,
                seed=404,
            )
            self.assertEqual(precise.eval_iters, 2)
            self.assertEqual(precise.seed, 404)
            self.assertEqual(precise.bands, experiment.bands)
            for endpoint in (precise.anchor, precise.selected_best):
                for result in (
                    endpoint.full_natural,
                    endpoint.last_three_global_natural,
                    endpoint.last_two_global_natural,
                    endpoint.window_natural,
                ):
                    self.assertEqual(len(result.batch_losses), 2)

                directional_pairs = (
                    (
                        endpoint.last_three_global_minus_last_two_global,
                        endpoint.last_three_global_natural,
                        endpoint.last_two_global_natural,
                    ),
                    (
                        endpoint.full_minus_last_three_global,
                        endpoint.full_natural,
                        endpoint.last_three_global_natural,
                    ),
                    (
                        endpoint.last_three_global_minus_window,
                        endpoint.last_three_global_natural,
                        endpoint.window_natural,
                    ),
                    (
                        endpoint.full_minus_last_two_global,
                        endpoint.full_natural,
                        endpoint.last_two_global_natural,
                    ),
                    (
                        endpoint.last_two_global_minus_window,
                        endpoint.last_two_global_natural,
                        endpoint.window_natural,
                    ),
                    (
                        endpoint.full_minus_window,
                        endpoint.full_natural,
                        endpoint.window_natural,
                    ),
                )
                for delta, candidate, baseline in directional_pairs:
                    self.assertEqual(delta.candidate, candidate.name)
                    self.assertEqual(delta.baseline, baseline.name)

                for delta in (
                    endpoint.last_three_global_minus_last_two_global_range,
                    endpoint.full_minus_last_three_global_range,
                    endpoint.last_three_global_minus_window_range,
                    endpoint.full_minus_last_two_global_range,
                    endpoint.last_two_global_minus_window_range,
                    endpoint.full_minus_window_range,
                ):
                    self.assertEqual((delta.start, delta.end), (3, 12))

                self.assert_position_losses_add(
                    endpoint.full_minus_window.mean_delta,
                    endpoint.full_minus_last_three_global.mean_delta,
                    endpoint.last_three_global_minus_window.mean_delta,
                )
                self.assert_position_losses_add(
                    endpoint.last_three_global_minus_window.mean_delta,
                    endpoint.last_three_global_minus_last_two_global.mean_delta,
                    endpoint.last_two_global_minus_window.mean_delta,
                )
                self.assert_position_losses_add(
                    endpoint.full_minus_last_two_global.mean_delta,
                    endpoint.full_minus_last_three_global.mean_delta,
                    endpoint.last_three_global_minus_last_two_global.mean_delta,
                )
                self.assertAlmostEqual(
                    endpoint.full_minus_window_range.mean_delta,
                    endpoint.full_minus_last_three_global_range.mean_delta
                    + endpoint.last_three_global_minus_window_range.mean_delta,
                )
                self.assertAlmostEqual(
                    endpoint.last_three_global_minus_window_range.mean_delta,
                    endpoint.last_three_global_minus_last_two_global_range.mean_delta
                    + endpoint.last_two_global_minus_window_range.mean_delta,
                )
                denominator = endpoint.full_minus_window.mean_delta.overall
                for actual, numerator in (
                    (
                        endpoint.last_three_global_recovery_fraction,
                        endpoint.last_three_global_minus_window.mean_delta.overall,
                    ),
                    (
                        endpoint.last_two_global_recovery_fraction,
                        endpoint.last_two_global_minus_window.mean_delta.overall,
                    ),
                ):
                    if denominator == 0.0:
                        self.assertIsNone(actual)
                    else:
                        self.assertAlmostEqual(actual, numerator / denominator)

            self.assertEqual(
                tuple(path.read_bytes() for path in persisted_paths),
                persisted_bytes,
            )

            corrupt_path = directory / "corrupt-last-three.pt"
            corrupt = dict(best)
            corrupt_state = dict(corrupt["model_state_dict"])
            mask_name = next(
                name
                for name in corrupt_state
                if name.startswith("blocks.3.") and name.endswith(".tril")
            )
            corrupt_mask = corrupt_state[mask_name].clone()
            corrupt_mask[3, 0] = 0
            corrupt_state[mask_name] = corrupt_mask
            corrupt["model_state_dict"] = corrupt_state
            torch.save(corrupt, corrupt_path)
            with self.assertRaisesRegex(ValueError, "mask buffer fingerprint"):
                STAGE_33.validate_stage_33_checkpoint(
                    corrupt_path,
                    *validation_args,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    checkpoint_kind="best",
                )

            original_stage_32_bytes = last_two.checkpoint_path.read_bytes()
            try:
                last_two.checkpoint_path.write_bytes(
                    original_stage_32_bytes + b"stage-32-corruption"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "read-only reference changed"
                ):
                    STAGE_33.run_paired_validation(
                        data,
                        config,
                        last_three,
                        last_two,
                        window,
                        full,
                        2,
                        6,
                        torch.device("cpu"),
                        schedule,
                        experiment,
                        last_three_anchor,
                        last_two_anchor,
                        window_anchor,
                        eval_iters=2,
                        seed=404,
                    )
            finally:
                last_two.checkpoint_path.write_bytes(original_stage_32_bytes)

            self.assertEqual(
                tuple(path.read_bytes() for path in reference_paths),
                reference_bytes,
            )

    def assert_position_losses_add(
        self,
        total: object,
        first: object,
        second: object,
    ) -> None:
        self.assertAlmostEqual(total.overall, first.overall + second.overall)
        self.assertEqual(len(total.bands), len(first.bands))
        self.assertEqual(len(total.bands), len(second.bands))
        for total_band, first_band, second_band in zip(
            total.bands,
            first.bands,
            second.bands,
            strict=True,
        ):
            self.assertAlmostEqual(total_band, first_band + second_band)


if __name__ == "__main__":
    unittest.main()
