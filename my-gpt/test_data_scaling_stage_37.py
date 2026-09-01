from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary, DEFAULT_DATA_PATH


MODULE_PATH = Path(__file__).with_name("37_data_scaling.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_37_data_scaling_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_37 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_37
SPEC.loader.exec_module(STAGE_37)


def make_character_data(
    *,
    vocabulary_size: int = 5,
    train_tokens: int = 1_024,
    val_tokens: int = 512,
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


def make_tiny_protocol() -> tuple[TrainingConfig, object]:
    schedule = STAGE_37.LearningRateSchedule(
        initial_learning_rate=2e-2,
        first_decay_step=1,
        middle_learning_rate=7e-3,
        second_decay_step=2,
        final_learning_rate=2e-3,
    )
    config = TrainingConfig(
        batch_size=2,
        block_size=24,
        n_embd=8,
        learning_rate=schedule.initial_learning_rate,
        max_iters=3,
        eval_interval=1,
        eval_iters=1,
        seed=101,
    )
    return config, schedule


class Stage37DataScalingTests(unittest.TestCase):
    def test_locked_defaults_topology_budget_and_cli(self) -> None:
        self.assertEqual(STAGE_37.EXPERIMENT_STAGE, 37)
        self.assertEqual(
            (
                STAGE_37.DEFAULT_BATCH_SIZE,
                STAGE_37.DEFAULT_BLOCK_SIZE,
                STAGE_37.DEFAULT_N_EMBD,
                STAGE_37.DEFAULT_N_HEAD,
                STAGE_37.DEFAULT_HEAD_SIZE,
                STAGE_37.DEFAULT_FEED_FORWARD_WIDTH,
                STAGE_37.DEFAULT_N_LAYER,
                STAGE_37.DEFAULT_DROPOUT,
            ),
            (8, 256, 128, 4, 32, 512, 6, 0.10),
        )
        self.assertEqual(
            STAGE_37.DEFAULT_LAYER_CONTEXTS,
            (23, 23, 23, 23, None, None),
        )
        self.assertEqual(STAGE_37.DEFAULT_TARGETS_PER_UPDATE, 2_048)
        self.assertEqual(STAGE_37.DEFAULT_MAX_ITERS, 18_000)
        self.assertEqual(STAGE_37.DEFAULT_TOTAL_TARGET_CHARACTERS, 36_864_000)
        self.assertEqual(
            STAGE_37.DEFAULT_TARGETS_PER_UPDATE * STAGE_37.DEFAULT_MAX_ITERS,
            STAGE_37.DEFAULT_TOTAL_TARGET_CHARACTERS,
        )
        self.assertFalse(STAGE_37.HISTORICAL_CONTROL_RERUN)
        self.assertFalse(STAGE_37.RAW_LOSS_CROSS_CORPUS_COMPARABLE)

        masks = STAGE_37.build_layer_attention_masks(256)
        self.assertEqual(
            tuple(int(mask.sum()) for mask in masks),
            STAGE_37.DEFAULT_LOGICAL_EDGES_BY_LAYER,
        )
        audit = STAGE_37._STAGE_36.audit_layerwise_reachability(masks)
        self.assertEqual(audit.widths_by_layer, (23, 45, 67, 89, 256, 256))
        self.assertEqual(audit.first_global_reachability_layer, 5)
        self.assertTrue(audit.all_causal_sources_reachable)

        with patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_37.parse_args()
        self.assertEqual(args.data, STAGE_37.DEFAULT_COMPLETE_DATA_PATH)
        self.assertEqual(
            (args.batch_size, args.block_size, args.max_iters),
            (8, 256, 18_000),
        )
        self.assertNotEqual(DEFAULT_DATA_PATH, STAGE_37.DEFAULT_COMPLETE_DATA_PATH)
        self.assertEqual(DEFAULT_DATA_PATH.name, "input.txt")
        with self.assertRaisesRegex(ValueError, "collides"):
            STAGE_37.validate_checkpoint_path(
                STAGE_37.DEFAULT_COMPLETE_DATA_PATH,
                source_path=STAGE_37.DEFAULT_COMPLETE_DATA_PATH,
            )

    def test_complete_corpus_identity_initialization_and_parameters(self) -> None:
        data = CharacterData.from_file(STAGE_37.DEFAULT_COMPLETE_DATA_PATH)
        config = TrainingConfig(
            batch_size=STAGE_37.DEFAULT_BATCH_SIZE,
            block_size=STAGE_37.DEFAULT_BLOCK_SIZE,
            n_embd=STAGE_37.DEFAULT_N_EMBD,
            learning_rate=STAGE_37.DEFAULT_INITIAL_LEARNING_RATE,
            max_iters=STAGE_37.DEFAULT_MAX_ITERS,
            eval_interval=STAGE_37.DEFAULT_EVAL_INTERVAL,
            eval_iters=STAGE_37.DEFAULT_EVAL_ITERS,
            seed=STAGE_37.DEFAULT_SEED,
        )
        summary = STAGE_37.summarize_corpus(
            data,
            config,
            source_path=STAGE_37.DEFAULT_COMPLETE_DATA_PATH,
        )
        self.assertIn(
            (summary.source_bytes, summary.source_sha256),
            {
                (
                    STAGE_37.DEFAULT_COMPLETE_SOURCE_BYTES,
                    STAGE_37.DEFAULT_COMPLETE_SOURCE_SHA256,
                ),
                (
                    STAGE_37.DEFAULT_COMPLETE_LF_SOURCE_BYTES,
                    STAGE_37.DEFAULT_COMPLETE_LF_SOURCE_SHA256,
                ),
            },
        )
        self.assertEqual(
            (
                summary.data_fingerprint,
                summary.num_characters,
                summary.train_characters,
                summary.val_characters,
                summary.vocabulary_size,
            ),
            (
                STAGE_37.DEFAULT_COMPLETE_DATA_FINGERPRINT,
                5_343_766,
                4_809_389,
                534_377,
                98,
            ),
        )
        self.assertAlmostEqual(summary.train_pool_multiplier_vs_tiny, 4.7909, places=4)
        self.assertAlmostEqual(summary.train_corpus_equivalents, 7.6650, places=4)
        self.assertAlmostEqual(
            STAGE_37.DEFAULT_HISTORICAL_CONTROL.train_corpus_equivalents,
            36.7225,
            places=4,
        )
        self.assertAlmostEqual(math.log(summary.vocabulary_size), 4.5850, places=4)

        STAGE_37.seed_everything(config.seed)
        model = STAGE_37.build_model(data, config, 4, 6)
        STAGE_37.verify_model_masks(model)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            STAGE_37.DEFAULT_PARAMETER_COUNT,
        )
        self.assertEqual(
            STAGE_37.tensor_mapping_sha256(model.state_dict()),
            STAGE_37.DEFAULT_INITIAL_STATE_SHA256,
        )

    def test_sampler_matches_stage_36_long_panel_geometry(self) -> None:
        data = make_character_data(vocabulary_size=17, train_tokens=3_000)
        config, _ = make_tiny_protocol()
        generator = torch.Generator().manual_seed(77)
        inputs, targets = STAGE_37.get_stage_36_matched_batch(
            data,
            "train",
            config,
            generator,
            torch.device("cpu"),
        )
        self.assertEqual(tuple(inputs.shape), (2, 24))
        self.assertEqual(tuple(targets.shape), (2, 24))
        # The two T24 rows are consecutive halves of one sampled T48 panel.
        self.assertEqual(int(inputs[1, 0]), int((inputs[0, -1] + 1) % 17))
        self.assertTrue(torch.equal(targets[:, :-1], inputs[:, 1:]))
        self.assertEqual(int(targets[0, -1]), int(inputs[1, 0]))

    def test_single_branch_training_checkpoint_and_wrong_corpus_rejection(self) -> None:
        data = make_character_data()
        config, schedule = make_tiny_protocol()
        with tempfile.TemporaryDirectory() as raw_directory:
            checkpoint_path = Path(raw_directory) / "stage37.pt"
            original_build_model = STAGE_37.build_model
            with (
                patch.object(
                    STAGE_37,
                    "build_model",
                    wraps=original_build_model,
                ) as build_model_mock,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = STAGE_37.run_experiment(
                    data,
                    config,
                    2,
                    6,
                    torch.device("cpu"),
                    0,
                    schedule,
                    checkpoint_path,
                    source_path=None,
                    training_batch_seed=102,
                    fixed_panel_seed=103,
                    training_rng_seed=104,
                )
            self.assertEqual(build_model_mock.call_count, 1)
            self.assertEqual(
                tuple(record.step for record in report.training.history),
                (0, 1, 2, 3),
            )
            self.assertEqual(report.corpus.targets_per_update, 48)
            self.assertEqual(report.corpus.total_target_presentations, 144)
            self.assertEqual(report.attention_shape, (2, 2, 24, 24))
            self.assertTrue(checkpoint_path.exists())

            payload = STAGE_37.validate_stage_37_checkpoint(
                checkpoint_path,
                data,
                config,
                2,
                6,
                schedule,
            )
            self.assertEqual(payload["experiment"]["stage"], 37)
            self.assertEqual(payload["experiment"]["trained_branches"], 1)
            self.assertFalse(payload["experiment"]["tiny_shakespeare_rerun"])
            self.assertFalse(payload["experiment"]["vocabulary_matched"])
            self.assertFalse(payload["experiment"]["validation_split_matched"])
            self.assertFalse(payload["experiment"]["training_stream_paired"])
            self.assertTrue(payload["experiment"]["training_sampler_geometry_matched"])
            self.assertEqual(payload["architecture"]["topology"], "LLLLGG")
            self.assertEqual(
                tuple(payload["architecture"]["layer_contexts"]),
                STAGE_37.DEFAULT_LAYER_CONTEXTS,
            )
            tiny_masks = STAGE_37.build_layer_attention_masks(config.block_size)
            tiny_audit = STAGE_37._STAGE_36.audit_layerwise_reachability(tiny_masks)
            self.assertEqual(
                tuple(payload["architecture"]["logical_edges_by_layer_per_head"]),
                tuple(int(mask.sum()) for mask in tiny_masks),
            )
            self.assertEqual(
                payload["architecture"]["pre_global_effective_receptive_field"],
                tiny_audit.widths_by_layer[3],
            )
            self.assertEqual(payload["training_config"]["targets_per_update"], 48)
            self.assertEqual(
                payload["training_config"]["total_target_presentations"],
                144,
            )
            self.assertEqual(
                payload["historical_control"]["checkpoint_sha256"],
                STAGE_37.HISTORICAL_CONTROL_CHECKPOINT_SHA256,
            )

            tampered = dict(payload)
            tampered["historical_control"] = dict(payload["historical_control"])
            tampered["historical_control"]["selected_validation_loss"] = 0.0
            tampered_path = Path(raw_directory) / "tampered.pt"
            torch.save(tampered, tampered_path)
            with self.assertRaisesRegex(ValueError, "historical-control"):
                STAGE_37.validate_stage_37_checkpoint(
                    tampered_path,
                    data,
                    config,
                    2,
                    6,
                    schedule,
                )

            wrong_data = make_character_data()
            wrong_train = wrong_data.train_data.clone()
            wrong_train[0] = (wrong_train[0] + 1) % wrong_data.vocabulary.size
            wrong_data = CharacterData(
                vocabulary=wrong_data.vocabulary,
                train_data=wrong_train,
                val_data=wrong_data.val_data,
                num_characters=wrong_data.num_characters,
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                STAGE_37.validate_stage_37_checkpoint(
                    checkpoint_path,
                    wrong_data,
                    config,
                    2,
                    6,
                    schedule,
                )


if __name__ == "__main__":
    unittest.main()
