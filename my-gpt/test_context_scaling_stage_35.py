from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("35_context_scaling.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_35_context_scaling_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_35 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_35
SPEC.loader.exec_module(STAGE_35)


def make_character_data(
    *,
    vocabulary_size: int = 5,
    train_tokens: int = 640,
    val_tokens: int = 384,
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


def make_tiny_protocol(directory: Path) -> tuple[object, ...]:
    schedule = STAGE_35.LearningRateSchedule(
        initial_learning_rate=2e-2,
        first_decay_step=1,
        middle_learning_rate=7e-3,
        second_decay_step=2,
        final_learning_rate=2e-3,
    )
    common = {
        "n_embd": 8,
        "learning_rate": schedule.initial_learning_rate,
        "max_iters": 3,
        "eval_interval": 1,
        "eval_iters": 1,
        "seed": 101,
    }
    control_config = TrainingConfig(batch_size=2, block_size=24, **common)
    treatment_config = TrainingConfig(batch_size=1, block_size=48, **common)
    control = STAGE_35.ContextSpec(
        "tiny T=24 control",
        2,
        24,
        directory / "stage35-control.pt",
    )
    treatment = STAGE_35.ContextSpec(
        "tiny T=48 treatment",
        1,
        48,
        directory / "stage35-treatment.pt",
    )
    return control_config, treatment_config, control, treatment, schedule


class Stage35ContextScalingTests(unittest.TestCase):
    def test_defaults_topology_reachability_and_cli(self) -> None:
        self.assertEqual(STAGE_35.EXPERIMENT_STAGE, 35)
        self.assertEqual(
            (
                STAGE_35.DEFAULT_CONTROL_BATCH_SIZE,
                STAGE_35.DEFAULT_CONTROL_BLOCK_SIZE,
                STAGE_35.DEFAULT_TREATMENT_BATCH_SIZE,
                STAGE_35.DEFAULT_TREATMENT_BLOCK_SIZE,
            ),
            (16, 128, 8, 256),
        )
        self.assertEqual(STAGE_35.DEFAULT_TARGETS_PER_UPDATE, 2_048)
        self.assertEqual(STAGE_35.DEFAULT_LAYER_CONTEXTS, (23, 23, 23, 23, None, None))
        self.assertEqual(
            (
                STAGE_35.DEFAULT_N_EMBD,
                STAGE_35.DEFAULT_N_HEAD,
                STAGE_35.DEFAULT_HEAD_SIZE,
                STAGE_35.DEFAULT_FEED_FORWARD_WIDTH,
                STAGE_35.DEFAULT_N_LAYER,
                STAGE_35.DEFAULT_DROPOUT,
            ),
            (128, 4, 32, 512, 6, 0.10),
        )
        self.assertNotEqual(
            STAGE_35.DEFAULT_CONTROL_CHECKPOINT_PATH,
            STAGE_35._STAGE_32.DEFAULT_LAST_TWO_GLOBAL_CHECKPOINT_PATH,
        )

        masks_128 = STAGE_35.build_layer_attention_masks(128)
        masks_256 = STAGE_35.build_layer_attention_masks(256)
        self.assertEqual(
            tuple(int(mask.sum()) for mask in masks_128),
            (2_691, 2_691, 2_691, 2_691, 8_256, 8_256),
        )
        self.assertEqual(
            tuple(int(mask.sum()) for mask in masks_256),
            (5_635, 5_635, 5_635, 5_635, 32_896, 32_896),
        )
        for short, long in zip(masks_128, masks_256, strict=True):
            self.assertTrue(torch.equal(short, long[:128, :128]))
        audit_128 = STAGE_35.audit_layerwise_reachability(masks_128)
        audit_256 = STAGE_35.audit_layerwise_reachability(masks_256)
        self.assertEqual(audit_128.widths_by_layer[:4], (23, 45, 67, 89))
        self.assertEqual(audit_256.widths_by_layer[:4], (23, 45, 67, 89))
        self.assertEqual(audit_128.first_global_reachability_layer, 5)
        self.assertEqual(audit_256.first_global_reachability_layer, 5)
        self.assertTrue(audit_128.all_causal_sources_reachable)
        self.assertTrue(audit_256.all_causal_sources_reachable)

        with patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_35.parse_args()
        self.assertEqual(
            (
                args.control_batch_size,
                args.control_block_size,
                args.treatment_batch_size,
                args.treatment_block_size,
                args.precise_eval_seed,
            ),
            (16, 128, 8, 256, 1_352),
        )

    def test_long_segments_preserve_targets_and_band_alignment(self) -> None:
        source = torch.arange(900, dtype=torch.long) % 7
        paired = STAGE_35.build_paired_context_batch(
            source,
            torch.tensor([3, 400]),
            control_block_size=128,
            treatment_block_size=256,
        )
        self.assertEqual(tuple(paired.control_inputs.shape), (4, 128))
        self.assertEqual(tuple(paired.treatment_inputs.shape), (2, 256))
        self.assertTrue(
            torch.equal(
                paired.control_targets.reshape(-1),
                paired.treatment_targets.reshape(-1),
            )
        )
        control = STAGE_35.ContextSpec("control", 4, 128, Path("control.pt"))
        treatment = STAGE_35.ContextSpec("treatment", 2, 256, Path("treatment.pt"))
        self.assertEqual(
            STAGE_35.make_position_bands(control, treatment),
            STAGE_35.DEFAULT_POSITION_BANDS,
        )

        generator = torch.Generator().manual_seed(77)
        long_logits = torch.randn(2, 256, 7, generator=generator)
        short_logits = long_logits.reshape(4, 128, 7)
        long_losses = STAGE_35.position_band_cross_entropy(
            long_logits,
            paired.treatment_targets,
            treatment,
            control,
            treatment,
        )
        short_losses = STAGE_35.position_band_cross_entropy(
            short_logits,
            paired.control_targets,
            control,
            control,
            treatment,
        )
        self.assertEqual(len(long_losses), 6)
        for long_loss, short_loss in zip(long_losses, short_losses, strict=True):
            self.assertTrue(torch.equal(long_loss, short_loss))

    def test_overlap_initialization_masks_and_first_half_logits(self) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            control_config, treatment_config, control, treatment, _ = (
                make_tiny_protocol(directory)
            )
            initialization = STAGE_35.create_paired_initial_states(
                data,
                control_config,
                treatment_config,
                2,
                6,
            )
            self.assertEqual(
                initialization.treatment_parameter_count
                - initialization.control_parameter_count,
                (treatment.block_size - control.block_size) * control_config.n_embd,
            )
            self.assertEqual(
                STAGE_35.tensor_mapping_sha256(
                    STAGE_35._CORE._project_treatment_state_to_control(
                        initialization.treatment,
                        initialization.control,
                    )
                ),
                initialization.common_state_sha256,
            )

            control_model = STAGE_35.build_model(data, control_config, 2, 6)
            treatment_model = STAGE_35.build_model(data, treatment_config, 2, 6)
            control_model.load_state_dict(initialization.control, strict=True)
            treatment_model.load_state_dict(initialization.treatment, strict=True)
            STAGE_35.verify_model_masks(control_model)
            STAGE_35.verify_model_masks(treatment_model)
            control_parameters = dict(control_model.named_parameters())
            treatment_parameters = dict(treatment_model.named_parameters())
            self.assertEqual(control_parameters.keys(), treatment_parameters.keys())
            for name, control_parameter in control_parameters.items():
                treatment_parameter = treatment_parameters[name]
                if name == "position_embedding_table.weight":
                    self.assertTrue(
                        torch.equal(
                            control_parameter,
                            treatment_parameter[: control.block_size],
                        )
                    )
                else:
                    self.assertTrue(torch.equal(control_parameter, treatment_parameter))
            extra_rows = treatment_parameters[
                "position_embedding_table.weight"
            ][control.block_size :]
            self.assertEqual(
                STAGE_35.tensor_sha256(extra_rows),
                initialization.extra_position_rows_sha256,
            )
            difference = STAGE_35.verify_initial_first_128_equivalence(
                control_model,
                treatment_model,
                data,
                control,
                treatment,
                torch.device("cpu"),
                seed=107,
            )
            self.assertLessEqual(difference, 1e-6)

            treatment_model.blocks[0].sa.heads[0].tril[23, 0] = 1
            with self.assertRaisesRegex(RuntimeError, "incorrect attention-mask"):
                STAGE_35.verify_model_masks(treatment_model)

    def test_tiny_training_checkpoint_and_precise_band_protocol(self) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            control_config, treatment_config, control, treatment, schedule = (
                make_tiny_protocol(directory)
            )
            with contextlib.redirect_stdout(io.StringIO()):
                report = STAGE_35.run_experiment(
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
                    training_batch_seed=102,
                    training_rng_seed=104,
                    benchmark=None,
                )
            self.assertEqual(report.targets_per_update, 48)
            self.assertEqual(
                report.control.final_batch_generator_sha256,
                report.treatment.final_batch_generator_sha256,
            )
            self.assertEqual(
                tuple(record.step for record in report.control.training.history),
                (0, 1, 2, 3),
            )
            for spec, branch in (
                (control, report.control),
                (treatment, report.treatment),
            ):
                payload = torch.load(
                    spec.checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(payload["experiment"]["stage"], 35)
                self.assertEqual(payload["architecture"]["topology"], "LLLLGG")
                for metadata_name in ("initialization", "experiment"):
                    metadata = payload[metadata_name]
                    self.assertEqual(
                        metadata["common_control_projection_sha256"],
                        report.initialization.common_state_sha256,
                    )
                    self.assertEqual(
                        metadata["common_control_projection_block_size"],
                        control.block_size,
                    )
                    self.assertTrue(
                        metadata["common_t64_projection_sha256_is_legacy_alias"]
                    )
                self.assertEqual(
                    tuple(payload["architecture"]["layer_contexts"]),
                    STAGE_35.DEFAULT_LAYER_CONTEXTS,
                )
                self.assertEqual(
                    tuple(tuple(value) for value in payload["evaluation_protocol"]["position_bands"]),
                    STAGE_35.make_position_bands(control, treatment),
                )
                STAGE_35.validate_stage_35_checkpoint(
                    spec.checkpoint_path,
                    data,
                    control_config if spec is control else treatment_config,
                    spec,
                    control,
                    treatment,
                    2,
                    6,
                    schedule,
                    initialization_seed=101,
                    branch_initial_state_sha256=branch.initial_state_sha256,
                    common_state_sha256=report.initialization.common_state_sha256,
                    extra_position_rows_sha256=(
                        report.initialization.extra_position_rows_sha256
                    ),
                    training_batch_seed=102,
                    training_rng_seed=104,
                    shared_parameter_initialization_sha256=(
                        report.initialization.shared_parameter_initialization_sha256
                    ),
                )

            cpu_rng_before = torch.random.get_rng_state().clone()
            with contextlib.redirect_stdout(io.StringIO()):
                precise = STAGE_35.run_precise_validation(
                    data,
                    control_config,
                    treatment_config,
                    control,
                    treatment,
                    2,
                    6,
                    torch.device("cpu"),
                    schedule,
                    report.initialization,
                    initialization_seed=101,
                    training_batch_seed=102,
                    training_rng_seed=104,
                    eval_iters=2,
                    seed=109,
                    expected_control_checkpoint_sha256=(
                        report.control_checkpoint_sha256
                    ),
                    expected_treatment_checkpoint_sha256=(
                        report.treatment_checkpoint_sha256
                    ),
                )
            self.assertTrue(torch.equal(cpu_rng_before, torch.random.get_rng_state()))
            self.assertEqual(
                precise.position_bands,
                ((0, 24), (24, 48), (24, 30), (30, 36), (36, 42), (42, 48)),
            )
            self.assertEqual(len(precise.band_results), 2)
            self.assertEqual(len(precise.band_delta.mean_delta.values), 6)
            self.assertEqual(precise.band_delta.candidate, treatment.name)
            self.assertEqual(precise.band_delta.baseline, control.name)


if __name__ == "__main__":
    unittest.main()
