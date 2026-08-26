from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("27_context_isolation.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_27_context_isolation",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_27 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_27
SPEC.loader.exec_module(STAGE_27)


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
    schedule = STAGE_27.LearningRateSchedule(
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
    block = STAGE_27.CrossHalfSpec(
        "block-4-at-8 control",
        False,
        4,
        directory / "stage27_block_best.pt",
    )
    full = STAGE_27.CrossHalfSpec(
        "full-8 Stage 26 reference",
        True,
        4,
        directory / "stage26_full.pt",
    )
    anchor = directory / "stage27_block_anchor.pt"
    return config, schedule, block, full, anchor


def create_tiny_stage_26_reference(
    data: CharacterData,
    config: TrainingConfig,
    schedule: object,
    full_path: Path,
) -> object:
    control_config, treatment_config = STAGE_27._stage_26_configs(config, 4)
    control = STAGE_27._STAGE_26.ContextSpec(
        "T=4 control",
        control_config.batch_size,
        control_config.block_size,
        full_path.with_name("stage26_control.pt"),
    )
    treatment = STAGE_27._STAGE_26.ContextSpec(
        "T=8 treatment",
        treatment_config.batch_size,
        treatment_config.block_size,
        full_path,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return STAGE_27._STAGE_26.run_experiment(
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


class Stage27MaskAndProtocolTests(unittest.TestCase):
    def test_masks_have_exact_edges_hashes_and_hard_barrier(self) -> None:
        block = STAGE_27.build_cross_half_attention_mask(
            8,
            4,
            allow_cross_half=False,
        )
        full = STAGE_27.build_cross_half_attention_mask(
            8,
            4,
            allow_cross_half=True,
        )
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(block, expected, rtol=0, atol=0)
        self.assertEqual(int(full.sum()), 36)
        self.assertEqual(int(block.sum()), 20)
        self.assertEqual(int((full & ~block).sum()), 16)
        self.assertTrue(torch.equal(full[:4], block[:4]))
        self.assertFalse(block[4:, :4].any().item())
        self.assertTrue((full & ~block)[4:, :4].all().item())
        STAGE_27.assert_hard_cross_half_barrier(block, 4)

        reachability = block.clone()
        for _ in range(5):
            reachability = (
                reachability.to(torch.int32) @ block.to(torch.int32)
            ) > 0
        self.assertFalse(reachability[4:, :4].any().item())

        canonical_block = STAGE_27.build_cross_half_attention_mask(
            128,
            64,
            allow_cross_half=False,
        )
        canonical_full = STAGE_27.build_cross_half_attention_mask(
            128,
            64,
            allow_cross_half=True,
        )
        self.assertEqual(int(canonical_block.sum()), 4_160)
        self.assertEqual(int(canonical_full.sum()), 8_256)
        self.assertEqual(int((canonical_full & ~canonical_block).sum()), 4_096)
        self.assertEqual(
            STAGE_27.tensor_sha256(canonical_block),
            STAGE_27.DEFAULT_BLOCK_MASK_SHA256,
        )
        self.assertEqual(
            STAGE_27.tensor_sha256(canonical_full),
            STAGE_27.DEFAULT_FULL_MASK_SHA256,
        )
        self.assertEqual(
            STAGE_27.tensor_sha256(canonical_block.float()),
            STAGE_27.DEFAULT_BLOCK_FLOAT_MASK_SHA256,
        )
        self.assertEqual(
            STAGE_27.tensor_sha256(canonical_full.float()),
            STAGE_27.DEFAULT_FULL_FLOAT_MASK_SHA256,
        )

        corrupt = block.clone()
        corrupt[4, 0] = True
        with self.assertRaisesRegex(RuntimeError, "cross-half attention edge"):
            STAGE_27.assert_hard_cross_half_barrier(corrupt, 4)
        with self.assertRaisesRegex(ValueError, "twice boundary"):
            STAGE_27.build_cross_half_attention_mask(
                8,
                3,
                allow_cross_half=False,
            )

    def test_protocol_fixes_geometry_bands_and_distinct_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, _, block, full, anchor = make_tiny_protocol(directory)
            targets, bands = STAGE_27.validate_cross_half_protocol(
                config,
                block,
                full,
                2,
                1,
                anchor_checkpoint_path=anchor,
            )
            self.assertEqual(targets, 16)
            self.assertEqual(
                bands,
                ((0, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8)),
            )

            aliased = STAGE_27.CrossHalfSpec(
                full.name,
                True,
                4,
                block.checkpoint_path,
            )
            with self.assertRaises(ValueError):
                STAGE_27.validate_cross_half_protocol(
                    config,
                    block,
                    aliased,
                    2,
                    1,
                )
            with self.assertRaises(ValueError):
                STAGE_27.validate_cross_half_protocol(
                    config,
                    block,
                    full,
                    2,
                    1,
                    anchor_checkpoint_path=block.checkpoint_path,
                )


class Stage27ModelAndInitializationTests(unittest.TestCase):
    def test_block_model_has_no_functional_cross_half_path(self) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            config, _, block, _, _ = make_tiny_protocol(Path(directory_name))
            model = STAGE_27.build_model(data, config, block, 2, 3)

        STAGE_27.verify_model_mask(model, block)
        self.assertEqual(
            sum(name.endswith(".tril") for name in model.state_dict()),
            6,
        )
        model.eval()
        first = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]], dtype=torch.long)
        changed_prefix = torch.tensor(
            [[4, 4, 4, 4, 4, 0, 1, 2]],
            dtype=torch.long,
        )
        with torch.no_grad():
            first_logits, _ = model(first)
            changed_logits, _ = model(changed_prefix)
        torch.testing.assert_close(
            first_logits[:, 4:],
            changed_logits[:, 4:],
            rtol=0,
            atol=0,
        )

        for transformer_block in model.blocks:
            for head in transformer_block.sa.heads:
                torch.nn.init.zeros_(head.key.weight)
                torch.nn.init.zeros_(head.query.weight)
        weights = model.get_attention_weights(first)
        torch.testing.assert_close(
            weights[0, 0, 4],
            torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            rtol=0,
            atol=0,
        )

        model.blocks[0].sa.heads[0].tril[4, 0] = 1
        with self.assertRaisesRegex(RuntimeError, "incorrect attention-mask"):
            STAGE_27.verify_model_mask(model, block)

    def test_initial_states_match_parameters_and_first_half_exactly(self) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            config, _, block, full, _ = make_tiny_protocol(Path(directory_name))
            initialization = STAGE_27.create_stage_27_initial_states(
                data,
                config,
                block,
                full,
                2,
                2,
            )
            block_model = STAGE_27.build_model(data, config, block, 2, 2)
            full_model = STAGE_27.build_model(data, config, full, 2, 2)
            block_model.load_state_dict(initialization.block, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(initialization.parameter_count, 1_861)
        self.assertEqual(len(initialization.differing_state_names), 4)
        self.assertTrue(
            all(
                name.endswith(".tril")
                for name in initialization.differing_state_names
            )
        )
        parameter_names = dict(block_model.named_parameters())
        for name, block_tensor in initialization.block.items():
            full_tensor = initialization.full[name]
            self.assertNotEqual(block_tensor.data_ptr(), full_tensor.data_ptr())
            if name in parameter_names:
                torch.testing.assert_close(
                    block_tensor,
                    full_tensor,
                    rtol=0,
                    atol=0,
                )
        self.assertEqual(
            STAGE_27.tensor_mapping_sha256(
                STAGE_27.clone_parameters(block_model)
            ),
            initialization.parameter_state_sha256,
        )
        self.assertEqual(
            STAGE_27.verify_initial_first_half_equivalence(
                block_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=107,
            ),
            0.0,
        )

    def test_canonical_initialization_hashes_are_pinned(self) -> None:
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
            block = STAGE_27.CrossHalfSpec(
                "block",
                False,
                64,
                directory / "block.pt",
            )
            full = STAGE_27.CrossHalfSpec(
                "full",
                True,
                64,
                directory / "full.pt",
            )
            initialization = STAGE_27.create_stage_27_initial_states(
                data,
                config,
                block,
                full,
                4,
                6,
            )

        self.assertEqual(initialization.parameter_count, 1_220_673)
        self.assertEqual(len(initialization.differing_state_names), 24)
        self.assertEqual(
            initialization.parameter_state_sha256,
            STAGE_27.DEFAULT_PARAMETER_INITIALIZATION_SHA256,
        )
        self.assertEqual(
            initialization.block_state_sha256,
            STAGE_27.DEFAULT_BLOCK_INITIAL_STATE_SHA256,
        )
        self.assertEqual(
            initialization.full_state_sha256,
            STAGE_27.DEFAULT_FULL_INITIAL_STATE_SHA256,
        )
        self.assertEqual(
            initialization.source_common_t64_state_sha256,
            STAGE_27.DEFAULT_SOURCE_COMMON_T64_STATE_SHA256,
        )
        self.assertEqual(
            initialization.source_extra_position_rows_sha256,
            STAGE_27.DEFAULT_SOURCE_EXTRA_POSITION_ROWS_SHA256,
        )

    def test_temporary_crossed_mask_restores_state_on_success_and_error(self) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            config, _, block, full, _ = make_tiny_protocol(Path(directory_name))
            model = STAGE_27.build_model(data, config, full, 2, 2)
        original = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        original_parameter_hash = STAGE_27.tensor_mapping_sha256(
            STAGE_27.clone_parameters(model)
        )
        block_mask = STAGE_27.build_cross_half_attention_mask(
            8,
            4,
            allow_cross_half=False,
        )
        with STAGE_27.temporary_attention_mask(model, block_mask):
            STAGE_27.verify_model_mask(model, block)
        STAGE_27.verify_model_mask(model, full)

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with STAGE_27.temporary_attention_mask(model, block_mask):
                STAGE_27.verify_model_mask(model, block)
                raise RuntimeError("sentinel")
        STAGE_27.verify_model_mask(model, full)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertEqual(
            STAGE_27.tensor_mapping_sha256(STAGE_27.clone_parameters(model)),
            original_parameter_hash,
        )

        panels = STAGE_27.make_validation_panels(
            data,
            config,
            eval_iters=2,
            seed=404,
        )
        bands = STAGE_27.make_position_bands(8, 4)
        full_mask = STAGE_27.build_cross_half_attention_mask(
            8,
            4,
            allow_cross_half=True,
        )
        full_losses = STAGE_27._evaluate_on_panels(
            model,
            panels,
            torch.device("cpu"),
            bands,
            full_mask,
        )
        blocked_losses = STAGE_27._evaluate_on_panels(
            model,
            panels,
            torch.device("cpu"),
            bands,
            block_mask,
        )
        for natural, crossed in zip(
            full_losses,
            blocked_losses,
            strict=True,
        ):
            self.assertEqual(natural.first_half, crossed.first_half)
            self.assertAlmostEqual(
                natural.overall,
                (natural.first_half + natural.second_half) / 2,
                places=6,
            )
        corrupt_first_half = block_mask.clone()
        corrupt_first_half[1, 0] = False
        with self.assertRaisesRegex(RuntimeError, "first-half logits"):
            STAGE_27._evaluate_mask_pair_on_panels(
                model,
                panels,
                torch.device("cpu"),
                bands,
                full_mask,
                corrupt_first_half,
                boundary=4,
            )
        STAGE_27.verify_model_mask(model, full)


class Stage27TinyWorkflowTests(unittest.TestCase):
    def test_real_tiny_reference_checkpoint_and_four_artifact_evaluation(
        self,
    ) -> None:
        data = make_character_data()
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, schedule, block, full, anchor = make_tiny_protocol(directory)
            stage_26 = create_tiny_stage_26_reference(
                data,
                config,
                schedule,
                full.checkpoint_path,
            )
            source_bytes = full.checkpoint_path.read_bytes()
            source_hash = STAGE_27.checkpoint_sha256(full.checkpoint_path)

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
                experiment = STAGE_27.run_experiment(
                    data,
                    config,
                    block,
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
            for expected_step, (_, actual_inputs, actual_targets) in enumerate(
                updates
            ):
                expected_inputs, expected_targets = data.get_batch(
                    "train",
                    batch_size=config.batch_size,
                    block_size=config.block_size,
                    device=torch.device("cpu"),
                    generator=expected_generator,
                )
                self.assertEqual(updates[expected_step][0], expected_step)
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
                experiment.block_checkpoint_sha256,
                STAGE_27.checkpoint_sha256(block.checkpoint_path),
            )
            self.assertEqual(
                experiment.block_anchor_checkpoint_sha256,
                STAGE_27.checkpoint_sha256(anchor),
            )
            self.assertEqual(
                experiment.full.checkpoint_step,
                stage_26.treatment.best_step,
            )
            self.assertEqual(experiment.initial_first_half_max_difference, 0.0)
            self.assertTrue(block.checkpoint_path.is_file())
            self.assertTrue(anchor.is_file())

            best = STAGE_27.validate_stage_27_checkpoint(
                block.checkpoint_path,
                data,
                config,
                block,
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
            anchor_payload = STAGE_27.validate_stage_27_checkpoint(
                anchor,
                data,
                config,
                block,
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
            self.assertEqual(best["experiment"]["stage"], 27)
            self.assertTrue(
                best["experiment"]["hard_cross_half_information_barrier"]
            )
            self.assertTrue(best["experiment"]["full_reference_reused_read_only"])
            self.assertEqual(anchor_payload["step"], stage_26.treatment.best_step)

            persisted_bytes = (
                block.checkpoint_path.read_bytes(),
                anchor.read_bytes(),
                full.checkpoint_path.read_bytes(),
            )
            precise = STAGE_27.run_four_artifact_validation(
                data,
                config,
                block,
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
            for endpoint in (precise.anchor, precise.selected_best):
                self.assertTrue(endpoint.full_weight_first_half_exact)
                self.assertTrue(endpoint.block_weight_first_half_exact)
                for natural, crossed in (
                    (endpoint.full_natural, endpoint.full_blocked),
                    (endpoint.block_natural, endpoint.block_opened),
                ):
                    self.assertEqual(len(natural.batch_losses), 2)
                    for first, second in zip(
                        natural.batch_losses,
                        crossed.batch_losses,
                        strict=True,
                    ):
                        self.assertEqual(first.first_half, second.first_half)
                self.assertEqual(
                    endpoint.full_weight_mask_delta.mean_delta.first_half,
                    0.0,
                )
                self.assertEqual(
                    endpoint.full_weight_mask_delta.standard_error.first_half,
                    0.0,
                )
                self.assertEqual(
                    endpoint.block_weight_mask_delta.mean_delta.first_half,
                    0.0,
                )
                self.assertEqual(
                    endpoint.block_weight_mask_delta.standard_error.first_half,
                    0.0,
                )
            self.assertEqual(
                (
                    block.checkpoint_path.read_bytes(),
                    anchor.read_bytes(),
                    full.checkpoint_path.read_bytes(),
                ),
                persisted_bytes,
            )

            with self.assertRaisesRegex(RuntimeError, "artifact produced"):
                STAGE_27.run_four_artifact_validation(
                    data,
                    config,
                    block,
                    full,
                    2,
                    1,
                    torch.device("cpu"),
                    schedule,
                    replace(experiment, block_checkpoint_sha256="0" * 64),
                    anchor,
                    eval_iters=2,
                    seed=404,
                )

            corrupt_path = directory / "corrupt_block.pt"
            corrupt = dict(best)
            corrupt_state = dict(corrupt["model_state_dict"])
            mask_name = next(
                name for name in corrupt_state if name.endswith(".tril")
            )
            corrupt_mask = corrupt_state[mask_name].clone()
            corrupt_mask[4, 0] = 1
            corrupt_state[mask_name] = corrupt_mask
            corrupt["model_state_dict"] = corrupt_state
            torch.save(corrupt, corrupt_path)
            with self.assertRaisesRegex(RuntimeError, "incorrect attention-mask"):
                STAGE_27.validate_stage_27_checkpoint(
                    corrupt_path,
                    data,
                    config,
                    block,
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
