from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from config import TrainingConfig
from data_utils import CharacterData, CharacterVocabulary


MODULE_PATH = Path(__file__).with_name("19_receptive_field.py")
SPEC = importlib.util.spec_from_file_location(
    "stage_19_receptive_field",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
STAGE_19 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE_19
SPEC.loader.exec_module(STAGE_19)


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


def make_specs(
    directory: Path,
    *,
    block_size: int = 8,
    local_context: int = 3,
) -> tuple[STAGE_19.AttentionSpec, STAGE_19.AttentionSpec]:
    return (
        STAGE_19.AttentionSpec(
            f"local-{local_context}",
            local_context,
            directory / "local.pt",
        ),
        STAGE_19.AttentionSpec(
            f"full-{block_size}",
            None,
            directory / "full.pt",
        ),
    )


class Stage19ReceptiveFieldTests(unittest.TestCase):
    def test_defaults_define_the_intended_effective_context_comparison(self) -> None:
        with mock.patch.object(sys, "argv", [str(MODULE_PATH)]):
            args = STAGE_19.parse_args()

        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.block_size, 128)
        self.assertEqual(args.local_context, 17)
        self.assertEqual(args.n_embd, 64)
        self.assertEqual(args.n_head, 4)
        self.assertEqual(args.n_layer, 4)
        self.assertEqual(args.max_iters, 18_000)
        self.assertEqual(args.precise_eval_iters, 500)
        self.assertEqual(
            args.local_checkpoint_path.name,
            "stage_19_local_17_best_checkpoint.pt",
        )
        self.assertEqual(
            args.full_checkpoint_path.name,
            "stage_19_full_128_best_checkpoint.pt",
        )

        self.assertEqual(
            STAGE_19.receptive_field_widths_by_layer(4, 17),
            (17, 33, 49, 65),
        )
        self.assertEqual(
            STAGE_19.local_span_for_half_receptive_field(128, 4),
            17,
        )

        with tempfile.TemporaryDirectory() as directory_name:
            local, full = make_specs(
                Path(directory_name),
                block_size=128,
                local_context=17,
            )
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
            targets, bands = STAGE_19.validate_attention_protocol(
                config,
                local,
                full,
                n_layer=4,
            )
            model = STAGE_19._build_model(
                make_character_data(vocabulary_size=65),
                config,
                4,
                4,
                local,
            )

        self.assertEqual(targets, 2_048)
        self.assertEqual(targets * args.max_iters, 36_864_000)
        self.assertEqual(
            bands,
            ((0, 32), (32, 64), (64, 96), (96, 128)),
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            215_873,
        )
        self.assertEqual(
            STAGE_19.effective_receptive_field_width(128, 4, local),
            65,
        )
        self.assertEqual(
            STAGE_19.effective_receptive_field_width(128, 4, full),
            128,
        )

    def test_diagnostic_and_mask_reachability_are_exact(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            STAGE_19.print_receptive_field_diagnostic(4, 17)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "after block 1: max receptive field = 17",
                "after block 2: max receptive field = 33",
                "after block 3: max receptive field = 49",
                "after block 4: max receptive field = 65",
            ],
        )

        mask = STAGE_19.build_attention_mask(128, 17)
        self.assertTrue(mask[16, 0].item())
        self.assertTrue(mask[16, 16].item())
        self.assertFalse(mask[16, 17].item())
        self.assertFalse(mask[17, 0].item())
        self.assertTrue(mask[17, 1].item())
        self.assertTrue(mask[17, 17].item())
        self.assertFalse(mask[127, 110].item())
        self.assertTrue(mask[127, 111].item())

        reachability = mask.clone()
        for _ in range(3):
            reachability = (reachability.to(torch.int32) @ mask.to(torch.int32)) > 0
        self.assertEqual(
            torch.nonzero(reachability[127], as_tuple=False).flatten().tolist(),
            list(range(63, 128)),
        )
        self.assertTrue(reachability[64, 0].item())
        self.assertFalse(reachability[65, 0].item())
        self.assertTrue(reachability[65, 1].item())

    def test_protocol_rejects_the_stage_18_geometry(self) -> None:
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
            STAGE_19.validate_attention_protocol(
                config,
                local,
                full,
                n_layer=2,
            )

            for invalid_window in (2, 4, 8):
                with self.subTest(invalid_window=invalid_window):
                    invalid, _ = make_specs(
                        directory,
                        local_context=invalid_window,
                    )
                    with self.assertRaises(ValueError):
                        STAGE_19.validate_attention_protocol(
                            config,
                            invalid,
                            full,
                            n_layer=2,
                        )
            with self.assertRaises(ValueError):
                STAGE_19.validate_attention_protocol(config, local, full)

    def test_initial_audit_uses_only_the_identical_query_prefix(self) -> None:
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
            initialization = STAGE_19.create_paired_initial_states(
                data,
                config,
                2,
                2,
                local,
                full,
            )
            local_model = STAGE_19._build_model(data, config, 2, 2, local)
            full_model = STAGE_19._build_model(data, config, 2, 2, full)
            local_model.load_state_dict(initialization.local, strict=True)
            full_model.load_state_dict(initialization.full, strict=True)

        self.assertEqual(
            STAGE_19.verify_initial_shared_prefix_equivalence(
                local_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=29,
                prefix_length=3,
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "shared-prefix logits differ"):
            STAGE_19.verify_initial_shared_prefix_equivalence(
                local_model,
                full_model,
                data,
                config,
                torch.device("cpu"),
                seed=29,
                prefix_length=4,
            )

    def test_tiny_workflow_writes_stage_19_checkpoint_metadata(self) -> None:
        data = make_character_data()
        config = TrainingConfig(
            batch_size=2,
            block_size=8,
            n_embd=8,
            learning_rate=2e-2,
            max_iters=3,
            eval_interval=3,
            eval_iters=1,
            seed=101,
        )
        schedule = STAGE_19.LearningRateSchedule(
            initial_learning_rate=2e-2,
            first_decay_step=1,
            middle_learning_rate=7e-3,
            second_decay_step=2,
            final_learning_rate=2e-3,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            local, full = make_specs(Path(directory_name))
            with contextlib.redirect_stdout(io.StringIO()):
                report = STAGE_19.run_experiment(
                    data,
                    config,
                    local,
                    full,
                    2,
                    2,
                    torch.device("cpu"),
                    0,
                    schedule,
                    training_batch_seed=202,
                    training_rng_seed=303,
                    benchmark=None,
                )

            self.assertEqual(report.initial_shared_prefix_length, 3)
            self.assertEqual(report.initial_shared_prefix_max_difference, 0.0)
            self.assertEqual(
                report.local.final_batch_generator_sha256,
                report.full.final_batch_generator_sha256,
            )
            for spec, initial_hash in (
                (local, report.initialization.local_state_sha256),
                (full, report.initialization.full_state_sha256),
            ):
                checkpoint = STAGE_19.validate_stage_19_checkpoint(
                    spec.checkpoint_path,
                    data,
                    config,
                    spec,
                    local,
                    full,
                    2,
                    2,
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
                self.assertEqual(experiment["stage"], 19)
                self.assertEqual(
                    experiment["comparison_variable"],
                    "maximum_attention_span",
                )
                self.assertEqual(
                    experiment["effective_receptive_field_width"],
                    5 if spec.max_context is not None else 8,
                )


if __name__ == "__main__":
    unittest.main()
