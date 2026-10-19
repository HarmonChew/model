from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

import artifact_fixtures
from data_utils import CharacterData, CharacterVocabulary
from training import LossEstimate


SPEC = importlib.util.spec_from_file_location(
    "stage_50_under_test", Path(__file__).with_name("50_depth_revisit.py")
)
assert SPEC is not None and SPEC.loader is not None
STAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE
SPEC.loader.exec_module(STAGE)


class Stage50Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, width: int = 16, **overrides):
        vocabulary = CharacterVocabulary.from_text("abcde")
        tokens = torch.arange(2400) % vocabulary.size
        data = CharacterData(vocabulary, tokens[:1800], tokens[1800:], 2400)
        settings = dict(
            batch_size=2,
            block_size=24,
            n_embd=width,
            max_iters=3,
            eval_iters=1,
            eval_interval=1,
        )
        settings.update(overrides)
        return data, replace(STAGE.canonical_config(), **settings), torch.device("cpu")

    def spec(
        self,
        directory: Path,
        key: str,
        *,
        width: int,
        depth: int,
        contexts=None,
        dropout: float = STAGE.DEFAULT_DROPOUT,
    ):
        return STAGE.BranchSpec(
            key,
            width,
            depth,
            dropout,
            STAGE.layer_contexts_for_depth(depth) if contexts is None else contexts,
            directory / f"{key}_best.pt",
            directory / f"{key}_final.pt",
        )

    def run_pair(self, directory: Path, width: int = 16):
        data, config, device = self.fixture(width)
        control_spec = self.spec(
            directory, "control", width=config.n_embd, depth=STAGE.DEFAULT_CONTROL_DEPTH
        )
        treatment_spec = self.spec(
            directory,
            "treatment",
            width=config.n_embd,
            depth=STAGE.DEFAULT_TREATMENT_DEPTH,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            control = STAGE.run_branch(control_spec, data, config, device)
            treatment = STAGE.run_branch(treatment_spec, data, config, device)
        return control, treatment, data, config, device

    def peers(
        self,
        spec,
        data,
        config,
        *,
        foreign_device: str | None = None,
        with_stored_l6: bool = False,
    ):
        """Stored-peer loaders over one trained fixture checkpoint."""

        payload = STAGE.validate_branch_checkpoint(
            spec.final_path, spec, data, config, "final"
        )
        if foreign_device is not None:
            payload = {
                **payload,
                "runtime": {**payload["runtime"], "device_name": foreign_device},
            }

        def loader(key: str):
            def load(data: CharacterData, directory: Path):
                return STAGE.StoredEndpoint(
                    key,
                    spec.width,
                    STAGE.DEFAULT_CONTROL_DEPTH,
                    STAGE.DEFAULT_DROPOUT,
                    spec.final_path,
                    payload,
                    config,
                )

            return load

        loaders = [loader(STAGE.BUDGET_PEER_KEY)]
        if with_stored_l6:
            loaders.append(loader(STAGE.STORED_L6_PEER_KEY))
        return tuple(loaders)

    def panel(
        self,
        directory: Path,
        seed: int = 1369,
        eval_iters: int = 5,
        *,
        with_stored_l6: bool = False,
        foreign_peer_device: str | None = None,
    ):
        """A budget panel built from a fresh fixture pair plus stored peers."""

        control, treatment, data, config, device = self.run_pair(directory)
        report = STAGE.evaluate_budget_shootout(
            data,
            config,
            device,
            directory,
            specs=(control.spec, treatment.spec),
            peer_loaders=self.peers(
                control.spec,
                data,
                config,
                foreign_device=foreign_peer_device,
                with_stored_l6=with_stored_l6,
            ),
            expected_generator_sha256=None,
            seed=seed,
            eval_iters=eval_iters,
        )
        return report, control, treatment, data, config, device

    def test_protocol_locks_every_setting_except_depth(self):
        config = STAGE.canonical_config()
        self.assertEqual(asdict(config), asdict(STAGE.canonical_config(STAGE.DEFAULT_TREATMENT_DEPTH)))
        self.assertEqual((config.batch_size, config.block_size), (8, 256))
        self.assertEqual((config.n_embd, STAGE._STAGE_44.N_HEAD), (256, 4))
        self.assertEqual((config.max_iters, config.learning_rate), (50000, 1e-3))
        self.assertEqual((config.eval_interval, config.eval_iters), (500, 100))
        self.assertEqual(
            config.batch_size * config.block_size * config.max_iters, 102_400_000
        )
        self.assertEqual(
            (STAGE.DEFAULT_WIDTH, STAGE.DEFAULT_DROPOUT), (256, 0.15)
        )
        self.assertEqual(
            (STAGE.DEFAULT_CONTROL_DEPTH, STAGE.DEFAULT_TREATMENT_DEPTH), (6, 8)
        )
        # The topologies are the established ones: every local layer W23, and the
        # same two consecutive Full layers last at both depths.
        control = STAGE.layer_contexts_for_depth(6)
        treatment = STAGE.layer_contexts_for_depth(8)
        self.assertEqual(control, (23, 23, 23, 23, None, None))
        self.assertEqual(treatment, (23, 23, 23, 23, 23, 23, None, None))
        self.assertEqual((control[-2:], treatment[-2:]), ((None, None), (None, None)))
        self.assertEqual(treatment[:6], (23,) * 6)
        self.assertEqual(STAGE.topology_label(control), "LLLLGG")
        self.assertEqual(STAGE.topology_label(treatment), "LLLLLLGG")
        # The treatment is the control plus two local layers, not a re-topology.
        self.assertEqual(control[:4], treatment[:4])
        for depth in (4, 5, 7, 9, 0, True, "6"):
            with self.assertRaises(ValueError):
                STAGE.layer_contexts_for_depth(depth)
        schedule = STAGE.make_schedule()
        for step, expected in (
            (0, 1e-3),
            (9999, 1e-3),
            (10000, 3e-4),
            (13000, 1e-4),
            (17999, 1e-4),
            (18000, 3e-5),
            (49999, 3e-5),
        ):
            self.assertEqual(schedule.for_update(step), expected)

    def test_architectures_differ_only_in_depth(self):
        data, config, _ = self.fixture()
        canonical = replace(config, n_embd=256, block_size=256, batch_size=8)
        specs = (
            self.spec(
                Path("/tmp"),
                "control",
                width=256,
                depth=STAGE.DEFAULT_CONTROL_DEPTH,
            ),
            self.spec(
                Path("/tmp"),
                "treatment",
                width=256,
                depth=STAGE.DEFAULT_TREATMENT_DEPTH,
            ),
        )
        metadata = [
            STAGE.depth_architecture_metadata(data, canonical, spec) for spec in specs
        ]
        shared = set(STAGE.SHARED_ARCHITECTURE_FIELDS)
        for field in shared - {"n_layer", "layer_contexts", "topology"}:
            self.assertEqual(
                metadata[0][field], metadata[1][field], field
            )
        self.assertEqual(
            (
                metadata[0]["n_layer"],
                metadata[0]["topology"],
                metadata[0]["layer_contexts"],
            ),
            (6, "LLLLGG", (23, 23, 23, 23, None, None)),
        )
        self.assertEqual(
            (
                metadata[1]["n_layer"],
                metadata[1]["topology"],
                metadata[1]["layer_contexts"],
            ),
            (8, "LLLLLLGG", (23, 23, 23, 23, 23, 23, None, None)),
        )
        for entry, local in zip(metadata, (4, 6), strict=True):
            self.assertEqual(
                (entry["n_embd"], entry["n_head"], entry["head_size"]), (256, 4, 64)
            )
            self.assertEqual(
                (entry["feed_forward_width"], entry["block_size"]), (1024, 256)
            )
            self.assertEqual(entry["residual_dropout"], 0.15)
            self.assertEqual(entry["local_layer_count"], local)
            self.assertEqual(entry["global_layer_count"], 2)
            self.assertEqual(entry["global_layer_indices"], (local, local + 1))
            self.assertEqual(entry["first_global_reachability_layer"], local + 1)
            # W23 stacked over local layers, then full coverage at the first Full.
            self.assertEqual(
                entry["receptive_field_by_layer"],
                tuple(
                    min(256, 1 + layer * 22) for layer in range(1, local + 1)
                )
                + (256, 256),
            )
        # The compositional local receptive field before global refinement, the
        # one quantity depth moves as a side effect.
        self.assertEqual(metadata[0]["pre_global_receptive_field"], 89)
        self.assertEqual(metadata[1]["pre_global_receptive_field"], 133)
        self.assertEqual(
            (STAGE.DEFAULT_BLOCK_PARAMETER_COUNT, STAGE.DEFAULT_C320_PARAMETER_COUNT),
            (788_992, 7_537_378),
        )
        self.assertEqual(
            STAGE.DEFAULT_TREATMENT_PARAMETER_COUNT - STAGE.DEFAULT_CONTROL_PARAMETER_COUNT,
            2 * STAGE.DEFAULT_BLOCK_PARAMETER_COUNT,
        )

    def test_recorded_c256_initialization_fingerprints_hold(self):
        """The pinned native hashes are re-derived from the real corpus and width."""

        data = CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        config = STAGE.canonical_config()
        specs = STAGE.branch_specs(Path("/tmp"))
        STAGE.validate_branch_specs(specs)
        recorded = (
            (6, 4_850_274, "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e"),
            (8, 6_428_258, "6bef5bb6fa54e908ca06f64d299b1c3b35db7cf8676a8ade7d33c2b9cdb2f8c3"),
        )
        for spec, (depth, parameters, digest) in zip(specs, recorded, strict=True):
            _, _, _, initialization = STAGE.initialize_branch(
                data, config, spec, torch.device("cpu")
            )
            self.assertEqual(spec.depth, depth)
            self.assertEqual(initialization["parameter_count"], parameters)
            self.assertEqual(initialization["block_parameter_count"], 788_992)
            self.assertEqual(initialization["state_sha256"], digest)
            # Two independent factories must agree with the recorded constant.
            self.assertEqual(
                STAGE.verify_native_initialization(data, config, spec, initialization),
                digest,
            )
            self.assertEqual(STAGE.locked_initialization_fingerprint(spec), (parameters, digest))
        # The cross-stage constant the control must also match.
        self.assertEqual(
            STAGE.C256_L6_NATIVE_INITIALIZATION_SHA256,
            "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        )
        self.assertEqual(
            STAGE.C256_L6_NATIVE_INITIALIZATION_SHA256,
            STAGE._STAGE_49.C256_NATIVE_INITIALIZATION_SHA256,
        )

    def fixture_stored_peers(self, directory: Path):
        """Reduced stand-ins for the two stored endpoints the preflight reads.

        The preflight itself runs at the canonical width, so its initialization
        audit is untouched. Only the two read-only peers are substituted, and
        they are real endpoints written by their owning stages.
        """

        stage_49 = STAGE._STAGE_49
        stage_46 = STAGE._STAGE_46
        control_data, control_config, control_spec, _ = artifact_fixtures.build_branch(
            stage_49, directory, "control", stage_49.DEFAULT_CONTROL_DROPOUT
        )
        control_payload = stage_49.validate_branch_checkpoint(
            control_spec.final_path, control_spec, control_data, control_config, "final"
        )
        budget_data, budget_config, budget_spec, _ = artifact_fixtures.build_branch(
            stage_46, directory, "treatment", stage_46.DEFAULT_TREATMENT_DROPOUT
        )
        budget_payload = stage_46.validate_branch_checkpoint(
            budget_spec.final_path, budget_spec, budget_data, budget_config, "final"
        )

        def control_loader(data: CharacterData, unused: Path):
            del unused
            return STAGE.StoredEndpoint(
                STAGE.STORED_L6_PEER_KEY,
                control_spec.width,
                STAGE.DEFAULT_CONTROL_DEPTH,
                control_spec.dropout,
                control_spec.final_path,
                control_payload,
                control_config,
            )

        def budget_loader(data: CharacterData, unused: Path):
            del unused
            return STAGE.StoredEndpoint(
                STAGE.BUDGET_PEER_KEY,
                budget_spec.width,
                STAGE._STAGE_44.N_LAYER,
                budget_spec.dropout,
                budget_spec.final_path,
                budget_payload,
                budget_config,
            )

        return control_loader, budget_loader

    def test_preflight_authenticates_initializations_and_stored_peers(self):
        """The canonical preflight gates the two stored endpoints before training."""

        data = STAGE.CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        config = STAGE.canonical_config()
        specs = STAGE.branch_specs(STAGE.DEFAULT_OUTPUT_DIRECTORY)
        with tempfile.TemporaryDirectory() as raw:
            control_loader, budget_loader = self.fixture_stored_peers(Path(raw))
            with (
                patch.object(STAGE, "load_stage_49_control", new=control_loader),
                patch.object(STAGE, "load_c320_budget_peer", new=budget_loader),
            ):
                audit = STAGE.preflight(
                    data, (config, config), specs, torch.device("cpu")
                )
        self.assertFalse(audit["training_executed"])
        self.assertEqual(audit["protocol"], "paired_architecture_native_depth")
        self.assertEqual(
            (
                audit["parameter_counts"]["control"],
                audit["parameter_counts"]["treatment"],
                audit["parameter_counts"]["added"],
            ),
            (4_850_274, 6_428_258, 2 * STAGE.DEFAULT_BLOCK_PARAMETER_COUNT),
        )
        self.assertEqual(
            audit["control_initialization_equivalence"]["state_sha256"],
            STAGE.C256_L6_NATIVE_INITIALIZATION_SHA256,
        )
        self.assertEqual(
            audit["treatment_initialization_equivalence"]["state_sha256"],
            STAGE.C256_L8_NATIVE_INITIALIZATION_SHA256,
        )
        shared = audit["shared_initialization"]
        self.assertTrue(shared["treatment_is_a_key_superset"])
        self.assertEqual(
            (
                shared["control_key_count"],
                shared["treatment_key_count"],
                shared["added_key_count"],
                shared["identical_key_count"],
                shared["added_parameter_count"],
                shared["per_added_block_parameter_count"],
            ),
            (162, 214, 52, 152, 1_577_984, 788_992),
        )
        self.assertEqual(
            [branch["depth"] for branch in audit["branches"]], [6, 8]
        )
        self.assertEqual(
            [
                branch["architecture"]["pre_global_receptive_field"]
                for branch in audit["branches"]
            ],
            [89, 133],
        )
        self.assertEqual(audit["updates_per_branch"], 50_000)
        self.assertEqual(audit["target_presentations_per_branch"], 102_400_000)
        peers = audit["stored_peers"]
        control_peer = peers["stage_49_control"]
        self.assertEqual(
            (
                control_peer["key"],
                control_peer["depth"],
                control_peer["dropout"],
            ),
            ("c256_l6_lineage", 6, 0.15),
        )
        budget_peer = peers["c320_budget_peer"]
        self.assertEqual(
            (
                budget_peer["key"],
                budget_peer["depth"],
                budget_peer["dropout"],
            ),
            ("c320_l6", 6, 0.15),
        )

    def test_canonical_preflight_pins_the_published_initializations(self):
        """The published pair and peers, checked only where the artifacts exist."""

        directory = STAGE.DEFAULT_OUTPUT_DIRECTORY
        stage_46 = STAGE._STAGE_46
        stage_49 = STAGE._STAGE_49
        stage_49_specs = stage_49.branch_specs(directory)
        stage_49_control = next(
            spec for spec in stage_49_specs if spec.key == "control"
        )
        c320_spec = next(
            spec
            for spec in stage_46.branch_specs(directory)
            if spec.final_path.name == STAGE.C320_BUDGET_REF["filename"]
        )
        if not (
            stage_49_control.final_path.is_file()
            and stage_49_control.best_path.is_file()
            and c320_spec.final_path.is_file()
        ):
            self.skipTest("ignored canonical Stage 49/C320 artifacts are unavailable")
        data = STAGE.CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        config = STAGE.canonical_config()
        specs = STAGE.branch_specs(directory)
        audit = STAGE.preflight(data, (config, config), specs, torch.device("cpu"))
        self.assertFalse(audit["training_executed"])
        self.assertEqual(
            (
                audit["parameter_counts"]["control"],
                audit["parameter_counts"]["treatment"],
                audit["parameter_counts"]["added"],
            ),
            (4_850_274, 6_428_258, 2 * STAGE.DEFAULT_BLOCK_PARAMETER_COUNT),
        )
        self.assertEqual(
            audit["control_initialization_equivalence"]["state_sha256"],
            STAGE.C256_L6_NATIVE_INITIALIZATION_SHA256,
        )
        self.assertEqual(
            audit["treatment_initialization_equivalence"]["state_sha256"],
            STAGE.C256_L8_NATIVE_INITIALIZATION_SHA256,
        )
        shared = audit["shared_initialization"]
        self.assertEqual(
            (
                shared["control_key_count"],
                shared["treatment_key_count"],
                shared["added_key_count"],
                shared["identical_key_count"],
                shared["added_parameter_count"],
                shared["per_added_block_parameter_count"],
            ),
            (162, 214, 52, 152, 1_577_984, 788_992),
        )
        self.assertEqual(
            [
                branch["architecture"]["pre_global_receptive_field"]
                for branch in audit["branches"]
            ],
            [89, 133],
        )
        self.assertEqual(audit["target_presentations_per_branch"], 102_400_000)
        peers = audit["stored_peers"]
        control_peer = peers["stage_49_control"]
        self.assertEqual(
            (
                control_peer["key"],
                control_peer["width"],
                control_peer["depth"],
                control_peer["parameter_count"],
                control_peer["best_step"],
            ),
            ("c256_l6_lineage", 256, 6, 4_850_274, 48_500),
        )
        self.assertEqual(
            control_peer["file_sha256"], STAGE.DEFAULT_STAGE_49_CONTROL_SHA256
        )
        budget_peer = peers["c320_budget_peer"]
        self.assertEqual(
            (
                budget_peer["key"],
                budget_peer["width"],
                budget_peer["depth"],
                budget_peer["parameter_count"],
                budget_peer["best_step"],
            ),
            ("c320_l6", 320, 6, 7_537_378, 49_500),
        )
        self.assertEqual(budget_peer["file_sha256"], STAGE.C320_BUDGET_REF["file_sha256"])

    def test_shared_initialization_prefix_is_a_block_ordered_extension(self):
        data, config, _ = self.fixture()
        specs = STAGE.branch_specs(Path("/tmp"))
        audit = STAGE.shared_initialization_audit(specs[0], specs[1], data, config)
        self.assertTrue(audit["treatment_is_a_key_superset"])
        self.assertEqual(audit["changed_mask_layers"], (4, 5))
        # Two blocks of identical width are the entire parameter difference.
        self.assertEqual(
            audit["added_parameter_count"],
            2 * audit["per_added_block_parameter_count"],
        )
        self.assertEqual(
            audit["treatment_parameter_count"] - audit["control_parameter_count"],
            audit["added_parameter_count"],
        )
        self.assertEqual(audit["added_block_layers"], (6, 7))
        # Every added entry belongs to the two new blocks, and every shared entry
        # outside the licensed exceptions is byte-identical.
        expectations = STAGE.expected_initialization_keys(specs[0], specs[1])
        self.assertEqual(
            expectations["expected_differing"],
            tuple(
                f"blocks.{layer}.sa.heads.{head}.tril"
                for layer in (4, 5)
                for head in range(STAGE._STAGE_44.N_HEAD)
            )
            + ("lm_head.weight", "lm_head.bias"),
        )
        # Every added state entry belongs to the two new blocks, so the count is
        # exactly twice one block's entry count.
        model = STAGE.build_model(
            data,
            config,
            STAGE.spec_for(specs[1], config.block_size),
            STAGE._STAGE_44.N_HEAD,
            torch.device("cpu"),
        )
        block_entries = sum(
            1 for name in model.state_dict() if name.startswith("blocks.0.")
        )
        del model
        self.assertEqual(audit["added_key_count"], 2 * block_entries)
        self.assertEqual(
            audit["identical_key_count"],
            audit["control_key_count"] - len(expectations["expected_differing"]),
        )
        self.assertEqual(
            audit["control_key_count"] + audit["added_key_count"],
            audit["treatment_key_count"],
        )
        self.assertNotEqual(
            audit["control_masks_sha256"], audit["treatment_masks_sha256"]
        )
        self.assertEqual(
            audit["control_identical_keys_sha256"],
            audit["treatment_identical_keys_sha256"],
        )
        # The claim is load-bearing: a treatment drawn from a different seed must
        # be rejected rather than silently reported as an extension.
        calls = {"count": 0}
        genuine = STAGE._STAGE_41.seed_everything

        def shifting(seed: int) -> None:
            calls["count"] += 1
            genuine(seed if calls["count"] == 1 else seed + 1)

        with patch.object(STAGE._STAGE_41, "seed_everything", side_effect=shifting):
            with self.assertRaisesRegex(RuntimeError, "prefix changed"):
                STAGE.shared_initialization_audit(specs[0], specs[1], data, config)

    def test_paired_comparison_requires_the_exact_depth_step_at_fixed_width(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, device = self.run_pair(root)
            STAGE.verify_paired_branches(control, treatment)
            self.assertEqual(control.spec.depth, 6)
            self.assertEqual(treatment.spec.depth, 8)
            self.assertEqual(
                treatment.initialization["parameter_count"]
                - control.initialization["parameter_count"],
                2 * control.initialization["block_parameter_count"],
            )
            shallow = self.spec(
                root,
                "treatment",
                width=config.n_embd,
                depth=STAGE.DEFAULT_CONTROL_DEPTH,
            )
            with self.assertRaisesRegex(RuntimeError, "L=8 against L=6"):
                STAGE.verify_paired_specs(
                    control.spec,
                    shallow,
                    control.initialization["parameter_count"],
                    control.initialization["parameter_count"],
                    control.initialization["block_parameter_count"],
                )
            with self.assertRaisesRegex(RuntimeError, "hold width fixed"):
                STAGE.verify_paired_specs(
                    control.spec,
                    replace(treatment.spec, width=config.n_embd + 4),
                    control.initialization["parameter_count"],
                    treatment.initialization["parameter_count"],
                    control.initialization["block_parameter_count"],
                )
            with self.assertRaisesRegex(RuntimeError, "hold dropout fixed"):
                STAGE.verify_paired_specs(
                    control.spec,
                    replace(treatment.spec, dropout=0.20),
                    control.initialization["parameter_count"],
                    treatment.initialization["parameter_count"],
                    control.initialization["block_parameter_count"],
                )
            with self.assertRaisesRegex(RuntimeError, "LLLLGG"):
                STAGE.verify_paired_specs(
                    replace(control.spec, layer_contexts=(31, 31, 31, 31, None, None)),
                    treatment.spec,
                    control.initialization["parameter_count"],
                    treatment.initialization["parameter_count"],
                    control.initialization["block_parameter_count"],
                )
            with self.assertRaisesRegex(RuntimeError, "wrong parameter count"):
                STAGE.verify_paired_specs(
                    control.spec,
                    treatment.spec,
                    control.initialization["parameter_count"],
                    control.initialization["parameter_count"] + 1,
                    control.initialization["block_parameter_count"],
                )
            with self.assertRaisesRegex(RuntimeError, "different training batches"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        final_streams={
                            **treatment.final_streams,
                            "training_generator_sha256": "x",
                        },
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "different torch_rng_sha256"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        initial_streams={**treatment.initial_streams, "torch_rng_sha256": "x"},
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "identical weights"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        initial_streams={
                            **treatment.initial_streams,
                            "model_sha256": control.initial_streams["model_sha256"],
                        },
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "did not consume more dropout RNG"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        final_streams={
                            **treatment.final_streams,
                            "torch_rng_sha256": control.final_streams["torch_rng_sha256"],
                        },
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "budgets"):
                STAGE.verify_paired_branches(
                    control,
                    replace(
                        treatment,
                        training=replace(
                            treatment.training,
                            history=treatment.training.history[:-1],
                        ),
                    ),
                )

    def test_real_pair_shares_the_sampler_and_saves_resumable_endpoints(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, _ = self.run_pair(root)
            STAGE.verify_paired_branches(control, treatment)
            self.assertEqual(
                tuple(record.step for record in control.training.history), (0, 1, 2, 3)
            )
            # One batch stream, two draws: the sampler and the accelerator RNG
            # start identical, the weights and the dropout schedules do not.
            self.assertEqual(
                control.initial_streams["training_generator_sha256"],
                treatment.initial_streams["training_generator_sha256"],
            )
            self.assertEqual(
                control.initial_streams["torch_rng_sha256"],
                treatment.initial_streams["torch_rng_sha256"],
            )
            self.assertNotEqual(
                control.initial_streams["model_sha256"],
                treatment.initial_streams["model_sha256"],
            )
            self.assertEqual(
                control.final_streams["training_generator_sha256"],
                treatment.final_streams["training_generator_sha256"],
            )
            self.assertNotEqual(
                control.final_streams["model_sha256"],
                treatment.final_streams["model_sha256"],
            )
            self.assertNotEqual(
                control.final_streams["torch_rng_sha256"],
                treatment.final_streams["torch_rng_sha256"],
            )
            for report in (control, treatment):
                spec = report.spec
                final = STAGE.validate_branch_checkpoint(
                    spec.final_path, spec, data, config, kind="final"
                )
                self.assertEqual(final["step"], 3)
                self.assertEqual(final["architecture"]["n_layer"], spec.depth)
                self.assertEqual(
                    final["architecture"]["layer_contexts"], spec.layer_contexts
                )
                self.assertEqual(
                    final["experiment"]["branch_depth"], spec.depth
                )
                self.assertEqual(
                    final["initialization"]["parameter_count"],
                    STAGE.parameter_count_from_state_dict(final["model_state_dict"]),
                )
                self.assertEqual(
                    {
                        int(state["step"])
                        for state in final["optimizer_state_dict"]["state"].values()
                    },
                    {3},
                )
                self.assertEqual(
                    STAGE.file_hash(spec.final_path), report.final_sha256
                )
                self.assertEqual(STAGE.file_hash(spec.best_path), report.best_sha256)

    def test_checkpoint_rejects_structural_tampering_independently_of_its_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, _ = self.run_pair(root)
            original = torch.load(treatment.spec.final_path, weights_only=True)
            # Each of these fields is validated structurally, so the tamper must
            # be rejected even when the payload integrity hash is recomputed to
            # match the tampered content.
            structural = (
                lambda payload: payload.update(step=2),
                lambda payload: payload.update(best_step=4),
                lambda payload: payload.update(checkpoint_kind="best"),
                lambda payload: payload["optimizer_state_dict"]["param_groups"][
                    0
                ].update(lr=3e-5),
                lambda payload: payload["optimizer_state_dict"]["param_groups"][
                    0
                ].update(betas=(0.8, 0.99)),
                lambda payload: payload["optimizer_state_dict"]["param_groups"][
                    0
                ].update(weight_decay=0.0),
                lambda payload: payload["architecture"].update(n_layer=6),
                lambda payload: payload["architecture"].update(
                    layer_contexts=(23, 23, 23, 23, None, None)
                ),
                lambda payload: payload["architecture"].update(residual_dropout=0.10),
                lambda payload: payload["architecture"].update(
                    pre_global_receptive_field=89
                ),
                lambda payload: payload["architecture"].update(
                    first_global_reachability_layer=5
                ),
                lambda payload: payload["experiment"].update(branch_depth=6),
                lambda payload: payload["experiment"].update(stage=49),
                lambda payload: payload["experiment"].update(
                    branch_layer_contexts=(23, 23, 23, 23, None, None)
                ),
                lambda payload: payload["experiment"].update(
                    branch_residual_dropout=0.20
                ),
                lambda payload: payload["initialization"].update(depth=6),
                lambda payload: payload["initialization"].update(
                    layer_contexts=(23, 23, 23, 23, None, None)
                ),
                lambda payload: payload["initialization"].update(
                    residual_dropout=0.20
                ),
                lambda payload: payload["initialization"].update(origin_stage=49),
                lambda payload: payload["initialization"].update(
                    historical_weights_loaded=True
                ),
                lambda payload: payload["initialization"].update(parameter_count=1),
                lambda payload: payload.update(optimizer_restart_step=1),
                lambda payload: payload.update(training_config={
                    **payload["training_config"], "eval_iters": 99
                }),
            )
            for mutate in structural:
                payload = copy.deepcopy(original)
                mutate(payload)
                payload["payload_sha256"] = STAGE.state_hash(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "payload_sha256"
                    }
                )
                path = root / "tampered.pt"
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    STAGE.validate_branch_checkpoint(
                        path, treatment.spec, data, config, kind="final"
                    )
            # The control's own spec must reject the deeper checkpoint too.
            with self.assertRaises(ValueError):
                STAGE.validate_branch_checkpoint(
                    treatment.spec.final_path, control.spec, data, config, "final"
                )

    def test_checkpoint_rejects_content_tampering_through_its_integrity_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, _, data, config, _ = self.run_pair(root)
            original = torch.load(control.spec.final_path, weights_only=True)
            # These fields carry no independent structural check, so the stored
            # payload hash is the defense that must reject them.
            content = (
                lambda payload: payload["snapshot_losses"].update(train=9.0),
                lambda payload: payload.update(best_val_loss=9.0),
                lambda payload: payload.update(
                    training_generator_state=torch.Generator()
                    .manual_seed(999)
                    .get_state()
                ),
                lambda payload: next(iter(payload["model_state_dict"].values())).add_(1),
                lambda payload: payload.update(data_fingerprint="0" * 64),
            )
            for mutate in content:
                payload = copy.deepcopy(original)
                mutate(payload)
                path = root / "stale.pt"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "integrity"):
                    STAGE.validate_branch_checkpoint(
                        path, control.spec, data, config, kind="final"
                    )

    def test_fresh_panel_is_shared_repeatable_and_seed_disjoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, device = self.run_pair(root)
            torch.manual_seed(991)
            before = torch.get_rng_state().clone()
            first = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1368, eval_iters=5
            )
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            second = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1368, eval_iters=5
            )
            self.assertEqual(asdict(first), asdict(second))
            self.assertEqual(len(first.baseline.batch_losses), 5)
            self.assertEqual(len(first.candidate.batch_losses), 5)
            differences = [
                candidate - baseline
                for baseline, candidate in zip(
                    first.baseline.batch_losses, first.candidate.batch_losses
                )
            ]
            self.assertAlmostEqual(first.delta.mean_delta, sum(differences) / 5)
            # Every seed Stages 43-49 consumed, plus this stage's other panel, is
            # refused by both panel constructors.
            for seed in (STAGE.DEFAULT_FIXED_PANEL_SEED, *range(1354, 1368)):
                for evaluate in (
                    lambda s=seed: STAGE.evaluate_checkpoint_pair(
                        control, treatment, data, config, device, seed=s, eval_iters=5
                    ),
                    lambda s=seed: STAGE.evaluate_budget_shootout(
                        data,
                        config,
                        device,
                        root,
                        specs=(control.spec, treatment.spec),
                        peer_loaders=self.peers(control.spec, data, config),
                        expected_generator_sha256=None,
                        seed=s,
                        eval_iters=5,
                    ),
                ):
                    with self.assertRaises(ValueError):
                        evaluate()
            for count in (0, 1):
                with self.assertRaises(ValueError):
                    STAGE.evaluate_checkpoint_pair(
                        control,
                        treatment,
                        data,
                        config,
                        device,
                        seed=1368,
                        eval_iters=count,
                    )

    def test_budget_panel_ranks_pairs_and_mirrors_deltas(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, control, treatment, data, config, device = self.panel(root)
            self.assertEqual(
                sorted(report["endpoints"]),
                ["c16_l6", "c16_l8", STAGE.BUDGET_PEER_KEY],
            )
            self.assertEqual(
                report["panel_keys"],
                {
                    "control": "c16_l6",
                    "treatment": "c16_l8",
                    "budget_peer": STAGE.BUDGET_PEER_KEY,
                    "stored_l6_peer": None,
                },
            )
            self.assertEqual(len(report["comparisons"]), 3)
            self.assertTrue(report["shared_training_batch_stream_verified"])
            self.assertFalse(report["dropout_enabled"])
            self.assertEqual(report["seed"], 1369)
            self.assertEqual(
                [entry["rank"] for entry in report["ranking"]], [1, 2, 3]
            )
            losses = [entry["mean_validation_loss"] for entry in report["ranking"]]
            self.assertEqual(losses, sorted(losses))
            self.assertEqual(len(report["final_generator_sha256"]), 64)
            self.assertEqual(
                report["endpoints"]["c16_l8"]["depth"], 8
            )
            self.assertEqual(report["endpoints"]["c16_l6"]["parameter_ratio_vs_control"], 1.0)
            champion = report["ranking"][0]["key"]
            runner_up = report["ranking"][1]["key"]
            stored = STAGE.compared_delta(report["comparisons"], champion, runner_up)
            mirror = STAGE.compared_delta(report["comparisons"], runner_up, champion)
            self.assertAlmostEqual(mirror["mean_delta"], -stored["mean_delta"], places=15)
            self.assertAlmostEqual(
                mirror["confidence_low"], -stored["confidence_high"], places=15
            )
            self.assertAlmostEqual(
                mirror["confidence_high"], -stored["confidence_low"], places=15
            )
            self.assertEqual(mirror["standard_error"], stored["standard_error"])
            self.assertEqual(
                (stored["candidate"], stored["baseline"]), (champion, runner_up)
            )
            self.assertEqual(
                (mirror["candidate"], mirror["baseline"]), (runner_up, champion)
            )
            self.assertEqual(
                STAGE.compared_delta(report["comparisons"], champion, runner_up)[
                    "mean_delta"
                ],
                stored["mean_delta"],
            )
            self.assertIn(report["ranking"][0]["label"], STAGE.budget_panel_outcome(report))
            self.assertIn("spend", STAGE.substitution_outcome(report))
            with self.assertRaises(KeyError):
                STAGE.compared_delta(report["comparisons"], "c16_l6", "absent")
            with self.assertRaisesRegex(ValueError, "lineage stream"):
                STAGE.evaluate_budget_shootout(
                    data,
                    config,
                    device,
                    root,
                    specs=(control.spec, treatment.spec),
                    peer_loaders=self.peers(control.spec, data, config),
                    expected_generator_sha256="0" * 64,
                    seed=1369,
                    eval_iters=5,
                )

    def test_budget_panel_rejects_a_mismatched_width_or_batch_stream(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            control_spec = self.spec(
                root, "control", width=config.n_embd, depth=STAGE.DEFAULT_CONTROL_DEPTH
            )
            treatment_spec = self.spec(
                root,
                "treatment",
                width=config.n_embd,
                depth=STAGE.DEFAULT_TREATMENT_DEPTH,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                STAGE.run_branch(control_spec, data, config, device)
                with patch.object(STAGE, "DEFAULT_TRAINING_BATCH_SEED", 4242):
                    STAGE.run_branch(treatment_spec, data, config, device)
            # A treatment sampled differently from the control cannot be paired
            # with it, however close its architecture is.
            with self.assertRaisesRegex(
                RuntimeError, "different training_generator_sha256"
            ):
                STAGE.evaluate_budget_shootout(
                    data,
                    config,
                    device,
                    root,
                    specs=(control_spec, treatment_spec),
                    peer_loaders=self.peers(control_spec, data, config),
                    expected_generator_sha256=None,
                    seed=1369,
                    eval_iters=2,
                )
            with self.assertRaisesRegex(ValueError, "must match the evaluated width"):
                STAGE.budget_plan(
                    data,
                    replace(config, n_embd=config.n_embd + 4),
                    root,
                    (control_spec, treatment_spec),
                    (),
                )
            payload = STAGE.validate_branch_checkpoint(
                control_spec.final_path, control_spec, data, config, "final"
            )
            with self.assertRaisesRegex(ValueError, "unrecorded stored budget peer"):
                STAGE.budget_plan(
                    data,
                    config,
                    root,
                    (control_spec, treatment_spec),
                    (
                        lambda data, directory: STAGE.StoredEndpoint(
                            "c256_l6",
                            config.n_embd,
                            STAGE.DEFAULT_CONTROL_DEPTH,
                            STAGE.DEFAULT_DROPOUT,
                            control_spec.final_path,
                            payload,
                            config,
                        ),
                    ),
                )

    def test_stored_endpoint_pairing_rejects_divergence(self):
        """The stored-endpoint contract is checked directly, not only via a run."""

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, _ = self.run_pair(root)
            specs = (control.spec, treatment.spec)

            def entry(branch, spec):
                payload = STAGE.validate_branch_checkpoint(
                    branch.spec.final_path, branch.spec, data, config, "final"
                )
                return {"payload": payload, "spec": spec}

            control_entry = entry(control, control.spec)
            treatment_entry = entry(treatment, treatment.spec)
            STAGE.verify_paired_endpoints(control_entry, treatment_entry, specs)
            divergence = {
                "training_generator_state": torch.Generator()
                .manual_seed(4242)
                .get_state(),
            }
            with self.assertRaisesRegex(RuntimeError, "different training batches"):
                STAGE.verify_paired_endpoints(
                    control_entry,
                    {
                        **treatment_entry,
                        "payload": {
                            **treatment_entry["payload"],
                            **divergence,
                        },
                    },
                    specs,
                )
            with self.assertRaisesRegex(RuntimeError, "identical weights"):
                STAGE.verify_paired_endpoints(
                    control_entry,
                    {
                        **treatment_entry,
                        "payload": {
                            **treatment_entry["payload"],
                            "model_state_dict": control_entry["payload"][
                                "model_state_dict"
                            ],
                        },
                    },
                    specs,
                )
            with self.assertRaisesRegex(RuntimeError, "dropout RNG"):
                STAGE.verify_paired_endpoints(
                    control_entry,
                    {
                        **treatment_entry,
                        "payload": {
                            **treatment_entry["payload"],
                            "rng_state": control_entry["payload"]["rng_state"],
                            "runtime": control_entry["payload"]["runtime"],
                        },
                    },
                    specs,
                )

    def test_budget_panel_requires_a_stored_peer(self):
        """A peerless panel cannot answer the budget question, so it is refused."""

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, device = self.run_pair(Path(root))
            with self.assertRaisesRegex(ValueError, "needs both fresh branches"):
                STAGE.evaluate_budget_shootout(
                    data,
                    config,
                    device,
                    root,
                    specs=(control.spec, treatment.spec),
                    peer_loaders=(),
                    expected_generator_sha256=None,
                    seed=1369,
                    eval_iters=3,
                )

    def test_canonical_panel_measures_no_device_shift(self):
        """On one device the panel reports the budget question and no shift."""

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, control, treatment, data, config, device = self.panel(root)
            self.assertIsNone(report["panel_keys"]["stored_l6_peer"])
            self.assertFalse(report["device_confounded"])
            self.assertEqual(report["device_confounded_endpoints"], [])
            self.assertEqual(
                report["device_names"],
                {
                    "c16_l6": "cpu",
                    "c16_l8": "cpu",
                    STAGE.BUDGET_PEER_KEY: "cpu",
                },
            )
            self.assertEqual(report["fresh_device_name"], "cpu")
            self.assertIsNone(STAGE.device_shift_delta(report))
            self.assertIn(
                "no device shift was measured", STAGE.device_shift_outcome(report)
            )
            self.assertNotIn(
                "device-confounded", STAGE.substitution_outcome(report)
            )

    def test_foreign_device_panel_scores_the_stored_control_as_a_shift(self):
        """Off the lineage device the stored L=6 endpoint measures the shift."""

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            foreign = "NVIDIA GeForce RTX 3090"
            report, control, treatment, data, config, device = self.panel(
                root,
                with_stored_l6=True,
                foreign_peer_device=foreign,
            )
            self.assertEqual(
                report["panel_keys"]["stored_l6_peer"], STAGE.STORED_L6_PEER_KEY
            )
            self.assertEqual(
                sorted(report["endpoints"]),
                [
                    "c16_l6",
                    "c16_l8",
                    STAGE.STORED_L6_PEER_KEY,
                    STAGE.BUDGET_PEER_KEY,
                ],
            )
            self.assertTrue(report["device_confounded"])
            self.assertEqual(
                report["device_confounded_endpoints"],
                sorted([STAGE.BUDGET_PEER_KEY, STAGE.STORED_L6_PEER_KEY]),
            )
            self.assertEqual(report["fresh_device_name"], "cpu")
            self.assertEqual(report["device_names"][STAGE.STORED_L6_PEER_KEY], foreign)
            # Every unordered pair is still compared; the two extra peers add
            # three comparisons to the three the two-branch panel had.
            self.assertEqual(len(report["comparisons"]), 6)
            self.assertIn(
                f"{STAGE.BUDGET_PEER_KEY}_minus_{STAGE.STORED_L6_PEER_KEY}"
                in report["comparisons"]
                or f"{STAGE.STORED_L6_PEER_KEY}_minus_{STAGE.BUDGET_PEER_KEY}"
                in report["comparisons"],
                [True],
            )
            shift = STAGE.device_shift_delta(report)
            self.assertIsNotNone(shift)
            # The shift is the fresh control minus the stored one, re-derived from
            # the panel's own per-batch losses.
            endpoints = report["endpoints"]
            differences = [
                fresh - stored
                for stored, fresh in zip(
                    endpoints[STAGE.STORED_L6_PEER_KEY]["batch_losses"],
                    endpoints["c16_l6"]["batch_losses"],
                    strict=True,
                )
            ]
            self.assertAlmostEqual(
                shift["mean_delta"], sum(differences) / len(differences), places=12
            )
            self.assertEqual(shift["candidate"], "c16_l6")
            self.assertEqual(shift["baseline"], STAGE.STORED_L6_PEER_KEY)
            text = STAGE.device_shift_outcome(report)
            self.assertIn(foreign, text)
            self.assertIn("cpu", text)
            self.assertIn(
                "device-confounded", STAGE.substitution_outcome(report)
            )
            self.assertIn(foreign, STAGE.budget_panel_outcome(report))
            # A zero-width shift interval is still self-consistent.
            self.assertLessEqual(shift["confidence_low"], shift["mean_delta"])
            self.assertLessEqual(shift["mean_delta"], shift["confidence_high"])
            ladder = STAGE.depth_budget_ladder(
                STAGE.evaluate_checkpoint_pair(
                    control, treatment, data, config, device, seed=1368, eval_iters=3
                ),
                report,
            )
            self.assertEqual(len(ladder["rows"]), 4)
            self.assertEqual(
                ladder["device_confounded_endpoints"],
                sorted([STAGE.BUDGET_PEER_KEY, STAGE.STORED_L6_PEER_KEY]),
            )
            self.assertIsNotNone(ladder["device_shift"])
            # The two L=6 rows tie on parameter count, so rows are ordered by key
            # within a tie: the fresh control, the stored lineage control, the
            # wider stored spend, then the deeper fresh branch.
            self.assertEqual(
                [row["source_stage"] for row in ladder["rows"]], [50, 49, 46, 50]
            )

    def test_device_modes_gate_foreign_accelerators(self):
        """A foreign accelerator is a deliberate mode, never a silent fallback."""

        self.assertEqual(
            STAGE.device_mode(torch.device("cpu"), allow_foreign=False),
            STAGE.CANONICAL_DEVICE_MODE,
        )
        self.assertEqual(
            STAGE.device_mode(torch.device("cpu"), allow_foreign=True),
            STAGE.CANONICAL_DEVICE_MODE,
        )
        STAGE.verify_lineage_device(torch.device("cpu"))
        STAGE.verify_device_mode(STAGE.CANONICAL_DEVICE_MODE, torch.device("cpu"))
        self.assertEqual(
            STAGE.budget_peer_loaders(STAGE.CANONICAL_DEVICE_MODE),
            (STAGE.load_c320_budget_peer,),
        )
        self.assertEqual(
            STAGE.budget_peer_loaders(STAGE.FOREIGN_DEVICE_MODE),
            (STAGE.load_c320_budget_peer, STAGE.load_stage_49_control),
        )
        with self.assertRaises(ValueError):
            STAGE.budget_peer_loaders("nonsense")
        with self.assertRaises(ValueError):
            STAGE.verify_device_mode("nonsense", torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            STAGE.verify_device_mode(STAGE.FOREIGN_DEVICE_MODE, torch.device("cpu"))
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this host")
        names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        lineage = names.index(STAGE.DEFAULT_LINEAGE_DEVICE_NAME)
        lineage_device = torch.device("cuda", lineage)
        self.assertEqual(
            STAGE.device_mode(lineage_device, allow_foreign=False),
            STAGE.CANONICAL_DEVICE_MODE,
        )
        STAGE.verify_device_mode(STAGE.CANONICAL_DEVICE_MODE, lineage_device)
        other = next(
            (
                index
                for index, name in enumerate(names)
                if name != STAGE.DEFAULT_LINEAGE_DEVICE_NAME
            ),
            None,
        )
        if other is None:
            return
        foreign = torch.device("cuda", other)
        with self.assertRaisesRegex(RuntimeError, "allow-foreign-device"):
            STAGE.device_mode(foreign, allow_foreign=False)
        with self.assertRaisesRegex(RuntimeError, "requires"):
            STAGE.verify_lineage_device(foreign)
        self.assertEqual(
            STAGE.device_mode(foreign, allow_foreign=True), STAGE.FOREIGN_DEVICE_MODE
        )
        STAGE.verify_device_mode(STAGE.FOREIGN_DEVICE_MODE, foreign)
        # The canonical mode refuses a foreign accelerator outright, while a
        # foreign-mode claim about the lineage device is simply inconsistent.
        with self.assertRaisesRegex(RuntimeError, "requires"):
            STAGE.verify_device_mode(STAGE.CANONICAL_DEVICE_MODE, foreign)
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            STAGE.verify_device_mode(STAGE.FOREIGN_DEVICE_MODE, lineage_device)

    def test_probe_comparison_reports_where_trajectories_diverge(self):
        recorded = {
            0: {"train": 1.0, "validation": 2.0},
            500: {"train": 0.9, "validation": 1.9},
            1000: {"train": 0.8, "validation": 1.8},
        }
        comparison = STAGE.compare_probe_records(recorded, recorded)
        self.assertEqual(comparison["steps_compared"], 3)
        self.assertEqual(comparison["identical_steps"], 3)
        self.assertIsNone(comparison["first_divergent_step"])
        self.assertEqual(comparison["max_abs_validation_delta"], 0.0)
        self.assertIn(
            "reproduces the recorded control trajectory exactly",
            STAGE.probe_interpretation(comparison, same_device=True),
        )
        observed = {
            0: {"train": 1.0, "validation": 2.0},
            500: {"train": 0.5, "validation": 1.9},
            1000: {"train": 0.8, "validation": 1.75},
        }
        comparison = STAGE.compare_probe_records(observed, recorded)
        self.assertEqual(comparison["identical_steps"], 1)
        self.assertEqual(comparison["first_divergent_step"], 500)
        self.assertAlmostEqual(comparison["max_abs_train_delta"], 0.4)
        self.assertAlmostEqual(comparison["max_abs_validation_delta"], 0.05)
        rows = {row["step"]: row for row in comparison["rows"]}
        self.assertFalse(rows[500]["identical"])
        self.assertAlmostEqual(rows[500]["train_delta"], -0.4)
        self.assertIn(
            "not the one that produced the stored checkpoints",
            STAGE.probe_interpretation(comparison, same_device=True),
        )
        self.assertIn(
            "accelerator's numerics enter the training loop",
            STAGE.probe_interpretation(comparison, same_device=False),
        )
        divergent_at_zero = STAGE.compare_probe_records(
            {**observed, 0: {"train": 1.1, "validation": 2.0}}, recorded
        )
        self.assertEqual(divergent_at_zero["first_divergent_step"], 0)
        self.assertIn(
            "forward-pass numerics",
            STAGE.probe_interpretation(divergent_at_zero, same_device=False),
        )
        with self.assertRaisesRegex(ValueError, "share no steps"):
            STAGE.compare_probe_records({7: {"train": 1.0, "validation": 1.0}}, recorded)

    def test_control_probe_replays_the_recorded_trajectory_and_writes_its_report(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            spec = self.spec(
                root, "control", width=config.n_embd, depth=STAGE.DEFAULT_CONTROL_DEPTH
            )
            records = []
            model, optimizer, generator, _ = STAGE.initialize_branch(
                data, config, spec, device
            )
            STAGE.train_branch(
                model, optimizer, generator, data, config, device, records.append
            )
            del model, optimizer, generator
            history = [
                {
                    "step": record.step,
                    "losses": {
                        "train": record.losses.train,
                        "validation": record.losses.val,
                    },
                }
                for record in records
            ]
            reference = root / "reference.json"
            reference.write_text(
                json.dumps(
                    {"stage": 49, "training_executed": True, "control": {"training": {"history": history}}}
                ),
                encoding="utf-8",
            )
            probe_path = root / "probe.json"
            probe = STAGE.run_control_probe(
                spec,
                data,
                config,
                device,
                steps=config.max_iters,
                mode=STAGE.CANONICAL_DEVICE_MODE,
                reference_path=reference,
                probe_path=probe_path,
            )
            # The replay is deterministic, so it must reproduce its own record.
            self.assertIsNone(probe["comparison"]["first_divergent_step"])
            self.assertEqual(
                probe["comparison"]["steps_compared"], len(history)
            )
            self.assertEqual(probe["device"]["name"], "cpu")
            self.assertFalse(probe["same_device_as_lineage"])
            self.assertEqual(probe["depth"], 6)
            self.assertEqual(probe["steps"], config.max_iters)
            self.assertEqual(
                [row["step"] for row in probe["observed"]], [0, 1, 2, 3]
            )
            self.assertEqual(json.loads(probe_path.read_text())["kind"], "control_replay_probe")
            # A diagnostic must be replaceable, unlike an exclusive stage report.
            STAGE.run_control_probe(
                spec,
                data,
                config,
                device,
                steps=config.max_iters,
                mode=STAGE.CANONICAL_DEVICE_MODE,
                reference_path=reference,
                probe_path=probe_path,
            )
            # A divergent record is reported rather than smoothed over.
            tampered = json.loads(reference.read_text())
            tampered["control"]["training"]["history"][0]["losses"]["validation"] = 9.0
            reference.write_text(json.dumps(tampered), encoding="utf-8")
            probe = STAGE.run_control_probe(
                spec,
                data,
                config,
                device,
                steps=config.max_iters,
                mode=STAGE.CANONICAL_DEVICE_MODE,
                reference_path=reference,
                probe_path=probe_path,
            )
            self.assertEqual(probe["comparison"]["first_divergent_step"], 0)
            self.assertEqual(probe["comparison"]["identical_steps"], 3)
            with self.assertRaisesRegex(ValueError, "stage report"):
                STAGE.run_control_probe(
                    spec,
                    data,
                    config,
                    device,
                    steps=config.max_iters,
                    mode=STAGE.CANONICAL_DEVICE_MODE,
                    reference_path=reference,
                    probe_path=root / "stage_50_report.json",
                )
            with self.assertRaisesRegex(ValueError, "positive integer"):
                STAGE.run_control_probe(
                    spec,
                    data,
                    config,
                    device,
                    steps=0,
                    mode=STAGE.CANONICAL_DEVICE_MODE,
                    reference_path=reference,
                    probe_path=probe_path,
                )

    def test_recorded_control_history_is_read_from_the_stage_49_report(self):
        """The trajectory reader is exercised on a synthetic Stage 49 report.

        The published report is an ignored artifact, so this builds the same
        structure the reader expects: 101 rows from step 0 to step 50,000 whose
        terminal row carries the champion's fixed-panel losses.
        """

        steps = list(range(0, 50_001, 500))
        terminal = {
            "train": STAGE.DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_TRAIN_LOSS,
            "validation": STAGE.DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_VALIDATION_LOSS,
        }
        history = [
            {
                "step": step,
                "losses": dict(terminal) if step == steps[-1] else {"train": 2.0, "val": 2.1},
            }
            for step in steps
        ]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "stage_49_report.json"
            path.write_text(
                json.dumps(
                    {
                        "stage": 49,
                        "training_executed": True,
                        "control": {"training": {"history": history}},
                    }
                )
            )
            recorded = STAGE.load_recorded_control_history(path)
        self.assertEqual(len(recorded), 101)
        self.assertEqual(sorted(recorded)[:3], [0, 500, 1000])
        self.assertEqual(sorted(recorded)[-1], 50_000)
        # The terminal row is the stored champion's own fixed-panel record, so
        # the probe's reference is the right one.
        self.assertEqual(recorded[50_000]["validation"], terminal["validation"])
        self.assertEqual(recorded[50_000]["train"], terminal["train"])
        # A report that spells the field "val" is read the same way.
        self.assertEqual(recorded[500]["validation"], 2.1)
        with self.assertRaises(FileNotFoundError):
            STAGE.load_recorded_control_history(Path("/nonexistent/stage_49_report.json"))
        with tempfile.TemporaryDirectory() as raw:
            wrong = Path(raw) / "wrong.json"
            wrong.write_text(json.dumps({"stage": 50, "training_executed": True}))
            with self.assertRaisesRegex(ValueError, "Stage 49 run"):
                STAGE.load_recorded_control_history(wrong)

    def test_canonical_recorded_control_history_matches_the_stage_49_report(self):
        """The published trajectory, checked only where its report exists."""

        if not STAGE.DEFAULT_STAGE_49_REPORT_PATH.is_file():
            self.skipTest("recorded Stage 49 report is unavailable")
        history = STAGE.load_recorded_control_history(STAGE.DEFAULT_STAGE_49_REPORT_PATH)
        self.assertEqual(len(history), 101)
        self.assertEqual(sorted(history)[:3], [0, 500, 1000])
        self.assertEqual(sorted(history)[-1], 50_000)
        self.assertEqual(
            history[50_000]["validation"],
            STAGE.DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_VALIDATION_LOSS,
        )
        self.assertEqual(
            history[50_000]["train"],
            STAGE.DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_TRAIN_LOSS,
        )

    def test_diagnostic_writer_replaces_atomically_and_refuses_non_json(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "diag.json"
            STAGE.write_diagnostic(path, {"first": 1})
            self.assertEqual(json.loads(path.read_text()), {"first": 1})
            STAGE.write_diagnostic(path, {"second": 2, "path": root})
            self.assertEqual(
                json.loads(path.read_text()), {"second": 2, "path": str(root)}
            )
            self.assertEqual(
                [entry.name for entry in root.iterdir()], ["diag.json"]
            )
            with self.assertRaisesRegex(ValueError, "must be JSON"):
                STAGE.write_diagnostic(root / "diag.txt", {"first": 1})

    def test_outcome_classification_reports_resolution_direction(self):
        def paired(mean: float, low: float, high: float):
            return types.SimpleNamespace(
                delta=types.SimpleNamespace(
                    mean_delta=mean, confidence_low=low, confidence_high=high
                )
            )

        # The treatment is deeper, so a negative delta (deeper validating better)
        # is the improving direction.
        self.assertIn("L=8 improves", STAGE.paired_outcome(paired(-0.01, -0.02, -0.001)))
        self.assertIn("L=6 beats", STAGE.paired_outcome(paired(0.01, 0.001, 0.02)))
        self.assertIn("unresolved", STAGE.paired_outcome(paired(0.0, -0.01, 0.01)))
        self.assertNotIn("improves", STAGE.paired_outcome(paired(0.01, 0.001, 0.02)))

        def delta(low: float, high: float) -> dict:
            return {
                "candidate": "candidate",
                "baseline": "baseline",
                "mean_delta": (low + high) / 2,
                "standard_error": (high - low) / 4,
                "confidence_low": low,
                "confidence_high": high,
                "critical_value": 2.0,
            }

        def panel(
            depth_delta: tuple[float, float], peer_delta: tuple[float, float]
        ) -> dict:
            """`peer_delta` is the loss interval of each peer minus the L=8 leader."""

            return {
                "panel_keys": {
                    "control": "c256_l6",
                    "treatment": "c256_l8",
                    "budget_peer": "c320_l6",
                },
                "endpoints": {
                    "c256_l6": {"parameter_count": 4_850_274},
                    "c256_l8": {"parameter_count": 6_428_258},
                    "c320_l6": {"parameter_count": 7_537_378},
                },
                "comparisons": {
                    "c256_l8_minus_c320_l6": delta(*depth_delta),
                    "c256_l6_minus_c256_l8": delta(*peer_delta),
                    "c256_l6_minus_c320_l6": delta(*peer_delta),
                },
                "ranking": [
                    {"key": "c256_l8", "label": "C256 L=8 p=0.15"},
                    {"key": "c256_l6", "label": "C256 L=6 p=0.15"},
                    {"key": "c320_l6", "label": "C320 L=6 p=0.15"},
                ],
            }

        # Depth beats the wider spend and the panel resolves both peers.
        resolved = STAGE.budget_panel_outcome(
            panel((-0.03, -0.01), (0.005, 0.02))
        )
        self.assertIn("resolved champion", resolved)
        self.assertIn("beats all 2 peers", resolved)
        self.assertIn("depth is the better spend", resolved)
        self.assertIn("0.8529x", resolved)
        # A peer straddling zero leaves the leader unresolved against it, and the
        # substitution clause is a separate statement about the budget.
        unresolved = STAGE.budget_panel_outcome(panel((-0.03, -0.01), (-0.01, 0.01)))
        self.assertIn("does not resolve", unresolved)
        self.assertIn("C256 L=6 p=0.15", unresolved)
        self.assertNotIn("resolved champion", unresolved)
        # The width spend wins only when the depth-minus-width interval is above
        # zero, which is the opposite direction from the depth step.
        width = STAGE.budget_panel_outcome(panel((0.01, 0.03), (0.005, 0.02)))
        self.assertIn("width is the better spend", width)
        # An interval straddling zero prefers neither allocation.
        neither = STAGE.substitution_outcome(panel((-0.005, 0.005), (0.005, 0.02)))
        self.assertIn("neither allocation is preferred", neither)

    def test_depth_budget_ladder_carries_the_measured_step_and_the_prior(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, control, treatment, data, config, device = self.panel(root)
            paired = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1368, eval_iters=4
            )
            ladder = STAGE.depth_budget_ladder(paired, report)
            self.assertEqual(
                [row["key"] for row in ladder["rows"]],
                ["c16_l6", "c320_l6", "c16_l8"],
            )
            self.assertEqual(
                [row["parameters"] for row in ladder["rows"]],
                sorted(row["parameters"] for row in ladder["rows"]),
            )
            self.assertEqual(
                [row["source_stage"] for row in ladder["rows"]], [50, 46, 50]
            )
            self.assertEqual(ladder["control_key"], "c16_l6")
            self.assertEqual(ladder["measured_depth_step"]["source_stage"], 50)
            self.assertEqual(
                ladder["measured_depth_step"]["fresh_panel_seed"], paired.seed
            )
            self.assertEqual(
                ladder["measured_depth_step"]["mean_delta"], paired.delta.mean_delta
            )
            self.assertEqual(
                ladder["measured_depth_step"]["parameter_delta"],
                2 * STAGE.DEFAULT_BLOCK_PARAMETER_COUNT,
            )
            self.assertAlmostEqual(
                ladder["measured_depth_step"]["parameter_ratio"],
                STAGE.DEFAULT_TREATMENT_PARAMETER_COUNT
                / STAGE.DEFAULT_CONTROL_PARAMETER_COUNT,
            )
            self.assertIsNotNone(ladder["budget_substitution"])
            prior = ladder["prior_depth_reference"]
            self.assertEqual(
                (prior["source_stage"], prior["mean_delta"], prior["standard_error"]),
                (24, -0.011038, 0.000647),
            )
            self.assertLess(prior["confidence_low"], prior["mean_delta"])
            self.assertLess(prior["mean_delta"], prior["confidence_high"])
            self.assertEqual(
                (prior["control_parameter_count"], prior["treatment_parameter_count"]),
                (816_705, 1_212_481),
            )
            # The prior is transcribed, never recomputed, so mutating a copy
            # cannot change the module constant.
            prior["mean_delta"] = 999.0
            self.assertEqual(STAGE.PRIOR_DEPTH_REFERENCE["mean_delta"], -0.011038)

    def test_branch_specs_and_artifact_paths_name_and_protect_stage_50_outputs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            specs = STAGE.branch_specs(root)
            STAGE.validate_branch_specs(specs)
            self.assertEqual(
                [(spec.key, spec.width, spec.depth) for spec in specs],
                [("control", 256, 6), ("treatment", 256, 8)],
            )
            self.assertEqual(
                [spec.best_path.name for spec in specs],
                [
                    "stage_50_control_c256_l6_best_checkpoint.pt",
                    "stage_50_treatment_c256_l8_best_checkpoint.pt",
                ],
            )
            self.assertEqual(
                [spec.final_path.name for spec in specs],
                [
                    "stage_50_control_c256_l6_step_50000_final_checkpoint.pt",
                    "stage_50_treatment_c256_l8_step_50000_final_checkpoint.pt",
                ],
            )
            for changed in (
                (specs[1], specs[0]),
                (replace(specs[1], depth=6, layer_contexts=(23, 23, 23, 23, None, None)), specs[0]),
                (specs[0], replace(specs[1], dropout=0.20)),
                (specs[0], replace(specs[1], width=320)),
                (
                    specs[0],
                    replace(
                        specs[1], layer_contexts=(31, 31, 31, 31, 31, 31, None, None)
                    ),
                ),
            ):
                with self.assertRaises(ValueError):
                    STAGE.validate_branch_specs(changed)
            for dropout in (-0.1, 1.0, float("nan")):
                with self.assertRaises(ValueError):
                    replace(specs[0], dropout=dropout)
            with self.assertRaises(ValueError):
                replace(specs[0], key="other")
            with self.assertRaises(ValueError):
                replace(specs[0], width=True)
            with self.assertRaises(ValueError):
                replace(specs[0], depth=7)
            with self.assertRaises(ValueError):
                # Unequal local windows are not a legal layerwise policy.
                replace(specs[1], layer_contexts=(23, 23, 23, 23, 31, 31, None, None))
            with self.assertRaises(ValueError):
                # A single Full layer is not a two-global-layer topology.
                replace(specs[0], layer_contexts=(23, 23, 23, 23, 23, None))
            with self.assertRaises(ValueError):
                # The Full layers must be consecutive and final.
                replace(specs[0], layer_contexts=(23, None, 23, 23, 23, None))
            outputs = tuple(
                path for spec in specs for path in (spec.best_path, spec.final_path)
            )
            STAGE.validate_artifact_paths((root / "corpus.txt",), outputs)
            with self.assertRaises(ValueError):
                STAGE.validate_artifact_paths((specs[0].best_path,), outputs)
            with self.assertRaises(ValueError):
                STAGE.validate_artifact_paths(
                    (), (specs[0].best_path, specs[0].best_path)
                )
            temporary = specs[0].best_path.with_name(f".{specs[0].best_path.name}.tmp")
            temporary.write_bytes(b"partial")
            with self.assertRaises(FileExistsError):
                STAGE.validate_artifact_paths((), outputs)
            temporary.unlink()
            specs[0].best_path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                STAGE.validate_artifact_paths((), outputs)

    def test_protocol_rejects_a_changed_corpus_or_an_additional_variable(self):
        data, _, _ = self.fixture()
        configs = (STAGE.canonical_config(), STAGE.canonical_config())
        with self.assertRaisesRegex(ValueError, "Shakespeare"):
            STAGE.validate_protocol(data, configs)
        with patch.object(
            STAGE,
            "fingerprint_data",
            return_value=STAGE._STAGE_41.DEFAULT_COMPLETE_DATA_FINGERPRINT,
        ):
            STAGE.validate_protocol(data, configs)
            for changed in (
                replace(configs[1], seed=configs[1].seed + 1),
                replace(configs[1], batch_size=16),
                replace(configs[1], n_embd=320),
                replace(configs[1], max_iters=49999),
                replace(configs[1], learning_rate=3e-5),
                replace(configs[1], eval_iters=99),
            ):
                with self.assertRaisesRegex(ValueError, "locks"):
                    STAGE.validate_protocol(data, (configs[0], changed))

    def fixture_stage_49_control(self, directory: Path):
        """A reduced Stage 49 C256 p=0.15 control pinned to this stage's constants.

        Stage 50 authenticates the stored champion by name, file hash, and
        fixed-panel record. A small real endpoint written by Stage 49's own
        writer exercises that path without the ignored historical checkpoint.
        """

        stage_49 = STAGE._STAGE_49
        data, config, spec, _ = artifact_fixtures.build_branch(
            stage_49,
            directory,
            "control",
            stage_49.DEFAULT_CONTROL_DROPOUT,
        )
        payload = stage_49.validate_branch_checkpoint(
            spec.final_path, spec, data, config, "final"
        )
        fixtures = (
            patch.object(stage_49, "branch_specs", return_value=(spec, spec)),
            patch.object(stage_49, "canonical_config", return_value=config),
            patch.object(
                STAGE,
                "DEFAULT_STAGE_49_CONTROL_SHA256",
                STAGE.file_hash(spec.final_path),
            ),
            patch.object(
                STAGE,
                "DEFAULT_STAGE_49_CONTROL_BEST_SHA256",
                STAGE.file_hash(spec.best_path),
            ),
            patch.object(
                STAGE,
                "DEFAULT_STAGE_49_CONTROL_BEST_STEP",
                payload["best_step"],
            ),
            patch.object(
                STAGE,
                "DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_TRAIN_LOSS",
                payload["snapshot_losses"]["train"],
            ),
            patch.object(
                STAGE,
                "DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_VALIDATION_LOSS",
                payload["snapshot_losses"]["validation"],
            ),
        )
        return data, spec, payload, fixtures

    def test_stage_49_reference_is_authenticated_before_it_is_trusted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            data, reference_spec, payload, fixtures = (
                self.fixture_stage_49_control(directory)
            )
            # The champion this stage steps up from is pinned by name and by
            # content, so a rerun that changed either would invalidate the
            # comparison and every budget row it feeds rather than silently
            # measuring a different model.
            self.assertEqual(
                reference_spec.final_path.name,
                STAGE.DEFAULT_STAGE_49_CONTROL_FILENAME,
            )
            with contextlib.ExitStack() as stack:
                for fixture in fixtures:
                    stack.enter_context(fixture)
                self.assertEqual(
                    STAGE.file_hash(reference_spec.final_path),
                    STAGE.DEFAULT_STAGE_49_CONTROL_SHA256,
                )
                self.assertEqual(
                    STAGE.file_hash(reference_spec.best_path),
                    STAGE.DEFAULT_STAGE_49_CONTROL_BEST_SHA256,
                )
                endpoint = STAGE.load_stage_49_control(data, directory)
                self.assertEqual(
                    (endpoint.key, endpoint.width, endpoint.depth, endpoint.dropout),
                    ("c256_l6_lineage", 16, 6, 0.15),
                )
                self.assertEqual(
                    (endpoint.payload["step"], endpoint.payload["best_step"]),
                    (payload["step"], payload["best_step"]),
                )
                with patch.object(
                    STAGE, "DEFAULT_STAGE_49_CONTROL_SHA256", "0" * 64
                ):
                    with self.assertRaisesRegex(ValueError, "hash changed"):
                        STAGE.load_stage_49_control(data, directory)
                with patch.object(
                    STAGE, "DEFAULT_STAGE_49_CONTROL_BEST_SHA256", "0" * 64
                ):
                    with self.assertRaisesRegex(
                        ValueError, "best checkpoint hash changed"
                    ):
                        STAGE.load_stage_49_control(data, directory)
                with patch.object(
                    STAGE,
                    "DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_VALIDATION_LOSS",
                    1.0,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "fixed-panel record changed"
                    ):
                        STAGE.load_stage_49_control(data, directory)
                with patch.object(
                    STAGE, "DEFAULT_STAGE_49_CONTROL_FILENAME", "other.pt"
                ):
                    with self.assertRaisesRegex(ValueError, "changed its name"):
                        STAGE.load_stage_49_control(data, directory)

    def test_canonical_stage_49_artifact_is_authenticated_when_present(self):
        """The published Stage 49 champion, checked only where it exists."""

        directory = STAGE.DEFAULT_OUTPUT_DIRECTORY
        specs = STAGE._STAGE_49.branch_specs(directory)
        reference_spec = next(spec for spec in specs if spec.key == "control")
        if not (
            reference_spec.final_path.is_file()
            and reference_spec.best_path.is_file()
        ):
            self.skipTest("ignored canonical Stage 49 artifact is unavailable")
        data = STAGE.CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        self.assertEqual(
            STAGE.file_hash(reference_spec.final_path),
            STAGE.DEFAULT_STAGE_49_CONTROL_SHA256,
        )
        self.assertEqual(
            STAGE.file_hash(reference_spec.best_path),
            STAGE.DEFAULT_STAGE_49_CONTROL_BEST_SHA256,
        )
        endpoint = STAGE.load_stage_49_control(data, directory)
        self.assertEqual(
            (endpoint.key, endpoint.width, endpoint.depth, endpoint.dropout),
            ("c256_l6_lineage", 256, 6, 0.15),
        )
        self.assertEqual(
            (endpoint.payload["step"], endpoint.payload["best_step"]),
            (50000, 48500),
        )

    def fixture_c320_budget_peer(self, directory: Path):
        """A reduced C320 p=0.15 endpoint pinned to this stage's budget reference."""

        stage_46 = STAGE._STAGE_46
        data, config, spec, _ = artifact_fixtures.build_branch(
            stage_46,
            directory,
            "treatment",
            stage_46.DEFAULT_TREATMENT_DROPOUT,
        )
        payload = stage_46.validate_branch_checkpoint(
            spec.final_path, spec, data, config, "final"
        )
        reference = {
            **STAGE.C320_BUDGET_REF,
            "width": 16,
            "file_sha256": STAGE.file_hash(spec.final_path),
            "best_step": payload["best_step"],
            "fixed_panel_train_loss": payload["snapshot_losses"]["train"],
            "fixed_panel_validation_loss": payload["snapshot_losses"]["validation"],
        }
        fixtures = (
            patch.object(stage_46, "branch_specs", return_value=(spec, spec)),
            patch.object(stage_46, "canonical_config", return_value=config),
            patch.object(STAGE, "C320_BUDGET_REF", reference),
        )
        return data, spec, payload, fixtures

    def test_budget_peer_is_authenticated_against_its_recorded_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            data, reference_spec, payload, fixtures = (
                self.fixture_c320_budget_peer(directory)
            )
            self.assertEqual(STAGE.C320_BUDGET_REF["owner"], "stage_46")
            with contextlib.ExitStack() as stack:
                for fixture in fixtures:
                    stack.enter_context(fixture)
                self.assertEqual(STAGE.C320_BUDGET_REF["width"], 16)
                endpoint = STAGE.load_c320_budget_peer(data, directory)
                self.assertEqual(
                    (endpoint.key, endpoint.width, endpoint.depth, endpoint.dropout),
                    (STAGE.BUDGET_PEER_KEY, 16, 6, 0.15),
                )
                self.assertEqual(
                    endpoint.payload["initialization"]["parameter_count"],
                    payload["initialization"]["parameter_count"],
                )
                self.assertEqual(
                    (endpoint.payload["step"], endpoint.payload["best_step"]),
                    (payload["step"], payload["best_step"]),
                )
                changed = {**STAGE.C320_BUDGET_REF, "file_sha256": "0" * 64}
                with patch.object(STAGE, "C320_BUDGET_REF", changed):
                    with self.assertRaisesRegex(ValueError, "hash changed"):
                        STAGE.load_c320_budget_peer(data, directory)

    def test_canonical_c320_budget_peer_is_authenticated_when_present(self):
        """The published C320 endpoint, checked only where it exists."""

        directory = STAGE.DEFAULT_OUTPUT_DIRECTORY
        stage_46 = STAGE._STAGE_46
        specs = stage_46.branch_specs(directory)
        reference_spec = next(
            spec
            for spec in specs
            if spec.final_path.name == STAGE.C320_BUDGET_REF["filename"]
        )
        if not reference_spec.final_path.is_file():
            self.skipTest("ignored canonical C320 artifact is unavailable")
        data = STAGE.CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        endpoint = STAGE.load_c320_budget_peer(data, directory)
        self.assertEqual(
            (endpoint.key, endpoint.width, endpoint.depth, endpoint.dropout),
            (STAGE.BUDGET_PEER_KEY, 320, 6, 0.15),
        )
        self.assertEqual(
            endpoint.payload["initialization"]["parameter_count"],
            STAGE.DEFAULT_C320_PARAMETER_COUNT,
        )
        self.assertEqual(
            (endpoint.payload["step"], endpoint.payload["best_step"]),
            (50000, 49500),
        )

    def test_lineage_device_resolution_prefers_the_recorded_accelerator(self):
        self.assertEqual(STAGE.DEFAULT_LINEAGE_DEVICE_NAME, "NVIDIA GeForce RTX 3090")
        self.assertEqual(STAGE.resolve_lineage_device("cpu"), torch.device("cpu"))
        STAGE.verify_lineage_device(torch.device("cpu"))
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this host")
        available = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        resolved = STAGE.resolve_lineage_device("cuda")
        if STAGE.DEFAULT_LINEAGE_DEVICE_NAME not in available:
            with self.assertRaisesRegex(RuntimeError, "no device named"):
                STAGE.resolve_lineage_device("cuda")
            return
        self.assertEqual(resolved.type, "cuda")
        self.assertEqual(
            torch.cuda.get_device_name(resolved), STAGE.DEFAULT_LINEAGE_DEVICE_NAME
        )
        STAGE.verify_lineage_device(resolved)
        self.assertEqual(
            resolved.index, available.index(STAGE.DEFAULT_LINEAGE_DEVICE_NAME)
        )
        other = next(
            (
                index
                for index, name in enumerate(available)
                if name != STAGE.DEFAULT_LINEAGE_DEVICE_NAME
            ),
            None,
        )
        if other is not None:
            self.assertEqual(
                STAGE.resolve_lineage_device(f"cuda:{other}"),
                torch.device("cuda", other),
            )
            with self.assertRaisesRegex(RuntimeError, "requires"):
                STAGE.verify_lineage_device(torch.device("cuda", other))
        with self.assertRaisesRegex(ValueError, "not visible"):
            STAGE.resolve_lineage_device("cuda:99")

    def test_no_improvement_retains_the_native_initialization(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data, config, device = self.fixture()
            spec = self.spec(
                root, "control", width=config.n_embd, depth=STAGE.DEFAULT_CONTROL_DEPTH
            )
            initialized, _, _, _ = STAGE.initialize_branch(data, config, spec, device)
            initial_weights = {
                key: value.clone() for key, value in initialized.state_dict().items()
            }
            with (
                patch.object(
                    STAGE,
                    "evaluate_on_fixed_batches",
                    side_effect=[LossEstimate(1, 1)] + [LossEstimate(2, 2)] * 3,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = STAGE.run_branch(spec, data, config, device)
            best = torch.load(spec.best_path, weights_only=True)
            final = torch.load(spec.final_path, weights_only=True)
            self.assertEqual((report.best_step, best["step"], final["step"]), (0, 0, 3))
            self.assertFalse(best["optimizer_state_dict"]["state"])
            for key, value in initial_weights.items():
                self.assertTrue(torch.equal(value, best["model_state_dict"][key]), key)


if __name__ == "__main__":
    unittest.main()
