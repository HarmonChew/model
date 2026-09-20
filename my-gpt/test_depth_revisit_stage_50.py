from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

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

    def panel(self, directory: Path, seed: int = 1369, eval_iters: int = 5):
        """A budget panel built from a fresh fixture pair plus a stored peer."""

        control, treatment, data, config, device = self.run_pair(directory)
        payload = STAGE.validate_branch_checkpoint(
            control.spec.final_path, control.spec, data, config, "final"
        )

        def peer_loader(data: CharacterData, directory: Path):
            return STAGE.StoredEndpoint(
                STAGE.BUDGET_PEER_KEY,
                config.n_embd,
                STAGE.DEFAULT_CONTROL_DEPTH,
                STAGE.DEFAULT_DROPOUT,
                control.spec.final_path,
                payload,
                config,
            )

        report = STAGE.evaluate_budget_shootout(
            data,
            config,
            device,
            directory,
            specs=(control.spec, treatment.spec),
            peer_loader=peer_loader,
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

    def test_preflight_authenticates_initializations_and_stored_peers(self):
        """The canonical preflight gates the two stored endpoints before training."""

        data = STAGE.CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        config = STAGE.canonical_config()
        specs = STAGE.branch_specs(STAGE.DEFAULT_OUTPUT_DIRECTORY)
        audit = STAGE.preflight(data, (config, config), specs, torch.device("cpu"))
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
                control_peer["width"],
                control_peer["depth"],
                control_peer["parameter_count"],
                control_peer["best_step"],
            ),
            ("c256_l6_stage_49", 256, 6, 4_850_274, 48_500),
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
                        peer_loader=None,
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
                    peer_loader=None,
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
                    peer_loader=None,
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
                    None,
                )
            payload = STAGE.validate_branch_checkpoint(
                control_spec.final_path, control_spec, data, config, "final"
            )
            with self.assertRaisesRegex(ValueError, "recorded C320 endpoint"):
                STAGE.budget_plan(
                    data,
                    config,
                    root,
                    (control_spec, treatment_spec),
                    lambda data, directory: STAGE.StoredEndpoint(
                        "c256_l6",
                        config.n_embd,
                        STAGE.DEFAULT_CONTROL_DEPTH,
                        STAGE.DEFAULT_DROPOUT,
                        control_spec.final_path,
                        payload,
                        config,
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

    def test_budget_ladder_reports_no_substitution_without_a_stored_peer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control, treatment, data, config, device = self.run_pair(Path(root))
            report = STAGE.evaluate_budget_shootout(
                data,
                config,
                device,
                root,
                specs=(control.spec, treatment.spec),
                peer_loader=None,
                expected_generator_sha256=None,
                seed=1369,
                eval_iters=3,
            )
            self.assertIsNone(report["panel_keys"]["budget_peer"])
            self.assertEqual(len(report["comparisons"]), 1)
            self.assertIn(
                "was not scored", STAGE.substitution_outcome(report)
            )
            paired = STAGE.evaluate_checkpoint_pair(
                control, treatment, data, config, device, seed=1368, eval_iters=3
            )
            ladder = STAGE.depth_budget_ladder(paired, report)
            self.assertIsNone(ladder["budget_substitution"])
            self.assertEqual(len(ladder["rows"]), 2)

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

    def test_stage_49_reference_is_authenticated_before_it_is_trusted(self):
        directory = STAGE.DEFAULT_OUTPUT_DIRECTORY
        data = STAGE.CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        specs = STAGE._STAGE_49.branch_specs(directory)
        reference_spec = next(spec for spec in specs if spec.key == "control")
        # The champion this stage steps up from is pinned by name and by content,
        # so a rerun that changed either would invalidate the comparison and every
        # budget row it feeds rather than silently measuring a different model.
        self.assertEqual(
            reference_spec.final_path.name,
            STAGE.DEFAULT_STAGE_49_CONTROL_FILENAME,
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
            ("c256_l6_stage_49", 256, 6, 0.15),
        )
        self.assertEqual(
            (endpoint.payload["step"], endpoint.payload["best_step"]), (50000, 48500)
        )
        with patch.object(STAGE, "DEFAULT_STAGE_49_CONTROL_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "hash changed"):
                STAGE.load_stage_49_control(data, directory)
        with patch.object(STAGE, "DEFAULT_STAGE_49_CONTROL_BEST_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "best checkpoint hash changed"):
                STAGE.load_stage_49_control(data, directory)
        with patch.object(
            STAGE, "DEFAULT_STAGE_49_CONTROL_FIXED_PANEL_VALIDATION_LOSS", 1.0
        ):
            with self.assertRaisesRegex(ValueError, "fixed-panel record changed"):
                STAGE.load_stage_49_control(data, directory)
        with patch.object(STAGE, "DEFAULT_STAGE_49_CONTROL_FILENAME", "other.pt"):
            with self.assertRaisesRegex(ValueError, "changed its name"):
                STAGE.load_stage_49_control(data, directory)

    def test_budget_peer_is_authenticated_against_its_recorded_hash(self):
        directory = STAGE.DEFAULT_OUTPUT_DIRECTORY
        data = STAGE.CharacterData.from_file(
            STAGE.DEFAULT_COMPLETE_DATA_PATH, train_fraction=0.9
        )
        self.assertEqual(STAGE.C320_BUDGET_REF["owner"], "stage_46")
        self.assertEqual(STAGE.C320_BUDGET_REF["width"], 320)
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
            (endpoint.payload["step"], endpoint.payload["best_step"]), (50000, 49500)
        )
        changed = {**STAGE.C320_BUDGET_REF, "file_sha256": "0" * 64}
        with patch.object(STAGE, "C320_BUDGET_REF", changed):
            with self.assertRaisesRegex(ValueError, "hash changed"):
                STAGE.load_c320_budget_peer(data, directory)

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
