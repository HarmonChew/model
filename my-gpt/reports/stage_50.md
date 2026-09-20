# Stage 50 script output

- Script: `50_depth_revisit.py`
- Run time (seconds): `1761`
- Device: `cuda:0 (NVIDIA GeForce RTX 3090 Ti, foreign)`
- Peak GPU utilization: `98%`
- Peak GPU memory: `838 MiB`
- Exit code: `0`
- Run status: `ok`

## Stdout

```text
{
  "stage": 50,
  "protocol": "paired_architecture_native_depth",
  "training_executed": false,
  "device_mode": "foreign_device",
  "runtime": {
    "device_type": "cuda",
    "device_name": "NVIDIA GeForce RTX 3090 Ti",
    "torch_version": "2.14.0+cu130",
    "lineage_device_required": "NVIDIA GeForce RTX 3090",
    "lineage_device_verified": false,
    "foreign_device_authorized": true
  },
  "branches": [
    {
      "branch_key": "control",
      "depth": 6,
      "layer_contexts": [
        23,
        23,
        23,
        23,
        null,
        null
      ],
      "residual_dropout": 0.15,
      "architecture": {
        "n_embd": 256,
        "n_head": 4,
        "n_layer": 6,
        "head_size": 64,
        "feed_forward_width": 1024,
        "block_size": 256,
        "residual_dropout": 0.15,
        "layer_contexts": [
          23,
          23,
          23,
          23,
          null,
          null
        ],
        "topology": "LLLLGG",
        "vocabulary_size": 98,
        "local_layer_count": 4,
        "global_layer_count": 2,
        "global_layer_indices": [
          4,
          5
        ],
        "direct_attention_spans": [
          23,
          23,
          23,
          23,
          256,
          256
        ],
        "receptive_field_by_layer": [
          23,
          45,
          67,
          89,
          256,
          256
        ],
        "pre_global_receptive_field": 89,
        "first_global_reachability_layer": 5
      },
      "initialization": {
        "kind": "fresh_architecture_native",
        "origin_stage": 50,
        "step": 0,
        "seed": 1337,
        "device": "cpu",
        "state_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        "policy": "Both branches recreate their own architecture-native initialization from the same seed and corpus on the CPU. Residual dropout has no learnable parameters and consumes no initialization RNG. Depth does change the model, so byte-identical complete states are neither expected nor claimed; what is claimed and checked is that the treatment's six shared blocks, its two embedding tables, and its final LayerNorm are byte-identical to the control's, because the native construction draws in block order.",
        "depth": 6,
        "layer_contexts": [
          23,
          23,
          23,
          23,
          null,
          null
        ],
        "residual_dropout": 0.15,
        "historical_weights_loaded": false,
        "historical_optimizer_loaded": false,
        "optimizer_state_entries_at_initialization": 0,
        "parameter_count": 4850274,
        "block_parameter_count": 788992
      },
      "initial_streams": {
        "model_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        "optimizer_sha256": "c5ffcf6319b0bad4abbbeb62670b300e06483d88a13730c82ca9fb123eff7155",
        "training_generator_sha256": "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced",
        "torch_rng_sha256": "059ea222d9d04c0b43c3602c01097ad125978a3787153133bce80fd163d4b01b"
      }
    },
    {
      "branch_key": "treatment",
      "depth": 8,
      "layer_contexts": [
        23,
        23,
        23,
        23,
        23,
        23,
        null,
        null
      ],
      "residual_dropout": 0.15,
      "architecture": {
        "n_embd": 256,
        "n_head": 4,
        "n_layer": 8,
        "head_size": 64,
        "feed_forward_width": 1024,
        "block_size": 256,
        "residual_dropout": 0.15,
        "layer_contexts": [
          23,
          23,
          23,
          23,
          23,
          23,
          null,
          null
        ],
        "topology": "LLLLLLGG",
        "vocabulary_size": 98,
        "local_layer_count": 6,
        "global_layer_count": 2,
        "global_layer_indices": [
          6,
          7
        ],
        "direct_attention_spans": [
          23,
          23,
          23,
          23,
          23,
          23,
          256,
          256
        ],
        "receptive_field_by_layer": [
          23,
          45,
          67,
          89,
          111,
          133,
          256,
          256
        ],
        "pre_global_receptive_field": 133,
        "first_global_reachability_layer": 7
      },
      "initialization": {
        "kind": "fresh_architecture_native",
        "origin_stage": 50,
        "step": 0,
        "seed": 1337,
        "device": "cpu",
        "state_sha256": "6bef5bb6fa54e908ca06f64d299b1c3b35db7cf8676a8ade7d33c2b9cdb2f8c3",
        "policy": "Both branches recreate their own architecture-native initialization from the same seed and corpus on the CPU. Residual dropout has no learnable parameters and consumes no initialization RNG. Depth does change the model, so byte-identical complete states are neither expected nor claimed; what is claimed and checked is that the treatment's six shared blocks, its two embedding tables, and its final LayerNorm are byte-identical to the control's, because the native construction draws in block order.",
        "depth": 8,
        "layer_contexts": [
          23,
          23,
          23,
          23,
          23,
          23,
          null,
          null
        ],
        "residual_dropout": 0.15,
        "historical_weights_loaded": false,
        "historical_optimizer_loaded": false,
        "optimizer_state_entries_at_initialization": 0,
        "parameter_count": 6428258,
        "block_parameter_count": 788992
      },
      "initial_streams": {
        "model_sha256": "6bef5bb6fa54e908ca06f64d299b1c3b35db7cf8676a8ade7d33c2b9cdb2f8c3",
        "optimizer_sha256": "eba0ace8e4fc9148445e08dcfc712eaa374d815ab02a18df1d58c4e5f473f881",
        "training_generator_sha256": "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced",
        "torch_rng_sha256": "059ea222d9d04c0b43c3602c01097ad125978a3787153133bce80fd163d4b01b"
      }
    }
  ],
  "shared_initialization": {
    "control_key_count": 162,
    "treatment_key_count": 214,
    "treatment_is_a_key_superset": true,
    "identical_key_count": 152,
    "control_identical_keys_sha256": "24d2260f37d3af57a2f6c323524bc448ea6631bccc50079ebaf29b2c7ad05234",
    "treatment_identical_keys_sha256": "24d2260f37d3af57a2f6c323524bc448ea6631bccc50079ebaf29b2c7ad05234",
    "expected_differing": [
      "blocks.4.sa.heads.0.tril",
      "blocks.4.sa.heads.1.tril",
      "blocks.4.sa.heads.2.tril",
      "blocks.4.sa.heads.3.tril",
      "blocks.5.sa.heads.0.tril",
      "blocks.5.sa.heads.1.tril",
      "blocks.5.sa.heads.2.tril",
      "blocks.5.sa.heads.3.tril",
      "lm_head.weight",
      "lm_head.bias"
    ],
    "changed_mask_layers": [
      4,
      5
    ],
    "added_key_count": 52,
    "added_block_layers": [
      6,
      7
    ],
    "added_parameter_count": 1577984,
    "per_added_block_parameter_count": 788992,
    "control_parameter_count": 4850274,
    "treatment_parameter_count": 6428258,
    "control_masks_sha256": "6d9c302adb831ca02b318605f394b6cf7b078e710428cbadc291e25710638076",
    "treatment_masks_sha256": "2e8fea590a35c4234f3d8d4e213cae7578ae29b59a8428a91cc7de021068044a"
  },
  "control_initialization_equivalence": {
    "reference": "Stage 49 recorded C256 L=6 native initialization",
    "state_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
    "stage_46_factory_identical": true,
    "stage_30_factory_identical": true,
    "identical": true
  },
  "treatment_initialization_equivalence": {
    "reference": "Stage 50 recorded C256 L=8 native initialization",
    "state_sha256": "6bef5bb6fa54e908ca06f64d299b1c3b35db7cf8676a8ade7d33c2b9cdb2f8c3",
    "stage_30_factory_identical": true,
    "identical": true
  },
  "updates_per_branch": 50000,
  "target_presentations_per_branch": 102400000,
  "parameter_counts": {
    "control": 4850274,
    "treatment": 6428258,
    "added": 1577984,
    "added_fraction": 0.3253391457884647,
    "stored_c320_peer": 7537378
  },
  "learning_rate_schedule": [
    {
      "start_update": 0,
      "end_update_exclusive": 10000,
      "learning_rate": 0.001
    },
    {
      "start_update": 10000,
      "end_update_exclusive": 13000,
      "learning_rate": 0.0003
    },
    {
      "start_update": 13000,
      "end_update_exclusive": 18000,
      "learning_rate": 0.0001
    },
    {
      "start_update": 18000,
      "end_update_exclusive": 50000,
      "learning_rate": 3e-05
    }
  ],
  "stored_peers": {
    "stage_49_control": {
      "key": "c256_l6_lineage",
      "path": "my-gpt/checkpoints/stage_49_control_c256_p015_step_50000_final_checkpoint.pt",
      "file_sha256": "8aaf70ae60e546168a976ecc640a6b1dac3044d135ccf7f9d822dfa5598affec",
      "width": 256,
      "depth": 6,
      "dropout": 0.15,
      "parameter_count": 4850274,
      "best_step": 48500,
      "role": "the champion the trained control must reproduce"
    },
    "c320_budget_peer": {
      "key": "c320_l6",
      "path": "my-gpt/checkpoints/stage_46_treatment_c320_p015_step_50000_final_checkpoint.pt",
      "file_sha256": "e47016371b6a588448dbaa0bad6d05ff8564b7b21c6f6da81f20bbb56e916197",
      "width": 320,
      "depth": 6,
      "dropout": 0.15,
      "parameter_count": 7537378,
      "best_step": 49500,
      "role": "the wider spend the depth treatment is scored against"
    }
  },
  "data_fingerprint": "c6cbdd246799ad495f7813e5d20afa52c03e6c339fe7dbe15912fd81a45d2b48",
  "initialization_policy": "Both branches recreate their own architecture-native initialization from the same seed and corpus on the CPU. Residual dropout has no learnable parameters and consumes no initialization RNG. Depth does change the model, so byte-identical complete states are neither expected nor claimed; what is claimed and checked is that the treatment's six shared blocks, its two embedding tables, and its final LayerNorm are byte-identical to the control's, because the native construction draws in block order.",
  "randomness_policy": "Both branches begin from the same CPU and accelerator training RNG states and consume the same sampled training panels, because the sampler depends only on batch shape and its seed. The treatment applies two more dropout sites per update, so its terminal Torch RNG state is expected to differ from the control's; neither identical dropout masks nor identical terminal Torch RNG states are claimed.",
  "historical_lineage": {
    "stage_41_authenticated_44k_ancestor_sha256": "666c9e72345399cf63b412128a1427d1c778d58794473c0e8503516d2235db36",
    "stage_44_empirical_best": {
      "branch": "treatment_c256",
      "step": 50000,
      "best_step": 48500,
      "best_checkpoint_sha256": "65af019d75a84cbef7cb589a50c7474d0324c669c484d9be629ec188811bfea3",
      "final_checkpoint_sha256": "c9e0092cb1986885f8cef948b703bb4fbf18ae0714e4e3d062fecc337081b247",
      "fixed_panel_train_loss": 1.0690209597349167,
      "fixed_panel_validation_loss": 1.3538347363471985
    },
    "stage_45_c320_under_regularized_endpoint": {
      "branch": "treatment_c320",
      "dropout": 0.1,
      "step": 50000,
      "best_step": 39000,
      "best_checkpoint_sha256": "dbcaffe6a4d186c3c3d8c99fab215aafd5a97e178f0c50ff1d1e0b7140da4c86",
      "final_checkpoint_sha256": "a6f908c18d02ea3d7a9060d4fb7886320bff41eea06a0e97e2196ea57f34984b",
      "fixed_panel_train_loss": 1.0492555099725722,
      "fixed_panel_validation_loss": 1.364953727722168,
      "role": "the state this stage's p=0.10 control must re-derive from zero"
    },
    "stage_45_c256_champion_endpoint": {
      "branch": "control_c256",
      "dropout": 0.1,
      "step": 50000,
      "best_step": 48500,
      "final_checkpoint_sha256": "e58ce4532a3ce0623ea622b91ea7ee04232c8d1f12d8b2c53591c7efec299e9f",
      "fixed_panel_train_loss": 1.0690209597349167,
      "fixed_panel_validation_loss": 1.3538347363471985,
      "role": "the champion state this stage's p=0.10 control must re-derive"
    },
    "stage_46_c320_p015_endpoint": {
      "branch": "treatment_c320",
      "dropout": 0.15,
      "step": 50000,
      "final_checkpoint_sha256": "e47016371b6a588448dbaa0bad6d05ff8564b7b21c6f6da81f20bbb56e916197",
      "parameter_count": 7537378,
      "role": "the wider budget this stage's deeper treatment is scored against"
    },
    "stage_47_c320_p020_endpoint": {
      "branch": "treatment_c320",
      "dropout": 0.2,
      "step": 50000,
      "best_step": 49500,
      "best_checkpoint_sha256": "4e7a7b45203cdd842f9c812821109d39ddedefe6371d93f59149967b9bcf27da",
      "final_checkpoint_sha256": "7289527c6078b92f52abc98f39c3871c30aed7fa56a2d8a2901d5f4f035b170f",
      "fixed_panel_train_loss": 1.085713002681732,
      "fixed_panel_validation_loss": 1.3601680147647857,
      "role": "read-only champion shootout peer"
    },
    "role": "historical reference only; none is a Stage 50 initialization",
    "stage_48_c256_control_p010_endpoint": {
      "branch": "control_c256",
      "dropout": 0.1,
      "step": 50000,
      "best_step": 48500,
      "final_checkpoint_sha256": "8c817fd5ea8a557e60f693a577cad68bbd9d483402d8e90fe343b23590c8090e",
      "fixed_panel_train_loss": 1.0690209597349167,
      "fixed_panel_validation_loss": 1.3538347363471985,
      "role": "the C256 endpoint Stage 48 displaced; not read by this stage"
    },
    "stage_48_c256_treatment_p015_endpoint": {
      "branch": "treatment_c256",
      "dropout": 0.15,
      "step": 50000,
      "best_step": 48500,
      "best_checkpoint_sha256": "62d6b5ee7d9009cafd48561e5622c9a1e7b3254e3d26fbfc57c280662ce723ab",
      "final_checkpoint_sha256": "04952aeeab517f2a31ba4209737d4f8d495d40fb84aca8c9b5f3b58bb44f98b4",
      "fixed_panel_train_loss": 1.087766051888466,
      "fixed_panel_validation_loss": 1.3507258701324463,
      "role": "the C256 p=0.15 state this stage's control must re-derive"
    },
    "stage_49_c256_p015_champion_endpoint": {
      "branch": "control_c256",
      "depth": 6,
      "dropout": 0.15,
      "step": 50000,
      "best_step": 48500,
      "best_checkpoint_sha256": "a73a9912e150a2ba56aa29568f1abef3afd466dd8bb36aeccf973e9effa51c18",
      "final_checkpoint_sha256": "8aaf70ae60e546168a976ecc640a6b1dac3044d135ccf7f9d822dfa5598affec",
      "fixed_panel_train_loss": 1.087766051888466,
      "fixed_panel_validation_loss": 1.3507258701324463,
      "role": "the champion this stage's control must re-derive and the depth step is measured from"
    },
    "stage_49_c256_p020_endpoint": {
      "branch": "treatment_c256",
      "dropout": 0.2,
      "step": 50000,
      "final_checkpoint_sha256": "cff587a1b79b03e946702947cd2fe6811b08c44c60488c02a43c97bc7fac5105",
      "role": "the over-regularized C256 endpoint Stage 49 displaced"
    }
  }
}
Stage 50 on cuda:0 (foreign_device): native C256 L=6 and L=8 at p=0.15, 50,000 updates per branch
  control C256 L6 p0.15 step     0 | lr 0.001 | train 4.7088 | val 4.6970 | best 4.6970 @ 0
  control C256 L6 p0.15 step   500 | lr 0.001 | train 1.9425 | val 2.0012 | best 2.0012 @ 500
  control C256 L6 p0.15 step  1000 | lr 0.001 | train 1.6885 | val 1.7776 | best 1.7776 @ 1000
  control C256 L6 p0.15 step  1500 | lr 0.001 | train 1.5741 | val 1.6984 | best 1.6984 @ 1500
  control C256 L6 p0.15 step  2000 | lr 0.001 | train 1.4949 | val 1.6422 | best 1.6422 @ 2000
  control C256 L6 p0.15 step  2500 | lr 0.001 | train 1.4423 | val 1.6047 | best 1.6047 @ 2500
  control C256 L6 p0.15 step  3000 | lr 0.001 | train 1.4063 | val 1.5865 | best 1.5865 @ 3000
  control C256 L6 p0.15 step  3500 | lr 0.001 | train 1.3778 | val 1.5499 | best 1.5499 @ 3500
  control C256 L6 p0.15 step  4000 | lr 0.001 | train 1.3516 | val 1.5430 | best 1.5430 @ 4000
  control C256 L6 p0.15 step  4500 | lr 0.001 | train 1.3307 | val 1.5216 | best 1.5216 @ 4500
  control C256 L6 p0.15 step  5000 | lr 0.001 | train 1.3194 | val 1.5119 | best 1.5119 @ 5000
  control C256 L6 p0.15 step  5500 | lr 0.001 | train 1.2986 | val 1.4944 | best 1.4944 @ 5500
  control C256 L6 p0.15 step  6000 | lr 0.001 | train 1.2890 | val 1.4909 | best 1.4909 @ 6000
  control C256 L6 p0.15 step  6500 | lr 0.001 | train 1.2746 | val 1.4735 | best 1.4735 @ 6500
  control C256 L6 p0.15 step  7000 | lr 0.001 | train 1.2626 | val 1.4631 | best 1.4631 @ 7000
  control C256 L6 p0.15 step  7500 | lr 0.001 | train 1.2517 | val 1.4525 | best 1.4525 @ 7500
  control C256 L6 p0.15 step  8000 | lr 0.001 | train 1.2492 | val 1.4607 | best 1.4525 @ 7500
  control C256 L6 p0.15 step  8500 | lr 0.001 | train 1.2388 | val 1.4461 | best 1.4461 @ 8500
  control C256 L6 p0.15 step  9000 | lr 0.001 | train 1.2302 | val 1.4393 | best 1.4393 @ 9000
  control C256 L6 p0.15 step  9500 | lr 0.001 | train 1.2235 | val 1.4335 | best 1.4335 @ 9500
  control C256 L6 p0.15 step 10000 | lr 0.001 | train 1.2202 | val 1.4365 | best 1.4335 @ 9500
  control C256 L6 p0.15 step 10500 | lr 0.0003 | train 1.1779 | val 1.3973 | best 1.3973 @ 10500
  control C256 L6 p0.15 step 11000 | lr 0.0003 | train 1.1698 | val 1.3969 | best 1.3969 @ 11000
  control C256 L6 p0.15 step 11500 | lr 0.0003 | train 1.1630 | val 1.3911 | best 1.3911 @ 11500
  control C256 L6 p0.15 step 12000 | lr 0.0003 | train 1.1570 | val 1.3871 | best 1.3871 @ 12000
  control C256 L6 p0.15 step 12500 | lr 0.0003 | train 1.1523 | val 1.3861 | best 1.3861 @ 12500
  control C256 L6 p0.15 step 13000 | lr 0.0003 | train 1.1502 | val 1.3853 | best 1.3853 @ 13000
  control C256 L6 p0.15 step 13500 | lr 0.0001 | train 1.1381 | val 1.3749 | best 1.3749 @ 13500
  control C256 L6 p0.15 step 14000 | lr 0.0001 | train 1.1350 | val 1.3736 | best 1.3736 @ 14000
  control C256 L6 p0.15 step 14500 | lr 0.0001 | train 1.1317 | val 1.3740 | best 1.3736 @ 14000
  control C256 L6 p0.15 step 15000 | lr 0.0001 | train 1.1300 | val 1.3736 | best 1.3736 @ 15000
  control C256 L6 p0.15 step 15500 | lr 0.0001 | train 1.1283 | val 1.3720 | best 1.3720 @ 15500
  control C256 L6 p0.15 step 16000 | lr 0.0001 | train 1.1257 | val 1.3704 | best 1.3704 @ 16000
  control C256 L6 p0.15 step 16500 | lr 0.0001 | train 1.1242 | val 1.3705 | best 1.3704 @ 16000
  control C256 L6 p0.15 step 17000 | lr 0.0001 | train 1.1224 | val 1.3694 | best 1.3694 @ 17000
  control C256 L6 p0.15 step 17500 | lr 0.0001 | train 1.1207 | val 1.3681 | best 1.3681 @ 17500
  control C256 L6 p0.15 step 18000 | lr 0.0001 | train 1.1200 | val 1.3682 | best 1.3681 @ 17500
  control C256 L6 p0.15 step 18500 | lr 3e-05 | train 1.1174 | val 1.3658 | best 1.3658 @ 18500
  control C256 L6 p0.15 step 19000 | lr 3e-05 | train 1.1159 | val 1.3646 | best 1.3646 @ 19000
  control C256 L6 p0.15 step 19500 | lr 3e-05 | train 1.1149 | val 1.3633 | best 1.3633 @ 19500
  control C256 L6 p0.15 step 20000 | lr 3e-05 | train 1.1142 | val 1.3636 | best 1.3633 @ 19500
  control C256 L6 p0.15 step 20500 | lr 3e-05 | train 1.1136 | val 1.3629 | best 1.3629 @ 20500
  control C256 L6 p0.15 step 21000 | lr 3e-05 | train 1.1129 | val 1.3635 | best 1.3629 @ 20500
  control C256 L6 p0.15 step 21500 | lr 3e-05 | train 1.1124 | val 1.3635 | best 1.3629 @ 20500
  control C256 L6 p0.15 step 22000 | lr 3e-05 | train 1.1120 | val 1.3629 | best 1.3629 @ 22000
  control C256 L6 p0.15 step 22500 | lr 3e-05 | train 1.1115 | val 1.3629 | best 1.3629 @ 22000
  control C256 L6 p0.15 step 23000 | lr 3e-05 | train 1.1107 | val 1.3628 | best 1.3628 @ 23000
  control C256 L6 p0.15 step 23500 | lr 3e-05 | train 1.1100 | val 1.3610 | best 1.3610 @ 23500
  control C256 L6 p0.15 step 24000 | lr 3e-05 | train 1.1096 | val 1.3614 | best 1.3610 @ 23500
  control C256 L6 p0.15 step 24500 | lr 3e-05 | train 1.1089 | val 1.3608 | best 1.3608 @ 24500
  control C256 L6 p0.15 step 25000 | lr 3e-05 | train 1.1083 | val 1.3611 | best 1.3608 @ 24500
  control C256 L6 p0.15 step 25500 | lr 3e-05 | train 1.1079 | val 1.3604 | best 1.3604 @ 25500
  control C256 L6 p0.15 step 26000 | lr 3e-05 | train 1.1076 | val 1.3605 | best 1.3604 @ 25500
  control C256 L6 p0.15 step 26500 | lr 3e-05 | train 1.1070 | val 1.3597 | best 1.3597 @ 26500
  control C256 L6 p0.15 step 27000 | lr 3e-05 | train 1.1066 | val 1.3601 | best 1.3597 @ 26500
  control C256 L6 p0.15 step 27500 | lr 3e-05 | train 1.1059 | val 1.3594 | best 1.3594 @ 27500
  control C256 L6 p0.15 step 28000 | lr 3e-05 | train 1.1059 | val 1.3600 | best 1.3594 @ 27500
  control C256 L6 p0.15 step 28500 | lr 3e-05 | train 1.1049 | val 1.3588 | best 1.3588 @ 28500
  control C256 L6 p0.15 step 29000 | lr 3e-05 | train 1.1050 | val 1.3592 | best 1.3588 @ 28500
  control C256 L6 p0.15 step 29500 | lr 3e-05 | train 1.1043 | val 1.3580 | best 1.3580 @ 29500
  control C256 L6 p0.15 step 30000 | lr 3e-05 | train 1.1039 | val 1.3581 | best 1.3580 @ 29500
  control C256 L6 p0.15 step 30500 | lr 3e-05 | train 1.1037 | val 1.3582 | best 1.3580 @ 29500
  control C256 L6 p0.15 step 31000 | lr 3e-05 | train 1.1033 | val 1.3578 | best 1.3578 @ 31000
  control C256 L6 p0.15 step 31500 | lr 3e-05 | train 1.1027 | val 1.3581 | best 1.3578 @ 31000
  control C256 L6 p0.15 step 32000 | lr 3e-05 | train 1.1019 | val 1.3580 | best 1.3578 @ 31000
  control C256 L6 p0.15 step 32500 | lr 3e-05 | train 1.1017 | val 1.3572 | best 1.3572 @ 32500
  control C256 L6 p0.15 step 33000 | lr 3e-05 | train 1.1012 | val 1.3583 | best 1.3572 @ 32500
  control C256 L6 p0.15 step 33500 | lr 3e-05 | train 1.1009 | val 1.3573 | best 1.3572 @ 32500
  control C256 L6 p0.15 step 34000 | lr 3e-05 | train 1.1005 | val 1.3559 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 34500 | lr 3e-05 | train 1.1001 | val 1.3572 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 35000 | lr 3e-05 | train 1.0995 | val 1.3575 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 35500 | lr 3e-05 | train 1.0989 | val 1.3581 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 36000 | lr 3e-05 | train 1.0988 | val 1.3563 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 36500 | lr 3e-05 | train 1.0987 | val 1.3572 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 37000 | lr 3e-05 | train 1.0982 | val 1.3568 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 37500 | lr 3e-05 | train 1.0978 | val 1.3563 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 38000 | lr 3e-05 | train 1.0973 | val 1.3567 | best 1.3559 @ 34000
  control C256 L6 p0.15 step 38500 | lr 3e-05 | train 1.0969 | val 1.3555 | best 1.3555 @ 38500
  control C256 L6 p0.15 step 39000 | lr 3e-05 | train 1.0967 | val 1.3558 | best 1.3555 @ 38500
  control C256 L6 p0.15 step 39500 | lr 3e-05 | train 1.0962 | val 1.3564 | best 1.3555 @ 38500
  control C256 L6 p0.15 step 40000 | lr 3e-05 | train 1.0958 | val 1.3557 | best 1.3555 @ 38500
  control C256 L6 p0.15 step 40500 | lr 3e-05 | train 1.0955 | val 1.3562 | best 1.3555 @ 38500
  control C256 L6 p0.15 step 41000 | lr 3e-05 | train 1.0946 | val 1.3557 | best 1.3555 @ 38500
  control C256 L6 p0.15 step 41500 | lr 3e-05 | train 1.0944 | val 1.3556 | best 1.3555 @ 38500
  control C256 L6 p0.15 step 42000 | lr 3e-05 | train 1.0938 | val 1.3549 | best 1.3549 @ 42000
  control C256 L6 p0.15 step 42500 | lr 3e-05 | train 1.0934 | val 1.3541 | best 1.3541 @ 42500
  control C256 L6 p0.15 step 43000 | lr 3e-05 | train 1.0934 | val 1.3551 | best 1.3541 @ 42500
  control C256 L6 p0.15 step 43500 | lr 3e-05 | train 1.0928 | val 1.3544 | best 1.3541 @ 42500
  control C256 L6 p0.15 step 44000 | lr 3e-05 | train 1.0928 | val 1.3544 | best 1.3541 @ 42500
  control C256 L6 p0.15 step 44500 | lr 3e-05 | train 1.0921 | val 1.3545 | best 1.3541 @ 42500
  control C256 L6 p0.15 step 45000 | lr 3e-05 | train 1.0919 | val 1.3548 | best 1.3541 @ 42500
  control C256 L6 p0.15 step 45500 | lr 3e-05 | train 1.0915 | val 1.3538 | best 1.3538 @ 45500
  control C256 L6 p0.15 step 46000 | lr 3e-05 | train 1.0913 | val 1.3546 | best 1.3538 @ 45500
  control C256 L6 p0.15 step 46500 | lr 3e-05 | train 1.0909 | val 1.3536 | best 1.3536 @ 46500
  control C256 L6 p0.15 step 47000 | lr 3e-05 | train 1.0904 | val 1.3542 | best 1.3536 @ 46500
  control C256 L6 p0.15 step 47500 | lr 3e-05 | train 1.0900 | val 1.3531 | best 1.3531 @ 47500
  control C256 L6 p0.15 step 48000 | lr 3e-05 | train 1.0897 | val 1.3534 | best 1.3531 @ 47500
  control C256 L6 p0.15 step 48500 | lr 3e-05 | train 1.0896 | val 1.3522 | best 1.3522 @ 48500
  control C256 L6 p0.15 step 49000 | lr 3e-05 | train 1.0894 | val 1.3539 | best 1.3522 @ 48500
  control C256 L6 p0.15 step 49500 | lr 3e-05 | train 1.0888 | val 1.3523 | best 1.3522 @ 48500
  control C256 L6 p0.15 step 50000 | lr 3e-05 | train 1.0886 | val 1.3530 | best 1.3522 @ 48500
  the control was not trained on the lineage device, so byte-identical reproduction of the stored champion is not claimed; the stored endpoint is scored as the panel's device-shift row
  treatment C256 L8 p0.15 step     0 | lr 0.001 | train 4.7449 | val 4.7442 | best 4.7442 @ 0
  treatment C256 L8 p0.15 step   500 | lr 0.001 | train 1.9450 | val 2.0065 | best 2.0065 @ 500
  treatment C256 L8 p0.15 step  1000 | lr 0.001 | train 1.6829 | val 1.7778 | best 1.7778 @ 1000
  treatment C256 L8 p0.15 step  1500 | lr 0.001 | train 1.5657 | val 1.7032 | best 1.7032 @ 1500
  treatment C256 L8 p0.15 step  2000 | lr 0.001 | train 1.4915 | val 1.6470 | best 1.6470 @ 2000
  treatment C256 L8 p0.15 step  2500 | lr 0.001 | train 1.4364 | val 1.6026 | best 1.6026 @ 2500
  treatment C256 L8 p0.15 step  3000 | lr 0.001 | train 1.3980 | val 1.5797 | best 1.5797 @ 3000
  treatment C256 L8 p0.15 step  3500 | lr 0.001 | train 1.3672 | val 1.5380 | best 1.5380 @ 3500
  treatment C256 L8 p0.15 step  4000 | lr 0.001 | train 1.3403 | val 1.5289 | best 1.5289 @ 4000
  treatment C256 L8 p0.15 step  4500 | lr 0.001 | train 1.3219 | val 1.5134 | best 1.5134 @ 4500
  treatment C256 L8 p0.15 step  5000 | lr 0.001 | train 1.3067 | val 1.5036 | best 1.5036 @ 5000
  treatment C256 L8 p0.15 step  5500 | lr 0.001 | train 1.2890 | val 1.4878 | best 1.4878 @ 5500
  treatment C256 L8 p0.15 step  6000 | lr 0.001 | train 1.2761 | val 1.4808 | best 1.4808 @ 6000
  treatment C256 L8 p0.15 step  6500 | lr 0.001 | train 1.2628 | val 1.4686 | best 1.4686 @ 6500
  treatment C256 L8 p0.15 step  7000 | lr 0.001 | train 1.2506 | val 1.4618 | best 1.4618 @ 7000
  treatment C256 L8 p0.15 step  7500 | lr 0.001 | train 1.2401 | val 1.4540 | best 1.4540 @ 7500
  treatment C256 L8 p0.15 step  8000 | lr 0.001 | train 1.2339 | val 1.4521 | best 1.4521 @ 8000
  treatment C256 L8 p0.15 step  8500 | lr 0.001 | train 1.2267 | val 1.4420 | best 1.4420 @ 8500
  treatment C256 L8 p0.15 step  9000 | lr 0.001 | train 1.2173 | val 1.4357 | best 1.4357 @ 9000
  treatment C256 L8 p0.15 step  9500 | lr 0.001 | train 1.2111 | val 1.4326 | best 1.4326 @ 9500
  treatment C256 L8 p0.15 step 10000 | lr 0.001 | train 1.2039 | val 1.4305 | best 1.4305 @ 10000
  treatment C256 L8 p0.15 step 10500 | lr 0.0003 | train 1.1616 | val 1.3952 | best 1.3952 @ 10500
  treatment C256 L8 p0.15 step 11000 | lr 0.0003 | train 1.1518 | val 1.3916 | best 1.3916 @ 11000
  treatment C256 L8 p0.15 step 11500 | lr 0.0003 | train 1.1442 | val 1.3886 | best 1.3886 @ 11500
  treatment C256 L8 p0.15 step 12000 | lr 0.0003 | train 1.1383 | val 1.3855 | best 1.3855 @ 12000
  treatment C256 L8 p0.15 step 12500 | lr 0.0003 | train 1.1327 | val 1.3856 | best 1.3855 @ 12000
  treatment C256 L8 p0.15 step 13000 | lr 0.0003 | train 1.1315 | val 1.3843 | best 1.3843 @ 13000
  treatment C256 L8 p0.15 step 13500 | lr 0.0001 | train 1.1179 | val 1.3738 | best 1.3738 @ 13500
  treatment C256 L8 p0.15 step 14000 | lr 0.0001 | train 1.1142 | val 1.3706 | best 1.3706 @ 14000
  treatment C256 L8 p0.15 step 14500 | lr 0.0001 | train 1.1107 | val 1.3712 | best 1.3706 @ 14000
  treatment C256 L8 p0.15 step 15000 | lr 0.0001 | train 1.1089 | val 1.3710 | best 1.3706 @ 14000
  treatment C256 L8 p0.15 step 15500 | lr 0.0001 | train 1.1072 | val 1.3702 | best 1.3702 @ 15500
  treatment C256 L8 p0.15 step 16000 | lr 0.0001 | train 1.1048 | val 1.3694 | best 1.3694 @ 16000
  treatment C256 L8 p0.15 step 16500 | lr 0.0001 | train 1.1027 | val 1.3690 | best 1.3690 @ 16500
  treatment C256 L8 p0.15 step 17000 | lr 0.0001 | train 1.1003 | val 1.3684 | best 1.3684 @ 17000
  treatment C256 L8 p0.15 step 17500 | lr 0.0001 | train 1.0985 | val 1.3663 | best 1.3663 @ 17500
  treatment C256 L8 p0.15 step 18000 | lr 0.0001 | train 1.0971 | val 1.3641 | best 1.3641 @ 18000
  treatment C256 L8 p0.15 step 18500 | lr 3e-05 | train 1.0943 | val 1.3625 | best 1.3625 @ 18500
  treatment C256 L8 p0.15 step 19000 | lr 3e-05 | train 1.0928 | val 1.3629 | best 1.3625 @ 18500
  treatment C256 L8 p0.15 step 19500 | lr 3e-05 | train 1.0917 | val 1.3613 | best 1.3613 @ 19500
  treatment C256 L8 p0.15 step 20000 | lr 3e-05 | train 1.0909 | val 1.3609 | best 1.3609 @ 20000
  treatment C256 L8 p0.15 step 20500 | lr 3e-05 | train 1.0906 | val 1.3605 | best 1.3605 @ 20500
  treatment C256 L8 p0.15 step 21000 | lr 3e-05 | train 1.0897 | val 1.3612 | best 1.3605 @ 20500
  treatment C256 L8 p0.15 step 21500 | lr 3e-05 | train 1.0893 | val 1.3611 | best 1.3605 @ 20500
  treatment C256 L8 p0.15 step 22000 | lr 3e-05 | train 1.0884 | val 1.3602 | best 1.3602 @ 22000
  treatment C256 L8 p0.15 step 22500 | lr 3e-05 | train 1.0879 | val 1.3608 | best 1.3602 @ 22000
  treatment C256 L8 p0.15 step 23000 | lr 3e-05 | train 1.0871 | val 1.3615 | best 1.3602 @ 22000
  treatment C256 L8 p0.15 step 23500 | lr 3e-05 | train 1.0863 | val 1.3593 | best 1.3593 @ 23500
  treatment C256 L8 p0.15 step 24000 | lr 3e-05 | train 1.0857 | val 1.3595 | best 1.3593 @ 23500
  treatment C256 L8 p0.15 step 24500 | lr 3e-05 | train 1.0850 | val 1.3601 | best 1.3593 @ 23500
  treatment C256 L8 p0.15 step 25000 | lr 3e-05 | train 1.0846 | val 1.3594 | best 1.3593 @ 23500
  treatment C256 L8 p0.15 step 25500 | lr 3e-05 | train 1.0841 | val 1.3597 | best 1.3593 @ 23500
  treatment C256 L8 p0.15 step 26000 | lr 3e-05 | train 1.0836 | val 1.3592 | best 1.3592 @ 26000
  treatment C256 L8 p0.15 step 26500 | lr 3e-05 | train 1.0831 | val 1.3577 | best 1.3577 @ 26500
  treatment C256 L8 p0.15 step 27000 | lr 3e-05 | train 1.0826 | val 1.3586 | best 1.3577 @ 26500
  treatment C256 L8 p0.15 step 27500 | lr 3e-05 | train 1.0819 | val 1.3579 | best 1.3577 @ 26500
  treatment C256 L8 p0.15 step 28000 | lr 3e-05 | train 1.0817 | val 1.3585 | best 1.3577 @ 26500
  treatment C256 L8 p0.15 step 28500 | lr 3e-05 | train 1.0808 | val 1.3576 | best 1.3576 @ 28500
  treatment C256 L8 p0.15 step 29000 | lr 3e-05 | train 1.0806 | val 1.3572 | best 1.3572 @ 29000
  treatment C256 L8 p0.15 step 29500 | lr 3e-05 | train 1.0799 | val 1.3571 | best 1.3571 @ 29500
  treatment C256 L8 p0.15 step 30000 | lr 3e-05 | train 1.0796 | val 1.3568 | best 1.3568 @ 30000
  treatment C256 L8 p0.15 step 30500 | lr 3e-05 | train 1.0794 | val 1.3579 | best 1.3568 @ 30000
  treatment C256 L8 p0.15 step 31000 | lr 3e-05 | train 1.0789 | val 1.3574 | best 1.3568 @ 30000
  treatment C256 L8 p0.15 step 31500 | lr 3e-05 | train 1.0781 | val 1.3575 | best 1.3568 @ 30000
  treatment C256 L8 p0.15 step 32000 | lr 3e-05 | train 1.0776 | val 1.3564 | best 1.3564 @ 32000
  treatment C256 L8 p0.15 step 32500 | lr 3e-05 | train 1.0771 | val 1.3555 | best 1.3555 @ 32500
  treatment C256 L8 p0.15 step 33000 | lr 3e-05 | train 1.0762 | val 1.3563 | best 1.3555 @ 32500
  treatment C256 L8 p0.15 step 33500 | lr 3e-05 | train 1.0760 | val 1.3552 | best 1.3552 @ 33500
  treatment C256 L8 p0.15 step 34000 | lr 3e-05 | train 1.0755 | val 1.3545 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 34500 | lr 3e-05 | train 1.0752 | val 1.3550 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 35000 | lr 3e-05 | train 1.0743 | val 1.3566 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 35500 | lr 3e-05 | train 1.0735 | val 1.3560 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 36000 | lr 3e-05 | train 1.0733 | val 1.3550 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 36500 | lr 3e-05 | train 1.0731 | val 1.3550 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 37000 | lr 3e-05 | train 1.0725 | val 1.3550 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 37500 | lr 3e-05 | train 1.0719 | val 1.3553 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 38000 | lr 3e-05 | train 1.0713 | val 1.3552 | best 1.3545 @ 34000
  treatment C256 L8 p0.15 step 38500 | lr 3e-05 | train 1.0709 | val 1.3535 | best 1.3535 @ 38500
  treatment C256 L8 p0.15 step 39000 | lr 3e-05 | train 1.0706 | val 1.3539 | best 1.3535 @ 38500
  treatment C256 L8 p0.15 step 39500 | lr 3e-05 | train 1.0700 | val 1.3538 | best 1.3535 @ 38500
  treatment C256 L8 p0.15 step 40000 | lr 3e-05 | train 1.0696 | val 1.3547 | best 1.3535 @ 38500
  treatment C256 L8 p0.15 step 40500 | lr 3e-05 | train 1.0693 | val 1.3549 | best 1.3535 @ 38500
  treatment C256 L8 p0.15 step 41000 | lr 3e-05 | train 1.0685 | val 1.3541 | best 1.3535 @ 38500
  treatment C256 L8 p0.15 step 41500 | lr 3e-05 | train 1.0684 | val 1.3538 | best 1.3535 @ 38500
  treatment C256 L8 p0.15 step 42000 | lr 3e-05 | train 1.0675 | val 1.3542 | best 1.3535 @ 38500
  treatment C256 L8 p0.15 step 42500 | lr 3e-05 | train 1.0669 | val 1.3531 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 43000 | lr 3e-05 | train 1.0668 | val 1.3538 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 43500 | lr 3e-05 | train 1.0661 | val 1.3536 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 44000 | lr 3e-05 | train 1.0661 | val 1.3540 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 44500 | lr 3e-05 | train 1.0656 | val 1.3532 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 45000 | lr 3e-05 | train 1.0650 | val 1.3532 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 45500 | lr 3e-05 | train 1.0648 | val 1.3536 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 46000 | lr 3e-05 | train 1.0641 | val 1.3539 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 46500 | lr 3e-05 | train 1.0640 | val 1.3545 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 47000 | lr 3e-05 | train 1.0634 | val 1.3554 | best 1.3531 @ 42500
  treatment C256 L8 p0.15 step 47500 | lr 3e-05 | train 1.0629 | val 1.3527 | best 1.3527 @ 47500
  treatment C256 L8 p0.15 step 48000 | lr 3e-05 | train 1.0625 | val 1.3535 | best 1.3527 @ 47500
  treatment C256 L8 p0.15 step 48500 | lr 3e-05 | train 1.0623 | val 1.3525 | best 1.3525 @ 48500
  treatment C256 L8 p0.15 step 49000 | lr 3e-05 | train 1.0619 | val 1.3538 | best 1.3525 @ 48500
  treatment C256 L8 p0.15 step 49500 | lr 3e-05 | train 1.0615 | val 1.3515 | best 1.3515 @ 49500
  treatment C256 L8 p0.15 step 50000 | lr 3e-05 | train 1.0612 | val 1.3525 | best 1.3515 @ 49500
  C256 L=6 p=0.15 @ 50000: 1.3392857
  C256 L=8 p=0.15 @ 50000: 1.3376945
  paired candidate - baseline: -0.0015912; SE 0.0006339; 95% panel interval [-0.0028366, -0.0003458]
  1. C256 L=6 p=0.15 @ 50000 [stage 49 stored] [4,850,274 params]: 1.3481738 (SE 0.0031616)
  2. C256 L=8 p=0.15 @ 50000 [stage 50 fresh] [6,428,258 params]: 1.3487511 (SE 0.0033527)
  3. C256 L=6 p=0.15 @ 50000 [stage 50 fresh] [4,850,274 params]: 1.3489730 (SE 0.0032444)
  4. C320 L=6 p=0.15 @ 50000 [stage 46 stored] [7,537,378 params]: 1.3563111 (SE 0.0034741)
  C256 L=6 p=0.15 @ 50000 [stage 50 fresh] - C256 L=8 p=0.15 @ 50000 [stage 50 fresh]: +0.0002220; SE 0.0006614; 95% panel interval [-0.0010775, +0.0015214]
  C256 L=6 p=0.15 @ 50000 [stage 50 fresh] - C320 L=6 p=0.15 @ 50000 [stage 46 stored]: -0.0073381; SE 0.0008185; 95% panel interval [-0.0089461, -0.0057300]
  C256 L=6 p=0.15 @ 50000 [stage 50 fresh] - C256 L=6 p=0.15 @ 50000 [stage 49 stored]: +0.0007993; SE 0.0005694; 95% panel interval [-0.0003195, +0.0019180]
  C256 L=8 p=0.15 @ 50000 [stage 50 fresh] - C320 L=6 p=0.15 @ 50000 [stage 46 stored]: -0.0075600; SE 0.0007238; 95% panel interval [-0.0089820, -0.0061380]
  C256 L=8 p=0.15 @ 50000 [stage 50 fresh] - C256 L=6 p=0.15 @ 50000 [stage 49 stored]: +0.0005773; SE 0.0007027; 95% panel interval [-0.0008034, +0.0019579]
  C320 L=6 p=0.15 @ 50000 [stage 46 stored] - C256 L=6 p=0.15 @ 50000 [stage 49 stored]: +0.0081373; SE 0.0009091; 95% panel interval [+0.0063511, +0.0099235]
L=8 improves C256 validation at matched 50k updates on the fresh paired panel. C256 L=6 p=0.15 @ 50000 [stage 49 stored] leads the shared fresh panel but the panel does not resolve it against C256 L=8 p=0.15 @ 50000 [stage 50 fresh], C256 L=6 p=0.15 @ 50000 [stage 50 fresh]. depth is the better spend: the 6,428,258-parameter L=8 model costs 0.8529x the 7,537,378-parameter C320 L=6 budget and verifies better on the shared panel; this comparison is device-confounded, because the L=8 branch was trained on 'NVIDIA GeForce RTX 3090 Ti' while C320 was trained on 'NVIDIA GeForce RTX 3090'. moving the same L=6 architecture from 'NVIDIA GeForce RTX 3090' to 'NVIDIA GeForce RTX 3090 Ti' is worth +0.0007993 on this panel ([-0.0003195, +0.0019180]), measured on identical initialization, corpus, panel, and protocol. Report: my-gpt/checkpoints/stage_50_report.json
```
