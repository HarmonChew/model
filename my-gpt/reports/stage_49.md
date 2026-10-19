# Stage 49 script output

- Script: `49_dropout_strength.py`
- Run time (seconds): `1704`
- Device: `cuda`
- Peak GPU utilization: `100%`
- Peak GPU memory: `23291 MiB`
- Exit code: `0`
- Run status: `ok`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
{
  "stage": 49,
  "protocol": "paired_architecture_native_regularization",
  "training_executed": false,
  "runtime": {
    "device_type": "cuda",
    "device_name": "NVIDIA GeForce RTX 3090",
    "torch_version": "2.14.0+cu130",
    "lineage_device_required": "NVIDIA GeForce RTX 3090",
    "lineage_device_verified": true
  },
  "branches": [
    {
      "branch_key": "control",
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
        "vocabulary_size": 98
      },
      "initialization": {
        "kind": "fresh_architecture_native",
        "origin_stage": 49,
        "step": 0,
        "seed": 1337,
        "device": "cpu",
        "state_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        "policy": "Both branches recreate the same architecture-native initialization from the same seed and corpus. Residual dropout has no learnable parameters and consumes no initialization RNG, so the paired initial weights are claimed and checked to be byte-identical.",
        "residual_dropout": 0.15,
        "historical_weights_loaded": false,
        "historical_optimizer_loaded": false,
        "optimizer_state_entries_at_initialization": 0,
        "parameter_count": 4850274
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
      "residual_dropout": 0.2,
      "architecture": {
        "n_embd": 256,
        "n_head": 4,
        "n_layer": 6,
        "head_size": 64,
        "feed_forward_width": 1024,
        "block_size": 256,
        "residual_dropout": 0.2,
        "layer_contexts": [
          23,
          23,
          23,
          23,
          null,
          null
        ],
        "topology": "LLLLGG",
        "vocabulary_size": 98
      },
      "initialization": {
        "kind": "fresh_architecture_native",
        "origin_stage": 49,
        "step": 0,
        "seed": 1337,
        "device": "cpu",
        "state_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        "policy": "Both branches recreate the same architecture-native initialization from the same seed and corpus. Residual dropout has no learnable parameters and consumes no initialization RNG, so the paired initial weights are claimed and checked to be byte-identical.",
        "residual_dropout": 0.2,
        "historical_weights_loaded": false,
        "historical_optimizer_loaded": false,
        "optimizer_state_entries_at_initialization": 0,
        "parameter_count": 4850274
      },
      "initial_streams": {
        "model_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        "optimizer_sha256": "c5ffcf6319b0bad4abbbeb62670b300e06483d88a13730c82ca9fb123eff7155",
        "training_generator_sha256": "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced",
        "torch_rng_sha256": "059ea222d9d04c0b43c3602c01097ad125978a3787153133bce80fd163d4b01b"
      }
    }
  ],
  "control_initialization_equivalence": {
    "reference": "Stage 45 recorded C256 native initialization (dropout-invariant)",
    "state_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
    "stage_46_factory_identical": true,
    "stage_45_factory_identical": true,
    "identical": true
  },
  "updates_per_branch": 50000,
  "target_presentations_per_branch": 102400000,
  "parameter_count": 4850274,
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
  "data_fingerprint": "c6cbdd246799ad495f7813e5d20afa52c03e6c339fe7dbe15912fd81a45d2b48",
  "initialization_policy": "Both branches recreate the same architecture-native initialization from the same seed and corpus. Residual dropout has no learnable parameters and consumes no initialization RNG, so the paired initial weights are claimed and checked to be byte-identical.",
  "randomness_policy": "Both branches begin from the same CPU and accelerator training RNG states, reset after native initialization, and consume the same sampled training panels. Different dropout probabilities draw different masks, so neither identical dropout masks nor identical terminal Torch RNG states are claimed.",
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
      "best_step": 49500,
      "best_checkpoint_sha256": "897a432e4b1a94b0619e5235b06464a38b56261ca4e15a51f69d6daf40699eb4",
      "final_checkpoint_sha256": "e47016371b6a588448dbaa0bad6d05ff8564b7b21c6f6da81f20bbb56e916197",
      "fixed_panel_train_loss": 1.0690791565179825,
      "fixed_panel_validation_loss": 1.3597860205173493,
      "role": "read-only champion shootout peer"
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
    "role": "historical reference only; none is a Stage 49 initialization",
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
    }
  }
}
Stage 49 on cuda:1: native C256 at p=0.15 and p=0.20, 50,000 updates per branch
  control C256 p0.15 step     0 | lr 0.001 | train 4.7088 | val 4.6970 | best 4.6970 @ 0
  control C256 p0.15 step   500 | lr 0.001 | train 1.9418 | val 2.0019 | best 2.0019 @ 500
  control C256 p0.15 step  1000 | lr 0.001 | train 1.6911 | val 1.7794 | best 1.7794 @ 1000
  control C256 p0.15 step  1500 | lr 0.001 | train 1.5734 | val 1.6978 | best 1.6978 @ 1500
  control C256 p0.15 step  2000 | lr 0.001 | train 1.4944 | val 1.6458 | best 1.6458 @ 2000
  control C256 p0.15 step  2500 | lr 0.001 | train 1.4446 | val 1.6114 | best 1.6114 @ 2500
  control C256 p0.15 step  3000 | lr 0.001 | train 1.4060 | val 1.5879 | best 1.5879 @ 3000
  control C256 p0.15 step  3500 | lr 0.001 | train 1.3759 | val 1.5478 | best 1.5478 @ 3500
  control C256 p0.15 step  4000 | lr 0.001 | train 1.3507 | val 1.5408 | best 1.5408 @ 4000
  control C256 p0.15 step  4500 | lr 0.001 | train 1.3317 | val 1.5242 | best 1.5242 @ 4500
  control C256 p0.15 step  5000 | lr 0.001 | train 1.3167 | val 1.5093 | best 1.5093 @ 5000
  control C256 p0.15 step  5500 | lr 0.001 | train 1.3003 | val 1.4963 | best 1.4963 @ 5500
  control C256 p0.15 step  6000 | lr 0.001 | train 1.2856 | val 1.4835 | best 1.4835 @ 6000
  control C256 p0.15 step  6500 | lr 0.001 | train 1.2748 | val 1.4749 | best 1.4749 @ 6500
  control C256 p0.15 step  7000 | lr 0.001 | train 1.2602 | val 1.4574 | best 1.4574 @ 7000
  control C256 p0.15 step  7500 | lr 0.001 | train 1.2504 | val 1.4503 | best 1.4503 @ 7500
  control C256 p0.15 step  8000 | lr 0.001 | train 1.2480 | val 1.4581 | best 1.4503 @ 7500
  control C256 p0.15 step  8500 | lr 0.001 | train 1.2356 | val 1.4368 | best 1.4368 @ 8500
  control C256 p0.15 step  9000 | lr 0.001 | train 1.2275 | val 1.4350 | best 1.4350 @ 9000
  control C256 p0.15 step  9500 | lr 0.001 | train 1.2208 | val 1.4262 | best 1.4262 @ 9500
  control C256 p0.15 step 10000 | lr 0.001 | train 1.2166 | val 1.4281 | best 1.4262 @ 9500
  control C256 p0.15 step 10500 | lr 0.0003 | train 1.1764 | val 1.3958 | best 1.3958 @ 10500
  control C256 p0.15 step 11000 | lr 0.0003 | train 1.1690 | val 1.3958 | best 1.3958 @ 11000
  control C256 p0.15 step 11500 | lr 0.0003 | train 1.1622 | val 1.3883 | best 1.3883 @ 11500
  control C256 p0.15 step 12000 | lr 0.0003 | train 1.1558 | val 1.3838 | best 1.3838 @ 12000
  control C256 p0.15 step 12500 | lr 0.0003 | train 1.1512 | val 1.3820 | best 1.3820 @ 12500
  control C256 p0.15 step 13000 | lr 0.0003 | train 1.1494 | val 1.3809 | best 1.3809 @ 13000
  control C256 p0.15 step 13500 | lr 0.0001 | train 1.1374 | val 1.3715 | best 1.3715 @ 13500
  control C256 p0.15 step 14000 | lr 0.0001 | train 1.1337 | val 1.3689 | best 1.3689 @ 14000
  control C256 p0.15 step 14500 | lr 0.0001 | train 1.1309 | val 1.3683 | best 1.3683 @ 14500
  control C256 p0.15 step 15000 | lr 0.0001 | train 1.1291 | val 1.3687 | best 1.3683 @ 14500
  control C256 p0.15 step 15500 | lr 0.0001 | train 1.1272 | val 1.3688 | best 1.3683 @ 14500
  control C256 p0.15 step 16000 | lr 0.0001 | train 1.1252 | val 1.3671 | best 1.3671 @ 16000
  control C256 p0.15 step 16500 | lr 0.0001 | train 1.1236 | val 1.3663 | best 1.3663 @ 16500
  control C256 p0.15 step 17000 | lr 0.0001 | train 1.1217 | val 1.3659 | best 1.3659 @ 17000
  control C256 p0.15 step 17500 | lr 0.0001 | train 1.1198 | val 1.3646 | best 1.3646 @ 17500
  control C256 p0.15 step 18000 | lr 0.0001 | train 1.1189 | val 1.3634 | best 1.3634 @ 18000
  control C256 p0.15 step 18500 | lr 3e-05 | train 1.1161 | val 1.3619 | best 1.3619 @ 18500
  control C256 p0.15 step 19000 | lr 3e-05 | train 1.1147 | val 1.3613 | best 1.3613 @ 19000
  control C256 p0.15 step 19500 | lr 3e-05 | train 1.1138 | val 1.3597 | best 1.3597 @ 19500
  control C256 p0.15 step 20000 | lr 3e-05 | train 1.1132 | val 1.3597 | best 1.3597 @ 19500
  control C256 p0.15 step 20500 | lr 3e-05 | train 1.1129 | val 1.3602 | best 1.3597 @ 19500
  control C256 p0.15 step 21000 | lr 3e-05 | train 1.1120 | val 1.3600 | best 1.3597 @ 19500
  control C256 p0.15 step 21500 | lr 3e-05 | train 1.1115 | val 1.3603 | best 1.3597 @ 19500
  control C256 p0.15 step 22000 | lr 3e-05 | train 1.1111 | val 1.3592 | best 1.3592 @ 22000
  control C256 p0.15 step 22500 | lr 3e-05 | train 1.1104 | val 1.3593 | best 1.3592 @ 22000
  control C256 p0.15 step 23000 | lr 3e-05 | train 1.1100 | val 1.3598 | best 1.3592 @ 22000
  control C256 p0.15 step 23500 | lr 3e-05 | train 1.1091 | val 1.3572 | best 1.3572 @ 23500
  control C256 p0.15 step 24000 | lr 3e-05 | train 1.1087 | val 1.3574 | best 1.3572 @ 23500
  control C256 p0.15 step 24500 | lr 3e-05 | train 1.1081 | val 1.3580 | best 1.3572 @ 23500
  control C256 p0.15 step 25000 | lr 3e-05 | train 1.1075 | val 1.3577 | best 1.3572 @ 23500
  control C256 p0.15 step 25500 | lr 3e-05 | train 1.1070 | val 1.3577 | best 1.3572 @ 23500
  control C256 p0.15 step 26000 | lr 3e-05 | train 1.1066 | val 1.3571 | best 1.3571 @ 26000
  control C256 p0.15 step 26500 | lr 3e-05 | train 1.1061 | val 1.3562 | best 1.3562 @ 26500
  control C256 p0.15 step 27000 | lr 3e-05 | train 1.1056 | val 1.3572 | best 1.3562 @ 26500
  control C256 p0.15 step 27500 | lr 3e-05 | train 1.1051 | val 1.3567 | best 1.3562 @ 26500
  control C256 p0.15 step 28000 | lr 3e-05 | train 1.1049 | val 1.3568 | best 1.3562 @ 26500
  control C256 p0.15 step 28500 | lr 3e-05 | train 1.1039 | val 1.3563 | best 1.3562 @ 26500
  control C256 p0.15 step 29000 | lr 3e-05 | train 1.1039 | val 1.3564 | best 1.3562 @ 26500
  control C256 p0.15 step 29500 | lr 3e-05 | train 1.1034 | val 1.3551 | best 1.3551 @ 29500
  control C256 p0.15 step 30000 | lr 3e-05 | train 1.1030 | val 1.3560 | best 1.3551 @ 29500
  control C256 p0.15 step 30500 | lr 3e-05 | train 1.1028 | val 1.3565 | best 1.3551 @ 29500
  control C256 p0.15 step 31000 | lr 3e-05 | train 1.1024 | val 1.3558 | best 1.3551 @ 29500
  control C256 p0.15 step 31500 | lr 3e-05 | train 1.1018 | val 1.3560 | best 1.3551 @ 29500
  control C256 p0.15 step 32000 | lr 3e-05 | train 1.1012 | val 1.3554 | best 1.3551 @ 29500
  control C256 p0.15 step 32500 | lr 3e-05 | train 1.1011 | val 1.3549 | best 1.3549 @ 32500
  control C256 p0.15 step 33000 | lr 3e-05 | train 1.1005 | val 1.3556 | best 1.3549 @ 32500
  control C256 p0.15 step 33500 | lr 3e-05 | train 1.1000 | val 1.3545 | best 1.3545 @ 33500
  control C256 p0.15 step 34000 | lr 3e-05 | train 1.0998 | val 1.3534 | best 1.3534 @ 34000
  control C256 p0.15 step 34500 | lr 3e-05 | train 1.0995 | val 1.3540 | best 1.3534 @ 34000
  control C256 p0.15 step 35000 | lr 3e-05 | train 1.0988 | val 1.3550 | best 1.3534 @ 34000
  control C256 p0.15 step 35500 | lr 3e-05 | train 1.0982 | val 1.3546 | best 1.3534 @ 34000
  control C256 p0.15 step 36000 | lr 3e-05 | train 1.0979 | val 1.3534 | best 1.3534 @ 34000
  control C256 p0.15 step 36500 | lr 3e-05 | train 1.0977 | val 1.3535 | best 1.3534 @ 34000
  control C256 p0.15 step 37000 | lr 3e-05 | train 1.0974 | val 1.3540 | best 1.3534 @ 34000
  control C256 p0.15 step 37500 | lr 3e-05 | train 1.0967 | val 1.3527 | best 1.3527 @ 37500
  control C256 p0.15 step 38000 | lr 3e-05 | train 1.0964 | val 1.3524 | best 1.3524 @ 38000
  control C256 p0.15 step 38500 | lr 3e-05 | train 1.0961 | val 1.3520 | best 1.3520 @ 38500
  control C256 p0.15 step 39000 | lr 3e-05 | train 1.0959 | val 1.3522 | best 1.3520 @ 38500
  control C256 p0.15 step 39500 | lr 3e-05 | train 1.0952 | val 1.3522 | best 1.3520 @ 38500
  control C256 p0.15 step 40000 | lr 3e-05 | train 1.0950 | val 1.3525 | best 1.3520 @ 38500
  control C256 p0.15 step 40500 | lr 3e-05 | train 1.0946 | val 1.3528 | best 1.3520 @ 38500
  control C256 p0.15 step 41000 | lr 3e-05 | train 1.0940 | val 1.3516 | best 1.3516 @ 41000
  control C256 p0.15 step 41500 | lr 3e-05 | train 1.0938 | val 1.3516 | best 1.3516 @ 41500
  control C256 p0.15 step 42000 | lr 3e-05 | train 1.0930 | val 1.3522 | best 1.3516 @ 41500
  control C256 p0.15 step 42500 | lr 3e-05 | train 1.0925 | val 1.3504 | best 1.3504 @ 42500
  control C256 p0.15 step 43000 | lr 3e-05 | train 1.0924 | val 1.3515 | best 1.3504 @ 42500
  control C256 p0.15 step 43500 | lr 3e-05 | train 1.0920 | val 1.3519 | best 1.3504 @ 42500
  control C256 p0.15 step 44000 | lr 3e-05 | train 1.0920 | val 1.3522 | best 1.3504 @ 42500
  control C256 p0.15 step 44500 | lr 3e-05 | train 1.0914 | val 1.3523 | best 1.3504 @ 42500
  control C256 p0.15 step 45000 | lr 3e-05 | train 1.0911 | val 1.3516 | best 1.3504 @ 42500
  control C256 p0.15 step 45500 | lr 3e-05 | train 1.0907 | val 1.3514 | best 1.3504 @ 42500
  control C256 p0.15 step 46000 | lr 3e-05 | train 1.0904 | val 1.3514 | best 1.3504 @ 42500
  control C256 p0.15 step 46500 | lr 3e-05 | train 1.0899 | val 1.3515 | best 1.3504 @ 42500
  control C256 p0.15 step 47000 | lr 3e-05 | train 1.0897 | val 1.3517 | best 1.3504 @ 42500
  control C256 p0.15 step 47500 | lr 3e-05 | train 1.0891 | val 1.3501 | best 1.3501 @ 47500
  control C256 p0.15 step 48000 | lr 3e-05 | train 1.0887 | val 1.3504 | best 1.3501 @ 47500
  control C256 p0.15 step 48500 | lr 3e-05 | train 1.0884 | val 1.3492 | best 1.3492 @ 48500
  control C256 p0.15 step 49000 | lr 3e-05 | train 1.0884 | val 1.3513 | best 1.3492 @ 48500
  control C256 p0.15 step 49500 | lr 3e-05 | train 1.0878 | val 1.3500 | best 1.3492 @ 48500
  control C256 p0.15 step 50000 | lr 3e-05 | train 1.0878 | val 1.3507 | best 1.3492 @ 48500
  treatment C256 p0.2 step     0 | lr 0.001 | train 4.7088 | val 4.6970 | best 4.6970 @ 0
  treatment C256 p0.2 step   500 | lr 0.001 | train 1.9485 | val 2.0072 | best 2.0072 @ 500
  treatment C256 p0.2 step  1000 | lr 0.001 | train 1.7034 | val 1.7902 | best 1.7902 @ 1000
  treatment C256 p0.2 step  1500 | lr 0.001 | train 1.5876 | val 1.7123 | best 1.7123 @ 1500
  treatment C256 p0.2 step  2000 | lr 0.001 | train 1.5097 | val 1.6578 | best 1.6578 @ 2000
  treatment C256 p0.2 step  2500 | lr 0.001 | train 1.4605 | val 1.6278 | best 1.6278 @ 2500
  treatment C256 p0.2 step  3000 | lr 0.001 | train 1.4204 | val 1.5998 | best 1.5998 @ 3000
  treatment C256 p0.2 step  3500 | lr 0.001 | train 1.3906 | val 1.5586 | best 1.5586 @ 3500
  treatment C256 p0.2 step  4000 | lr 0.001 | train 1.3661 | val 1.5577 | best 1.5577 @ 4000
  treatment C256 p0.2 step  4500 | lr 0.001 | train 1.3468 | val 1.5323 | best 1.5323 @ 4500
  treatment C256 p0.2 step  5000 | lr 0.001 | train 1.3311 | val 1.5201 | best 1.5201 @ 5000
  treatment C256 p0.2 step  5500 | lr 0.001 | train 1.3134 | val 1.5079 | best 1.5079 @ 5500
  treatment C256 p0.2 step  6000 | lr 0.001 | train 1.3020 | val 1.4967 | best 1.4967 @ 6000
  treatment C256 p0.2 step  6500 | lr 0.001 | train 1.2868 | val 1.4857 | best 1.4857 @ 6500
  treatment C256 p0.2 step  7000 | lr 0.001 | train 1.2739 | val 1.4701 | best 1.4701 @ 7000
  treatment C256 p0.2 step  7500 | lr 0.001 | train 1.2647 | val 1.4620 | best 1.4620 @ 7500
  treatment C256 p0.2 step  8000 | lr 0.001 | train 1.2591 | val 1.4640 | best 1.4620 @ 7500
  treatment C256 p0.2 step  8500 | lr 0.001 | train 1.2493 | val 1.4512 | best 1.4512 @ 8500
  treatment C256 p0.2 step  9000 | lr 0.001 | train 1.2433 | val 1.4475 | best 1.4475 @ 9000
  treatment C256 p0.2 step  9500 | lr 0.001 | train 1.2372 | val 1.4386 | best 1.4386 @ 9500
  treatment C256 p0.2 step 10000 | lr 0.001 | train 1.2310 | val 1.4342 | best 1.4342 @ 10000
  treatment C256 p0.2 step 10500 | lr 0.0003 | train 1.1938 | val 1.4033 | best 1.4033 @ 10500
  treatment C256 p0.2 step 11000 | lr 0.0003 | train 1.1863 | val 1.4034 | best 1.4033 @ 10500
  treatment C256 p0.2 step 11500 | lr 0.0003 | train 1.1792 | val 1.3966 | best 1.3966 @ 11500
  treatment C256 p0.2 step 12000 | lr 0.0003 | train 1.1732 | val 1.3938 | best 1.3938 @ 12000
  treatment C256 p0.2 step 12500 | lr 0.0003 | train 1.1691 | val 1.3915 | best 1.3915 @ 12500
  treatment C256 p0.2 step 13000 | lr 0.0003 | train 1.1665 | val 1.3911 | best 1.3911 @ 13000
  treatment C256 p0.2 step 13500 | lr 0.0001 | train 1.1548 | val 1.3810 | best 1.3810 @ 13500
  treatment C256 p0.2 step 14000 | lr 0.0001 | train 1.1520 | val 1.3797 | best 1.3797 @ 14000
  treatment C256 p0.2 step 14500 | lr 0.0001 | train 1.1495 | val 1.3796 | best 1.3796 @ 14500
  treatment C256 p0.2 step 15000 | lr 0.0001 | train 1.1475 | val 1.3795 | best 1.3795 @ 15000
  treatment C256 p0.2 step 15500 | lr 0.0001 | train 1.1455 | val 1.3783 | best 1.3783 @ 15500
  treatment C256 p0.2 step 16000 | lr 0.0001 | train 1.1433 | val 1.3772 | best 1.3772 @ 16000
  treatment C256 p0.2 step 16500 | lr 0.0001 | train 1.1423 | val 1.3764 | best 1.3764 @ 16500
  treatment C256 p0.2 step 17000 | lr 0.0001 | train 1.1401 | val 1.3753 | best 1.3753 @ 17000
  treatment C256 p0.2 step 17500 | lr 0.0001 | train 1.1380 | val 1.3743 | best 1.3743 @ 17500
  treatment C256 p0.2 step 18000 | lr 0.0001 | train 1.1371 | val 1.3737 | best 1.3737 @ 18000
  treatment C256 p0.2 step 18500 | lr 3e-05 | train 1.1350 | val 1.3719 | best 1.3719 @ 18500
  treatment C256 p0.2 step 19000 | lr 3e-05 | train 1.1338 | val 1.3718 | best 1.3718 @ 19000
  treatment C256 p0.2 step 19500 | lr 3e-05 | train 1.1328 | val 1.3703 | best 1.3703 @ 19500
  treatment C256 p0.2 step 20000 | lr 3e-05 | train 1.1321 | val 1.3697 | best 1.3697 @ 20000
  treatment C256 p0.2 step 20500 | lr 3e-05 | train 1.1320 | val 1.3695 | best 1.3695 @ 20500
  treatment C256 p0.2 step 21000 | lr 3e-05 | train 1.1313 | val 1.3702 | best 1.3695 @ 20500
  treatment C256 p0.2 step 21500 | lr 3e-05 | train 1.1309 | val 1.3700 | best 1.3695 @ 20500
  treatment C256 p0.2 step 22000 | lr 3e-05 | train 1.1304 | val 1.3699 | best 1.3695 @ 20500
  treatment C256 p0.2 step 22500 | lr 3e-05 | train 1.1300 | val 1.3691 | best 1.3691 @ 22500
  treatment C256 p0.2 step 23000 | lr 3e-05 | train 1.1295 | val 1.3699 | best 1.3691 @ 22500
  treatment C256 p0.2 step 23500 | lr 3e-05 | train 1.1286 | val 1.3677 | best 1.3677 @ 23500
  treatment C256 p0.2 step 24000 | lr 3e-05 | train 1.1282 | val 1.3674 | best 1.3674 @ 24000
  treatment C256 p0.2 step 24500 | lr 3e-05 | train 1.1277 | val 1.3680 | best 1.3674 @ 24000
  treatment C256 p0.2 step 25000 | lr 3e-05 | train 1.1271 | val 1.3672 | best 1.3672 @ 25000
  treatment C256 p0.2 step 25500 | lr 3e-05 | train 1.1269 | val 1.3675 | best 1.3672 @ 25000
  treatment C256 p0.2 step 26000 | lr 3e-05 | train 1.1264 | val 1.3672 | best 1.3672 @ 25000
  treatment C256 p0.2 step 26500 | lr 3e-05 | train 1.1260 | val 1.3657 | best 1.3657 @ 26500
  treatment C256 p0.2 step 27000 | lr 3e-05 | train 1.1256 | val 1.3666 | best 1.3657 @ 26500
  treatment C256 p0.2 step 27500 | lr 3e-05 | train 1.1250 | val 1.3660 | best 1.3657 @ 26500
  treatment C256 p0.2 step 28000 | lr 3e-05 | train 1.1247 | val 1.3667 | best 1.3657 @ 26500
  treatment C256 p0.2 step 28500 | lr 3e-05 | train 1.1237 | val 1.3651 | best 1.3651 @ 28500
  treatment C256 p0.2 step 29000 | lr 3e-05 | train 1.1239 | val 1.3661 | best 1.3651 @ 28500
  treatment C256 p0.2 step 29500 | lr 3e-05 | train 1.1234 | val 1.3649 | best 1.3649 @ 29500
  treatment C256 p0.2 step 30000 | lr 3e-05 | train 1.1230 | val 1.3648 | best 1.3648 @ 30000
  treatment C256 p0.2 step 30500 | lr 3e-05 | train 1.1229 | val 1.3652 | best 1.3648 @ 30000
  treatment C256 p0.2 step 31000 | lr 3e-05 | train 1.1223 | val 1.3641 | best 1.3641 @ 31000
  treatment C256 p0.2 step 31500 | lr 3e-05 | train 1.1218 | val 1.3645 | best 1.3641 @ 31000
  treatment C256 p0.2 step 32000 | lr 3e-05 | train 1.1212 | val 1.3631 | best 1.3631 @ 32000
  treatment C256 p0.2 step 32500 | lr 3e-05 | train 1.1209 | val 1.3633 | best 1.3631 @ 32000
  treatment C256 p0.2 step 33000 | lr 3e-05 | train 1.1204 | val 1.3636 | best 1.3631 @ 32000
  treatment C256 p0.2 step 33500 | lr 3e-05 | train 1.1199 | val 1.3628 | best 1.3628 @ 33500
  treatment C256 p0.2 step 34000 | lr 3e-05 | train 1.1197 | val 1.3622 | best 1.3622 @ 34000
  treatment C256 p0.2 step 34500 | lr 3e-05 | train 1.1195 | val 1.3624 | best 1.3622 @ 34000
  treatment C256 p0.2 step 35000 | lr 3e-05 | train 1.1189 | val 1.3632 | best 1.3622 @ 34000
  treatment C256 p0.2 step 35500 | lr 3e-05 | train 1.1183 | val 1.3634 | best 1.3622 @ 34000
  treatment C256 p0.2 step 36000 | lr 3e-05 | train 1.1182 | val 1.3627 | best 1.3622 @ 34000
  treatment C256 p0.2 step 36500 | lr 3e-05 | train 1.1178 | val 1.3627 | best 1.3622 @ 34000
  treatment C256 p0.2 step 37000 | lr 3e-05 | train 1.1176 | val 1.3628 | best 1.3622 @ 34000
  treatment C256 p0.2 step 37500 | lr 3e-05 | train 1.1171 | val 1.3622 | best 1.3622 @ 34000
  treatment C256 p0.2 step 38000 | lr 3e-05 | train 1.1165 | val 1.3618 | best 1.3618 @ 38000
  treatment C256 p0.2 step 38500 | lr 3e-05 | train 1.1161 | val 1.3611 | best 1.3611 @ 38500
  treatment C256 p0.2 step 39000 | lr 3e-05 | train 1.1159 | val 1.3618 | best 1.3611 @ 38500
  treatment C256 p0.2 step 39500 | lr 3e-05 | train 1.1155 | val 1.3620 | best 1.3611 @ 38500
  treatment C256 p0.2 step 40000 | lr 3e-05 | train 1.1149 | val 1.3620 | best 1.3611 @ 38500
  treatment C256 p0.2 step 40500 | lr 3e-05 | train 1.1147 | val 1.3622 | best 1.3611 @ 38500
  treatment C256 p0.2 step 41000 | lr 3e-05 | train 1.1140 | val 1.3611 | best 1.3611 @ 38500
  treatment C256 p0.2 step 41500 | lr 3e-05 | train 1.1140 | val 1.3619 | best 1.3611 @ 38500
  treatment C256 p0.2 step 42000 | lr 3e-05 | train 1.1133 | val 1.3618 | best 1.3611 @ 38500
  treatment C256 p0.2 step 42500 | lr 3e-05 | train 1.1130 | val 1.3600 | best 1.3600 @ 42500
  treatment C256 p0.2 step 43000 | lr 3e-05 | train 1.1128 | val 1.3604 | best 1.3600 @ 42500
  treatment C256 p0.2 step 43500 | lr 3e-05 | train 1.1122 | val 1.3604 | best 1.3600 @ 42500
  treatment C256 p0.2 step 44000 | lr 3e-05 | train 1.1123 | val 1.3612 | best 1.3600 @ 42500
  treatment C256 p0.2 step 44500 | lr 3e-05 | train 1.1119 | val 1.3610 | best 1.3600 @ 42500
  treatment C256 p0.2 step 45000 | lr 3e-05 | train 1.1114 | val 1.3606 | best 1.3600 @ 42500
  treatment C256 p0.2 step 45500 | lr 3e-05 | train 1.1112 | val 1.3607 | best 1.3600 @ 42500
  treatment C256 p0.2 step 46000 | lr 3e-05 | train 1.1109 | val 1.3607 | best 1.3600 @ 42500
  treatment C256 p0.2 step 46500 | lr 3e-05 | train 1.1104 | val 1.3604 | best 1.3600 @ 42500
  treatment C256 p0.2 step 47000 | lr 3e-05 | train 1.1102 | val 1.3609 | best 1.3600 @ 42500
  treatment C256 p0.2 step 47500 | lr 3e-05 | train 1.1098 | val 1.3593 | best 1.3593 @ 47500
  treatment C256 p0.2 step 48000 | lr 3e-05 | train 1.1094 | val 1.3595 | best 1.3593 @ 47500
  treatment C256 p0.2 step 48500 | lr 3e-05 | train 1.1091 | val 1.3587 | best 1.3587 @ 48500
  treatment C256 p0.2 step 49000 | lr 3e-05 | train 1.1089 | val 1.3600 | best 1.3587 @ 48500
  treatment C256 p0.2 step 49500 | lr 3e-05 | train 1.1087 | val 1.3593 | best 1.3587 @ 48500
  treatment C256 p0.2 step 50000 | lr 3e-05 | train 1.1085 | val 1.3595 | best 1.3587 @ 48500
  control C256 p0.15 @ 50000: 1.3428665
  treatment C256 p0.2 @ 50000: 1.3492821
  paired candidate - baseline: +0.0064156; SE 0.0005430; 95% panel interval [+0.0053488, +0.0074824]
  1. C256 p=0.15 @ 50000: 1.3425371 (SE 0.0033186)
  2. C256 p=0.2 @ 50000: 1.3486411 (SE 0.0033161)
  3. C320 p=0.2 @ 50000: 1.3509843 (SE 0.0034710)
  4. C320 p=0.15 @ 50000: 1.3514700 (SE 0.0035450)
  C256 p=0.15 @ 50000 - C256 p=0.2 @ 50000: -0.0061040; SE 0.0005235; 95% panel interval [-0.0071325, -0.0050754]
  C256 p=0.15 @ 50000 - C320 p=0.15 @ 50000: -0.0089329; SE 0.0009200; 95% panel interval [-0.0107404, -0.0071253]
  C256 p=0.15 @ 50000 - C320 p=0.2 @ 50000: -0.0084472; SE 0.0007580; 95% panel interval [-0.0099364, -0.0069580]
  C256 p=0.2 @ 50000 - C320 p=0.15 @ 50000: -0.0028289; SE 0.0008664; 95% panel interval [-0.0045312, -0.0011266]
  C256 p=0.2 @ 50000 - C320 p=0.2 @ 50000: -0.0023432; SE 0.0007752; 95% panel interval [-0.0038663, -0.0008201]
  C320 p=0.15 @ 50000 - C320 p=0.2 @ 50000: +0.0004857; SE 0.0006438; 95% panel interval [-0.0007792, +0.0017506]
p=0.15 beats p=0.20 at C256 on the fresh paired panel. C256 p=0.15 @ 50000 is the resolved champion: it beats all 3 peers on the shared fresh panel. Report: my-gpt/checkpoints/stage_49_report.json
```
