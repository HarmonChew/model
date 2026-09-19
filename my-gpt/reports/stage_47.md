# Stage 47 script output

- Script: `47_dropout_strength.py`
- Run time (seconds): `2015`
- Device: `cuda`
- Peak GPU utilization: `99%`
- Peak GPU memory: `15250 MiB`
- Exit code: `0`
- Run status: `ok`

## Stdout

```text
{
  "stage": 47,
  "protocol": "paired_architecture_native_regularization",
  "training_executed": false,
  "branches": [
    {
      "branch_key": "control",
      "residual_dropout": 0.15,
      "architecture": {
        "n_embd": 320,
        "n_head": 4,
        "n_layer": 6,
        "head_size": 80,
        "feed_forward_width": 1280,
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
        "origin_stage": 47,
        "step": 0,
        "seed": 1337,
        "device": "cpu",
        "state_sha256": "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
        "policy": "Both branches recreate the same architecture-native initialization from the same seed and corpus. Residual dropout has no learnable parameters and consumes no initialization RNG, so the paired initial weights are claimed and checked to be byte-identical.",
        "residual_dropout": 0.15,
        "historical_weights_loaded": false,
        "historical_optimizer_loaded": false,
        "optimizer_state_entries_at_initialization": 0,
        "parameter_count": 7537378
      },
      "initial_streams": {
        "model_sha256": "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
        "optimizer_sha256": "c5ffcf6319b0bad4abbbeb62670b300e06483d88a13730c82ca9fb123eff7155",
        "training_generator_sha256": "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced",
        "torch_rng_sha256": "9940cf9e44812cb470574f2f6398e7f8809a5dc05d1f5f32f5832427f6b8bfc3"
      }
    },
    {
      "branch_key": "treatment",
      "residual_dropout": 0.2,
      "architecture": {
        "n_embd": 320,
        "n_head": 4,
        "n_layer": 6,
        "head_size": 80,
        "feed_forward_width": 1280,
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
        "origin_stage": 47,
        "step": 0,
        "seed": 1337,
        "device": "cpu",
        "state_sha256": "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
        "policy": "Both branches recreate the same architecture-native initialization from the same seed and corpus. Residual dropout has no learnable parameters and consumes no initialization RNG, so the paired initial weights are claimed and checked to be byte-identical.",
        "residual_dropout": 0.2,
        "historical_weights_loaded": false,
        "historical_optimizer_loaded": false,
        "optimizer_state_entries_at_initialization": 0,
        "parameter_count": 7537378
      },
      "initial_streams": {
        "model_sha256": "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
        "optimizer_sha256": "c5ffcf6319b0bad4abbbeb62670b300e06483d88a13730c82ca9fb123eff7155",
        "training_generator_sha256": "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced",
        "torch_rng_sha256": "9940cf9e44812cb470574f2f6398e7f8809a5dc05d1f5f32f5832427f6b8bfc3"
      }
    }
  ],
  "control_initialization_equivalence": {
    "reference": "Stage 46 C320 p=0.15 native initialization",
    "state_sha256": "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
    "identical": true
  },
  "updates_per_branch": 50000,
  "target_presentations_per_branch": 102400000,
  "parameter_count": 7537378,
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
    "stage_46_c320_p015_endpoint": {
      "branch": "treatment_c320",
      "dropout": 0.15,
      "step": 50000,
      "best_step": 49500,
      "best_checkpoint_sha256": "897a432e4b1a94b0619e5235b06464a38b56261ca4e15a51f69d6daf40699eb4",
      "final_checkpoint_sha256": "e47016371b6a588448dbaa0bad6d05ff8564b7b21c6f6da81f20bbb56e916197",
      "fixed_panel_train_loss": 1.0690791565179825,
      "fixed_panel_validation_loss": 1.3597860205173493,
      "role": "the state this stage's p=0.15 control must re-derive from zero"
    },
    "role": "historical reference only; none is a Stage 47 initialization"
  }
}
Stage 47 on cuda: native C320 at p=0.15 and p=0.20, 50,000 updates per branch
  control C320 p0.15 step     0 | lr 0.001 | train 4.7684 | val 4.7675 | best 4.7675 @ 0
  control C320 p0.15 step   500 | lr 0.001 | train 1.8785 | val 1.9401 | best 1.9401 @ 500
  control C320 p0.15 step  1000 | lr 0.001 | train 1.6602 | val 1.7594 | best 1.7594 @ 1000
  control C320 p0.15 step  1500 | lr 0.001 | train 1.5543 | val 1.6872 | best 1.6872 @ 1500
  control C320 p0.15 step  2000 | lr 0.001 | train 1.4881 | val 1.6452 | best 1.6452 @ 2000
  control C320 p0.15 step  2500 | lr 0.001 | train 1.4357 | val 1.6028 | best 1.6028 @ 2500
  control C320 p0.15 step  3000 | lr 0.001 | train 1.4024 | val 1.5855 | best 1.5855 @ 3000
  control C320 p0.15 step  3500 | lr 0.001 | train 1.3710 | val 1.5471 | best 1.5471 @ 3500
  control C320 p0.15 step  4000 | lr 0.001 | train 1.3477 | val 1.5402 | best 1.5402 @ 4000
  control C320 p0.15 step  4500 | lr 0.001 | train 1.3283 | val 1.5239 | best 1.5239 @ 4500
  control C320 p0.15 step  5000 | lr 0.001 | train 1.3183 | val 1.5166 | best 1.5166 @ 5000
  control C320 p0.15 step  5500 | lr 0.001 | train 1.2973 | val 1.4932 | best 1.4932 @ 5500
  control C320 p0.15 step  6000 | lr 0.001 | train 1.2850 | val 1.4894 | best 1.4894 @ 6000
  control C320 p0.15 step  6500 | lr 0.001 | train 1.2697 | val 1.4787 | best 1.4787 @ 6500
  control C320 p0.15 step  7000 | lr 0.001 | train 1.2592 | val 1.4670 | best 1.4670 @ 7000
  control C320 p0.15 step  7500 | lr 0.001 | train 1.2500 | val 1.4624 | best 1.4624 @ 7500
  control C320 p0.15 step  8000 | lr 0.001 | train 1.2452 | val 1.4706 | best 1.4624 @ 7500
  control C320 p0.15 step  8500 | lr 0.001 | train 1.2351 | val 1.4517 | best 1.4517 @ 8500
  control C320 p0.15 step  9000 | lr 0.001 | train 1.2247 | val 1.4494 | best 1.4494 @ 9000
  control C320 p0.15 step  9500 | lr 0.001 | train 1.2167 | val 1.4392 | best 1.4392 @ 9500
  control C320 p0.15 step 10000 | lr 0.001 | train 1.2130 | val 1.4403 | best 1.4392 @ 9500
  control C320 p0.15 step 10500 | lr 0.0003 | train 1.1701 | val 1.4026 | best 1.4026 @ 10500
  control C320 p0.15 step 11000 | lr 0.0003 | train 1.1612 | val 1.3992 | best 1.3992 @ 11000
  control C320 p0.15 step 11500 | lr 0.0003 | train 1.1538 | val 1.3942 | best 1.3942 @ 11500
  control C320 p0.15 step 12000 | lr 0.0003 | train 1.1467 | val 1.3880 | best 1.3880 @ 12000
  control C320 p0.15 step 12500 | lr 0.0003 | train 1.1419 | val 1.3901 | best 1.3880 @ 12000
  control C320 p0.15 step 13000 | lr 0.0003 | train 1.1400 | val 1.3911 | best 1.3880 @ 12000
  control C320 p0.15 step 13500 | lr 0.0001 | train 1.1265 | val 1.3806 | best 1.3806 @ 13500
  control C320 p0.15 step 14000 | lr 0.0001 | train 1.1223 | val 1.3776 | best 1.3776 @ 14000
  control C320 p0.15 step 14500 | lr 0.0001 | train 1.1189 | val 1.3765 | best 1.3765 @ 14500
  control C320 p0.15 step 15000 | lr 0.0001 | train 1.1170 | val 1.3772 | best 1.3765 @ 14500
  control C320 p0.15 step 15500 | lr 0.0001 | train 1.1152 | val 1.3758 | best 1.3758 @ 15500
  control C320 p0.15 step 16000 | lr 0.0001 | train 1.1126 | val 1.3748 | best 1.3748 @ 16000
  control C320 p0.15 step 16500 | lr 0.0001 | train 1.1108 | val 1.3746 | best 1.3746 @ 16500
  control C320 p0.15 step 17000 | lr 0.0001 | train 1.1086 | val 1.3732 | best 1.3732 @ 17000
  control C320 p0.15 step 17500 | lr 0.0001 | train 1.1064 | val 1.3716 | best 1.3716 @ 17500
  control C320 p0.15 step 18000 | lr 0.0001 | train 1.1050 | val 1.3716 | best 1.3716 @ 18000
  control C320 p0.15 step 18500 | lr 3e-05 | train 1.1022 | val 1.3687 | best 1.3687 @ 18500
  control C320 p0.15 step 19000 | lr 3e-05 | train 1.1007 | val 1.3669 | best 1.3669 @ 19000
  control C320 p0.15 step 19500 | lr 3e-05 | train 1.0995 | val 1.3667 | best 1.3667 @ 19500
  control C320 p0.15 step 20000 | lr 3e-05 | train 1.0988 | val 1.3669 | best 1.3667 @ 19500
  control C320 p0.15 step 20500 | lr 3e-05 | train 1.0983 | val 1.3666 | best 1.3666 @ 20500
  control C320 p0.15 step 21000 | lr 3e-05 | train 1.0976 | val 1.3673 | best 1.3666 @ 20500
  control C320 p0.15 step 21500 | lr 3e-05 | train 1.0972 | val 1.3667 | best 1.3666 @ 20500
  control C320 p0.15 step 22000 | lr 3e-05 | train 1.0963 | val 1.3670 | best 1.3666 @ 20500
  control C320 p0.15 step 22500 | lr 3e-05 | train 1.0958 | val 1.3668 | best 1.3666 @ 20500
  control C320 p0.15 step 23000 | lr 3e-05 | train 1.0952 | val 1.3679 | best 1.3666 @ 20500
  control C320 p0.15 step 23500 | lr 3e-05 | train 1.0942 | val 1.3650 | best 1.3650 @ 23500
  control C320 p0.15 step 24000 | lr 3e-05 | train 1.0936 | val 1.3647 | best 1.3647 @ 24000
  control C320 p0.15 step 24500 | lr 3e-05 | train 1.0928 | val 1.3648 | best 1.3647 @ 24000
  control C320 p0.15 step 25000 | lr 3e-05 | train 1.0922 | val 1.3647 | best 1.3647 @ 24000
  control C320 p0.15 step 25500 | lr 3e-05 | train 1.0915 | val 1.3640 | best 1.3640 @ 25500
  control C320 p0.15 step 26000 | lr 3e-05 | train 1.0914 | val 1.3647 | best 1.3640 @ 25500
  control C320 p0.15 step 26500 | lr 3e-05 | train 1.0908 | val 1.3637 | best 1.3637 @ 26500
  control C320 p0.15 step 27000 | lr 3e-05 | train 1.0905 | val 1.3640 | best 1.3637 @ 26500
  control C320 p0.15 step 27500 | lr 3e-05 | train 1.0896 | val 1.3641 | best 1.3637 @ 26500
  control C320 p0.15 step 28000 | lr 3e-05 | train 1.0893 | val 1.3638 | best 1.3637 @ 26500
  control C320 p0.15 step 28500 | lr 3e-05 | train 1.0885 | val 1.3629 | best 1.3629 @ 28500
  control C320 p0.15 step 29000 | lr 3e-05 | train 1.0881 | val 1.3633 | best 1.3629 @ 28500
  control C320 p0.15 step 29500 | lr 3e-05 | train 1.0875 | val 1.3628 | best 1.3628 @ 29500
  control C320 p0.15 step 30000 | lr 3e-05 | train 1.0872 | val 1.3630 | best 1.3628 @ 29500
  control C320 p0.15 step 30500 | lr 3e-05 | train 1.0871 | val 1.3637 | best 1.3628 @ 29500
  control C320 p0.15 step 31000 | lr 3e-05 | train 1.0863 | val 1.3626 | best 1.3626 @ 31000
  control C320 p0.15 step 31500 | lr 3e-05 | train 1.0859 | val 1.3637 | best 1.3626 @ 31000
  control C320 p0.15 step 32000 | lr 3e-05 | train 1.0850 | val 1.3624 | best 1.3624 @ 32000
  control C320 p0.15 step 32500 | lr 3e-05 | train 1.0848 | val 1.3624 | best 1.3624 @ 32000
  control C320 p0.15 step 33000 | lr 3e-05 | train 1.0839 | val 1.3631 | best 1.3624 @ 32000
  control C320 p0.15 step 33500 | lr 3e-05 | train 1.0835 | val 1.3624 | best 1.3624 @ 33500
  control C320 p0.15 step 34000 | lr 3e-05 | train 1.0831 | val 1.3616 | best 1.3616 @ 34000
  control C320 p0.15 step 34500 | lr 3e-05 | train 1.0830 | val 1.3632 | best 1.3616 @ 34000
  control C320 p0.15 step 35000 | lr 3e-05 | train 1.0822 | val 1.3637 | best 1.3616 @ 34000
  control C320 p0.15 step 35500 | lr 3e-05 | train 1.0815 | val 1.3639 | best 1.3616 @ 34000
  control C320 p0.15 step 36000 | lr 3e-05 | train 1.0810 | val 1.3617 | best 1.3616 @ 34000
  control C320 p0.15 step 36500 | lr 3e-05 | train 1.0809 | val 1.3624 | best 1.3616 @ 34000
  control C320 p0.15 step 37000 | lr 3e-05 | train 1.0804 | val 1.3618 | best 1.3616 @ 34000
  control C320 p0.15 step 37500 | lr 3e-05 | train 1.0798 | val 1.3610 | best 1.3610 @ 37500
  control C320 p0.15 step 38000 | lr 3e-05 | train 1.0791 | val 1.3620 | best 1.3610 @ 37500
  control C320 p0.15 step 38500 | lr 3e-05 | train 1.0788 | val 1.3601 | best 1.3601 @ 38500
  control C320 p0.15 step 39000 | lr 3e-05 | train 1.0784 | val 1.3609 | best 1.3601 @ 38500
  control C320 p0.15 step 39500 | lr 3e-05 | train 1.0778 | val 1.3610 | best 1.3601 @ 38500
  control C320 p0.15 step 40000 | lr 3e-05 | train 1.0776 | val 1.3617 | best 1.3601 @ 38500
  control C320 p0.15 step 40500 | lr 3e-05 | train 1.0773 | val 1.3623 | best 1.3601 @ 38500
  control C320 p0.15 step 41000 | lr 3e-05 | train 1.0765 | val 1.3607 | best 1.3601 @ 38500
  control C320 p0.15 step 41500 | lr 3e-05 | train 1.0762 | val 1.3619 | best 1.3601 @ 38500
  control C320 p0.15 step 42000 | lr 3e-05 | train 1.0755 | val 1.3610 | best 1.3601 @ 38500
  control C320 p0.15 step 42500 | lr 3e-05 | train 1.0751 | val 1.3602 | best 1.3601 @ 38500
  control C320 p0.15 step 43000 | lr 3e-05 | train 1.0750 | val 1.3604 | best 1.3601 @ 38500
  control C320 p0.15 step 43500 | lr 3e-05 | train 1.0744 | val 1.3604 | best 1.3601 @ 38500
  control C320 p0.15 step 44000 | lr 3e-05 | train 1.0741 | val 1.3611 | best 1.3601 @ 38500
  control C320 p0.15 step 44500 | lr 3e-05 | train 1.0736 | val 1.3606 | best 1.3601 @ 38500
  control C320 p0.15 step 45000 | lr 3e-05 | train 1.0730 | val 1.3611 | best 1.3601 @ 38500
  control C320 p0.15 step 45500 | lr 3e-05 | train 1.0727 | val 1.3601 | best 1.3601 @ 38500
  control C320 p0.15 step 46000 | lr 3e-05 | train 1.0723 | val 1.3602 | best 1.3601 @ 38500
  control C320 p0.15 step 46500 | lr 3e-05 | train 1.0719 | val 1.3606 | best 1.3601 @ 38500
  control C320 p0.15 step 47000 | lr 3e-05 | train 1.0715 | val 1.3609 | best 1.3601 @ 38500
  control C320 p0.15 step 47500 | lr 3e-05 | train 1.0707 | val 1.3591 | best 1.3591 @ 47500
  control C320 p0.15 step 48000 | lr 3e-05 | train 1.0704 | val 1.3596 | best 1.3591 @ 47500
  control C320 p0.15 step 48500 | lr 3e-05 | train 1.0702 | val 1.3586 | best 1.3586 @ 48500
  control C320 p0.15 step 49000 | lr 3e-05 | train 1.0700 | val 1.3603 | best 1.3586 @ 48500
  control C320 p0.15 step 49500 | lr 3e-05 | train 1.0694 | val 1.3586 | best 1.3586 @ 49500
  control C320 p0.15 step 50000 | lr 3e-05 | train 1.0691 | val 1.3598 | best 1.3586 @ 49500
  treatment C320 p0.2 step     0 | lr 0.001 | train 4.7684 | val 4.7675 | best 4.7675 @ 0
  treatment C320 p0.2 step   500 | lr 0.001 | train 1.8783 | val 1.9408 | best 1.9408 @ 500
  treatment C320 p0.2 step  1000 | lr 0.001 | train 1.6655 | val 1.7667 | best 1.7667 @ 1000
  treatment C320 p0.2 step  1500 | lr 0.001 | train 1.5597 | val 1.6950 | best 1.6950 @ 1500
  treatment C320 p0.2 step  2000 | lr 0.001 | train 1.4912 | val 1.6491 | best 1.6491 @ 2000
  treatment C320 p0.2 step  2500 | lr 0.001 | train 1.4418 | val 1.6096 | best 1.6096 @ 2500
  treatment C320 p0.2 step  3000 | lr 0.001 | train 1.4064 | val 1.5949 | best 1.5949 @ 3000
  treatment C320 p0.2 step  3500 | lr 0.001 | train 1.3769 | val 1.5537 | best 1.5537 @ 3500
  treatment C320 p0.2 step  4000 | lr 0.001 | train 1.3533 | val 1.5445 | best 1.5445 @ 4000
  treatment C320 p0.2 step  4500 | lr 0.001 | train 1.3351 | val 1.5320 | best 1.5320 @ 4500
  treatment C320 p0.2 step  5000 | lr 0.001 | train 1.3229 | val 1.5193 | best 1.5193 @ 5000
  treatment C320 p0.2 step  5500 | lr 0.001 | train 1.3018 | val 1.4966 | best 1.4966 @ 5500
  treatment C320 p0.2 step  6000 | lr 0.001 | train 1.2936 | val 1.4940 | best 1.4940 @ 6000
  treatment C320 p0.2 step  6500 | lr 0.001 | train 1.2770 | val 1.4876 | best 1.4876 @ 6500
  treatment C320 p0.2 step  7000 | lr 0.001 | train 1.2667 | val 1.4753 | best 1.4753 @ 7000
  treatment C320 p0.2 step  7500 | lr 0.001 | train 1.2563 | val 1.4666 | best 1.4666 @ 7500
  treatment C320 p0.2 step  8000 | lr 0.001 | train 1.2530 | val 1.4765 | best 1.4666 @ 7500
  treatment C320 p0.2 step  8500 | lr 0.001 | train 1.2442 | val 1.4522 | best 1.4522 @ 8500
  treatment C320 p0.2 step  9000 | lr 0.001 | train 1.2355 | val 1.4516 | best 1.4516 @ 9000
  treatment C320 p0.2 step  9500 | lr 0.001 | train 1.2276 | val 1.4455 | best 1.4455 @ 9500
  treatment C320 p0.2 step 10000 | lr 0.001 | train 1.2240 | val 1.4458 | best 1.4455 @ 9500
  treatment C320 p0.2 step 10500 | lr 0.0003 | train 1.1814 | val 1.4060 | best 1.4060 @ 10500
  treatment C320 p0.2 step 11000 | lr 0.0003 | train 1.1729 | val 1.4035 | best 1.4035 @ 11000
  treatment C320 p0.2 step 11500 | lr 0.0003 | train 1.1646 | val 1.3985 | best 1.3985 @ 11500
  treatment C320 p0.2 step 12000 | lr 0.0003 | train 1.1590 | val 1.3943 | best 1.3943 @ 12000
  treatment C320 p0.2 step 12500 | lr 0.0003 | train 1.1540 | val 1.3928 | best 1.3928 @ 12500
  treatment C320 p0.2 step 13000 | lr 0.0003 | train 1.1535 | val 1.3932 | best 1.3928 @ 12500
  treatment C320 p0.2 step 13500 | lr 0.0001 | train 1.1399 | val 1.3832 | best 1.3832 @ 13500
  treatment C320 p0.2 step 14000 | lr 0.0001 | train 1.1363 | val 1.3812 | best 1.3812 @ 14000
  treatment C320 p0.2 step 14500 | lr 0.0001 | train 1.1330 | val 1.3809 | best 1.3809 @ 14500
  treatment C320 p0.2 step 15000 | lr 0.0001 | train 1.1310 | val 1.3809 | best 1.3809 @ 14500
  treatment C320 p0.2 step 15500 | lr 0.0001 | train 1.1290 | val 1.3801 | best 1.3801 @ 15500
  treatment C320 p0.2 step 16000 | lr 0.0001 | train 1.1262 | val 1.3782 | best 1.3782 @ 16000
  treatment C320 p0.2 step 16500 | lr 0.0001 | train 1.1244 | val 1.3805 | best 1.3782 @ 16000
  treatment C320 p0.2 step 17000 | lr 0.0001 | train 1.1227 | val 1.3785 | best 1.3782 @ 16000
  treatment C320 p0.2 step 17500 | lr 0.0001 | train 1.1207 | val 1.3778 | best 1.3778 @ 17500
  treatment C320 p0.2 step 18000 | lr 0.0001 | train 1.1191 | val 1.3758 | best 1.3758 @ 18000
  treatment C320 p0.2 step 18500 | lr 3e-05 | train 1.1167 | val 1.3745 | best 1.3745 @ 18500
  treatment C320 p0.2 step 19000 | lr 3e-05 | train 1.1154 | val 1.3741 | best 1.3741 @ 19000
  treatment C320 p0.2 step 19500 | lr 3e-05 | train 1.1142 | val 1.3722 | best 1.3722 @ 19500
  treatment C320 p0.2 step 20000 | lr 3e-05 | train 1.1134 | val 1.3727 | best 1.3722 @ 19500
  treatment C320 p0.2 step 20500 | lr 3e-05 | train 1.1127 | val 1.3718 | best 1.3718 @ 20500
  treatment C320 p0.2 step 21000 | lr 3e-05 | train 1.1123 | val 1.3724 | best 1.3718 @ 20500
  treatment C320 p0.2 step 21500 | lr 3e-05 | train 1.1118 | val 1.3721 | best 1.3718 @ 20500
  treatment C320 p0.2 step 22000 | lr 3e-05 | train 1.1111 | val 1.3716 | best 1.3716 @ 22000
  treatment C320 p0.2 step 22500 | lr 3e-05 | train 1.1107 | val 1.3716 | best 1.3716 @ 22500
  treatment C320 p0.2 step 23000 | lr 3e-05 | train 1.1100 | val 1.3722 | best 1.3716 @ 22500
  treatment C320 p0.2 step 23500 | lr 3e-05 | train 1.1091 | val 1.3695 | best 1.3695 @ 23500
  treatment C320 p0.2 step 24000 | lr 3e-05 | train 1.1086 | val 1.3694 | best 1.3694 @ 24000
  treatment C320 p0.2 step 24500 | lr 3e-05 | train 1.1079 | val 1.3696 | best 1.3694 @ 24000
  treatment C320 p0.2 step 25000 | lr 3e-05 | train 1.1074 | val 1.3694 | best 1.3694 @ 25000
  treatment C320 p0.2 step 25500 | lr 3e-05 | train 1.1070 | val 1.3686 | best 1.3686 @ 25500
  treatment C320 p0.2 step 26000 | lr 3e-05 | train 1.1067 | val 1.3691 | best 1.3686 @ 25500
  treatment C320 p0.2 step 26500 | lr 3e-05 | train 1.1062 | val 1.3678 | best 1.3678 @ 26500
  treatment C320 p0.2 step 27000 | lr 3e-05 | train 1.1055 | val 1.3679 | best 1.3678 @ 26500
  treatment C320 p0.2 step 27500 | lr 3e-05 | train 1.1048 | val 1.3680 | best 1.3678 @ 26500
  treatment C320 p0.2 step 28000 | lr 3e-05 | train 1.1045 | val 1.3684 | best 1.3678 @ 26500
  treatment C320 p0.2 step 28500 | lr 3e-05 | train 1.1037 | val 1.3674 | best 1.3674 @ 28500
  treatment C320 p0.2 step 29000 | lr 3e-05 | train 1.1036 | val 1.3679 | best 1.3674 @ 28500
  treatment C320 p0.2 step 29500 | lr 3e-05 | train 1.1032 | val 1.3670 | best 1.3670 @ 29500
  treatment C320 p0.2 step 30000 | lr 3e-05 | train 1.1027 | val 1.3661 | best 1.3661 @ 30000
  treatment C320 p0.2 step 30500 | lr 3e-05 | train 1.1024 | val 1.3667 | best 1.3661 @ 30000
  treatment C320 p0.2 step 31000 | lr 3e-05 | train 1.1018 | val 1.3660 | best 1.3660 @ 31000
  treatment C320 p0.2 step 31500 | lr 3e-05 | train 1.1013 | val 1.3661 | best 1.3660 @ 31000
  treatment C320 p0.2 step 32000 | lr 3e-05 | train 1.1006 | val 1.3652 | best 1.3652 @ 32000
  treatment C320 p0.2 step 32500 | lr 3e-05 | train 1.1003 | val 1.3645 | best 1.3645 @ 32500
  treatment C320 p0.2 step 33000 | lr 3e-05 | train 1.0994 | val 1.3664 | best 1.3645 @ 32500
  treatment C320 p0.2 step 33500 | lr 3e-05 | train 1.0992 | val 1.3652 | best 1.3645 @ 32500
  treatment C320 p0.2 step 34000 | lr 3e-05 | train 1.0989 | val 1.3642 | best 1.3642 @ 34000
  treatment C320 p0.2 step 34500 | lr 3e-05 | train 1.0987 | val 1.3648 | best 1.3642 @ 34000
  treatment C320 p0.2 step 35000 | lr 3e-05 | train 1.0978 | val 1.3664 | best 1.3642 @ 34000
  treatment C320 p0.2 step 35500 | lr 3e-05 | train 1.0974 | val 1.3655 | best 1.3642 @ 34000
  treatment C320 p0.2 step 36000 | lr 3e-05 | train 1.0968 | val 1.3640 | best 1.3640 @ 36000
  treatment C320 p0.2 step 36500 | lr 3e-05 | train 1.0969 | val 1.3650 | best 1.3640 @ 36000
  treatment C320 p0.2 step 37000 | lr 3e-05 | train 1.0963 | val 1.3642 | best 1.3640 @ 36000
  treatment C320 p0.2 step 37500 | lr 3e-05 | train 1.0958 | val 1.3633 | best 1.3633 @ 37500
  treatment C320 p0.2 step 38000 | lr 3e-05 | train 1.0951 | val 1.3642 | best 1.3633 @ 37500
  treatment C320 p0.2 step 38500 | lr 3e-05 | train 1.0947 | val 1.3627 | best 1.3627 @ 38500
  treatment C320 p0.2 step 39000 | lr 3e-05 | train 1.0945 | val 1.3640 | best 1.3627 @ 38500
  treatment C320 p0.2 step 39500 | lr 3e-05 | train 1.0938 | val 1.3635 | best 1.3627 @ 38500
  treatment C320 p0.2 step 40000 | lr 3e-05 | train 1.0937 | val 1.3634 | best 1.3627 @ 38500
  treatment C320 p0.2 step 40500 | lr 3e-05 | train 1.0933 | val 1.3635 | best 1.3627 @ 38500
  treatment C320 p0.2 step 41000 | lr 3e-05 | train 1.0924 | val 1.3625 | best 1.3625 @ 41000
  treatment C320 p0.2 step 41500 | lr 3e-05 | train 1.0923 | val 1.3632 | best 1.3625 @ 41000
  treatment C320 p0.2 step 42000 | lr 3e-05 | train 1.0916 | val 1.3630 | best 1.3625 @ 41000
  treatment C320 p0.2 step 42500 | lr 3e-05 | train 1.0910 | val 1.3615 | best 1.3615 @ 42500
  treatment C320 p0.2 step 43000 | lr 3e-05 | train 1.0910 | val 1.3625 | best 1.3615 @ 42500
  treatment C320 p0.2 step 43500 | lr 3e-05 | train 1.0902 | val 1.3615 | best 1.3615 @ 42500
  treatment C320 p0.2 step 44000 | lr 3e-05 | train 1.0903 | val 1.3626 | best 1.3615 @ 42500
  treatment C320 p0.2 step 44500 | lr 3e-05 | train 1.0896 | val 1.3616 | best 1.3615 @ 42500
  treatment C320 p0.2 step 45000 | lr 3e-05 | train 1.0890 | val 1.3625 | best 1.3615 @ 42500
  treatment C320 p0.2 step 45500 | lr 3e-05 | train 1.0888 | val 1.3613 | best 1.3613 @ 45500
  treatment C320 p0.2 step 46000 | lr 3e-05 | train 1.0886 | val 1.3618 | best 1.3613 @ 45500
  treatment C320 p0.2 step 46500 | lr 3e-05 | train 1.0880 | val 1.3613 | best 1.3613 @ 45500
  treatment C320 p0.2 step 47000 | lr 3e-05 | train 1.0878 | val 1.3631 | best 1.3613 @ 45500
  treatment C320 p0.2 step 47500 | lr 3e-05 | train 1.0873 | val 1.3617 | best 1.3613 @ 45500
  treatment C320 p0.2 step 48000 | lr 3e-05 | train 1.0867 | val 1.3621 | best 1.3613 @ 45500
  treatment C320 p0.2 step 48500 | lr 3e-05 | train 1.0865 | val 1.3601 | best 1.3601 @ 48500
  treatment C320 p0.2 step 49000 | lr 3e-05 | train 1.0862 | val 1.3611 | best 1.3601 @ 48500
  treatment C320 p0.2 step 49500 | lr 3e-05 | train 1.0858 | val 1.3598 | best 1.3598 @ 49500
  treatment C320 p0.2 step 50000 | lr 3e-05 | train 1.0857 | val 1.3602 | best 1.3598 @ 49500
  control C320 p0.15 @ 50000: 1.3611457
  treatment C320 p0.2 @ 50000: 1.3611574
  paired candidate - baseline: +0.0000117; SE 0.0007087; 95% panel interval [-0.0013807, +0.0014042]
unresolved within paired panel uncertainty; this does not prove equivalence. Report: my-gpt/checkpoints/stage_47_report.json
```
