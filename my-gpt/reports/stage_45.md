# Stage 45 script output

- Script: `45_width_scaling.py`
- Run time (seconds): `2173`
- Device: `cuda`
- Peak GPU utilization: `99%`
- Peak GPU memory: `21896 MiB`
- Exit code: `0`
- Run status: `ok`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
{
  "stage": 45,
  "protocol": "paired_architecture_native_width_scaling",
  "training_executed": false,
  "branches": [
    {
      "architecture": {
        "n_embd": 256,
        "n_head": 4,
        "n_layer": 6,
        "head_size": 64,
        "feed_forward_width": 1024,
        "block_size": 256,
        "residual_dropout": 0.1,
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
        "origin_stage": 45,
        "step": 0,
        "seed": 1337,
        "device": "cpu",
        "state_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        "policy": "Architecture-native initializations recreated from the same seed; their tensors are not claimed byte-identical.",
        "cross_width_tensor_identity_claimed": false,
        "historical_weights_loaded": false,
        "historical_optimizer_loaded": false,
        "optimizer_state_entries_at_initialization": 0,
        "parameter_count": 4850274
      },
      "initial_streams": {
        "model_sha256": "bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e",
        "optimizer_sha256": "c5ffcf6319b0bad4abbbeb62670b300e06483d88a13730c82ca9fb123eff7155",
        "training_generator_sha256": "a209d256b1a57c186263709545dea10bb4c49dc1173a3aae9376012fcd3f1ced",
        "torch_rng_sha256": "9940cf9e44812cb470574f2f6398e7f8809a5dc05d1f5f32f5832427f6b8bfc3"
      }
    },
    {
      "architecture": {
        "n_embd": 320,
        "n_head": 4,
        "n_layer": 6,
        "head_size": 80,
        "feed_forward_width": 1280,
        "block_size": 256,
        "residual_dropout": 0.1,
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
        "origin_stage": 45,
        "step": 0,
        "seed": 1337,
        "device": "cpu",
        "state_sha256": "f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98",
        "policy": "Architecture-native initializations recreated from the same seed; their tensors are not claimed byte-identical.",
        "cross_width_tensor_identity_claimed": false,
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
  "updates_per_branch": 50000,
  "target_presentations_per_branch": 102400000,
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
  "initialization_policy": "Architecture-native initializations recreated from the same seed; their tensors are not claimed byte-identical.",
  "randomness_policy": "Both branches begin from the same CPU and accelerator training RNG states, reset after native initialization. Different widths can subsequently consume different draws, including before later equal-shape calls. Neither identical dropout masks nor identical terminal Torch RNG states are claimed.",
  "historical_lineage": {
    "stage_41_authenticated_44k_ancestor_sha256": "666c9e72345399cf63b412128a1427d1c778d58794473c0e8503516d2235db36",
    "stage_42_empirical_best": {
      "branch": "control_3e_5",
      "step": 50000,
      "best_checkpoint_sha256": "1a66c6bcae77aede45af77432d93cdf8b4c51b72a100fd04b6cdb84cafbe39e6",
      "final_checkpoint_sha256": "57024bd7bb9e698ba3c16c23533aa5e5d52562eebe87841ac25a992def2b9baf",
      "fixed_panel_train_loss": 1.1931665575504302,
      "fixed_panel_validation_loss": 1.4177960669994354
    },
    "stage_43_empirical_best": {
      "branch": "treatment_c192",
      "step": 50000,
      "best_step": 48500,
      "best_checkpoint_sha256": "de624fde73bef1a641d971809041ed6c1779270b6d0e3371a75e9fbafaf2aaa4",
      "final_checkpoint_sha256": "7506703a1643510812a1bbbb94a6a3327073dc77be8397d6a25c69297a4248f5",
      "fixed_panel_train_loss": 1.121634880900383,
      "fixed_panel_validation_loss": 1.3692117285728456
    },
    "stage_44_empirical_best": {
      "branch": "treatment_c256",
      "step": 50000,
      "best_step": 48500,
      "best_checkpoint_sha256": "65af019d75a84cbef7cb589a50c7474d0324c669c484d9be629ec188811bfea3",
      "final_checkpoint_sha256": "c9e0092cb1986885f8cef948b703bb4fbf18ae0714e4e3d062fecc337081b247",
      "fixed_panel_train_loss": 1.0690209597349167,
      "fixed_panel_validation_loss": 1.3538347363471985
    },
    "role": "historical reference only; none is a Stage 45 initialization"
  }
}
Stage 45 on cuda: native C256/C320, 50,000 updates per branch
  control C256 step     0 | lr 0.001 | train 4.7088 | val 4.6970 | best 4.6970 @ 0
  control C256 step   500 | lr 0.001 | train 1.9389 | val 1.9972 | best 1.9972 @ 500
  control C256 step  1000 | lr 0.001 | train 1.6832 | val 1.7755 | best 1.7755 @ 1000
  control C256 step  1500 | lr 0.001 | train 1.5659 | val 1.6959 | best 1.6959 @ 1500
  control C256 step  2000 | lr 0.001 | train 1.4871 | val 1.6425 | best 1.6425 @ 2000
  control C256 step  2500 | lr 0.001 | train 1.4356 | val 1.6020 | best 1.6020 @ 2500
  control C256 step  3000 | lr 0.001 | train 1.3983 | val 1.5829 | best 1.5829 @ 3000
  control C256 step  3500 | lr 0.001 | train 1.3704 | val 1.5464 | best 1.5464 @ 3500
  control C256 step  4000 | lr 0.001 | train 1.3441 | val 1.5356 | best 1.5356 @ 4000
  control C256 step  4500 | lr 0.001 | train 1.3249 | val 1.5205 | best 1.5205 @ 4500
  control C256 step  5000 | lr 0.001 | train 1.3116 | val 1.5098 | best 1.5098 @ 5000
  control C256 step  5500 | lr 0.001 | train 1.2931 | val 1.4960 | best 1.4960 @ 5500
  control C256 step  6000 | lr 0.001 | train 1.2812 | val 1.4851 | best 1.4851 @ 6000
  control C256 step  6500 | lr 0.001 | train 1.2671 | val 1.4738 | best 1.4738 @ 6500
  control C256 step  7000 | lr 0.001 | train 1.2552 | val 1.4640 | best 1.4640 @ 7000
  control C256 step  7500 | lr 0.001 | train 1.2444 | val 1.4530 | best 1.4530 @ 7500
  control C256 step  8000 | lr 0.001 | train 1.2401 | val 1.4565 | best 1.4530 @ 7500
  control C256 step  8500 | lr 0.001 | train 1.2295 | val 1.4429 | best 1.4429 @ 8500
  control C256 step  9000 | lr 0.001 | train 1.2216 | val 1.4405 | best 1.4405 @ 9000
  control C256 step  9500 | lr 0.001 | train 1.2151 | val 1.4327 | best 1.4327 @ 9500
  control C256 step 10000 | lr 0.001 | train 1.2097 | val 1.4267 | best 1.4267 @ 10000
  control C256 step 10500 | lr 0.0003 | train 1.1665 | val 1.3902 | best 1.3902 @ 10500
  control C256 step 11000 | lr 0.0003 | train 1.1570 | val 1.3897 | best 1.3897 @ 11000
  control C256 step 11500 | lr 0.0003 | train 1.1508 | val 1.3840 | best 1.3840 @ 11500
  control C256 step 12000 | lr 0.0003 | train 1.1438 | val 1.3809 | best 1.3809 @ 12000
  control C256 step 12500 | lr 0.0003 | train 1.1396 | val 1.3810 | best 1.3809 @ 12000
  control C256 step 13000 | lr 0.0003 | train 1.1376 | val 1.3812 | best 1.3809 @ 12000
  control C256 step 13500 | lr 0.0001 | train 1.1242 | val 1.3698 | best 1.3698 @ 13500
  control C256 step 14000 | lr 0.0001 | train 1.1201 | val 1.3685 | best 1.3685 @ 14000
  control C256 step 14500 | lr 0.0001 | train 1.1171 | val 1.3688 | best 1.3685 @ 14000
  control C256 step 15000 | lr 0.0001 | train 1.1155 | val 1.3699 | best 1.3685 @ 14000
  control C256 step 15500 | lr 0.0001 | train 1.1131 | val 1.3680 | best 1.3680 @ 15500
  control C256 step 16000 | lr 0.0001 | train 1.1105 | val 1.3669 | best 1.3669 @ 16000
  control C256 step 16500 | lr 0.0001 | train 1.1092 | val 1.3665 | best 1.3665 @ 16500
  control C256 step 17000 | lr 0.0001 | train 1.1069 | val 1.3658 | best 1.3658 @ 17000
  control C256 step 17500 | lr 0.0001 | train 1.1052 | val 1.3666 | best 1.3658 @ 17000
  control C256 step 18000 | lr 0.0001 | train 1.1042 | val 1.3638 | best 1.3638 @ 18000
  control C256 step 18500 | lr 3e-05 | train 1.1011 | val 1.3622 | best 1.3622 @ 18500
  control C256 step 19000 | lr 3e-05 | train 1.0997 | val 1.3621 | best 1.3621 @ 19000
  control C256 step 19500 | lr 3e-05 | train 1.0986 | val 1.3601 | best 1.3601 @ 19500
  control C256 step 20000 | lr 3e-05 | train 1.0978 | val 1.3604 | best 1.3601 @ 19500
  control C256 step 20500 | lr 3e-05 | train 1.0975 | val 1.3596 | best 1.3596 @ 20500
  control C256 step 21000 | lr 3e-05 | train 1.0963 | val 1.3595 | best 1.3595 @ 21000
  control C256 step 21500 | lr 3e-05 | train 1.0960 | val 1.3591 | best 1.3591 @ 21500
  control C256 step 22000 | lr 3e-05 | train 1.0954 | val 1.3597 | best 1.3591 @ 21500
  control C256 step 22500 | lr 3e-05 | train 1.0948 | val 1.3596 | best 1.3591 @ 21500
  control C256 step 23000 | lr 3e-05 | train 1.0940 | val 1.3600 | best 1.3591 @ 21500
  control C256 step 23500 | lr 3e-05 | train 1.0933 | val 1.3579 | best 1.3579 @ 23500
  control C256 step 24000 | lr 3e-05 | train 1.0926 | val 1.3576 | best 1.3576 @ 24000
  control C256 step 24500 | lr 3e-05 | train 1.0921 | val 1.3583 | best 1.3576 @ 24000
  control C256 step 25000 | lr 3e-05 | train 1.0915 | val 1.3572 | best 1.3572 @ 25000
  control C256 step 25500 | lr 3e-05 | train 1.0909 | val 1.3575 | best 1.3572 @ 25000
  control C256 step 26000 | lr 3e-05 | train 1.0906 | val 1.3580 | best 1.3572 @ 25000
  control C256 step 26500 | lr 3e-05 | train 1.0900 | val 1.3569 | best 1.3569 @ 26500
  control C256 step 27000 | lr 3e-05 | train 1.0896 | val 1.3576 | best 1.3569 @ 26500
  control C256 step 27500 | lr 3e-05 | train 1.0888 | val 1.3569 | best 1.3569 @ 26500
  control C256 step 28000 | lr 3e-05 | train 1.0886 | val 1.3570 | best 1.3569 @ 26500
  control C256 step 28500 | lr 3e-05 | train 1.0876 | val 1.3566 | best 1.3566 @ 28500
  control C256 step 29000 | lr 3e-05 | train 1.0872 | val 1.3570 | best 1.3566 @ 28500
  control C256 step 29500 | lr 3e-05 | train 1.0869 | val 1.3564 | best 1.3564 @ 29500
  control C256 step 30000 | lr 3e-05 | train 1.0864 | val 1.3563 | best 1.3563 @ 30000
  control C256 step 30500 | lr 3e-05 | train 1.0862 | val 1.3564 | best 1.3563 @ 30000
  control C256 step 31000 | lr 3e-05 | train 1.0858 | val 1.3563 | best 1.3563 @ 30000
  control C256 step 31500 | lr 3e-05 | train 1.0852 | val 1.3564 | best 1.3563 @ 30000
  control C256 step 32000 | lr 3e-05 | train 1.0844 | val 1.3554 | best 1.3554 @ 32000
  control C256 step 32500 | lr 3e-05 | train 1.0843 | val 1.3553 | best 1.3553 @ 32500
  control C256 step 33000 | lr 3e-05 | train 1.0836 | val 1.3565 | best 1.3553 @ 32500
  control C256 step 33500 | lr 3e-05 | train 1.0831 | val 1.3561 | best 1.3553 @ 32500
  control C256 step 34000 | lr 3e-05 | train 1.0828 | val 1.3550 | best 1.3550 @ 34000
  control C256 step 34500 | lr 3e-05 | train 1.0823 | val 1.3557 | best 1.3550 @ 34000
  control C256 step 35000 | lr 3e-05 | train 1.0816 | val 1.3566 | best 1.3550 @ 34000
  control C256 step 35500 | lr 3e-05 | train 1.0810 | val 1.3563 | best 1.3550 @ 34000
  control C256 step 36000 | lr 3e-05 | train 1.0808 | val 1.3557 | best 1.3550 @ 34000
  control C256 step 36500 | lr 3e-05 | train 1.0806 | val 1.3559 | best 1.3550 @ 34000
  control C256 step 37000 | lr 3e-05 | train 1.0800 | val 1.3555 | best 1.3550 @ 34000
  control C256 step 37500 | lr 3e-05 | train 1.0794 | val 1.3552 | best 1.3550 @ 34000
  control C256 step 38000 | lr 3e-05 | train 1.0788 | val 1.3546 | best 1.3546 @ 38000
  control C256 step 38500 | lr 3e-05 | train 1.0784 | val 1.3537 | best 1.3537 @ 38500
  control C256 step 39000 | lr 3e-05 | train 1.0780 | val 1.3536 | best 1.3536 @ 39000
  control C256 step 39500 | lr 3e-05 | train 1.0776 | val 1.3543 | best 1.3536 @ 39000
  control C256 step 40000 | lr 3e-05 | train 1.0773 | val 1.3550 | best 1.3536 @ 39000
  control C256 step 40500 | lr 3e-05 | train 1.0771 | val 1.3553 | best 1.3536 @ 39000
  control C256 step 41000 | lr 3e-05 | train 1.0762 | val 1.3544 | best 1.3536 @ 39000
  control C256 step 41500 | lr 3e-05 | train 1.0761 | val 1.3548 | best 1.3536 @ 39000
  control C256 step 42000 | lr 3e-05 | train 1.0752 | val 1.3542 | best 1.3536 @ 39000
  control C256 step 42500 | lr 3e-05 | train 1.0749 | val 1.3531 | best 1.3531 @ 42500
  control C256 step 43000 | lr 3e-05 | train 1.0748 | val 1.3550 | best 1.3531 @ 42500
  control C256 step 43500 | lr 3e-05 | train 1.0740 | val 1.3544 | best 1.3531 @ 42500
  control C256 step 44000 | lr 3e-05 | train 1.0737 | val 1.3552 | best 1.3531 @ 42500
  control C256 step 44500 | lr 3e-05 | train 1.0732 | val 1.3549 | best 1.3531 @ 42500
  control C256 step 45000 | lr 3e-05 | train 1.0728 | val 1.3544 | best 1.3531 @ 42500
  control C256 step 45500 | lr 3e-05 | train 1.0725 | val 1.3544 | best 1.3531 @ 42500
  control C256 step 46000 | lr 3e-05 | train 1.0722 | val 1.3543 | best 1.3531 @ 42500
  control C256 step 46500 | lr 3e-05 | train 1.0717 | val 1.3542 | best 1.3531 @ 42500
  control C256 step 47000 | lr 3e-05 | train 1.0712 | val 1.3544 | best 1.3531 @ 42500
  control C256 step 47500 | lr 3e-05 | train 1.0706 | val 1.3534 | best 1.3531 @ 42500
  control C256 step 48000 | lr 3e-05 | train 1.0702 | val 1.3537 | best 1.3531 @ 42500
  control C256 step 48500 | lr 3e-05 | train 1.0699 | val 1.3523 | best 1.3523 @ 48500
  control C256 step 49000 | lr 3e-05 | train 1.0696 | val 1.3548 | best 1.3523 @ 48500
  control C256 step 49500 | lr 3e-05 | train 1.0693 | val 1.3529 | best 1.3523 @ 48500
  control C256 step 50000 | lr 3e-05 | train 1.0690 | val 1.3538 | best 1.3523 @ 48500
  treatment C320 step     0 | lr 0.001 | train 4.7684 | val 4.7675 | best 4.7675 @ 0
  treatment C320 step   500 | lr 0.001 | train 1.8740 | val 1.9384 | best 1.9384 @ 500
  treatment C320 step  1000 | lr 0.001 | train 1.6525 | val 1.7548 | best 1.7548 @ 1000
  treatment C320 step  1500 | lr 0.001 | train 1.5481 | val 1.6796 | best 1.6796 @ 1500
  treatment C320 step  2000 | lr 0.001 | train 1.4759 | val 1.6380 | best 1.6380 @ 2000
  treatment C320 step  2500 | lr 0.001 | train 1.4345 | val 1.6173 | best 1.6173 @ 2500
  treatment C320 step  3000 | lr 0.001 | train 1.3939 | val 1.5783 | best 1.5783 @ 3000
  treatment C320 step  3500 | lr 0.001 | train 1.3661 | val 1.5398 | best 1.5398 @ 3500
  treatment C320 step  4000 | lr 0.001 | train 1.3384 | val 1.5316 | best 1.5316 @ 4000
  treatment C320 step  4500 | lr 0.001 | train 1.3206 | val 1.5128 | best 1.5128 @ 4500
  treatment C320 step  5000 | lr 0.001 | train 1.3078 | val 1.5047 | best 1.5047 @ 5000
  treatment C320 step  5500 | lr 0.001 | train 1.2885 | val 1.4869 | best 1.4869 @ 5500
  treatment C320 step  6000 | lr 0.001 | train 1.2800 | val 1.4844 | best 1.4844 @ 6000
  treatment C320 step  6500 | lr 0.001 | train 1.2646 | val 1.4719 | best 1.4719 @ 6500
  treatment C320 step  7000 | lr 0.001 | train 1.2526 | val 1.4637 | best 1.4637 @ 7000
  treatment C320 step  7500 | lr 0.001 | train 1.2427 | val 1.4534 | best 1.4534 @ 7500
  treatment C320 step  8000 | lr 0.001 | train 1.2363 | val 1.4615 | best 1.4534 @ 7500
  treatment C320 step  8500 | lr 0.001 | train 1.2283 | val 1.4445 | best 1.4445 @ 8500
  treatment C320 step  9000 | lr 0.001 | train 1.2219 | val 1.4407 | best 1.4407 @ 9000
  treatment C320 step  9500 | lr 0.001 | train 1.2137 | val 1.4330 | best 1.4330 @ 9500
  treatment C320 step 10000 | lr 0.001 | train 1.2064 | val 1.4337 | best 1.4330 @ 9500
  treatment C320 step 10500 | lr 0.0003 | train 1.1611 | val 1.3960 | best 1.3960 @ 10500
  treatment C320 step 11000 | lr 0.0003 | train 1.1509 | val 1.3928 | best 1.3928 @ 11000
  treatment C320 step 11500 | lr 0.0003 | train 1.1432 | val 1.3894 | best 1.3894 @ 11500
  treatment C320 step 12000 | lr 0.0003 | train 1.1362 | val 1.3858 | best 1.3858 @ 12000
  treatment C320 step 12500 | lr 0.0003 | train 1.1303 | val 1.3872 | best 1.3858 @ 12000
  treatment C320 step 13000 | lr 0.0003 | train 1.1296 | val 1.3868 | best 1.3858 @ 12000
  treatment C320 step 13500 | lr 0.0001 | train 1.1137 | val 1.3761 | best 1.3761 @ 13500
  treatment C320 step 14000 | lr 0.0001 | train 1.1094 | val 1.3735 | best 1.3735 @ 14000
  treatment C320 step 14500 | lr 0.0001 | train 1.1055 | val 1.3738 | best 1.3735 @ 14000
  treatment C320 step 15000 | lr 0.0001 | train 1.1037 | val 1.3744 | best 1.3735 @ 14000
  treatment C320 step 15500 | lr 0.0001 | train 1.1013 | val 1.3744 | best 1.3735 @ 14000
  treatment C320 step 16000 | lr 0.0001 | train 1.0985 | val 1.3732 | best 1.3732 @ 16000
  treatment C320 step 16500 | lr 0.0001 | train 1.0968 | val 1.3714 | best 1.3714 @ 16500
  treatment C320 step 17000 | lr 0.0001 | train 1.0940 | val 1.3719 | best 1.3714 @ 16500
  treatment C320 step 17500 | lr 0.0001 | train 1.0913 | val 1.3714 | best 1.3714 @ 17500
  treatment C320 step 18000 | lr 0.0001 | train 1.0902 | val 1.3715 | best 1.3714 @ 17500
  treatment C320 step 18500 | lr 3e-05 | train 1.0870 | val 1.3684 | best 1.3684 @ 18500
  treatment C320 step 19000 | lr 3e-05 | train 1.0851 | val 1.3682 | best 1.3682 @ 19000
  treatment C320 step 19500 | lr 3e-05 | train 1.0840 | val 1.3665 | best 1.3665 @ 19500
  treatment C320 step 20000 | lr 3e-05 | train 1.0831 | val 1.3675 | best 1.3665 @ 19500
  treatment C320 step 20500 | lr 3e-05 | train 1.0828 | val 1.3671 | best 1.3665 @ 19500
  treatment C320 step 21000 | lr 3e-05 | train 1.0818 | val 1.3676 | best 1.3665 @ 19500
  treatment C320 step 21500 | lr 3e-05 | train 1.0812 | val 1.3669 | best 1.3665 @ 19500
  treatment C320 step 22000 | lr 3e-05 | train 1.0803 | val 1.3676 | best 1.3665 @ 19500
  treatment C320 step 22500 | lr 3e-05 | train 1.0795 | val 1.3675 | best 1.3665 @ 19500
  treatment C320 step 23000 | lr 3e-05 | train 1.0789 | val 1.3672 | best 1.3665 @ 19500
  treatment C320 step 23500 | lr 3e-05 | train 1.0780 | val 1.3653 | best 1.3653 @ 23500
  treatment C320 step 24000 | lr 3e-05 | train 1.0774 | val 1.3653 | best 1.3653 @ 24000
  treatment C320 step 24500 | lr 3e-05 | train 1.0765 | val 1.3661 | best 1.3653 @ 24000
  treatment C320 step 25000 | lr 3e-05 | train 1.0758 | val 1.3654 | best 1.3653 @ 24000
  treatment C320 step 25500 | lr 3e-05 | train 1.0752 | val 1.3652 | best 1.3652 @ 25500
  treatment C320 step 26000 | lr 3e-05 | train 1.0748 | val 1.3659 | best 1.3652 @ 25500
  treatment C320 step 26500 | lr 3e-05 | train 1.0742 | val 1.3638 | best 1.3638 @ 26500
  treatment C320 step 27000 | lr 3e-05 | train 1.0735 | val 1.3646 | best 1.3638 @ 26500
  treatment C320 step 27500 | lr 3e-05 | train 1.0727 | val 1.3642 | best 1.3638 @ 26500
  treatment C320 step 28000 | lr 3e-05 | train 1.0723 | val 1.3651 | best 1.3638 @ 26500
  treatment C320 step 28500 | lr 3e-05 | train 1.0714 | val 1.3644 | best 1.3638 @ 26500
  treatment C320 step 29000 | lr 3e-05 | train 1.0711 | val 1.3650 | best 1.3638 @ 26500
  treatment C320 step 29500 | lr 3e-05 | train 1.0705 | val 1.3641 | best 1.3638 @ 26500
  treatment C320 step 30000 | lr 3e-05 | train 1.0698 | val 1.3650 | best 1.3638 @ 26500
  treatment C320 step 30500 | lr 3e-05 | train 1.0696 | val 1.3648 | best 1.3638 @ 26500
  treatment C320 step 31000 | lr 3e-05 | train 1.0691 | val 1.3639 | best 1.3638 @ 26500
  treatment C320 step 31500 | lr 3e-05 | train 1.0685 | val 1.3647 | best 1.3638 @ 26500
  treatment C320 step 32000 | lr 3e-05 | train 1.0676 | val 1.3647 | best 1.3638 @ 26500
  treatment C320 step 32500 | lr 3e-05 | train 1.0671 | val 1.3639 | best 1.3638 @ 26500
  treatment C320 step 33000 | lr 3e-05 | train 1.0662 | val 1.3646 | best 1.3638 @ 26500
  treatment C320 step 33500 | lr 3e-05 | train 1.0660 | val 1.3645 | best 1.3638 @ 26500
  treatment C320 step 34000 | lr 3e-05 | train 1.0653 | val 1.3640 | best 1.3638 @ 26500
  treatment C320 step 34500 | lr 3e-05 | train 1.0649 | val 1.3641 | best 1.3638 @ 26500
  treatment C320 step 35000 | lr 3e-05 | train 1.0641 | val 1.3651 | best 1.3638 @ 26500
  treatment C320 step 35500 | lr 3e-05 | train 1.0632 | val 1.3645 | best 1.3638 @ 26500
  treatment C320 step 36000 | lr 3e-05 | train 1.0629 | val 1.3635 | best 1.3635 @ 36000
  treatment C320 step 36500 | lr 3e-05 | train 1.0628 | val 1.3641 | best 1.3635 @ 36000
  treatment C320 step 37000 | lr 3e-05 | train 1.0623 | val 1.3633 | best 1.3633 @ 37000
  treatment C320 step 37500 | lr 3e-05 | train 1.0615 | val 1.3644 | best 1.3633 @ 37000
  treatment C320 step 38000 | lr 3e-05 | train 1.0609 | val 1.3639 | best 1.3633 @ 37000
  treatment C320 step 38500 | lr 3e-05 | train 1.0604 | val 1.3626 | best 1.3626 @ 38500
  treatment C320 step 39000 | lr 3e-05 | train 1.0601 | val 1.3624 | best 1.3624 @ 39000
  treatment C320 step 39500 | lr 3e-05 | train 1.0592 | val 1.3639 | best 1.3624 @ 39000
  treatment C320 step 40000 | lr 3e-05 | train 1.0588 | val 1.3638 | best 1.3624 @ 39000
  treatment C320 step 40500 | lr 3e-05 | train 1.0583 | val 1.3643 | best 1.3624 @ 39000
  treatment C320 step 41000 | lr 3e-05 | train 1.0574 | val 1.3640 | best 1.3624 @ 39000
  treatment C320 step 41500 | lr 3e-05 | train 1.0573 | val 1.3641 | best 1.3624 @ 39000
  treatment C320 step 42000 | lr 3e-05 | train 1.0564 | val 1.3649 | best 1.3624 @ 39000
  treatment C320 step 42500 | lr 3e-05 | train 1.0562 | val 1.3629 | best 1.3624 @ 39000
  treatment C320 step 43000 | lr 3e-05 | train 1.0560 | val 1.3633 | best 1.3624 @ 39000
  treatment C320 step 43500 | lr 3e-05 | train 1.0552 | val 1.3638 | best 1.3624 @ 39000
  treatment C320 step 44000 | lr 3e-05 | train 1.0549 | val 1.3636 | best 1.3624 @ 39000
  treatment C320 step 44500 | lr 3e-05 | train 1.0543 | val 1.3636 | best 1.3624 @ 39000
  treatment C320 step 45000 | lr 3e-05 | train 1.0536 | val 1.3642 | best 1.3624 @ 39000
  treatment C320 step 45500 | lr 3e-05 | train 1.0536 | val 1.3637 | best 1.3624 @ 39000
  treatment C320 step 46000 | lr 3e-05 | train 1.0530 | val 1.3641 | best 1.3624 @ 39000
  treatment C320 step 46500 | lr 3e-05 | train 1.0523 | val 1.3651 | best 1.3624 @ 39000
  treatment C320 step 47000 | lr 3e-05 | train 1.0521 | val 1.3653 | best 1.3624 @ 39000
  treatment C320 step 47500 | lr 3e-05 | train 1.0514 | val 1.3634 | best 1.3624 @ 39000
  treatment C320 step 48000 | lr 3e-05 | train 1.0507 | val 1.3642 | best 1.3624 @ 39000
  treatment C320 step 48500 | lr 3e-05 | train 1.0504 | val 1.3628 | best 1.3624 @ 39000
  treatment C320 step 49000 | lr 3e-05 | train 1.0502 | val 1.3649 | best 1.3624 @ 39000
  treatment C320 step 49500 | lr 3e-05 | train 1.0496 | val 1.3636 | best 1.3624 @ 39000
  treatment C320 step 50000 | lr 3e-05 | train 1.0493 | val 1.3650 | best 1.3624 @ 39000
  control C256 @ 50000: 1.3462395
  treatment C320 @ 50000: 1.3556985
  paired candidate - baseline: +0.0094590; SE 0.0008036; 95% panel interval [+0.0078801, +0.0110379]
C256 wins at matched 50k updates under the established schedule. Report: my-gpt/checkpoints/stage_45_report.json
```
