# Stage 53 script output

- Script: `53_paired_width.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 53: paired width C320 vs C256 | device NVIDIA GeForce RTX 3090
seed0: training C320 {'init_seed': 1337, 'batch_seed': 1338, 'rng_seed': 1340}
  seed0 update   5000 | train batch 1.3223
  seed0 update  10000 | train batch 1.2172
  seed0 update  15000 | train batch 1.1731
  seed0 update  20000 | train batch 1.1279
  seed0 update  25000 | train batch 1.0959
  seed0 update  30000 | train batch 1.0940
  seed0 update  35000 | train batch 1.1218
  seed0 update  40000 | train batch 1.1949
  seed0 update  45000 | train batch 1.0432
  seed0 update  50000 | train batch 1.1862
seed0: reproduces Stage 46 treatment, model 37a523f2ddfb
seed1: training C320 {'init_seed': 2337, 'batch_seed': 2338, 'rng_seed': 2340}
  seed1 update   5000 | train batch 1.3468
  seed1 update  10000 | train batch 1.3830
  seed1 update  15000 | train batch 1.0937
  seed1 update  20000 | train batch 1.0325
  seed1 update  25000 | train batch 1.2646
  seed1 update  30000 | train batch 1.1089
  seed1 update  35000 | train batch 1.0863
  seed1 update  40000 | train batch 1.0063
  seed1 update  45000 | train batch 1.1613
  seed1 update  50000 | train batch 1.0357
seed2: training C320 {'init_seed': 3337, 'batch_seed': 3338, 'rng_seed': 3340}
  seed2 update   5000 | train batch 1.3706
  seed2 update  10000 | train batch 1.2221
  seed2 update  15000 | train batch 0.9505
  seed2 update  20000 | train batch 1.1457
  seed2 update  25000 | train batch 1.0544
  seed2 update  30000 | train batch 1.1324
  seed2 update  35000 | train batch 1.1164
  seed2 update  40000 | train batch 1.0620
  seed2 update  45000 | train batch 1.1220
  seed2 update  50000 | train batch 1.0761
seed3: training C320 {'init_seed': 4337, 'batch_seed': 4338, 'rng_seed': 4340}
  seed3 update   5000 | train batch 1.3310
  seed3 update  10000 | train batch 1.2825
  seed3 update  15000 | train batch 1.1924
  seed3 update  20000 | train batch 1.1107
  seed3 update  25000 | train batch 1.1347
  seed3 update  30000 | train batch 1.0642
  seed3 update  35000 | train batch 1.1866
  seed3 update  40000 | train batch 1.1137
  seed3 update  45000 | train batch 1.1521
  seed3 update  50000 | train batch 0.9009

Full-split loss per seed set (C320 - C256, positive favours C256):
seed         C256       C320      delta  95% window interval
seed0    1.344749   1.352732  +0.007983  [+0.005783, +0.010183]
seed1    1.347499   1.337010  -0.010489  [-0.012036, -0.008941]
seed2    1.350152   1.345714  -0.004438  [-0.006370, -0.002506]
seed3    1.355584   1.345020  -0.010564  [-0.012250, -0.008877]

Across 4 seed sets: mean -0.004377 | sd 0.008725 | 95% seed interval [-0.018261, +0.009507] | C256 better in 1/4

Report: my-gpt/checkpoints/stage_53_report.json
```
