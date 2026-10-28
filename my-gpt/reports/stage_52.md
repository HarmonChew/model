# Stage 52 script output

- Script: `52_paired_dropout.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 52: paired dropout 0.10 vs 0.15 | device NVIDIA GeForce RTX 3090
seed0: training p=0.1 {'init_seed': 1337, 'batch_seed': 1338, 'rng_seed': 1340}
  seed0 update   5000 | train batch 1.3180
  seed0 update  10000 | train batch 1.2119
  seed0 update  15000 | train batch 1.1695
  seed0 update  20000 | train batch 1.1267
  seed0 update  25000 | train batch 1.0961
  seed0 update  30000 | train batch 1.1163
  seed0 update  35000 | train batch 1.1252
  seed0 update  40000 | train batch 1.1888
  seed0 update  45000 | train batch 1.0428
  seed0 update  50000 | train batch 1.1946
seed0: reproduces Stage 48 control, model 3c4ded6a3653
seed1: training p=0.1 {'init_seed': 2337, 'batch_seed': 2338, 'rng_seed': 2340}
  seed1 update   5000 | train batch 1.3526
  seed1 update  10000 | train batch 1.4198
  seed1 update  15000 | train batch 1.0949
  seed1 update  20000 | train batch 1.0558
  seed1 update  25000 | train batch 1.2853
  seed1 update  30000 | train batch 1.1139
  seed1 update  35000 | train batch 1.0847
  seed1 update  40000 | train batch 1.0347
  seed1 update  45000 | train batch 1.1636
  seed1 update  50000 | train batch 1.0386
seed2: training p=0.1 {'init_seed': 3337, 'batch_seed': 3338, 'rng_seed': 3340}
  seed2 update   5000 | train batch 1.3792
  seed2 update  10000 | train batch 1.2223
  seed2 update  15000 | train batch 0.9509
  seed2 update  20000 | train batch 1.1541
  seed2 update  25000 | train batch 1.0608
  seed2 update  30000 | train batch 1.1113
  seed2 update  35000 | train batch 1.1239
  seed2 update  40000 | train batch 1.0652
  seed2 update  45000 | train batch 1.1310
  seed2 update  50000 | train batch 1.0738
seed3: training p=0.1 {'init_seed': 4337, 'batch_seed': 4338, 'rng_seed': 4340}
  seed3 update   5000 | train batch 1.3481
  seed3 update  10000 | train batch 1.2911
  seed3 update  15000 | train batch 1.1995
  seed3 update  20000 | train batch 1.1220
  seed3 update  25000 | train batch 1.1337
  seed3 update  30000 | train batch 1.0502
  seed3 update  35000 | train batch 1.1919
  seed3 update  40000 | train batch 1.1174
  seed3 update  45000 | train batch 1.1517
  seed3 update  50000 | train batch 0.9004

Full-split loss per seed set (p0.15 - p0.10, negative favours p0.15):
seed        p0.10      p0.15      delta  95% window interval
seed0    1.346796   1.344749  -0.002046  [-0.003756, -0.000336]
seed1    1.347013   1.347499  +0.000486  [-0.001113, +0.002084]
seed2    1.345487   1.350152  +0.004665  [+0.003121, +0.006209]
seed3    1.348721   1.355584  +0.006863  [+0.005348, +0.008378]

Across 4 seed sets: mean +0.002492 | sd 0.004019 | 95% seed interval [-0.003903, +0.008886] | p0.15 better in 1/4

Report: my-gpt/checkpoints/stage_52_report.json
```
