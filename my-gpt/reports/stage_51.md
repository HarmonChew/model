# Stage 51 script output

- Script: `51_noise_floor.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 51: noise floor | device NVIDIA GeForce RTX 3090
seed0: training {'init_seed': 1337, 'batch_seed': 1338, 'rng_seed': 1340}
  seed0 update   5000 | train batch 1.3307
  seed0 update  10000 | train batch 1.2195
  seed0 update  15000 | train batch 1.1985
  seed0 update  20000 | train batch 1.1512
  seed0 update  25000 | train batch 1.1276
  seed0 update  30000 | train batch 1.1135
  seed0 update  35000 | train batch 1.1504
  seed0 update  40000 | train batch 1.2185
  seed0 update  45000 | train batch 1.0631
  seed0 update  50000 | train batch 1.2145
seed0: reproduces Stage 49 control, model 22bcc9eecff8
seed1: training {'init_seed': 2337, 'batch_seed': 2338, 'rng_seed': 2340}
  seed1 update   5000 | train batch 1.3995
  seed1 update  10000 | train batch 1.4297
  seed1 update  15000 | train batch 1.1037
  seed1 update  20000 | train batch 1.0667
  seed1 update  25000 | train batch 1.3156
  seed1 update  30000 | train batch 1.1263
  seed1 update  35000 | train batch 1.1175
  seed1 update  40000 | train batch 1.0556
  seed1 update  45000 | train batch 1.1963
  seed1 update  50000 | train batch 1.0656
seed2: training {'init_seed': 3337, 'batch_seed': 3338, 'rng_seed': 3340}
  seed2 update   5000 | train batch 1.3942
  seed2 update  10000 | train batch 1.2387
  seed2 update  15000 | train batch 0.9666
  seed2 update  20000 | train batch 1.1827
  seed2 update  25000 | train batch 1.0730
  seed2 update  30000 | train batch 1.1376
  seed2 update  35000 | train batch 1.1480
  seed2 update  40000 | train batch 1.0904
  seed2 update  45000 | train batch 1.1443
  seed2 update  50000 | train batch 1.1072
seed3: training {'init_seed': 4337, 'batch_seed': 4338, 'rng_seed': 4340}
  seed3 update   5000 | train batch 1.3570
  seed3 update  10000 | train batch 1.3055
  seed3 update  15000 | train batch 1.2324
  seed3 update  20000 | train batch 1.1583
  seed3 update  25000 | train batch 1.1581
  seed3 update  30000 | train batch 1.0834
  seed3 update  35000 | train batch 1.2087
  seed3 update  40000 | train batch 1.1320
  seed3 update  45000 | train batch 1.2046
  seed3 update  50000 | train batch 0.9346

Full validation split: 2087 windows x 256 = 534272 targets
endpoint           trained on                     params       loss
c256_l6_p010       NVIDIA GeForce RTX 3090       4850274   1.346796
c256_l6_p015       NVIDIA GeForce RTX 3090       4850274   1.344749
c256_l6_p020       NVIDIA GeForce RTX 3090       4850274   1.351961
c320_l6_p015       NVIDIA GeForce RTX 3090       7537378   1.352732
c256_l6_p015_ti    NVIDIA GeForce RTX 3090 Ti    4850274   1.346904
c256_l8_p015_ti    NVIDIA GeForce RTX 3090 Ti    6428258   1.345339
seed0              NVIDIA GeForce RTX 3090       4850274   1.344749
seed1              NVIDIA GeForce RTX 3090       4850274   1.347499
seed2              NVIDIA GeForce RTX 3090       4850274   1.350152
seed3              NVIDIA GeForce RTX 3090       4850274   1.355584

Seed spread, C256 L=6 p=0.15 on the 3090 (4 replicates): mean 1.349496 | sd 0.004619 | range 1.344749..1.355584
Two single-seed runs must differ by more than 0.013065 (2*sqrt(2)*sd)

Historical calls, rescored on the full split (candidate - baseline):
  Stage 48 dropout 0.10 -> 0.15     -0.002046 [-0.003756, -0.000336]  within seed noise
  Stage 49 dropout 0.15 -> 0.20     +0.007212 [+0.005821, +0.008603]  within seed noise
  Stage 50 depth L=6 -> L=8 (Ti)    -0.001565 [-0.003263, +0.000132]  within seed noise
  Stage 50 device 3090 -> Ti        +0.002155 [+0.000728, +0.003582]  within seed noise
  Stage 46 width C256 -> C320       +0.007983 [+0.005783, +0.010183]  within seed noise

Report: my-gpt/checkpoints/stage_51_report.json
```
