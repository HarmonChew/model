# Stage 58 script output

- Script: `58_weight_decay.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 58: C320 weight decay 0.1 | device NVIDIA GeForce RTX 3090

Full-split loss per seed set (val | train | gap = val - train):
seed   trained on                        arm        val      train        gap
seed0  NVIDIA GeForce RTX 3090       wd 0.01   1.352732   1.071134   0.281598
seed0  NVIDIA GeForce RTX 3090        wd 0.1   1.347055   1.067135   0.279920
seed1  NVIDIA GeForce RTX 3090       wd 0.01   1.337010   1.062369   0.274641
seed1  NVIDIA GeForce RTX 3090        wd 0.1   1.339390   1.063254   0.276137
seed2  NVIDIA GeForce RTX 3090       wd 0.01   1.345714   1.064869   0.280845
seed2  NVIDIA GeForce RTX 3090        wd 0.1   1.345430   1.063878   0.281552
seed3  NVIDIA GeForce RTX 3090       wd 0.01   1.345020   1.065965   0.279055
seed3  NVIDIA GeForce RTX 3090        wd 0.1   1.341880   1.064218   0.277662
seed4  NVIDIA GeForce RTX 3090       wd 0.01   1.345406   1.065126   0.280280
seed4  NVIDIA GeForce RTX 3090        wd 0.1   1.341658   1.063443   0.278216
seed5  NVIDIA GeForce RTX 3090       wd 0.01   1.349327   1.067301   0.282026
seed5  NVIDIA GeForce RTX 3090        wd 0.1   1.354928   1.070470   0.284458
seed6  NVIDIA GeForce RTX 3090 Ti    wd 0.01   1.348643   1.066159   0.282483
seed6  NVIDIA GeForce RTX 3090 Ti     wd 0.1   1.336357   1.061420   0.274937
seed7  NVIDIA GeForce RTX 3090 Ti    wd 0.01   1.342572   1.064126   0.278446
seed7  NVIDIA GeForce RTX 3090 Ti     wd 0.1   1.341454   1.062231   0.279223

C320 wd0.1 - C320 wd0.01 champion (negative favours wd0.1):
seed   trained on                      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090     -0.005677  [-0.007175, -0.004179]
seed1  NVIDIA GeForce RTX 3090     +0.002380  [+0.000326, +0.004434]
seed2  NVIDIA GeForce RTX 3090     -0.000284  [-0.001999, +0.001431]
seed3  NVIDIA GeForce RTX 3090     -0.003140  [-0.004934, -0.001346]
seed4  NVIDIA GeForce RTX 3090     -0.003748  [-0.005296, -0.002200]
seed5  NVIDIA GeForce RTX 3090     +0.005600  [+0.003796, +0.007405]
seed6  NVIDIA GeForce RTX 3090 Ti  -0.012285  [-0.014155, -0.010416]
seed7  NVIDIA GeForce RTX 3090 Ti  -0.001118  [-0.002839, +0.000603]
Across 8 seed sets: mean -0.002284 | sd 0.005394 | 95% seed interval [-0.006794, +0.002226] | wd0.1 better than wd0.01 in 6/8

Seed-to-seed sd of val loss over 8 seed sets:
C320 wd0.1 0.005670 | C320 wd0.01 0.004736

Mean generalization gap (val - train):
C320 wd0.1 0.279013 | C320 wd0.01 0.279922

Report: my-gpt/checkpoints/stage_58_report.json
```
