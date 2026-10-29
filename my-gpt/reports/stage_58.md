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
seed6  NVIDIA GeForce RTX 3090 Ti    wd 0.01   1.348643   1.066159   0.282483
seed6  NVIDIA GeForce RTX 3090 Ti     wd 0.1   1.336357   1.061420   0.274937
seed7  NVIDIA GeForce RTX 3090 Ti    wd 0.01   1.342572   1.064126   0.278446
seed7  NVIDIA GeForce RTX 3090 Ti     wd 0.1   1.341454   1.062231   0.279223

C320 wd0.1 - C320 wd0.01 champion (negative favours wd0.1):
seed   trained on                      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090     -0.005677  [-0.007175, -0.004179]
seed1  NVIDIA GeForce RTX 3090     +0.002380  [+0.000326, +0.004434]
seed6  NVIDIA GeForce RTX 3090 Ti  -0.012285  [-0.014155, -0.010416]
seed7  NVIDIA GeForce RTX 3090 Ti  -0.001118  [-0.002839, +0.000603]
Across 4 seed sets: mean -0.004175 | sd 0.006334 | 95% seed interval [-0.014254, +0.005904] | wd0.1 better than wd0.01 in 3/4

Seed-to-seed sd of val loss over 4 seed sets:
C320 wd0.1 0.004509 | C320 wd0.01 0.006894

Mean generalization gap (val - train):
C320 wd0.1 0.277554 | C320 wd0.01 0.279292

Report: my-gpt/checkpoints/stage_58_report.json
```
