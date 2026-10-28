# Stage 56 script output

- Script: `56_learning_rate_width.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 56: C384 peak learning rate 0.0005 | device NVIDIA GeForce RTX 3090

Full-split loss per seed set:
seed   trained on                  C320 1e-3  C384 1e-3  C384 5e-4
seed0  NVIDIA GeForce RTX 3090      1.352732   1.345260   1.348824
seed1  NVIDIA GeForce RTX 3090      1.337010   1.358969   1.346647
seed6  NVIDIA GeForce RTX 3090 Ti   1.348643   1.361670   1.344523
seed7  NVIDIA GeForce RTX 3090 Ti   1.342572   1.342669   1.346184

C384 lr5e-4 - C384 lr1e-3 (negative favours C384 lr5e-4):
seed   trained on                      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090     +0.003565  [+0.001716, +0.005413]
seed1  NVIDIA GeForce RTX 3090     -0.012322  [-0.014051, -0.010593]
seed6  NVIDIA GeForce RTX 3090 Ti  -0.017147  [-0.019657, -0.014638]
seed7  NVIDIA GeForce RTX 3090 Ti  +0.003515  [+0.001344, +0.005687]
Across 4 seed sets: mean -0.005597 | sd 0.010733 | 95% seed interval [-0.022676, +0.011482] | C384 lr5e-4 better than C384 lr1e-3 in 2/4

C384 lr5e-4 - C320 lr1e-3 champion (negative favours C384 lr5e-4):
seed   trained on                      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090     -0.003908  [-0.005822, -0.001994]
seed1  NVIDIA GeForce RTX 3090     +0.009637  [+0.007707, +0.011566]
seed6  NVIDIA GeForce RTX 3090 Ti  -0.004120  [-0.006172, -0.002068]
seed7  NVIDIA GeForce RTX 3090 Ti  +0.003613  [+0.001797, +0.005429]
Across 4 seed sets: mean +0.001305 | sd 0.006617 | 95% seed interval [-0.009224, +0.011834] | C384 lr5e-4 better than C320 lr1e-3 in 2/4

Seed-to-seed sd of loss over 4 seed sets:
C384 lr5e-4 0.001772 | C384 lr1e-3 (Stage 54) 0.009565 | C320 lr1e-3 0.006894

Report: my-gpt/checkpoints/stage_56_report.json
```
