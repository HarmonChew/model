# Stage 54 script output

- Script: `54_paired_width.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 54: paired width C384 vs C320 | device NVIDIA GeForce RTX 3090

Full-split loss per seed set (C384 - C320, positive favours C320):
seed   trained on                       C320       C384      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090      1.352732   1.345260  -0.007472  [-0.009289, -0.005656]
seed1  NVIDIA GeForce RTX 3090      1.337010   1.358969  +0.021959  [+0.019722, +0.024196]
seed6  NVIDIA GeForce RTX 3090 Ti   1.348643   1.361670  +0.013027  [+0.011018, +0.015036]
seed7  NVIDIA GeForce RTX 3090 Ti   1.342572   1.342669  +0.000097  [-0.002066, +0.002260]

Across 4 seed sets: mean +0.006903 | sd 0.013130 | 95% seed interval [-0.013989, +0.027795] | C384 better in 1/4

Report: my-gpt/checkpoints/stage_54_report.json
```
