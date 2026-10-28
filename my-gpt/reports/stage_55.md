# Stage 55 script output

- Script: `55_paired_dropout.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 55: paired dropout at C320 | device NVIDIA GeForce RTX 3090

p010 vs p015 at C320 (negative favours p010):
seed   trained on                       p015       p010      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090      1.352732   1.357571  +0.004839  [+0.003133, +0.006546]
seed1  NVIDIA GeForce RTX 3090      1.337010   1.347386  +0.010376  [+0.008375, +0.012377]
seed6  NVIDIA GeForce RTX 3090 Ti   1.348643   1.344108  -0.004534  [-0.006217, -0.002852]
seed7  NVIDIA GeForce RTX 3090 Ti   1.342572   1.355770  +0.013198  [+0.011105, +0.015291]
Across 4 seed sets: mean +0.005970 | sd 0.007816 | 95% seed interval [-0.006467, +0.018407] | p015 better in 3/4

p020 vs p015 at C320 (negative favours p020):
seed   trained on                       p015       p020      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090      1.352732   1.352873  +0.000141  [-0.001515, +0.001798]
seed1  NVIDIA GeForce RTX 3090      1.337010   1.346908  +0.009898  [+0.008237, +0.011558]
seed6  NVIDIA GeForce RTX 3090 Ti   1.348643   1.344427  -0.004215  [-0.005970, -0.002461]
seed7  NVIDIA GeForce RTX 3090 Ti   1.342572   1.344179  +0.001607  [+0.000051, +0.003163]
Across 4 seed sets: mean +0.001858 | sd 0.005903 | 95% seed interval [-0.007535, +0.011250] | p015 better in 3/4

Report: my-gpt/checkpoints/stage_55_report.json
```
