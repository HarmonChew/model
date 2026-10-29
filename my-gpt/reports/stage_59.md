# Stage 59 script output

- Script: `59_half_data.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 59: C320 on half the training split (2,409,389 of 4,809,389 characters) | device NVIDIA GeForce RTX 3090

Full-split loss per seed set (val | train | gap = val - train):
seed   trained on                        arm        val      train        gap
seed0  NVIDIA GeForce RTX 3090          full   1.352732   1.071134   0.281598
seed0  NVIDIA GeForce RTX 3090          half   1.435378   0.992471   0.442908
seed1  NVIDIA GeForce RTX 3090          full   1.337010   1.062369   0.274641
seed1  NVIDIA GeForce RTX 3090          half   1.445253   0.998560   0.446693
seed6  NVIDIA GeForce RTX 3090 Ti       full   1.348643   1.066159   0.282483
seed6  NVIDIA GeForce RTX 3090 Ti       half   1.440835   1.008758   0.432077
seed7  NVIDIA GeForce RTX 3090 Ti       full   1.342572   1.064126   0.278446
seed7  NVIDIA GeForce RTX 3090 Ti       half   1.450850   0.993651   0.457199

C320 half data - C320 full data champion (positive means half data is worse):
seed   trained on                      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090     +0.082646  [+0.079120, +0.086173]
seed1  NVIDIA GeForce RTX 3090     +0.108243  [+0.103961, +0.112526]
seed6  NVIDIA GeForce RTX 3090 Ti  +0.092193  [+0.088583, +0.095803]
seed7  NVIDIA GeForce RTX 3090 Ti  +0.108278  [+0.104427, +0.112128]
Across 4 seed sets: mean +0.097840 | sd 0.012648 | 95% seed interval [+0.077714, +0.117966] | half data better than full in 0/4

Seed-to-seed sd of val loss over 4 seed sets:
C320 half 0.006569 | C320 full 0.006894

Mean generalization gap (val - train):
C320 half 0.444719 | C320 full 0.279292

Report: my-gpt/checkpoints/stage_59_report.json
```
