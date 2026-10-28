# Stage 53 script output

- Script: `53_paired_width.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 53: paired width C320 vs C256 | device NVIDIA GeForce RTX 3090

Full-split loss per seed set (C320 - C256, positive favours C256):
seed   trained on                       C256       C320      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090      1.344749   1.352732  +0.007983  [+0.005783, +0.010183]
seed1  NVIDIA GeForce RTX 3090      1.347499   1.337010  -0.010489  [-0.012036, -0.008941]
seed2  NVIDIA GeForce RTX 3090      1.350152   1.345714  -0.004438  [-0.006370, -0.002506]
seed3  NVIDIA GeForce RTX 3090      1.355584   1.345020  -0.010564  [-0.012250, -0.008877]
seed4  NVIDIA GeForce RTX 3090      1.353974   1.345406  -0.008567  [-0.010189, -0.006945]
seed5  NVIDIA GeForce RTX 3090      1.352373   1.349327  -0.003046  [-0.004741, -0.001351]
seed6  NVIDIA GeForce RTX 3090 Ti   1.353963   1.348643  -0.005320  [-0.007057, -0.003583]
seed7  NVIDIA GeForce RTX 3090 Ti   1.349862   1.342572  -0.007290  [-0.008963, -0.005617]

Across 8 seed sets: mean -0.005216 | sd 0.005994 | 95% seed interval [-0.010227, -0.000206] | C256 better in 1/8

Report: my-gpt/checkpoints/stage_53_report.json
```
