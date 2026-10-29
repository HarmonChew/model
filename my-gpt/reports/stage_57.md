# Stage 57 script output

- Script: `57_learning_rate_champion.py`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 57: C320 peak learning rate 0.0005 | device NVIDIA GeForce RTX 3090

Full-split loss per seed set:
seed   trained on                  C320 1e-3  C320 5e-4  C384 5e-4
seed0  NVIDIA GeForce RTX 3090      1.352732   1.346524   1.348824
seed1  NVIDIA GeForce RTX 3090      1.337010   1.344303   1.346647
seed6  NVIDIA GeForce RTX 3090 Ti   1.348643   1.349070   1.344523
seed7  NVIDIA GeForce RTX 3090 Ti   1.342572   1.347549   1.346184

C320 lr5e-4 - C320 lr1e-3 champion (negative favours C320 lr5e-4):
seed   trained on                      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090     -0.006208  [-0.008158, -0.004258]
seed1  NVIDIA GeForce RTX 3090     +0.007293  [+0.005730, +0.008856]
seed6  NVIDIA GeForce RTX 3090 Ti  +0.000427  [-0.001360, +0.002214]
seed7  NVIDIA GeForce RTX 3090 Ti  +0.004977  [+0.003315, +0.006639]
Across 4 seed sets: mean +0.001622 | sd 0.005948 | 95% seed interval [-0.007843, +0.011087] | C320 lr5e-4 better than C320 lr1e-3 in 1/4

C320 lr5e-4 - C384 lr5e-4 (Stage 56) (negative favours C320 lr5e-4):
seed   trained on                      delta  95% window interval
seed0  NVIDIA GeForce RTX 3090     -0.002300  [-0.004216, -0.000384]
seed1  NVIDIA GeForce RTX 3090     -0.002344  [-0.004409, -0.000279]
seed6  NVIDIA GeForce RTX 3090 Ti  +0.004547  [+0.002698, +0.006396]
seed7  NVIDIA GeForce RTX 3090 Ti  +0.001364  [-0.000416, +0.003145]
Across 4 seed sets: mean +0.000317 | sd 0.003313 | 95% seed interval [-0.004954, +0.005588] | C320 lr5e-4 better than C384 lr5e-4 in 2/4

Seed-to-seed sd of loss over 4 seed sets:
C320 lr5e-4 0.002001 | C320 lr1e-3 0.006894 | C384 lr5e-4 (Stage 56) 0.001772

Report: my-gpt/checkpoints/stage_57_report.json
```
