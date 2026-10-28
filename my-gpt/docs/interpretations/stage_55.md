# Stage 55 interpretation: p=0.15 holds at C320, but nothing is resolved

## Question

Stage 53 made C320 `L=6` `p=0.15` the champion. Its dropout was tuned on C256 at the
biased lineage seed set (Stages 48–49), and Stage 52 showed that tuning was seed luck.
Stage 55 trains C320 at `p=0.10` and `p=0.20` at four seed sets and pairs each run
with the Stage 53 C320 `p=0.15` run at the same seeds and card.

- Seed sets 0–1 on the 3090, 6–7 on the 3090 Ti (matching the Stage 53 runs).
- Seed set 0 reproduced the stored Stage 46 control (`p=0.10`, model
  `db300b8756e4…`) and Stage 47 treatment (`p=0.20`, model `d0e3a4f340f3…`) byte
  for byte. All endpoints were scored together on the 3090.

## Finding: p=0.15 is best on 3 of 4 seed sets against both neighbours

| Seed set | Card | p=0.10 | p=0.15 | p=0.20 |
| --- | --- | ---: | ---: | ---: |
| 0 | 3090 | 1.357571 | 1.352732 | 1.352873 |
| 1 | 3090 | 1.347386 | **1.337010** | 1.346908 |
| 6 | Ti | **1.344108** | 1.348643 | 1.344427 |
| 7 | Ti | 1.355770 | **1.342572** | 1.344179 |
| mean | | 1.351209 | **1.345239** | 1.347097 |

| Comparison | Mean delta | 95% seed interval | p=0.15 better |
| --- | ---: | --- | ---: |
| p=0.10 − p=0.15 | +0.005970 | [-0.006467, +0.018407] | 3/4 |
| p=0.20 − p=0.15 | +0.001858 | [-0.007535, +0.011250] | 3/4 |

- Both intervals include zero. The data lean toward keeping `p=0.15`, with `p=0.10`
  the weaker neighbour, but neither call is resolved.
- Seed set 6 favours both neighbours over `p=0.15`, so the pattern is not uniform.

## Caveat: the p=0.15 runs were already selected once

Stage 53 chose C320 using these same `p=0.15` runs. Seed sets 1 and 7 are two of its
best C320 runs, so the control arm here is mildly flattered, the same selection effect
Stage 53 found for seed set 0. It is weak with four seed sets but pushes in the
direction of the finding.

## Consequences

- Keep **C320 `L=6` `p=0.15`** as the champion. There is no evidence that another
  dropout in `0.10`–`0.20` is better.
- Dropout in this range is a small lever at C320: differences of about `0.002`–`0.006`
  against a paired seed noise of `0.006`–`0.008`. Resolving them would need eight or
  more seed sets per arm, which is poor value.
- Tuning the remaining knobs one at a time at this noise level has stopped paying.
  Stages 52–55 changed the champion once (width) in about 30 runs.

## Recommendation

Stop fine-tuning dropout and width. The next lever should be one expected to move loss
by more than `0.01`, judged on at least four seed sets:

1. **Learning rate at larger width.** C384's seed spread (Stage 54) was about twice
   C320's, which suggests the C256-tuned learning rate is too high above C320. A lower
   peak learning rate at C384 tests whether width can still pay.
2. **Weight decay** (currently `0.01`), which attacks the `0.26`–`0.29`
   generalization gap differently from dropout.

## Artifacts

- Verbatim run output (scoring): [`reports/stage_55.md`](../../reports/stage_55.md)
- Machine-readable measurements: `checkpoints/stage_55_report.json`
- Replicates: `checkpoints/stage_55_c320_l6_{p010,p020}_seed{0,1,6,7}_final_checkpoint.pt`
- Raw logs: `checkpoints/logs/55_55_paired_dropout*.log`
- Tests: [`test_paired_dropout_stage_55.py`](../../test_paired_dropout_stage_55.py)
