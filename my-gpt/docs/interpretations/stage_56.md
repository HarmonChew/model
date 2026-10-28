# Stage 56 interpretation: a lower peak learning rate calms C384 but does not beat the champion

## Question

Stage 54 found C384 `L=6` `p=0.15` beat the C320 champion on only 1 of 4 seed sets, with
about twice C320's seed-to-seed spread. That suggested the learning rate, tuned at smaller
widths, is too high at C384. Stage 56 halves only the first schedule phase (`1e-3` →
`5e-4`, updates 0–9,999); updates 10,000 onward (`3e-4`, `1e-4`, `3e-5`) are identical.
`config.learning_rate` was set to `5e-4` to match.

- Seed sets 0–1 on the 3090, 6–7 on the 3090 Ti, matching the baseline runs. The script
  refuses to score a pair trained on different cards. All endpoints were scored together
  on the 3090.
- No stored checkpoint exists for this configuration, so nothing was verified byte for
  byte.
- The schedule assertions (equal to the old schedule at 10,000, 13,000, 18,000 and
  49,999; `5e-4` at update 0) are in the script and in the test.

## Finding: the lower learning rate is a wash on average, but much steadier

| Seed set | Card | C320 lr 1e-3 | C384 lr 1e-3 | C384 lr 5e-4 |
| --- | --- | ---: | ---: | ---: |
| 0 | 3090 | 1.352732 | 1.345260 | 1.348824 |
| 1 | 3090 | **1.337010** | 1.358969 | 1.346647 |
| 6 | Ti | 1.348643 | 1.361670 | **1.344523** |
| 7 | Ti | **1.342572** | 1.342669 | 1.346184 |
| mean | | **1.345239** | 1.352142 | 1.346545 |

| Comparison | Mean delta | sd | 95% seed interval | Win count |
| --- | ---: | ---: | --- | ---: |
| C384 lr 5e-4 − C384 lr 1e-3 | −0.005597 | 0.010733 | [-0.022676, +0.011482] | lower LR better 2/4 |
| C384 lr 5e-4 − C320 lr 1e-3 champion | +0.001305 | 0.006617 | [-0.009224, +0.011834] | C384 better 2/4 |

Seed-to-seed sd of the loss, four seed sets each:

| Arm | sd |
| --- | ---: |
| C384 lr 5e-4 | 0.001772 |
| C384 lr 1e-3 (Stage 54) | 0.009565 |
| C320 lr 1e-3 | 0.006894 |

- **Does the lower LR help C384?** Possibly a little on average (−0.0056), but both
  intervals include zero and it won on only 2 of 4 seed sets. It helped on seed sets 1
  and 6 (−0.012, −0.017), where Stage 54's C384 was bad, and hurt on 0 and 7 (+0.004,
  +0.004), where Stage 54's C384 was good.
- **Does C384 now beat the champion?** No. The mean is +0.0013 in the champion's favour,
  the interval is wide and straddles zero, and the wins split 2/4.
- **Spread.** The C384 sd fell from 0.0096 to 0.0018, a variance ratio of about 29. With
  3 degrees of freedom per arm, the two-sided 5% F threshold is about 15, so this is
  unlikely to be chance alone, but it is four numbers per arm.

## Reading

The pattern fits the original hypothesis in one respect: the lower peak LR removed the
bad C384 runs (1.359, 1.362) and also the good ones (1.345, 1.343), leaving all four
runs between 1.3445 and 1.3488. The cost is that the best C384 runs no longer appear,
so the mean is no better than the champion's.

## Caveats

- Four seed sets. The mean differences (0.001–0.006) are smaller than the paired
  seed noise (sd 0.007–0.011) and nothing here is resolved.
- The sd comparison uses four runs on each side and mixes two cards; the card shift
  (~0.002) is small against the old spread but not against the new one (0.0018).
- The C320 and C384 lr 1e-3 baselines were already looked at in Stages 53–54, and the
  champion's arm was mildly flattered by selection (Stage 55 caveat), which pushes
  against C384.
- Only one lower value (`5e-4`) and only the first phase were tried. Whether the
  later phases or an even lower peak would do better is unknown.
- Whether C320 itself would gain from `5e-4` is untested; this stage cannot say that
  the lower LR is specific to width 384.

## Consequences

- Keep **C320 `L=6` `p=0.15`, peak LR `1e-3`** as the champion.
- C384 with `5e-4` is a viable, low-variance alternative, but nothing in this evidence
  says it is better, and it costs more to train (~17–18 min per run).
- The learning rate matters more to C384's run-to-run variance than to its mean.

## Recommendation

Not run here:

1. Test the same `5e-4` first phase at **C320** on the same four seed sets. If it also
   lowers the loss or its spread, the schedule is miscalibrated at every width and the
   champion itself may improve; if not, the effect is specific to C384.
2. If C384 is pursued further, add seed sets (8 or more) before drawing conclusions,
   since the current intervals are ±0.011–0.017.
3. Otherwise move to **weight decay** (currently `0.01`), the other lever for the
   `0.26`–`0.29` generalization gap.

## Artifacts

- Verbatim run output (scoring): [`reports/stage_56.md`](../../reports/stage_56.md)
- Machine-readable measurements: `checkpoints/stage_56_report.json`
- Replicates: `checkpoints/stage_56_c384_l6_p015_lr5e4_seed{0,1,6,7}_final_checkpoint.pt`
- Raw logs: `checkpoints/logs/56_56_learning_rate_width*.log`
- Tests: [`test_learning_rate_width_stage_56.py`](../../test_learning_rate_width_stage_56.py)
