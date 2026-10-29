# Stage 57 interpretation: the lower learning rate steadies C320 but does not lower its loss

## Question

Stage 56 found that halving the first schedule phase (`1e-3` → `5e-4`, updates
0–9,999) left C384's mean loss where it was but cut its seed-to-seed spread about
fivefold. Stage 57 applies the identical schedule to the champion, C320 `L=6`
`p=0.15`, to see whether the schedule is too hot at every width (in which case the
champion should improve) or only at C384.

- Seed sets 0–1 on the 3090, 6–7 on the 3090 Ti, matching the Stage 53 baselines.
  The script refuses to score a pair trained on different cards. All endpoints were
  scored together on the 3090.
- The schedule is Stage 56's `make_schedule`, unchanged; updates 10,000 onward are
  identical to the champion's.
- No stored checkpoint exists for this configuration, so nothing was verified byte
  for byte.

## Finding: no gain in the mean, a narrower spread again

| Seed set | Card | C320 lr 1e-3 | C320 lr 5e-4 | C384 lr 5e-4 |
| --- | --- | ---: | ---: | ---: |
| 0 | 3090 | 1.352732 | **1.346524** | 1.348824 |
| 1 | 3090 | **1.337010** | 1.344303 | 1.346647 |
| 6 | Ti | 1.348643 | 1.349070 | **1.344523** |
| 7 | Ti | **1.342572** | 1.347549 | 1.346184 |
| mean | | **1.345239** | 1.346862 | 1.346545 |

| Comparison | Mean delta | sd | 95% seed interval | Win count |
| --- | ---: | ---: | --- | ---: |
| C320 lr 5e-4 − C320 lr 1e-3 champion | +0.001622 | 0.005948 | [-0.007843, +0.011087] | lower LR better 1/4 |
| C320 lr 5e-4 − C384 lr 5e-4 | +0.000317 | 0.003313 | [-0.004954, +0.005588] | C320 better 2/4 |

Seed-to-seed sd of the loss, four seed sets each:

| Arm | sd |
| --- | ---: |
| C320 lr 5e-4 | 0.002001 |
| C320 lr 1e-3 | 0.006894 |
| C384 lr 5e-4 (Stage 56) | 0.001772 |

- **Does the lower LR help the champion?** No. The mean moves `+0.0016` against it,
  it wins on only 1 of 4 seed sets, and the interval straddles zero. It helped on
  seed set 0 (−0.006), where the champion's run was its worst, and hurt on seed set 1
  (+0.007), where the champion's run was its best.
- **Spread.** The sd fell from `0.0069` to `0.0020`, a variance ratio of about 12.
  That is below the two-sided 5% F threshold for 3 and 3 degrees of freedom (about
  15), so on its own it is not conclusive. But it repeats the Stage 56 pattern at a
  second width.
- **Width at the lower LR.** C320 and C384 are tied at `5e-4` (`+0.0003`, 2/4), both
  with sd near `0.002`.

## Reading

The lower first-phase learning rate does the same thing at both widths: it removes
the tails. The worst runs get better and the best runs get worse, and the mean
stays put or gets slightly worse. So the effect is not specific to C384, and it is
not a miscalibration that was holding the champion back. At `1e-3` the early phase
sends each seed to a more different place; at `5e-4` every seed lands near
`1.345–1.349`.

The best single run in the lineage so far (`1.3370`, C320 seed set 1 at `1e-3`)
comes from the noisier schedule. That does not make `1e-3` better in expectation,
but there is no evidence that `5e-4` is better either.

## Caveats

- Four seed sets. The mean differences (`0.0003`–`0.0016`) are far smaller than the
  paired seed noise (sd `0.003`–`0.006`).
- The champion's arm was mildly flattered by selection (Stage 55 caveat), which
  pushes slightly against the treatment.
- Only one alternative value (`5e-4`, first phase only) has been tried at either
  width.
- The sd comparison mixes two cards; the card shift (`~0.002`) is the same size as
  the new spread.

## Consequences

- Keep **C320 `L=6` `p=0.15`, peak LR `1e-3`** as the champion (mean `1.3452`).
- A lower first-phase LR is a way to get more repeatable runs, not a better model.
  If later stages need fewer seeds per call, `5e-4` would make comparisons cheaper,
  but switching would change the baseline and needs its own decision.
- The learning-rate lever is exhausted for now: two widths, same answer.

## Recommendation

Not run here:

1. Move to **weight decay** (currently `0.01`) at the champion, four seed sets
   (0, 1, 6, 7), with one higher value such as `0.1`. The `0.26`–`0.29`
   generalization gap is still the clearest target.
2. Consider adopting the `5e-4` schedule as the comparison schedule for future
   stages, because its sd (`~0.002`) would let four seed sets resolve effects of
   about `0.003` instead of about `0.01`. This is a protocol choice for the owner.

## Artifacts

- Verbatim run output (scoring): [`reports/stage_57.md`](../../reports/stage_57.md)
- Machine-readable measurements: `checkpoints/stage_57_report.json`
- Replicates: `checkpoints/stage_57_c320_l6_p015_lr5e4_seed{0,1,6,7}_final_checkpoint.pt`
- Raw logs: `checkpoints/logs/57_57_learning_rate_champion*.log`
- Tests: [`test_learning_rate_champion_stage_57.py`](../../test_learning_rate_champion_stage_57.py)
