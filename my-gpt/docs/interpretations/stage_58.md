# Stage 58 interpretation: weight decay 0.1 leans slightly better but is not a clear win

## Question

The champion, C320 `L=6` `p=0.15`, has a `0.26`–`0.29` train/val generalization gap.
Stages 56–57 showed the learning-rate lever is exhausted. Does raising AdamW weight
decay from `0.01` to `0.1` help?

- Same four seed sets (0, 1, 6, 7) and cards as Stage 53: sets 0–1 on the 3090, 6–7
  on the 3090 Ti. The script refuses to score a pair trained on different cards. All
  endpoints were scored together on the 3090.
- The schedule and learning rate are the champion's (`1e-3` peak), unchanged; only
  `weight_decay` differs. The script asserts the default learning rate equals the
  schedule's value at update 0.
- Gap = val loss − train loss, with train loss scored on the first `len(val)` train
  tokens by the same full-split scorer. Val loss is the decision metric.

## Finding: three of four seed sets favour wd 0.1, the mean effect is small

| Seed set | Card | wd 0.01 val | wd 0.1 val | Delta | wd 0.01 gap | wd 0.1 gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3090 | 1.352732 | **1.347055** | -0.005677 | 0.281598 | 0.279920 |
| 1 | 3090 | **1.337010** | 1.339390 | +0.002380 | 0.274641 | 0.276137 |
| 6 | Ti | 1.348643 | **1.336357** | -0.012285 | 0.282483 | 0.274937 |
| 7 | Ti | 1.342572 | **1.341454** | -0.001118 | 0.278446 | 0.279223 |
| mean | | 1.345239 | **1.341064** | -0.004175 | 0.279292 | 0.277554 |

| Comparison | Mean delta | sd | 95% seed interval | Win count |
| --- | ---: | ---: | --- | ---: |
| C320 wd0.1 − C320 wd0.01 champion | -0.004175 | 0.006334 | [-0.014254, +0.005904] | wd0.1 better 3/4 |

Seed-to-seed sd of val loss: wd 0.1 `0.004509`, wd 0.01 `0.006894`.

- **Val loss.** Mean `-0.0042`, 3 of 4 seed sets favour `0.1`, and the seed interval
  straddles zero. The largest single gain (`-0.0123`, seed set 6) is about twice the
  paired sd.
- **Gap.** Mean gap `0.2776` vs `0.2793`, a reduction of `0.0017`. It shrank on seed
  sets 0 and 6 and grew on 1 and 7. Train loss rose slightly on average in the
  seeds where val improved, but the gap barely moved.
- **Spread.** sd `0.0045` vs `0.0069`; with four runs each this is not distinguishable.

## Reading

Ten times the weight decay did almost nothing to the generalization gap, and the val
change is smaller than the seed noise. The `0.28` gap does not look like something
regularising the weights by this route will close. The `-0.0042` mean is suggestive
but below the roughly `0.01` that four seed sets with paired sd `~0.006` can resolve,
and it is not a 4/4 sweep.

## Caveats

- Four seed sets, paired sd `0.006`: only a mean effect of about `≥0.01` or a 4/4
  sweep would be meaningful. Neither occurred.
- One alternative value (`0.1`) only; no curve, and nothing tells us whether `0.3`
  or `0.03` would differ.
- The champion arm was mildly flattered by selection (Stage 55 caveat), which pushes
  slightly against the treatment.
- The gap is measured on a train slice the size of val, taken from the start of the
  train split, so it is a consistent comparison, not an unbiased train loss estimate.

## Consequences

- Keep **C320 `L=6` `p=0.15`, weight decay `0.01`, peak LR `1e-3`** as the champion.
- Learning rate and weight decay are both flat around the champion at this noise
  level; the remaining levers are structural (data, architecture, training length).

## Recommendation

Not run here:

1. Optional: replicate wd `0.1` on four more seed sets (2, 3, 4, 5, each paired with a
   Stage 53-style baseline) if a `0.004` improvement is worth resolving; otherwise
   leave it.
2. Otherwise move to a different kind of lever than optimizer hyperparameters, since
   the gap did not respond.

## Artifacts

- Verbatim run output (scoring): [`reports/stage_58.md`](../../reports/stage_58.md)
- Machine-readable measurements: `checkpoints/stage_58_report.json`
- Replicates: `checkpoints/stage_58_c320_l6_p015_wd01_seed{0,1,6,7}_final_checkpoint.pt`
- Raw logs: `checkpoints/logs/58_58_weight_decay*.log`
- Tests: [`test_weight_decay_stage_58.py`](../../test_weight_decay_stage_58.py)
