# Stage 58 interpretation: weight decay 0.1 is at most a small gain and does not close the gap

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

## Eight seed sets: the lean shrinks and stays unresolved

Seed sets 2–5 were added with `--seeds 2 3 4 5`, all on the 3090 because their Stage 53
baselines were trained there. All 16 endpoints were rescored together on the 3090.

| Seed set | Card | wd 0.01 val | wd 0.1 val | Delta | wd 0.01 gap | wd 0.1 gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 3090 | 1.345714 | **1.345430** | -0.000284 | 0.280845 | 0.281552 |
| 3 | 3090 | 1.345020 | **1.341880** | -0.003140 | 0.279055 | 0.277662 |
| 4 | 3090 | 1.345406 | **1.341658** | -0.003748 | 0.280280 | 0.278216 |
| 5 | 3090 | **1.349327** | 1.354928 | +0.005600 | 0.282026 | 0.284458 |
| mean of 8 | | 1.345803 | **1.343519** | -0.002284 | 0.279922 | 0.279013 |

| Comparison | Mean delta | sd | 95% seed interval | Win count |
| --- | ---: | ---: | --- | ---: |
| C320 wd0.1 − C320 wd0.01, 8 seed sets | -0.002284 | 0.005394 | [-0.006794, +0.002226] | wd0.1 better 6/8 |

Seed-to-seed sd of val loss over eight: wd 0.1 `0.005670`, wd 0.01 `0.004736`.

- The new seed sets averaged `-0.0004`, so the mean effect fell from `-0.0042` (four)
  to `-0.0023` (eight). The four-seed lean was partly seed set 6.
- Six of eight favour `0.1`, but the interval still includes zero (a two-sided sign
  test on 6/8 gives p ≈ 0.29). Compare Stage 53, which adopted C320 on 7/8 with an
  interval that excluded zero.
- The gap is unchanged: `0.2790` vs `0.2799` over eight.
- The lower four-seed sd for `0.1` did not hold up (`0.0057` vs `0.0047` over eight).

## Reading

Ten times the weight decay did almost nothing to the generalization gap, and the val
change is smaller than the seed noise. Eight seed sets confirm this: if `0.1` helps at
all, it is by about `0.002`, which this protocol cannot separate from zero. The `0.28` gap does not look like something
regularising the weights by this route will close. The `-0.0042` mean is suggestive
but below the roughly `0.01` that four seed sets with paired sd `~0.006` can resolve,
and it is not a 4/4 sweep.

## Caveats

- Eight seed sets, paired sd `0.005`: the interval half-width is about `0.0045`, so a
  true `0.002` effect would need roughly 40 seed sets to resolve.
- One alternative value (`0.1`) only; no curve, and nothing tells us whether `0.3`
  or `0.03` would differ.
- The champion arm was mildly flattered by selection (Stage 55 caveat), which pushes
  slightly against the treatment.
- The gap is measured on a train slice the size of val, taken from the start of the
  train split, so it is a consistent comparison, not an unbiased train loss estimate.

## Consequences

- Keep **C320 `L=6` `p=0.15`, weight decay `0.01`, peak LR `1e-3`** as the champion
  (eight-seed mean `1.3458`). Weight decay `0.1` (`1.3435`) is an acceptable
  alternative but not a proven improvement.
- Learning rate and weight decay are both flat around the champion at this noise
  level; the remaining levers are structural (data, architecture, training length).

## Recommendation

Not run here:

1. Stop tuning optimizer hyperparameters: learning rate and weight decay are both flat
   to within `~0.002` around the champion, and the gap did not respond to either.
2. Test the data limit directly: train the champion on half of the Complete
   Shakespeare training split (same updates, same val split) at four seed sets. A
   large loss increase means data is the binding constraint and more text is the
   next lever.

## Artifacts

- Verbatim run output (four seed sets): [`reports/stage_58.md`](../../reports/stage_58.md)
- Verbatim run output (eight seed sets): [`reports/stage_58_eight_seeds.md`](../../reports/stage_58_eight_seeds.md)
- Machine-readable measurements: `checkpoints/stage_58_report.json` (eight seed sets),
  `checkpoints/stage_58_report_four_seeds.json`
- Replicates: `checkpoints/stage_58_c320_l6_p015_wd01_seed{0..7}_final_checkpoint.pt`
- Raw logs: `checkpoints/logs/58_58_weight_decay*.log`
- Tests: [`test_weight_decay_stage_58.py`](../../test_weight_decay_stage_58.py)
