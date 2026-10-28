# Stage 52 interpretation: the Stage 48 dropout gain does not survive a new seed

## Question

Stage 48 picked dropout `p=0.15` over `p=0.10` at C256 `L=6` on a `-0.0020` gain from
one seed set. Stage 51 showed that the seed spread of single runs is `0.011`, but every
lineage call was paired (both arms share seeds), so the noise that matters is the
spread of the *difference*. Stage 52 retrains `p=0.10` at Stage 51's seed sets 1–3 and
pairs each with the matching Stage 51 `p=0.15` replicate.

Seed set 0 at `p=0.10` reproduced the stored Stage 48 control byte for byte (model
`3c4ded6a3653…`), so setting dropout after construction is exactly the Stage 48 arm.

## Finding: p=0.15 wins on 1 seed set of 4

| Seed set | p=0.10 | p=0.15 | p0.15 − p0.10 |
| --- | ---: | ---: | ---: |
| 0 (lineage) | 1.346796 | 1.344749 | **-0.002046** |
| 1 | 1.347013 | 1.347499 | +0.000486 |
| 2 | 1.345487 | 1.350152 | +0.004665 |
| 3 | 1.348721 | 1.355584 | +0.006863 |
| mean / sd | 1.347004 / 0.001328 | 1.349496 / 0.004619 | **+0.002492 / 0.004019** |

- The lineage seed set is the only one where `p=0.15` wins. The Stage 48 call was a
  seed draw, not a dropout effect.
- Across seed sets the mean leans toward `p=0.10` (`+0.0025`), but the 95% seed
  interval `[-0.0039, +0.0089]` includes zero. The honest call is **no detectable
  difference** between `p=0.10` and `p=0.15`.
- Each seed set's own window interval is tight (about ±0.0016) and three of four
  exclude zero, in different directions. Window intervals measure which text is held
  out, not training luck; they were never evidence for a configuration call.

## Finding: pairing helps less than hoped

The paired difference has sd `0.0040`. Unpaired runs differ with sd `√2 · 0.0046 =
0.0065`. Sharing seeds removes only part of the seed noise. With four seed sets a
paired call needs an effect of roughly `0.006` to clear the seed interval, and a
single seed set (as in every stage since 43) needs about `0.008`.

One more observation: `p=0.10` is far steadier across seeds (sd `0.0013`) than
`p=0.15` (sd `0.0046`). With four runs each this is suggestive only, but it points the
same way as the mean.

## Consequences

- Stage 48's choice of `p=0.15` is not supported. `p=0.10` is at least as good and
  more stable; the champion's `1.3447` is the best of eight draws, not a typical one.
- Calls of about `0.002` from one seed set (Stage 50 depth, the device shift) carry
  no information about the configuration.
- Larger single-seed calls are not settled either: Stage 49 `p=0.20` (`+0.0072`) and
  Stage 46 width C320 (`+0.0080`) sit right at the single-seed-set threshold.

## Caveats

- Four seed sets. The sd of the differences is itself rough (95% range roughly
  `0.0023`–`0.015`).
- One configuration and one comparison. Other calls may have a different
  difference spread.

## Recommendation

1. Treat C256 `L=6` at `p=0.10` and `p=0.15` as tied. Do not re-tune dropout further
   in this range from single seeds.
2. Make multi-seed paired comparison the protocol for any further call: at least four
   seed sets, judged on the seed interval of the paired difference.
3. Next, apply it to the larger call: Stage 46 width C320 `p=0.15` at seed sets 1–3,
   paired with Stage 51. If even the `0.008` width call does not hold, the lineage
   has reached the noise floor of this setup and the next gains need a bigger change
   (data, training length) rather than tuning.

## Artifacts

- Verbatim run output: [`reports/stage_52.md`](../../reports/stage_52.md)
- Machine-readable measurements: `checkpoints/stage_52_report.json`
- Replicates: `checkpoints/stage_52_c256_l6_p010_seed{0..3}_final_checkpoint.pt`
- Raw log: `checkpoints/logs/52_52_paired_dropout.log`
- Tests: [`test_paired_dropout_stage_52.py`](../../test_paired_dropout_stage_52.py)
