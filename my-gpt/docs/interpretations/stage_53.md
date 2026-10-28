# Stage 53 interpretation: the Stage 46 width reversal does not survive a new seed either

## Question

Stage 46 found that C320 loses to C256 (both `L=6` `p=0.15`) by `+0.0080`, the largest
single-seed call in the lineage and the basis for "extra capacity only widens the
generalization gap". Stage 52 showed the Stage 48 dropout call was seed luck. Stage 53
applies the same test to width: C320 at Stage 51's seed sets, each paired with the
matching C256 replicate.

Seed set 0 reproduced the stored Stage 46 treatment byte for byte (model
`37a523f2ddfb…`).

## Finding: C320 wins on 3 seed sets of 4

| Seed set | C256 | C320 | C320 − C256 |
| --- | ---: | ---: | ---: |
| 0 (lineage) | 1.344749 | 1.352732 | **+0.007983** |
| 1 | 1.347499 | **1.337010** | -0.010489 |
| 2 | 1.350152 | 1.345714 | -0.004438 |
| 3 | 1.355584 | 1.345020 | -0.010564 |
| mean / sd | 1.349496 / 0.004619 | 1.345119 / 0.006431 | **-0.004377 / 0.008725** |

- The lineage seed set is again the only one that supports the historical call. On the
  other three, C320 is better by `0.004`–`0.011`.
- The 95% seed interval `[-0.0183, +0.0095]` includes zero, so this does not prove C320
  is better. It does remove the evidence that it is worse.
- C320 seed set 1 (`1.337010`) is the best full-split loss in the lineage, `0.0077`
  below the champion.

## Finding: the lineage seed set is biased

Across Stages 52 and 53 the lineage seed set disagrees with the other three on both
calls. That is the expected outcome of selection, not a property of the seed: every
lineage choice since Stage 43 was made because it won on seed set 0, so seed set 0
flatters whatever was chosen. Any configuration picked on one seed set will look
better on that seed set than it is.

## Finding: the difference noise is larger at width than at dropout

The paired-difference sd is `0.0087` here against `0.0040` for dropout. Changing width
changes the shape of every weight tensor, so shared seeds produce unrelated initial
weights and pairing cancels almost nothing (unpaired sd would be about `0.0079`). For
width or depth calls, pairing is worth little and more seed sets are the only remedy.

## Consequences

- "Capacity is absorbed by the generalization gap" (Stages 46, 50) is not supported.
  The data-limit argument rested on this call.
- The best mean configuration measured so far is C320 `p=0.15` (`1.3451`), then C256
  `p=0.10` (`1.3470`), then the champion C256 `p=0.15` (`1.3495`). None of the gaps is
  resolved with four seed sets.
- Every lineage call from Stage 43 onward was made on seed set 0 and should be treated
  as unconfirmed.

## Caveats

- Four seed sets. With a difference sd near `0.009`, four seed sets can only resolve
  effects of about `0.014`.
- Training seconds per C320 run were about 890 on the 3090 (C256: about 760).

## Recommendation

1. Retire seed set 0 as the decision seed. Judge configurations on the mean over
   several seed sets.
2. Decide the width question properly: add seed sets 4–7 for both C256 and C320
   (eight runs, about 3.7 hours on one card). Eight paired differences resolve about
   `0.007`.
3. If C320 holds, revisit the capacity ladder (C384, `L=8`) under the multi-seed
   protocol before concluding anything about the data limit.

## Artifacts

- Verbatim run output: [`reports/stage_53.md`](../../reports/stage_53.md)
- Machine-readable measurements: `checkpoints/stage_53_report.json`
- Replicates: `checkpoints/stage_53_c320_l6_p015_seed{0..3}_final_checkpoint.pt`
- Raw log: `checkpoints/logs/53_53_paired_width.log`
- Tests: [`test_paired_width_stage_53.py`](../../test_paired_width_stage_53.py)
