# Stage 53 interpretation: C320 beats C256; the Stage 46 width reversal was seed luck

## Question

Stage 46 found that C320 loses to C256 (both `L=6` `p=0.15`) by `+0.0080`, the largest
single-seed call in the lineage and the basis for "extra capacity only widens the
generalization gap". Stage 52 showed the Stage 48 dropout call was seed luck. Stage 53
applies the same test to width: C320 at eight seed sets, each paired with a C256 run
at the same seeds.

- Seed sets 0–3 were trained on the lineage 3090. C256 is Stage 51's replicates.
  Seed set 0 reproduced the stored Stage 46 treatment byte for byte (model
  `37a523f2ddfb…`).
- Seed sets 4–7 were added with `--seeds`: both arms of 4–5 on the 3090, both arms of
  6–7 on the 3090 Ti. Each pair shares one card, so the `~0.002` device shift cancels
  in the difference. All 16 endpoints were scored together on the 3090.

## Finding: C320 wins on 7 seed sets of 8

| Seed set | Card | C256 | C320 | C320 − C256 |
| --- | --- | ---: | ---: | ---: |
| 0 (lineage) | 3090 | 1.344749 | 1.352732 | **+0.007983** |
| 1 | 3090 | 1.347499 | **1.337010** | -0.010489 |
| 2 | 3090 | 1.350152 | 1.345714 | -0.004438 |
| 3 | 3090 | 1.355584 | 1.345020 | -0.010564 |
| 4 | 3090 | 1.353974 | 1.345406 | -0.008567 |
| 5 | 3090 | 1.352373 | 1.349327 | -0.003046 |
| 6 | Ti | 1.353963 | 1.348643 | -0.005320 |
| 7 | Ti | 1.349862 | 1.342572 | -0.007290 |
| mean / sd | | 1.351020 / 0.003662 | 1.345803 / 0.004736 | **-0.005216 / 0.005994** |

- The 95% seed interval of the paired difference is `[-0.0102, -0.0002]`. It excludes
  zero: **C320 is better than C256**, by about `0.005` on average.
- The lineage seed set is the only one that supports Stage 46. Without it, the other
  seven differences are tightly grouped: mean `-0.0071`, sd `0.0030`, every one
  negative.
- C320 seed set 1 (`1.337010`) is the best full-split loss in the lineage.

## Finding: the lineage seed set is biased by selection

Across Stages 52 and 53 the lineage seed set disagrees with the other seed sets on both
calls. That is the expected outcome of selection, not a property of the seed: every
lineage choice since Stage 43 was made because it won on seed set 0, so seed set 0
flatters whatever was chosen. The champion's `1.3447` is the best of the eight C256
runs; a typical C256 run scores `1.351`.

## Finding: width differences are noisier than dropout differences

The paired-difference sd is `0.0060` here against `0.0040` for dropout. Changing width
changes the shape of every weight tensor, so shared seeds produce unrelated initial
weights and pairing cancels little (unpaired sd would be about `0.0060`). For width or
depth calls, more seed sets are the only remedy; eight was enough here because the
effect is `0.005`.

## Consequences

- "Capacity is absorbed by the generalization gap" (Stages 46, 50) is wrong at this
  step: C320 converts its extra capacity into better held-out loss. The argument that
  the lineage was near the limit of the data rested on this call.
- The new champion configuration is **C320 `L=6` `p=0.15`** (mean `1.3458` over eight
  seed sets). Its dropout was tuned on C256 seed set 0 and has not been re-checked.
- Every lineage call from Stage 43 onward was made on seed set 0 and should be treated
  as unconfirmed, including Stage 45's width ladder and Stage 50's depth result.

## Caveats

- The interval only just excludes zero (`-0.0002`). Seven of eight in one direction
  is strong; the size of the effect is less certain than its sign.
- Two of the eight pairs were trained on the Ti. The pairing removes the device shift
  from the difference, but those four endpoints' absolute losses are not comparable
  with 3090 runs.
- Training seconds per C320 run were about 890 on the 3090 (C256: about 760).

## Recommendation

1. Use the mean over at least four seed sets, never seed set 0 alone, for every call.
   Width and depth calls need about eight.
2. Climb the capacity ladder again under this protocol: C384 `L=6` `p=0.15` against
   C320 at four seed sets first, extending to eight if the result is close.
3. Re-check dropout at C320 (`p=0.10` and `p=0.20`) once the width is settled.

## Artifacts

- Verbatim run output, seed sets 0–3 (training): [`reports/stage_53.md`](../../reports/stage_53.md)
- Verbatim run output, all eight seed sets (scoring): [`reports/stage_53_eight_seeds.md`](../../reports/stage_53_eight_seeds.md)
- Machine-readable measurements: `checkpoints/stage_53_report.json`
- Replicates: `checkpoints/stage_53_c320_l6_p015_seed{0..7}_final_checkpoint.pt`,
  `checkpoints/stage_51_c256_l6_p015_seed{4..7}_final_checkpoint.pt`
- Raw logs: `checkpoints/logs/53_53_paired_width*.log`
- Tests: [`test_paired_width_stage_53.py`](../../test_paired_width_stage_53.py)
