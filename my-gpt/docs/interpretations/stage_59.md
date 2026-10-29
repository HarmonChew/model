# Stage 59 interpretation: half the training text costs 0.098; data is the binding constraint

## Question

Stages 56–58 found the learning rate and weight decay flat to within about `0.002`
around the champion, C320 `L=6` `p=0.15`, and its train/val gap stayed near `0.28`
whatever the optimizer did. Is the training text the limit? Stage 59 trains the
unchanged champion recipe on half of the Complete Shakespeare training split and
measures how much worse it gets.

- Half = every other 100,000-character block of the training split (2,409,389 of
  4,809,389 characters), so it samples the whole corpus rather than its first plays.
  The joins create 23 seams; a training window crosses one with probability under 1%.
- Vocabulary and validation split are unchanged; a test confirms the training sampler
  reads only the halved split.
- Recipe unchanged: same schedule (peak `1e-3`), 50,000 updates, weight decay `0.01`,
  same seeds. Seed sets 0–1 on the 3090, 6–7 on the 3090 Ti, paired with Stage 53's
  full-data runs on the same card. All endpoints were scored together on the 3090.
- Train loss for each arm is scored on the first `len(val)` characters of the text
  that arm was trained on.

## Finding: half data is worse on every seed set, by about 0.1

| Seed set | Card | Full val | Half val | Delta | Full gap | Half gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3090 | **1.352732** | 1.435378 | +0.082646 | 0.281598 | 0.442908 |
| 1 | 3090 | **1.337010** | 1.445253 | +0.108243 | 0.274641 | 0.446693 |
| 6 | Ti | **1.348643** | 1.440835 | +0.092193 | 0.282483 | 0.432077 |
| 7 | Ti | **1.342572** | 1.450850 | +0.108278 | 0.278446 | 0.457199 |
| mean | | **1.345239** | 1.443079 | +0.097840 | 0.279292 | 0.444719 |

| Comparison | Mean delta | sd | 95% seed interval | Win count |
| --- | ---: | ---: | --- | ---: |
| C320 half − C320 full | +0.097840 | 0.012648 | [+0.077714, +0.117966] | half better 0/4 |

- **Val loss.** `+0.098` worse, 4 of 4 seed sets, and the interval is far from zero.
  This is about 40 times the largest optimizer effect in Stages 56–58 (`~0.002`) and
  20 times the width gain that made C320 the champion (`0.005`, Stage 53).
- **Train loss** went *down* with half the data (`~0.998` vs `~1.066`): the model fits
  the text it sees better, because it sees each character twice as often (about 42
  passes instead of 21).
- **Gap.** `0.445` vs `0.279`. Halving the text widened the gap by `0.165`, far more
  than any regulariser moved it (weight decay: `0.001`).

## Reading

The model is limited by how much text it has, not by its optimizer settings. With less
text it memorises more (lower train loss) and generalises worse (higher val loss); with
more text the reverse. The gap responds strongly to data and barely at all to weight
decay or the learning rate, which is why Stages 56–58 found nothing.

A rough expectation, not a measurement: if loss falls by a similar amount for each
doubling, doubling the training text to about 10M characters could be worth
`0.05`–`0.1`, which is more than everything since Stage 43 combined. Returns usually
shrink with each doubling, so treat this as an upper-end guess.

## Caveats

- Four seed sets, but the effect is so large that more would not change the call.
- Fixed compute: both arms get 50,000 updates, so the half-data model also repeats its
  text twice as often. Part of the penalty is over-repetition; an early-stopped
  half-data model would probably lose less. This measures "same training budget, half
  the text", which is the question that matters for adding text.
- One fraction (half). Nothing here gives the shape of the curve between half and full,
  or beyond full.
- Stage 37 (TinyShakespeare → Complete Shakespeare) already pointed the same way, but
  changed vocabulary and corpus at the same time; this stage isolates the amount of text.

## Consequences

- The champion is unchanged: C320 `L=6` `p=0.15`, weight decay `0.01`, peak LR `1e-3`,
  full Complete Shakespeare (four-seed mean `1.3452`, eight-seed `1.3458`).
- Data is the binding constraint. Further tuning on this corpus is not worth the GPU
  time; the next real gain needs more training text.
- The Shakespeare corpus is exhausted as a source: Complete Shakespeare is already all
  of it.

## Recommendation

Not run here:

1. Close the tuning line at this stage.
2. If the project continues, change the data rather than the settings, for example
   adding other early-modern English drama (Marlowe, Jonson, the King James Bible) and
   keeping the same Shakespeare validation split, so the number stays comparable.

## Artifacts

- Verbatim run output (scoring): [`reports/stage_59.md`](../../reports/stage_59.md)
- Machine-readable measurements: `checkpoints/stage_59_report.json`
- Replicates: `checkpoints/stage_59_c320_l6_p015_half_seed{0,1,6,7}_final_checkpoint.pt`
- Raw logs: `checkpoints/logs/59_59_half_data*.log`
- Tests: [`test_half_data_stage_59.py`](../../test_half_data_stage_59.py)
