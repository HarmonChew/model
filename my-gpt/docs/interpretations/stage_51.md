# Stage 51 interpretation: the seed-to-seed spread is larger than every call since Stage 46

## Question

Stages 46–50 made capacity and regularization calls on effects of `0.0015`–`0.009`,
each from one training seed scored on a sampled 500-batch panel. Stage 50 showed
the panel alone could flip a sign. Stage 51 measures the two noise sources directly
before any further ~0.002 call is made:

1. **Panel noise:** score every stored endpoint on the *entire* validation split
   (2,087 non-overlapping `T=256` windows, 534,272 targets, each scored once).
2. **Seed noise:** retrain the champion (C256 `L=6` `p=0.15`) from scratch on the
   lineage RTX 3090 at four seed sets, changing only the initialization, batch-stream,
   and dropout seeds (`+1000·k`).

Replicate 0 uses the lineage seeds and reproduced the stored Stage 49 control byte
for byte (model `22bcc9eecff8…`), so the short training loop is the Stage 43–50
protocol and the other three replicates differ from the champion only by seed.

## Finding 1: training seed moves held-out loss by 0.011

| Replicate | Full-split loss |
| --- | ---: |
| seed0 (lineage, = Stage 49 champion) | **1.344749** |
| seed1 | 1.347499 |
| seed2 | 1.350152 |
| seed3 | 1.355584 |
| mean / sd | 1.349496 / **0.004619** |

The range is `0.0108` for an unchanged configuration. The lineage seed is the best
of the four, so the champion's absolute loss is about `0.005` better than a typical
draw of the same configuration.

## Finding 2: the full split settles the panel questions

| Call | Full-split delta | 95% window interval |
| --- | ---: | --- |
| Stage 48 dropout 0.10 → 0.15 | -0.002046 | [-0.003756, -0.000336] |
| Stage 49 dropout 0.15 → 0.20 | +0.007212 | [+0.005821, +0.008603] |
| Stage 50 depth L=6 → L=8 (Ti) | -0.001565 | [-0.003263, +0.000132] |
| Stage 50 device 3090 → Ti | +0.002155 | [+0.000728, +0.003582] |
| Stage 46 width C256 → C320 | +0.007983 | [+0.005783, +0.010183] |

- **Depth:** the full split agrees with Stage 50's primary panel (`-0.0016`) and its
  interval just reaches zero. Stage 50's two panels disagreed through panel sampling,
  not because the checkpoints are unstable.
- **Device:** with panel noise removed, the 3090 → Ti shift is resolved at
  `+0.0022`. That is as large as the Stage 48 dropout gain.
- These intervals cover *which text is held out* for fixed checkpoints. They say
  nothing about training seed.

## Finding 3: every historical call is smaller than the unpaired seed threshold

Two independent single-seed runs differ with SD `√2 · 0.00462 = 0.0065`, so a
95%-style threshold is `0.0131`. All five calls are below it, including the two
largest (width `+0.0080`, `p=0.20` `+0.0072`).

**This does not show those calls are wrong.** Every lineage comparison was *paired*:
both arms shared the init, batch, and dropout seeds. Shared seeds cancel much of the
seed variance in the difference, so the right noise for those calls is the spread of
the *difference* across seed sets, which Stage 51 did not measure. What it does show:

- a single-seed comparison between *unpaired* runs is useless at this scale;
- the paired calls are only as reliable as the seed-to-seed spread of their
  differences, which is still unknown;
- `0.002`-scale calls (Stage 48 dropout, Stage 50 depth, the device shift) need
  that measurement before they are treated as settled.

## Caveats

- Four replicates give a rough SD; its 95% range is roughly `0.0026`–`0.017`.
- One configuration only. The seed spread at C320 or `L=8` may differ.
- Window losses are adjacent pieces of one text, so the window intervals slightly
  understate held-out uncertainty.

## Recommendation

Measure the seed spread of the **differences**, reusing these replicates as the
control arm. Retrain the other arm of a historical call at seeds 1–3 and pair it
with `seed1`–`seed3`:

- **Stage 48 dropout:** C256 `p=0.10` at seeds 1–3 (three ~12-minute runs). This is
  the call the champion rests on, and its effect is closest to the noise.
- **Stage 46 width:** C320 `p=0.15` at seeds 1–3, if the dropout result holds.

Four paired differences per call give a direct answer to "does this effect survive a
new seed?". Use that before the deferred `L=8 p=0.20` run or any data change, and
judge those against it.

## Artifacts

- Verbatim run output: [`reports/stage_51.md`](../../reports/stage_51.md)
- Machine-readable measurements (including per-window losses): `checkpoints/stage_51_report.json`
- Replicates: `checkpoints/stage_51_c256_l6_p015_seed{0..3}_final_checkpoint.pt`
- Raw log: `checkpoints/logs/51_51_noise_floor.log`
- Tests: [`test_noise_floor_stage_51.py`](../../test_noise_floor_stage_51.py)
