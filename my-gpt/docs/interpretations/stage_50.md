# Stage 50 interpretation: the depth gain does not survive at the mature operating point

## Question

The original depth result came from a much smaller era of this project: at `C=128`,
`T=64`, `p=0.10`, 1.1 million characters, and 18,000 updates, going from `L=4` to
`L=6` bought `-0.011038` validation loss for +48.46% parameters (Stage 24). Every
capacity decision since was made at a dramatically larger operating point — 4.8
million characters, `C=256`, `T=256`, a 50,000-update schedule, and regularization
tuned to `p=0.15` — and in that era width has been *negative*: `C=320` costs
1.5540x the champion's parameters and loses to it by `0.0087`–`0.0091` on every
panel measured (Stages 45–49).

Stage 50 therefore asks how that mature model should spend **more** parameters, by
re-asking the original depth question under the mature conditions and widening the
answer into a budget comparison:

**given a larger parameter budget, is it better spent on depth or on width?**

| | width | depth | contexts | p | params |
| --- | ---: | ---: | --- | ---: | ---: |
| control | 256 | 6 | `LLLLGG` | 0.15 | 4,850,274 |
| treatment | 256 | 8 | `LLLLLLGG` | 0.15 | **6,428,258** |
| width peer (stored, Stage 46) | 320 | 6 | `LLLLGG` | 0.15 | 7,537,378 |

Both branches are fresh from architecture-native initialization under the
unchanged Stage 43–49 protocol: 50,000 updates, 102.4 million target
presentations, one batch stream, one fixed panel, one corpus split. The treatment
inserts two W23 local layers *before* the same two consecutive final Full layers,
so the global refiners move from layers 5–6 to layers 7–8 and the compositional
local receptive field before them grows from `89` to `133` positions. Each C256
block costs 788,992 parameters, so the step is exactly two blocks (+32.53%).

Both branches were trained on the **RTX 3090 Ti**, off the lineage device, with the
stored 3090 endpoints scored beside them. That decision is what makes this stage's
methodological half interesting, and it is treated in Finding 4.

## Finding 1: two more layers buy almost nothing, and the two panels disagree about the sign

| Readout (all on the training device) | `L=6` control | `L=8` treatment | Delta (`L=8 − L=6`) |
| --- | ---: | ---: | ---: |
| Fresh paired panel, seed 1,368 | 1.3392857 | 1.3376945 | **-0.0015912** |
| Fresh paired panel, seed 1,369 | 1.3489730 | 1.3487511 | **+0.0002220** |
| Fixed-panel final, step 50,000 | 1.3530255 | 1.3524957 | -0.0005298 |
| Fixed-panel selected best | 1.3522 @ 48,500 | 1.3515 @ 49,500 | -0.0006867 |
| Step-50k train loss | 1.0886314 | 1.0611878 | **-0.0274436** |
| Step-50k generalization gap | 0.26439 | 0.29131 | **+0.026914** |

The primary panel (seed 1,368) resolves a small win: `-0.0015912`, SE `0.0006339`,
two-sided Student-t 95% interval **`[-0.0028366, -0.0003458]`**, lower loss on
`288 / 500` shared batches. On the lineage device that would be a positive result.

It does not replicate. The budget panel (seed 1,369), scoring the *same two
checkpoints* on 500 different shared batches, puts the same comparison at
`+0.0002220`, interval `[-0.0010775, +0.0015214]`, straddling zero. The two
intervals are disjoint, and the spread between the two point estimates
(`0.0018`) is **2.9x the within-panel standard error** (`0.00064`).

So the honest statement is not "depth helps a little" or "depth is neutral". It is
that **this stage's design cannot resolve the depth step at all**: the effect is
at or below the panel-to-panel variation of the measurement. Stage 24's
`-0.011038` was seven times the primary panel's estimate and, unlike this effect,
did not depend on which held-out batches were drawn.

The interior readouts agree with the panels that it is at most tiny: the
fixed-panel deltas are `-0.00053` (final) and `-0.00069` (selected best), and both
branches selected a best checkpoint within 1,000 updates of each other (48,500 vs
49,500). Nothing here looks like a frontier move.

## Finding 2: the extra capacity is absorbed by the generalization gap

The training-loss column is where the two more layers show up, and it is the
clearest result in the stage. `L=8` reaches `1.0612` against `L=6`'s `1.0886` —
it fits the training split **0.0274 better** — while validating within `0.0016` of
it on any panel. The gap therefore widens from `0.26439` to `0.29131`
(`+0.026914`), which is 17x the largest held-out gain any readout shows.

For comparison, Stage 24's depth step lowered *both* sides: eval-train fell
`0.0030` and validation fell `0.0116`, so six layers generalized better rather
than merely fitting harder. Stage 50's two extra layers do the opposite. This is
the overfitting signature rather than a capacity or optimization failure: the model
is not failing to use the parameters, it is using them on the training split.

That is direct evidence for the recommendation Stages 48 and 49 both ended on —
that the remaining constraint is **data**, not capacity. Both directions of
capacity spend are now measured at this operating point: width is negative
(`C=320` loses by `0.0073`–`0.0081` to C256 L=6 on this stage's own panels, see
Finding 3), and depth is indistinguishable from zero. Adding parameters in either
direction is absorbed by a `0.26`–`0.29` generalization gap.

## Finding 3: the budget question is answered, and the answer survives the device change

All four endpoints were scored on one shared 500-batch panel (seed 1,369), so
every comparison below is paired per batch. Dropout is inactive in eval mode;
only stored weights differ.

| Endpoint | Params | Trained on | Fresh-panel loss | SE |
| --- | ---: | --- | ---: | ---: |
| C256 `L=6` p=0.15, Stage 49 stored | 4,850,274 | 3090 | **1.3481738** | 0.0031616 |
| C256 `L=8` p=0.15, fresh | 6,428,258 | 3090 Ti | 1.3487511 | 0.0033527 |
| C256 `L=6` p=0.15, fresh | 4,850,274 | 3090 Ti | 1.3489730 | 0.0032444 |
| C320 `L=6` p=0.15, Stage 46 stored | 7,537,378 | 3090 | 1.3563111 | 0.0034741 |

| Comparison | Mean delta | 95% panel interval |
| --- | ---: | --- |
| C256 `L=8` − C320 `L=6` | **-0.0075600** | [-0.0089820, -0.0061380] |
| C256 `L=6` (fresh) − C320 `L=6` | -0.0073381 | [-0.0089461, -0.0057300] |
| C320 `L=6` − C256 `L=6` (stored, 3090) | +0.0081373 | [+0.0063511, +0.0099235] |
| C256 `L=6` (fresh) − C256 `L=8` (fresh) | +0.0002220 | [-0.0010775, +0.0015214] |

**The width conclusion reproduces on a second accelerator.** `C=320` loses to a
C256 `L=6` model trained *today, on different hardware* by `0.00734`, and to the
stored 3090 champion by `0.00814` — the same magnitude the lineage measured on the
3090 (`0.0087`–`0.0091`), slightly attenuated. The sign and size of the
width deficit do not depend on which card trained the reference model.

**And depth beats width as a way to spend the budget.** `L=8` at 6,428,258
parameters (`0.8529x` C320's budget) validates `0.00756` ahead of `C320` at
7,537,378. Even after correcting for the measured device shift (Finding 4), the
margin stays near `0.008`. So of the two directions available for the next
parameter block, depth is the *better* one — while Finding 1 says the block itself
is close to worthless. Both statements are true and neither should be quoted
without the other: **depth is the better spend, and neither spend is worth making.**

## Finding 4: the device shift is real, small, and the same size as the effect under test

Off the lineage device the stage cannot claim that its control reproduces the
stored champion, because Stage 48 established that the same seed on the other
accelerator produces a different trajectory. Rather than assert the difference
away, the stage measured it. The two `L=6` endpoints are the same architecture,
seed, corpus, protocol, and batch stream, and because initialization happens on
the CPU they start from the **same native state** (`bdfe3e38…`, checked, not
assumed). The only difference is the card.

| Device-shift readout | Value |
| --- | ---: |
| Replay probe, step 0 fixed-panel loss (`Ti` vs `3090`) | identical to 7 decimals |
| Replay probe, first divergent step | **500** |
| Replay probe, step-1000 deltas (train / validation) | -0.0026 / -0.0017 |
| Live control curve, validation drift at matched steps (101 steps) | mean +0.00285, range -0.0067 to +0.0093 |
| Terminal fixed-panel validation (`Ti` − `3090` champion) | **+0.0022996** |
| Terminal fixed-panel train (`Ti` − `3090` champion) | +0.0008653 |
| Paired panel device shift (`Ti` − `3090`, 500 batches) | **+0.0007993** `[-0.0003195, +0.0019180]` |
| Selected best step, both `L=6` runs | 48,500 (identical) |

The probe is the sharpest instrument here and it is worth reading carefully: the
**step-0 forward pass is bit-identical** on both cards, so the cards do not
disagree about arithmetic. They diverge only after updates begin, by `~1e-3` within
1,000 updates. This is floating-point reduction order entering the *training* loop,
not a different forward pass — which is exactly what Stage 48's device pin was
protecting, and why pinning was necessary rather than fussy.

The paired shift (`+0.0008`) is unresolved against zero at 95%, but its sign is
consistent across three independent readouts (probe, live curve, terminal fixed
panel), and its fixed-panel twin (`+0.0023`) is the largest of them.

**This is the stage's most consequential number, and it is larger than it looks.**
The depth step's whole measured range is `+0.0002` to `-0.0016`. The accelerator
change is `+0.0008`. **The two quantities are the same size.** Had the shift been
assumed rather than measured — or had the depth comparison been made across
devices, as the rejected designs would have — the entire result would have been
inside the confound. Instead the confound is a row of the panel, and the robust
finding (Finding 3: width loses by `0.008`, an order of magnitude above the shift)
stands on its own.

## Caveats

- **Both panels are conditional on the same two checkpoints.** One initialization,
  one batch stream, one corpus split, one device pair. These are not training-seed
  or corpus-IID intervals.
- **Panel-to-panel variation exceeds the within-panel interval for this effect.**
  The two disjoint 500-batch panels put the same pair of checkpoints `0.0018`
  apart, 2.8x their own standard errors. Every interval in this stage should be
  read as a *lower* bound on the uncertainty of the depth step. Stages 47–49
  recorded the same phenomenon on cross-stage levels (`~0.002`); Stage 50 shows it
  applies to a single stage's own headline number too.
- **The treatment is not "depth" in the abstract.** It is two more W23 local
  layers before the same two global refiners, which also moves the global layers
  from 5–6 to 7–8 and grows the pre-global local receptive field `89 → 133`. A
  null result is a null for *this* way of adding depth. The topology constraint
  that the two Full layers must be final and consecutive is what makes a clean
  "depth only" control unavailable in this architecture.
- **Regularization was held at the control's optimum, not the treatment's.** `p=0.15`
  is where C256 `L=6` peaks (Stages 48–49), and the deeper model has 32.5% more
  parameters. The widened generalization gap (`+0.027`) means the natural
  next question — does `L=8` at a *higher* `p` recover held-out loss? — was not
  asked here. This is a real gap in the conclusion: "depth does not help at
  matched regularization" is what was measured.
- **The device-shift row is one pair of runs, not a device calibration.** It says
  what moving *this* architecture between *these two* cards cost, at this protocol
  and this panel. It is not a general 3090/3090 Ti offset.
- The `L=8` branch cost 40% more wall-clock per update than `L=6` (1,015 s vs
  727 s for the full run). Tokens presented, not time, defined the budget.

## Recommendation

**Do not spend further budget on capacity, in either direction.** At this operating
point the model is data-limited, not capacity-limited:

- width costs `+55.4%` parameters and loses `0.0073`–`0.0081` (this stage, and
  Stages 45–49 on the lineage device);
- depth costs `+32.5%` parameters and moves held-out loss by an amount this design
  cannot distinguish from zero (`-0.0016` on one panel, `+0.0002` on another),
  while fitting the training split `0.0274` harder and widening the gap to `0.291`.

`C=256` `L=6` `T=256` `p=0.15` remains the project's configuration, now on the
strength of a third independent replication of the width result as well as its own
dropout bracket.

If capacity is to be revisited, the one live thread is the **regularization of the
deeper model** rather than more capacity: `L=8` at `p=0.20` is a single 50,000-update
run, and it is the only configuration that could convert the observed extra
fitting into held-out loss. Otherwise the next high-information experiment is the
one Stages 48 and 49 both pointed at and this stage now supports with a mechanism:
**data**. The `0.264`–`0.291` gap is where every capacity increment has gone.

## Controls and provenance

- **The depth comparison is internally matched.** Both branches trained on one
  accelerator, from one seed, on one batch stream, with byte-identical shared
  initialization proved structurally: 152 of 162 state entries identical, the 52
  added entries exactly the two new blocks, and the only licensed differences the
  language head plus the mask buffers of layers 4–5 (whose attention policy
  changed). Re-derived independently by the audit.
- **The width comparison is cross-device by design and flagged as such.** The
  stored `C320` and stored `C256 L=6` peers trained on the 3090; both fresh
  branches trained on the 3090 Ti. The panel records which endpoints are
  confounded, and the measured shift is what makes the comparison interpretable.
- **The batch stream is one stream across all four endpoints.** The sampler is a
  CPU generator depending only on batch shape and seed, so the check is that all
  four end on the recorded shared generator state (`986ffc56…`) — replayed from
  the seed and corpus by the audit, not assumed.
- **The device pin was replaced by a device measurement, not removed.** The
  canonical protocol still refuses any accelerator but the recorded one; the
  foreign mode has to be requested, records `foreign_device_authorized`, and
  changes what is claimed rather than failing later.
- **Every stored endpoint is re-authenticated.** Stage 49's control and Stage 46's
  C320 treatment are validated through their owning stages' own validators, plus
  their file hashes, terminal steps, and fixed-panel records.
- The independent CPU audit re-derived both paired panels from stored per-batch
  losses with its own incomplete-beta Student-t inversion, replayed the sampler,
  re-authenticated all four checkpoints and the fourteen historical artifacts,
  rebuilt both native initializations, re-derived the shared-prefix relation, and
  checked the device-shift row and the ladder against the panel. Status: `passed`.
  It found two defects in itself while doing so — a hard-coded comparison ordering
  and a silently skipped outcome check — which is the audit working as intended.

## Artifacts

- Verbatim run output: [`reports/stage_50.md`](../../reports/stage_50.md)
- Machine-readable measurements: `checkpoints/stage_50_report.json`
- Budget panel (standalone rescore): `checkpoints/stage_50_budget_shootout.json`
- Device-shift probe: `checkpoints/stage_50_control_probe.json`
- Independent audit: `checkpoints/stage_50_audit.json`, `checkpoints/stage_50_budget_audit.json`
- Launch manifest: `checkpoints/stage_50_run_status.json`
- Raw log: `checkpoints/logs/50_50_depth_revisit.log`
- Figure: [`figures/stage_50_depth_revisit.png`](../../figures/stage_50_depth_revisit.png)
- Audit and plot tools: [`experiments/stage_50_audit.py`](../../experiments/stage_50_audit.py),
  [`experiments/stage_50_plot.py`](../../experiments/stage_50_plot.py)
- Tests: [`test_depth_revisit_stage_50.py`](../../test_depth_revisit_stage_50.py)
