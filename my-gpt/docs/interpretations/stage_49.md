# Stage 49 interpretation: the C256 dropout bracket closes at p=0.15

## Question

Stage 48 measured one dropout step at C256 and found `p=0.10 → p=0.15` worth
`-0.0016`. It also refused to extrapolate from that single step, because Stage 47
had already shown that one step's slope does not predict the next, and it
recorded `p=0.20` at C256 as explicitly untested. So the C256 regularization
curve had one measured interior point and no upper bound: the search could not
distinguish "the optimum is at `p=0.15`" from "the optimum is somewhere at or
above `p=0.15`".

Stage 49 supplies the missing step and nothing else:

**is `C=256` at `p=0.15` at its regularization optimum, or is there more
headroom above it?**

Both branches are `C=256` (`H=4`, `D=64`, `FF=1024`, `L=6`, `T=256`, `LLLLGG`),
trained fresh from architecture-native initialization under the unchanged
Stage 43–48 protocol for 50,000 updates / 102.4 million target presentations.
Only residual dropout differs: control `p=0.15` — Stage 48's own winning
treatment — against treatment `p=0.20`. Dropout adds no parameters and consumes
no initialization RNG, so both branches start from byte-identical weights
(`bdfe3e38…`, the hash every stage since 45 has recorded at C256), the same
4,850,274 parameters, and the same sampler and Torch RNG states.

## Finding 1: p=0.20 is too much — the treatment loses and by more than the first step gained

| Readout (C256) | `p=0.15` control | `p=0.20` treatment | Delta (`p=0.20 − p=0.15`) |
| --- | ---: | ---: | ---: |
| Fresh paired, step-50k endpoints | 1.3428665 | 1.3492821 | **+0.0064156** |
| Fixed-panel final, step 50,000 | 1.3507259 | 1.3595269 | +0.0088010 |
| Fixed-panel selected best | 1.3491719 @ 48,500 | 1.3586973 @ 48,500 | +0.0095254 |
| Step-50k generalization gap | 0.26296 | 0.25103 | **-0.01193** |
| Step-50k train loss | 1.0877661 | 1.1084951 | **+0.02073** |

The paired fresh delta has standard error `0.0005430` and a two-sided Student-t
95% interval of **`[+0.0053488, +0.0074824]`** — entirely positive, i.e. the
treatment is worse at matched updates. It is lower-loss on only `143 / 500`
shared batches. Every interior readout agrees in sign.

The direction of the *mechanism* is unchanged and confirms the manipulation
worked: `p=0.20` pays `+0.0207` training loss and still cuts the generalization
gap by `0.0119`. This is a model that is fitting less well, not one that failed to
regularize. What changed is that the extra regularization no longer buys anything
on held-out data.

The relevant comparison is against the step below it. `p=0.10 → p=0.15`
**gained** `0.0016`; `p=0.15 → p=0.20` **loses** `0.0064` — a penalty `4.1x`
larger than the gain that the first step produced. Both branches selected their
best checkpoint at the **same step, 48,500**, exactly as Stage 48's pair did, so
the deficit is not a selection-timing artifact.

## Finding 2: two disjoint panels agree on magnitude, and the shootout replicates the pair

All four stored endpoints were rescored on one shared fresh 500-batch panel
(seed 1,367), paired per batch. Dropout is inactive in eval mode; only stored
weights differ.

| Endpoint | Fresh mean validation loss | SE |
| --- | ---: | ---: |
| C256 `p=0.15` (this stage, control) | **1.3425371** | 0.0033186 |
| C256 `p=0.20` (this stage, treatment) | 1.3486411 | 0.0033161 |
| C320 `p=0.20` (Stage 47) | 1.3509843 | 0.0034710 |
| C320 `p=0.15` (Stage 46) | 1.3514700 | 0.0035450 |

| Comparison | Mean delta | 95% panel interval |
| --- | ---: | --- |
| C256 `p=0.15` − C256 `p=0.20` | -0.0061040 | [-0.0071325, -0.0050754] |
| C256 `p=0.15` − C320 `p=0.15` | -0.0089329 | [-0.0107404, -0.0071253] |
| C256 `p=0.15` − C320 `p=0.20` | -0.0084472 | [-0.0099364, -0.0069580] |
| C256 `p=0.20` − C320 `p=0.15` | -0.0028289 | [-0.0045312, -0.0011266] |
| C256 `p=0.20` − C320 `p=0.20` | -0.0023432 | [-0.0038663, -0.0008201] |
| C320 `p=0.15` − C320 `p=0.20` | +0.0004857 | [-0.0007792, +0.0017506] |

`C=256` `p=0.15` is the **resolved champion**: its interval lies entirely below
all three peers. Two independent 500-batch panels measure the same C256 dropout
step at `+0.0064` (fresh, seed 1,366) and `+0.0061` (shootout, seed 1,367), so the
magnitude of the loss is stable rather than a single panel's accident.

The last row re-measures the C320 plateau on yet another disjoint panel:
`+0.0004857`, straddling zero, with Stage 47's own panel at `+0.0000117` and the
Stage 48 shootout at `-0.0003948`. The sign has now flipped across three panels
for the same pair of stored endpoints. The two C320 dropout settings remain
interchangeable, and this stage did not retrain them.

One result here is stronger than anything Stage 48 could state. The
over-regularized `C256` `p=0.20` endpoint **still beats both C320 endpoints** by
`0.0023`–`0.0028` with intervals excluding zero. Being wrong about C256's dropout
by a full step is cheaper than being at C320 at all, so the width conclusion does
not depend on hitting C256's regularization optimum.

## Finding 3: the C256 response is asymmetric, and differently shaped from C320's

| Width | Dropout step | Fresh paired delta | Source |
| ---: | --- | ---: | --- |
| 256 | `p=0.10 → p=0.15` | -0.0015667 | Stage 48, seed 1,364 |
| 256 | `p=0.15 → p=0.20` | **+0.0064156** | **this stage, seed 1,366** |
| 320 | `p=0.10 → p=0.15` | -0.0050533 | Stage 46, seed 1,360 |
| 320 | `p=0.15 → p=0.20` | +0.0000117 | Stage 47, seed 1,362 |

The two widths do not share a curve shape:

- **C256** rises shallowly to `p=0.15` and then falls sharply: `+0.0016`, then
  `-0.0064`. The optimum is a narrow peak with a steep high-dropout penalty.
- **C320** rises steeply to `p=0.15` and then flattens: `+0.0051`, then
  `-0.0000`. The optimum is a broad plateau bracketed at `0.15–0.20`.

So C256 is the *more* dropout-sensitive model on the high side and the *less*
sensitive one on the low side — the two widths differ in which direction their
response is sharp, not merely in scale. That reinforces Stage 48's conclusion
that dropout sensitivity is a property of the specific width rather than a
monotone function of capacity, and it is why Stage 47's linear projection from a
single step failed.

**Bracket closed.** With `p=0.10` measured too little (worse by `0.0016`),
`p=0.15` best, and `p=0.20` measured too much (worse by `0.0064`), the C256
dropout optimum is interior and bracketed. There is no remaining dropout
direction to search at either width.

## Caveats

- Intervals are **panel-sampling uncertainty conditional on these checkpoints**:
  one initialization, one batch stream, one split, one device, one backend per
  branch. They are not training-seed or corpus-IID confidence bands.
- This is a **matched-budget** statement. Both branches were evaluated at 50,000
  updates and `p=0.20`'s own fixed-panel validation curve was still drifting
  slightly downward at its selected step (1.3593 at 47,500, 1.3587 at 48,500,
  1.3595 at 50,000 — a noisy flat decline, not a turned curve). The `0.0064`
  deficit therefore says `p=0.20` is worse at the protocol's budget; it does not
  say the deficit would survive an unbounded run. Extending the budget for one
  dropout setting to test that is a different experiment, and the established
  50,000-update protocol is what every width and dropout claim in this lineage is
  conditioned on.
- The `p=0.15` control is not merely *statistically* the Stage 48 treatment: it
  **reproduced it byte for byte**. Terminal weights
  (`22bcc9ee…`), AdamW state (`4dc3ddc7…`), batch-generator state (`986ffc56…`),
  CPU RNG (`6c3a57a8…`), and the training device's RNG entry (`0b8d137f…`) all
  match, as do the fixed-panel losses, the architecture record, and the selected
  best step 48,500. Because both captures were taken on this same two-GPU host,
  the full RNG container hashes equal as well — a stronger reproduction claim
  than Stage 48 could make against a single-accelerator host, and it means the
  control is the champion it is being compared against rather than a rerun that
  merely resembles it.
- The two `C320` endpoints are stored Stage 46 and Stage 47 treatments, not
  branches retrained here. They were rescored on the same panel as both fresh
  endpoints, so those comparisons are paired; they are not same-stage controls.
  The panel-dependence of a cross-stage *level* recorded by Stages 47 and 48
  still applies to the `0.0084`–`0.0091` margins, but not to their sign: the
  `C256 → C320` deficit reproduced on this stage's panel with the identical
  reference model (Stage 48's treatment is this stage's byte-identical control)
  at the same magnitude Stage 48 measured.
- The ladder's top row is this stage's own measured step and the second row is
  Stage 48's, transcribed rather than recomputed. Both were measured on
  different, disjoint fresh panels under the same protocol; they are comparable
  in construction, not paired to each other.
- Only two probabilities are sampled at C256, the second of which is the worst
  point measured. The optimum's *location* is bracketed; its shape between
  `0.15` and `0.20` is not resolved, and nothing here tests whether some
  probability between them is slightly better than `0.15`. The effect sizes that
  would distinguish such a point (`<0.0006`) are below this protocol's
  resolution, so that search is not worth its budget.

## Controls and provenance

- **The control is the endpoint it is trying to beat.** Preflight rebuilds the
  C256 native initialization through the Stage 46 exposed-dropout factory and
  Stage 45's unchanged factory and requires `bdfe3e38…`; after training, the
  control is compared field by field against the stored Stage 48 `p=0.15`
  treatment and required to match on weights, AdamW state, batch-generator
  state, CPU RNG, training-device RNG projection, fixed-panel losses, and
  selected best step.
- **Dropout is the only variable.** Preflight and the paired check require equal
  parameter counts, byte-identical initial weights, and identical initial sampler
  and Torch RNG states. Model weights and Torch RNG states necessarily differ at
  the end (different masks); no mask identity is claimed.
- **All four champion endpoints share one batch stream.** The sampler depends
  only on batch shape and seed, so the same 50,000 updates present the same
  102.4 million targets at every width; the check is that all four endpoints end
  on the recorded shared generator state (`986ffc56…`), not an assumption.
- **Every stored endpoint is re-authenticated.** The Stage 48 treatment and both
  C320 peers are validated through their owning stages' own validators, plus
  their file hashes, terminal steps, and fixed-panel losses before scoring.
- **The lineage device is pinned by name.** The canonical protocol refuses any
  accelerator other than the recorded `NVIDIA GeForce RTX 3090`, because kernel
  selection is device-dependent and the host also exposes a `3090 Ti`; the
  control reproduction is only defined on the recorded device.
- The independent CPU audit (`experiments/stage_49_audit.py`) re-derived every
  paired statistic from stored per-batch losses with its own incomplete-beta
  Student-t inversion, replayed the 50,000-update sampler, re-authenticated all
  four checkpoint payloads and the eleven historical artifacts, verified the
  four-row ladder against its sources, and checked the launch manifest. Status:
  `passed`.

## Recommendation

**Lock `C=256` at `p=0.15`.** It is the best-validating configuration measured,
and its own neighbouring dropout settings are both measurably worse than it on
the fresh panels that were run for exactly that comparison: `p=0.10` by
`0.0016`–`0.0022` (Stage 48's seeds 1,364 and 1,365), `p=0.20` by `0.0061`–
`0.0064` (this stage's seeds 1,366 and 1,367), and the C320 plateau endpoints by
`0.0084`–`0.0091` on the same shared-panel construction.

**Do not spend further budget on dropout at any width.** The C256 bracket is
closed on both sides and the C320 plateau is a saturated null. The remaining
untested direction is still **data**: the same generalization gap attacked from
the other side, on a corpus where C256's `0.263` gap is not the binding
constraint. Regularization has now been tuned to its optimum at both widths and
the frontier has not moved for three stages.

## Artifacts

- Verbatim run output: [`reports/stage_49.md`](../../reports/stage_49.md)
- Machine-readable measurements: `checkpoints/stage_49_report.json`
- Champion shootout panel: `checkpoints/stage_49_champion_shootout.json`
- Independent audit: `checkpoints/stage_49_audit.json`
- Launch manifest: `checkpoints/stage_49_run_status.json`
- Raw log: `checkpoints/logs/49_49_dropout_strength.log`
- Figure: [`figures/stage_49_dropout_strength.png`](../../figures/stage_49_dropout_strength.png)
- Audit and plot tools: [`experiments/stage_49_audit.py`](../../experiments/stage_49_audit.py),
  [`experiments/stage_49_plot.py`](../../experiments/stage_49_plot.py)
- Tests: [`test_dropout_strength_stage_49.py`](../../test_dropout_strength_stage_49.py)
