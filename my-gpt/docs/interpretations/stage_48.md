# Stage 48 interpretation: the C256 champion was itself under-regularized

## Question

Stages 46–47 established that `C=320` needs stronger regularization than the
`p=0.10` fixed in every width stage, and that its dropout optimum is a plateau
bracketed at `p=0.15`–`0.20`. But that whole investigation held one thing
constant without testing it: the `C=256` champion that `C=320` was losing to had
**never had its own dropout tuned**. Stage 44 and Stage 45 both ran `C=256` at
`p=0.10`, the value inherited from the early architecture stages, so the frontier
was identified at a regularization setting that Stage 46 had just shown to be a
real, mis-settable lever at the wider model.

Stage 48 therefore moves the question back to the current champion:

**is `C=256` at `p=0.10` itself optimally regularized, or was its apparent
capacity ceiling partly caused by under-regularization?**

Both branches are `C=256` (`H=4`, `D=64`, `FF=1024`, `L=6`, `T=256`, `LLLLGG`),
trained fresh from architecture-native initialization under the unchanged
Stage 43–47 protocol for 50,000 updates / 102.4 million target presentations.
Only residual dropout differs: control `p=0.10`, treatment `p=0.15`. Dropout adds
no parameters and consumes no initialization RNG, so both branches start from
byte-identical weights
(`bdfe3e386128cc72a954aaf5e2cd529aad8d5c69a486f3bb075ec391b1f3943e`, the hash
Stage 45 recorded at C256), the same 4,850,274 parameters, and the same sampler
and Torch RNG states.

## Finding 1: p=0.15 improves C256, and the control is the Stage 45 champion exactly

| Readout (C256) | `p=0.10` control | `p=0.15` treatment | Delta (`p=0.15 − p=0.10`) |
| --- | ---: | ---: | ---: |
| Fresh paired, step-50k endpoints | 1.3427660 | 1.3411993 | **-0.0015667** |
| Fixed-panel final, step 50,000 | 1.3538347 | 1.3507259 | -0.0031089 |
| Fixed-panel selected best | 1.3522927 @ 48,500 | 1.3491719 @ 48,500 | -0.0031207 |
| Step-50k generalization gap | 0.28481 | 0.26296 | **-0.02185** |
| Step-50k train loss | 1.0690209 | 1.0877661 | **+0.01875** |

The paired fresh delta has standard error `0.0006895` and a two-sided Student-t
95% interval of **`[-0.0029214, -0.0002120]`**, which excludes zero: the
treatment won the metric at matched updates on a panel (seed 1,364) unused by any
prior stage. Every interior readout agrees in sign, including both fixed-panel
figures, which are reported without intervals and are therefore weaker evidence
on their own.

The operating point moved exactly as the earlier dropout steps did. `p=0.15`
pays `+0.0188` training loss to buy `-0.0016` validation loss and cuts the
generalization gap by `0.0219`, from `0.2848` to `0.2630`. That is the textbook
regularizer trade, and it is the same trade Stage 46 measured at `C=320`
(`+0.0198` train for `-0.0051` validation): the *price* of the first dropout step
is nearly width-independent, while its *yield* is not.

One difference is worth stating plainly. At `C=320` the first dropout step moved
the selected best checkpoint from an early step 39,000 to a late interior step
49,500 — the signature of a model whose validation curve had already turned.
Here both branches select their best checkpoint at the **same step, 48,500**.
`C=256` was never drifting; it was simply fitting less well than it could.

## Finding 2: the 256/320 frontier is unchanged in sign, and the champion moves further ahead

All four stored and fresh endpoints were rescored on one shared fresh 500-batch
panel (seed 1,365), so every comparison below is paired per batch. Dropout is
inactive in eval mode; only stored weights differ.

| Endpoint | Fresh mean validation loss | SE |
| --- | ---: | ---: |
| C256 `p=0.15` (this stage, treatment) | **1.3462870** | 0.0034788 |
| C256 `p=0.10` (this stage, control) | 1.3484581 | 0.0035604 |
| C320 `p=0.15` (Stage 46) | 1.3550250 | 0.0036563 |
| C320 `p=0.20` (Stage 47) | 1.3554197 | 0.0036200 |

| Comparison | Mean delta | 95% panel interval | First model lower-loss batches |
| --- | ---: | --- | ---: |
| C256 `p=0.10` − C256 `p=0.15` | +0.0021711 | [+0.0008697, +0.0034725] | 237 / 500 |
| C256 `p=0.15` − C320 `p=0.15` | -0.0087379 | [-0.0104575, -0.0070184] | 320 / 500 |
| C256 `p=0.15` − C320 `p=0.20` | -0.0091327 | [-0.0105424, -0.0077230] | 350 / 500 |
| C320 `p=0.15` − C320 `p=0.20` | -0.0003948 | [-0.0017194, +0.0009298] | 274 / 500 |

`C=256` `p=0.15` is the **resolved champion**: its interval lies entirely below
every one of the three other endpoints on this panel. The last row independently
replicates Stage 47's null on a *third* disjoint 500-batch panel (Stage 47's own
panel gave `+0.0000117`, its shared panel `-0.0000025`, this one `-0.0003948`);
the two `C=320` plateau endpoints remain interchangeable.

The `C256 → C320` deficit is now `0.0087`–`0.0091`, measured against the
correctly-regularized `C=256`. Stage 46 and Stage 47 had reported `0.0053` and
`0.0071` against the *under-regularized* `C=256` `p=0.10`; the frontier margin was
therefore understated by roughly a third in both, because the reference point
itself was mis-set. Note also that the direction of Finding 1 is stable across
panels while its magnitude is not: the fresh panel (seed 1,364) gives `-0.0016`
and the champion panel (seed 1,365) gives `-0.0022` for the same stored weights.

## What this establishes about the frontier

The capacity/regularization story now reads as a two-parameter surface rather
than a single width ladder:

| Width | Dropout step | Fresh paired delta | Source |
| ---: | --- | ---: | --- |
| 256 | `p=0.10 → p=0.15` | **-0.0015667** | this stage, seed 1,364 |
| 320 | `p=0.10 → p=0.15` | -0.0050533 | Stage 46, seed 1,360 |
| 320 | `p=0.15 → p=0.20` | +0.0000117 | Stage 47, seed 1,362 |

The first dropout step pays off at **both** widths, so the predicted signature of
capacity-dependent regularization is confirmed in direction — but the magnitude
is *inverted* against the prediction that motivated the stage:

- `p=0.15` buys C256 **0.0016**, not the ~0.005 that a capacity-scaled effect
  would suggest.
- The same `0.05` of dropout buys C320 **0.0051**, i.e. **3.2x more**, and C320
  still absorbs a second step at `p=0.20` without penalty while C256's gain is
  already the smallest of the three measured steps.

So the larger model has the *larger* regularization headroom, the opposite of the
capacity-monotone story the stage was built to test. The honest reading is that
dropout sensitivity is a property of the specific width, not a monotone function
of capacity: C256 was mildly under-regularized (0.0016 of held-out loss was
being left on the table), while C320 was badly so (0.0051). Neither model was
"too large"; C320's Stage 45 reversal was a tuning artifact that a proper
bracket resolves without ever beating C256.

**The champion changes, but the frontier does not move.** The project's best
configuration is now `C=256` at `p=0.15`, better than the previous champion by
`0.0016`–`0.0022` depending on panel. Because `p=0.15` also *reduces* the
generalization gap and the selected best step is unchanged at 48,500, this is a
strict improvement rather than a different point on the same frontier.

## Caveats

- Intervals are **panel-sampling uncertainty conditional on these checkpoints**:
  one initialization, one batch stream, one split, one device, one backend per
  branch. They are not training-seed or corpus-IID confidence bands.
- **`p=0.15` is the smallest effect this protocol has resolved.** Its interval is
  `±0.00135` wide and the point estimate `-0.0016` sits only `2.27` standard
  errors from zero. It is a real effect under this panel, but it is close enough
  to the resolution limit that it should not be treated as a large margin, and
  the sign was not predeclared — the stage asked a two-sided question and the
  answer came back positive.
- Only one dropout step at C256 is sampled. Stage 47 showed that extrapolating a
  single step's slope to predict the next is unreliable (its own linear
  projection was refuted by measurement), so **nothing here implies that
  `p=0.20` would help C256 further**, and `p=0.20` at C256 is untested.
- The two `C=320` endpoints are stored Stage 46 and Stage 47 treatments, not
  branches retrained in this stage. They were rescored on the same panel as both
  fresh C256 endpoints, so those comparisons are paired; they are not same-stage
  controls. The same caveat that Stage 47 recorded applies: the level of a
  cross-panel deficit is itself panel-dependent to roughly `0.002`, larger than a
  within-panel standard error, so the `0.009` margin should be read as
  `0.007`–`0.011` rather than as a single panel's number.
- The `C=256` `p=0.10` control reproduced Stage 45's champion byte-for-byte on
  weights, AdamW state, batch-generator state, the CPU RNG state, and the
  training device's own CUDA RNG state, with identical fixed-panel losses and the
  same selected best step. The *container* of the Torch RNG capture differs (see
  the reproducibility note below); that field is not a training-state claim.

## Reproducibility note: the lineage device must be pinned

The first attempt at this run reproduced the Stage 45 control trajectory on a
different GPU. The host exposes two accelerators, and CUDA enumerated the
`RTX 3090 Ti` as `cuda:0` where Stages 43–47 had trained on the `RTX 3090`.
Kernel selection is device-dependent, so the same seed produced a genuinely
different curve. The stage now resolves an unqualified `--device cuda` to the
**recorded lineage device by name** and refuses the canonical protocol on any
other accelerator, and the rerun reproduced the Stage 45 control trajectory at
every one of the 101 recorded evaluation boundaries (steps 0 through 50,000
inclusive) with zero mismatches — including bit-identical terminal weights, AdamW
state, and fixed-panel losses.

A related trap is recorded in the checkpoints. `capture_rng_state` stores the RNG
state of *every visible* accelerator, so its container is a function of the
host's GPU set rather than of the training run: Stage 45 stored one CUDA entry
and this stage stores two, yet the CPU state and the training device's own entry
are byte-identical (`0b8d137f…`), as are the terminal batch-generator states and
the AdamW states. Cross-stage comparisons in this stage therefore project the
capture onto `{cpu, training device}` and record the projection in each
checkpoint as `rng_normalization`, rather than comparing containers.

## Controls and provenance

- **The control is the champion it is trying to beat.** Preflight rebuilds the
  C256 native initialization through both the Stage 46 exposed-dropout factory
  and Stage 45's unchanged factory and requires `bdfe3e38…`; after training, the
  control's terminal weights, AdamW state, batch-generator state, CPU RNG state,
  and training-device RNG state are compared field by field against the stored
  Stage 45 endpoint and required to match.
- **Dropout is the only variable.** Preflight and the paired check require equal
  parameter counts, byte-identical initial weights, and identical initial sampler
  and Torch RNG states. Model weights and Torch RNG states necessarily differ at
  the end (different masks); no mask identity is claimed.
- **All four champion endpoints share one batch stream.** The sampler depends
  only on batch shape and seed, so the same 50,000 updates present the same
  102.4 million targets at every width; the check is that all four endpoints end
  on the recorded shared generator state, not an assumption.
- **Every stored endpoint is re-authenticated.** The Stage 45 champion and both
  C320 peers are validated through their owning stages' own validators, plus
  their file hashes, terminal steps, and fixed-panel losses before scoring.
- The independent CPU audit (`experiments/stage_48_audit.py`) re-derived every
  paired statistic from stored per-batch losses with its own incomplete-beta
  Student-t inversion, replayed the 50,000-update sampler, re-authenticated all
  four checkpoint payloads and the eight historical artifacts, and verified the
  launch manifest. Status: `passed`.

## Recommendation

**Adopt `C=256` at `p=0.15` as the project's architecture.** It is the best
validating configuration measured, it is `1.554x` cheaper in parameters than
`C=320`, and it is now better on both axes.

Do not spend further budget on dropout at either width. The measured curve says
the first step is exhausted at C256 (this stage), and the second step is
exhausted at C320 (Stages 47 and the third-panel replication here). If the
frontier is to move, the remaining untested direction is still **data**, not
width and not regularization: the same generalization gap attacked from the
other side, on a corpus where `C=256`'s `0.263` gap is not the binding
constraint.

## Artifacts

- Verbatim run output: [`reports/stage_48.md`](../../reports/stage_48.md)
- Machine-readable measurements: `checkpoints/stage_48_report.json`
- Champion shootout panel: `checkpoints/stage_48_champion_shootout.json`
- Independent audit: `checkpoints/stage_48_audit.json`
- Launch manifest: `checkpoints/stage_48_run_status.json`
- Raw log: `checkpoints/logs/48_48_dropout_strength.log`
- Figure: [`figures/stage_48_dropout_strength.png`](../../figures/stage_48_dropout_strength.png)
- Audit and plot tools: [`experiments/stage_48_audit.py`](../../experiments/stage_48_audit.py),
  [`experiments/stage_48_plot.py`](../../experiments/stage_48_plot.py)
- Tests: [`test_dropout_strength_stage_48.py`](../../test_dropout_strength_stage_48.py)

## Superseded in part by Stage 49

The recommendation above says not to spend further budget on dropout at C256
because nothing in Stage 48 implied `p=0.20` would help. Stage 49 tested that
directly and it does not: `p=0.20` is worse than `p=0.15` by `+0.0064156`
(`[+0.0053488, +0.0074824]`), a penalty `4.1x` larger than this stage's own
`0.0016` gain from the step below it. The C256 bracket is now closed on both
sides, so this stage's refusal to extrapolate was correct and the lock on
`p=0.15` stands. The champion panel in Stage 49 also re-measures both C320
endpoints against the same reference model on a fresh panel and reproduces this
stage's `0.0084`–`0.0089` margin.
