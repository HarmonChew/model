# Stage 46 interpretation: stronger regularization partially rescues C320

## Question

Stage 45 showed the width ladder reversing at `C=320`: it fit the training stream
better than `C=256` while validating worse, and its selected best checkpoint
arrived early (step 39,000) before a drifting validation curve. That is the
signature of a capacity/regularization mismatch rather than an optimization
failure, so Stage 46 asked the direct follow-up: **is `C=320` intrinsically
beyond what Complete Shakespeare supports, or is it merely under-regularized at
residual dropout `p=0.10`?**

Both branches are `C=320`, `H=4`, `D=80`, `FF=1280`, `L=6`, `T=256`, `LLLLGG`,
trained fresh from architecture-native initialization under the unchanged
Stage 43–45 protocol for 50,000 updates / 102.4 million target presentations.
Only residual dropout differs: control `p=0.10`, treatment `p=0.15`. Because
dropout adds no parameters and consumes no initialization RNG, the two branches
started from **byte-identical weights**, from the same sampler state, with the
same parameter count (7,537,378) and the same batch stream.

## Finding 1: `p=0.15` substantially improves `C=320`

| Readout (C320) | `p=0.10` control | `p=0.15` treatment | Delta (`p=0.15 − p=0.10`) |
| --- | ---: | ---: | ---: |
| Fresh paired, step-50k endpoints | 1.3564144 | 1.3513503 | **-0.0050533** |
| Fixed-panel final, step 50,000 | 1.3649537 | 1.3597860 | -0.0051677 |
| Fixed-panel selected best | 1.3624469 @ 39,000 | 1.3585851 @ 49,500 | -0.0038618 |
| Step-50k generalization gap | 0.31570 | 0.29071 | **-0.02499** |

The paired fresh delta has standard error `0.0006239` and a two-sided Student-t
95% interval of **`[-0.0062791, -0.0038274]`**, excluding zero. The treatment won
309 of the 500 shared fresh batches. This is the first intervention in the width
ladder that moved `C=320` in the right direction, and it moved exactly the
quantity the previous stage identified as binding: the generalization gap fell
by `0.025`, from `0.3157` to `0.2907`, while the treatment's selected best
checkpoint moved from an early step 39,000 to a late interior step 49,500.

Operating point moved, not just the endpoint: at step 50,000 the treatment pays
`+0.0198` train loss (`1.0691` versus `1.0493`) to buy `-0.0052` validation
loss. That is the textbook trade of a regularizer, not a schedule artifact.

## Finding 2: the rescue recovers only half the deficit — C256 still wins

Both `C=320` endpoints and the stored Stage 45 `C=256` control were rescored on
one shared fresh 500-batch panel (seed 1,361, unused by every prior stage), so
all three comparisons below are paired per batch. Dropout is inactive in eval
mode; only stored weights differ.

| Endpoint | Fresh mean validation loss | SE |
| --- | ---: | ---: |
| C256 `p=0.10` (Stage 45 control) | **1.3460741** | 0.0033440 |
| C320 `p=0.15` (rescued) | 1.3513503 | 0.0034532 |
| C320 `p=0.10` (under-regularized) | 1.3564144 | 0.0034954 |

| Comparison | Mean delta | 95% panel interval | Lower-loss batches |
| --- | ---: | --- | ---: |
| C320 `p=0.15` − C256 | **+0.0052762** | [+0.0037693, +0.0067830] | 195 / 500 |
| C320 `p=0.10` − C256 | +0.0103403 | [+0.0087633, +0.0119173] | 143 / 500 |
| C320 `p=0.15` − C320 `p=0.10` | -0.0050641 | [-0.0063277, -0.0038005] | 321 / 500 |

Stronger regularization recovered `0.0051` of the `0.0103` deficit against
C256 — **49%** — and the interval for the rescued model is still entirely on the
losing side. At `p=0.15`, `C=320` remains worse than `C=256` by about half the
margin it showed at `p=0.10`. The fixed-panel readouts agree: step-50,000
`1.3597860` (C320 `p=0.15`) versus `1.3538347` (C256), and selected best
`1.3585851` versus `1.3522927`, both C256-favourable by about `0.006`.

The third row is an independent replication of Finding 1 on a second panel: the
stage's own 500-batch run (seed 1,360) gave `-0.0050533`, this shared panel
(seed 1,361) gives `-0.0050641`. A `1.1e-5` discrepancy across two disjoint
500-batch validation panels indicates the effect is stable and its uncertainty
is dominated by batch sampling rather than by the specific panel.

## What this establishes about the corpus limit

Two facts now hold at `C=320` under this protocol:

- **Regularization is a real lever, and it was mis-set.** The regularizer that
  was load-bearing but fixed in every earlier width stage was too weak at this
  width. `C=320` was never given its best operating point in Stage 45.
- **The lever is not sufficient.** Halving the deficit does not close it.
  Extrapolating the measured `0.0051`-per-`0.05`-dropout slope linearly would put
  parity with C256 near `p≈0.20`, and regularizer benefits are typically
  sublinear in strength — while `p=0.15` is already paying `+0.0198` train loss
  and would begin to impair the fit long before that. There is no evidence of a
  dropout setting at which `C=320` clearly beats `C=256` on this corpus.

So the answer to the stage's question is a qualified both: `C=320` is *partly*
under-regularized — Stage 45's reversal overstated the case against it — but its
deficit is not a tuning artifact. At 5,343,777 characters of Complete
Shakespeare, a 534,377-character validation split, and 102.4 million target
presentations, the extra 2,687,104 parameters of `C=320` are not convertible into
validation improvement by the one knob tested. That is consistent with the
frontier sitting at `C=256` for a data reason, not merely a regularization one.

## Recommendation

**Lock `C=256` as the project's width.** The width ladder is now:

| Control → treatment | Width step | Parameter ratio | Fresh paired delta | Outcome |
| --- | ---: | ---: | ---: | --- |
| C128 → C192 (Stage 43) | 1.50x | 2.210x | -0.0478292 | wider wins |
| C192 → C256 (Stage 44) | 1.33x | 1.762x | -0.0146869 | wider wins, 3.3x less |
| C256 → C320 (Stage 45) | 1.25x | 1.554x | +0.0094590 | wider loses |
| C256 vs C320 `p=0.15` (Stage 46) | 1.25x | 1.554x | +0.0052762 | wider still loses |

The `C=320` penalty is reduced but not removed. Given that `p=0.15` costs
`+0.0198` train loss and still trails, `C=256` at `p=0.10` remains the
best-supported configuration, and it is also `1.554x` cheaper in parameters, so
there is no efficiency normalization under which `C=320` recovers.

Fine-grained dropout tuning at `C=320` (`p=0.125`, `p=0.175`) is **not**
recommended: the measured slope implies parity needs roughly double the
regularization increase that already costs `0.02` train loss, so the plausible
outcome is a narrower loss, not a win. If the frontier is to move, the
evidence points at data rather than width or regularization — the same
generalization gap attacked from the other side.

> **Superseded in part by Stage 47.** The `p=0.15 → p=0.20` step this section
> projected was run directly
> ([`stage_47.md`](stage_47.md)). The measured second-step delta is `+0.0000117`
> (`[-0.0013807, +0.0014042]`), not the projected approach to parity: the
> regularization response saturates, and both plateau endpoints still trail
> `C=256`. The extrapolation above is therefore refuted while its conclusion —
> lock `C=256`, stop tuning `C=320` — is confirmed by a tighter bracket.

## Controls and provenance

- **Cross-stage determinism re-confirmed, one level out.** The fresh `C=320`
  `p=0.10` control reproduced Stage 45's `C=320` treatment's fixed-panel
  trajectory to all recorded digits: best step 39,000 with train `1.0600597` /
  val `1.3624470`, and terminal train `1.0492555` / val `1.3649537`. Only
  provenance metadata differs between the two checkpoint files.
- **The exposed-dropout factory is the Stage 45 factory.** Preflight rebuilds
  the `p=0.10` model through the new dropout-parameterized factory and through
  Stage 45's unchanged factory from the same seed, and requires identical state
  hashes (`f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98`,
  the recorded Stage 45 native initialization). The architecture was not
  re-implemented, only re-parameterized.
- **Dropout is the only variable.** Preflight and the paired check require equal
  parameter counts, byte-identical initial weights, identical initial sampler and
  Torch RNG states, and identical terminal sampler state. Model weights and
  Torch RNG states necessarily differ (different masks); no mask identity is
  claimed.
- The independent CPU audit (`experiments/stage_46_audit.py`) re-derived the
  paired statistics from stored per-batch losses for the stage panel and for all
  three frontier comparisons, replayed the 50,000-update sampler, and verified
  payload integrity, AdamW state, and historical artifact hashes.

## Caveats

- Intervals are **panel-sampling uncertainty conditional on these checkpoints**:
  one initialization, one batch stream, one split, one device, one backend per
  branch. They are not training-seed or corpus-IID confidence bands.
- The `C=256` peer is Stage 45's stored endpoint, not a branch retrained in this
  stage. It is nonetheless a verified deterministic re-derivation of the Stage 44
  `C=256` treatment, and it was rescored on the same panel as both `C=320`
  endpoints, so the comparison is paired; it is not a same-stage control.
- One dropout step (`0.10 → 0.15`) is sampled. The slope quoted above is a
  two-point estimate, adequate for the parity argument but not a fitted curve.
- `p=0.15` was chosen before the run as the predeclared treatment; no other
  probability was tried, so no selection over dropout values inflates the
  reported effect.

## Artifacts

- Verbatim run output: [`reports/stage_46.md`](../../reports/stage_46.md)
- Machine-readable measurements: `checkpoints/stage_46_report.json`
- Cross-stage C256 comparison: `checkpoints/stage_46_c256_comparison.json`
- Launch manifest: `checkpoints/stage_46_run_status.json`
- Raw log: `checkpoints/logs/46_46_regularization_rescue.log`
- Figure: [`figures/stage_46_regularization_rescue.png`](../../figures/stage_46_regularization_rescue.png)
- Audit: `experiments/stage_46_audit.py`
