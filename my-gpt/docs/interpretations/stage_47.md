# Stage 47 interpretation: the C320 dropout optimum is broad and bracketed at 0.15–0.20

## Question

Stage 46 moved `C=320` from residual dropout `p=0.10` to `p=0.15` and recovered
`0.0051` of its `0.0103` deficit against `C=256`: a real rescue, worth
`-0.0050533` on a fresh paired panel at the cost of `+0.0198` training loss.
That left the *direction* open. If `0.15` was still too weak, the next step
should keep paying; if `0.15` was already the optimum, the next step should
flatten or reverse. Stage 47 asks that question directly by advancing the same
architecture one more dropout step:

**is `C=320` still under-regularized at `p=0.15`, or has it reached its
regularization optimum?**

Both branches are `C=320`, `H=4`, `D=80`, `FF=1280`, `L=6`, `T=256`, `LLLLGG`,
trained fresh from architecture-native initialization under the unchanged
Stage 43–46 protocol for 50,000 updates / 102.4 million target presentations.
Only residual dropout differs: control `p=0.15`, treatment `p=0.20`. Dropout adds
no parameters and consumes no initialization RNG, so both branches start from
byte-identical weights (`f438a0a742b54760749f7396aabbb91a5a53dc2eae9c7951594057cba5876b98`,
the same hash Stage 45 recorded at C320), the same 7,537,378 parameters, and the
same sampler and Torch RNG states.

## Finding 1: the second dropout step buys nothing

| Readout (C320) | `p=0.15` control | `p=0.20` treatment | Delta (`p=0.20 − p=0.15`) |
| --- | ---: | ---: | ---: |
| Fresh paired, step-50k endpoints | 1.3611457 | 1.3611574 | **+0.0000117** |
| Fixed-panel final, step 50,000 | 1.3597860 | 1.3601680 | +0.0003820 |
| Fixed-panel selected best | 1.3585851 @ 49,500 | 1.3598166 @ 49,500 | +0.0012315 |
| Step-50k generalization gap | 0.29071 | 0.27446 | **-0.01625** |
| Step-50k train loss | 1.0690792 | 1.0857130 | **+0.01663** |

The paired fresh delta has standard error `0.0007087` and a two-sided Student-t
95% interval of **`[-0.0013807, +0.0014042]`**, straddling zero; the treatment had
the lower loss on exactly 250 of the 500 shared fresh batches. The point estimate
is `+0.0000117` — two orders of magnitude smaller than the interval — and the two
single-panel interior readouts (final `+0.0003820`, selected best `+0.0012315`)
both sit on the losing side but are reported without intervals and are therefore
weak evidence against the null. Stage 46's own comparison of the same `p=0.15`
weights, by contrast, produced a `-0.0050533` point estimate whose interval
excluded zero.

The operating point moved exactly as before, and the readout ignored it:
`p=0.20` pays `+0.0166` additional training loss and cuts the generalization gap
by another `0.0163`, yet validation loss does not move. Regularization is no
longer converting into held-out improvement; it is only reducing fit.

Together with Stage 46's `p=0.10 → p=0.15` step (`-0.0050641` at `+0.0198` train
cost), the C320 dropout curve under this protocol reads:

| Dropout step | Fresh paired validation delta | Train-loss cost | Selected best step |
| --- | ---: | ---: | ---: |
| `p=0.10 → p=0.15` | **-0.0050641** (`[-0.0063277, -0.0038005]`) | +0.0198 | 39,000 → 49,500 |
| `p=0.15 → p=0.20` | **+0.0000117** (`[-0.0013807, +0.0014042]`) | +0.0166 | 49,500 → 49,500 |

Both branches selected their best checkpoint at the same late interior step,
49,500, so the erosion of the early-checkpoint signature that Stage 45 diagnosed
is complete by `p=0.15` and does not change further at `p=0.20`.

Of the three possibilities the stage was designed to distinguish, the result
selects the second: **`p=0.20 ≈ p=0.15`**. A broad regularization optimum has
been bracketed at `C=320`. Surpassing `0.15` is refuted, and `0.20` losing is not
supported — the estimate is statistically indistinguishable from zero in either
direction. Stage 46's linear extrapolation, which placed parity with `C=256`
near `p≈0.20`, is refuted by measurement: the second step delivered a slope 400x
smaller than the first, not a continuation, exactly the sublinear response that
Stage 46 flagged as the likely outcome.

## Finding 2: the second step does not move the frontier either

All three stored endpoints — Stage 45's `C=256 p=0.10` control, this stage's
`C=320 p=0.15` control, and its `C=320 p=0.20` treatment — were rescored on one
shared fresh 500-batch panel (seed 1,363, unused by every prior stage), so each
comparison below is paired per batch. Dropout is inactive in eval mode; only
stored weights differ.

| Endpoint | Fresh mean validation loss | SE |
| --- | ---: | ---: |
| C256 `p=0.10` (Stage 45 control) | **1.3441435** | 0.0032963 |
| C320 `p=0.20` | 1.3512438 | 0.0033350 |
| C320 `p=0.15` | 1.3512463 | 0.0033948 |

| Comparison | Mean delta | 95% panel interval | Candidate lower-loss batches |
| --- | ---: | --- | ---: |
| C320 `p=0.20` − C256 | **+0.0071003** | [+0.0057427, +0.0084580] | 163 / 500 |
| C320 `p=0.15` − C256 | +0.0071028 | [+0.0056051, +0.0086006] | 172 / 500 |
| C320 `p=0.20` − C320 `p=0.15` | -0.0000025 | [-0.0013347, +0.0013297] | 244 / 500 |

The two C320 endpoints are interchangeable on this panel and both lose to `C256`
by about `0.0071`, with intervals entirely on the losing side. The width ladder
therefore terminates the same way whether the second dropout step is taken or
not, and the third row is an independent replication of Finding 1 on a second
disjoint 500-batch panel: the stage's own panel (seed 1,362) gave `+0.0000117`,
this shared panel (seed 1,363) gives `-0.0000025`. A `1.4e-5` discrepancy across
two disjoint panels confirms the effect is a null, not a small win or loss.

## What this establishes about the C320 frontier

Three facts now hold under this protocol:

- **Regularization is a real lever at `C=320`, and it was mis-set once.** The
  first `0.05` of dropout bought `0.0051`; Stage 45's `p=0.10` reversal was
  partly a tuning artifact, as Stage 46 argued.
- **The lever saturates between `0.15` and `0.20`.** The second `0.05` bought
  `0.0000 ± 0.0013` while still costing `0.0166` train loss. The optimum is a
  plateau, not a peak, and `p=0.15` sits inside it.
- **The frontier does not move.** Both plateau endpoints trail `C=256` by the
  same margin, so the residual `C=320` deficit is not a regularization artifact
  and not fixable by the knob that Stage 46 identified as binding.

Stage 46's conclusion — "`C=320` at `p=0.10` is under-regularized; `p=0.15`
substantially rescues it, but it has not demonstrated superiority over `C=256`"
— is now closed from both sides. The apparent model-capacity frontier does
depend on regularization, and that dependence has been fully resolved: it moves
the `C=320` operating point by `0.005` and stops there, which is not enough to
displace `C=256`.

## Recommendation

> **Superseded in part by Stage 48.** The recommendation below locks `C=256` at
> `p=0.10`. Stage 48 tested that inherited setting directly and found `C=256` was
> itself under-regularized: `p=0.15` improves it by `-0.0015667`
> (`[-0.0029214, -0.0002120]`) and makes it the resolved champion against both
> `C=320` plateau endpoints, by `0.0087` and `0.0091`. The instruction to stop
> tuning `C=320`, and the conclusion that its residual deficit is not a
> regularization artifact, both stand — but the champion configuration is now
> `C=256` at `p=0.15`
> ([`stage_48.md`](stage_48.md)).

**Lock `C=256` at `p=0.10`, and stop tuning `C=320`.** The width ladder is now:

| Control → treatment | Width step | Parameter ratio | Fresh paired delta | Outcome |
| --- | ---: | ---: | ---: | --- |
| C128 → C192 (Stage 43) | 1.50x | 2.210x | -0.0478292 | wider wins |
| C192 → C256 (Stage 44) | 1.33x | 1.762x | -0.0146869 | wider wins, 3.3x less |
| C256 → C320 (Stage 45) | 1.25x | 1.554x | +0.0094590 | wider loses |
| C256 vs C320 `p=0.15` (Stage 46) | 1.25x | 1.554x | +0.0052762 | wider still loses |
| C256 vs C320 `p=0.20` (Stage 47) | 1.25x | 1.554x | +0.0071003 | wider still loses |

Intermediate dropout values (`p=0.125`, `p=0.175`) are **not** recommended, and
the bracket is now the reason rather than Stage 46's extrapolation: the
`0.15 → 0.20` interval is `±0.0013` wide, so any untested interior point must
beat both neighbours by less than the panel can resolve while the deficit
against `C=256` is `0.005–0.007`. There is no dropout setting in the bracket
that closes that gap.

If the frontier is to move, the evidence still points at data rather than width
or regularization: the same generalization gap attacked from the other side.
`C=256` remains both the best-validating and the `1.554x` cheaper
configuration, so no efficiency normalization favours `C=320`.

## Controls and provenance

- **Cross-stage determinism re-confirmed at the endpoint, exactly.** This
  stage's fresh `C=320 p=0.15` control reproduced Stage 46's `C=320 p=0.15`
  treatment byte-for-byte: model `37a523f2ddfb70794038144d834ff89fb6f37cfaa4f941410a199b08859b3984`,
  AdamW state `9c759acc82ae815b4c2a558a840527980412ed573ed9b3f164e80113218eb381`,
  batch generator `986ffc5650abe72ba270a01cc94ab5a247a946e1f4f040d1eef1508f8a7c86b9`,
  and model RNG `054107eae18e3861fbf7c54577d40e580c29cd3b328f2d759375fd5f6ca20608`
  all match, with identical fixed-panel losses to all recorded digits
  (train `1.0690791565179825` / val `1.3597860205173493`) and the same selected
  best step 49,500. Only provenance metadata differs between the files.
- **The native initialization is dropout-invariant and lineage-checked.**
  Preflight rebuilds the `p=0.15` control through the Stage 46 factory and
  requires the Stage 46 recorded hash; both branches share it, and it is the
  hash Stage 45 first recorded at C320 (`f438a0a7…`).
- **Dropout is the only variable.** Preflight and the paired check require equal
  parameter counts, byte-identical initial weights, and identical initial
  sampler and Torch RNG states. Model weights and Torch RNG states necessarily
  differ at the end (different masks); no mask identity is claimed.
- The independent CPU audit (`experiments/stage_47_audit.py`) re-derived every
  paired statistic from stored per-batch losses with its own incomplete-beta
  Student-t inversion (critical-value error `-2.6e-11`), replayed the
  50,000-update sampler and the seed-1,362 fresh panel, and verified payload
  integrity, AdamW slot coverage and counters, tensor shapes, and all historical
  artifact hashes. Status: `passed`.

## Caveats

- Intervals are **panel-sampling uncertainty conditional on these checkpoints**:
  one initialization, one batch stream, one split, one device, one backend per
  branch. They are not training-seed or corpus-IID confidence bands.
- **The level of the deficit against `C=256` is panel-dependent.** The same
  stored `p=0.15` weights scored `1.3513503` on seed 1,361 (Stage 46's panel)
  and `1.3512463` on seed 1,363 — a `0.0001` move — while the identical `C=256`
  weights moved `0.0019` (`1.3460741` → `1.3441435`) between the same two
  panels. The estimated deficit therefore reads `0.0052762` on one panel and
  `0.0071028` on the other. Cross-panel uncertainty on the deficit level is
  therefore of order `0.002`, larger than the within-panel standard error; the
  sign and the resolution of the comparison are unaffected, but the deficit
  should be quoted as `0.005–0.007`, not as a single panel's number.
- The `C=256` peer and the `p=0.10` endpoint are stored Stage 45 endpoints, not
  branches retrained in this stage. Both were rescored on the same panel as the
  two `C=320` endpoints, so those comparisons are paired; they are not
  same-stage controls.
- Two dropout steps (`0.10 → 0.15 → 0.20`) are sampled, and both probabilities
  were predeclared before their runs; no other probability was tried, so no
  selection over dropout values inflates the reported effects.
- The audit's limits are the same as Stage 46's: stored batch losses were
  recomputed statistically, model forward losses were not rerun.

## Artifacts

- Verbatim run output: [`reports/stage_47.md`](../../reports/stage_47.md)
- Machine-readable measurements: `checkpoints/stage_47_report.json`
- Cross-stage C256 comparison: `checkpoints/stage_47_c256_comparison.json`
- Independent audit: `checkpoints/stage_47_audit.json`
- Launch manifest: `checkpoints/stage_47_run_status.json`
- Raw log: `checkpoints/logs/47_47_dropout_strength.log`
- Figure: [`figures/stage_47_dropout_strength.png`](../../figures/stage_47_dropout_strength.png)
- Audit script: [`experiments/stage_47_audit.py`](../../experiments/stage_47_audit.py)
