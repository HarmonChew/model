# Stage 45 interpretation: width scaling from C256 to C320 saturates and reverses

## Question

Stage 45 asked whether the third native-width step of the width ladder — the
`5/4` increase from `C=256` to `C=320` (head size 64 → 80, feed-forward width
1024 → 1280) — still buys validation improvement at the locked 50,000-update /
102.4-million-target-presentation budget, or whether width-scaling returns are
saturating. Both branches were trained fresh from architecture-native
initialization under the unchanged Stage 44 protocol; only `n_embd` differed.
Stage 44 weights and optimizer state were never loaded.

## Finding

`C=320` lost. On the fresh 500-batch endpoint panel (seed 1,359, an unused
panel seed; same fresh-panel methodology as Stage 44):

| Readout | C256 control | C320 treatment | Delta (C320 − C256) |
| --- | ---: | ---: | ---: |
| Fresh paired, step-50k endpoints | 1.3462395 | 1.3556985 | **+0.0094590** |
| Fixed-panel final, step 50,000 | 1.3538347 | 1.3649537 | +0.0111190 |
| Fixed-panel selected best | 1.3522927 @ 48,500 | 1.3624470 @ 39,000 | +0.0101543 |

The paired fresh delta has standard error `0.0008036` and a two-sided Student-t
95% interval of **`[+0.0078801, +0.0110379]`**, which excludes zero on the
wrong side: `C=320` is worse, not merely unimproved. The treatment had lower
loss on only 155 of the 500 shared fresh batches.

**Scope of the interval.** The interval is panel-sampling uncertainty
conditional on these two trained checkpoints. It does not include training-seed
uncertainty: one initialization, one batch stream, one split, one device, and
one backend per branch. It should not be read as a run-to-run confidence band.

## The width curve, now three steps deep

All three steps use fresh architecture-native initializations, the identical
schedule, and fresh 500-batch paired panels:

| Control → treatment | Width step | Parameter ratio | Fresh paired delta | Per log-parameter |
| --- | ---: | ---: | ---: | ---: |
| C128 → C192 (Stage 43) | 1.50x | 2.210x | -0.0478292 | -0.0603 |
| C192 → C256 (Stage 44) | 1.33x | 1.762x | -0.0146869 | -0.0259 |
| C256 → C320 (Stage 45) | 1.25x | 1.554x | **+0.0094590** | **+0.0215** |

The return per log-parameter of capacity fell by ~2.3x at each step and has
now changed sign. On this corpus and budget, native width scaling has passed
its sweet spot at C256: the marginal parameter is not just less valuable, it is
harmful.

## Why it reverses: the generalization gap

Step-50k `validation − train` across the ladder: 0.2246 (C128), 0.2476 (C192),
0.2848 (C256), **0.3157 (C320)**. The gap grows monotonically with width. C320
fits the training stream better than C256 (1.0493 versus 1.0690 final train
loss) while validating worse — a capacity/regularization mismatch, not an
optimization failure. Consistent with early overfitting, C320's selected best
occurred at step 39,000 and its validation curve drifted upward from there,
whereas C256 kept improving to a late interior minimum at 48,500. At fixed
dropout 0.1, fixed 102.4M target presentations, and this 534,377-character
validation split, 320-wide is too large.

## Efficiency relative to added cost

The treatment consumed 1.554x the parameters (7,537,378 versus 4,850,274;
+2,687,104) and 1.596x the branch wall time (1,328.9 s versus 832.9 s on the
shared GPU used for this run) to produce a regression: negative absolute
improvement at positive extra cost. There is no efficiency normalization under
which this step pays. The prior step (Stage 44) already bought ~3.3x less
improvement than the first; this step buys none. C256 remains the width to
keep at this update/token budget.

## Controls and provenance

- Paired batch streams verified: identical initial and terminal sampler
  states; the terminal training-generator hash also replays independently in
  the audit.
- Same seed and RNG-reset policy as Stage 44 (seed 1,337 initialization, RNG
  reset to 1,340 after native initialization). No cross-width tensor or
  dropout-mask identity is claimed; only shared initial stream states are.
- Cross-stage determinism re-confirmed: the fresh C256 control reproduced the
  Stage 44 C256 treatment's model, AdamW, batch-generator, and model-RNG
  terminal states bit-for-bit (all four stream hashes equal).
- The independent CPU audit (`experiments/stage_45_audit.py`) re-derived the
  paired mean, SE, and Student-t interval from the stored per-batch losses and
  passed, as did the 10-test protocol suite (`test_width_scaling_stage_45.py`).

## Next experiment

The binding constraint at C320 is generalization, not capacity delivery. The
direct test is to re-open width with the regularizer that is now load-bearing:
a paired C256-versus-C320 (or C320-only) run at higher residual dropout, since
the ladder has shown the gap grows ~0.03 per width step at fixed dropout 0.1.
Alternatively hold width at C256 and test data scaling, which attacks the same
gap from the other side.

## Artifacts

- Verbatim run output: [`reports/stage_45.md`](../../reports/stage_45.md)
- Machine-readable measurements: `checkpoints/stage_45_report.json`
- Launch manifest: `checkpoints/stage_45_run_status.json`
- Raw log: `checkpoints/logs/45_45_width_scaling.log`
- Figure: [`figures/stage_45_width_scaling.png`](../../figures/stage_45_width_scaling.png)
- Audit: `experiments/stage_45_audit.py`
