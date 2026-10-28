# Stage 54 interpretation: C384 does not beat C320; C320 stays champion

## Question

Stage 53 made C320 `L=6` `p=0.15` the champion (it beat C256 on 7 of 8 seed sets).
Stage 54 climbs the next rung: does C384 `L=6` `p=0.15` beat C320 under the same
paired-seed protocol?

- C384 was trained at seed sets 0, 1, 6 and 7 (50,000 updates each, nothing else
  changed from Stage 53). Each is paired with the Stage 53 C320 run at the same seeds.
- Pairs share a card: seed sets 0–1 on the 3090, 6–7 on the 3090 Ti. All eight
  endpoints were scored together on the 3090.
- There is no stored C384 checkpoint, so nothing was verified byte for byte; the
  architecture was built once on CPU before training to confirm `384 / 4 heads` works.

## Finding: C384 is no better, and probably worse

| Seed set | Card | C320 | C384 | C384 − C320 |
| --- | --- | ---: | ---: | ---: |
| 0 (lineage) | 3090 | 1.352732 | **1.345260** | **-0.007472** |
| 1 | 3090 | 1.337010 | 1.358969 | +0.021959 |
| 6 | Ti | 1.348643 | 1.361670 | +0.013027 |
| 7 | Ti | 1.342572 | 1.342669 | +0.000097 |
| mean / sd | | 1.345239 | 1.352142 | **+0.006903 / 0.013130** |

- The 95% seed interval of the paired difference is `[-0.0140, +0.0278]`. It spans
  zero, so the sign is not established.
- **C384 is better on 1 seed set of 4** (seed set 0, the lineage seed, which is
  biased towards whatever was chosen on it — see Stage 53). Seed set 7 is a tie
  (`+0.0001`).
- The spread is large: the paired sd is `0.0131`, more than twice Stage 53's width
  sd (`0.0060`). Seed set 1 is a `+0.022` loss for C384, the biggest single gap in
  the lineage.

## Consequences

- Nothing here justifies moving off C320. The point estimate is `+0.0069` against C384
  and 3 of 4 seed sets favour C320 or tie.
- The Stage 53 gain from C256 → C320 (`-0.005`) does not continue to C384 at four
  seed sets. If there is a width optimum it is at or below C384.
- C384 trains about `1,050–1,150` s per run (C320: about 890 s on the 3090).

## Caveats

- Four seed sets with sd `0.013` give a wide interval; a real C384 advantage or
  disadvantage of `±0.01` cannot be ruled out in either direction.
- The extension rule for a close result (interval spans zero and mean within
  `~0.005` of zero) is **not** met: the mean is `+0.0069`, so no extension was
  proposed or run.
- Dropout `p=0.15` was tuned on C256 and has not been re-checked at C384. A wider
  model may want more dropout, so this is a test of C384 at fixed `p`, not of C384
  with its own best regularisation.
- Two of the four pairs were trained on the Ti; pairing cancels the device shift in
  the difference, but absolute losses are not comparable across cards.

## Recommendation

1. Keep **C320 `L=6` `p=0.15`** as the champion; do not adopt C384.
2. If width remains a priority, add seed sets 2–5 (3090) for C384 to narrow the
   interval before closing the width ladder. This was not run.
3. Otherwise move on to re-checking dropout at C320 (`p=0.10`, `p=0.20`), as
   recommended in Stage 53.

## Artifacts

- Verbatim run output (scoring): [`reports/stage_54.md`](../../reports/stage_54.md)
- Machine-readable measurements: `checkpoints/stage_54_report.json`
- Replicates: `checkpoints/stage_54_c384_l6_p015_seed{0,1,6,7}_final_checkpoint.pt`;
  C320 baselines `checkpoints/stage_53_c320_l6_p015_seed{0,1,6,7}_final_checkpoint.pt`
- Raw logs: `checkpoints/logs/54_54_paired_width*.log`
- Tests: [`test_paired_width_stage_54.py`](../../test_paired_width_stage_54.py)
