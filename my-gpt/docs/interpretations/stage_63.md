# Stage 63 interpretation: the champion copies a little, but not much

## Question

Stage 62 found that layers 5-6 look about 55 characters back in a way no simple pattern
explained. Stage 63 tests the leading idea, induction: "the current character appeared
before; copy what came after it". Nothing is trained.

- 64 sequences, each 100 random lowercase letters repeated once (200 characters).
  Random letters cannot be predicted from Shakespeare.
- 64 control sequences of 200 random letters with no repeat. The loss can drift down
  along any sequence, so the copying effect is the drop on the repeated sequences minus
  the drop on the controls.
- 100 is more than the 23 characters layers 1-4 can see, so only layers 5-6 can reach
  the first copy.
- Each of the 24 heads is scored on the second copy: **induction** (attention on the
  character 99 back, the one right after the earlier copy of the current character) and
  **duplicate** (attention 100 back, the earlier copy of the current character). A head
  is labelled at 0.3 or more.

Figures: [loss by position](../../figures/stage_63_loss_by_position.png) and
[the strongest induction head](../../figures/stage_63_induction_head.png).

## Finding

| Sequences | First half | Second half | Drop |
| --- | ---: | ---: | ---: |
| Repeated | 8.09 | 7.45 | 0.64 |
| Control, no repeat | 8.02 | 7.83 | 0.18 |

A uniform guesser would score ln 26 = 3.26 on every character.

- **The model copies, but only a little.** The repeat lowers the loss by 0.46 more than
  the controls do (0.64 against 0.18). A model that really copied would fall far below
  the guessing level. This one stays far above it.
- **No head reaches the 0.3 threshold.** The strongest induction head is L6 H2 at 0.13.
  Every other head scores 0.02 or less on both induction and duplicate. Layers 1-4 score
  0.00, as expected, since they cannot see 99 characters back.
- **L6 H2 does show the pattern, faintly.** Its map has a thin diagonal stripe 99 back
  that appears in the second copy and is absent in the first, with a few sharp dots on it.

## Reading

The champion has, at most, one weak induction head, and it is not enough to copy well.
The loss on random letters is far above guessing because the model is overconfident.
On the first copy its top guess gets 56% of the probability on average, and the guesses
are confident wrong letters (`e`, `a`, `u`, `i`, as if the text were English). It is not
mainly expecting spaces or capitals: 79% of its probability is on lowercase letters, and
the loss is still 7.58 when only those are counted. The copying effect is real but small
next to that. So the long look-back of layers 5-6 is mostly not induction.

## Caveats

- One checkpoint (seed set 0), one random draw of 64 sequences and 64 controls.
- Random uniform letters are very unlike the training text, so heads may behave
  differently here than on real text. A head that copies only in real words would score
  low in this test.
- The 0.3 threshold is a convention. L6 H2 at 0.13 is clearly above the rest, but still
  low.
- Attention shows where a head looks, not what it passes on. Switching heads off
  (Stage 64) tests which head does the copying.

## Next

Stage 64: switch heads off one at a time and see which ones the model needs, on real
text and for this copying test.
