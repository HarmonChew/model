# Stage 64 interpretation: layer 1 carries the model, and copying runs through two heads

## Question

Stages 62-63 looked at where heads put their attention. That shows where a head looks,
not whether the model needs it. Stage 64 switches heads off (sets their output to zero)
and measures the damage. Nothing is trained.

- **Real text:** the loss on the same 64 validation passages as Stage 62. With nothing
  switched off it is 1.354.
- **Copying:** the Stage 63 test. The first-half minus second-half loss on the repeated
  random letters, minus the same drop on the unrepeated controls (which drift down 0.18
  with nothing to copy). With nothing switched off it is 0.46. 0 means no copying.
- Each of the 24 heads alone, then each layer's 4 heads together. The controls are
  re-measured with the same heads off, so each row compares like with like.

Figure: [loss increase and copying left, per head](../../figures/stage_64_head_ablation.png).

## Finding

| Switched off | Real-text loss increase | Copying left (0.46 with nothing off) |
| --- | ---: | ---: |
| L1 H2 (previous character) | +1.663 | 0.05 |
| L1 H1 (two back) | +0.769 | 0.43 |
| L6 H2 (Stage 63's best induction head) | +0.017 | 0.08 |
| Any other single head | +0.001 to +0.035 | 0.42-0.55 |
| All of layer 1 | +2.518 | 0.02 |
| All of layer 2 | +0.186 | 0.48 |
| All of layers 3 / 4 | +0.081 / +0.077 | 0.62 / 0.53 |
| All of layers 5 / 6 | +0.024 / +0.024 | 0.52 / 0.07 |

- **Two layer-1 heads carry the model.** Without L1 H2 (the previous character) the loss
  jumps from 1.35 to 3.02, and without L1 H1 (the character before that) to 2.12. No
  other head or group of later heads makes up for them.
- **The other 22 heads matter only a little each.** Any one of them costs at most 0.035.
  Whole layers cost more (layer 2: +0.19, layers 3-4: +0.08), so the work there is shared
  between heads, and the model can partly cover for any single one.
- **The long-range layers 5-6 barely help on real text.** Switching off a whole layer
  costs only +0.024. Looking far back is worth very little here.
- **Copying runs through two heads: L1 H2 and L6 H2.** Switching off either one removes
  almost all of it (0.46 to 0.05 and 0.08). No other head moves it by more than 0.09.
  This is the textbook two-head induction circuit: an early head notes "the character
  before me was X", and a late head finds the earlier place that came after X and copies
  what is there. L6 H2's attention in Stage 63 was weak (0.13), but it is the head that
  does the copying.

## Reading

The champion is mostly a short-range model. It leans hard on the last two characters
from layer 1, and builds on them in layers 2-4. Layers 5-6 can see 256 characters but
add little on real text. Their one clear job is copying earlier text, done by L6 H2 with
help from L1 H2, and that helps little on Shakespeare, where exact repeats 100
characters apart are rare. This fits Stage 63: the copying circuit exists but is weak.

## Caveats

- Switching a head off sets its output to zero. That is a harsh change the model never
  saw in training, so some of the damage may come from the shock rather than from lost
  information. Replacing the output with its average ("mean ablation") is the gentler
  test.
- L1 H2's part in copying is the least certain. Switching it off also breaks real text
  (loss 3.02), so copying may stop because the whole model is confused, not because L6
  H2 loses its input. The same goes for all of layer 1. Checking that L6 H2's attention
  stripe disappears with L1 H2 off would settle it.
- One checkpoint (seed set 0), one set of passages and random sequences.
- Single-head switch-offs miss shared work: the layer rows show that heads in the same
  layer cover for each other.

## Next

Stage 65: the logit lens. Read the model's prediction after each layer to see where the
answer forms, and whether layers 5-6 change it at all.
