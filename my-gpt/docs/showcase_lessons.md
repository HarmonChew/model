# What I learned building a GPT from scratch

This project builds a small character-level GPT in PyTorch, one piece at a time, and
trains it on Shakespeare. It has 59 numbered stages, and each one changes a single thing
and measures the effect against a control. Validation loss went from 2.04, the first
full transformer, to 1.35 for the final model, which has 6 layers, width 320, a
256-character context and about 7.5M parameters. Compare the
[text samples](showcase_samples.md) and the [progress chart](../figures/showcase_progress.png).

These are the five lessons that changed how I think about language models.

## 1. A transformer is a short list of simple parts, and each one has a job

Stages 1–8 add one part at a time. Stage 1 is a lookup table that predicts each
character on its own. Stage 2 lets a character average over the characters before it.
Stage 3 replaces the fixed average with learned self-attention. Stage 4 runs four
attention heads side by side. Stage 5 adds a feed-forward network for per-position
processing. Stage 6 adds residual connections, and Stage 7 adds LayerNorm, both of which
let deeper stacks train at all. Stage 8 stacks the blocks into a real GPT. No part is
mysterious on its own. The capability comes from stacking them and training on enough
text.

## 2. Whether a change helps depends on the model around it

- **Dropout.** It hurt the width-64 model whether it was added late or from the start
  (Stages 14–16). At width 128 it was the most useful change available (Stages 21–22).
  A small model has nothing to spare for regularisation; a bigger one overfits without it.
- **Context.** Doubling context from 64 to 128 characters lost on the small model
  (Stage 17). Once the model was wider and deeper, the same change won and set a new
  best (Stage 26).

A "this doesn't work" result is only true for the model it was tested on. Good ideas get
rejected if they're tried too early.

## 3. Attention really does use the long context

Stages 27–34 tested whether the model actually used distant characters or just the last
few. Blocking attention between the two halves of the window hurt (Stage 27). Limiting
attention to a short local window lost to full attention (Stages 28–29), even when
stacked local layers could in theory reach just as far. Mixing local layers with one or
two full-attention layers didn't beat plain full attention either (Stages 30–34). The
model learns to use information from more than 100 characters back, which is the whole
reason attention replaced older architectures.

## 4. One training run can lie, so measure the noise first

Late in the project, Stage 51 trained the same recipe four times with different random
seeds. The final loss varied by about 0.011 between seeds. That is larger than many of
the improvements I had been accepting from single runs. Re-checking with paired,
multi-seed comparisons overturned two earlier calls:

- Stage 48's dropout gain turned out to be luck (Stage 52).
- Stage 46's finding that width 320 was worse than 256 was also luck. With eight seeds,
  width 320 won on 7 of 8 and became the new best model (Stage 53).

Since then, every decision uses at least four paired seeds, and a change is only adopted
when the evidence clearly exceeds the noise. This changed the project more than any
single architecture idea.

## 5. In the end, data beats tuning

Stages 56–58 tuned the learning rate and weight decay around the best model. Neither
moved the loss by more than about 0.002, which is within seed noise. The train/val gap
(about 0.28) showed the model was memorising, but neither regulariser closed it.

Stage 59 explains why. The same model trained on half the text was 0.098 worse, on every
seed, about 40 times any tuning effect. With less text it memorised more and generalised
worse. The model had learned most of what Complete Shakespeare can teach it, and since
that is all of Shakespeare, the next real gain needs more text, not better settings.

## How the work was done

- Every stage has a script, tests, a verbatim report of its output, and a written
  interpretation that separates evidence from my conclusions.
- Comparisons are paired: same seeds, same GPU, one change. Runs on the two GPUs, an
  RTX 3090 and a 3090 Ti, are never mixed within a pair, because the cards differ by
  about 0.002 in loss.
- Negative results are kept. 30 of the 59 stages tested something that did not
  become part of the final model.

## What I would do next

- **Use a real tokenizer (BPE)** instead of single characters, which is how production
  GPT models read text.
- **Train on a larger, more varied dataset**, for example other Elizabethan drama, keeping
  the Shakespeare validation text so results stay comparable.
- **Fine-tune** the base model for a new style or task.
