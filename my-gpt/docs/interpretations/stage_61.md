# Stage 61 interpretation: on this corpus, BPE tokens learn faster but overfit, and do not beat characters

## Question

Stage 60 built a BPE tokenizer that makes Complete Shakespeare 2.45x shorter at 1,024
tokens. Does the champion predict Shakespeare better when it reads tokens instead of
characters?

- Same architecture as the character champion: C320 `L=6` `p=0.15`, LLLLGG with W23
  local windows, 256 positions, the Stage 50 schedule, 50,000 updates of 8×256 targets,
  weight decay `0.01`. Only the input units and the vocabulary (98 → 1,024) differ.
  The bigger embedding and output layers bring the token model to 8.13M parameters,
  against 7.54M.
- The comparison is in **bits per character** (bpc): total scored loss in bits divided
  by the characters those targets cover. Loss per token cannot be compared with loss
  per character. Both models score essentially the whole validation split (534,272
  vs 533,990 characters).
- Seed sets 0–1 on the 3090, 6–7 on the 3090 Ti, each paired with the Stage 53
  character run on the same card. The full validation split was scored every 5,000
  updates, and the best of those checkpoints was kept alongside the final one.

## Finding: the token model peaks at 15,000 updates, then overfits

| Seed set | Card | Characters | Tokens, final | Tokens, best (step) |
| --- | --- | ---: | ---: | ---: |
| 0 | 3090 | **1.9516** | 2.0171 | 1.9606 (15k) |
| 1 | 3090 | **1.9289** | 2.0160 | 1.9554 (15k) |
| 6 | Ti | **1.9457** | 2.0167 | 1.9512 (15k) |
| 7 | Ti | **1.9369** | 2.0166 | 1.9534 (15k) |
| mean | | **1.9408** | 2.0166 | 1.9552 |

| Comparison | Mean delta (bpc) | sd | 95% seed interval | Tokens better |
| --- | ---: | ---: | --- | ---: |
| Tokens final − characters | +0.0758 | 0.0095 | [+0.0607, +0.0910] | 0/4 |
| Tokens best checkpoint − characters | +0.0144 | 0.0093 | [-0.0004, +0.0292] | 0/4 |

Mean token-model validation bpc during training: 1.992 (5k), 1.970 (10k),
**1.955 (15k)**, 1.968 (20k), 1.976, 1.985, 1.993, 2.001, 2.009, 2.017 (50k).

- **Final models:** tokens are clearly worse, by 0.076 bpc on every seed set.
- **Best checkpoints:** with early stopping, tokens come within 0.014 bpc of characters
  but still lose on all four seed sets. This comparison also flatters the tokens,
  because the best step was picked using the same validation text.
- **Speed:** the token model reached 1.955 bpc by update 15,000. The character model
  needs the full 50,000 updates to reach 1.941. Per update, tokens learn much faster.
- **Overfitting:** after 15,000 updates the token model's validation loss rises
  steadily while its training loss keeps falling. By 50,000 updates it has made about
  52 passes over its 1.98M training tokens, against 21 passes for the character model.

## Reading

Tokens make each update cover about 2.45 times as much text, so the model learns
faster, but it also runs out of new text 2.45 times sooner. On a 4.8M-character
corpus that is the deciding factor. This is the Stage 59 lesson again from another
angle: the project is limited by data, and BPE spends the same data faster.

The best step, 15,000, also falls in the middle of a schedule that was tuned for
characters. Updates 13,000–17,999 run at `1e-4`, and the final `3e-5` phase only helps
a model that is still improving. A token model with a schedule that decays by about
update 15,000 would likely close some of the 0.014 gap, but probably not turn it into
a clear win.

Production models use BPE because they train on far more text than they could ever
repeat. In that regime the "learns faster" side wins and "runs out sooner" never
applies. Shakespeare is the opposite regime.

## Caveats

- Four seed sets; the final-model result is unambiguous, while the best-checkpoint
  comparison is close and its interval touches zero.
- Best checkpoints were selected on the scored validation split, so their numbers are
  optimistic.
- One vocabulary size (1,024), the character-tuned schedule and dropout, and the same
  256-position context. That window now spans about 627 characters, so the token model
  also had more context, which did not rescue it.
- The token model has 0.6M more parameters, all in the embedding and output layers.

## Consequences

- The character model remains the champion: C320 `L=6` `p=0.15`, mean 1.345 nats per
  character (1.941 bpc on these four seed sets).
- BPE works as engineered (Stage 60) and learns faster per update, but on this much
  text it is not better.

## Recommendation

Not run here:

1. Optional: retrain the token model with a schedule that finishes decaying by about
   update 15,000, or with stronger dropout, to see whether it ties or beats
   characters. This answers "fair schedule", not "more data".
2. The more interesting next step is more text: with a corpus several times larger,
   the faster-learning token model should pull ahead. That is the regime real GPTs
   train in.

## Artifacts

- Verbatim run output (scoring): [`reports/stage_61.md`](../../reports/stage_61.md)
- Machine-readable measurements: `checkpoints/stage_61_report.json`
- Replicates: `checkpoints/stage_61_c320_l6_p015_bpe1024_seed{0,1,6,7}_{final,best}_checkpoint.pt`
- Raw logs: `checkpoints/logs/61_61_token_model*.log`
- Tests: [`test_token_model_stage_61.py`](../../test_token_model_stage_61.py)
