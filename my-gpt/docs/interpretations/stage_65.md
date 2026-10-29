# Stage 65 interpretation: the answer builds up layer by layer, and copying appears only at layer 6

## Question

Stage 64 showed which heads the champion needs. Stage 65 asks where its prediction
forms. After the embeddings (layer 0) and after each of the 6 blocks, the residual
stream goes through the model's own final layer norm and output layer (`ln_f` +
`lm_head`), as if the model stopped there. This is the "logit lens". Nothing is trained
or switched off.

- **Real text:** the loss and top-1 accuracy on the same 64 validation passages as
  Stage 62.
- **Copying:** the Stage 63 test, as in Stage 64. 0.46 for the full model; 0 means no
  copying.
- The reading after layer 6 is checked to equal the model's normal output exactly
  (it does).

Figure: [loss and copying after each layer](../../figures/stage_65_logit_lens.png).

## Finding

| Read after | Real-text loss | Improvement | Top-1 accuracy | Copying |
| --- | ---: | ---: | ---: | ---: |
| embeddings | 4.127 | - | 7.6% | 0.01 |
| layer 1 | 2.739 | +1.388 | 24.1% | -0.03 |
| layer 2 | 2.174 | +0.565 | 38.1% | 0.00 |
| layer 3 | 1.812 | +0.362 | 47.8% | 0.01 |
| layer 4 | 1.555 | +0.257 | 54.2% | 0.03 |
| layer 5 | 1.408 | +0.147 | 57.4% | 0.06 |
| layer 6 | 1.354 | +0.054 | 59.4% | 0.46 |

- **Layer 1 gives the biggest step.** Half of the total improvement (1.39 of 2.77)
  comes after layer 1, which fits Stage 64: the two layer-1 heads that read the last two
  characters carry the model. Part of this step is the lens itself, though: the reading
  after the embeddings (4.13) is poor partly because the output layer was never trained
  to read it (see Caveats).
- **Each later layer adds less than the one before.** Layers 2-4 add 0.57, 0.36 and
  0.26. By layer 4 the reading is already 1.55, and layers 5-6 together add only 0.20.
- **Readable copying appears at layer 6, as expected.** It is about 0 up to layer 4
  (-0.03 to 0.03), a little at layer 5 (0.06), and jumps to the full 0.46 at layer 6.
  The 0 up to layer 4 is guaranteed: layers 1-4 cannot see 100 characters back. It does
  show the noise level, about ±0.03. The jump fits Stage 64, where switching off L6 H2
  (or L1 H2, which feeds it) removed almost all copying.

## Reading

The champion refines its guess step by step. Layer 1 turns "which character is this"
into a usable guess from the last two characters, and layers 2-4 sharpen it with the
rest of the 23-character window. The long-range layers 5-6 polish the answer on real
text, and layer 6 is where copying from far back first shows in the reading. Copying
may start inside earlier layers in a form the lens cannot read; this test cannot tell.

One point does not line up neatly with Stage 64. There, switching off all of layer 5
cost only 0.024; here the reading improves by 0.147 across layer 5. These measure
different things. The lens asks how much closer the stream is to a good answer *when
read through the final layer norm and output layer*; switching off asks how much the
model needs that layer. Layer 6 may be able to do much of layer 5's work when layer 5
is gone, and layer 5 may partly be putting the stream into a shape the output layer
reads well. So "layer 5 improves the reading by 0.15" does not mean the model needs
layer 5 for 0.15.

## Caveats

- The early layers were never trained to be read through `ln_f` and `lm_head`; only
  the final layer was. A high early loss partly reflects that the stream is in a form
  the output layer does not read, not only that the model "does not know yet". A tuned
  lens (a small trained read-out per layer) would separate the two.
- On the random letters the loss rises with depth (about 5.5 after the embeddings, 8.1
  after layer 6, against 3.26 for guessing), probably because each layer pushes the guess
  towards Shakespeare. Copying is measured as a difference against the controls, so
  this rise cancels out, but the raw losses there are not "quality" readings.
- The small layer-5 copying (0.06) is only about twice the ±0.03 noise seen in layers
  1-4. It may be a first trace of copying or noise; one set of random sequences cannot
  tell.
- One checkpoint (seed set 0), one set of passages and random sequences.

## Next

Stage 66 could check whether L6 H2 still copies when L1 H2 is left alone but its
previous-character information is removed in a gentler way (mean ablation), settling
Stage 64's least certain caveat.
