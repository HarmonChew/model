# Stage 62 interpretation: the first layer reads the last few characters, the last two look far back

## Question

Every stage so far trained a model and measured its loss. Stage 62 trains nothing. It
opens the Stage 53 champion (C320, 6 layers x 4 heads = 24 heads, seed set 0) and asks
where each attention head looks.

- 64 validation passages of 256 characters, run through the model's own forward pass
  with each layer's attention weights kept. The script checks that this gives exactly
  the model's normal predictions (loss 1.354 on these passages).
- Each head is scored on three patterns, averaged over query positions 32-255: its best
  **fixed offset** (attention on the character exactly k places back, k = 0-8), **word
  start** (the most recent space or newline) and **line start** (the most recent
  newline). A head gets a label when one pattern takes at least 30% of its attention.
- Layers 1-4 of the champion can only see the last 23 characters. That is the
  local/global design kept since Stage 29. Layers 5-6 see all 256.

Figures: [all 24 heads](../../figures/stage_62_attention_grid.png) and
[a labelled close-up](../../figures/stage_62_attention_closeup.png).

## Finding

| Layer | What its heads do |
| --- | --- |
| 1 | H2 looks at the previous character (99%), H1 at the one before that (99%), H4 leans 3 back (33%). H3 is spread. |
| 2-3 | Softer short offsets (33-66%) mixed with other positions. L3 H2 looks at the start of the line (28%). |
| 4 | Spread over its 23-character window, no simple pattern. |
| 5-6 | Look about 50-60 characters back on average. Many queries all look at the same few keys (vertical stripes in the grid). None of the three patterns explains them. |

- **Layer 1 builds a short "last three characters" summary.** Two heads are almost
  exact copy machines. One hands each position the previous character, the other the
  one before that. Nothing in training asked for this; it is simply what the first
  layer found most useful.
- **Line start is real, word start is not proven.** L3 H2 puts 28% on the last newline,
  against 2-4% for most heads, and the close-up shows it locking onto the newlines
  around `JULIA.`. The best word-start score (L3 H4, 22%) is barely above the 18-19%
  that L1 H1 and L1 H2 get by accident: when the previous character is a space, "one
  back" and "start of word" are the same position. The close-up does show L3 H4 on
  spaces, but this measure cannot separate the two.
- **Most heads (17 of 24) are "spread".** Three hand-picked patterns explain the early
  layers but not the later ones, and especially not the long-range layers 5-6.

## Reading

The picture matches the usual story for small transformers. Early layers do simple
bookkeeping (what came just before), middle layers pick up text structure (lines),
and the last layers fetch information from far back. The open question is what layers
5-6 fetch. The leading idea is induction: "find where the current text appeared
before, and copy what came next".

## Caveats

- One checkpoint (seed set 0). Other seeds may put the same jobs in different heads.
- Attention weights show where a head looks, not what it passes on or whether the
  model needs it. Ablation (switching heads off) would test that.
- The figures show only the last 64 characters of one passage. Layers 5-6 mostly look
  further back than that, so their panels show only part of what they do.

## Next

Stage 63: test for induction heads in layers 5-6. Feed random text repeated twice. If
the loss collapses on the second copy, the model is copying, and the attention should
point one character after the earlier copy of the current character.
