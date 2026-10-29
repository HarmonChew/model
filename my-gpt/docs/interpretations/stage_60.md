# Stage 60 interpretation: a from-scratch BPE tokenizer shortens Shakespeare 2.5x at 1,024 tokens

## Question

Every model so far reads one character at a time. Production GPTs read sub-word
tokens learned by byte-pair encoding (BPE). Stage 60 builds a BPE tokenizer from
scratch ([`bpe.py`](../../bpe.py)), trains it on the Complete Shakespeare training split
only, and measures what it does to the validation split at several vocabulary sizes.

- Method (GPT-2's recipe without the byte layer): split text into pre-tokens with a
  regex (a word with its leading space, numbers, punctuation runs, whitespace), start
  from the 98 corpus characters, and repeatedly merge the most frequent adjacent pair.
- Pair counts are updated incrementally for the words that contain the merged pair,
  so 3,998 merges train in 13 seconds in plain Python.
- Smaller vocabularies are prefixes of the merge list, so one training run gives
  every size.
- Unit tests ([`test_bpe.py`](../../test_bpe.py)) cover exact round trips, merge order,
  truncation, determinism and save/load.

## Finding

| Vocab | Val tokens | Characters per token | Tokens seen < 10 times in train |
| ---: | ---: | ---: | ---: |
| 98 (characters) | 534,377 | 1.000 | 10 |
| 256 | 303,203 | 1.762 | 10 |
| 512 | 252,954 | 2.113 | 11 |
| 1,024 | 218,013 | 2.451 | 15 |
| 2,048 | 194,400 | 2.749 | 64 |
| 4,096 | 176,653 | 3.025 | 316 |

- **Exact.** `decode(encode(val)) == val` at every size, including words never seen
  in training: anything outside the merge list falls back to smaller pieces, down to
  single characters.
- **What it learns first** is English structure: `' t'`, `'he'`, `' a'`, `'ou'`, then
  `' the'` by merge 12. At 1,024 tokens, whole common words are single tokens
  (`' sweet'`, `' lady'`, `' your'`, `' daughter'`), while rarer words split into
  pieces (`' b|orrow|s'`, `' comp|an|y'`, `'Th|u|ri|o'`).
- **Diminishing returns.** Each doubling of the vocabulary buys less compression:
  +0.35, +0.34, +0.30, +0.28 characters per token.
- **Rare tokens explode past 1,024.** At 2,048 and 4,096 the vocabulary starts spending
  slots on whitespace runs (the complete works indent sonnets and stage directions)
  and on tokens the training text barely contains: 316 tokens appear fewer than 10
  times at 4,096. With only 4.8M training characters, the model would see too few
  examples to learn those embeddings.

## Reading

Tokenization is a trade. A bigger vocabulary makes the sequence shorter, so the same
256-position context window covers more text (about 627 characters at 1,024 tokens,
versus 256 now). But each token is a bigger, rarer thing to predict, and a small corpus
cannot teach thousands of rare tokens. For this corpus, 512–1,024 is the sweet spot:
2.1–2.5x shorter sequences with almost no rare tokens.

## Caveats

- The pre-tokenizer uses ASCII letter classes (Python's `re` has no `\p{L}`), so
  accented letters and curly apostrophes (`’`) are treated as punctuation and never
  merge into words. That costs a little compression on this corpus.
- There is no byte layer, so the tokenizer only handles the 98 characters in Complete
  Shakespeare. That is fine for this project but not for general text.
- Compression is not quality. Whether a token model predicts Shakespeare better can
  only be measured by training one (Stage 61).

## Consequences and next step

- Tokenizer: `bpe.py`, trained merges saved locally as
  `checkpoints/stage_60_bpe_4096.json` (any smaller size is a prefix).
- **Stage 61:** train the champion architecture (C320 `L=6` `p=0.15`, 256-token
  context) on vocab-1,024 tokens. Compare it with the character champion fairly in
  **bits per character**: token loss × tokens ÷ characters, converted to bits. Loss per
  token is not comparable with loss per character, because each token carries 2.45
  characters.

## Artifacts

- Verbatim run output: [`reports/stage_60.md`](../../reports/stage_60.md)
- Machine-readable measurements: `checkpoints/stage_60_report.json`
- Tokenizer merges: `checkpoints/stage_60_bpe_4096.json`
- Raw log: `checkpoints/logs/60_60_bpe_tokenizer.log`
