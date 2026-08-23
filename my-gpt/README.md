# Character language-model checkpoints

The token-local embedding model remains intact as **Stage 1**. **Stage 2**
adds fixed causal averaging so tokens can communicate, without adding any
trainable parameters.

## Stage 1: embeddings

This stage separates character lookup from next-character prediction:

```text
(B, T) token IDs
    -> token embeddings + position embeddings
(B, T, C) internal representations
    -> linear language-model head
(B, T, V) logits
```

The code is split by responsibility:

- `data_utils.py` owns the vocabulary, encoding/decoding, data split, and batches.
- `model.py` defines `EmbeddingLanguageModel` and cropped generation.
- `training.py` owns seeding, evaluation, and optimization.
- `train.py` runs the checkpoint and prints its reports.
- `inspect_data.py` only inspects the corpus; importing it no longer trains.
- `test_stage1.py` verifies shapes, parameters, gradients, cropping, and the
  architectural ceiling.

All data paths are resolved relative to these scripts, so commands work from
the repository root.

## Run the checkpoint

Train the token + position model with the Stage 1 defaults:

```powershell
python .\my-gpt\train.py
```

Train the requested A/B experiment with matched token/head initialization and
the same sampled-batch stream. Every reported evaluation also reuses the same
fixed set of sampled blocks, making initial/final losses directly comparable:

```powershell
python .\my-gpt\train.py --compare-positions
```

Run a quick CPU smoke check:

```powershell
python .\my-gpt\train.py --device cpu --max-iters 10 --eval-iters 2 --sample-length 40
```

Run the tests without third-party test tooling:

```powershell
python -m unittest discover -s .\my-gpt -p "test_*.py" -v
```

## What the ceiling check proves

`train.py` constructs two contexts whose final token is identical but whose
earlier tokens are all different. Their final-position logits remain exactly
the same. Each position is still processed independently, so a context window
of eight tokens is only an input-size limit—not eight-token understanding.

There is an additional subtlety in this particular model. Because the head is
linear,

```text
W(token_embedding + position_embedding) + bias
```

is just a token-dependent term plus a position-dependent term. Position does
not create token-to-token communication (or even a token-by-position
interaction). Attention is the next ingredient that will let information move
between positions.

## Stage 2: fixed causal averaging

`causal_average_model.py` defines `CausalAverageLanguageModel`. After adding
token and position embeddings, it applies this fixed lower-triangular
operation:

```text
position 0 <- position 0
position 1 <- average(positions 0, 1)
position 2 <- average(positions 0, 1, 2)
...
```

The `tril` connectivity matrix is a registered buffer, not a parameter. With
the default vocabulary, context, and embedding sizes, both Stage 1 and Stage 2
therefore have exactly 4,481 trainable parameters.

Run the new checkpoint:

```powershell
python .\my-gpt\03_causal_average.py
```

Run a matched comparison against the retained Stage 1 model. Both models use
identical learned-parameter initialization, training batches, and evaluation
batches:

```powershell
python .\my-gpt\03_causal_average.py --compare-embedding
```

Quick CPU smoke comparison:

```powershell
python .\my-gpt\03_causal_average.py --compare-embedding `
    --device cpu --max-iters 10 --eval-iters 2 --sample-length 40
```

The Stage 1 prefix test remains exactly zero. With the default multi-token
context, the Stage 2 prefix test is non-zero because changing an earlier token
now changes the averaged final representation. The weights are still uniform
and fixed; learned Q/K/V self-attention is the intended next checkpoint.

## Stage 3: one causal self-attention head

`04_single_head_attention.py` replaces the fixed average with one learned
query/key/value head. It prints the full Stage 3 diagnostic set: parameter and
shape reports, normalization and causal-mask checks, prefix sensitivity,
initial/final losses, generated text, and the learned attention matrix for an
eight-character input.

Run the full FP32/eager checkpoint (5,000 steps at a `1e-3` learning rate):

```powershell
python .\my-gpt\04_single_head_attention.py
```

Run a quick CPU smoke check:

```powershell
python .\my-gpt\04_single_head_attention.py --device cpu `
    --max-iters 10 --eval-iters 2 --sample-length 40
```

The default attention probe is `"To be or"`. Use `--attention-text` to inspect
another non-empty string no longer than `block_size` whose characters occur in
the training corpus.

## Stage 4: four causal self-attention heads

`05_multi_head_attention.py` splits the 32 representation channels across four
independent heads. Each head produces eight channels and its own causal
attention matrix; concatenating the four outputs restores `(B,T,32)`. There is
still no output projection, residual connection, feed-forward network, or
LayerNorm, so the experiment isolates multi-head routing while retaining the
same 7,553 trainable parameters as Stage 3.

Run the full checkpoint with the Stage 3 training conditions:

```powershell
python .\my-gpt\05_multi_head_attention.py
```

Run a quick CPU smoke check:

```powershell
python .\my-gpt\05_multi_head_attention.py --device cpu `
    --max-iters 10 --eval-iters 2 --sample-length 40
```

The report includes per-head Q/K/V shapes, combined attention weights with
shape `(B,H,T,T)`, row-sum and causal-mask checks, train/validation loss,
generated text, all four learned attention matrices, and each head's routing
distribution for the final character of `"To be or"`.

## Stage 5: position-wise feed-forward network

`05_feed_forward.py` keeps the four causal attention heads and adds a ReLU
feed-forward network after them:

```text
(B,T,32) -> Linear(32,128) -> ReLU -> Linear(128,32) -> (B,T,32)
```

The FFN is applied independently at every position. This stage intentionally
still has no residual connection, LayerNorm, dropout, attention output
projection, or GELU. With the default dimensions, the FFN contributes 8,352
parameters and the complete model has 15,905 trainable parameters.

Run the full 5,000-step checkpoint:

```powershell
python .\my-gpt\05_feed_forward.py
```

Run a quick CPU smoke check:

```powershell
python .\my-gpt\05_feed_forward.py --device cpu `
    --max-iters 10 --eval-iters 2 --sample-length 40
```

The report prints the `(B,T,4C)` hidden shape, verifies attention remains
normalized and causal, and performs a direct FFN locality experiment: changing
only position 0 must leave the FFN outputs at positions 1 through 7 exactly
unchanged.

## Stage 6: residual connections

`06_residual_connections.py` preserves the Stage 5 attention and FFN modules
and changes only how their outputs update the representation:

```python
x = x + self.sa(x)
x = x + self.ffwd(x)
```

Both additions preserve `(B,T,C)`, add no trainable parameters, and leave the
complete model at exactly 15,905 parameters. The checkpoint reports `x0`, both
sublayer updates, both post-residual representations, and their norms. It also
retains the causal-attention and position-wise FFN checks from Stage 5.

Run the full 5,000-step checkpoint with the same controlled settings as Stage
5:

```powershell
python .\my-gpt\06_residual_connections.py
```

Run a quick CPU smoke check:

```powershell
python .\my-gpt\06_residual_connections.py --device cpu `
    --max-iters 10 --eval-iters 2 --sample-length 40
```

`test_residual_connections.py` additionally proves the exact two-addition
dataflow, the zero-update identity behavior, the direct residual gradient
route, unchanged parameter count, and causal masking.

## Stage 7: pre-norm LayerNorm and GELU

`07_layer_norm.py` adds two learned LayerNorms around the Stage 6 residual
branches and switches the feed-forward activation from ReLU to GELU:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

This is a pre-norm arrangement: each complicated sublayer receives a
normalized `(B,T,C)` view, while the residual stream itself keeps a direct
identity path. LayerNorm acts independently on the `C` features of every
token and contributes a learned scale and bias of shape `(C,)`. With two
LayerNorms and the default `C=32`, Stage 7 adds 128 trainable parameters, for
16,033 total. GELU changes the activation behavior but adds no parameters.

Run the full 5,000-step checkpoint with the same controlled settings as Stage
6:

```powershell
python .\my-gpt\07_layer_norm.py
```

Run a quick CPU smoke check:

```powershell
python .\my-gpt\07_layer_norm.py --device cpu `
    --max-iters 10 --eval-iters 2 --sample-length 40
```

The report includes a standalone LayerNorm experiment, all pre-norm and
residual shapes, one token's mean and population variance before and after
`ln1`, initial LayerNorm affine parameters, residual magnitudes, train and
validation loss, and generated text. `test_layer_norm.py` verifies the exact
pre-norm dataflow, feature-axis normalization without token or batch mixing,
GELU, causal masking, residual identity path, gradients, and the 16,033
parameter count.

## Stage 8: stacked Transformer blocks

`08_transformer_block.py` packages the Stage 7 pre-norm attention and FFN
branches into a reusable `Block`, gives multi-head attention its learned
output projection, and stacks four independent blocks:

```text
token + position embeddings
    -> Block 0 -> Block 1 -> Block 2 -> Block 3
    -> final LayerNorm
    -> language-model head
```

Each block retains the two residual updates:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

The attention output projection mixes the concatenated head slices before
they re-enter the residual stream. At the default `C=32`, one block has
12,608 parameters: 3,072 Q/K/V weights, 1,056 output-projection parameters,
8,352 FFN parameters, and 128 LayerNorm parameters. With four blocks, the
actual 65-character vocabulary, final LayerNorm, embeddings, and LM head, the
complete model has exactly 54,977 trainable parameters.

Run the full 5,000-step checkpoint on the automatically selected accelerator:

```powershell
python .\my-gpt\08_transformer_block.py
```

Run a quick CPU smoke check:

```powershell
python .\my-gpt\08_transformer_block.py --device cpu `
    --max-iters 10 --eval-iters 2 --sample-length 40
```

The report records the embedding and every block's `(B,T,C)` shape and
residual-stream norm, the final LayerNorm norm, causal attention checks, and
one first-head query gradient from every block before training. With the
default `(32,8,32)` diagnostic batch, the initial final-LayerNorm norm should
be close to `sqrt(8192) = 90.51`.

`test_transformer_block.py` verifies the projected multi-head path, exact
pre-norm block flow, residual identity behavior, independent block instances,
the final normalization, end-to-end causality, gradient propagation through
all four blocks, context cropping, validation errors, and both exact parameter
budgets.

## Stage 9: scale context from 8 to 64

`09_context_length.py` reuses the exact Stage 8 model and changes the default
context length only:

```text
B=32, T=64, C=32, H=4, D=8, L=4
```

The position embedding grows from `(8,32)` to `(64,32)`, and each head's
causal mask grows from `(8,8)` to `(64,64)`. The model therefore has 56,769
trainable parameters--only 1,792 more than Stage 8--while each full attention
matrix contains 64 times as many relationships.

Run the full FP32/eager XPU experiment:

```powershell
python .\my-gpt\09_context_length.py --device xpu
```

Run a quick CPU smoke check while keeping the benchmark inside the shortened
training budget:

```powershell
python .\my-gpt\09_context_length.py --device cpu `
    --max-iters 10 --eval-iters 2 --sample-length 40 `
    --benchmark-warmup 2 --benchmark-steps 5
```

The benchmark warms up on 20 optimizer steps and times the following 100 real
training steps by default. Those steps are part of `max_iters`; benchmarking
does not add updates, consume extra training batches, or reset AdamW state.
The timer covers batch construction and transfer, forward pass, loss,
backward pass, and optimizer update. Accelerator synchronization brackets the
timed window. On XPU, the report also records peak PyTorch allocated and
reserved memory. Pass `--benchmark-steps 0` to disable it.

The first full run on the Intel Arc 140T with PyTorch 2.13.0+xpu produced:

| Run | T | C | H | L | Params | train loss | val loss | bench seconds | iter/s | tokens/s | allocated MB | reserved MB |
| --- | -: | -: | -: | -: | -----: | ---------: | -------: | ------------: | -----: | -------: | -----------: | ----------: |
| Stage 8 | 8 | 32 | 4 | 4 | 54,977 | 1.9500 | 2.0425 | ... | ... | ... | ... | ... |
| Stage 9 | 64 | 32 | 4 | 4 | 56,769 | 1.7684 | 1.9225 | 3.836 | 26.066 | 53,383.602 | 27.698 | 50.000 |

Both runs used 5,000 optimizer steps, but they did not use the same token
budget: Stage 8 trained on 1.28 million token positions and Stage 9 trained on
10.24 million. The lower Stage 9 validation loss therefore reflects the
combined experiment of longer context and eight times as many token
predictions, not a context-only causal estimate.

The seeded 500-character Stage 9 sample was:

```text
siclan where make in more stonvant hirs
Than what therk thengts.
Buth soon ome, he mors Enguman cong
With tracters art a lequingly ever armlian
His is himselff, no?


DUKE VOLINGBY:

Hidjour you twaph'd! be coite Laray vurrour?
So mun mark he master, away of this I, he's
Wknamings I havan: dithtly and bloody me
not umzemand, if joil aurved come,
Engel have crothinngour, sir, by reast we
Bustsizers, dor mornighore do binss!
And park thy father vachile, and whose,
But all the no dobldest!

ISABET:
```
