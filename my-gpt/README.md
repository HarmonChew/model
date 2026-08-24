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
| Stage 9 | 64 | 32 | 4 | 4 | 56,769 | 1.7684 | 1.9225 | 2.739 | 36.514 | 74,780.672 | 27.698 | 50.000 |

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

## Stage 10: scale width from 32 to 64

`10_scale_width.py` retains the Stage 9 token budget and context, then doubles
only the residual-stream width:

```text
B=32, T=64, C=64, H=4, D=16, FF=256, L=4
```

The attention maps remain `(32,4,64,64)`, because their final two dimensions
depend on context length. The token representations become `(32,64,64)`, each
head's Q/K/V projection becomes `(32,64,16)`, and the FFN grows from
`32 -> 128 -> 32` to `64 -> 256 -> 64`.

Each block has 49,792 parameters and the complete 65-character model has
exactly 211,777:

```text
token embedding       4,160
position embedding    4,096
four blocks          199,168
final LayerNorm         128
LM head               4,225
                     -------
total                211,777
```

Run the full FP32/eager XPU experiment:

```powershell
python .\my-gpt\10_scale_width.py --device xpu
```

Run a quick CPU smoke check:

```powershell
python .\my-gpt\10_scale_width.py --device cpu `
    --batch-size 2 --max-iters 2 --eval-interval 1 --eval-iters 1 `
    --sample-length 40 --benchmark-warmup 1 --benchmark-steps 2 `
    --checkpoint-path .\my-gpt\checkpoints\stage_10_smoke.pt
```

Stage 10 benchmarks a separately initialized, disposable model and optimizer.
Its warmup and timed updates therefore cannot change the model used for the
5,000-step experiment. The actual model is reseeded before construction so
benchmark RNG consumption cannot alter its initialization either.

Every periodic validation result is compared using a strict improvement rule.
The separate final evaluation at step 5,000 is also considered, and the best
weights are written to `my-gpt/checkpoints/stage_10_best_model.pt`. Checkpoint
artifacts are ignored by Git. The final report records the full architecture,
parameter count, final train/validation losses, generalization gap, best
validation loss and step, throughput, allocator peaks, and generated sample.

The first full Stage 10 run on the Intel Arc 140T produced:

| Run | T | C | H | L | Params | train loss | val loss | gap | best val / step | iter/s | tokens/s | allocated MB | reserved MB |
| --- | -: | -: | -: | -: | -----: | ---------: | -------: | --: | --------------: | -----: | -------: | -----------: | ----------: |
| Stage 10 | 64 | 64 | 4 | 4 | 211,777 | 1.4930 | 1.6800 | 0.1870 | 1.6800 / 5,000 | 15.487 | 31,717.613 | 46.270 | 54.000 |

The Stage 9 row above used its legacy in-training timing window, whereas Stage
10 benchmarks a disposable model before the experiment. Both time the same
optimizer-step work after 20 warmups, but their surrounding harness placement
differs, so the exact throughput ratio is informative rather than a perfectly
matched A/B measurement. Future stages use the independent Stage 10 method.

The seeded 500-character sample from the best checkpoint was:

```text
Me lack, thou have beard a good exture.

TANIrch:
Bredk thoughts, in a ofuld right.
Thou wounder, thou would at thurn arms But of here,

JUlie aloner:
A bloody, sir, go weeet alonion
In pour you telposienteds,
That bay very wine me: nor here horself,
But usir I will he sayk; my respect!

Second ELAUREL:
With meaning may arrow to like.

FRIAR go:
To be had into thy hough smord your stood
Thousime of dartion, good to things,
And part thy fathor vain's angelio,
And distancer them I come, Darls: bur
```

## Stage 11: train the same model longer

`11_train_longer.py` keeps the complete Stage 10 setup unchanged and increases
only the training duration from 5,000 to 10,000 optimizer steps:

```text
B=32, T=64, C=64, H=4, D=16, FF=256, L=4, learning_rate=1e-3
```

The existing `stage_10_best_model.pt` contains only a raw model state dict. It
does not contain AdamW's moving averages, a training step, or the prior best
validation loss. Loading it with a new optimizer would therefore be a warm
start, not an exact continuation. To preserve the one-variable experiment,
the default Stage 11 command performs a fresh seeded run through all 10,000
steps. Its measurements at steps 4,000, 4,500, and 5,000 exactly reproduced
the Stage 10 values.

Run the clean FP32/eager XPU experiment:

```powershell
python .\my-gpt\11_train_longer.py --device xpu
```

Stage 11 checkpoints are fully resumable. Resume one to a higher absolute
target step with:

```powershell
python .\my-gpt\11_train_longer.py --device xpu `
    --max-iters 15000 `
    --resume-from .\my-gpt\checkpoints\stage_11_best_checkpoint.pt
```

The saved payload contains the model and optimizer states, absolute step, best
loss and step, training-batch generator state, CPU and accelerator RNG states,
architecture and training metadata, a corpus fingerprint, and optimizer-lineage
provenance. Loading validates the metadata and restores AdamW's state tensors
to the model's device.

On XPU, the saver synchronizes the device and copies an owned snapshot of every
model and optimizer tensor to CPU before calling `torch.save`. This prevents
asynchronous device storage from producing a checkpoint whose scalar metadata
is newer than its tensor contents. The test suite covers repeated XPU saves and
reloads, including non-empty AdamW state and RNG restoration.

A weights-only checkpoint is rejected by default. The explicitly confounded
alternative remains available under an opt-in flag and writes to a separate
path:

```powershell
python .\my-gpt\11_train_longer.py --device xpu `
    --resume-from .\my-gpt\checkpoints\stage_10_best_model.pt `
    --allow-optimizer-restart --legacy-step 5000 `
    --checkpoint-path `
        .\my-gpt\checkpoints\stage_11_restarted_adamw_best_checkpoint.pt
```

That mode is reported as `Stage-10 weights + restarted AdamW state`, and the
restart provenance remains in subsequent full checkpoints.

The clean Stage 11 run produced this fixed-batch evaluation curve:

| Step | Train loss | Validation loss | Gap | Best validation / step |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4.3394 | 4.3465 | 0.0071 | 4.3465 / 0 |
| 500 | 2.2155 | 2.2332 | 0.0177 | 2.2332 / 500 |
| 1,000 | 1.9576 | 2.0240 | 0.0664 | 2.0240 / 1,000 |
| 1,500 | 1.7871 | 1.9079 | 0.1208 | 1.9079 / 1,500 |
| 2,000 | 1.6989 | 1.8478 | 0.1489 | 1.8478 / 2,000 |
| 2,500 | 1.6294 | 1.7787 | 0.1493 | 1.7787 / 2,500 |
| 3,000 | 1.5853 | 1.7522 | 0.1668 | 1.7522 / 3,000 |
| 3,500 | 1.5585 | 1.7272 | 0.1687 | 1.7272 / 3,500 |
| 4,000 | 1.5287 | 1.7100 | 0.1813 | 1.7100 / 4,000 |
| 4,500 | 1.5121 | 1.6919 | 0.1798 | 1.6919 / 4,500 |
| 5,000 | 1.4930 | 1.6800 | 0.1870 | 1.6800 / 5,000 |
| 5,500 | 1.4877 | 1.6827 | 0.1950 | 1.6800 / 5,000 |
| 6,000 | 1.4698 | 1.6603 | 0.1905 | 1.6603 / 6,000 |
| 6,500 | 1.4564 | 1.6542 | 0.1977 | 1.6542 / 6,500 |
| 7,000 | 1.4486 | 1.6520 | 0.2034 | 1.6520 / 7,000 |
| 7,500 | 1.4414 | 1.6486 | 0.2073 | 1.6486 / 7,500 |
| 8,000 | 1.4300 | 1.6401 | 0.2101 | 1.6401 / 8,000 |
| 8,500 | 1.4270 | 1.6314 | 0.2044 | 1.6314 / 8,500 |
| 9,000 | 1.4177 | 1.6361 | 0.2184 | 1.6314 / 8,500 |
| 9,500 | 1.4134 | 1.6257 | 0.2123 | 1.6257 / 9,500 |
| 10,000 | 1.4100 | 1.6167 | 0.2068 | 1.6167 / 10,000 |

Validation improved by `0.0633` from step 5,000 to step 10,000, and the final
measurement was a fresh best. The gap grew from `0.1870` to `0.2068`, but the
validation curve did not show a sustained reversal: the isolated increases at
5,500 and 9,000 were followed by lower losses. Stage 10 was undertrained, and
this run does not provide evidence for adding dropout next.

The constant learning rate also has not produced a settled plateau. Before
another capacity change, the most informative Stage 12 would be a controlled
learning-rate experiment: fork the exact step-10,000 checkpoint into a constant
`1e-3` continuation and a decayed-learning-rate continuation, with matching
batch and evaluation streams.

The independent XPU benchmark measured 22.360 iterations/s, 45,792.771
tokens/s, 46.270 MB peak allocated memory, and 54.000 MB peak reserved memory.
The best checkpoint is `my-gpt/checkpoints/stage_11_best_checkpoint.pt`; the
complete console record is `my-gpt/checkpoints/stage_11_training.log`.

The seeded 500-character sample from the step-10,000 checkpoint was:

```text
Me lanchery come, to morney.
That therefold troubly, dared not suding for any
To you, I and my envented on rough murre and Butoor't;
Such mlaint in itset love, you?

KING RICHARD II:
I can die more!

CATESBY:
I can band with thee held make he made heeds?

GLOUCESTER:
Not tk this so were, for thy countinusing me
Bagy man a powery like.

VOLUMNIA:
To the wretched is hour, sir, by this? I do buried
on drawn's world to the wretchers all brother?

FRIAR LAURENCE:
Set you all the news about are smet m
```

## Stage 12: controlled learning-rate drop

`12_learning_rate_drop.py` turns the proposed learning-rate test into one
paired experiment. It loads the exact Stage 11 step-10,000 checkpoint twice
and advances each independent branch to the same absolute target step:

```text
                              Stage 11 source
                       step 10,000, lr = 1e-3
                                  |
                   +--------------+--------------+
                   |                             |
             control branch                LR-drop branch
               lr = 1e-3                     lr = 3e-4
                   |                             |
             step 15,000                   step 15,000
```

The architecture and other training defaults remain unchanged:

```text
B=32, T=64, C=64, H=4, D=16, FF=256, L=4
source_step=10000, max_iters=15000, eval_interval=500, eval_iters=100
```

Because `max_iters` is an absolute target, each branch performs exactly 5,000
additional optimizer updates, numbered 10,001 through 15,000. Run the default
FP32/eager XPU experiment with:

```powershell
python .\my-gpt\12_learning_rate_drop.py --device xpu
```

The default checkpoint paths are:

```text
source:   my-gpt/checkpoints/stage_11_best_checkpoint.pt
control:  my-gpt/checkpoints/stage_12_control_best_checkpoint.pt
LR drop:  my-gpt/checkpoints/stage_12_lr_drop_best_checkpoint.pt
```

They can be changed with `--source-checkpoint`,
`--control-checkpoint-path`, and `--lr-drop-checkpoint-path`. The fork and
learning-rate defaults can likewise be changed with `--source-step`,
`--max-iters`, `--source-learning-rate`, `--control-learning-rate`, and
`--reduced-learning-rate`. The control rate must equal the source rate, and
the reduced rate must be lower than the control rate.

### Exact branch restoration

Each branch is constructed and restored independently from the source file.
The loader first validates the Stage 11 architecture, training metadata, and
corpus fingerprint, then restores the model weights, AdamW state, explicit
training-batch generator, and saved RNG state. Only after
`optimizer.load_state_dict(...)` has restored AdamW's parameter groups and
moment estimates does Stage 12 set every parameter group's learning rate:

```python
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

for param_group in optimizer.param_groups:
    param_group["lr"] = branch_learning_rate
```

The order matters because loading the optimizer state also restores its saved
`1e-3` learning rate. Applying the override first would silently undo the
intended LR drop when the checkpoint was loaded. The control branch writes
`1e-3` back, while the LR-drop branch writes `3e-4`; the AdamW first- and
second-moment estimates and step counters remain restored in both cases.

Both branches begin with the same model, optimizer history, training-batch
generator state, and global RNG state. They therefore consume matching
training batches after the fork. The learning rate is the only intentional
training difference.

### Fixed evaluation panel

Stage 11 and Stage 12 do not draw a fresh validation panel at every reported
step. `evaluate_on_fixed_batches` recreates the evaluation generator from the
same seed for every measurement, so all measurements and both branches use the
same 100 sampled training batches and the same 100 sampled validation batches
by default.

This corrects an easy misreading of the Stage 11 curve: its isolated increases
at steps 5,500 and 9,000 are not draw-to-draw noise from newly sampled
evaluation batches. They are changes in model loss on a fixed, finite panel.
That panel is still an estimate rather than the full corpus, but holding it
fixed makes the Stage 12 branch comparison more direct.

### Checkpoint safety and provenance

Before training, Stage 12 requires a full checkpoint at the requested source
step whose `best_step` equals its saved `step`. It also requires non-empty
AdamW moment state, a saved training-generator state, a CPU RNG state, known
optimizer provenance, and no recorded optimizer restart. A weights-only or
confounded warm-start checkpoint is rejected.

The source, control-output, and LR-drop-output paths must all resolve to
different files. Stage 12 hashes the source before training and verifies the
same hash after both branches, preventing either output from silently replacing
the common fork. Each branch output is materialized at step 10,000 under its
active learning rate before its first new update, so a valid resumable output
still exists if that branch never improves on the source validation loss.

Branch checkpoints retain the complete Stage 11 resumable payload and add:

```text
checkpoint_kind = "best"
experiment.stage
experiment.branch
experiment.source_checkpoint_sha256
experiment.source_step
experiment.source_learning_rate
experiment.branch_learning_rate
experiment.learning_rate_changed
```

The standard optimizer state and `training_config.learning_rate` also record
the branch's active rate. The generated samples use each branch's final
step-15,000 model with the same prompt, seed, and token count; the saved
checkpoint for each branch remains the best fixed-panel validation state seen
through that branch.

### Results

The full XPU experiment showed a clear advantage for lowering the learning
rate. At the equal-budget step-15,000 endpoint, the `3e-4` branch had a
validation loss `0.0201` below the `1e-3` control. It also had the lower train
loss, so the result is not a regularization trade in which the reduced rate
merely accepts a worse training fit.

| Metric | Control (`1e-3`) | LR drop (`3e-4`) | Drop - control |
| --- | ---: | ---: | ---: |
| Final train loss at step 15,000 | 1.3743 | **1.3470** | -0.0273 |
| Final validation loss at step 15,000 | 1.6029 | **1.5829** | -0.0201 |
| Final generalization gap | 0.2286 | 0.2358 | +0.0072 |
| Best validation loss / step | 1.5989 / 13,500 | **1.5791 / 13,000** | -0.0198 |

The complete paired fixed-panel curve was:

| Step | Control train | Control val | LR-drop train | LR-drop val |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1.4100 | 1.6167 | 1.4100 | 1.6167 |
| 10,500 | 1.4031 | 1.6134 | 1.3699 | 1.5845 |
| 11,000 | 1.4024 | 1.6083 | 1.3641 | 1.5812 |
| 11,500 | 1.3965 | 1.6160 | 1.3606 | 1.5864 |
| 12,000 | 1.3908 | 1.6097 | 1.3572 | 1.5819 |
| 12,500 | 1.3897 | 1.6132 | 1.3556 | 1.5846 |
| 13,000 | 1.3811 | 1.6061 | 1.3512 | **1.5791** |
| 13,500 | 1.3803 | **1.5989** | 1.3501 | 1.5815 |
| 14,000 | 1.3785 | 1.6068 | 1.3474 | 1.5815 |
| 14,500 | 1.3766 | 1.6061 | 1.3466 | 1.5843 |
| 15,000 | 1.3743 | 1.6029 | 1.3470 | 1.5829 |

The control still improved on the Stage 11 source, from `1.6167` to `1.6029`
at the final step, so `1e-3` had not stopped making progress. The stronger
result is that the restored AdamW optimizer made substantially finer progress
immediately after its global step size was reduced: by step 10,500 the
LR-drop branch was already at `1.5845`. This is direct evidence that `1e-3`
had become too large for efficient late-stage refinement, while `3e-4` was
not so small that useful learning stopped.

```text
control checkpoint: my-gpt/checkpoints/stage_12_control_best_checkpoint.pt
LR-drop checkpoint: my-gpt/checkpoints/stage_12_lr_drop_best_checkpoint.pt
complete console record: my-gpt/checkpoints/stage_12_training.log
```

## Stage 13: second controlled learning-rate drop

`13_second_learning_rate_drop.py` repeats the paired fork after Stage 12. It
loads the exact `3e-4` branch winner at step 13,000 twice and advances both
independent continuations to the same absolute target:

```text
                         Stage 12 LR-drop winner
                          step 13,000, lr = 3e-4
                                    |
                     +--------------+--------------+
                     |                             |
               control branch              second-drop branch
                 lr = 3e-4                      lr = 1e-4
                     |                             |
               step 18,000                   step 18,000
```

The architecture and all other defaults remain fixed:

```text
B=32, T=64, C=64, H=4, D=16, FF=256, L=4
source_step=13000, max_iters=18000, eval_interval=500, eval_iters=100
```

Run the full FP32/eager XPU experiment with:

```powershell
python .\my-gpt\13_second_learning_rate_drop.py --device xpu
```

The default artifacts are:

```text
source:       my-gpt/checkpoints/stage_12_lr_drop_best_checkpoint.pt
control:      my-gpt/checkpoints/stage_13_control_best_checkpoint.pt
second drop:  my-gpt/checkpoints/stage_13_lr_drop_best_checkpoint.pt
console log:  my-gpt/checkpoints/stage_13_training.log
```

Stage 13 adds a stricter source gate to the Stage 12 safeguards. The source
must be a full best checkpoint with `step == best_step == 13000`, uninterrupted
AdamW provenance, non-empty moment state, the saved batch-generator and RNG
states, an active `3e-4` LR in both its training metadata and every optimizer
parameter group, and explicit provenance identifying it as Stage 12's
`lr_drop` branch. An XPU or CUDA run also requires the matching saved device
RNG state.

Each branch independently restores that source before its LR is overwritten.
The source is hashed before and after validation, around each branch load, and
after training. Stage 13 checkpoints use new paths and record
`experiment.stage = 13` plus the Stage 12 source hash, source branch, source
step, source LR, and active branch LR. The source SHA-256 remained:

```text
46eb29ebb24e4ed18d1271f80447e873e99f822cefc969a876a420383238b88f
```

### Results

The second reduction won on both training and validation loss. At the equal
step-18,000 endpoint, `1e-4` was `0.0057` lower on train loss and `0.0093`
lower on validation loss. Both branches selected step 17,000 as their best
fixed-panel checkpoint, where the second drop led by `0.0082` validation-loss
units.

| Metric | Control (`3e-4`) | Second drop (`1e-4`) | Drop - control |
| --- | ---: | ---: | ---: |
| Final train loss at step 18,000 | 1.3358 | **1.3301** | -0.0057 |
| Final validation loss at step 18,000 | 1.5803 | **1.5710** | -0.0093 |
| Final generalization gap | 0.2445 | **0.2410** | -0.0035 |
| Best validation loss / step | 1.5770 / 17,000 | **1.5688 / 17,000** | -0.0082 |

The complete paired fixed-panel curve was:

| Step | Control train | Control val | Second-drop train | Second-drop val |
| ---: | ---: | ---: | ---: | ---: |
| 13,000 | 1.3512 | 1.5791 | 1.3512 | 1.5791 |
| 13,500 | 1.3501 | 1.5815 | 1.3390 | 1.5718 |
| 14,000 | 1.3474 | 1.5815 | 1.3376 | 1.5715 |
| 14,500 | 1.3466 | 1.5843 | 1.3358 | 1.5715 |
| 15,000 | 1.3470 | 1.5829 | 1.3356 | 1.5729 |
| 15,500 | 1.3436 | 1.5799 | 1.3338 | 1.5694 |
| 16,000 | 1.3419 | 1.5815 | 1.3330 | 1.5707 |
| 16,500 | 1.3397 | 1.5803 | 1.3323 | 1.5715 |
| 17,000 | 1.3384 | **1.5770** | 1.3316 | **1.5688** |
| 17,500 | 1.3373 | 1.5774 | 1.3310 | 1.5708 |
| 18,000 | 1.3358 | 1.5803 | 1.3301 | 1.5710 |

The control's measurements through step 15,000 exactly reproduce the prior
Stage 12 `3e-4` branch to four decimals. This provides an end-to-end check that
the model, optimizer, and training-batch stream resumed from the intended
step-13,000 state.

Like the first reduction, `1e-4` did not improve validation by accepting a
worse training fit: it found lower-loss parameters on both datasets. The
second decay therefore still addressed an optimization limitation. Its return
was smaller, however. Most of the validation gain appeared in the first 500
steps, and the best `1e-4` result was only `0.0030` below its step-13,500
measurement after another 3,500 updates. Training loss continued downward
after the best step while validation rose slightly at 17,500 and 18,000.

That is stronger evidence of diminishing optimization returns and a developing
generalization plateau, although two post-best measurements are not enough to
declare definitive overfitting. Generated text remains a qualitative sample;
both branches use the same newline prompt, seed, sampling procedure, and token
count, while the loss curves remain the comparison metric.

## Stage 14: controlled residual-path dropout

`14_dropout.py` tests whether regularization helps after the Stage 13
optimization gains have flattened. It loads the exact Stage 13 `1e-4` winner
at step 17,000 twice and advances both branches to step 22,000:

```text
                         Stage 13 LR-drop winner
                          step 17,000, lr = 1e-4
                                    |
                     +--------------+--------------+
                     |                             |
               control branch                dropout branch
                    p = 0.0                       p = 0.1
                     |                             |
               step 22,000                   step 22,000
```

Dropout is applied only after the attention output projection and after the
FFN's final `C`-dimensional projection, immediately before each residual
addition:

```text
attention heads -> output projection -> dropout -> residual addition
FFN hidden GELU -> final projection   -> dropout -> residual addition
```

There are two `nn.Dropout` modules per block and eight in the four-block
model. They contain no parameters or persistent buffers, so the Stage 13 model
state loads strictly without renaming any tensor, AdamW's parameter order is
unchanged, and both branches still have exactly 211,777 trainable parameters.
The probability and placement are written explicitly into Stage 14 checkpoint
metadata because they cannot be recovered from the state dict itself.

Run the complete FP32/eager XPU experiment with:

```powershell
python .\my-gpt\14_dropout.py --device xpu
```

The default run first evaluates the Stage 12 winner and both Stage 13 winners
on 500 newly sampled validation batches. It then performs the paired Stage 14
fork. Pass `--precise-eval-iters 0` to skip the preflight. A short CPU plumbing
check can use an absolute target just above the fork:

```powershell
python .\my-gpt\14_dropout.py --device cpu `
    --max-iters 17001 --precise-eval-iters 2 --sample-length 0 `
    --control-checkpoint-path .\my-gpt\checkpoints\stage_14_control_smoke.pt `
    --dropout-checkpoint-path .\my-gpt\checkpoints\stage_14_dropout_smoke.pt
```

### Precise validation preflight

The fixed 100-batch panel remains the checkpoint-selection and training-curve
metric. Because those samples have been reused across several stages, Stage 14
also uses a separate validation-only panel of 500 batches with seed 1,340. All
three checkpoints see the same new batches, enabling paired loss differences.
This reduces sampling uncertainty but is still a new sample from the existing
validation split, not a third held-out dataset.

| Checkpoint | Step | Fixed-panel loss | Fresh 500-batch loss | SE |
| --- | ---: | ---: | ---: | ---: |
| Stage 12 `3e-4` winner | 13,000 | 1.5791 | 1.579669 | 0.002627 |
| Stage 13 `3e-4` control | 17,000 | 1.5770 | 1.576908 | 0.002684 |
| Stage 13 `1e-4` winner | 17,000 | 1.5688 | **1.569129** | 0.002672 |

The paired deltas were more precise than the standalone standard errors:

| Paired comparison | Mean delta | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Stage 13 control - Stage 12 winner | -0.002761 | 0.000353 | [-0.003452, -0.002069] |
| Stage 13 `1e-4` - Stage 13 control | **-0.007779** | 0.000178 | **[-0.008128, -0.007430]** |

The fresh panel therefore confirms rather than reverses the Stage 13 result.
The second learning-rate drop retains a clear advantage over its paired
control, and both Stage 13 branches improve on the Stage 12 source in this
larger measurement.

### Exact fork and train/eval behavior

Each branch independently restores the same model tensors, all 94 non-empty
AdamW state entries, the explicit training-batch generator, and the saved CPU
and XPU RNG states. Both branches receive the same post-fork training batches.
The `p=0.1` branch then consumes global XPU RNG for masks; that divergence is
the intended treatment rather than a loss of pairing.

Every reported train and validation measurement calls `model.eval()` and then
restores the previous mode. Dropout is therefore off for the complete table
below. Training calls `model.train()` explicitly, so dropout is active only
during optimizer updates. The source checkpoint remained unchanged at SHA-256:

```text
650eb2399a29bced5fa5398dc129228e15e2f4a68fb09db9c44a5400b1f739f1
```

### Results

The control made a small further improvement. Residual dropout introduced at
step 17,000 caused a large loss increase and did not recover within 5,000
updates:

| Step | Control train | Control val | Dropout train | Dropout val |
| ---: | ---: | ---: | ---: | ---: |
| 17,000 | 1.3316 | 1.5688 | 1.3316 | 1.5688 |
| 17,500 | 1.3310 | 1.5708 | 1.4535 | 1.6283 |
| 18,000 | 1.3301 | 1.5710 | 1.4491 | 1.6279 |
| 18,500 | 1.3300 | 1.5713 | 1.4479 | 1.6285 |
| 19,000 | 1.3283 | 1.5700 | 1.4476 | 1.6284 |
| 19,500 | 1.3270 | 1.5693 | 1.4449 | 1.6260 |
| 20,000 | 1.3268 | 1.5683 | 1.4438 | **1.6250** |
| 20,500 | 1.3265 | 1.5706 | 1.4433 | 1.6271 |
| 21,000 | 1.3249 | **1.5682** | 1.4419 | 1.6252 |
| 21,500 | 1.3239 | 1.5709 | 1.4416 | 1.6294 |
| 22,000 | 1.3237 | 1.5686 | 1.4403 | 1.6266 |

| Metric | Control (`p=0.0`) | Dropout (`p=0.1`) | Dropout - control |
| --- | ---: | ---: | ---: |
| Final train loss at step 22,000 | **1.3237** | 1.4403 | +0.1166 |
| Final validation loss at step 22,000 | **1.5686** | 1.6266 | +0.0581 |
| Final generalization gap | 0.2449 | **0.1864** | -0.0585 |
| Best validation loss / step | **1.5682 / 21,000** | 1.5688 / 17,000 | +0.0007 |

The smaller dropout-branch gap is not evidence of better generalization here.
Its measured train loss worsened roughly twice as much as its validation loss,
and both losses are substantially above the control. No post-fork dropout
checkpoint beat the common source, so the saved dropout "best" remains the
step-17,000 source weights with explicit `p=0.1` runtime metadata.

A post-run evaluation of the two saved best checkpoints on the same fresh
500-batch panel also favors the control:

| Best checkpoint | Fresh loss | SE |
| --- | ---: | ---: |
| Stage 14 control, step 21,000 | **1.566965** | 0.002689 |
| Stage 14 dropout, step 17,000 | 1.569129 | 0.002672 |

The paired `dropout - control` delta is `+0.002164 +/- 0.000202 SE`, with a
95% interval of `[+0.001767, +0.002561]`.

Stage 14 therefore rejects adding `p=0.1` residual-path dropout during
late-stage refinement of this already-trained model. It does not establish
that dropout would hurt if present from step zero, and it does not test a
smaller probability or a different placement. Those require separate
experiments rather than reinterpretation of this fork.

```text
control checkpoint: my-gpt/checkpoints/stage_14_control_best_checkpoint.pt
dropout checkpoint: my-gpt/checkpoints/stage_14_dropout_best_checkpoint.pt
complete console record: my-gpt/checkpoints/stage_14_training.log
post-run precise record: my-gpt/checkpoints/stage_14_posthoc_precise.log
```
