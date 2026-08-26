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

## Stage 15: residual dropout from initialization

`15_dropout_from_initialization.py` tests the question left open by Stage 14:
does `p=0.1` residual-path dropout help when it is present for the model's
entire training run? The script creates one seeded model state on CPU, clones
that exact state into both branches, and constructs a separate AdamW optimizer
for each:

```text
                     one cloned initialization
                               |
                    +----------+----------+
                    |                     |
              control, p=0.0        dropout, p=0.1
                    |                     |
                    +--- same batches ---+
```

The architecture remains `B=32, T=64, C=64, H=4, D=16, FF=256, L=4` by
default. Dropout remains in the two Stage 14 locations only: after the
attention output projection and after the FFN output projection, immediately
before their residual additions. Attention-weight dropout is not added.

Each branch owns an independent CPU batch generator initialized with the same
seed (`seed + 1` by default), so both consume the same training examples even
though the dropout branch also samples masks. After model construction and
state loading, the global CPU and accelerator RNGs are reset to the same
training seed (`seed + 3` by default) immediately before each branch begins
its optimizer updates.

Both branches use the discovered learning-rate schedule, indexed by the
zero-based optimizer update:

```text
[0, 10000)       lr = 1e-3
[10000, 13000)   lr = 3e-4
[13000, 18000)   lr = 1e-4
```

Thus the state reported as step 10,000 has completed 10,000 updates at
`1e-3`; update index 10,000 (the next update) uses `3e-4`.

Run the full experiment with:

```powershell
python .\my-gpt\15_dropout_from_initialization.py --device xpu
```

The default outputs are:

```text
control:  my-gpt/checkpoints/stage_15_control_best_checkpoint.pt
dropout:  my-gpt/checkpoints/stage_15_dropout_best_checkpoint.pt
```

A tiny CPU plumbing run can compress both decay boundaries and use separate
output paths:

```powershell
python .\my-gpt\15_dropout_from_initialization.py --device cpu `
    --first-decay-step 1 --second-decay-step 2 --max-iters 3 `
    --eval-interval 1 --eval-iters 2 --precise-eval-iters 2 `
    --sample-length 0 `
    --control-checkpoint-path .\my-gpt\checkpoints\stage_15_control_smoke.pt `
    --dropout-checkpoint-path .\my-gpt\checkpoints\stage_15_dropout_smoke.pt
```

Every reported training-curve measurement reuses the same fixed train and
validation panels and evaluates with dropout disabled. After both runs, the
script loads their best checkpoints and compares them on a fresh reproducible
validation panel shared by both models. The default uses 500 batches and
reports the paired dropout-minus-control difference, its standard error, and
a 95% confidence interval.

### Results

The completed default XPU run trained both 211,777-parameter branches for
18,000 updates. They used the same initial state, training-batch seed 1,338,
training RNG seed 1,340, and learning-rate schedule. Their shared initial
state SHA-256 was:

```text
88dae91952fe315e838766caa8df2e8624fdcd9dbf0c0ab870cfeeca5ea4bb88
```

Both branches began with identical fixed-panel losses. The selected curve
below includes the first comparison, points within each phase, both phase
boundaries, both post-decay responses, the best-checkpoint step, and the final
step. Above step zero, the LR shown is the rate used to reach that model state;
the step-zero row shows the initial scheduled rate.

| Step | LR | Control train | Control val | Control gap | Dropout train | Dropout val | Dropout gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1e-3 | 4.3394 | 4.3465 | 0.0071 | 4.3394 | 4.3465 | 0.0071 |
| 500 | 1e-3 | 2.2155 | 2.2332 | 0.0177 | 2.2392 | 2.2513 | 0.0121 |
| 1,000 | 1e-3 | 1.9576 | 2.0240 | 0.0664 | 2.0092 | 2.0615 | 0.0523 |
| 2,500 | 1e-3 | 1.6294 | 1.7787 | 0.1493 | 1.6948 | 1.8364 | 0.1416 |
| 5,000 | 1e-3 | 1.4930 | 1.6800 | 0.1870 | 1.5489 | 1.7132 | 0.1643 |
| 7,500 | 1e-3 | 1.4414 | 1.6486 | 0.2073 | 1.4870 | 1.6620 | 0.1750 |
| 9,500 | 1e-3 | 1.4134 | 1.6257 | 0.2123 | 1.4613 | 1.6461 | 0.1848 |
| 10,000 | 1e-3 | 1.4100 | 1.6167 | 0.2068 | 1.4533 | 1.6290 | 0.1758 |
| 10,500 | 3e-4 | 1.3699 | 1.5845 | 0.2146 | 1.4271 | 1.6063 | 0.1792 |
| 13,000 | 3e-4 | 1.3512 | 1.5791 | 0.2279 | 1.4122 | 1.5996 | 0.1875 |
| 13,500 | 1e-4 | 1.3390 | 1.5718 | 0.2328 | 1.4045 | 1.5909 | 0.1864 |
| 15,000 | 1e-4 | 1.3356 | 1.5729 | 0.2374 | 1.4013 | 1.5893 | 0.1880 |
| 17,000 | 1e-4 | 1.3316 | **1.5688** | 0.2373 | 1.3973 | **1.5843** | 0.1870 |
| 18,000 | 1e-4 | 1.3301 | 1.5710 | 0.2410 | 1.3957 | 1.5847 | 0.1890 |

The final and best fixed-panel comparisons were:

| Metric | Control (`p=0.0`) | Dropout (`p=0.1`) | Dropout - control |
| --- | ---: | ---: | ---: |
| Final train loss at step 18,000 | **1.3301** | 1.3957 | +0.0656 |
| Final validation loss at step 18,000 | **1.5710** | 1.5847 | +0.0137 |
| Final train-validation gap | 0.2410 | 0.1890 | -0.0520 |
| Best checkpoint step | 17,000 | 17,000 | -- |
| Train loss at best checkpoint | **1.3316** | 1.3973 | +0.0657 |
| Best validation loss | **1.5688** | 1.5843 | +0.0155 |
| Gap at best checkpoint | 0.2373 | 0.1870 | -0.0503 |

The smaller dropout gap is not a validation win. Dropout raised both losses;
its training loss rose much more, which compressed the gap while leaving its
validation loss above the control.

The control's step-17,000 train, validation, and gap values (`1.3316`,
`1.5688`, and `0.2373`) reproduce the Stage 13 `1e-4` branch to four decimal
places. That match provides an end-to-end check of the from-initialization
control path under the intended protocol. A post-run audit also found exact
equality between the Stage 15 control and Stage 13 model tensors, AdamW state,
and training-generator state at step 17,000.

The fresh paired post-run evaluation used 500 validation batches with shared
panel seed 1,341 and dropout disabled:

| Best checkpoint | Step | Fixed val | Fresh val | SE |
| --- | ---: | ---: | ---: | ---: |
| Control | 17,000 | 1.5688 | **1.575934** | 0.002738 |
| Dropout | 17,000 | 1.5843 | 1.590406 | 0.002382 |

The paired `dropout - control` delta was `+0.014472`, with paired SE
`0.000835` and a 95% confidence interval of `[+0.012834, +0.016109]`. The
interval is wholly positive, so the fresh panel confirms the fixed-panel
ordering. This interval measures validation-batch sampling uncertainty for
these trained weights; it does not measure variation across independent
training seeds.

For this initialization and training protocol, `p=0.1` residual-path dropout
loses even when present from the first update. Stage 14's late introduction
was therefore not the sole reason `p=0.1` failed. Stage 15 contains no late
switch and shows no comparable abrupt shock: the dropout branch learned
continuously from step zero, but remained behind the control instead of
reaching a better validation optimum.

If the dropout line continues, a from-initialization `p=0.05` comparison under
the same protocol is the next motivated test. Stage 15 suggests that `p=0.1`
may be too strong for this model and data; it does not establish that `p=0.05`
will help.

```text
control checkpoint: my-gpt/checkpoints/stage_15_control_best_checkpoint.pt
dropout checkpoint: my-gpt/checkpoints/stage_15_dropout_best_checkpoint.pt
complete console record: my-gpt/checkpoints/stage_15_training.log

control checkpoint SHA-256:
4d05d242344f6020a56d93c21507e8b0bc7a73a23d51d65e673982192b180168

dropout checkpoint SHA-256:
8b3d781476cfbdbe304a54ff6883264697f16d2291ce724997ead29cc28ecfd6
```

## Stage 16: residual-dropout strength

`16_dropout_strength.py` performs the final motivated dropout-strength test:
does halving residual-path dropout from `p=0.10` to `p=0.05` recover enough
fit to beat the Stage 15 `p=0` control? It trains only the new treatment and
reuses both saved Stage 15 checkpoints as read-only references:

```text
                 same verified initialization
                            |
          +-----------------+-----------------+
          |                 |                 |
       p=0.00            p=0.05            p=0.10
       Stage 15          Stage 16           Stage 15
       reused             trained            reused
```

The default command pins the exact Stage 15 checkpoint hashes before doing
any expensive work. It regenerates the original CPU initialization from seed
1,337, requires SHA-256
`88dae91952fe315e838766caa8df2e8624fdcd9dbf0c0ab870cfeeca5ea4bb88`,
strictly loads it into the `p=0.05` model, and resets the training RNG to
1,340 immediately before updates. Its independent batch generator uses seed
1,338, the same architecture and 18,000-update LR schedule are retained, and
every curve measurement uses the same fixed panels as Stage 15.

At the Stage 15 reference step 17,000, the running treatment's batch-generator
state must exactly equal both saved Stage 15 generator states. At the end of
training it is also checked against a separately advanced 18,000-update
generator. Both Stage 15 files are hashed again after training and after fresh
evaluation, so the run fails if either supposedly read-only input changes.

Run the experiment after the two Stage 15 checkpoints exist:

```powershell
python .\my-gpt\16_dropout_strength.py --device xpu
```

The new treatment checkpoint and complete console record are:

```text
my-gpt/checkpoints/stage_16_dropout_best_checkpoint.pt
my-gpt/checkpoints/stage_16_training.log
```

Stage 16 has its own checkpoint contract because its treatment and baseline
come from different stages. The treatment checkpoint records the complete
schedule, initialization and RNG seeds, dropout probability and placement,
and hashes of both reused Stage 15 references. A mixed-stage evaluator first
validates each checkpoint under its own contract, then measures all three on
one paired validation panel with dropout disabled.

### Results

The canonical XPU run trained one 211,777-parameter `p=0.05` model for 18,000
updates. Its selected fixed-panel curve is shown alongside the already saved
Stage 15 curves:

| Step | LR | `p=0` val | `p=0.05` val | `p=0.10` val |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1e-3 | 4.3465 | 4.3465 | 4.3465 |
| 2,500 | 1e-3 | **1.7787** | 1.8109 | 1.8364 |
| 7,500 | 1e-3 | **1.6486** | 1.6531 | 1.6620 |
| 10,000 | 1e-3 | **1.6167** | 1.6237 | 1.6290 |
| 10,500 | 3e-4 | **1.5845** | 1.5991 | 1.6063 |
| 13,500 | 1e-4 | **1.5718** | 1.5832 | 1.5909 |
| 15,500 | 1e-4 | **1.5694** | 1.5779 | 1.5847 |
| 17,000 | 1e-4 | **1.5688** | 1.5787 | 1.5843 |
| 18,000 | 1e-4 | **1.5710** | 1.5799 | 1.5847 |

The best fixed-panel checkpoints and final-step diagnostics were:

| Metric | `p=0` | `p=0.05` | `p=0.10` |
| --- | ---: | ---: | ---: |
| Best checkpoint step | 17,000 | 15,500 | 17,000 |
| Train loss at best checkpoint | **1.3316** | 1.3787 | 1.3973 |
| Best fixed validation loss | **1.5688** | 1.5779 | 1.5843 |
| Gap at best checkpoint | 0.2373 | 0.1992 | 0.1870 |
| Final train loss | **1.3301** | 1.3751 | 1.3957 |
| Final validation loss | **1.5710** | 1.5799 | 1.5847 |
| Final gap | 0.2410 | 0.2048 | 0.1890 |

Halving dropout reduced the damage but did not reverse it. Relative to the
reused control, `p=0.05` was `+0.0091` worse at its best fixed checkpoint;
`p=0.10` was `+0.0155` worse. The generalization gap again moved in the
opposite direction from validation quality because dropout raised training
loss much more than it helped validation.

The post-run evaluator used 500 newly sampled validation batches with shared
seed 1,343. This seed is distinct from the fixed panel (1,339), the Stage 15
fresh panel (1,341), and generation (1,342):

| Best checkpoint | Step | Fixed val | Fresh val | SE |
| --- | ---: | ---: | ---: | ---: |
| `p=0` | 17,000 | **1.5688** | **1.572696** | 0.002754 |
| `p=0.05` | 15,500 | 1.5779 | 1.580936 | 0.002517 |
| `p=0.10` | 17,000 | 1.5843 | 1.588543 | 0.002431 |

Because every model saw the same batches, paired differences are substantially
more precise than comparing the three marginal standard errors:

| Paired fresh delta | Mean | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| `p=0.05 - p=0` | +0.008240 | 0.000750 | [+0.006771, +0.009710] |
| `p=0.10 - p=0` | +0.015847 | 0.000858 | [+0.014164, +0.017529] |
| `p=0.10 - p=0.05` | +0.007606 | 0.000648 | [+0.006336, +0.008876] |

All three intervals are wholly positive. Validation loss therefore increases
cleanly with dropout strength for these trained weights:

```text
p=0.00  <  p=0.05  <  p=0.10
 better                  worse
```

Stage 16 closes the current dropout line. `p=0.05` is not an ambiguous tie;
it is reliably worse than no residual-path dropout, while `p=0.10` is worse
again. The model should retain `p=0`, and further searches such as `0.04`,
`0.03`, or `0.02` are not justified by this dose response. As in Stage 15,
the confidence intervals quantify validation-batch sampling uncertainty for
these particular trained weights, not variation across independent training
seeds.

The next useful experiment should return to architecture or data use. Context
scaling from `T=64` to `T=128` is a natural candidate, but it needs a protocol
that separates the benefit of longer context from the doubled token targets
per batch and the quadratic attention cost.

```text
reused p=0 checkpoint: my-gpt/checkpoints/stage_15_control_best_checkpoint.pt
new p=.05 checkpoint:  my-gpt/checkpoints/stage_16_dropout_best_checkpoint.pt
reused p=.10 checkpoint: my-gpt/checkpoints/stage_15_dropout_best_checkpoint.pt
complete console record: my-gpt/checkpoints/stage_16_training.log

p=.05 checkpoint SHA-256:
94035435b62b84cc930d4b97759ca99ed98bb71b7c9d32695b3c94795814cca9

console record SHA-256:
e68ef8da30e7f2efc343f8a34ed1e3d3e50ef24a9f5322f6215586638355d424
```

## Stage 17: controlled context scaling

`17_context_length.py` tests whether extending context from 64 to 128
characters improves held-out next-character prediction enough to justify the
additional attention cost. Both branches retain the Stage 16 winner:
residual dropout is zero.

| Branch | Batch | Context | Targets/update | Parameters |
| --- | ---: | ---: | ---: | ---: |
| `T=64` control | 32 | 64 | 2,048 | 211,777 |
| `T=128` treatment | 16 | 128 | 2,048 | 215,873 |

Both use `C=64, H=4, D=16, FF=256, L=4`, with residual dropout fixed at
zero.

Both branches train from initialization for 18,000 updates under the
established schedule:

```text
[0, 10000)       lr = 1e-3
[10000, 13000)   lr = 3e-4
[13000, 18000)   lr = 1e-4
```

The experiment does not reuse the saved Stage 15 control. That run sampled 32
independent 64-character windows per update, which is a different training
stream. Stage 17 instead samples 16 starting positions. For each sampled
128-character input and its 128 next-character targets, the control receives
the two adjacent 64-character windows:

```text
long treatment:  [ s ....................................... s+127 ]
short control A: [ s ............... s+63 ]
short control B:                     [ s+64 ........... s+127 ]
```

The short windows are interleaved by sampled segment and built by reshaping
the long CPU tensor. This makes the flattened target tensors exactly equal at
every update. Only the selected branch's view is transferred to the
accelerator, so the memory comparison does not retain both representations.
Over 18,000 updates, each branch therefore trains on 36,864,000 target
characters.

### Matched initialization

Changing the positional-table size changes constructor RNG consumption, so
constructing both models with the same seed would not by itself initialize
their later layers equally. Stage 17 constructs the canonical seeded `T=64`
state, then draws rows 64 through 127 from the continuation of that seeded RNG
stream. It copies every shared parameter and positional rows 0 through 63 into
`T=128`; the newly drawn rows are the treatment's only additional learned
parameters. The correctly sized causal-mask buffers remain branch-specific.

The script records separate full-state hashes and a hash of the common
`T=64` projection, plus a hash of the additional positional rows. On one
seeded validation audit batch, it also checks before training that the two
models produce matching logits on the first 64 positions of the same long
segments. The checkpoint metadata calls this an overlap-matched
initialization; the full states are not identical because their positional
tables and causal masks have different shapes.

### Context and systems diagnostics

Every fixed-panel evaluation reports overall train and validation loss plus
first- and second-half validation loss. Best checkpoints are selected
strictly by overall validation loss. The second-half comparison is the
central context diagnostic: both models predict the same characters, while
`T=128` can attend to the preceding half that the second `T=64` window cannot
see.

The residual activation budget is unchanged:

```text
32 * 64 * 64  = 131,072
16 * 128 * 64 = 131,072
```

The attention-weight tensor shapes are checked directly:

```text
T=64:  (32, 4, 64, 64)   =   524,288 elements per block
T=128: (16, 4, 128, 128) = 1,048,576 elements per block
```

The driver benchmarks disposable initialized clones in the counterbalanced
order `T=64, T=128, T=128, T=64`. Each run measures 100 updates after 20
warmup updates by default. The four runs are adjacent in time, and none of
their updates touch the models, optimizers, or batch generators used for the
quality experiment. The timed region includes paired CPU batch construction,
transfer, forward loss, backward, and the optimizer update. The two runs for
each context are aggregated before reporting iterations/second, target
characters/second, and the maximum peak allocated and reserved accelerator
memory.

After training, the driver reloads both overall-best checkpoints on a fresh
paired validation panel (seed 1,344 by default) and reports
treatment-minus-control deltas, standard errors, and 95% confidence intervals
for overall, first-half, and second-half loss. Those intervals measure
validation-panel sampling uncertainty for the two trained models, not
variation across independent training seeds. It also prints final-step text
from each branch using the same generation seed (`seed + 5`); those samples
are not generated from the reloaded best checkpoints.

Run the full experiment with:

```powershell
python .\my-gpt\17_context_length.py --device xpu 2>&1 | Tee-Object -FilePath .\my-gpt\checkpoints\stage_17_training.log
```

A small CPU plumbing run can compress the schedule and use disposable
checkpoint paths:

```powershell
python .\my-gpt\17_context_length.py --device cpu `
    --first-decay-step 1 --second-decay-step 2 --max-iters 3 `
    --eval-interval 1 --eval-iters 1 --precise-eval-iters 2 `
    --benchmark-warmup 1 --benchmark-steps 2 --sample-length 0 `
    --control-checkpoint-path .\my-gpt\checkpoints\stage_17_t64_smoke.pt `
    --treatment-checkpoint-path .\my-gpt\checkpoints\stage_17_t128_smoke.pt
```

The default best-checkpoint outputs are:

```text
my-gpt/checkpoints/stage_17_control_best_checkpoint.pt
my-gpt/checkpoints/stage_17_context_128_best_checkpoint.pt
```

As in the preceding stages, capture the complete canonical console record
externally as `my-gpt/checkpoints/stage_17_training.log`; the driver does not
create the log itself.

### Results

The canonical XPU run completed all 18,000 updates on an Intel Arc 140T. Exact
target pairing and overlap-matched initialization both passed: the common
`T=64` projection had SHA-256
`88dae91952fe315e838766caa8df2e8624fdcd9dbf0c0ab870cfeeca5ea4bb88`,
the 64 additional positional rows had SHA-256
`ea3c74d982c0643b34a7c8951033e24949efe7c9545ae7b14794dabc36db73d1`,
and the initial first-half logits matched with a maximum difference of
`0.000e+00`.

The counterbalanced training benchmark found a modest end-to-end cost even
though the attention-score tensor doubled:

| Metric | `T=64` control | `T=128` treatment | Treatment change |
| --- | ---: | ---: | ---: |
| Parameters | 211,777 | 215,873 | +1.9% |
| Attention elements/block | 524,288 | 1,048,576 | +100.0% |
| Iterations/second | **46.542** | 44.447 | -4.5% |
| Target characters/second | **95,318.928** | 91,027.556 | -4.5% |
| Peak allocated memory | **46.003 MiB** | 54.988 MiB | +19.5% |
| Peak reserved memory | **52.000 MiB** | 62.000 MiB | +19.2% |

At this model size on this accelerator, the measurements suggest that
attention was not the dominant end-to-end cost: doubling its score elements
reduced throughput by about 4.5% and raised measured peak accelerator memory
by about 19%.

The control had lower overall validation loss at every reported fixed-panel
evaluation. Its selected checkpoint and final-step measurements were also
better:

| Branch | Best step | Best fixed val | Final train | Final val | Final gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `T=64` control | 14,500 | **1.5639** | **1.3282** | **1.5673** | 0.2391 |
| `T=128` treatment | 16,500 | 1.5820 | 1.3454 | 1.5857 | 0.2403 |

The best fixed-panel delta was `+0.0181`, and the final-step delta was
`+0.0184`, where positive `T=128 - T=64` values favor the control. The higher
training loss and nearly equal final train/validation gaps do not resemble an
overfitting penalty; they are more consistent with a harder optimization or
capacity-allocation problem, although this run does not identify the cause.

Fresh evaluation used 500 paired validation batches with seed 1,344:

| Best checkpoint | Step | Overall | First half/window | Second half/window |
| --- | ---: | ---: | ---: | ---: |
| `T=64` control | 14,500 | **1.574628** | **1.571067** | 1.578189 |
| `T=128` treatment | 16,500 | 1.593972 | 1.613175 | **1.574769** |

The paired differences separate the overall result into two opposing pieces:

| Paired delta, `T=128 - T=64` | Mean | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Overall | +0.019344 | 0.000844 | [+0.017689, +0.020999] |
| First half/window | +0.042107 | 0.001103 | [+0.039945, +0.044269] |
| Second half/window | -0.003419 | 0.001211 | [-0.005793, -0.001046] |

Because the two halves contain the same number of targets, their paired
deltas reproduce the overall difference exactly:

```text
(+0.042107 - 0.003419) / 2 = +0.019344
```

The `T=128` checkpoint was slightly better on the second-window targets, where
it could condition on the preceding 64 characters. That result is consistent
with useful predictive signal beyond a 64-character window. It is not, by
itself, a context-only causal estimate: the comparison uses separately
trained weights, and the treatment also learns positional rows 64 through
127. A same-checkpoint evaluation that blocks cross-half attention would test
the treatment's actual use of the earlier half more directly.

The much larger first-window penalty is decisive for the standard objective.
Those predictions receive no additional history, and their logits were
identical across branches before training, so the penalty developed during
optimization rather than coming from an overlap-initialization mismatch.
Learned absolute positions add another relevant difference: every control
position row serves both short windows, whereas the 128-row table distributes
the same target budget across two sets of absolute positions. This is an
inherent part of the tested longer-context model, but it means Stage 17 cannot
attribute the first-half divergence to context length alone.

Stage 17 therefore rejects `T=128` as a drop-in replacement under the tested
equal-target budget and established learning-rate schedule:

```text
retain T=64
```

The fresh overall loss was `+0.019344` worse, with its entire 95% confidence
interval above zero. The fixed-panel best and final results agree with that
ordering. As in earlier stages, these intervals measure validation-batch
sampling uncertainty for these two trained checkpoints, not variation across
independent training seeds. The generated samples are not used to override
the quantitative result.

```text
control checkpoint:
my-gpt/checkpoints/stage_17_control_best_checkpoint.pt

control checkpoint SHA-256:
e7b3823dbf13a75a7c06b87bd55f86a13b07dfa877ceaa3a478a3334471253c9

treatment checkpoint:
my-gpt/checkpoints/stage_17_context_128_best_checkpoint.pt

treatment checkpoint SHA-256:
26a0bfb8d056f7371eb5eb1eb16d29b91e4fb1552d8784aabaac39e00447c22b

canonical console transcript:
my-gpt/checkpoints/stage_17_training.log

console transcript SHA-256:
5bc50ce7bd84562c5753c6f71260afbc89c3f60172945d4bb4842fa3d5b0ab09
```

`stage_17_training.log` was reconstructed verbatim from the complete console
output returned after the run. It was not captured directly from the process
with `Tee-Object`, so its hash authenticates the persisted transcript rather
than the original terminal byte stream. The numerical record and line order
are preserved.

## Stage 18: isolated direct attention span

`18_attention_span.py` follows up the opposing Stage 17 results without
changing batch geometry or absolute-position usage. Both branches use the
same model and training configuration:

| Setting | Local control | Full treatment |
| --- | ---: | ---: |
| Batch size | 16 | 16 |
| Sequence length | 128 | 128 |
| Targets/update | 2,048 | 2,048 |
| Learned position rows | 128 | 128 |
| Parameters | 215,873 | 215,873 |
| Attention tensor/block | `(16, 4, 128, 128)` | `(16, 4, 128, 128)` |
| Direct attention span | latest 64 positions | full causal prefix |

Both retain `C=64, H=4, D=16, FF=256, L=4`, zero residual dropout, and the
18,000-update three-phase learning-rate schedule established in Stage 15.
Every learned tensor, including all 128 absolute positional embeddings, is
initialized identically. Independent batch generators start from the same
seed and consume the same sequence of `B=16,T=128` batches. The only initial
state differences are the persistent attention-mask buffers.

The local mask implements the requested rolling window:

```python
positions = torch.arange(block_size)
query_pos = positions[:, None]
key_pos = positions[None, :]

causal = key_pos <= query_pos
within_window = key_pos >= query_pos - max_context + 1
mask = causal & within_window
```

With `block_size=128` and `max_context=64`, query 100 directly attends to
keys 37 through 100. The full branch directly attends to keys 0 through 100.
Rows 0 through 63 are identical across masks; row 64 is the first to differ.
An initialization audit checks that first-half logits match to within `1e-6`,
and the test suite also verifies identical first-half parameter gradients
when loss is restricted to those positions.

### Scope of the intervention

The local setting limits each attention operation to 64 positions. It is not
a hard 64-character end-to-end receptive-field limit in a multilayer model.
Information can relay through intermediate positions across blocks. With four
blocks, the theoretical receptive-field width is capped by the sequence length
but otherwise grows as:

```text
1 + 4 * (64 - 1) = 253 positions
```

Stage 18 therefore asks whether direct attention edges longer than 63
positions improve prediction when every other tested condition is held fixed.
It does not compare literal access versus no access to all characters more
than 63 positions back. A stricter information-access experiment would use a
block-diagonal causal mask with separate 64-position segments, preventing
cross-half information from relaying at any layer.

### Evaluation and safeguards

Fixed-panel evaluations report overall, first-half, second-half, and four
equal position bands:

```text
[0, 32)   [32, 64)   [64, 96)   [96, 128)
```

Best checkpoints are selected by overall validation loss. The fresh
best-checkpoint comparison uses the same independently sampled validation
batches for both branches and computes paired `full - local` differences
before the standard error and two-sided Student-t 95% interval. At least two
paired batches are required. The second-half delta is the primary context
diagnostic. The four band intervals are exploratory, pointwise diagnostics;
they are not multiplicity-adjusted evidence for separate significance claims.
These intervals quantify validation-panel sampling uncertainty for the two
trained checkpoints, not training-seed variability.

Each branch's fresh result uses its independently selected overall-best
checkpoint, so the comparison includes the common early-stopping rule and the
checkpoints may represent different update counts. The fixed-panel final-step
losses are also reported as the matched-update comparison.

Checkpoint validation records and checks the shared learned-initialization
hash, branch mask-policy hashes, batch and RNG seeds, position bands, schedule,
data fingerprint, optimizer learning rate, and exact training-generator state.
Because local and full mask buffers have identical shapes, strict state-dict
loading alone could silently accept the wrong mask. Stage 18 explicitly checks
every saved mask against the branch policy after loading.

The optional benchmark uses disposable initialized clones in the
counterbalanced order `local, full, full, local`. Both paths still construct
dense attention scores, so it measures mask-policy overhead under identical
tensor shapes; it is not a benchmark of a sparse local-attention kernel.

Run the canonical experiment with:

```powershell
python .\my-gpt\18_attention_span.py --device xpu 2>&1 | Tee-Object -FilePath .\my-gpt\checkpoints\stage_18_training.log
```

A small CPU plumbing run can compress the schedule and use disposable
checkpoint paths:

```powershell
python .\my-gpt\18_attention_span.py --device cpu `
    --batch-size 2 --block-size 8 --local-context 4 `
    --n-embd 8 --n-head 2 --n-layer 1 `
    --first-decay-step 1 --second-decay-step 2 --max-iters 3 `
    --eval-interval 1 --eval-iters 1 --precise-eval-iters 2 `
    --benchmark-warmup 1 --benchmark-steps 2 --sample-length 0 `
    --local-checkpoint-path .\my-gpt\checkpoints\stage_18_local_smoke.pt `
    --full-checkpoint-path .\my-gpt\checkpoints\stage_18_full_smoke.pt
```

The default best-checkpoint outputs are:

```text
my-gpt/checkpoints/stage_18_local_64_best_checkpoint.pt
my-gpt/checkpoints/stage_18_full_128_best_checkpoint.pt
```

### Results

The canonical XPU run completed all 18,000 updates on an Intel Arc 140T. The
controlled protocol passed its initialization audit. Benchmark, training
history, final-step, and generated-text values below come from the supplied
console output. Best-checkpoint metadata and the fresh 500-batch evaluation
were independently reproduced from the saved checkpoints.

```text
shared learned-parameter SHA-256:
bfccda39e33b36457f947fa1a41c9f230e860b96f25531cb61e3cf965ffcf02c

local mask-policy SHA-256:
756634e0909fbbd2bb7d7fc5cebc20666e435aca6fe6b37257f1ae6cde79a4d6

full mask-policy SHA-256:
91fc1df848a40db0decca26c27935e4ae74bc4ec8a09f3bf8ee6dda243dceabd

initial first-half maximum logit difference: 0.000e+00
```

The dense local and full implementations had indistinguishable systems
requirements:

| Metric | Local-64 | Full-128 | Full change |
| --- | ---: | ---: | ---: |
| Iterations/second | 41.819 | 42.228 | +1.0% |
| Target characters/second | 85,645.851 | 86,482.053 | +1.0% |
| Peak allocated memory | 54.988 MiB | 54.988 MiB | 0 |
| Peak reserved memory | 62.000 MiB | 62.000 MiB | 0 |

The small throughput difference is within what this two-run-per-branch
benchmark can plausibly attribute to timing noise. Both policies materialize
the same dense attention tensors, so this experiment does not demonstrate a
computational benefit for local attention.

Full attention had lower fixed-panel loss at the final common update and at
its independently selected best checkpoint. It lagged through the first 1,500
updates and the ordering remained mixed through update 6,000; from update
6,500 onward, full attention had lower overall validation loss at every fixed
panel report.

| Branch | Best step | Best fixed val | Final train | Final val |
| --- | ---: | ---: | ---: | ---: |
| Local-64 | 17,000 | 1.5688 | 1.3375 | 1.5695 |
| Full-128 | 16,500 | **1.5628** | **1.3345** | **1.5649** |

The best fixed-panel difference was `-0.005977`, and the same-step final
difference was approximately `-0.0046`, where negative `full - local` values
favor full attention. The final fixed-panel band differences were `-0.0055`,
`-0.0074`, `-0.0051`, and `-0.0005` from the earliest to latest band. These
same-step measurements have no repeated-panel uncertainty interval.

Fresh evaluation reloaded the two overall-best checkpoints on 500 paired
validation batches with seed 1,344:

| Best checkpoint | Step | Overall | First half | Second half |
| --- | ---: | ---: | ---: | ---: |
| Local-64 | 17,000 | 1.579086 | 1.597815 | 1.560356 |
| Full-128 | 16,500 | **1.577779** | **1.595293** | **1.560265** |

The paired Student-t intervals separate the small overall difference into two
very different effects:

| Paired delta, `full - local` | Mean | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Overall | -0.001306 | 0.000690 | [-0.002661, +0.000049] |
| First half | -0.002522 | 0.001036 | [-0.004557, -0.000487] |
| Second half | -0.000090 | 0.001020 | [-0.002094, +0.001914] |
| Positions 0-31 | -0.002841 | 0.001385 | [-0.005562, -0.000120] |
| Positions 32-63 | -0.002204 | 0.001545 | [-0.005238, +0.000831] |
| Positions 64-95 | -0.002937 | 0.001497 | [-0.005878, +0.000005] |
| Positions 96-127 | +0.002756 | 0.001485 | [-0.000161, +0.005673] |

The second half was the predeclared context-sensitive diagnostic. It was
effectively tied: the point estimate was `-0.000090`, and its interval spans
both practically small benefits and costs. Stage 18 therefore does not show a
measurable predictive benefit from direct attention edges longer than 63
positions.

The apparent overall gap also shrank from `-0.005977` on the repeatedly used
checkpoint-selection panel to `-0.001306` on the fresh panel. That contraction
is consistent with selection-panel optimism and is another reason not to treat
the fixed best gap as the context effect.

The small overall advantage is instead concentrated in the first half, where
the branch masks are identical for every layer. Half of that first-half delta
contributes `-0.001261` to the overall mean, versus only `-0.000045` from the
second-half delta. Approximately 96.5% of the apparent overall improvement is
therefore an indirect consequence of training different shared weights. The
fresh overall interval narrowly crosses zero.

The exploratory bands do not show the hypothesized advantage increasing with
position. The final band trends toward local attention, and none of the four
pointwise band results supports a separate claim after accounting for the
four exploratory comparisons. The first-half result and differing best steps
also mean that the best-checkpoint comparison includes training dynamics and
the shared early-stopping rule, not just inference-time mask access.

This substantially narrows the Stage 17 interpretation. Its large first-half
penalty disappears and reverses when batch geometry, all 128 position rows,
initial learned tensors, and tensor shapes are matched. That is consistent
with the Stage 17 first-half loss arising from its positional and optimization
changes rather than from longer attention itself. The Stage 17 second-half
benefit does not reappear here as a benefit from direct long-range edges.

The result does not establish that information beyond 64 characters is
useless. With four local-64 blocks, information can relay across the entire
128-position sequence. The supported conclusion is narrower:

```text
full causal attention produced no detectable second-half improvement over
four-layer local-64 attention, while costing essentially the same in this
dense implementation
```

Under the predefined overall-checkpoint rule, Full-128 is the nominal Stage 18
winner: its fixed best and final losses were lower, and it carried no measured
systems penalty. It should not be recorded as a demonstrated context-length
win: the fresh overall result is borderline, the primary second-half result is
null, and the experiment contains only one paired training seed.

```text
local checkpoint:
my-gpt/checkpoints/stage_18_local_64_best_checkpoint.pt

local checkpoint SHA-256:
ae8c3e7652eaacd40242ce508d6b98fadb99bb95d699a1e67fe2ced1e319c827

full checkpoint:
my-gpt/checkpoints/stage_18_full_128_best_checkpoint.pt

full checkpoint SHA-256:
93db6fed4f96ef9ba6a362eca676635c7e4b344a9f61e29f042c9c5b8f03c6c9
```

The complete canonical console output supplied after the run is archived in
`my-gpt/checkpoints/stage_18_training.log`. This is a post-run transcript of
the supplied output, not the original live `Tee-Object` capture. The checkpoint
metadata, hashes, and fresh 500-batch evaluation above were independently
revalidated from the saved artifacts.

## Stage 19: bounded theoretical receptive field

`19_receptive_field.py` follows the null Stage 18 context diagnostic with a
local branch whose stacked receptive field no longer covers the full sequence.
Stage 18 used a 64-position window in each of four blocks, giving an uncapped
theoretical receptive-field width of `253`, larger than `T=128`. Stage 19 uses
the smallest integer window whose four-block receptive field reaches roughly
half of the sequence:

```text
W = 17
R = 1 + L(W - 1)
  = 1 + 4(17 - 1)
  = 65
```

The controlled configuration remains:

| Setting | Local control | Full treatment |
| --- | ---: | ---: |
| Batch size | 16 | 16 |
| Sequence length | 128 | 128 |
| Targets/update | 2,048 | 2,048 |
| Learned position rows | 128 | 128 |
| Parameters | 215,873 | 215,873 |
| Attention tensor/block | `(16, 4, 128, 128)` | `(16, 4, 128, 128)` |
| Per-layer attention span | latest 17 positions | full causal prefix |
| Maximum theoretical RF | 65 positions | 128 positions |

Both branches retain `C=64, H=4, D=16, FF=256, L=4`, zero residual dropout,
the same 18,000-update learning-rate schedule, identical learned
initialization, identical `B=16,T=128` training batches, and identical absolute
positional embeddings. Only the per-layer causal mask policy changes.

Before training, the program prints the local receptive-field calculation for
every block:

```text
after block 1: max receptive field = 17
after block 2: max receptive field = 33
after block 3: max receptive field = 49
after block 4: max receptive field = 65
```

For query position `q` after block `l`, the position-dependent width is:

```text
R_l(q) = min(q + 1, 1 + l(W - 1))
```

At query 127, Local-17 can theoretically depend on inputs 63 through 127;
Full-128 can depend on inputs 0 through 127. Query 64 still has its complete
65-position causal prefix available locally. Position 65 is the first query
for which the full branch has a reachable input origin that the four-block
local branch cannot reach, making zero-based positions `[65,128)` the cleanest
extra-history region.

### Safeguards and interpretation

The masks are identical only for query positions 0 through 16; row 17 is the
first to differ. Stage 19 therefore replaces Stage 18's initial first-half
equivalence check with an exact shared-prefix logit audit over positions
`[0,17)`. It retains the identical-parameter, mask-buffer, training-batch,
checkpoint-provenance, dense-shape, and fresh paired-validation safeguards.
Overall, half, and four-band losses are unchanged for comparability with Stage
18. The second-half aggregate `[64,128)` is an approximate extra-history proxy
that includes the single boundary query 64, where both branches can still
reach the complete prefix.

The value 65 is a maximum theoretical dependency width, not a guarantee that
the trained model preserves or uses information from every reachable
position. This is also not equivalent to an ordinary `T=64` model. Distant
information in Local-17 must travel through intermediate representations,
whereas Full-128 can reach its complete causal prefix directly in every block.
The intervention therefore bundles maximum reachable history, information
path length, and the training effects of a different mask policy.

Both implementations still form dense `128 x 128` attention score tensors and
mask unwanted local entries afterward. As in Stage 18, the benchmark measures
dense mask-policy overhead, not the potential speed or memory benefit of a
specialized sparse local-attention kernel.

Run the canonical experiment with:

```powershell
python .\my-gpt\19_receptive_field.py --device xpu 2>&1 | Tee-Object -FilePath .\my-gpt\checkpoints\stage_19_training.log
```

A small CPU plumbing run uses `T=8,W=3,L=2`, whose local receptive field is 5:

```powershell
python .\my-gpt\19_receptive_field.py --device cpu `
    --batch-size 2 --block-size 8 --local-window 3 `
    --n-embd 8 --n-head 2 --n-layer 2 `
    --first-decay-step 1 --second-decay-step 2 --max-iters 3 `
    --eval-interval 1 --eval-iters 1 --precise-eval-iters 2 `
    --benchmark-warmup 1 --benchmark-steps 2 --sample-length 0 `
    --local-checkpoint-path .\my-gpt\checkpoints\stage_19_local_smoke.pt `
    --full-checkpoint-path .\my-gpt\checkpoints\stage_19_full_smoke.pt
```

The default best-checkpoint outputs are:

```text
my-gpt/checkpoints/stage_19_local_17_best_checkpoint.pt
my-gpt/checkpoints/stage_19_full_128_best_checkpoint.pt
```

### Results

The canonical XPU run completed all 18,000 updates on an Intel Arc 140T. The
saved checkpoints independently passed the complete Stage 19 contract: data,
architecture, all 16 attention-mask buffers, learned initialization, seeds,
schedule, optimizer provenance, and batch-generator replay. Re-evaluating the
fixed 100-batch panels and the fresh 500 paired batches reproduced every saved
or displayed loss, standard error, and confidence interval exactly.

```text
shared learned-parameter SHA-256:
bfccda39e33b36457f947fa1a41c9f230e860b96f25531cb61e3cf965ffcf02c

local mask-policy SHA-256:
fc25dd44948a4f2e3be00e5f305b4412c62e4c3b3d05cf70ae007a3c693e20ad

full mask-policy SHA-256:
91fc1df848a40db0decca26c27935e4ae74bc4ec8a09f3bf8ee6dda243dceabd

initial shared-prefix maximum logit difference: 0.000e+00
```

The dense implementations again had the same measured memory requirements.
Local-17 was about 1.6% faster in this short counterbalanced benchmark, a
small difference that does not establish a systems advantage:

| Metric | Local-17 | Full-128 | Full change |
| --- | ---: | ---: | ---: |
| Iterations/second | 41.283 | 40.632 | -1.6% |
| Target characters/second | 84,547.187 | 83,214.077 | -1.6% |
| Peak allocated memory | 54.988 MiB | 54.988 MiB | 0 |
| Peak reserved memory | 62.000 MiB | 62.000 MiB | 0 |

Local-17 learned faster through most of training on the repeatedly evaluated
fixed panel. Full-128 first moved ahead at step 13,000 and remained lower
thereafter. It had the lower final fixed-panel validation loss and the lower
independently selected fixed-panel best loss:

| Branch | Best step | Best fixed val | Final train | Final val |
| --- | ---: | ---: | ---: | ---: |
| Local-17 | 17,000 | 1.5684 | **1.3060** | 1.5715 |
| Full-128 | 16,500 | **1.5628** | 1.3345 | **1.5649** |

The best fixed-panel difference was `-0.005562`, and the same-step final
difference was approximately `-0.0066`, where negative `full - local` values
favor Full-128. The final band differences were `+0.0016`, `-0.0121`,
`-0.0125`, and `-0.0034` from earliest to latest. These are measurements on
the reused selection panel and have no repeated-panel uncertainty interval.

Fresh evaluation reloaded the two overall-best checkpoints on 500 paired
validation batches with seed 1,344:

| Best checkpoint | Step | Overall | First half | Second half |
| --- | ---: | ---: | ---: | ---: |
| Local-17 | 17,000 | 1.577164 | 1.594120 | 1.560208 |
| Full-128 | 16,500 | 1.577779 | 1.595293 | 1.560265 |

The fresh point estimates slightly favor Local-17, but all three aggregate
intervals include zero:

| Paired delta, `full - local` | Mean | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Overall | +0.000615 | 0.000792 | [-0.000940, +0.002171] |
| First half | +0.001173 | 0.001121 | [-0.001030, +0.003376] |
| Second half | +0.000058 | 0.001089 | [-0.002081, +0.002196] |
| Positions 0-31 | +0.004495 | 0.001504 | [+0.001540, +0.007450] |
| Positions 32-63 | -0.002148 | 0.001580 | [-0.005252, +0.000955] |
| Positions 64-95 | -0.003296 | 0.001640 | [-0.006518, -0.000073] |
| Positions 96-127 | +0.003411 | 0.001562 | [+0.000342, +0.006480] |

The fixed-panel advantage therefore did not survive the fresh comparison: it
reversed from `-0.005562` to `+0.000615`. The second half is the closest
reported proxy for positions where Full-128 has additional reachable input
origins, and its difference is effectively zero. Stage 19 provides no
convincing overall or second-half advantage for Full-128 over a four-layer
Local-17 model with a maximum theoretical receptive field of 65 positions.

The exploratory bands are mixed rather than distance-monotonic. Positions
64-95 favor Full-128 pointwise, while both the earliest and latest bands favor
Local-17 pointwise. These four intervals are not multiplicity-adjusted, so
they do not support separate positional-effect claims. In particular, the
latest band does not show the increasing full-context benefit predicted by a
simple history-length account.

The Full-128 model state, optimizer state, selected step, saved losses, and
training-generator state are tensor-for-tensor identical to the Stage 18
Full-128 branch. Only the experiment and initialization-policy metadata differ,
as expected. This reproduces the full branch while isolating the new Local-17
control.

Under the predefined fixed-panel checkpoint-selection rule, Full-128 is the
nominal Stage 19 winner. It should not be recorded as a demonstrated
context-length win: the more trustworthy fresh aggregate and second-half
results are ties, the experiment uses one paired training seed, and its
confidence intervals measure validation-batch sampling uncertainty rather
than training-seed variability. The supported conclusion is narrower:

```text
full causal attention produced no detectable fresh-validation improvement
over four-layer local-17 attention with a theoretical 65-position receptive
field in this model and training run
```

```text
local checkpoint:
my-gpt/checkpoints/stage_19_local_17_best_checkpoint.pt

local checkpoint SHA-256:
7e670e63cd63f5f745febfffc0a2e2e4b235f4cc4c8d3e5f6fb630b60a587ba7

full checkpoint:
my-gpt/checkpoints/stage_19_full_128_best_checkpoint.pt

full checkpoint SHA-256:
a767130bae617ceef518d2c0020995319d52a659fa1fa97fb13c528526ec9e20

canonical console transcript:
my-gpt/checkpoints/stage_19_training.log

console transcript SHA-256:
2c4bba31081230feb2f0806cba6a6d70aa69d7d960023f71b9c7ac6fb3c46d7e
```

`stage_19_training.log` is the post-run console transcript supplied after the
canonical run, not the original live `Tee-Object` byte stream. Its hash
authenticates the persisted transcript. Checkpoint metadata and both
validation panels were independently revalidated from the saved artifacts;
benchmark timings, final-step metrics, and generated samples rely on the
supplied transcript.

## Stage 20: controlled width scaling from 64 to 128

`20_scale_width.py` closes the Stage 17-19 context-length investigation and
returns to representation capacity. The fresh Stage 19 panel found no
measurable overall benefit from full 128-character attention over Local-17's
theoretical 65-character receptive field, so both Stage 20 branches use the
smaller `B=32,T=64` geometry. The only architectural treatment is width:

| Setting | C=64 control | C=128 treatment |
| --- | ---: | ---: |
| Batch size | 32 | 32 |
| Sequence length | 64 | 64 |
| Targets/update | 2,048 | 2,048 |
| Residual width | 64 | 128 |
| Attention heads | 4 | 4 |
| Channels/head | 16 | 32 |
| FFN hidden width | 256 | 512 |
| Blocks | 4 | 4 |
| Residual dropout | 0 | 0 |
| Parameters | 211,777 | 816,705 |

For this repository's architecture, one block contains
`12C^2 + 10C` parameters. With the 65-character vocabulary and 64 learned
position rows, the treatment count is:

```text
token embedding          8,320
position embedding       8,192

per block:
Q/K/V                    49,152
attention projection    16,512
FFN                     131,712
two LayerNorms              512
                        -------
per block               197,888

four blocks             791,552
final LayerNorm             256
LM head                   8,385
                        -------
total                    816,705
```

The resulting parameter ratio is approximately `3.86x`, as expected when the
block-dominant terms scale quadratically with width.

### Controlled training and checkpoint protocol

Both branches train from step zero for 18,000 updates with fresh AdamW
optimizers and the established schedule:

```text
update indices 0-9,999:       1e-3
update indices 10,000-12,999: 3e-4
update indices 13,000-17,999: 1e-4
```

A dedicated batch generator is recreated from seed 1,338 for each branch, so
every optimizer update sees byte-identical inputs and targets. Fixed-panel
train and validation measurements recreate seed 1,339 at every report. The
program verifies that both training generators finish in the same exact
state.

The two widths cannot have identical learned initial tensors: their shapes
and correct fan-in-dependent initialization ranges differ. Stage 20 therefore
resets the same initialization seed before constructing each architecture and
records two separate state hashes. It deliberately does not copy or slice the
wide tensors into the narrow model, and checkpoint metadata explicitly marks
the states as non-identical. Consequently, a one-seed width comparison still
contains width-specific initialization variation even though its batches are
paired.

Each strict overall-validation improvement atomically writes a full best
checkpoint containing model and optimizer state, batch-generator and
global/device RNG state, the branch-specific initialization hash, exact
architecture and parameter count, schedule, data fingerprint, seeds, fixed
position bands, and experiment contract. The validator replays the expected
training generator through the selected step and strictly loads the saved
state into the requested width, preventing swapped-width checkpoints from
passing silently.

Generated samples are also produced from each branch's reloaded overall-best
checkpoint, and the console labels the selected step explicitly.

After training, the two independently selected overall-best checkpoints are
reloaded on 500 fresh paired validation batches from seed 1,344. Loss is
reported overall, by half, and in four 16-position bands. The program first
forms each per-batch `C=128 - C=64` difference, then reports its mean, paired
standard error, and two-sided Student-t 95% interval. Negative deltas favor
the wider model. Overall is the primary width result; the half and position
band intervals are exploratory and are not adjusted for multiple comparisons.
These intervals measure fresh-panel sampling uncertainty conditional on the
two selected checkpoints, not variation across training seeds. The repeatedly
used fixed panel remains the checkpoint-selection panel; the fresh panel is
the primary guard against selection-panel optimism.

### Counterbalanced hardware benchmark

Before quality training, disposable clones run in the counterbalanced order
`C64, C128, C128, C64`. Each trial uses the appropriate native initial state,
the same batch and model RNG seeds, 20 warmup updates, and 100 timed updates.
The two trials per width are aggregated by total timed updates divided by
total elapsed time. Reports include iterations/second, target
characters/second, and peak allocated and reserved MiB. The benchmark models,
optimizers, and updates are discarded before either experimental branch
trains.

Run the canonical XPU experiment with:

```powershell
python .\my-gpt\20_scale_width.py --device xpu 2>&1 | Tee-Object -FilePath .\my-gpt\checkpoints\stage_20_training.log
```

A small CPU plumbing run compresses the model and schedule while preserving
the complete paired workflow:

```powershell
python .\my-gpt\20_scale_width.py --device cpu `
    --batch-size 2 --block-size 8 `
    --control-width 8 --treatment-width 16 `
    --n-head 2 --n-layer 1 `
    --first-decay-step 1 --second-decay-step 2 --max-iters 3 `
    --eval-interval 1 --eval-iters 1 --precise-eval-iters 2 `
    --benchmark-warmup 1 --benchmark-steps 2 --sample-length 0 `
    --control-checkpoint-path .\my-gpt\checkpoints\stage_20_c8_smoke.pt `
    --treatment-checkpoint-path .\my-gpt\checkpoints\stage_20_c16_smoke.pt
```

The default best-checkpoint outputs are:

```text
my-gpt/checkpoints/stage_20_width_64_best_checkpoint.pt
my-gpt/checkpoints/stage_20_width_128_best_checkpoint.pt
```

### Results

The canonical Arc 140T run completed all 18,000 updates for both branches. The
supplied transcript reports the following counterbalanced benchmark:

| Metric | C=64 | C=128 |
| --- | ---: | ---: |
| Iterations/second | 31.373 | 26.379 |
| Target characters/second | 64,251.521 | 54,024.683 |
| Peak allocated memory | 46.003 MiB | 87.609 MiB |
| Peak reserved memory | 52.000 MiB | 100.000 MiB |

Despite having `3.856x` as many parameters, C=128 retained `84.08%` of C=64's
measured training throughput. Its peak allocated and reserved memory were
`1.904x` and `1.923x` the control values.

The repeatedly evaluated fixed panel selected different best steps:

| Branch | Best step | Best fixed val | Final train | Final val |
| --- | ---: | ---: | ---: | ---: |
| C=64 | 17,000 | 1.5688 | 1.3301 | 1.5710 |
| C=128 | 10,500 | **1.5462** | **1.0789** | 1.6230 |

These are checkpoint-selection results, not a final-step win for C=128. From
its selected step to step 18,000, the wider model's training loss fell from
`1.1695` to `1.0789` while its fixed-panel validation loss rose from `1.5462`
to `1.6230`. The final-step validation delta was `+0.051942`, favoring C=64,
whereas the independently selected best-checkpoint point estimates differed by
`-0.022660`, favoring C=128.

The fresh panel did not select new checkpoints. It reloaded the two already
selected models and evaluated them on 500 paired validation batches:

| Best checkpoint | Overall | First half | Second half |
| --- | ---: | ---: | ---: |
| C=64 at step 17,000 | 1.561340 | 1.601042 | 1.521639 |
| C=128 at step 10,500 | **1.535457** | **1.572447** | **1.498467** |

Paired `C=128 - C=64` inference gave:

| Scope | Mean delta | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Overall | -0.025883 | 0.000946 | [-0.027742, -0.024024] |
| First half | -0.028595 | 0.001178 | [-0.030910, -0.026280] |
| Second half | -0.023172 | 0.001380 | [-0.025884, -0.020460] |
| Positions 0-15 | -0.028183 | 0.001640 | [-0.031404, -0.024961] |
| Positions 16-31 | -0.029007 | 0.001793 | [-0.032530, -0.025483] |
| Positions 32-47 | -0.022989 | 0.001895 | [-0.026713, -0.019265] |
| Positions 48-63 | -0.023354 | 0.001933 | [-0.027152, -0.019556] |

The primary overall interval excludes zero. For this paired training seed and
these selected checkpoints, doubling width produced a clear fresh-validation
improvement. Every exploratory band also favors C=128, so the observed gain is
broad across the sequence rather than confined to later positions. The band
intervals are not multiplicity-adjusted and do not establish separate
position-specific effects.

The run also shows substantially stronger overfitting pressure at C=128: its
best checkpoint arrives 7,500 updates before the end, and continued training
improves training loss while degrading validation loss. This does not by
itself identify data scarcity as the cause; schedule length, regularization,
and other width-dependent optimization choices remain possible contributors.

This remains a one-training-seed result. The paired interval measures
fresh-validation batch uncertainty conditional on the two selected models. It
does not measure training-seed or width-specific initialization variability,
and the two architecture-native initial states are not identical.

```text
C=64 checkpoint:
my-gpt/checkpoints/stage_20_width_64_best_checkpoint.pt

C=64 checkpoint SHA-256:
a68fcbd716226b311bfe0651b7d4a689b9143ba331d8f78630d4fe8aae6724fb

C=128 checkpoint:
my-gpt/checkpoints/stage_20_width_128_best_checkpoint.pt

C=128 checkpoint SHA-256:
fc42fd0b58d3e197db3cb4d687e841b9db3f8c38a4208f9b9db2cf329ab1d3e9

canonical console transcript:
my-gpt/checkpoints/stage_20_training.log

console transcript SHA-256:
dc832c09980208869ab33c71dff03839d5519db1d2177252486475f6765833e0
```

`stage_20_training.log` is the post-run console transcript supplied after the
canonical run, not the original live `Tee-Object` byte stream. Its hash
authenticates the persisted transcript. Both checkpoints passed the strict
Stage 20 validator, and the complete 500-batch fresh panel was independently
replayed from them with an exact six-decimal match. Benchmark timings,
training-trajectory rows, final-step metrics, and generated samples rely on
the supplied transcript.

## Stage 21: residual dropout revisited at width 128

`21_dropout.py` tests whether the residual dropout that over-regularized the
`C=64` model becomes useful after Stage 20 exposed strong overfitting at
`C=128`. It trains only one new treatment and keeps the existing Stage 20
width-128 best checkpoint as a read-only `p=0` control:

| Setting | Reused control | New treatment |
| --- | ---: | ---: |
| Batch size | 32 | 32 |
| Sequence length | 64 | 64 |
| Residual width | 128 | 128 |
| Attention heads / channels per head | 4 / 32 | 4 / 32 |
| FFN width / blocks | 512 / 4 | 512 / 4 |
| Residual dropout | 0 | 0.05 |
| Parameters | 816,705 | 816,705 |
| Updates | 18,000 | 18,000 |

Dropout remains restricted to the two established residual paths in every
block: after the attention output projection and after the FFN output
projection, immediately before their residual additions. There is no
embedding dropout or attention-weight dropout. The eight `Dropout` modules
add no parameters or buffers, so the `p=0` and `p=0.05` models have
byte-identical state dictionaries at initialization.

The treatment recreates Stage 20's native C=128 initialization from seed
1,337, strictly loads that state into a `p=0.05` model, creates a fresh AdamW
optimizer with the same defaults (including weight decay `0.01`), and resets
the model/dropout RNG to seed 1,340 immediately before update zero. Training
batches use the same independent seed 1,338 generator. The fixed panel is
recreated from seed 1,339 at every report, and the update schedule is unchanged:

```text
update indices 0-9,999:       1e-3
update indices 10,000-12,999: 3e-4
update indices 13,000-17,999: 1e-4
```

The Stage 20 source is pinned by SHA-256 and expected best step. The program
verifies the treatment's batch-generator state against that source at step
10,500, independently replays the expected generator through all 18,000
updates, and hashes the source before and after training and evaluation.
Treatment checkpoints atomically store full model and optimizer state, RNG
state, both train and validation losses at the selected step, exact dropout
placement, initialization and source hashes, seeds, schedule, data fingerprint,
position bands, and optimizer settings. The validator checks optimizer steps
and shapes, the exact causal-mask contents, and all protocol metadata. The
selected treatment weights are reloaded and replayed on the fixed panel to tie
the stored best metrics to the saved tensors.

The source checkpoint contains Stage 20's best `p=0` weights at step 10,500,
not its final step-18,000 weights. Stage 21 can therefore make an artifact-based
best-checkpoint comparison, including a replayed train loss and generalization
gap, but it does not claim to reload or remeasure the `p=0` final state. That
branch's final curve remains transcript evidence from Stage 20.

Run the canonical treatment-only XPU experiment with:

```powershell
python .\my-gpt\21_dropout.py --device xpu 2>&1 |
    Tee-Object -FilePath .\my-gpt\checkpoints\stage_21_training.log
```

Run the focused CPU workflow and provenance tests with:

```powershell
python -m unittest discover -s .\my-gpt `
    -p "test_width_128_dropout.py" -v
```

### Results

Dropout changed the conclusion decisively. Selected points from the paired
fixed-panel trajectories are:

| Step | `p=0` train | `p=0` val | `p=0.05` train | `p=0.05` val |
| ---: | ---: | ---: | ---: | ---: |
| 8,500 | 1.2559 | 1.5683 | 1.2839 | **1.5383** |
| 10,000 | 1.2269 | 1.5559 | 1.2617 | **1.5347** |
| 10,500 | 1.1695 | 1.5462 | 1.2196 | **1.5110** |
| 13,000 | 1.1254 | 1.5835 | 1.1902 | **1.5109** |
| 13,500 | 1.1076 | 1.5800 | 1.1784 | **1.5052** |
| 15,000 | 1.0952 | 1.5980 | 1.1706 | **1.5077** |
| 18,000 | 1.0789 | 1.6230 | 1.1604 | **1.5134** |

The best-checkpoint and final treatment diagnostics were:

| Metric | Reused `p=0` | New `p=0.05` |
| --- | ---: | ---: |
| Best checkpoint step | 10,500 | 13,500 |
| Train loss at best | 1.169517 | 1.178411 |
| Best fixed validation loss | 1.546158 | **1.505233** |
| Gap at best | 0.376641 | **0.326822** |
| Final train loss | 1.0789 | 1.1604 |
| Final validation loss | 1.6230 | **1.5134** |
| Final gap | 0.5441 | **0.3529** |

The new optimum arrives 3,000 updates later and improves the fixed selection
panel by `-0.040925`. Dropout does not eliminate overfitting: after step
13,500, training loss continues down while validation eventually rises. It
does make the late deterioration far shallower. The treatment worsens only
from `1.5052` to `1.5134` by step 18,000, while the Stage 20 control worsened
from `1.5462` to `1.6230` over its post-best interval.

After checkpoint selection, both best models were reloaded and evaluated on
the same 500 fresh validation batches from Stage 20's seed 1,344:

| Best checkpoint | Overall | First half | Second half |
| --- | ---: | ---: | ---: |
| `p=0` at step 10,500 | 1.535457 | 1.572447 | 1.498467 |
| `p=0.05` at step 13,500 | **1.495556** | **1.538885** | **1.452227** |

Paired `p=0.05 - p=0` inference gave:

| Scope | Mean delta | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Overall | -0.039901 | 0.000728 | [-0.041332, -0.038470] |
| First half | -0.033562 | 0.000992 | [-0.035511, -0.031612] |
| Second half | -0.046240 | 0.001129 | [-0.048458, -0.044023] |
| Positions 0-15 | -0.030751 | 0.001346 | [-0.033396, -0.028106] |
| Positions 16-31 | -0.036373 | 0.001499 | [-0.039319, -0.033427] |
| Positions 32-47 | -0.044216 | 0.001597 | [-0.047354, -0.041078] |
| Positions 48-63 | -0.048265 | 0.001592 | [-0.051392, -0.045137] |

The primary interval is wholly below zero, and every exploratory positional
band favors dropout. The benefit grows later in the context rather than being
confined to the earliest, easiest positions. The band intervals are not
multiplicity-adjusted.

This is outcome A with an additional element of outcome C: `p=0.05` improves
the optimum substantially and delays overfitting. It also demonstrates the
capacity interaction motivating the experiment. At `C=64`, the Stage 16 fresh
paired delta for the same dose was `+0.008240`, reliably worse than no dropout;
at `C=128`, it is `-0.039901`, reliably better under this protocol. Regularizer
strength is not an intrinsic dataset setting independent of model capacity.

The result remains conditional on one initialization/training seed and one
dropout-mask stream. The paired intervals quantify fresh-validation batch
uncertainty for the selected checkpoints; they do not measure training-seed
or mask-seed uncertainty. A replicated training-seed comparison would be
needed to estimate that broader uncertainty.

The canonical artifacts are:

```text
reused p=0 checkpoint:
my-gpt/checkpoints/stage_20_width_128_best_checkpoint.pt

new p=0.05 checkpoint:
my-gpt/checkpoints/stage_21_dropout_005_best_checkpoint.pt

p=0.05 checkpoint SHA-256:
2845538e8a92443eaf959ec0c1086ef27a46960c01fe86319ffbd79aede8dca0

canonical console transcript:
my-gpt/checkpoints/stage_21_training.log

console transcript SHA-256:
afe9792676c56637bf0a13ab6d29cbe65e71525f9d288e376cfc0121f165447f
```

The checkpoint passed the current strict validator, including exact causal
masks and optimizer state. Its best fixed panel was independently replayed
from the saved tensors (`train=1.1784111536`, `val=1.5052327192`), and the full
500-batch fresh panel was independently replayed with an exact nine-decimal
match to the transcript.

## Stage 22: increase residual dropout from 0.05 to 0.10

`22_dropout_strength.py` asks whether the wider model continues to benefit
from stronger regularization or whether dropout begins interfering with
useful learning. It trains only a new `p=0.10` branch and keeps the Stage 21
best checkpoint as a read-only `p=0.05` control:

| Setting | Reused control | New treatment |
| --- | ---: | ---: |
| Batch size | 32 | 32 |
| Sequence length | 64 | 64 |
| Residual width | 128 | 128 |
| Attention heads / channels per head | 4 / 32 | 4 / 32 |
| FFN width / blocks | 512 / 4 | 512 / 4 |
| Residual dropout | 0.05 | 0.10 |
| Parameters | 816,705 | 816,705 |
| Updates | 18,000 | 18,000 |

The only intended treatment variable is the probability at the same eight
residual-path dropout sites used by Stage 21. There is still no embedding or
attention-weight dropout. Because dropout modules have no state, both models
share the exact learned initialization with SHA-256:

```text
ea5ac1a1b7fc912a1f5535e6bbabc43216b48aafbd86869dffd5372e7fc0bfad
```

The treatment starts from that recreated state with a fresh AdamW optimizer.
It uses the same training-batch seed 1,338, model/dropout RNG seed 1,340,
fixed-panel seed 1,339, fresh-validation seed 1,344, learning-rate schedule,
18,000-update budget, and 500-step evaluation cadence as Stages 20 and 21.
The run is deliberately not stopped early if its early validation is worse;
the complete curve distinguishes slower learning from harmful regularization.

The Stage 21 source is pinned to best step 13,500 and checkpoint SHA-256
`2845538e8a92443eaf959ec0c1086ef27a46960c01fe86319ffbd79aede8dca0`.
Stage 22 validates all Stage 21 provenance, checks the treatment batch-generator
state against the source at step 13,500, and hashes the source before and after
training and evaluation. Its own checkpoint records Stage 22, `p=0.05` as the
Stage 21 control, `p=0.10` as the treatment, and the shared initialization
lineage. The strict validator also checks optimizer state, exact causal masks,
loss metadata, seeds, schedule, data fingerprint, and dropout placement.

Run the canonical treatment-only XPU experiment with:

```powershell
python .\my-gpt\22_dropout_strength.py --device xpu 2>&1 |
    Tee-Object -FilePath .\my-gpt\checkpoints\stage_22_training.log
```

Run the focused CPU workflow and provenance tests with:

```powershell
python -m unittest discover -s .\my-gpt `
    -p "test_dropout_strength_128.py" -v
```

### Results

Stronger dropout improved generalization while deliberately worsening the
training fit. Selected points from the matched fixed-panel trajectories are:

| Step | `p=0.05` train | `p=0.05` val | `p=0.10` train | `p=0.10` val |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | **1.3454** | **1.5707** | 1.3637 | 1.5788 |
| 8,500 | **1.2839** | **1.5383** | 1.3045 | 1.5436 |
| 10,000 | **1.2617** | 1.5347 | 1.2847 | **1.5343** |
| 10,500 | **1.2196** | 1.5110 | 1.2508 | **1.5094** |
| 11,000 | **1.2108** | 1.5112 | 1.2430 | **1.5049** |
| 13,000 | **1.1902** | 1.5109 | 1.2255 | **1.5064** |
| 13,500 | **1.1784** | 1.5052 | 1.2161 | **1.4960** |
| 15,000 | **1.1706** | 1.5077 | 1.2099 | **1.4984** |
| 18,000 | **1.1604** | 1.5134 | 1.2016 | **1.5008** |

The best-checkpoint and final diagnostics were:

| Metric | Reused `p=0.05` | New `p=0.10` |
| --- | ---: | ---: |
| Best checkpoint step | 13,500 | 13,500 |
| Train loss at best | **1.178411** | 1.216079 |
| Best fixed validation loss | 1.505233 | **1.495999** |
| Gap at best | 0.326822 | **0.279920** |
| Final train loss | **1.1604** | 1.2016 |
| Final validation loss | 1.5134 | **1.5008** |
| Final gap | 0.3529 | **0.2992** |

The fixed selection panel improves by `-0.009234`. At the shared optimum,
`p=0.10` raises training loss by `+0.037668` but lowers validation loss, which
is the clean regularization outcome this stage was designed to test. Both
branches select step 13,500, so the hypothesis that stronger dropout might
move the optimum later is not supported here. Stronger dropout still makes
the late deterioration shallower: its validation loss rises only from
`1.4960` to `1.5008` by step 18,000, compared with `1.5052` to `1.5134` for
`p=0.05`.

After checkpoint selection, both best models were reloaded and evaluated on
the same 500 fresh validation batches from seed 1,344:

| Best checkpoint | Overall | First half | Second half |
| --- | ---: | ---: | ---: |
| `p=0.05` at step 13,500 | 1.495556 | 1.538885 | 1.452227 |
| `p=0.10` at step 13,500 | **1.487083** | **1.531501** | **1.442665** |

Paired `p=0.10 - p=0.05` inference gave:

| Scope | Mean delta | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Overall | -0.008473 | 0.000614 | [-0.009679, -0.007266] |
| First half | -0.007384 | 0.000816 | [-0.008986, -0.005781] |
| Second half | -0.009562 | 0.000902 | [-0.011333, -0.007790] |
| Positions 0-15 | -0.004531 | 0.001040 | [-0.006574, -0.002488] |
| Positions 16-31 | -0.010237 | 0.001295 | [-0.012780, -0.007693] |
| Positions 32-47 | -0.011095 | 0.001277 | [-0.013604, -0.008587] |
| Positions 48-63 | -0.008028 | 0.001236 | [-0.010455, -0.005600] |

The primary interval is wholly below zero, and every exploratory position
band favors `p=0.10`. The largest improvements occur in the two middle-late
bands rather than growing monotonically with position. As before, the band
intervals are exploratory and not multiplicity-adjusted.

This is the first outcome posed for Stage 22: the `C=128` model continues to
benefit from stronger residual regularization at `p=0.10`. The improvement is
smaller than the Stage 21 gain from `p=0` to `p=0.05`, suggesting diminishing
returns, but this experiment has not yet located the point where additional
dropout becomes harmful. The conclusion remains conditional on this one
initialization, training-batch stream, and dropout-mask RNG stream. The paired
interval measures validation-batch sampling uncertainty for these selected
checkpoints, not uncertainty across training seeds.

The canonical artifacts are:

```text
reused p=0.05 checkpoint:
my-gpt/checkpoints/stage_21_dropout_005_best_checkpoint.pt

new p=0.10 checkpoint:
my-gpt/checkpoints/stage_22_dropout_010_best_checkpoint.pt

p=0.10 checkpoint SHA-256:
dabb5b9d235460b9c3ab4b7b1a43d70c0ecd3403d48b447bf7f5d45f68823fc9

canonical console transcript:
my-gpt/checkpoints/stage_22_training.log

console transcript SHA-256:
91e1d2465885b4348f5754603d86741dae970ae0fadea985ea0edae2174996ce
```

The checkpoint passed the strict Stage 22 validator, its selected fixed-panel
losses were replayed from the saved tensors, and the fresh 500-batch panel was
run on identically sampled validation batches for both checkpoints.

One measurement detail is now explicit: the fixed-panel `train` values in
Stages 20-22 were already produced by `model.eval()` under `torch.no_grad()`.
Dropout was therefore disabled for both the fixed training and validation
panels. In particular, Stage 22's `1.216079` training-panel loss measures the
fit of the learned weights with dropout off; it is not mechanically inflated
by active dropout noise. What is not reported by those curves is the noisy
single-batch loss seen by an individual optimizer update.

## Stage 23: increase residual dropout from 0.10 to 0.15

`23_dropout_strength.py` continues the controlled regularization sweep with
the next local increment. It trains only a new `p=0.15` branch and keeps the
Stage 22 best checkpoint as a read-only `p=0.10` control:

| Setting | Reused control | New treatment |
| --- | ---: | ---: |
| Batch size | 32 | 32 |
| Sequence length | 64 | 64 |
| Residual width | 128 | 128 |
| Attention heads / channels per head | 4 / 32 | 4 / 32 |
| FFN width / blocks | 512 / 4 | 512 / 4 |
| Residual dropout | 0.10 | 0.15 |
| Parameters | 816,705 | 816,705 |
| Updates | 18,000 | 18,000 |

The Stage 22 control is pinned to best step 13,500 and SHA-256
`dabb5b9d235460b9c3ab4b7b1a43d70c0ecd3403d48b447bf7f5d45f68823fc9`.
Stage 23 validates its full provenance and hashes it before and after training,
checkpoint replay, and fresh validation. The treatment starts from the same
byte-identical initialization, uses a fresh AdamW optimizer, and preserves the
training batches, RNG seeds, fixed panels, LR schedule, evaluation cadence,
18,000-update budget, architecture, and parameter count.

A non-default Stage 22 control path must be accompanied by both
`--control-checkpoint-sha256` and `--control-checkpoint-step`; Stage 23 never
infers a custom control's expected identity from that checkpoint's own
metadata. The selected Stage 23 treatment hash is also carried in the
in-memory experiment report, so replacing that file between training and
fresh validation is rejected.

At every evaluation checkpoint, the console now labels the three comparable
diagnostics explicitly:

```text
eval-train   fixed training panel, model.eval(), dropout off
eval-val     fixed validation panel, model.eval(), dropout off
eval-gap     eval-val - eval-train
```

This is a clarification of the existing evaluator rather than an extra model
pass or a protocol change. The Stage 23 checkpoint records and strictly
validates the fixed-panel seed, eval mode, disabled gradients/dropout, split
order, and batch count, along with `identical_fixed_training_batches=True`.

Run the canonical treatment-only XPU experiment with:

```powershell
python .\my-gpt\23_dropout_strength.py --device xpu 2>&1 |
    Tee-Object -FilePath .\my-gpt\checkpoints\stage_23_training.log
```

Run the focused CPU workflow, provenance, and measurement-mode tests with:

```powershell
python -m unittest discover -s .\my-gpt `
    -p "test_dropout_strength_015.py" -v
```

The canonical run will write the selected treatment checkpoint to:

```text
my-gpt/checkpoints/stage_23_dropout_015_best_checkpoint.pt
```

The primary Stage 23 question is whether the smaller `0.05 -> 0.10` gain
continues at `p=0.15` or turns around. Fresh paired validation compares the
Stage 22 and Stage 23 best checkpoints on the same 500 validation batches;
negative `p=0.15 - p=0.10` loss favors the new treatment.

### Stage 23 result

The canonical run found the turning point. The Stage 22 `p=0.10` checkpoint
remains the champion:

| Metric | `p=0.10` | `p=0.15` | `p=.15 - p=.10` |
| --- | ---: | ---: | ---: |
| Best fixed validation | **1.4960** | 1.4986 | +0.0026 |
| Fresh validation | **1.487083** | 1.488836 | **+0.001753** |
| Eval-train at best | **1.2161** | 1.2371 | +0.0210 |
| Eval gap at best | 0.2799 | **0.2615** | -0.0184 |
| Best step | 13,500 | 15,500 | +2,000 |

The fresh paired delta was `+0.001753 +/- 0.000512 SE`, with a two-sided 95%
interval of `[+0.000746,+0.002759]`. Positive favors `p=0.10`, so the paired
panel resolves the small regression conditional on these checkpoints. The
result also demonstrates why validation loss, rather than the smallest
train/validation gap, remains the objective: `p=0.15` reduced the gap by
weakening training fit but did not improve validation.

## Stage 24: controlled depth scaling from 4 to 6 layers

`24_scale_depth.py` locks the winning residual dropout at `p=0.10` and asks a
new architectural question: what do two additional rounds of attention and
nonlinear transformation buy at fixed representation width?

| Setting | Reused control | New treatment |
| --- | ---: | ---: |
| Batch size | 32 | 32 |
| Sequence length | 64 | 64 |
| Targets/update | 2,048 | 2,048 |
| Residual width | 128 | 128 |
| Attention heads / channels per head | 4 / 32 | 4 / 32 |
| FFN hidden width | 512 | 512 |
| Transformer blocks | 4 | 6 |
| Residual dropout | 0.10 | 0.10 |
| Parameters | 816,705 | 1,212,481 |
| Updates | 18,000 | 18,000 |

Each `C=128` block contains 197,888 parameters, so the two added blocks add
395,776 parameters, a 48.46% increase over the four-layer model. Attention
geometry within a layer remains `(B,H,T,T) = (32,4,64,64)`, while the number
of causal-mask buffers rises from 16 to 24 and the number of residual dropout
sites rises from 8 to 12.

### Native initialization and read-only control

The architectures cannot have byte-identical complete initial states. Stage
24 resets initialization seed 1,337 before constructing each native model and
records a separate state hash for each branch. It does not initialize `L=6`
by copying the trained `L=4` checkpoint or by grafting new layers onto it.

The `L=4` branch does not need another 18,000-update run: it is exactly the
Stage 22 `C=128,L=4,p=0.10` champion produced with the requested batch stream,
optimizer, schedule, update count, and initialization. Stage 24 therefore
pins that checkpoint at step 13,500 and SHA-256
`dabb5b9d235460b9c3ab4b7b1a43d70c0ecd3403d48b447bf7f5d45f68823fc9`,
strictly validates its provenance, and keeps it read-only. A custom control
path requires both `--control-checkpoint-sha256` and
`--control-checkpoint-step`; expected identity is never inferred from a custom
checkpoint's own metadata.

The new `L=6` branch uses a fresh AdamW optimizer and the established schedule:

```text
update indices 0-9,999:       1e-3
update indices 10,000-12,999: 3e-4
update indices 13,000-17,999: 1e-4
```

Its explicit training-batch generator is recreated from seed 1,338, so its
inputs and targets match the Stage 22 stream exactly. Global model/dropout RNG
is reset to seed 1,340 before update zero. The RNG seed is shared, but the
models do not receive identical dropout masks: six layers consume more random
draws than four. Checkpoint selection again uses fixed eval-mode train and
validation panels from seed 1,339, with dropout and gradients disabled. This
keeps each best-step generalization gap apples-to-apples.

The Stage 24 checkpoint validator checks the treatment depth and parameter
count, 24 causal masks, 12 `p=0.10` dropout sites, optimizer state and step,
schedule, data fingerprint, explicit batch-generator replay, both native
initialization hashes, fixed-panel protocol, and pinned control identity. The
selected treatment hash is carried into fixed-panel replay and fresh
validation so replacing either checkpoint is rejected.

### Compute benchmark and paired validation

Disposable native-initialized models run before quality training in the
counterbalanced order `L4,L6,L6,L4`. Each trial uses 20 warmup and 100 timed
optimizer updates with the real dropout-active training path. The aggregated
report includes parameters, iterations/second, target characters/second, and
peak allocated and reserved accelerator memory. These models and optimizers
are discarded before the `L=6` treatment starts, and tests verify that enabling
the benchmark cannot change its quality trajectory.

After training, the independently selected Stage 22 and Stage 24 best
checkpoints run on 500 fresh paired validation batches from seed 1,344. The
primary statistic is the paired `L=6 - L=4` overall-loss delta; negative favors
the deeper model. First/second-half and four 16-position band intervals remain
exploratory and are not multiplicity-adjusted. As in prior stages, these
intervals quantify validation-batch sampling uncertainty conditional on the
selected checkpoints, not training-seed uncertainty.

Run the canonical XPU experiment with:

```powershell
python .\my-gpt\24_scale_depth.py --device xpu 2>&1 |
    Tee-Object -FilePath .\my-gpt\checkpoints\stage_24_training.log
```

The canonical comparison should use XPU, matching the numerical training path
that produced the reused Stage 22 checkpoint. CPU, CUDA, or MPS runs remain
useful for plumbing or separate replications, but device-dependent numerical
differences mean they are not the single-intervention canonical depth run.

Run the focused protocol and workflow tests with:

```powershell
python -m unittest discover -s .\my-gpt -p "test_depth_scaling.py" -v
```

The only new canonical checkpoint is:

```text
my-gpt/checkpoints/stage_24_depth_6_best_checkpoint.pt
```

### Stage 24 result

Depth produced a clear validation gain, although most of the added capacity
went into fitting the training panel:

| Metric | `L=4` | `L=6` | `L=6 - L=4` |
| --- | ---: | ---: | ---: |
| Parameters | 816,705 | 1,212,481 | +48.5% |
| Best fixed validation | 1.4960 | **1.4871** | -0.0089 |
| Paired-panel validation | 1.487083 | **1.478018** | **-0.009065** |
| Eval-train at best | 1.2161 | **1.1559** | -0.0602 |
| Gap at best | 0.2799 | 0.3312 | +0.0513 |
| Selected step | 13,500 | 15,500 | +2,000 |
| Training throughput | 64,191 chars/s | 46,424 chars/s | -27.7% |
| Peak allocated memory | 90.4 MiB | 132.1 MiB | +46.2% |

The paired `L=6 - L=4` estimate was `-0.009065 +/- 0.000656 SE`, with a
two-sided 95% interval of `[-0.010354,-0.007777]`. Every 16-position band
favored `L=6`; the band deltas ranged from `-0.007723` to `-0.011132`.
Conditional on these checkpoints, the deeper model therefore wins across the
sequence rather than only at late positions.

The wider generalization gap does not reverse that conclusion: validation
loss selects the model, while the gap diagnoses how it got there. At the
selected checkpoints, the additional layers reduced eval-train loss by
`0.0602` but fixed validation by only `0.0089`. After the L=6 optimum,
eval-train continued from `1.1559` to `1.1468` while eval-val worsened from
`1.4871` to `1.4914`, making the added overfitting pressure directly visible.
The fixed-panel values from steps 13,500 through 15,500 differ by at most
about `0.0015`, so the selected step is best read as part of a broad late
plateau rather than evidence that six layers intrinsically need exactly 2,000
more updates.

The improvement costs about 38% more time per update and roughly 46% more
allocated memory. Memory still remains small in absolute terms for this
experiment; throughput is the more immediate scaling constraint.

The selected Stage 24 checkpoint is pinned at step 15,500 with SHA-256:

```text
b44f591123bb3679557b07135dd1c44ca0392ecfa4f5b96f1ffc24f4c65a2727
```

## Stage 25: dropout strength at six layers

`25_dropout_strength.py` tests whether the stronger overfitting pressure at
`L=6` shifts the useful residual-dropout dose upward. It holds the Stage 24
architecture fixed and changes only `p=0.10 -> p=0.15`:

| Setting | Reused control | New treatment |
| --- | ---: | ---: |
| Batch size | 32 | 32 |
| Sequence length | 64 | 64 |
| Residual width | 128 | 128 |
| Attention heads / channels per head | 4 / 32 | 4 / 32 |
| FFN hidden width | 512 | 512 |
| Transformer blocks | 6 | 6 |
| Residual dropout | 0.10 | 0.15 |
| Parameters | 1,212,481 | 1,212,481 |
| Updates | 18,000 | 18,000 |

The Stage 24 control is read-only and pinned to the hash and step above. The
new treatment recreates the byte-identical L=6 initialization with seed 1,337
and state SHA-256:

```text
2a4639d1eff064c51d7cd103e11ddd32b782314e7d0f98b560b860224a937ab0
```

Because dropout modules have no learned state, changing the probability does
not change initialization or parameter count. The treatment uses a fresh
AdamW optimizer, training-batch seed 1,338, model/dropout RNG seed 1,340, the
same 18,000-update learning-rate schedule, and the same fixed panels as Stage
24. The common RNG seed does not imply identical dropout masks: different
probabilities retain different elements even when the random stream is
matched.

The Stage 25 validator checks the complete L=6 state identity, 24 causal-mask
buffers, 12 `p=0.15` dropout sites, optimizer state and step, batch-generator
replay, schedule, data fingerprint, fixed eval-mode panels, and the pinned
Stage 24 source identity. It also records that depth and parameter count are
fixed while dropout masks are not identical. Both checkpoint files are
hashed around training, replay, and paired validation so the control stays
read-only and replacement of either selected artifact is rejected.

After selection, `p=0.10` and `p=0.15` run on the same 500 paired validation
batches from seed 1,344. The primary statistic is `p=0.15 - p=0.10`; a
negative value favors stronger regularization. The interval remains
conditional on the two selected checkpoints and measures validation-batch
sampling uncertainty, not variation across training seeds or dropout streams.

Run the canonical XPU experiment with:

```powershell
python .\my-gpt\25_dropout_strength.py --device xpu 2>&1 |
    Tee-Object -FilePath .\my-gpt\checkpoints\stage_25_training.log
```

Run the focused CPU workflow and provenance tests with:

```powershell
python -m unittest discover -s .\my-gpt `
    -p "test_dropout_strength_depth_6.py" -v
```

The selected treatment checkpoint was written to:

```text
my-gpt/checkpoints/stage_25_depth_6_dropout_015_best_checkpoint.pt
```

### Stage 25 result

Increasing residual dropout to `p=0.15` did not improve the six-layer model.
The Stage 24 `p=0.10` checkpoint remains the champion:

| Metric | `p=0.10` | `p=0.15` | `p=.15 - p=.10` |
| --- | ---: | ---: | ---: |
| Best fixed validation | **1.487101** | 1.497398 | +0.010298 |
| Paired-panel validation | **1.478018** | 1.486290 | **+0.008272** |
| Eval-train at best | **1.155896** | 1.180082 | +0.024186 |
| Gap at best | 0.331204 | **0.317316** | -0.013888 |
| Selected step | 15,500 | 17,000 | +1,500 |

The paired `p=0.15 - p=0.10` estimate was `+0.008272 +/- 0.000673 SE`,
with a two-sided 95% interval of `[+0.006950,+0.009595]`. Positive favors
`p=0.10`, so the primary comparison resolves the regression conditional on
these checkpoints. The stronger dose worsened paired-panel validation by
about 0.56%.

Every exploratory 16-position band also worsened. The deltas were
`+0.009727`, `+0.007915`, `+0.005415`, and `+0.010031` from earliest to
latest, and every unadjusted interval excluded zero. The effect is therefore
not confined to one context region, although these bands do not establish
that every individual position became worse.

The smaller gap does not make `p=0.15` the better model. Stronger dropout
raised eval-train loss by `0.024186` while validation also rose; the gap
shrank because fitting weakened, not because generalization improved. The
treatment still showed mild late overfitting: from its selected step to step
18,000, eval-train improved from `1.1801` to `1.1778` while eval-val worsened
from `1.4974` to `1.5001`. The late fixed-panel optimum is broad enough that
step 17,000 should not be interpreted as proof that stronger dropout
intrinsically requires 1,500 more updates.

Both learning-rate reductions were followed by new best regions. The
treatment reached `1.5314` at step 10,000, improved to `1.5032` after the
first reduction, and did not move below `1.50` until the second reduction.
This mirrors the earlier trajectories, although this run does not provide a
no-decay counterfactual that isolates the schedule's causal effect.

The cross-stage pattern goes in the opposite direction from the motivating
hypothesis. On the same 500 validation batches, the `p=.15 - p=.10` penalty
was `+0.001753` at `L=4` and `+0.008272` at `L=6`. A post-hoc paired
difference-in-differences contrast gives:

```text
[L6(.15 - .10)] - [L4(.15 - .10)]
    = +0.006519 +/- 0.000872 SE

95% CI:
[+0.004805,+0.008234]
```

Positive means the move to `p=0.15` was more harmful at six layers under
this protocol. This interaction interval measures validation-batch sampling
uncertainty conditional on four selected checkpoints. It does not include
training-seed, dropout-stream, or checkpoint-selection uncertainty, and it
does not establish `p=0.10` as globally optimal or rule out an intermediate
probability.

Depth still helped at `p=0.15`: the paired `L=6 - L=4` fresh delta was
`-0.002546 +/- 0.000706 SE`, with a 95% interval of
`[-0.003933,-0.001159]`. That is only about 28.1% of the depth gain measured
at `p=0.10`; the stronger dose erased roughly 71.9% of that gain without
reversing it completely.

The selected `p=0.15` checkpoint is pinned at step 17,000 with SHA-256:

```text
9bab97c690d31b164e8210ae29d96e2ade1ed25bfd4e8aa1dc7a27a6e324e4a3
```

The checkpoint passed the strict validator, and its fixed panel and complete
500-batch paired panel were independently replayed without changing either
artifact. The supplied console transcript was not persisted as
`stage_25_training.log` in this workspace, so the intermediate trajectory
and rounded final-step values above are documented from that transcript
rather than an authenticated log file.

## Stage 26: controlled context scaling from 64 to 128 characters

`26_context_length.py` locks the Stage 25 verdict at the winning
`C=128,H=4,L=6,p=0.10` architecture and tests whether predictions benefit
from another 64 characters of available history:

| Setting | New control | New treatment |
| --- | ---: | ---: |
| Batch size | 32 | 16 |
| Sequence length | 64 | 128 |
| Targets/update | 2,048 | 2,048 |
| Residual width | 128 | 128 |
| Attention heads / channels per head | 4 / 32 | 4 / 32 |
| FFN hidden width | 512 | 512 |
| Transformer blocks | 6 | 6 |
| Residual dropout | 0.10 | 0.10 |
| Parameters | 1,212,481 | 1,220,673 |
| Updates | 18,000 | 18,000 |

Both branches must be trained again from step zero. The saved Stage 24 model
is the right architecture and initialization source, but it is not a valid
trained control for this experiment: it saw 32 independently sampled
64-character windows per update. Stage 26 instead samples 16 long segments
and derives both branches from each segment:

```text
T=128: 16 x [---------------- 128 ----------------]
T=64:  16 x [------ 64 ------][------ 64 ------]
        = 32 short windows
```

The reshape preserves flattened input and target order exactly. Consequently,
each optimizer update in both branches contains the same 2,048 corresponding
target characters, and both branches consume 36,864,000 targets over the
canonical run. Batch size and sequence geometry differ, but target identity,
update count, learning-rate schedule, dropout probability, initialization
seed, model/dropout RNG seed, and optimizer configuration are held fixed.
The metadata conservatively does not claim identical dropout masks across
the different tensor geometries.

### Overlap-matched initialization

The `T=64` control reproduces the exact Stage 24 six-layer initialization.
Every shared trainable parameter and positional row 0-63 is copied
byte-for-byte into the `T=128` treatment. Only positional rows 64-127 are new,
and they are drawn with the embedding layer's native normal initializer from
an isolated continuation of the control initialization stream. Constructing
the larger positional table therefore cannot shift the initialization of any
later Transformer parameter.

The canonical provenance pins are:

```text
corpus fingerprint:
fa03e609cfc0769eb1fa73fb4e81239fee0c6e9ab0608c549b0a742a25535493

T=64 complete initial state (also the Stage 24 L=6 hash):
2a4639d1eff064c51d7cd103e11ddd32b782314e7d0f98b560b860224a937ab0

shared trainable parameters:
9f4a599cf43ab3b5e38dc697767afd6f4e8eb908639fec6c5fc139c97ba1d1ee

new positional rows 64-127:
e2776f1eedcb60b39b36e2ce4bbf210f7fbbdc221178871b39645d2546158da5

T=128 complete initial state:
c27f642dfb3ba186430d1f553b14a30fe3ce90c5062e9b2fe0ff448e88b6c8e8
```

The 8,192-parameter difference is exactly `64 x 128` additional positional
embedding values, or about 0.676% of the control. An initialization audit also
requires the two models' first-half logits to agree within `1e-6` with
dropout disabled before training begins.

### Paired panel interpretation

Fixed and fresh panels report overall loss plus two diagnostics:

```text
panel positions 000-063
panel positions 064-127
```

For the treatment, these are model positions 0-63 and 64-127. For the
control, the second panel half is the second derived short window, so its
model positions reset to 0-63. The first panel half has no more than 63 prior
within-sequence characters in either branch; the second is where `T=128` can
use up to 127. A useful-context result should therefore preferentially improve
the second half, although the halves are diagnostics and overall validation
loss remains the checkpoint-selection objective.

The first half is not a pure post-training internal control. The branches are
optimized separately, may select different steps, and the control reuses
absolute positions 0-63 for both windows while the treatment learns positions
0-127. Likewise, a second-half gain is consistent with useful longer context
but remains conditional on the complete training treatment.

After selection, both best checkpoints run on the same fresh paired batches
from seed 1,344. Stage 26 forms per-batch `T=128 - T=64` differences and uses
an unbiased paired SE with a two-sided Student-t 95% interval. Negative favors
longer context. At least two batches are required, and the fresh seed may not
equal fixed-panel seed 1,339. These intervals measure validation-panel
sampling uncertainty conditional on the selected checkpoints, not variation
across training seeds or checkpoint selection.

Run the canonical XPU experiment with:

```powershell
python .\my-gpt\26_context_length.py --device xpu 2>&1 |
    Tee-Object -FilePath .\my-gpt\checkpoints\stage_26_training.log
```

Run the focused pairing, initialization, workflow, RNG-isolation, checkpoint,
and paired-statistics tests with:

```powershell
python -m unittest discover -s .\my-gpt `
    -p "test_context_length_scaling_depth_6.py" -v
```

The canonical run writes two newly trained selected checkpoints:

```text
my-gpt/checkpoints/stage_26_context_64_best_checkpoint.pt
my-gpt/checkpoints/stage_26_context_128_best_checkpoint.pt
```

### Stage 26 result: longer context wins through the second half

Stage 26 gives a decisive answer for the current `C=128,H=4,L=6,p=0.10`
model: use `T=128`. On 500 fresh paired validation batches, the selected
`T=128` checkpoint lowered overall loss by `0.014058 +/- 0.000810 SE`; its
two-sided 95% Student-t interval is `[-0.015649,-0.012468]`. The entire
interval favors the longer context.

The selected-checkpoint measurements are:

| Metric | `T=64` control | `T=128` treatment | `T=128 - T=64` |
| --- | ---: | ---: | ---: |
| Parameters | 1,212,481 | 1,220,673 | +8,192 (+0.676%) |
| Best step | 14,000 | 13,500 | -500 |
| Best fixed val | 1.491739 | **1.476912** | **-0.014827** |
| Eval-train at best step | 1.167894 | **1.153023** | -0.014871 |
| Eval gap at best step | 0.323845 | 0.323890 | +0.000045 |

The nearly identical changes in fixed validation and eval-train loss leave
the selected-checkpoint gap unchanged to four decimal places. This does not
look like a regularization effect. The longer-context branch learned a better
fit without paying an additional validation penalty relative to that fit.

Fresh evaluation used seed 1,344 and the predeclared 500 paired batches:

| Best checkpoint | Overall | First half/window | Second half/window |
| --- | ---: | ---: | ---: |
| `T=64` control | 1.500019 | **1.497060** | 1.502977 |
| `T=128` treatment | **1.485961** | 1.506290 | **1.465631** |

| Paired delta, `T=128 - T=64` | Mean | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Overall | **-0.014058** | 0.000810 | [-0.015649, -0.012468] |
| First half/window | +0.009230 | 0.000968 | [+0.007328, +0.011132] |
| Second half/window | **-0.037346** | 0.001191 | [-0.039687, -0.035006] |

The position result is more informative than the overall number alone. The
two equal-sized halves reproduce the overall effect exactly, apart from the
displayed rounding:

```text
(+0.009230 - 0.037346) / 2 = -0.014058
```

The first half became significantly worse even though the branches' initial
first-half logits were identical. The second half improved by about four times
the magnitude of that penalty, more than compensating for it. This is the
specific pattern predicted when positions 64-127 can exploit useful history
from positions 0-63. It is strong evidence that the current model is partly
context-limited at 64 characters.

It is not a context-only causal estimate. The branches are trained separately;
the treatment also halves batch size, adds positional rows 64-127, distributes
the target budget across twice as many absolute positions, and selects a
checkpoint 500 updates earlier. The small parameter increase is entirely the
new position rows, and the localization of the gain to the context-sensitive
half makes a generic parameter-count explanation unlikely, but this comparison
does not eliminate the other training-treatment differences.

### Audit against the incumbent Stage 24 champion

The Stage 26 pair answers the controlled treatment question, but its newly
trained `T=64` branch is not the incumbent model. A read-only three-way audit
therefore evaluated the Stage 24 `L=6,p=0.10,T=64` champion and both Stage 26
checkpoints on the exact same 500 long panels from seed 1,344. Each `T=64`
model saw the two reset 64-character halves in the same target order used by
the Stage 26 evaluator.

| Checkpoint on Stage 26 panels | Overall | First half/window | Second half/window |
| --- | ---: | ---: | ---: |
| Stage 24 `T=64` champion | 1.492336 | **1.488982** | 1.495690 |
| Stage 26 `T=64` control | 1.500019 | 1.497060 | 1.502977 |
| Stage 26 `T=128` treatment | **1.485961** | 1.506290 | **1.465631** |

| Paired delta against Stage 24 | Mean | Paired SE | 95% CI |
| --- | ---: | ---: | ---: |
| Stage 26 `T=64 -` Stage 24 `T=64` | +0.007683 | 0.000801 | [+0.006108, +0.009257] |
| Stage 26 `T=128 -` Stage 24 `T=64` | **-0.006375** | 0.000783 | [-0.007913, -0.004837] |

This establishes `T=128` as the new project champion, not merely the winner
against a specially retrained control. It also tightens the effect-size story.
The Stage 26 `T=64` retraining is `0.007683` worse than Stage 24, accounting
descriptively for about 54.65% of the within-stage `0.014058` difference. The
net improvement over the actual incumbent is `0.006375`, or the remaining
45.35%:

```text
+0.007683 - 0.014058 = -0.006375
```

This decomposition does not identify why the `T=64` retraining regressed; its
training stream and trajectory differ from Stage 24. It prevents that
regression from being misreported as part of the operational context gain.
Against the incumbent, `T=128` is worse by `+0.017308` on the first half but
better by `-0.030059` on the second, so the champion result remains localized
to positions with access to the preceding half.

The outcome also reverses Stage 17 at the smaller
`C=64,H=4,L=4,p=0.00` configuration:

| Fresh paired delta, `T=128 - T=64` | Stage 17 | Stage 26 |
| --- | ---: | ---: |
| Overall | +0.019344 | **-0.014058** |
| First half/window | +0.042107 | +0.009230 |
| Second half/window | -0.003419 | **-0.037346** |

The larger current model has a much smaller first-half penalty and a much
larger context-sensitive gain. Because width, depth, dropout, initialization,
and training trajectories differ between stages, this cross-stage reversal
does not identify which change enabled the gain. It does show that the Stage 17
rejection of `T=128` was conditional on that earlier model rather than a corpus-
wide verdict about 64-character history.

The supplied counterbalanced benchmark records the systems cost:

| Metric | `T=64` control | `T=128` treatment | Treatment change |
| --- | ---: | ---: | ---: |
| Attention elements/block | 524,288 | 1,048,576 | +100.0% |
| Iterations/second | **30.473** | 27.997 | -8.1% |
| Target characters/second | **62,409.217** | 57,338.106 | -8.1% |
| Peak allocated memory | **132.107 MiB** | 145.607 MiB | +10.2% |
| Peak reserved memory | **144.000 MiB** | 158.000 MiB | +9.7% |

Thus the longer window doubles each block's attention-score elements but costs
only about 8.1% measured training throughput and 13.5 MiB of peak allocated
accelerator memory in this benchmark. Given both the controlled Stage 26 win
and the direct incumbent-checkpoint win, `T=128` is the new default for this
architecture.

The selected artifacts are pinned as:

```text
Stage 24 incumbent T=64 checkpoint SHA-256:
b44f591123bb3679557b07135dd1c44ca0392ecfa4f5b96f1ffc24f4c65a2727

T=64 checkpoint SHA-256:
fbe973aee9fb82dc74278e9d77aae0b9bba7708ae134f81cb1357daaa18b9e42

T=128 checkpoint SHA-256:
3f2ba55ab544569351e4e4b5f01432c9967f5d7629598ff6483db55c17b00be6
```

Both Stage 26 checkpoints passed strict validation. An independent XPU replay
reproduced every reported fresh mean, SE, paired delta, and confidence interval
at the displayed precision, and the subsequent three-way champion audit left
all three artifact hashes unchanged. The supplied console transcript reports
the benchmark, complete training curves, final-step values, and generated
samples, but it was not persisted as
`stage_26_training.log` in this workspace. Those transcript-only observations
therefore do not have a local log hash.

## Stage 27: hard cross-half context isolation at T=128

`27_context_isolation.py` tests whether Stage 26 improved because predictions
at positions 64-127 can use information from positions 0-63. Both branches
have the Stage 26 treatment geometry:

| Setting | Blocked control | Full-context reference |
| --- | ---: | ---: |
| Batch size | 16 | 16 |
| Sequence length | 128 | 128 |
| Targets/update | 2,048 | 2,048 |
| Residual width | 128 | 128 |
| Attention heads / channels per head | 4 / 32 | 4 / 32 |
| FFN hidden width | 512 | 512 |
| Transformer blocks | 6 | 6 |
| Residual dropout | 0.10 | 0.10 |
| Parameters | 1,220,673 | 1,220,673 |

The full branch uses ordinary causal attention. The blocked branch uses a
hard block-diagonal causal mask in every head of every layer:

```text
allowed(q, k) = (k <= q) and (q // 64 == k // 64)
```

Each 128-character sequence is therefore two independent causal islands.
Queries 64-127 cannot attend to keys 0-63 directly, and the same boundary in
all six layers prevents information from relaying across it indirectly. This
is deliberately different from a rolling 64-character attention window,
which could propagate older information through intermediate representations
as layers accumulate.

The full mask has 8,256 allowed query-key edges per head. The blocked mask has
4,160 and removes exactly `64 x 64 = 4,096` cross-half edges. Tensor geometry,
parameter count, positional embeddings, targets per update, optimizer,
schedule, and dropout probability remain unchanged.

### Reused reference and matched anchor

Stage 27 does not retrain the full-context branch. It validates and reuses the
selected Stage 26 `T=128` checkpoint at step 13,500, pinned by SHA-256:

```text
3f2ba55ab544569351e4e4b5f01432c9967f5d7629598ff6483db55c17b00be6
```

Only the blocked branch is trained from step zero. Its learned parameters are
initialized byte-for-byte from the Stage 26 `T=128` initialization; the 24
attention-mask buffers are the only differing state entries. The canonical
initialization pins are:

```text
learned parameters:
c29ae7a57c7db105f572b58c17578aea86564c7c9ae8dfb203e3179e61600477

full initial state:
c27f642dfb3ba186430d1f553b14a30fe3ce90c5062e9b2fe0ff448e88b6c8e8

blocked initial state:
f2cd4762e4bc96f0086df2f2cbff0e364ddf06f66b67aac4b5a24e9e530f686d
```

The blocked run uses the same training-batch seed and model RNG seed as the
Stage 26 treatment. It saves both its independently selected overall-best
checkpoint and an unconditional step-13,500 anchor. The anchor validator
requires its batch-generator and CPU/XPU RNG states to equal the Stage 26
reference states exactly. This verifies matching stream consumption at the
comparison endpoint; it does not expand the confidence interval to cover
training-seed or backend variability.

The primary trained-model comparison is therefore:

```text
full-context Stage 26 checkpoint @ 13,500
minus
blocked Stage 27 anchor @ 13,500
```

Comparing the two independently selected best checkpoints is secondary,
because their selected steps may differ.

### Four-condition mechanistic evaluation

The fresh evaluator loads two persisted weight checkpoints and applies both
masks to each without saving either crossed condition:

| Weights | Inference mask | Role |
| --- | --- | --- |
| Full-trained | Full | Natural Stage 26 model |
| Full-trained | Blocked | Same-weight removal of cross-half access |
| Block-trained | Blocked | Natural Stage 27 control |
| Block-trained | Full | Exploratory same-weight opening of access |

Within either same-weight pair, positions 0-63 are mathematically unaffected
by the mask change, so their first-half logits and batchwise losses must be
exactly equal. First-half values need not match between independently trained
weight sets.

The canonical fresh panel uses 500 validation batches from seed 1,345 and
reports overall loss, both halves, and six contiguous position bands:

```text
000-031  032-063  064-079  080-095  096-111  112-127
```

The reported paired deltas are:

```text
natural:           full/full - blocked/blocked
full-weight mask:  full/full - full/blocked
block-weight mask: block/full - blocked/blocked
```

Negative values favor cross-half information access. The natural matched-step
delta is the primary training-treatment endpoint, with second-half loss as the
primary mechanistic diagnostic. Blocking the full-trained model measures its
same-weight reliance on earlier-half information. Opening the block-trained
model is exploratory because those weights were never optimized with the new
connections. Student-t intervals quantify validation-panel sampling only,
conditional on the persisted checkpoints. The matched-step natural overall
endpoint and its predeclared second-half diagnostic carry primary evidential
weight. Selected-best, crossed-mask, and position-band intervals are
exploratory, pointwise, and unadjusted for multiplicity.

The canonical CLI requires `Intel(R) Arc(TM) 140T GPU (16GB)`, not merely an
arbitrary XPU, so the newly trained branch uses the same accelerator model as
its reused reference:

```powershell
python .\my-gpt\27_context_isolation.py --device xpu 2>&1 |
    Tee-Object -FilePath .\my-gpt\checkpoints\stage_27_training.log
```

It writes two new artifacts and verifies that the Stage 26 reference hash is
unchanged before returning:

```text
my-gpt/checkpoints/stage_27_block_64_at_128_best_checkpoint.pt
my-gpt/checkpoints/stage_27_block_64_at_128_step_13500_checkpoint.pt
```

Run the focused mask, initialization, checkpoint, matched-stream, and
four-condition workflow tests with:

```powershell
python -m unittest discover -s .\my-gpt `
    -p "test_long_range_information.py" -v
```
