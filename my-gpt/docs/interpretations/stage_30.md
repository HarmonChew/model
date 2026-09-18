# Stage 30 interpretation: one global routing layer

## Question

Stage 30 tested whether one global attention layer in the middle of an otherwise
local six-layer model could close the remaining gap between the Stage 29 W=23
model and Full-128 attention. The hybrid used local W=23 attention in layers 1,
2, 4, 5, and 6, with Full-128 attention only in layer 3.

## Finding

The layer-3 global exchange did not close the Full-128 gap. On the fresh matched
step-13,500 panel, Full-128 reached validation loss 1.480932, W=23 reached
1.487418, and the hybrid reached 1.493886. The primary Full-minus-Hybrid delta
was -0.012955 with a 95% interval of [-0.014497, -0.011412]. The
Hybrid-minus-W23 delta was +0.006468 with a 95% interval of
[+0.004645, +0.008291]. Lower loss is better, so both comparisons disfavored
the hybrid.

The selected-best comparison had the same ordering: Full-128, W=23, then the
hybrid. The topology-divergent positions `[23,128)` also agreed with the overall
result.

## Interpretation boundary

This run shows that a single global layer at depth 3 was worse than both the
fully global reference and the all-local W=23 model under this protocol. It
does not establish that global mixing itself is harmful or that the local
layers after the global layer caused the regression. The result is conditional
on one initialization, training stream, corpus split, device, and backend.

## Next experiment

Test placement directly by moving the single global layer to the final block.
That removes subsequent local processing as a possible explanation and isolates
whether a late global readout behaves differently from the layer-3 exchange.

The complete observed output is in
[`reports/stage_30.md`](../../reports/stage_30.md). The original local run record
is `checkpoints/logs/30_30_sparse_global_shortcut.log`.

