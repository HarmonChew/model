# Experiment results

A short record of the experiments this change verified. Each row says what was
run, on which device, and what came back. This is a verification record for the
benchmark-timing fix and the test-fixture work, not a new set of published
measurements: the canonical runs, their checkpoints, and their interpretations
are unchanged.

The environment is CPU-only (PyTorch 2.14.0, Python 3.14), so every run below is
a reduced-size smoke run. Full-size runs require the lineage accelerator and the
ignored checkpoints, which are not present here.

| Experiment | Command shape | Result |
| --- | --- | --- |
| Stage 1 checkpoint | `train.py --device cpu --max-iters 10 --eval-iters 2 --eval-interval 5 --sample-length 20` | Exit 0; initial 4.6310/4.6894 to final 3.5070/3.5449; sampled text produced. |
| Stage 9 context | `09_context_length.py --device cpu --max-iters 10 --eval-iters 2 --eval-interval 5 --sample-length 20 --benchmark-warmup 2 --benchmark-steps 3` | Exit 0; initial 4.3943/4.3974 to final 3.8595/3.8610; benchmark reported. |
| Stage 10 width | `10_scale_width.py --device cpu --max-iters 10 --eval-iters 2 --eval-interval 5 --sample-length 20 --benchmark-warmup 2 --benchmark-steps 4` | Exit 0; final 3.3996/3.4521; 211,777 parameters; benchmark reported. |
| Stages 43-49 protocol audit | `NN_*.py --validate-only --device cpu` | Exit 0 for every stage; each audits corpus, architecture, initialization, and its stored peers. |
| Stage 50 depth | `50_depth_revisit.py --validate-only --device cpu` | Exits non-zero by design: the preflight requires the ignored Stage 49 and C320 endpoints. Its ordinary test coverage now runs without them. |

## Benchmark timing check

The point of the `train_model` change is that evaluation cadence cannot move
the reported training throughput. Running the Stage 9 script with an evaluation
every update versus almost never, inside the same benchmark window, gives the
same measured seconds (about 0.09 s at this tiny size, where evaluation is cheap
relative to process startup). The deterministic regression test is the precise
proof: two 100-second evaluations inside the window leave the reported seconds
at exactly 3.0.
