# Artifact locations

Each kind of project data has one owner. Link to that owner instead of copying
the same material into multiple documents.

| Location | Contents | Source of truth |
| --- | --- | --- |
| `README.md` | Project introduction, commands, and stage index | Navigation only |
| `NN_stage_name.py` | Experiment definition, locked protocol, and emitted output | Experiment behavior |
| `reports/stage_NN.md` | Complete, verbatim script stdout plus separate runner metadata | Human-readable run evidence |
| `docs/interpretations/stage_NN.md` | Conclusions, cross-stage comparisons, caveats, and next steps | Human interpretation |
| `checkpoints/logs/` | Original captured process output and runner metadata | Raw local run record |
| `checkpoints/*_report.json` | Full structured measurements | Machine-readable results |
| `checkpoints/*.pt` | Model, optimizer, and reproducibility state | Binary training state |
| `figures/` | Plots generated from recorded measurements | Visual presentation |
| `data/` | Input corpora | Training and evaluation inputs |
| `test_*.py` | Behavioral, protocol, and reproducibility checks | Executable verification |

The entire `checkpoints/` directory is intentionally ignored by Git. A tracked
stage report therefore preserves the script's textual output, while the local
log, JSON, and checkpoint files retain the larger or machine-oriented evidence.

## Publishing a stage report

Capture stdout independently from stderr and runner instrumentation when
possible. Then publish it with:

```text
python experiments/publish_report.py \
  --stage 30 \
  --script 30_sparse_global_shortcut.py \
  --log checkpoints/logs/30_30_sparse_global_shortcut.log \
  --output reports/stage_30.md
```

Legacy logs append instrumentation lines beginning with `__RUN_`, `__DEVICE__`,
or `__GPU_`. The publisher removes only a valid trailing instrumentation block
and records its fields as metadata. Everything before that block is preserved
as program output.

Use `--force` only when intentionally regenerating an existing report from its
authoritative log.

## Migration policy

Stage 30 is the reference layout. Other historical stages remain in the legacy
cumulative documents until they are touched or explicitly selected for
migration.

