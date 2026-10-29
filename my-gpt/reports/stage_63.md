# Stage 63 script output

- Script: `63_induction_heads.py`

## Stdout

```text
Model: stage_53_c320_l6_p015_seed0_final_checkpoint.pt (7,537,378 parameters, 6 layers x 4 heads)
Sequences: 64 x 100 random letters, repeated once (200 characters)
Controls: 64 x 200 random letters, no repeat
Guessing baseline ln(26): 3.2581

| Sequences | First half (positions 1-99) | Second half (positions 101-199) | Drop |
| --- | ---: | ---: | ---: |
| Repeated | 8.0921 | 7.4476 | 0.6446 |
| Control, no repeat | 8.0151 | 7.8306 | 0.1845 |

Copying (repeated drop minus control drop): 0.4600

| Head | Induction | Duplicate | Label |
| --- | ---: | ---: | --- |
| L1 H1 | 0.00 | 0.00 | - |
| L1 H2 | 0.00 | 0.00 | - |
| L1 H3 | 0.00 | 0.00 | - |
| L1 H4 | 0.00 | 0.00 | - |
| L2 H1 | 0.00 | 0.00 | - |
| L2 H2 | 0.00 | 0.00 | - |
| L2 H3 | 0.00 | 0.00 | - |
| L2 H4 | 0.00 | 0.00 | - |
| L3 H1 | 0.00 | 0.00 | - |
| L3 H2 | 0.00 | 0.00 | - |
| L3 H3 | 0.00 | 0.00 | - |
| L3 H4 | 0.00 | 0.00 | - |
| L4 H1 | 0.00 | 0.00 | - |
| L4 H2 | 0.00 | 0.00 | - |
| L4 H3 | 0.00 | 0.00 | - |
| L4 H4 | 0.00 | 0.00 | - |
| L5 H1 | 0.00 | 0.00 | - |
| L5 H2 | 0.01 | 0.01 | - |
| L5 H3 | 0.01 | 0.01 | - |
| L5 H4 | 0.02 | 0.01 | - |
| L6 H1 | 0.01 | 0.01 | - |
| L6 H2 | 0.13 | 0.01 | - |
| L6 H3 | 0.00 | 0.00 | - |
| L6 H4 | 0.01 | 0.00 | - |

Strongest induction head: L6 H2 (0.13)
Induction heads: 0, duplicate heads: 0

Wrote figures/stage_63_loss_by_position.png, figures/stage_63_induction_head.png and checkpoints/stage_63_report.json
```
