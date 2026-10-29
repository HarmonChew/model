# Stage 62 script output

- Script: `62_attention_maps.py`

## Stdout

```text
Model: stage_53_c320_l6_p015_seed0_final_checkpoint.pt (7,537,378 parameters, 6 layers x 4 heads)
Passages: 64 x 256 validation characters, scoring queries 32-255
Loss on these passages: 1.3540 (the champion's usual validation loss is about 1.35)

| Head | Best offset | Offset share | Word start | Line start | Mean distance | Label |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| L1 H1 | 2 | 0.99 | 0.18 | 0.02 | 2.1 | offset 2 |
| L1 H2 | 1 | 0.99 | 0.19 | 0.03 | 1.1 | offset 1 |
| L1 H3 | 0 | 0.09 | 0.04 | 0.02 | 9.1 | spread |
| L1 H4 | 3 | 0.33 | 0.14 | 0.04 | 4.1 | offset 3 |
| L2 H1 | 2 | 0.33 | 0.08 | 0.02 | 3.5 | offset 2 |
| L2 H2 | 2 | 0.25 | 0.06 | 0.02 | 3.9 | spread |
| L2 H3 | 1 | 0.35 | 0.05 | 0.02 | 3.4 | offset 1 |
| L2 H4 | 1 | 0.66 | 0.19 | 0.03 | 1.9 | offset 1 |
| L3 H1 | 2 | 0.20 | 0.10 | 0.02 | 5.1 | spread |
| L3 H2 | 2 | 0.05 | 0.06 | 0.28 | 11.3 | spread |
| L3 H3 | 1 | 0.35 | 0.11 | 0.03 | 2.9 | offset 1 |
| L3 H4 | 3 | 0.13 | 0.22 | 0.02 | 5.4 | spread |
| L4 H1 | 1 | 0.28 | 0.14 | 0.04 | 4.4 | spread |
| L4 H2 | 3 | 0.10 | 0.07 | 0.04 | 7.2 | spread |
| L4 H3 | 2 | 0.11 | 0.20 | 0.04 | 6.7 | spread |
| L4 H4 | 2 | 0.05 | 0.09 | 0.09 | 10.9 | spread |
| L5 H1 | 1 | 0.01 | 0.03 | 0.04 | 58.7 | spread |
| L5 H2 | 2 | 0.01 | 0.02 | 0.04 | 61.1 | spread |
| L5 H3 | 0 | 0.02 | 0.02 | 0.01 | 55.1 | spread |
| L5 H4 | 0 | 0.02 | 0.01 | 0.01 | 59.3 | spread |
| L6 H1 | 1 | 0.02 | 0.02 | 0.01 | 57.8 | spread |
| L6 H2 | 0 | 0.03 | 0.03 | 0.02 | 63.0 | spread |
| L6 H3 | 2 | 0.02 | 0.08 | 0.02 | 49.3 | spread |
| L6 H4 | 0 | 0.02 | 0.02 | 0.03 | 52.6 | spread |

Heads per label: offset 2 2, offset 1 4, spread 17, offset 3 1
Clearest offset 1 head: L1 H2 (0.99)
Clearest word start head: L3 H4 (0.22)
Clearest line start head: L3 H2 (0.28)

Example passage (last 64 characters):
'ou, why is it?\n\nJULIA.\nMarry, mine host, because I cannot be mer'

Wrote figures/stage_62_attention_grid.png, figures/stage_62_attention_closeup.png and checkpoints/stage_62_report.json
```
