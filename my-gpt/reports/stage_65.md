# Stage 65 script output

- Script: `65_logit_lens.py`

## Stdout

```text
Model: stage_53_c320_l6_p015_seed0_final_checkpoint.pt (7,537,378 parameters, 6 layers x 4 heads)
Real text: 64 x 256 validation characters (as Stage 62)
Copying: 64 x 100 random letters repeated once, and 64 unrepeated controls (as Stage 63)
Layer-6 reading equals the model's own output exactly: True

| Read after | Real-text loss | Improvement | Top-1 accuracy |
| --- | ---: | ---: | ---: |
| embeddings | 4.1271 | - | 7.6% |
| layer 1 | 2.7390 | +1.3881 | 24.1% |
| layer 2 | 2.1738 | +0.5652 | 38.1% |
| layer 3 | 1.8118 | +0.3620 | 47.8% |
| layer 4 | 1.5547 | +0.2571 | 54.2% |
| layer 5 | 1.4077 | +0.1470 | 57.4% |
| layer 6 | 1.3540 | +0.0537 | 59.4% |

| Read after | Repeated first | Repeated second | Control first | Control second | Copying |
| --- | ---: | ---: | ---: | ---: | ---: |
| embeddings | 5.4586 | 5.4602 | 5.4524 | 5.4630 | 0.0090 |
| layer 1 | 5.6790 | 5.6963 | 5.6608 | 5.6501 | -0.0280 |
| layer 2 | 5.9157 | 5.9450 | 5.8786 | 5.9123 | 0.0044 |
| layer 3 | 6.2845 | 6.2964 | 6.2672 | 6.2885 | 0.0095 |
| layer 4 | 6.6856 | 6.7126 | 6.6561 | 6.7127 | 0.0296 |
| layer 5 | 7.1087 | 7.0255 | 7.0482 | 7.0246 | 0.0596 |
| layer 6 | 8.0921 | 7.4476 | 8.0151 | 7.8306 | 0.4600 |

Total improvement from embeddings to layer 6: 2.7731
Biggest single-layer improvement: layer 1 (1.3881, 50% of the total)
Copying per layer: 0: 0.009, 1: -0.028, 2: 0.004, 3: 0.009, 4: 0.030, 5: 0.060, 6: 0.460

Wrote figures/stage_65_logit_lens.png and checkpoints/stage_65_report.json
```
