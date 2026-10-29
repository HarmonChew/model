# Stage 64 script output

- Script: `64_head_ablation.py`

## Stdout

```text
Model: stage_53_c320_l6_p015_seed0_final_checkpoint.pt (7,537,378 parameters, 6 layers x 4 heads)
Real text: 64 x 256 validation characters (as Stage 62)
Copying: 64 x 100 random letters repeated once, and 64 unrepeated controls (as Stage 63)
Nothing switched off: real-text loss 1.3540, copying 0.4600

| Switched off | Real-text loss | Increase | Copying |
| --- | ---: | ---: | ---: |
| L1 H1 | 2.1228 | +0.7688 | 0.4282 |
| L1 H2 | 3.0172 | +1.6632 | 0.0489 |
| L1 H3 | 1.3597 | +0.0057 | 0.4373 |
| L1 H4 | 1.3690 | +0.0150 | 0.5152 |
| L2 H1 | 1.3688 | +0.0148 | 0.4794 |
| L2 H2 | 1.3590 | +0.0050 | 0.5139 |
| L2 H3 | 1.3608 | +0.0068 | 0.4616 |
| L2 H4 | 1.3889 | +0.0349 | 0.4233 |
| L3 H1 | 1.3599 | +0.0059 | 0.4827 |
| L3 H2 | 1.3878 | +0.0338 | 0.5093 |
| L3 H3 | 1.3639 | +0.0099 | 0.5534 |
| L3 H4 | 1.3645 | +0.0105 | 0.4290 |
| L4 H1 | 1.3650 | +0.0110 | 0.5071 |
| L4 H2 | 1.3627 | +0.0087 | 0.4534 |
| L4 H3 | 1.3621 | +0.0081 | 0.4684 |
| L4 H4 | 1.3842 | +0.0303 | 0.4801 |
| L5 H1 | 1.3584 | +0.0044 | 0.4656 |
| L5 H2 | 1.3557 | +0.0017 | 0.4970 |
| L5 H3 | 1.3586 | +0.0046 | 0.4993 |
| L5 H4 | 1.3579 | +0.0039 | 0.4479 |
| L6 H1 | 1.3545 | +0.0006 | 0.4588 |
| L6 H2 | 1.3712 | +0.0172 | 0.0820 |
| L6 H3 | 1.3547 | +0.0008 | 0.4612 |
| L6 H4 | 1.3576 | +0.0036 | 0.4591 |

| Switched off | Real-text loss | Increase | Copying |
| --- | ---: | ---: | ---: |
| all of L1 | 3.8721 | +2.5182 | 0.0190 |
| all of L2 | 1.5397 | +0.1857 | 0.4772 |
| all of L3 | 1.4351 | +0.0811 | 0.6153 |
| all of L4 | 1.4314 | +0.0774 | 0.5264 |
| all of L5 | 1.3780 | +0.0240 | 0.5197 |
| all of L6 | 1.3784 | +0.0244 | 0.0728 |

Most needed heads: L1 H2 (+1.663), L1 H1 (+0.769), L2 H4 (+0.035), L3 H2 (+0.034), L4 H4 (+0.030)
Least needed heads: L5 H4 (+0.004), L6 H4 (+0.004), L5 H2 (+0.002), L6 H3 (+0.001), L6 H1 (+0.001)
Head whose removal cuts copying most: L1 H2 (0.4600 -> 0.0489)

Wrote figures/stage_64_head_ablation.png and checkpoints/stage_64_report.json
```
