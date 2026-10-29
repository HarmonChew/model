# Stage 61 script output

- Script: `61_token_model.py`

## Stdout

```text
Stage 61: C320 on BPE-1024 tokens | train 1,979,208 tokens | val 218,013 tokens | device NVIDIA GeForce RTX 3090

Validation bits per character (lower is better):
seed   trained on                 characters tokens final  tokens best best step
seed0  NVIDIA GeForce RTX 3090        1.9516       2.0171       1.9606     15000
seed1  NVIDIA GeForce RTX 3090        1.9289       2.0160       1.9554     15000
seed6  NVIDIA GeForce RTX 3090 Ti     1.9457       2.0167       1.9512     15000
seed7  NVIDIA GeForce RTX 3090 Ti     1.9369       2.0166       1.9534     15000

Scored characters: character model 534,272 | token model 533,990 | parameters: token model 8,130,944

tokens (final) - characters (negative favours tokens), same seed set and card:
Across 4 seed sets: mean +0.0758 bpc | sd 0.0095 | 95% seed interval [+0.0607, +0.0910] | tokens better in 0/4

tokens (best checkpoint) - characters (negative favours tokens), same seed set and card:
Across 4 seed sets: mean +0.0144 bpc | sd 0.0093 | 95% seed interval [-0.0004, +0.0292] | tokens better in 0/4

Token model validation bpc during training (mean over seed sets):
  update   5000: 1.9916
  update  10000: 1.9701
  update  15000: 1.9551
  update  20000: 1.9681
  update  25000: 1.9755
  update  30000: 1.9849
  update  35000: 1.9928
  update  40000: 2.0006
  update  45000: 2.0087
  update  50000: 2.0166

Report: my-gpt/checkpoints/stage_61_report.json
```
