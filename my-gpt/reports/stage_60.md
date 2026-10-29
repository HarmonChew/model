# Stage 60 script output

- Script: `60_bpe_tokenizer.py`

## Stdout

```text
Stage 60: BPE on Complete Shakespeare | train 4,809,389 chars | val 534,377 chars
Trained 3,998 merges over 98 base characters in 13.3 s

First 20 merges:
  ' t' | 'he' | ' a' | 'ou' | ' s' | 'in' | ' m' | ' w' | 're' | 'ha' | 'nd' | ' the' | ' b' | 'is' | 'er' | 'or' | 'll' | ' f' | 'on' | 'it'

Validation split (534,377 characters), round trip exact at every size:
 vocab  val tokens  chars/token  unused <10 in train  longest tokens
    98     534,377        1.000       2           10  '\n', ' ', '!'
   256     303,203        1.762       2           10  ' that', ' with', ' your'
   512     252,954        2.113       3           11  ' should', ' shall', ' would'
  1024     218,013        2.451       3           15  ' daughter', '        ', ' brother'
  2048     194,400        2.749      22           64  '\n\n\n                   ', '\n                   ', '\n                '
  4096     176,653        3.025     131          316  '\n\n\n                   ', '\n                   ', '\n                '

The first 200 validation characters at vocab 1024, tokens separated by |:
N|E|.|
|Your|self|,| sweet| lady|,| for| you| g|ave| the| f|ire|.| Sir| |Th|u|ri|o| b|orrow|s| his| wit| from| your| lady|s|hip|’|s| look|s|,| and| sp|end|s| what| he| b|orrow|s| kind|ly| in| your| comp|an|y|.|
|
|TH|UR|IO|.|
|Sir|,| if| you| sp|end| word| for| word| with

Tokenizer: my-gpt/checkpoints/stage_60_bpe_4096.json
Report: my-gpt/checkpoints/stage_60_report.json
```
