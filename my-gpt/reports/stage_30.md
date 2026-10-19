# Stage 30 script output

- Script: `30_sparse_global_shortcut.py`
- Run time (seconds): `167.32`
- Device: `cuda`
- Peak GPU utilization: `97%`
- Peak GPU memory: `852 MiB`
- Exit code: `0`
- Run status: `ok`

> Publication note: machine-specific absolute paths in the stdout block were replaced with repository-relative paths. Numeric results, hashes, ordering, and all other text are unchanged.

## Stdout

```text
Stage 30: sparse global shortcut
Device: cuda
Accelerator: NVIDIA GeForce RTX 3090
Characters: 1,115,394
Vocabulary size: 65
Uniform-loss baseline: 4.1744
All: B=16, T=128, C=128, H=4, D=32, FF=512, L=6, dropout=0.1
Hybrid layer contexts: (23, 23, None, 23, 23, 23)
Schedule by zero-based update: [0, 10,000) lr=0.001; [10,000, 13,000) lr=0.0003; [13,000, 18,000) lr=0.0001
Seeds: initialization=1337, batches=1338, fixed=1339, training RNG=1340, held-out=1347

Controlled sparse-global-shortcut protocol
  Stage 26 Full and both Stage 29 W23 artifacts are pinned read-only.
  Only the one-global-layer hybrid is trained from step zero.
  layer contexts:              (23, 23, None, 23, 23, 23)
  learned parameter SHA-256:  c22ae640e5d1e70a0b789a30bb5b9f9be833dfc090147151a8a303409e41c53e
  hybrid initial SHA-256:     f96825b691f3b77542222838f2e0fb308b0ce39318e0f51abfbaa2fc014304d5
  mask-buffer differences:    20
  RF widths after layers:     (23, 45, 128, 128, 128, 128)
  Initial [0,23) max logit difference: 0.000e+00

Counterbalanced disposable dense training benchmark

W23 with one Full layer at depth 3 dense training benchmark
  seconds:               1.488
  iterations/sec:        134.411
  target characters/sec: 275273.187
  peak allocated MiB:    163.858
  peak reserved MiB:     178.000

full-128 Stage 26 reference dense training benchmark
  seconds:               1.484
  iterations/sec:        134.736
  target characters/sec: 275939.448
  peak allocated MiB:    163.858
  peak reserved MiB:     178.000

W23 with one Full layer at depth 3 branch (B=16, T=128, contexts=(23, 23, None, 23, 23, 23), dropout=0.1)
  branch initialization: f96825b691f3b77542222838f2e0fb308b0ce39318e0f51abfbaa2fc014304d5
  learned parameters:    c22ae640e5d1e70a0b789a30bb5b9f9be833dfc090147151a8a303409e41c53e
  training-batch seed:   1338
  training RNG seed:     1340
  matched anchor step:   13500
  step     0 | lr 0.001 | train 4.3391 | val 4.3399 | val first 4.3265 | val second 4.3534 | val 000-022 4.3061 | val 023-044 4.3609 | val 045-066 4.3154 | val 067-088 4.3704 | val 089-110 4.3652 | val 111-127 4.3181 | gap 0.0008 | best 4.3399 @ 0  * best
  step   500 | lr 0.001 | train 1.9953 | val 2.0623 | val first 2.0746 | val second 2.0499 | val 000-022 2.0888 | val 023-044 2.0680 | val 045-066 2.0628 | val 067-088 2.0520 | val 089-110 2.0462 | val 111-127 2.0525 | gap 0.0670 | best 2.0623 @ 500  * best
  step  1000 | lr 0.001 | train 1.6956 | val 1.8384 | val first 1.8579 | val second 1.8188 | val 000-022 1.8969 | val 023-044 1.8344 | val 045-066 1.8371 | val 067-088 1.8139 | val 089-110 1.8166 | val 111-127 1.8256 | gap 0.1427 | best 1.8384 @ 1000  * best
  step  1500 | lr 0.001 | train 1.5608 | val 1.7266 | val first 1.7489 | val second 1.7043 | val 000-022 1.7959 | val 023-044 1.7244 | val 045-066 1.7173 | val 067-088 1.6950 | val 089-110 1.7029 | val 111-127 1.7191 | gap 0.1657 | best 1.7266 @ 1500  * best
  step  2000 | lr 0.001 | train 1.4868 | val 1.6610 | val first 1.6841 | val second 1.6379 | val 000-022 1.7364 | val 023-044 1.6555 | val 045-066 1.6524 | val 067-088 1.6296 | val 089-110 1.6367 | val 111-127 1.6494 | gap 0.1742 | best 1.6610 @ 2000  * best
  step  2500 | lr 0.001 | train 1.4462 | val 1.6404 | val first 1.6649 | val second 1.6159 | val 000-022 1.7259 | val 023-044 1.6340 | val 045-066 1.6252 | val 067-088 1.6113 | val 089-110 1.6136 | val 111-127 1.6249 | gap 0.1941 | best 1.6404 @ 2500  * best
  step  3000 | lr 0.001 | train 1.4045 | val 1.5973 | val first 1.6228 | val second 1.5719 | val 000-022 1.6838 | val 023-044 1.5884 | val 045-066 1.5862 | val 067-088 1.5698 | val 089-110 1.5645 | val 111-127 1.5844 | gap 0.1929 | best 1.5973 @ 3000  * best
  step  3500 | lr 0.001 | train 1.3714 | val 1.5754 | val first 1.6025 | val second 1.5483 | val 000-022 1.6655 | val 023-044 1.5716 | val 045-066 1.5597 | val 067-088 1.5438 | val 089-110 1.5434 | val 111-127 1.5610 | gap 0.2040 | best 1.5754 @ 3500  * best
  step  4000 | lr 0.001 | train 1.3539 | val 1.5679 | val first 1.5937 | val second 1.5420 | val 000-022 1.6600 | val 023-044 1.5560 | val 045-066 1.5552 | val 067-088 1.5366 | val 089-110 1.5352 | val 111-127 1.5577 | gap 0.2139 | best 1.5679 @ 4000  * best
  step  4500 | lr 0.001 | train 1.3381 | val 1.5633 | val first 1.5884 | val second 1.5381 | val 000-022 1.6524 | val 023-044 1.5544 | val 045-066 1.5480 | val 067-088 1.5383 | val 089-110 1.5291 | val 111-127 1.5504 | gap 0.2251 | best 1.5633 @ 4500  * best
  step  5000 | lr 0.001 | train 1.3157 | val 1.5463 | val first 1.5746 | val second 1.5179 | val 000-022 1.6436 | val 023-044 1.5385 | val 045-066 1.5307 | val 067-088 1.5142 | val 089-110 1.5106 | val 111-127 1.5323 | gap 0.2306 | best 1.5463 @ 5000  * best
  step  5500 | lr 0.001 | train 1.2991 | val 1.5397 | val first 1.5666 | val second 1.5129 | val 000-022 1.6382 | val 023-044 1.5305 | val 045-066 1.5192 | val 067-088 1.5106 | val 089-110 1.5049 | val 111-127 1.5278 | gap 0.2406 | best 1.5397 @ 5500  * best
  step  6000 | lr 0.001 | train 1.2858 | val 1.5380 | val first 1.5661 | val second 1.5099 | val 000-022 1.6379 | val 023-044 1.5293 | val 045-066 1.5194 | val 067-088 1.5086 | val 089-110 1.5014 | val 111-127 1.5237 | gap 0.2522 | best 1.5380 @ 6000  * best
  step  6500 | lr 0.001 | train 1.2719 | val 1.5320 | val first 1.5589 | val second 1.5052 | val 000-022 1.6313 | val 023-044 1.5211 | val 045-066 1.5129 | val 067-088 1.5028 | val 089-110 1.4965 | val 111-127 1.5205 | gap 0.2601 | best 1.5320 @ 6500  * best
  step  7000 | lr 0.001 | train 1.2614 | val 1.5246 | val first 1.5524 | val second 1.4967 | val 000-022 1.6252 | val 023-044 1.5132 | val 045-066 1.5099 | val 067-088 1.4930 | val 089-110 1.4899 | val 111-127 1.5079 | gap 0.2632 | best 1.5246 @ 7000  * best
  step  7500 | lr 0.001 | train 1.2520 | val 1.5234 | val first 1.5518 | val second 1.4950 | val 000-022 1.6236 | val 023-044 1.5117 | val 045-066 1.5086 | val 067-088 1.4951 | val 089-110 1.4881 | val 111-127 1.5043 | gap 0.2714 | best 1.5234 @ 7500  * best
  step  8000 | lr 0.001 | train 1.2442 | val 1.5256 | val first 1.5529 | val second 1.4982 | val 000-022 1.6261 | val 023-044 1.5152 | val 045-066 1.5069 | val 067-088 1.5011 | val 089-110 1.4889 | val 111-127 1.5063 | gap 0.2814 | best 1.5234 @ 7500
  step  8500 | lr 0.001 | train 1.2359 | val 1.5164 | val first 1.5466 | val second 1.4861 | val 000-022 1.6189 | val 023-044 1.5090 | val 045-066 1.5018 | val 067-088 1.4848 | val 089-110 1.4784 | val 111-127 1.4961 | gap 0.2805 | best 1.5164 @ 8500  * best
  step  9000 | lr 0.001 | train 1.2275 | val 1.5183 | val first 1.5477 | val second 1.4890 | val 000-022 1.6202 | val 023-044 1.5073 | val 045-066 1.5048 | val 067-088 1.4876 | val 089-110 1.4802 | val 111-127 1.5015 | gap 0.2908 | best 1.5164 @ 8500
  step  9500 | lr 0.001 | train 1.2202 | val 1.5163 | val first 1.5439 | val second 1.4887 | val 000-022 1.6156 | val 023-044 1.5075 | val 045-066 1.4990 | val 067-088 1.4906 | val 089-110 1.4760 | val 111-127 1.5011 | gap 0.2961 | best 1.5163 @ 9500  * best
  step 10000 | lr 0.001 | train 1.2112 | val 1.5126 | val first 1.5409 | val second 1.4843 | val 000-022 1.6152 | val 023-044 1.5023 | val 045-066 1.4949 | val 067-088 1.4865 | val 089-110 1.4738 | val 111-127 1.4941 | gap 0.3014 | best 1.5126 @ 10000  * best
  step 10500 | lr 0.0003 | train 1.1705 | val 1.4883 | val first 1.5166 | val second 1.4599 | val 000-022 1.5904 | val 023-044 1.4785 | val 045-066 1.4703 | val 067-088 1.4641 | val 089-110 1.4487 | val 111-127 1.4683 | gap 0.3178 | best 1.4883 @ 10500  * best
  step 11000 | lr 0.0003 | train 1.1587 | val 1.4928 | val first 1.5221 | val second 1.4634 | val 000-022 1.5938 | val 023-044 1.4852 | val 045-066 1.4773 | val 067-088 1.4664 | val 089-110 1.4517 | val 111-127 1.4732 | gap 0.3340 | best 1.4883 @ 10500
  step 11500 | lr 0.0003 | train 1.1527 | val 1.4869 | val first 1.5158 | val second 1.4580 | val 000-022 1.5888 | val 023-044 1.4774 | val 045-066 1.4719 | val 067-088 1.4609 | val 089-110 1.4462 | val 111-127 1.4672 | gap 0.3342 | best 1.4869 @ 11500  * best
  step 12000 | lr 0.0003 | train 1.1464 | val 1.4950 | val first 1.5223 | val second 1.4678 | val 000-022 1.5949 | val 023-044 1.4848 | val 045-066 1.4773 | val 067-088 1.4683 | val 089-110 1.4570 | val 111-127 1.4800 | gap 0.3486 | best 1.4869 @ 11500
  step 12500 | lr 0.0003 | train 1.1401 | val 1.4905 | val first 1.5179 | val second 1.4632 | val 000-022 1.5910 | val 023-044 1.4785 | val 045-066 1.4740 | val 067-088 1.4654 | val 089-110 1.4511 | val 111-127 1.4751 | gap 0.3504 | best 1.4869 @ 11500
  step 13000 | lr 0.0003 | train 1.1340 | val 1.4929 | val first 1.5199 | val second 1.4659 | val 000-022 1.5916 | val 023-044 1.4832 | val 045-066 1.4748 | val 067-088 1.4685 | val 089-110 1.4530 | val 111-127 1.4781 | gap 0.3588 | best 1.4869 @ 11500
  step 13500 | lr 0.0001 | train 1.1227 | val 1.4903 | val first 1.5172 | val second 1.4634 | val 000-022 1.5883 | val 023-044 1.4802 | val 045-066 1.4737 | val 067-088 1.4676 | val 089-110 1.4496 | val 111-127 1.4742 | gap 0.3676 | best 1.4869 @ 11500
  step 14000 | lr 0.0001 | train 1.1184 | val 1.4904 | val first 1.5175 | val second 1.4632 | val 000-022 1.5888 | val 023-044 1.4820 | val 045-066 1.4723 | val 067-088 1.4661 | val 089-110 1.4498 | val 111-127 1.4752 | gap 0.3720 | best 1.4869 @ 11500
  step 14500 | lr 0.0001 | train 1.1155 | val 1.4900 | val first 1.5177 | val second 1.4624 | val 000-022 1.5883 | val 023-044 1.4826 | val 045-066 1.4724 | val 067-088 1.4653 | val 089-110 1.4498 | val 111-127 1.4735 | gap 0.3745 | best 1.4869 @ 11500
  step 15000 | lr 0.0001 | train 1.1123 | val 1.4903 | val first 1.5175 | val second 1.4631 | val 000-022 1.5875 | val 023-044 1.4826 | val 045-066 1.4724 | val 067-088 1.4661 | val 089-110 1.4501 | val 111-127 1.4751 | gap 0.3780 | best 1.4869 @ 11500
  step 15500 | lr 0.0001 | train 1.1099 | val 1.4905 | val first 1.5177 | val second 1.4633 | val 000-022 1.5879 | val 023-044 1.4832 | val 045-066 1.4725 | val 067-088 1.4664 | val 089-110 1.4497 | val 111-127 1.4752 | gap 0.3806 | best 1.4869 @ 11500
  step 16000 | lr 0.0001 | train 1.1075 | val 1.4926 | val first 1.5202 | val second 1.4650 | val 000-022 1.5898 | val 023-044 1.4859 | val 045-066 1.4755 | val 067-088 1.4682 | val 089-110 1.4511 | val 111-127 1.4771 | gap 0.3851 | best 1.4869 @ 11500
  step 16500 | lr 0.0001 | train 1.1052 | val 1.4922 | val first 1.5197 | val second 1.4647 | val 000-022 1.5884 | val 023-044 1.4855 | val 045-066 1.4757 | val 067-088 1.4679 | val 089-110 1.4509 | val 111-127 1.4772 | gap 0.3871 | best 1.4869 @ 11500
  step 17000 | lr 0.0001 | train 1.1025 | val 1.4925 | val first 1.5201 | val second 1.4648 | val 000-022 1.5896 | val 023-044 1.4857 | val 045-066 1.4757 | val 067-088 1.4685 | val 089-110 1.4502 | val 111-127 1.4774 | gap 0.3900 | best 1.4869 @ 11500
  step 17500 | lr 0.0001 | train 1.1007 | val 1.4946 | val first 1.5220 | val second 1.4671 | val 000-022 1.5900 | val 023-044 1.4888 | val 045-066 1.4780 | val 067-088 1.4696 | val 089-110 1.4537 | val 111-127 1.4795 | gap 0.3939 | best 1.4869 @ 11500
  step 18000 | lr 0.0001 | train 1.0980 | val 1.4938 | val first 1.5214 | val second 1.4662 | val 000-022 1.5898 | val 023-044 1.4875 | val 045-066 1.4771 | val 067-088 1.4692 | val 089-110 1.4525 | val 111-127 1.4787 | gap 0.3957 | best 1.4869 @ 11500
  Full/W23/Hybrid matched batch and RNG states: identical

Stage 30 experimental summary
  all branches B/T:             16/128
  hybrid contexts:              (23, 23, None, 23, 23, 23)
  effective RF widths/layer:    23, 45, 128, 128, 128, 128
  first global reachability:    layer 3
  parameter count (all):        1,220,673
  attention shape (all):        (16, 4, 128, 128)
  hybrid best train/val/gap:    1.1527 / 1.4869 / 0.3342 @ 11500
  W23 best train/val:           1.1151 / 1.4803 @ 13500
  Full reference train/val:     1.1527 / 1.4728 @ 13500
  matched Full-Hybrid delta:     -0.0175
  matched Hybrid-W23 delta:      +0.0100
  selected Full-Hybrid delta:    -0.0141
  Negative candidate-baseline deltas favor the named candidate.
  hybrid best checkpoint:        my-gpt/checkpoints/stage_30_hybrid_layer_3_at_128_best_checkpoint.pt
  hybrid best SHA-256:           5c1392578efdd68166ff77b1a0763c84f253e051564f613bb52ea4a43438e046
  hybrid matched anchor:         my-gpt/checkpoints/stage_30_hybrid_layer_3_at_128_step_13500_checkpoint.pt
  hybrid anchor SHA-256:         1903abcda330062b6e74ddf487c310b6e738cc04f26d041ae7503d5cfbcfc426
  W23 references:                my-gpt/checkpoints/stage_29_window_23_at_128_best_checkpoint.pt
  Full reference:                my-gpt/checkpoints/stage_26_context_128_best_checkpoint.pt

W23 with one Full layer at depth 3 selected-best generated text

Petruchio, some poace of all Margaret.

BUCKINGHAM:
Give me them hence, is guard his kneesness
With us that if he answer lose this crown, they communish'd.

GLOUCESTER:
Friend, minds, levite small complainture
Some mouth part and shapes to for comfort
By Warwick at Oxford, or my eyes of Edward's
Please ships you? play, Richard be unto me.

HASTINGS:
Ireland, Lord Angelo.

DUCHESS OF YORK:
Not all the utterness, sitting, and for the field
That I should to give dive: give your kingly
Ladded captai

Held-out three-branch paired validation of Stage 30
  500 shared validation panels, seed=1347; natural trained masks only

matched step endpoint
  full-trained/full-mask | step 13500 | overall 1.480932 +/- 0.003412 SE | first 1.507290 +/- 0.004090 | second 1.454574 +/- 0.004155
    positions 000-022: 1.583564 +/- 0.005238 SE
    positions 023-044: 1.470033 +/- 0.006223 SE
    positions 045-066: 1.457589 +/- 0.005785 SE
    positions 067-088: 1.462812 +/- 0.005997 SE
    positions 089-110: 1.452913 +/- 0.005733 SE
    positions 111-127: 1.446097 +/- 0.006077 SE
  hybrid-trained/hybrid-mask (matched-anchor) | step 13500 | overall 1.493886 +/- 0.003424 SE | first 1.521053 +/- 0.004121 | second 1.466720 +/- 0.004210
    positions 000-022: 1.597502 +/- 0.005282 SE
    positions 023-044: 1.485552 +/- 0.006325 SE
    positions 045-066: 1.468446 +/- 0.005853 SE
    positions 067-088: 1.476762 +/- 0.006260 SE
    positions 089-110: 1.462536 +/- 0.005764 SE
    positions 111-127: 1.460142 +/- 0.006388 SE
  W23-trained/W23-mask (matched-anchor) | step 13500 | overall 1.487418 +/- 0.003475 SE | first 1.513701 +/- 0.004198 | second 1.461136 +/- 0.004216
    positions 000-022: 1.590749 +/- 0.005337 SE
    positions 023-044: 1.476771 +/- 0.006341 SE
    positions 045-066: 1.462466 +/- 0.005711 SE
    positions 067-088: 1.470639 +/- 0.006168 SE
    positions 089-110: 1.457872 +/- 0.005738 SE
    positions 111-127: 1.453641 +/- 0.006435 SE
  primary natural Full-minus-Hybrid: full-trained/full-mask - hybrid-trained/hybrid-mask (matched-anchor); paired-t critical=1.964729
    overall                 -0.012955 +/- 0.000785 SE (95% CI [-0.014497, -0.011412])
    first half              -0.013763 +/- 0.001101 SE (95% CI [-0.015926, -0.011601])
    second half             -0.012146 +/- 0.001133 SE (95% CI [-0.014373, -0.009919])
    positions 000-022       -0.013938 +/- 0.001656 SE (95% CI [-0.017192, -0.010684])
    positions 023-044       -0.015519 +/- 0.001933 SE (95% CI [-0.019317, -0.011721])
    positions 045-066       -0.010857 +/- 0.001898 SE (95% CI [-0.014586, -0.007128])
    positions 067-088       -0.013950 +/- 0.001890 SE (95% CI [-0.017662, -0.010237])
    positions 089-110       -0.009622 +/- 0.001963 SE (95% CI [-0.013479, -0.005765])
    positions 111-127       -0.014045 +/- 0.002125 SE (95% CI [-0.018220, -0.009870])
  secondary natural Hybrid-minus-W23: hybrid-trained/hybrid-mask (matched-anchor) - W23-trained/W23-mask (matched-anchor); paired-t critical=1.964729
    overall                 +0.006468 +/- 0.000928 SE (95% CI [+0.004645, +0.008291])
    first half              +0.007352 +/- 0.001222 SE (95% CI [+0.004952, +0.009752])
    second half             +0.005584 +/- 0.001284 SE (95% CI [+0.003061, +0.008107])
    positions 000-022       +0.006753 +/- 0.001789 SE (95% CI [+0.003237, +0.010268])
    positions 023-044       +0.008781 +/- 0.001982 SE (95% CI [+0.004887, +0.012675])
    positions 045-066       +0.005980 +/- 0.002041 SE (95% CI [+0.001970, +0.009990])
    positions 067-088       +0.006124 +/- 0.002034 SE (95% CI [+0.002127, +0.010120])
    positions 089-110       +0.004664 +/- 0.002015 SE (95% CI [+0.000704, +0.008624])
    positions 111-127       +0.006501 +/- 0.002490 SE (95% CI [+0.001610, +0.011392])
  secondary natural Full-minus-W23: full-trained/full-mask - W23-trained/W23-mask (matched-anchor); paired-t critical=1.964729
    overall                 -0.006487 +/- 0.000762 SE (95% CI [-0.007984, -0.004989])
    first half              -0.006411 +/- 0.001002 SE (95% CI [-0.008380, -0.004443])
    second half             -0.006562 +/- 0.001078 SE (95% CI [-0.008681, -0.004443])
    positions 000-022       -0.007185 +/- 0.001452 SE (95% CI [-0.010037, -0.004333])
    positions 023-044       -0.006738 +/- 0.001736 SE (95% CI [-0.010149, -0.003328])
    positions 045-066       -0.004877 +/- 0.001705 SE (95% CI [-0.008228, -0.001526])
    positions 067-088       -0.007826 +/- 0.001701 SE (95% CI [-0.011167, -0.004485])
    positions 089-110       -0.004958 +/- 0.001786 SE (95% CI [-0.008467, -0.001450])
    positions 111-127       -0.007544 +/- 0.002041 SE (95% CI [-0.011555, -0.003533])
  predeclared Full-minus-Hybrid topology-divergent aggregate [23,128): -0.012739 +/- 0.000868 SE; 95% CI [-0.014444, -0.011034]
  secondary Hybrid-minus-W23 topology-divergent aggregate [23,128): +0.006406 +/- 0.001029 SE; 95% CI [+0.004384, +0.008427]
  secondary Full-minus-W23 topology-divergent aggregate [23,128): -0.006334 +/- 0.000866 SE; 95% CI [-0.008035, -0.004633]
  descriptive overall recovery fraction: -99.7%
  descriptive topology-range recovery fraction: -101.1%

selected best endpoint
  full-trained/full-mask | step 13500 | overall 1.480932 +/- 0.003412 SE | first 1.507290 +/- 0.004090 | second 1.454574 +/- 0.004155
    positions 000-022: 1.583564 +/- 0.005238 SE
    positions 023-044: 1.470033 +/- 0.006223 SE
    positions 045-066: 1.457589 +/- 0.005785 SE
    positions 067-088: 1.462812 +/- 0.005997 SE
    positions 089-110: 1.452913 +/- 0.005733 SE
    positions 111-127: 1.446097 +/- 0.006077 SE
  hybrid-trained/hybrid-mask (selected-best) | step 11500 | overall 1.490819 +/- 0.003323 SE | first 1.517903 +/- 0.003973 | second 1.463736 +/- 0.004084
    positions 000-022: 1.595887 +/- 0.005106 SE
    positions 023-044: 1.482328 +/- 0.006073 SE
    positions 045-066: 1.464479 +/- 0.005653 SE
    positions 067-088: 1.473361 +/- 0.006003 SE
    positions 089-110: 1.460383 +/- 0.005579 SE
    positions 111-127: 1.455729 +/- 0.006190 SE
  W23-trained/W23-mask (selected-best) | step 13500 | overall 1.487418 +/- 0.003475 SE | first 1.513701 +/- 0.004198 | second 1.461136 +/- 0.004216
    positions 000-022: 1.590749 +/- 0.005337 SE
    positions 023-044: 1.476771 +/- 0.006341 SE
    positions 045-066: 1.462466 +/- 0.005711 SE
    positions 067-088: 1.470639 +/- 0.006168 SE
    positions 089-110: 1.457872 +/- 0.005738 SE
    positions 111-127: 1.453641 +/- 0.006435 SE
  secondary natural Full-minus-Hybrid: full-trained/full-mask - hybrid-trained/hybrid-mask (selected-best); paired-t critical=1.964729
    overall                 -0.009888 +/- 0.000780 SE (95% CI [-0.011420, -0.008356])
    first half              -0.010613 +/- 0.001069 SE (95% CI [-0.012714, -0.008513])
    second half             -0.009162 +/- 0.001104 SE (95% CI [-0.011331, -0.006994])
    positions 000-022       -0.012323 +/- 0.001645 SE (95% CI [-0.015555, -0.009090])
    positions 023-044       -0.012295 +/- 0.001838 SE (95% CI [-0.015907, -0.008683])
    positions 045-066       -0.006890 +/- 0.001873 SE (95% CI [-0.010570, -0.003211])
    positions 067-088       -0.010549 +/- 0.001768 SE (95% CI [-0.014022, -0.007075])
    positions 089-110       -0.007470 +/- 0.001923 SE (95% CI [-0.011248, -0.003691])
    positions 111-127       -0.009632 +/- 0.002027 SE (95% CI [-0.013614, -0.005650])
  secondary natural Hybrid-minus-W23: hybrid-trained/hybrid-mask (selected-best) - W23-trained/W23-mask (selected-best); paired-t critical=1.964729
    overall                 +0.003401 +/- 0.000922 SE (95% CI [+0.001589, +0.005213])
    first half              +0.004202 +/- 0.001208 SE (95% CI [+0.001828, +0.006576])
    second half             +0.002600 +/- 0.001275 SE (95% CI [+0.000095, +0.005105])
    positions 000-022       +0.005138 +/- 0.001812 SE (95% CI [+0.001577, +0.008698])
    positions 023-044       +0.005557 +/- 0.001945 SE (95% CI [+0.001735, +0.009378])
    positions 045-066       +0.002013 +/- 0.002045 SE (95% CI [-0.002004, +0.006031])
    positions 067-088       +0.002722 +/- 0.001980 SE (95% CI [-0.001168, +0.006613])
    positions 089-110       +0.002511 +/- 0.002038 SE (95% CI [-0.001493, +0.006515])
    positions 111-127       +0.002088 +/- 0.002371 SE (95% CI [-0.002570, +0.006746])
  secondary natural Full-minus-W23: full-trained/full-mask - W23-trained/W23-mask (selected-best); paired-t critical=1.964729
    overall                 -0.006487 +/- 0.000762 SE (95% CI [-0.007984, -0.004989])
    first half              -0.006411 +/- 0.001002 SE (95% CI [-0.008380, -0.004443])
    second half             -0.006562 +/- 0.001078 SE (95% CI [-0.008681, -0.004443])
    positions 000-022       -0.007185 +/- 0.001452 SE (95% CI [-0.010037, -0.004333])
    positions 023-044       -0.006738 +/- 0.001736 SE (95% CI [-0.010149, -0.003328])
    positions 045-066       -0.004877 +/- 0.001705 SE (95% CI [-0.008228, -0.001526])
    positions 067-088       -0.007826 +/- 0.001701 SE (95% CI [-0.011167, -0.004485])
    positions 089-110       -0.004958 +/- 0.001786 SE (95% CI [-0.008467, -0.001450])
    positions 111-127       -0.007544 +/- 0.002041 SE (95% CI [-0.011555, -0.003533])
  predeclared Full-minus-Hybrid topology-divergent aggregate [23,128): -0.009354 +/- 0.000861 SE; 95% CI [-0.011047, -0.007662]
  secondary Hybrid-minus-W23 topology-divergent aggregate [23,128): +0.003021 +/- 0.001026 SE; 95% CI [+0.001005, +0.005037]
  secondary Full-minus-W23 topology-divergent aggregate [23,128): -0.006334 +/- 0.000866 SE; 95% CI [-0.008035, -0.004633]
  descriptive overall recovery fraction: -52.4%
  descriptive topology-range recovery fraction: -47.7%

  The matched-step Full-minus-Hybrid natural overall endpoint is primary. Hybrid-minus-W23, Full-minus-W23, selected-best results, position bands, and recovery fractions are secondary/descriptive.
  Recovery fractions are ratios of paired point estimates and do not have inferential intervals.

```
