# len384 deployability benchmark

Checked on `avi-ling-gpu03`, physical A100 GPU1, with two untouched idle
resident kernels totaling 5,220 MiB. Input is the full local scale: 13,397,761
items and 365,654 pairs. The model is the confirmed fold-04 len384 RuBERT
checkpoint; architecture and runtime are representative, but it is not the
deployable all-hand refit.

## Exact two-direction result

- text/item build: 270.60 s
- symmetric two-direction inference: 629.20 s
- total: 906.87 s (15.11 min)
- throughput: 581.14 pairs/s
- torch peak allocated/reserved: 3,600.6 / 5,550 MiB
- board peak: 11,282 MiB; process increment over resident baseline: 6,062 MiB
- prediction float32 SHA256:
  `2ae70e0626ad8e4ead0b89ae3df4dbbaecf7701795d483e3c8fe67b5e381cafe`

Conclusion: memory is safe, but exact two-direction len384 is not a plausible
20-minute T4 path.

## Exact single-direction result

- text/item build: 274.52 s
- canonical `id1 -> id2` inference: 338.67 s
- total: 620.38 s (10.34 min)
- throughput: 1,079.68 pairs/s
- torch peak allocated/reserved: 3,600.8 / 5,550 MiB
- board peak: 11,282 MiB; process increment: 6,062 MiB
- prediction float32 SHA256:
  `6a33b7d74169ec998255feca3d9a7f8382b08e18320a91ed3f4786df278d5146`

This is a dedicated measured run, not an inference from halving the symmetric
runtime. Memory is safe. A T4 runtime claim is still unchecked; the candidate
therefore retains the hard 15.5-minute guard and tail-length fallback.
