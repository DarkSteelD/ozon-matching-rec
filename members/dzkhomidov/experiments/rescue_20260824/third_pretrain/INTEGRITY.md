# Input integrity

- `llm_pairs_full.parquet`: 11,187,780 rows, 91 row groups.
- token ids: `(11,187,780, 128)` int32; labels: `(11,187,780,)`
  float32, finite, range 0..1.
- Original tokenizer log enumerates all 46 slices and ends `cache complete`.
- Exact retoken comparison passed on 12 rows spanning the first row, slice
  boundaries, interior rows, and final row using attrs + category at len128.
- `rubase_llmfull_e2/config.json` records `_name_or_path` as the epoch-1
  checkpoint. Its model hash differs from epoch 1 and matches the disk copy.
  `train_ce_fast.py` saves the non-interim target directory only after its
  entire configured loop, and the archived epoch-2 OOF result is present.
- Limitation: the original epoch-2 stdout/command was not retained, so the
  claim that it was exactly one continuation pass is supported by checkpoint
  provenance + iteration log, not by optimizer state or a completion log.
- Archived epoch-2 predictions align 1:1 with canonical rows and independently
  reproduce official fold scores: fold_01 0.8309671088, fold_02 0.8406228737.
