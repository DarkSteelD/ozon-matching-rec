# Reproducibility datasets

This directory contains the smallest dataset snapshot needed to reproduce the
winning cross-encoder line without relying on `~/matching-work`.

- `hand_pairs.parquet`: canonical 365,654-row hand-labelled fold dataset.
- `hand_pairs_pd_v3cal.parquet`: calibrated v3 teacher-target dataset used by
  the winning distillation recipe.
- `hand_pairs_prio_distill.parquet`: prioritized-text distillation dataset.
- `llm_pairs_full/`: the 11,187,780-row LLM-labelled pretraining dataset,
  split into four ordered ZSTD parquet shards. Concatenating the shards in
  lexical filename order preserves the source row order.

The parquet files are stored with Git LFS. Token caches and model checkpoints
are deliberately excluded: `members/dzkhomidov/src/tokenize_llm.py` rebuilds
the former, while the latter are generated training outputs rather than source
datasets.

See `MANIFEST.json` for byte hashes, row counts, schemas and the source-to-shard
logical-content check.
