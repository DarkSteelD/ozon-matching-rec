# Final matching artifact manifest

Audited source directory:
`/home/dzkhomidov/matching-work/container/sub_final2`.

Consolidated archive:
`/home/dzkhomidov/matching-work/container/ecup_matching_consolidated_v2.zip`.

The container is the current-best consolidated recipe: a priority-text student
at length 224 and weight 1.0, blended with mDeBERTa at length 160 and weight 0.5,
plus the fashion-size penalty in `run.py`. The source `run.py` is byte-identical
to `members/dzkhomidov/container/run_v4.py`.

The directory contains 16 files totaling 939,289,334 bytes. The ZIP contains
exactly the same 16 files, passes `unzip -t`, and has byte-identical uncompressed
content for every entry.

## SHA-256 inventory

| Relative path | Bytes | SHA-256 |
|---|---:|---|
| `metadata.json` | 83 | `6b544b0fb38853da684458e6a261be2e9022490f63b3e19db86a08d8ae3f006c` |
| `models.json` | 146 | `d2f600bd2ae202cf0e105d870db50338ca8d620ae070decfabaaabc6ea8cd82b` |
| `models/mdeb/added_tokens.json` | 23 | `fb697283833d25e2c711f1bc37730ecd8b20f4bd5f015db1d84aefe0adc9155a` |
| `models/mdeb/config.json` | 986 | `32cf20485711534b020af1acfe5f22d896521c7cc7233cd70b5d87b9313de77e` |
| `models/mdeb/model.safetensors` | 557,644,770 | `965e3913d01e5715ff20f3b5940223c616f291460814913e7b945040339ec142` |
| `models/mdeb/special_tokens_map.json` | 286 | `9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9` |
| `models/mdeb/spm.model` | 4,305,025 | `13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3` |
| `models/mdeb/tokenizer.json` | 16,351,029 | `b0b5830e1b34447deba02ce1541f8802c6e5ef3d145ab6c431169c142ad27f2b` |
| `models/mdeb/tokenizer_config.json` | 19,675 | `8032d12bf0b260fc43a05ceb8e640f31847c5f97c9ef7c0229109a62458147c7` |
| `models/student/config.json` | 977 | `4c51f6b4219ff3867d3bd9bc9af279ecf7ecf0a2ab3dc43c07738b4f057ced8d` |
| `models/student/model.safetensors` | 355,731,962 | `ee67d9f0dd450b292e9c7931d1d6df7ca575674d82f8dee73c614124d348983d` |
| `models/student/special_tokens_map.json` | 125 | `b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3` |
| `models/student/tokenizer.json` | 3,573,819 | `a74460ab9429b147d10e1a6a90a47ef46f40c380d42654f29016fc1e2e5e9238` |
| `models/student/tokenizer_config.json` | 1,271 | `0eb95891c3de474ec9ea0cd665b43f1e9fdd3604f76e8d66b3f85e9b301232a1` |
| `models/student/vocab.txt` | 1,649,718 | `78106a3d3ae8600d1ba573b967b9bb731d2c2282957cbc6e26ab20935c3da02b` |
| `run.py` | 9,439 | `c7c3a5c089ccf2fc089bb44ca416f543cdae15d4bb4f51b16c362ba836ac6a23` |
| `ecup_matching_consolidated_v2.zip` | 849,643,430 | `7c115e7dd77653b4d19ecef80cf6772202e2e79fc690005a3970d0a68bc48bad` |

## Git storage decision

Git LFS 3.3.0 is installed on the machine, but this repository has no
`.gitattributes`, no LFS-tracked files, and no rules for ZIP or Safetensors
artifacts. Therefore LFS is not safely configured for this repository.

The following required blobs exceed GitHub's 100 MB regular-object limit and
were not added:

- `models/mdeb/model.safetensors` — 557,644,770 bytes
- `models/student/model.safetensors` — 355,731,962 bytes
- `ecup_matching_consolidated_v2.zip` — 849,643,430 bytes

Before preserving the artifact in Git, configure and verify LFS for at least
`*.safetensors` and `*.zip`, confirm that the remote accepts LFS objects, then
add the complete artifact in a separate commit. Partial small-file copies were
intentionally not committed because they cannot reproduce the submission.

No credentials, authentication state, submission receipts, labels, predictions,
validation files, caches, or logs are part of this manifest.
