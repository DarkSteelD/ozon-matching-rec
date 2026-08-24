# Commands

The frozen trainer is
`/home/dzkhomidov/matching-work/rescue_20260824/student_long_context/train.py`
(SHA256 `46b10630fc1bb11140eb5326fded09f99c2e0c6cd9f7154c898c7e27b355970d`).

Both variants use physical GPU6 and init
`/home/dzkhomidov/matching-work/rescue_20260824/third_pretrain/ckpt/rubase_llmfull_e3`.
Exact invocations are retained in `runner.log` and differ only in
`--variant/--max-len` (`e3_len224/224`, then `e3_len384/384`).
