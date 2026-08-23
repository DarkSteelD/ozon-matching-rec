# Track 1 (matching) — iteration log

Goal: max mean_prauc on frozen 4-fold validation in ≤30 scored iterations.
Baseline to beat: lgbm_cheap_v1 = 0.63787 (team lb). Scoring: scripts/score.sh (scratch only, no auto-publish).
ODS public context (2026-08-13): top-1 total_prauc 0.4827, median 0.3044.

| it | exp | mean_prauc | delta vs best | notes |
| -- | --- | ---------- | ------------- | ----- |
| 1 | ce_tiny_hand | 0.56419 | -0.0737 | rubert-tiny2 CE hand-only, name+cat, len64 |
| 2 | ce_tiny_hand_attrs | 0.56452 | -0.0733 | +attrs len160: no gain for tiny |
| 3 | ce_tiny_llm2m_zs | 0.43625 | -0.2016 | zero-shot after 2M LLM pretrain; label shift but signal real |
| 4 | ce_tiny_llm2m_hand | 0.57957 | -0.0583 | LLM pretrain +0.015 at tiny scale |
| 5 | lgbm_ce_tiny | (killed, unscored — superseded by it9) | | |
| 6 | ce_rubase_hand | 0.78877 | +0.1509 | rubert-base hand-only OOF len128 2ep — NEW BEST |
| 7 | ce_rubase_llmft_hand | 0.82660 | +0.0378 | 11.2M LLM pretrain -> hand FT — NEW BEST |
| 8 | ens_rubase_pair | 0.82666 | +0.0001 | rank ens: same-arch adds nothing |
| 9 | lgbm_ce_rubase | 0.82708 | +0.0005 | LGBM stack: CE dominates |
| 10 | ce_rubase_llmft_len160 | 0.83287 | +0.0058 | len160 FT — attrs truncation costs; NEW BEST |
| 11 | ce_rubase_e2_hand | 0.83474 | +0.0019 | 2-epoch LLM pretrain, FT len160 — NEW BEST |
| 12 | ce_rubase_llmft_len224 | 0.83830 | +0.0036 | len224+attrs800 from e1 ckpt — NEW BEST |
| 13 | ce_e5_llmft_hand | 0.82589 | -0.0124 | e5-base diversity model, len160 |
| 14a | ens3 | 0.84455 | +0.0063 | rank ens 3 models — NEW BEST |
| 14 | ce_rubase_e2_len224 | 0.84073 | best single | e2 + len224 |
| 15 | ce_e5_len224 | 0.83183 | | e5 len224 |
| 16 | ens4 | 0.84694 | +0.0024 | 4-model weighted rank ens — NEW BEST |
| 17 | lgbm_stack5 | 0.84636 | -0.0006 | stack < rank ens |
| 18 | ce_rubase_e2_len288 | 0.84390 | best single | len288 |
| 19 | ens5 | 0.84931 | +0.0024 | rank ens with len288 — NEW BEST |
| 20 | ce_e5_len288 | 0.83444 | | e5 len288 |
| 21 | ens6 | 0.84969 | +0.0004 | NEW BEST |
| 22 | ens7 | 0.85020 | +0.0005 | + seed-2 bag — NEW BEST |
| 23 | ce_rubase_e2_len384 | 0.84532 | best single | len384 |
| 24 | ens8 | 0.85161 | +0.0014 | ens with len384 — NEW BEST |
| 25 | ce_mdeb_len224 | 0.83108 | | mdeberta diversity member |
| 26 | ens9 | 0.85406 | +0.0024 | 7-member grand ens — NEW BEST |
| 27 | ens10 | 0.85405 | -0.0000 | weight probe, no gain |
| 28 | lgbm_stack_final | 0.85432 | +0.0003 | learned stack — NEW BEST |
| 29 | final_combo | 0.85548 | +0.0012 | rank avg stack+ens9 — FINAL BEST |
| 30 | final_combo2 | 0.85546 | -0.0000 | weighted variant, no gain |

Final: **0.85548 mean_prauc** (`final_combo` = rank-avg of lgbm_stack_final + ens9). Predictions: preds_disk/final_combo. Recipe: rubert-base + e5-base + mdeberta-v3, each BCE-pretrained on 11.19M soft-label LLM pairs (leak-free, disjoint items), fine-tuned per-fold on hand pairs (name | category | attrs<=800ch), lengths 224-384, rank-ensembled + LGBM stack with 21 cheap features.
