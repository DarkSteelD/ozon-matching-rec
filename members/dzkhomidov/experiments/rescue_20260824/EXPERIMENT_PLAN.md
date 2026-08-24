# Matching rescue — 2026-08-24

## Goal

Find a reproducible gain over the current cross-encoder recipe without using ODS
as the primary evaluator. Exploratory outputs stay outside the repository's
`validation/` tree. No submission, push, or external message is authorized.

## Frozen context

- Repository: `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec`
- Starting SHA: `2da459984a1207677ff9eb863ca28589027a4bc3`
- Primary split: existing grouped four-fold hand split
- Primary metrics: mean fold PR-AUC and macro category PR-AUC
- Secondary: per-category PR-AUC, especially Shoes, Clothing, Accessories
- Current OOF reference: `final_stack_v3` mean PR-AUC `0.85768258`
- Deployable reference: calibrated/symmetric rubase student + mDeBERTa + soft
  fashion size penalty; ODS `0.4969291743`

## Noise and acceptance

The cheap LightGBM pipeline has measured seed sigma `0.00031` and range
`0.00076`. Neural candidates are not assumed less noisy. Screening uses folds
1–2, but a positive conclusion requires all four folds. A candidate advances
when the first two folds have the same delta sign and either mean or macro
category PR-AUC improves by more than `0.001`, with no catastrophic category.

## Parallel arms

| Arm | Measurable claim | Controls | Initial coverage |
|---|---|---|---|
| category balancing | equal category contribution improves macro AP | ordinary BCE | folds 1–2 |
| hard-negative weighting | train-only hard negatives improve ranking | ordinary BCE + random negatives at equal coverage | folds 1–2 |
| long context | len 448/512 improves over len 384 where truncation exists | len 384 + len 224 sanity control | folds 1–2 |
| third LLM-pretrain epoch | epoch 3 transfers beyond epoch 2 | epoch-2 checkpoint with identical hand FT | folds 1–2 |
| category blend | per-category architecture diversity transfers out of fold | global blend + permutation/random weights | leave-one-fold-out, all folds |
| fashion stress audit | a label-free slice reproduces the hidden-like collapse | prevalence-only resampling | all OOF folds |
| ruRoBERTa-large | model scale gives a teacher-side gain | comparable base model | folds 1–2 if feasible |
| pairwise ranking loss | within-category RankNet improves PR-AUC over BCE | BCE + random-pair control | folds 1–2 |

## Explicitly not repeated

- zero-shot LLM sweeps;
- fashion-only CE;
- hard zero for size mismatch;
- exact train-pair lookup;
- automatic label flipping;
- same-architecture seed ensembling;
- KNRM blending.

## Completion contract

Every arm must save exact commands, host/GPU/PID, data/checkpoint identity,
fold-level metrics, runtime, checked/unchecked variants, and a positive,
negative, or inconclusive conclusion. Only arms passing the screening gate run
folds 3–4.
