# Field dropout + consistency gate

Claim: dropping complete attribute fields at rate 0.05 in two views of the same
training row, with Bernoulli symmetric-KL consistency weight 0.1, improves macro
category AP by more than 0.001 and has positive deltas on both folds 01–02.

All arms use the same ruBERT epoch-2 initialization, clean evaluation text,
max length 224, batch 256 rows, two views (512 encoded sequences/update), two
epochs, update count, seed, folds and batch permutations.

Arms: two-clean-view BCE baseline; field dropout 0.05; negative control with an
equal character-rate non-field-aligned span corruption and consistency against a
one-row-rolled/mismatched second view. Field rate 0.10 runs only if 0.05 passes.
Only a dose passing the same-sign and +0.001 pooled macro gate advances to folds
03–04.

After phase 1, the passing field dose with the largest pooled gate delta is
selected. Phase 2 reruns both the matched two-view BCE baseline and that dose on
folds 03–04, then scores the complete four-fold OOF union.

Evaluation slices are predeclared: either/one-side empty attrs, attribute-key
Jaccard below 0.25, field-count ratio at least 2, total-text length ratio at
least 2, and each category. No predictions enter `validation/`.
