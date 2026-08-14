.PHONY: validation-targets validation-targets-v2 score score-v2 leaderboard leaderboard-v2 \
	leaderboard-md darksteeld-baselines train train-list

# Any Python with polars + numpy (builder) and numpy (evaluator) works.
# Default: repo-local .venv if present, else python3. Override: make score PY=...
PY ?= $(or $(wildcard .venv/bin/python), python3)

MEMBER ?=
EXPERIMENT ?=
NOTES ?=
PUBLIC_PRAUC ?=

# Fold specification in use. v1 = hashed components (no stratification);
# v2 = stratified by category and label. Switching specs is a validation
# version change, not a tweak: OOF predictions belong to the partition they
# were produced on, so an experiment must be RE-RUN, never re-scored, when the
# spec changes. Override the whole set together, or use the *-v2 targets.
SPEC ?= validation/folds.json
TARGETS ?= validation/targets
RESULTS ?= validation/results
LEADERBOARD ?= validation/leaderboard.csv
PREDICTIONS ?= validation/predictions/$(MEMBER)/$(EXPERIMENT)

V2 = SPEC=validation/folds_v2.json TARGETS=validation/targets_v2 \
     RESULTS=validation/results_v2 LEADERBOARD=validation/leaderboard_v2.csv

validation-targets:
	$(PY) -m validation.build_folds --spec "$(SPEC)" --targets-dir "$(TARGETS)"

validation-targets-v2:
	$(MAKE) validation-targets PY=$(PY) $(V2)

score:
	@test -n "$(MEMBER)" || (echo "MEMBER is required" && exit 2)
	@test -n "$(EXPERIMENT)" || (echo "EXPERIMENT is required" && exit 2)
	$(PY) -m validation.evaluate \
		--member "$(MEMBER)" \
		--experiment "$(EXPERIMENT)" \
		--predictions-dir "$(PREDICTIONS)" \
		--spec "$(SPEC)" --targets-dir "$(TARGETS)" \
		--results-dir "$(RESULTS)" --leaderboard "$(LEADERBOARD)" \
		--notes "$(NOTES)" $(if $(strip $(PUBLIC_PRAUC)),--public-prauc "$(PUBLIC_PRAUC)",)

score-v2:
	$(MAKE) score PY=$(PY) MEMBER="$(MEMBER)" EXPERIMENT="$(EXPERIMENT)" \
		PREDICTIONS="$(PREDICTIONS)" NOTES="$(NOTES)" $(V2)

leaderboard:
	$(PY) -m validation.evaluate --rebuild-only \
		--spec "$(SPEC)" --targets-dir "$(TARGETS)" \
		--results-dir "$(RESULTS)" --leaderboard "$(LEADERBOARD)"

leaderboard-v2:
	$(MAKE) leaderboard PY=$(PY) $(V2)

leaderboard-md:
	$(PY) -m validation.render_leaderboard

# Reference baselines (member darksteeld): writes predictions for all shared
# folds, then registers every baseline on the team leaderboard.
darksteeld-baselines:
	$(PY) members/darksteeld/src/validation_baselines.py --baseline all
	$(MAKE) score PY=$(PY) MEMBER=darksteeld EXPERIMENT=const_prior \
		NOTES="Constant prediction = global hand-label prior 0.2568; PR-AUC of a constant equals fold prevalence"
	$(MAKE) score PY=$(PY) MEMBER=darksteeld EXPERIMENT=name_exact \
		NOTES="1.0 if normalized names are equal else 0.0 (lowercase, yo->ye, non-alnum->space)"
	$(MAKE) score PY=$(PY) MEMBER=darksteeld EXPERIMENT=name_tfidf_cos \
		NOTES="char_wb 3-5gram TF-IDF cosine of names; fit on items_human names only (test-legal transductive fit)"
	$(MAKE) score PY=$(PY) MEMBER=darksteeld EXPERIMENT=attr_jaccard \
		NOTES="Jaccard over attributes key=value token sets parsed from the attributes JSON"
	$(MAKE) score PY=$(PY) MEMBER=darksteeld EXPERIMENT=name_tfidf_attr_blend \
		NOTES="0.5 * name TF-IDF cosine + 0.5 * attributes key=value Jaccard"

train:
	@test -n "$(MEMBER)" || (echo "MEMBER is required" && exit 2)
	@test -n "$(EXPERIMENT)" || (echo "EXPERIMENT is required" && exit 2)
	$(PY) validation/ops/train.py \
		--member "$(MEMBER)" --experiment "$(EXPERIMENT)" $(if $(strip $(GPUS)),--gpus "$(GPUS)",) $(if $(strip $(DRY)),--dry-run,)

train-list:
	$(PY) validation/ops/train.py --list
