.PHONY: validation-targets score leaderboard leaderboard-md darksteeld-baselines

# Any Python with polars + numpy (builder) and numpy (evaluator) works.
# Default: repo-local .venv if present, else python3. Override: make score PY=...
PY ?= $(or $(wildcard .venv/bin/python), python3)

MEMBER ?=
EXPERIMENT ?=
PREDICTIONS ?= validation/predictions/$(MEMBER)/$(EXPERIMENT)
NOTES ?=
PUBLIC_PRAUC ?=

validation-targets:
	$(PY) -m validation.build_folds

score:
	@test -n "$(MEMBER)" || (echo "MEMBER is required" && exit 2)
	@test -n "$(EXPERIMENT)" || (echo "EXPERIMENT is required" && exit 2)
	$(PY) -m validation.evaluate \
		--member "$(MEMBER)" \
		--experiment "$(EXPERIMENT)" \
		--predictions-dir "$(PREDICTIONS)" \
		--notes "$(NOTES)" $(if $(strip $(PUBLIC_PRAUC)),--public-prauc "$(PUBLIC_PRAUC)",)

leaderboard:
	$(PY) -m validation.evaluate --rebuild-only

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
