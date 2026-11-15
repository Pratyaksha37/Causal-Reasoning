# Probability modelling, reasoning, and evidence review

Probability modelling is part of the same project, in `src/bayesian/`, `src/causal/`, `src/data/`, and the shared evaluation/annotation modules. It fits a discrete Bayesian network to real BRFSS observations, answers observational queries, and simulates interventions under explicit graph assumptions. It also provides an independent-review workflow; missing human annotations never block software execution and are never fabricated.

## Install and run

From the project root, using Python 3.11:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m src download
MPLBACKEND=Agg .venv/bin/python -m src run --data data/brfss/raw/LLCP2015XPT.zip --output results/modelling/brfss2015
```

The explicit download command accesses the official CDC endpoint. Subsequent execution is offline; it does not require Gemini keys; the integrated run rebuilds evidence in `results/graph/current/` before fitting. CSV, XPT, and a ZIP containing one XPT member are supported. Downloads and large row-level outputs are excluded from Git.

For a reproducible **synthetic software demonstration**, without downloading data:

```bash
MPLBACKEND=Agg .venv/bin/python -m src run --demo
```

Demo results are separately labelled and saved to `results/modelling/demo`. Synthetic probabilities are arbitrary test inputs, not biomedical findings. Missing real data raises an error unless `--demo` was explicitly supplied.

Open `results/index.html` locally for the interactive BMI comparison, fitted CPDs, calibration, review coverage, and downloadable results. The CLI is the query interface; no web server or additional UI dependency is required.

## Query a saved model

```bash
.venv/bin/python -m src query --model results/modelling/brfss2015/model.json --evidence BMICat=obese
.venv/bin/python -m src intervene --model results/modelling/brfss2015/model.json --do BMICat=normal
```

The default network contains only `BMICat → Outcome`. Age and activity are preprocessed when available, but are not silently included in this sparse model. Queries using absent variables or invalid categories return a readable error and exit code 2. State names are case-sensitive strings from the canonical registry.

`query` calculates P(Outcome | observed evidence). `intervene` cuts incoming edges to the treatment and replaces its CPD with a point mass, then performs variable elimination. It does **not** use conditioning as a substitute for intervention.

By default, intervention results have status `model_simulation_only`. Three optional CLI flags explicitly assert assumptions:

```bash
.venv/bin/python -m src intervene --model results/modelling/brfss2015/model.json --do BMICat=normal --assume-causal-sufficiency --assume-consistency --acknowledge-structural-prior
```

Even with those flags, identification is reported only *under the asserted DAG assumptions*, requires empirical treatment support in every parent configuration, and never establishes real-world causal validity. Bayesian pseudocounts cannot establish empirical positivity. Latent-confounder identification, front-door identification, and transportability are not implemented.

## Human review without invented labels

```bash
.venv/bin/python -m src review-report --export-form /tmp/reviewer_a.csv
.venv/bin/python -m src review-report --reviews /path/to/submitted_reviews.csv
.venv/bin/python -m src run --data data/brfss/raw/LLCP2015XPT.zip --reviews /path/to/reviews.jsonl --adjudications /path/to/adjudications.jsonl
```

Give each reviewer a separate blank form. Complete `annotator_id`, timezone-aware `timestamp`, `decision` (`causal`, `association`, `none`, `uncertain`), `direction`, `independent_review=true`, and `source=human_submission`; notes are optional. Causal decisions require `entity_1->entity_2` or its reverse; all other decisions use `not_applicable`. The sentence and PMID are joined from the frozen queue rather than trusted from edited form cells.

JSONL submissions use the same review fields, excluding form context columns. Blank CSV decisions are pending. Duplicate submissions by one reviewer for a pair are rejected. Agreement is computed separately for every reviewer pair with overlapping items. Kappa is null when undefined; raw agreement remains available. Blinding and expertise cannot be established by software; independence is a reviewer attestation.

Adjudications require `pair_id`, `adjudicator_id`, a timezone-aware `timestamp`, `decision`, `direction`, `rationale`, and `source=human_adjudication`. At least two distinct reviewers must have submitted the item first, and the adjudicator must be a separate person. Adjudicated non-uncertain **relation/direction** labels are exported separately as `human_gold.jsonl`; this does not certify polarity or language status, or automatically validate every model edge. literature extraction pseudo-labels are never overwritten.

## Models and CPDs

`configs/bayesian.yaml` controls input mappings, model variables, seed, split, estimation, assumptions, and output directory. Read canonical variable names/states from `configs/aliases.yaml`.

Default CPDs use BDeu posterior means: each cell receives `equivalent_sample_size / (number_of_child_states * number_of_parent_configurations)`. Default ESS is 5. All canonical states are represented, including states absent from training data. MLE is also supported (`estimation.method: mle`); it rejects unobserved parent configurations rather than inventing probabilities. Every CPD must be finite, nonnegative, and normalized.

For an exploratory model, set `model.mode: exploratory`, list the canonical variables, and supply `exploratory_edges` as objects with `source`, `target`, and a nonempty `rationale`. The prior edge is retained. Extra edges cannot target baseline AgeCat/Sex, reference absent nodes, or introduce cycles. Additional isolated nodes require `acknowledge_independence: true`; independence is an assumption, not recovered scientific knowledge. Exploratory edges remain separately recorded and never modify the literature extraction graph.

## Evaluation and reproducibility

A fixed-seed, outcome-stratified 80/20 split is made before CPD estimation. Reported predictive metrics use only the held-out set. The comparison baseline is training outcome prevalence. Calibration bins, log loss, Brier score, smoothing sensitivity (ESS 1/5/20), and an explicitly exploratory edge-removal ablation are saved. No causal accuracy is reported.

Artifacts include `model.json`, `cpds.csv`, `processed.csv`, `splits.csv`, `preprocessing.json`, `validation.json`, `summary.json`, `sensitivity.json`, `queries.json`, `config.json`, `report.md`, `index.html`, and five figures. `summary.json` records the source hash, configuration, package versions, class counts, splits, and assumptions. JSON model persistence avoids executable pickle. `splits.csv` refers to zero-based rows of `processed.csv`.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
./run.sh test
```

See `docs/BRFSS_DATASET.md`, `docs/ARCHITECTURE.md` for mappings, methodological boundaries, and executed results.


Current-graph paths are canonical for direct Bayesian-stage calls as well as integrated runs. Saved input provenance uses project-relative paths or external basenames, with SHA-256 retained. An ablation records its empty active edge set separately from the original evidence prior and re-estimates counts for the new parent sets.

Calibration reports include the number of populated bins, distinct probability values, and prediction range. Empty high-risk bins provide no calibration evidence. `causal_accuracy` remains null: a future evaluation first needs a prespecified causal estimand, defensible identification assumptions, and independent intervention or credible quasi-experimental outcomes. An RCT linkage alone does not define a generic causal-accuracy score.
