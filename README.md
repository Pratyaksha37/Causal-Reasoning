# Neuro-Symbolic Causal Inference for Diabetes Evidence

One Python project connects biomedical language processing, a citation-backed evidence DAG, Bayesian probability estimation, and observational versus hypothetical intervention reasoning.

**Biomedical text → relation extraction and comparison → evidence DAG → BRFSS CPDs → probability queries and interventions**

## Run the project

Use Python 3.11 from the project root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m spacy download en_core_web_sm
./run.sh prepare-encoder --allow-download
./run.sh download
./run.sh run --data data/brfss/raw/LLCP2015XPT.zip
```

`prepare-encoder` copies the pinned public MiniLM checkpoint from the Hugging Face cache; `--allow-download` permits downloading it if missing. The dataset download uses official CDC BRFSS 2015. After those assets are present, execution is offline and does not call an LLM. Running `./run.sh` without arguments uses the downloaded BRFSS archive if available, otherwise shows help. The equivalent Python entry point is `.venv/bin/python -m src`.

Open **[results/index.html](results/index.html)** for the integrated viewer. It contains interactive literature examples, the controlled model comparison, citation evidence, the DAG, CPDs, BMI observation/intervention controls, calibration, and limitations.

For an explicit synthetic probability demonstration:

```bash
./run.sh run --demo
```

The NLP stage still uses real frozen abstracts; the Bayesian observations in this mode are explicitly synthetic. Demo model files are separate from real-data model files. No real observations are silently replaced.

## What the product demonstrates

| Capability | Implementation |
|---|---|
| NLP-based causal relationship extraction | spaCy rules, keyword baseline, mapped entity mentions, sentence-level relation labels and source provenance. `predict-text` also runs saved learned heads on new text. |
| Fair rule/ML/transformer comparison | spaCy rules, KeywordHeuristic, train-only TF-IDF + logistic regression, and genuine pretrained MiniLM + a train-only logistic head; identical frozen document splits and task metrics. |
| Evidence-backed DAG construction | Saved literature predictions are filtered, evidence is deduplicated by sentence/paper, cycles are checked/repaired, and the resulting structural prior initializes the Bayesian model. Associations remain separate. |
| Bayesian probability estimation | Training-only MLE or Dirichlet-smoothed BDeu CPDs from BRFSS; normalized tables and a portable saved model. |
| Observation and intervention | Variable elimination for P(Y|X); graph surgery and point-mass treatment CPDs for P(Y|do(X)), with explicit identification assumptions and empirical positivity checks. |
| Transparent evaluation | Document holdouts for NLP; respondent holdouts for Bayesian prediction; support counts, confusion matrices, log loss, Brier score, calibration, sensitivity, provenance, and limitations. |

## Use the model

```bash
./run.sh predict-text --text "Obesity causes type 2 diabetes."
./run.sh query --model results/modelling/brfss2015/model.json --evidence BMICat=obese
./run.sh intervene --model results/modelling/brfss2015/model.json --do BMICat=normal
./run.sh benchmark
./run.sh review-report --export-form /tmp/independent_review.csv
./run.sh test
```

The default graph contains only `BMICat → Outcome`. Age, activity, and the other canonical variables are preprocessed where present but are not silently given invented causal edges. Additional hypotheses require an explicitly exploratory model configuration. Unsupported query variables return a readable error.

For literature-only execution, the existing commands remain available:

```bash
./run.sh extract --from-saved-predictions
# Explicit live extraction may use the configured LLM API:
./run.sh extract --max-pairs 250 --sample-strategy stratified
```

## Measured model comparison

The controlled benchmark uses 150 training, 50 development, and 50 test pairs, with document-level separation. Both learned systems use the same bounded, mention-marked text and logistic classifier settings. TF-IDF vocabulary and classifier heads are fitted on training records only. MiniLM's pretrained encoder weights are actually loaded and frozen; **this is a transformer feature baseline, not full transformer fine-tuning**. KeywordHeuristic remains a separate non-neural baseline.

| Method | Primary: supported-class macro F1 | Secondary: all 3 classes |
|---|---:|---:|
| spaCy-Rules | 0.8786 | 0.5857 |
| KeywordHeuristic | 0.7772 | 0.5181 |
| TF-IDF + LogisticRegression | 0.8300 | 0.5533 |
| MiniLM frozen encoder + LogisticRegression | 0.7326 | 0.4884 |

**Interpretation:** These are comparisons against provisional, rule-generated pseudo-labels, which favour related rules. Records identify `rule_heuristic_regex_v1`; metadata alone does not verify the exact generator code or prove spaCy generated these labels. Related-rule evaluation may be circular. Test and dev contain **zero causal examples**, so held-out causal generalization is unmeasured. Supported-class F1 covers association and none only. There are only four causal training examples. No conclusion about independent clinical accuracy or universal transformer superiority is justified. Historical LLM pilots use different subsets and are excluded from this controlled table.

The historical `DistilBERTExtractor` name is a compatibility alias for KeywordHeuristic; it is never counted as a trained transformer. The actual transformer in this product is `sentence-transformers/all-MiniLM-L6-v2` with a fitted logistic head. Learned heads estimate relation class only; their outputs are not promoted to directed causal edges without direction and evidence validation.

## Measured probability-model results

The executed official BRFSS 2015 run retained **401,327 of 441,456 records** for complete BMI/outcome modelling, using 321,061 training and 80,266 held-out observations. Held-out log loss is **0.396413** and Brier score **0.120483**, compared with training-prevalence baseline values of 0.421152 and 0.126872. These are unweighted observational prediction metrics.

The resulting prior DAG has 2 nodes and 1 edge; 31 observational association edges are preserved separately. The graph is a project-accepted literature prior, not independently established biological causation. No additional causal relationships are fabricated to make it larger.

## Essential limitations

- Human review is pending. The independent review/adjudication workflow works without manufacturing annotations or blocking computation.
- Literature claims, predictive associations, and causal identification are different forms of evidence.
- BRFSS is cross-sectional self-report. `Outcome` combines prediabetes and diabetes; it does not establish incident T2DM.
- Survey weights and the complex sample design are not used; results are respondent-level estimates, not nationally representative prevalence.
- Default interventions are **model simulations only**. Equal observational and intervention probabilities in the sparse two-node graph follow from its assumed structure, not proof of no confounding.
- Causal sufficiency, consistency, positivity, temporal ordering, and transport from literature to survey measures require justification. No latent-variable ID solver or clinical recommendation system is claimed.

## Project layout

```text
src/
  retrieval/       frozen literature and Europe PMC
  text/            sentences, mentions, candidate pairs
  extractors/      rules, keyword, LLM, real transformer and learned heads
  mapping/         canonical variables and precision rules
  annotation/      sampling, independent review and adjudication
  graph/           evidence aggregation, DAG checks and provenance
  data/            BRFSS loading and preprocessing
  bayesian/        structure, CPDs, persistence and inference
  causal/          interventions and identification conditions
  evaluation/      common NLP benchmark and probability evaluation
  visualization/   the integrated viewer and figures
  pipeline.py      whole-project orchestration
  cli.py           one command interface
configs/           experiment, benchmark, Bayesian, aliases and model settings
tests/             unit and integration tests for the whole project
results/           one viewer plus benchmark, graph, modelling and verification artifacts
docs/              architecture, dataset and modelling explanations
report/            existing academic report artifacts
```

There is no separate implementation project or phase directory. Feature modules share schemas, variable definitions, configuration utilities, provenance, and the root runner.

See [architecture](docs/ARCHITECTURE.md), [NLP benchmark protocol](docs/NLP_PIPELINE.md), [BRFSS mappings](docs/BRFSS_DATASET.md), [reasoning and review commands](docs/MODELLING.md).


## Review fixes and evaluation artifacts

The integrated benchmark now writes [per-split error analysis](results/benchmark/error_analysis.json), with full direction/polarity diagnostics only for extractors that actually predict them. The shared `RelationEvaluation` schema makes supported-class macro F1 primary; historical `macro_f1` fields retain their three-class meaning for compatibility.

The current graph is consistently configured under `results/graph/current/`. Reconstruction lives in `src/graph/pipeline.py`, with root-script compatibility imports preserved. BRFSS provenance paths are project-relative; external files are referenced by basename plus hash rather than a user's machine directory.

Generated viewers use a packaged `viewer.js` and inert, escaped JSON data. Abstract markup is removed only for display; frozen texts and entity spans are unchanged.

Dependency policy: the numerical stack is validated with NumPy 1.26, so the <2 cap remains until a clean NumPy 2 environment passes the suite. pgmpy 1.0.0 is pinned to the tested CPD/inference API. Hugging Face Hub is a direct dependency because checkpoint preparation imports it directly.
