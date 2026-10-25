# Methodological Limitations and Remaining Work

**Project:** Neuro-Symbolic Causal Inference for Diabetes-Related Health Evidence  
**Phase:** 1 Baseline Finalization  
**Date:** October 2026  
**Status:** Clean Baseline Established; Scientific Caveats Disclosed  

---

## 1. Executive Summary of Limitations

Phase 1 establishes a reproducible, deterministic, offline-capable evidence extraction and graph construction pipeline. However, several critical scientific and methodological boundaries must be explicitly understood before these outputs are interpreted or consumed by Phase 2:

1. **Literature Claims $\neq$ Absolute Biological Truth:**  
   The evidence graph captures relationships *asserted in published biomedical abstracts*. Published claims may be subject to publication bias, residual confounding, observational bias, or domain-specific cohort constraints.
2. **Provisional Pseudo-Labels $\neq$ Clinical Gold Truth:**  
   The current 250-pair benchmark (`data/annotations/sample.jsonl`) is labeled using rule-based heuristics (`PROVISIONAL_RULE_PSEUDO_GOLD`, `human_reviewed: false`). Formal double-blind human clinical adjudication has not yet occurred.
3. **Causal Test Generalization is Strictly Unmeasured:**  
   The held-out 50-pair test split contains **zero gold causal instances** (33 none, 17 association), reflecting real-world biomedical class imbalance. Causal generalization on unseen literature is mathematically unmeasured.
4. **Extractor B is a Heuristic Baseline, Not a Fine-Tuned Transformer:**  
   `KeywordHeuristicExtractor` uses entity-marker tokenization (`[E1]...[/E1]`, `[E2]...[/E2]`) and regex triggers. No fine-tuned Transformer weights are currently evaluated; genuine neural fine-tuning is future work.
5. **Single Validated Causal Edge in Evidence DAG:**  
   After rigorous epidemiological filtering of secondary endpoints (dementia risk, CTS, IVDD) and biological mediators, the causal DAG contains exactly one validated edge: `BMICat -> Outcome` (supported by Mendelian Randomization and longitudinal pediatric etiology). All other empirical evidence is preserved as observational associations (31 edges).

---

## 2. Detailed Dimension Breakdown

### A. Literature Sourcing & Corpus Boundaries
- **Corpus Size:** Exactly 500 abstracts retrieved from Europe PMC and frozen into `data/manifest/corpus_manifest.jsonl` with SHA-256 hashes.
- **Coverage Gap:** Full-text articles were not retrieved; abstracts may omit negative controls, detailed covariate adjustments, or non-significant exploratory findings.
- **Query Bias:** The search query focused on Type 2 Diabetes and cardiometabolic comorbidities, which skews the candidate pair distribution toward known risk factors (BMI, hypertension, age) and underrepresents protective lifestyle interventions.

### B. Annotation Benchmark & Provenance
- **Dataset Size:** 250 candidate pairs sampled across 41 concept pairs and 124 papers without document leakage.
- **Current Provenance:** All 250 records are labeled via `rule_heuristic_regex_v1`. They are suitable for unit test regression, pipeline validation, and error analysis, but **must not be published as human gold truth**.
- **Synthetic Inter-Annotator Agreement:** The reported Cohen's Kappa ($\kappa = 0.8816$) reflects programmatic perturbation between rule passes, serving as a pipeline consistency sanity check rather than human agreement.
- **Action Required for Gold Standard:** Complete independent human review using the prioritized queue in `data/annotations/human_annotation_queue.jsonl` (50 diverse candidate pairs) and adjudicate disagreements.

### C. Evaluation & Metric Boundaries
- **Split Distribution:**
  - Train ($N=150$): 101 none, 45 association, 4 causal.
  - Dev ($N=50$): 27 none, 23 association, 0 causal.
  - Test ($N=50$): 33 none, 17 association, 0 causal.
- **Metric Interpretation:**
  - Overall `spaCy-Rules` Macro-F1 across all 250 provisional pairs is `0.9316`.
  - On the held-out test split, Macro-F1 is `0.5857` (Assoc-F1 = `0.8276`, None-F1 = `0.9296`, Causal-F1 = `0.0000`).
  - Because test causal support is 0, test Macro-F1 is mathematically bounded at $\frac{0.8276 + 0.9296 + 0.0}{3} \approx 0.5857$. This must be interpreted as honest metric reporting under zero-support conditions, not poor model behavior on tested classes.

### D. Extractor Implementations
- **spaCy-Rules:** Highly accurate on standard biomedical active/passive assertions and lexical triggers; brittle on inverted syntax, multi-sentence discourse, or complex co-reference.
- **KeywordHeuristic:** Operates as a fast baseline (`0.6012` Macro-F1). Does not utilize deep contextual semantic representations.
- **Prompted LLMs (Gemini / Groq):** Evaluated on a 20-pair pilot sample due to API quotas and financial constraints. Offline reconstruction uses saved cache hits and does not execute live API calls.
- **Fine-Tuned DistilBERT:** Historical file `finetuned-distilbert_predictions.jsonl` was identified as an identical byte-copy of the keyword heuristic predictions. It has been disclaimed, deprecated, and excluded from multi-model comparison tables.

### E. Graph Construction & Evidence Filtering
- **Strict Epidemiological DAG Validity:**
  - Excluded secondary complications and non-target disease outcomes: `PMID:42828458:s04` asserts that insulin resistance increases the risk of *dementia in patients with pre-existing T2DM*. This was rejected from `InsulinResistance -> Outcome` and routed to the association graph.
  - Excluded biological mediators/mechanisms: `PMID:42901975:s03` describes poor dietary quality causing T2DM *through visceral adiposity and insulin resistance*. Insulin resistance is a mediator, not the primary exposure; routed to association graph.
  - Excluded reverse causation: `Outcome -> Exposure` edges are rejected.
  - Excluded demographic immutability: `Sex` and `AgeCat` cannot be causally caused by disease or lifestyle.
- **Final Graph Counts:**
  - **Causal DAG:** 2 nodes (`BMICat`, `Outcome`), 1 directed causal edge (`BMICat -> Outcome`). Strict DAG (`is_dag = True`, 0 cycle removals needed).
  - **Observational Association Graph:** 10 nodes, 31 undirected edges with complete citation and sentence provenance.
  - **Combined Multi-Relational Graph:** 10 nodes, 32 edges (1 causal, 31 association).

---

## 3. Remaining Work Roadmap

| Priority | Task Description | Dependencies | Phase Target |
|:---:|---|---|:---:|
| **P1** | **Human Expert Annotation:** Annotate the 50 prioritized candidates in `data/annotations/human_annotation_queue.jsonl` by 2 independent biomedical reviewers; resolve disagreements; update `sample.jsonl` to genuine human gold. | Domain annotators | Pre-Publication |
| **P2** | **Held-Out Causal Gold Instances:** Specifically enrich the held-out Dev and Test splits with adjudicated causal sentences so that causal generalization F1 can be empirically evaluated. | Annotation protocol | Pre-Publication |
| **P3** | **Neural Transformer Fine-Tuning:** Train a real Bio-DistilBERT or PubMedBERT sequence classification model on SemEval-2010 Task 8 and adapt on biomedical gold pairs; evaluate against rule baseline. | PyTorch, GPU/MPS, Gold annotations | Phase 1 Extension |
| **P4** | **Full-Text Literature Ingestion:** Extend Europe PMC client to fetch open-access XML full-text for comprehensive methods and results section extraction. | Europe PMC BioC API | Future Work |
| **P5** | **Phase 2 Bayesian Network Prototype:** Ingest `BMICat -> Outcome` DAG and Phase 2 alias configs into `pgmpy`; fit CPDs on CDC BRFSS dataset; execute observational vs interventional queries. | CDC BRFSS data, pgmpy | Phase 2 |
