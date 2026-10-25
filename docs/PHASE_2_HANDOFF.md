# Phase 2 Handoff Specification: Bayesian Network & Intervention Modeling

**Project:** Neuro-Symbolic Causal Inference for Diabetes-Related Health Evidence  
**Author:** Phase 1 Engineering & Audit Team  
**Date:** October 2026  
**Status:** Approved for Phase 2 Prototyping (Safe Boundary Mode)  

---

## 1. Objective of Phase 2

Phase 2 builds upon Phase 1 by combining the **biomedical literature evidence structure** with large-scale observational population data (**CDC Diabetes Health Indicators / BRFSS**) to construct Bayesian Networks, fit Conditional Probability Distributions (CPDs), and perform Pearlian causal interventional queries ($P(\text{Outcome} \mid do(X = x))$).

> [!CAUTION]
> **Fundamental Safety Rule for Phase 2:**  
> The literature evidence graph records *claims published in scientific literature*. It must **never** be consumed as unassailable biological ground truth. Phase 2 must explicitly differentiate empirical statistical association observed in CDC microdata from true interventional causal claims supported by external experimental or Mendelian Randomization literature evidence.

---

## 2. Consumable Phase 1 Artifacts

Phase 2 may safely consume the following verified Phase 1 outputs:

| Artifact | Location | Schema / Format | Permitted Phase 2 Usage |
|---|---|---|---|
| **Causal Evidence DAG** | `results/graph/evidence_graph_after_repair.graphml` | GraphML, DiGraph (`is_dag=True`), 2 nodes, 1 edge | DAG structural skeleton for initializing Bayesian Network structure (`pgmpy.models.BayesianNetwork`). |
| **Complete Evidence Provenance** | `results/graph/provenance.json` | JSON, structured edge-level literature citations | Audit trace justifying why structural edges exist (used for sensitivity analyses and reporting). |
| **Observational Association Graph** | `results/graph/association_graph.graphml` | GraphML, Graph, 10 nodes, 31 undirected edges | Feature selection, correlation benchmarks, and latent confounding hypotheses. |
| **Variable Alias Ontology** | `configs/aliases.yaml` | YAML dictionary | Standardizes column mappings and value binning between CDC BRFSS variables and literature concepts. |
| **Offline Graph Reconstruction** | `./run.sh --from-saved-predictions` | Bash script | Deterministically regenerates Phase 1 graph artifacts from local saved predictions in <4s without network access. |

---

## 3. Provisional Artifacts Requiring Caution

| Artifact | Location | Provisional Nature | Precaution for Phase 2 |
|---|---|---|---|
| **Evaluation Benchmark** | `data/annotations/sample.jsonl` | Labeled via rule heuristics (`PROVISIONAL_RULE_PSEUDO_GOLD`), `human_reviewed: false`. | Use strictly for internal regression testing. Do **not** cite these Macro-F1 scores as verified human benchmark accuracy in academic publications. |
| **Synthetic Consistency Report** | `results/tables/inter_annotator_agreement.json` | Programmatic perturbation check ($\kappa = 0.8816$), `is_human_annotated: false`. | Do not cite as human inter-rater reliability. Awaiting genuine dual-human annotation using `data/annotations/human_annotation_queue.csv`. |
| **Pilot Prompted LLM Predictions** | `results/predictions/prompted-llm-*_predictions.jsonl` | Evaluated across a 20-pair pilot sample. | Do not treat pilot LLM predictions as comprehensive corpus extractions. |

---

## 4. Edge-Level Inclusion & Exclusion Status

### A. Accepted Causal Edge for Phase 2 DAG
- **`BMICat -> Outcome`**
  - **Relation:** Causal, polarity: `increases` ($\text{BMICat}=\text{obese} \xrightarrow{\text{increases}} \text{Outcome}=1$).
  - **Supporting Papers:** 2 unique, independent high-impact studies:
    1. `PMID:41424208:s12` — Mendelian Randomization (MR) analysis establishing abdominal obesity as the dominant causal driver of T2DM.
    2. `PMID:42415555:s01` — Pediatric longitudinal etiology demonstrating obesity leading to dysglycemia and T2DM.
  - **Recommendation:** Safe for initial exploratory Bayesian Network modeling.

### B. Excluded Edges (Do NOT Include in Causal DAG)
1. **`InsulinResistance -> Outcome`:**  
   - *Reason:* Current candidate sentences represent dementia risk among pre-existing diabetes cohorts (`PMID:42828458:s04`) or biological mediation of dietary quality (`PMID:42901975:s03`). Neither establishes direct interventional exposure on diabetes in the current corpus. Preserved in the observational association graph (31 evidence sentences).
2. **Reverse Causation (`Outcome -> Any Exposure`):**  
   - *Reason:* T2DM diagnosis does not retroactively alter baseline genetics or past demographic history.
3. **Demographic Targets (`Any -> Sex` or `Any -> AgeCat`):**  
   - *Reason:* Biological sex and chronological age are exogenous demographic strata.

---

## 5. Mandatory Methodological Assumptions for Phase 2

When fitting Bayesian Networks on the CDC Diabetes Health Indicators dataset, Phase 2 must explicitly state the following assumptions:

1. **Causal Sufficiency Assumption:**  
   Standard Bayesian Network parameter estimation assumes no unobserved confounders (e.g., unmeasured genetic variants, early childhood socioeconomic conditions) that drive both exposure and outcome. In observational health survey data (BRFSS), this assumption is almost certainly violated.
2. **Faithfulness & Markov Condition:**  
   Observed conditional independencies in the data are assumed to reflect the causal DAG structure.
3. **Positivity:**  
   Every demographic subgroup must have a non-zero probability of receiving any lifestyle/BMI exposure state.
4. **Distinction Between Observational and Interventional Distributions:**  
   - Observational conditioning: $P(\text{Outcome} = 1 \mid \text{BMICat} = \text{obese})$ (predictive correlation).
   - Pearl's do-operator: $P(\text{Outcome} = 1 \mid do(\text{BMICat} = \text{normal}))$ (expected outcome under an intervention).  
   Without an unconfounded graph, observational conditional probabilities cannot be interpreted as the effect of an intervention.

---

## 6. What Phase 2 Can and Cannot Conclude

### What Phase 2 CAN Test Computationally:
- Fit discrete Conditional Probability Distributions (CPDs) on CDC survey microdata using Maximum Likelihood or Bayesian Dirichlet estimators (BDeu).
- Compare log-likelihood, BIC, and AIC scores between literature-constrained DAG structures vs purely data-driven discovery structures (e.g. PC algorithm, Hill Climbing).
- Perform simulated $do$-interventions under explicitly stated structural assumptions and quantify estimated risk differences.
- Conduct structural sensitivity analysis by testing how interventional estimates shift when candidate literature edges (e.g., adding `PhysActivity -> BMICat`) are added or removed.

### What Phase 2 CANNOT Conclude From Observational Data Alone:
- Cannot prove that an observed risk difference is free from residual confounding.
- Cannot establish clinical treatment efficacy or replace randomized controlled clinical trials (RCTs).
- Cannot provide personalized clinical recommendations for individual patients.

---

## 7. Baseline Reproduction Commands for Phase 2

To reproduce the exact Phase 1 inputs before running Phase 2 code:

```bash
# 1. Verify virtual environment and run the complete test suite (68 tests)
./run.sh test

# 2. Reconstruct the evidence graph offline from verified saved predictions (zero API calls)
./run.sh --from-saved-predictions

# 3. Verify generated Phase 2 GraphML input
ls -lh results/graph/evidence_graph_after_repair.graphml
ls -lh results/graph/provenance.json
```
