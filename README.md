# Neuro-Symbolic Causal Inference for Diabetes Health Evidence
### Biomedical Relation Extraction, Evaluation, and Evidence Graph Construction

## 📖 Table of Contents
1. [Overview & Research Mission](#overview--research-mission)
2. [Quickstart & Environment Setup](#quickstart--environment-setup)
3. [Running the Pipeline](#running-the-pipeline)
4. [The Three Relation Extractors](#the-three-relation-extractors)
5. [CDC Variable Mapping & Precision Rules](#cdc-variable-mapping--precision-rules)
6. [Evidence Graph & Deterministic Cycle Repair](#evidence-graph--deterministic-cycle-repair)
7. [Testing & Quality Assurance](#testing--quality-assurance)
8. [Phase 2 Boundary & Handoff Artifacts](#phase-2-boundary--handoff-artifacts)
9. [Dataset Citations](#dataset-citations)

---

## Overview & Research Mission

This repository contains **Phase 1** of a two-phase research initiative:
- **Phase 1 (In Scope):** Retrieve a frozen corpus of scientific abstracts from Europe PMC, extract asserted causal and associational relationships using three diverse NLP extractors, evaluate them intrinsically and end-to-end, and build a directed **evidence graph** with full provenance and deterministic cycle repair.
- **Phase 2 (Out of Scope):** Bayesian Network structure learning, fitting Conditional Probability Distributions (CPDs) using CDC survey data (BRFSS/NHANES), and executing do-calculus intervention queries. Phase 1 produces the clean, typed graph export that directly feeds Phase 2.

---

## Quickstart & Environment Setup

### 1. Prerequisites
- **Python 3.11** (Homebrew: `brew install python@3.11`)

### 2. Create and Activate Virtual Environment
```bash
# Clone the repository and navigate into it
cd Causal

# Create Python 3.11 virtual environment
/opt/homebrew/bin/python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

> [!TIP]
> Always ensure your virtual environment is active (`source .venv/bin/activate`) so that `python` points to `.venv/bin/python` where all dependencies (`spacy`, `torch`, `transformers`) are installed. Alternatively, prefix commands with `.venv/bin/python`.

### 3. Install Package & Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

### 4. Configure API Key
Copy the template and add your Groq API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b
LOG_LEVEL=INFO
```

---

## Running the Pipeline

### One-Command Full Pipeline Execution
To run all stages end-to-end (corpus loading, 3 extractors, benchmark evaluation, graph assembly, cycle repair, and visualization):
```bash
python run_pipeline.py
```
*(Or directly: `.venv/bin/python run_pipeline.py`)*

---

### Step-by-Step Individual Commands

#### Step 1: Europe PMC Corpus Retrieval & Frozen Mode (M1)
The corpus of 500 diabetes abstracts is permanently frozen in `data/manifest/corpus_manifest.jsonl`.
To verify hashes in strict frozen mode (network access blocked):
```bash
python -m src.retrieval.corpus --config configs/retrieval.yaml
```

#### Step 2: Sentence Splitting & Candidate Generation (M2)
Extracts mentions, applies alias mapping with CDC precision rules, and generates candidate pairs across the corpus:
```bash
python -m src.text.process_corpus
```
- Outputs: `data/candidates_corpus.jsonl` (2,586 pairs), `results/mapping_report.json`.

#### Step 3: Candidate Sampling & Annotation Tooling (M3)
Samples 250 pairs with document-level train/dev/test holdouts (seed 42) and exports CSVs for human annotation:
```bash
python -m src.annotation.sampler
python -c '
from src.annotation.exporter import export_candidates_to_csv
export_candidates_to_csv("data/annotations/sample.jsonl", "data/annotations/annotator_1_to_label.csv", "annotator_1")
export_candidates_to_csv("data/annotations/sample.jsonl", "data/annotations/annotator_2_to_label.csv", "annotator_2")
'
```

#### Step 4: Inter-Annotator Agreement (Cohen's Kappa)
Calculates chance-corrected agreement across the double-annotated subset:
```bash
python -c '
from src.annotation.importer import import_annotations_from_csv
from src.annotation.kappa import compute_agreement, save_agreement_report

a1 = import_annotations_from_csv("data/annotations/annotator_1_to_label.csv", allow_unlabeled=True)
a2 = import_annotations_from_csv("data/annotations/annotator_2_to_label.csv", allow_unlabeled=True)
report = compute_agreement(a1, a2)
print("Cohen Kappa:", report["relation_class_kappa"])
'
```

---

## The Three Relation Extractors

All three extractors implement a common, typed abstract interface:
`BaseExtractor.predict(list[CandidatePair]) -> list[Prediction]`

| Extractor | Architecture | Key Characteristics |
|---|---|---|
| **Extractor A: spaCy Rules** | Lexical triggers + dependency path heuristics | Fully deterministic, inspectable, fast. Accurately handles active/passive syntax, lexical negation (`not`, `fails to`), and hedging (`may`, `suggests`). |
| **Extractor B: DistilBERT** | Sequence classification with entity markers | Uses `[E1]...[/E1]` and `[E2]...[/E2]` delimiter tokens. Hardware accelerated (`mps` on Apple Silicon / `cuda`). |
| **Extractor C: Prompted LLM (Groq)** | `openai/gpt-oss-120b` via Groq REST API | Temperature 0.0, seed 42, structured JSON object mode, persistent file-backed SHA-256 cache (`data/cache/llm_cache.jsonl`), and rate limit retries. |

### Intrinsic Benchmark Results

| Extractor | Macro-F1 | Causal-F1 | Assoc-F1 | None-F1 | Direction Acc (Causal) | Polarity Acc | Lang Status Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **spaCy-Rules** | `0.9030` | `0.9091` | `1.0000` | `0.8000` | `1.0000` | `0.8750` | `1.0000` |
| **FineTuned-DistilBERT** | `0.9030` | `0.9091` | `1.0000` | `0.8000` | `1.0000` | `0.8750` | `1.0000` |
| **Prompted-LLM-Groq** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | `0.6250` | `1.0000` |

---

## CDC Variable Mapping & Precision Rules

Literature terms are linked to the **8 canonical Phase 2 variables** in `configs/aliases.yaml`:
1. `AgeCat`, `Sex`, `Education`, `Income`, `Smoker`, `PhysActivity`, `BMICat`, `Outcome`.

### Critical Mapping Mechanics
- **Polarity Flipping on Inactivity:**  
  *"Physical inactivity increases T2DM risk"* ($\text{PhysActivity}=0 \xrightarrow{\text{increases}} \text{Outcome}$).  
  When mapped to the CDC exposure ($\text{PhysActivity}=1$), the normalizer flips the edge: $\text{PhysActivity}=1 \xrightarrow{\text{decreases}} \text{Outcome}$.
- **Strict Smoker 100-Cigarette Criterion:**  
  Only explicit lifetime 100+ cigarette mentions map to `Smoker=1` (or `never smoked` to `0`). General mentions (`smoking`, `tobacco`, `smoker`, `smoking cessation`) strictly default to **`unmapped`** to prevent data drift against CDC definitions.
- **Obesity Classification:**  
  `obesity / BMI >= 30` maps to highest `BMICat` (`obese`), cleanly distinguished from clinical BMI values.
- **Outcome Precision:**  
  `type 2 diabetes` / `prediabetes` $\rightarrow$ `Outcome=1`; `no diabetes` / `normoglycemic` $\rightarrow$ `Outcome=0`.
- **Unmapped Fallback:**  
  Phrases that do not match closely default to `unmapped`. Fuzzy matches are never forced.

---

## Evidence Graph & Deterministic Cycle Repair

1. **NetworkX DiGraph Construction:**
   - Nodes represent canonical variables or open concepts.
   - Edges only accept `causal` relations. Association-only claims are stored in a separate undirected store and never silently promoted to causal edges.
   - Edges record support count, mean aggregated confidence, extractor names, and list of supporting sentences with character offsets.
2. **Cycle Detection:**
   - Detects all simple cycles (`nx.simple_cycles`) and Strongly Connected Components with $>1$ node.
3. **Deterministic Cycle Repair Heuristic (`IterativeMinConfidenceRemoval`):**
   - Iteratively removes the edge in each cycle with the **minimum aggregated confidence**.
   - Tie-breaking: lowest support count (fewer citations), then lexicographical order of `(source, target)`.
   - Logs every removed edge to `results/graph/cycle_repair.log`.
4. **Export:**
   - Generates `results/graph/evidence_graph_after_repair.graphml` and `results/graph/provenance.json` for Phase 2.
   - Generates visual plot: `results/figures/evidence_graph_after_repair.png`.

---

## Testing & Quality Assurance

The codebase includes 39 comprehensive unit and integration tests:
```bash
# Run all unit and integration tests
pytest -v

# Run static type checking
mypy src/ run_pipeline.py

# Run code style and lint checks
ruff check .
```

---

## Dataset Citations

- **Europe PMC:**  
  Europe PMC Consortium. (2025). *Europe PMC: a full-text literature database for the life sciences*. Nucleic Acids Research. https://europepmc.org
- **SemEval-2010 Task 8:**  
  Hendrickx, I., et al. (2010). *SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals*. Proceedings of SemEval-2010.
- **scispaCy / spaCy:**  
  Neumann, M., et al. (2019). *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing*. Proceedings of BIONLP 2019.
