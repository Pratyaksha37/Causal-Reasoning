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
8. [Academic Project Report & PDF](#academic-project-report--pdf)
9. [Phase 2 Boundary & Handoff Artifacts](#phase-2-boundary--handoff-artifacts)
10. [Dataset Citations](#dataset-citations)

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
Copy the template and add your Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_MODEL=gemini-3.5-flash
LOG_LEVEL=INFO
```

---

## Running the Pipeline

### Running the Pipeline

#### 1. Full-Corpus Processing (All 1,143 Non-Self Pairs)
To process all 1,143 non-self candidate pairs across all 453 documents in the frozen corpus:
```bash
./run.sh --max-pairs all
```
*(Alternatively: `.venv/bin/python run_pipeline.py --max-pairs all`)*

You can also use reproducible sampling strategies for subset evaluation:
```bash
# Seeded stratified sampling across distinct concept pairs
./run.sh --max-pairs 250 --sample-strategy stratified

# Seeded random sampling across the full corpus
./run.sh --max-pairs 250 --sample-strategy random
```

#### 2. Offline Graph Reconstruction (Zero API Calls)
To reconstruct the evidence graph, aggregate evidence, apply cycle repair, and render figures completely offline from existing saved predictions without invoking external APIs:
```bash
./run.sh --from-saved-predictions
```
*(Alternatively: `.venv/bin/python run_pipeline.py --from-saved-predictions`)*

> [!NOTE]
> By default, the evidence graph is built strictly from real literature corpus candidates (`data/candidates_corpus.jsonl`). Synthetic evaluation fixtures (`data/fixtures/synthetic_candidates.jsonl`) are used only for calculating intrinsic F1/direction metrics. To include synthetic fixtures in the graph for debugging, pass the `--include-synthetic` flag.

To run all automated tests:
```bash
./run.sh test
```

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
- Outputs: `data/candidates_corpus.jsonl` (2,586 pairs total; 1,143 non-self pairs), `results/mapping_report.json`.

#### Step 3: Candidate Sampling & Annotation Tooling (M3)
Samples 250 diverse non-self candidate pairs with document-level train/dev/test holdouts (seed 42) and exports CSVs for human annotation:
```bash
python -m src.annotation.sampler
python -c '
from src.annotation.exporter import export_candidates_to_csv
export_candidates_to_csv("data/annotations/sample.jsonl", "data/annotations/annotator_1_to_label.csv", "annotator_1")
export_candidates_to_csv("data/annotations/sample.jsonl", "data/annotations/annotator_2_to_label.csv", "annotator_2")
'
```
*Note: All sampled candidates are explicitly flagged with `review_status: "NEEDS_EXPERT_ANNOTATION"` and `human_reviewed: false`. See [docs/ANNOTATION_PROTOCOL.md](docs/ANNOTATION_PROTOCOL.md) for full annotation guidelines.*

#### Step 4: Inter-Annotator Agreement (Cohen's Kappa)
Calculates chance-corrected agreement across the 50 double-annotated pairs once independent human annotators submit their labels:
```bash
python -c '
from src.annotation.importer import import_annotations_from_csv
from src.annotation.kappa import compute_agreement, save_agreement_report

a1 = import_annotations_from_csv("data/annotations/annotator_1_to_label.csv", allow_unlabeled=True)
a2 = import_annotations_from_csv("data/annotations/annotator_2_to_label.csv", allow_unlabeled=True)
report = compute_agreement(a1, a2)
save_agreement_report(report, "results/tables/inter_annotator_agreement.json")
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
| **Extractor B: KeywordHeuristic** | Keyword heuristic with entity markers | Fast baseline extractor using `[E1]...[/E1]` and `[E2]...[/E2]` entity marker delimiters and keyword triggers. *(Baseline extractor; does not load trained neural weights)*. |
| **Extractor C: Prompted LLM (Gemini)** | Structured JSON via Gemini REST API | Structured JSON mode, persistent file-backed SHA-256 local cache (`data/cache/llm_cache.jsonl`), automatic retry pacing, and cache hit metrics. |

### Real Literature Benchmark Evaluation (N=250 Candidate Pairs)

> [!IMPORTANT]
> **Annotation Provenance Notice:** The 250 evaluation pairs in `data/annotations/sample.jsonl` are currently labeled via **provisional rule-based heuristic pseudo-labels** (`PROVISIONAL_RULE_PSEUDO_GOLD`, `human_reviewed: false`). Formal clinical expert adjudication has not yet taken place; an unannotated human review queue prioritizing diverse candidates is maintained in `data/annotations/human_annotation_queue.jsonl`.
>
> **Model Provenance Notice:** Extractor B is a lightweight keyword heuristic baseline operating on entity-marker tokenized text. No fine-tuned neural Transformer weights are currently evaluated; fine-tuned DistilBERT inference remains future work.

| Extractor | Macro-F1 | Causal-F1 (N=4) | Assoc-F1 (N=85) | None-F1 (N=161) | Direction Acc (Causal) | Polarity Acc | Lang Status Acc | N | Benchmark Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **spaCy-Rules** | **`0.9316`** | **`1.0000`** | **`0.8608`** | **`0.9341`** | **`1.0000`** | **`0.7753`** | `0.8760` | 250 | `PROVISIONAL_RULE_PSEUDO_GOLD` |
| **KeywordHeuristic** | `0.6012` | `0.2000` | `0.7412` | `0.8625` | `1.0000` | `0.6404` | **`0.9000`** | 250 | `PROVISIONAL_RULE_PSEUDO_GOLD` |

*Artifacts: `results/tables/intrinsic_evaluation.csv`, `results/metrics/error_analysis.json`, `results/figures/confusion_matrix_*.png`.*

### Document-Holdout Split Evaluation (Zero Leakage)

Abstracts were partitioned into document-level splits (`data/annotations/doc_splits.json`, seed 42) to guarantee zero leakage between train, dev, and test:

| Split | Extractor | Macro-F1 | Causal-F1 | Causal Support | Assoc-F1 | None-F1 | Polarity Acc | Split Support | Causal Generalization Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Test (Holdout)** | **spaCy-Rules** | **`0.5857`** | `0.0000` | 0 | **`0.8276`** | **`0.9296`** | **`0.5882`** | 50 | **UNMEASURED (0 gold causal examples)** |
| Test (Holdout) | KeywordHeuristic | `0.5181` | `0.0000` | 0 | `0.7059` | `0.8485` | `0.5882` | 50 | **UNMEASURED (0 gold causal examples)** |
| **Dev (Tuning)** | **spaCy-Rules** | **`0.6263`** | `0.0000` | 0 | **`0.9333`** | **`0.9455`** | **`0.9130`** | 50 | **UNMEASURED (0 gold causal examples)** |
| Dev (Tuning) | KeywordHeuristic | `0.5506` | `0.0000` | 0 | `0.8000` | `0.8519` | `0.6522` | 50 | **UNMEASURED (0 gold causal examples)** |
| **Train** | **spaCy-Rules** | **`0.9220`** | **`1.0000`** | 4 | **`0.8333`** | **`0.9327`** | **`0.7755`** | 150 | MEASURED |
| Train | KeywordHeuristic | `0.6058` | `0.2222` | 4 | `0.7253` | `0.8700` | `0.6531` | 150 | MEASURED |

> [!WARNING]
> **Causal Generalization Honesty:** Because the held-out Dev and Test splits naturally contain 0 causal examples (reflecting extreme real-world biomedical literature class imbalance), causal generalization performance on held-out literature is **strictly unmeasured**. Macro-F1 on the test split is mathematically bounded by non-causal classes. It is scientifically dishonest to claim high causal F1 on held-out data until human-adjudicated causal test instances are added. Detailed split table saved in `results/tables/literature_evaluation_splits.csv`.

### Synthetic Pipeline-Consistency Check (Synthetic Agreement)

> [!NOTE]
> The agreement metric below is a **synthetic pipeline-consistency sanity check** between programmatic rule passes, NOT human inter-annotator agreement:
- **Agreement Type:** `SYNTHETIC_PIPELINE_CONSISTENCY_CHECK` (`is_human_annotated: false`)
- **Synthetic Cohen's Kappa ($\kappa$):** **`0.8816`** (pipeline consistency check)
- **Observed Agreement Rate:** **`95.74%`** (45/47 matches)
- **True Human Agreement:** Pending completion of dual human review using `data/annotations/human_annotation_queue.csv`.
- **Report Location:** Saved in `results/tables/inter_annotator_agreement.json`

### Unit Test Synthetic Fixture (10 Synthetic Candidate Pairs)

| Extractor | Macro-F1 | Causal-F1 | Assoc-F1 | None-F1 | Direction Acc (Causal) | Polarity Acc | Lang Status Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **spaCy-Rules** | `0.9030` | `0.9091` | `1.0000` | `0.8000` | `1.0000` | `0.8750` | `1.0000` |
| **KeywordHeuristic** | `0.9030` | `0.9091` | `1.0000` | `0.8000` | `1.0000` | `0.8750` | `1.0000` |
| **Prompted-LLM-Gemini** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | `0.6250` | `1.0000` |

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

### 1. Dual-Graph Architecture & Exports
The pipeline produces two complementary representations of biomedical evidence:
- **Causal DAG (`results/figures/causal_dag.png` & `results/graph/evidence_graph_after_repair.graphml`):**  
  Contains strictly validated causal assertions (directed edges). Must be a strict Directed Acyclic Graph (DAG) for Phase 2 Bayesian Network initialization.
- **Multi-Relational Evidence Graph (`results/figures/evidence_graph_after_repair.png` & `results/graph/combined_evidence_graph.graphml`):**  
  Visualizes the full literature network across all 9 canonical concepts, rendering solid arrows for causal claims and dashed lines for empirical observational associations (22+ associations).
- **Association Graph Export (`results/graph/association_graph.graphml`):**  
  Dedicated GraphML containing all observational associations with support and citation metadata.
- **Provenance Metadata (`results/graph/provenance.json`):**  
  Full evidence provenance detailing unique PubMed papers, unique sentences, and extractor agreement counts per edge.

### 2. Evidence Filtering & Empty DAG Conditions
> [!NOTE]
> An empty causal DAG in a limited candidate sample does **not** imply that the scientific literature lacks causal findings. Slicing candidates by sequential row order can exclude valid causal sentences appearing later in the corpus (e.g., Mendelian Randomization claims in `PMID:41424208` or pediatric etiology in `PMID:42415555`). Furthermore, strict target filters legitimately route false positives (such as Carpal Tunnel Syndrome `CTS`, Intervertebral Disc Degeneration `IVDD`, parallel secular registry trends, or brain morphometry) to observational associations rather than polluting the causal DAG.

### 3. Deterministic Cycle Repair Heuristic (`IterativeMinConfidenceRemoval`)
If any cycles occur among directed causal candidates:
1. Detects all simple cycles (`nx.simple_cycles`) and Strongly Connected Components with $>1$ node.
2. Iteratively removes the edge in each cycle with the **minimum aggregated confidence**.
3. Tie-breaking: lowest unique citation support count, then lexicographical order of `(source, target)`.
4. Logs every removed edge to `results/graph/cycle_repair.log`.

---

## Testing & Quality Assurance

The codebase includes an extensive suite of automated unit, integration, and regression tests:
```bash
# Run all unit, integration, and regression tests
./run.sh test

# Run static type checking
mypy src/ run_pipeline.py
```

---

## Academic Project Report & PDF

A comprehensive 12-page academic publication-style report is prepared and compiled:

| Document | Location | Description |
|---|---|---|
| **Compiled PDF Report** | [`report/project_report.pdf`](report/project_report.pdf) | Fully compiled 12-page report (~1.59 MB) with embedded high-resolution figures, evaluation tables, and bibliography. |
| **Master LaTeX Source** | [`report/project_report.tex`](report/project_report.tex) | Complete LaTeX document formatted according to academic publication standards. |
| **BibTeX Citations** | [`report/references.bib`](report/references.bib) | 11 comprehensive citations covering biomedical NLP, causality, epidemiology, and dataset resources. |
| **Report Guide** | [`report/README.md`](report/README.md) | Compilation guide for Tectonic, TeX Live/pdflatex, and Overleaf. |

To recompile locally at any time:
```bash
tectonic report/project_report.tex
```

---

## Phase 2 Boundary & Handoff Artifacts

Phase 1 produces the evidence graph and variable ontology for Phase 2:

| Artifact Description | Location | Format / Schema |
|---|---|---|
| **Causal DAG (Phase 2 Input)** | `results/graph/evidence_graph_after_repair.graphml` | GraphML, directed acyclic (`is_dag=True`), 2 nodes, 1 validated causal edge (`BMICat -> Outcome`) |
| **Observational Association Graph** | `results/graph/association_graph.graphml` | GraphML, undirected, 10 nodes, 31 association edges |
| **Combined Multi-Relational Graph** | `results/graph/combined_evidence_graph.graphml` | GraphML, multi-relational edge types (`causal` vs `association_only`) |
| **Complete Citation Provenance** | `results/graph/provenance.json` | JSON with PMIDs, sentences, character offsets, support & agreement counts |
| **Deterministic Cycle Repair Log** | `results/graph/cycle_repair.log` | Log of removed cycle edges with confidence, support, and timestamp |
| **Primary Evaluation Benchmark** | `results/tables/intrinsic_evaluation.csv` | CSV reporting Macro-F1, per-class F1, support counts, and benchmark provenance |
| **Document-Holdout Split Table** | `results/tables/literature_evaluation_splits.csv` | CSV breaking down train (150), dev (50), test (50) with class support and causal status |
| **Diagnostic Error Analysis** | `results/metrics/error_analysis.json` | JSON categorizing errors across 5 diagnostic failure categories |
| **Synthetic Consistency Check** | `results/tables/inter_annotator_agreement.json` | Synthetic Kappa ($\kappa=0.8816$), raw agreement, and unannotated human review queue reference |
| **Confusion Matrix Visualizations** | `results/figures/confusion_matrix_*.png` | High-contrast publication-ready PNG plots per extractor |
| **Causal DAG Plot** | `results/figures/causal_dag.png` | Directed graph visualization of accepted causal claims |
| **Multi-Relational Evidence Graph** | `results/figures/evidence_graph_after_repair.png` | Full network visualization with solid causal and dashed association edges |

---

## Dataset Citations

- **Europe PMC:**  
  Europe PMC Consortium. (2025). *Europe PMC: a full-text literature database for the life sciences*. Nucleic Acids Research. https://europepmc.org
- **SemEval-2010 Task 8:**  
  Hendrickx, I., et al. (2010). *SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals*. Proceedings of SemEval-2010.
- **scispaCy / spaCy:**  
  Neumann, M., et al. (2019). *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing*. Proceedings of BIONLP 2019.