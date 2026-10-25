# Expert Biomedical Relation Annotation Protocol
## Neuro-Symbolic Causal Inference for Diabetes Health Evidence (Phase 1)

> [!IMPORTANT]
> **Adherence Mandate:** Annotate exclusively what the authors assert in the explicit sentence text. Do not project unstated biological assumptions, downstream disease pathology, or external medical knowledge into the sentence labels.

---

## 1. Overview & Annotation Architecture

This protocol governs the human expert adjudication of candidate entity pairs extracted from the 500-abstract diabetes literature corpus (`data/manifest/corpus_manifest.jsonl`).

Each annotation record in [`data/annotations/sample.jsonl`](file:///Users/pratyaksha_003/Desktop/Causal/data/annotations/sample.jsonl) presents:
- **`sentence_text`**: The unedited scientific sentence from PubMed.
- **`entity_1`**: Mention text, character span, and canonical variable (`AgeCat`, `Sex`, `BMICat`, `PhysActivity`, `Smoker`, `Hypertension`, `InsulinResistance`, `SleepDuration`, `Outcome`, or `unmapped`).
- **`entity_2`**: Mention text, character span, and canonical variable.

For every candidate pair, annotators independently assign four categorical variables:
1. `relation_class` (`causal` | `association_only` | `none`)
2. `direction` (`entity_1->entity_2` | `entity_2->entity_1` | `not_applicable`)
3. `polarity` (`increases` | `decreases` | `mixed_or_unclear` | `not_applicable`)
4. `language_status` (`affirmed` | `negated` | `uncertain`)

---

## 2. Decision Trees & Class Definitions

### A. Relation Class (`relation_class`)

```
                                  [Candidate Pair]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ Does sentence assert a functional or          │
                 │ statistical relationship between E1 and E2?   │
                 └───────────────────────┬───────────────────────┘
                                         │
                       ┌─────────────────┴─────────────────┐
                      NO                                  YES
                       │                                   │
                [relation: none]                           │
             [dir: not_applicable]                         │
            [polar: not_applicable]                        │
                                         ┌─────────────────┴─────────────────┐
                                         │ Direct mechanism, intervention,   │
                                         │ Mendelian Randomization, or       │
                                         │ longitudinal causal assertion?    │
                                         └─────────────────┬─────────────────┘
                                                           │
                                         ┌─────────────────┴─────────────────┐
                                        NO                                  YES
                                         │                                   │
                             [relation: association_only]           [relation: causal]
                             - "is associated with"                 - "causes", "leads to"
                             - "correlated with"                    - "reduces risk of"
                             - "linked to", "predicts"              - "dominant causal link"
```

1. **`causal`**: The authors state that one entity directly alters, induces, causes, triggers, prevents, or modifies the risk of the second entity.
   - *Key triggers:* `"causes"`, `"leads to"`, `"induces"`, `"prevents"`, `"reduces the risk of"`, `"increases the risk of"`, `"results in"`, `"dominant causal link"`, `"protects against"`.
   - *Mendelian Randomization:* Sentences summarizing Mendelian Randomization analyses that identify an exposure as a causal determinant of an outcome are classified as `causal`.
   - *Exclusion - Unrelated Endpoints:* If the sentence asserts causality for an unrelated endpoint (e.g. Carpal Tunnel Syndrome `CTS`, Intervertebral Disc Degeneration `IVDD`, or dementia) and diabetes is merely a comorbidity, classify the relation between diabetes and the exposure as `association_only` or `none`.
   - *Exclusion - Case Reports:* Individual case reports describing a single patient developing an outcome (`"we report the case of..."`) must not be classified as general population `causal` claims.

2. **`association_only`**: The authors report statistical correlation, cross-sectional co-occurrence, regression association, or observational predictive value without asserting causal direction or mechanism.
   - *Key triggers:* `"is associated with"`, `"correlated with"`, `"linked to"`, `"related to"`, `"predicts"`.

3. **`none`**: No functional or statistical relationship is asserted between the two entities in this sentence.
   - *Listing demographics:* *"We recruited 400 women and assessed type 2 diabetes."* $\rightarrow$ `none`
   - *Parallel comorbidity listing:* *"Obesity and hypertension were both recorded at baseline."* $\rightarrow$ `none`

---

### B. Direction (`direction`)
- **`entity_1->entity_2`**: Entity 1 is asserted as the exposure/source/cause and Entity 2 is the outcome/target/effect.
- **`entity_2->entity_1`**: Entity 2 is asserted as the exposure/source/cause and Entity 1 is the outcome/target/effect (e.g. passive construction: *"Incident T2DM is caused by severe obesity"*).
- **`not_applicable`**: Mandatory for `none`, and standard for symmetric observational associations where no exposure-outcome arrow is defined.

---

### C. Polarity (`polarity`)
- **`increases`**: The exposure increases the risk, probability, or physiological magnitude of the target.
- **`decreases`**: The exposure decreases, prevents, or protects against the target.
- **`mixed_or_unclear`**: Non-linear curves (e.g. U-shaped), conflicting findings, or unspecified polarity.
- **`not_applicable`**: Mandatory for `none`.

> [!NOTE]
> **Inactivity Polarity Convention:** Annotate what the raw text literally asserts. If the text says *"physical inactivity increases diabetes risk"*, record `increases`. The downstream pipeline automatically applies `polarity_flip: true` when mapping to `PhysActivity=1`.

---

### D. Language Status (`language_status`)
- **`affirmed`**: Direct assertion in the indicative mood without negation or hedging.
- **`negated`**: Explicit syntactic or lexical negation (*"was not associated with"*, *"failed to reduce"*).
- **`uncertain`**: Hedged, speculative, or probabilistic assertion (*"may contribute to"*, *"suggests a potential link"*, *"could reduce"*).

---

## 3. Double-Annotation & Cohen's Kappa Adjudication Protocol

1. **Independent Labeling:**
   - Annotator 1 receives [`data/annotations/annotator_1_to_label.csv`](file:///Users/pratyaksha_003/Desktop/Causal/data/annotations/annotator_1_to_label.csv) (all 250 candidate pairs).
   - Annotator 2 receives [`data/annotations/annotator_2_to_label.csv`](file:///Users/pratyaksha_003/Desktop/Causal/data/annotations/annotator_2_to_label.csv) (the 47 double-annotated pairs).
   - Neither annotator may communicate or view model predictions during labeling.
2. **Inter-Annotator Agreement Computation:**
   - Run agreement calculation:
     ```bash
     .venv/bin/python -c '
     from src.annotation.importer import import_annotations_from_csv
     from src.annotation.kappa import compute_agreement, save_agreement_report

     a1 = import_annotations_from_csv("data/annotations/annotator_1_to_label.csv")
     a2 = import_annotations_from_csv("data/annotations/annotator_2_to_label.csv")
     report = compute_agreement(a1, a2)
     save_agreement_report(report, "results/tables/inter_annotator_agreement.json")
     print("Cohen Kappa:", report["relation_class_kappa"])
     '
     ```
   - Target benchmark quality standard: Cohen's Kappa $\kappa \ge 0.70$.
3. **Adjudication & Final Benchmark Export:**
   - Any pair with disagreement on `relation_class`, `direction`, or `polarity` is logged in `results/tables/inter_annotator_agreement.json` and resolved by a senior epidemiologist.
   - Only after adjudication is complete are the final held-out benchmark splits (`dev` and `test`) considered finalized.
