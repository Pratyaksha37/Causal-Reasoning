# Biomedical Relation Annotation Guidelines
## Neuro-Symbolic Causal Inference for Diabetes-Related Health Evidence (Phase 1)

> [!IMPORTANT]
> **Claim Boundary Reminder:** Annotate what the publication asserts in the sentence text. Do not annotate biological ground truth or external knowledge.

---

### 1. Overview
You are annotating candidate pairs of biomedical entities occurring in the same sentence extracted from diabetes-related scientific literature.
Each item presents:
- **Sentence text**
- **Entity 1** (mention text, character span, canonical variable if mapped)
- **Entity 2** (mention text, character span, canonical variable if mapped)

For each candidate pair, you must provide 4 categorical labels:
1. `relation_class` (`causal` | `association_only` | `none`)
2. `direction` (`entity_1->entity_2` | `entity_2->entity_1` | `not_applicable`)
3. `polarity` (`increases` | `decreases` | `mixed_or_unclear` | `not_applicable`)
4. `language_status` (`affirmed` | `negated` | `uncertain`)

---

### 2. Label Definitions and Decision Criteria

#### A. Relation Class (`relation_class`)
- **`causal`**: The authors state that one entity directly causes, induces, leads to, triggers, prevents, or alters the risk or level of the other entity.
  - *Key triggers*: "causes", "leads to", "induces", "prevents", "reduces the risk of", "increases the risk of", "results in", "triggers", "protects against".
  - *Example*: "Physical activity reduces the risk of type 2 diabetes." -> `causal`
  - *Example*: "Obesity triggers chronic low-grade inflammation leading to insulin resistance." -> `causal`

- **`association_only`**: The authors report statistical correlation, association, prediction, cross-sectional relationship, or co-occurrence without claiming direct causal mechanisms.
  - *Key triggers*: "is associated with", "correlated with", "is related to", "predicts", "co-occurs with", "is linked to".
  - *Example*: "Higher BMI was positively associated with fasting glucose." -> `association_only`
  - *Example*: "Sleep deprivation correlates with insulin resistance." -> `association_only`

- **`none`**: No meaningful functional or statistical relationship is asserted between the two entities in this sentence.
  - *Example*: "We recruited 500 women and assessed type 2 diabetes." (merely describes demographics) -> `none`
  - *Example*: "Both obesity and hypertension are common chronic conditions." (listing, no link between them) -> `none`

---

#### B. Direction (`direction`)
- **`entity_1->entity_2`**: Entity 1 is asserted as the exposure/cause/source, and Entity 2 is the outcome/effect/target.
  - *Example*: Entity 1 = "Physical activity", Entity 2 = "T2DM". "Physical activity reduces T2DM." -> `entity_1->entity_2`
- **`entity_2->entity_1`**: Entity 2 is asserted as the exposure/cause, and Entity 1 is the outcome/target.
  - *Example*: Entity 1 = "T2DM", Entity 2 = "obesity". "T2DM risk is markedly heightened by obesity." -> `entity_2->entity_1`
- **`not_applicable`**: Mandatory for `none`. Also standard for symmetric `association_only` where no directional exposure->outcome is claimed.

---

#### C. Polarity (`polarity`)
- **`increases`**: The exposure/source increases the risk, probability, or physiological magnitude of the target.
  - *Example*: "Obesity increases T2DM risk." -> `increases`
  - *Example*: "Physical inactivity elevates incident diabetes." -> `increases` (Note: for inactivity, polarity is 'increases' in raw text, which our normalizer automatically flips for PhysActivity=1).
- **`decreases`**: The exposure/source decreases, prevents, or protects against the target.
  - *Example*: "Regular exercise reduces diabetes risk." -> `decreases`
- **`mixed_or_unclear`**: The evidence reported is non-linear (e.g. U-shaped curve), conflicting, or direction is not specified.
- **`not_applicable`**: Mandatory for `none`.

---

#### D. Language Status (`language_status`)
- **`affirmed`**: Direct affirmative claim in the indicative mood.
  - *Example*: "Brisk walking decreases insulin resistance." -> `affirmed`
- **`negated`**: Explicit syntactic or lexical negation of the relation.
  - *Example*: "Dietary fiber was not associated with reduced diabetes risk." -> `negated`
  - *Example*: "Smoking cessation did not eliminate hypertension." -> `negated`
- **`uncertain`**: Speculative, hedged, or probabilistic language.
  - *Key triggers*: "may", "might", "could", "suggests", "possibly", "potential mechanism".
  - *Example*: "Short sleep duration may contribute to incident diabetes." -> `uncertain`

---

### 3. Special Cases & Precision Boundaries

1. **Inactivity vs Activity (Polarity Normalization)**:
   - When annotating: Record what the raw sentence says. If the text says "physical inactivity increases risk", record `polarity: increases`.
   - The downstream graph builder checks `polarity_flip: true` and will normalize `PhysActivity=0 increases Risk` into `PhysActivity=1 decreases Risk`.

2. **Population vs Individual Claims**:
   - Annotate epidemiological/population-level assertions and biological mechanisms. Exclude single clinical case study observations ("In this 52-year-old patient...").

3. **Smoking Mentions**:
   - Only phrases specifying the 100-cigarette lifetime criterion ("ever smoked at least 100 cigarettes") map to `Smoker=1`.
   - If the sentence says "smoking increases risk", mark the relation as `causal`, but note that the mention itself is `unmapped`. Never force `Smoker` if the 100-cigarette criterion is absent.

---

### 4. Double-Annotation Protocol
Roughly 20% of candidate pairs are flagged with `double_annotate: true`. Both Annotator 1 and Annotator 2 must annotate these items independently without consulting each other. Cohen's Kappa will be calculated before any adjudication.
