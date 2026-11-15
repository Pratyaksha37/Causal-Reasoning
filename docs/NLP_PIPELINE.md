# NLP extraction and fair comparison protocol

The frozen corpus contains 500 Europe PMC abstracts with SHA-256 verification. Sentence-level alias matching produces mentions mapped to canonical variables; longer aliases are preferred. Physical inactivity flips exposure polarity when normalized to PhysActivity=1. Smoker mapping preserves the explicit 100-cigarette lifetime criterion; vague smoking remains an open-domain concept rather than being forced into that survey variable.

The existing spaCy rule extractor predicts relation class, direction, polarity and language status. The keyword heuristic is a separate shallow baseline. The prompted LLM adapter supports cached typed outputs, but its small historical pilots are not a common-sample benchmark.

## Actual trained comparison

`configs/benchmark.yaml` defines the pinned MiniLM revision, CPU batch size, maximum input length, fixed logistic hyperparameters, source records and output paths. `./run.sh benchmark`:

1. Loads the existing 250 records and rejects duplicate pair IDs or document leakage across splits.
2. Keeps the frozen 150/50/50 train/dev/test allocation.
3. Runs the rules and keyword baseline on exactly those candidates.
4. Builds explicit entity-marked text. The same tokenizer-bounded text is used for both learned methods.
5. Fits TF-IDF vocabulary and its logistic head only on train.
6. Loads genuine pretrained `sentence-transformers/all-MiniLM-L6-v2` weights, performs attention-masked mean pooling and L2 normalization, and fits a logistic head only on train. Encoder weights remain frozen.
7. Reports every method on the exact same dev/test IDs, with common relation classes, support counts and confusion matrices.

No test-based tuning is performed. Settings are fixed in configuration. The encoder ID, revision, architecture, weight SHA-256, parameter count, pooling, device, truncation count, timings and split IDs are recorded. Timing includes warm/cached encodings and is not a standardized hardware speed benchmark. Heads and TF-IDF vocabulary/IDF are saved as portable JSON; prediction does not load executable pickle. The encoder cache key includes its weights and exact input texts.

The historical name `DistilBERTExtractor` is retained only as a compatibility alias for the keyword heuristic. It is not the transformer's implementation. Full neural fine-tuning is not claimed.

## Interpretation

The reference labels are `PROVISIONAL_RULE_PSEUDO_GOLD`, not independent expert truth. This favours related rule systems. Causal support is 4 in train and 0 in dev/test. Supported-class macro F1 is primary; the all-three-class aggregate is retained as secondary; zero-support causal performance is explicitly unmeasured. Strong rule F1 cannot validate causal generalization, and a lower MiniLM score cannot establish that transformers are intrinsically worse.

The learned classifiers predict relation class only. They do not supply unsupported causal direction, polarity or human review, and are not automatically added as directed DAG evidence. The graph retains its own provenance and conservative inclusion rules.

The model card documents MiniLM's architecture and mean-pooling method: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## New text

```bash
./run.sh predict-text --text "Obesity causes type 2 diabetes."
./run.sh predict-text --rules-only --text "Physical inactivity increases the risk of type 2 diabetes."
```

New user text is marked as user input, not as a PubMed citation. Predictions do not update the literature DAG automatically.


Both evaluator APIs use `src/schemas/evaluation.py` and the single relation-class calculation in `src/evaluation/relation.py`. Rich extractors expose additional evaluated dimensions; learned class-only heads never receive invented direction/polarity scores. The old rounded `macro_f1` alias is retained for compatibility.

`results/benchmark/error_analysis.json` includes every split/method's misclassified pair IDs and reference/predicted classes. The rule and keyword extractors additionally use the existing diagnostic categorizer for direction, polarity, relation confusion and language errors. These remain diagnostics against provisional references, not adjudicated clinical mistakes.
