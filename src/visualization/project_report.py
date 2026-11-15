"""One results viewer connects NLP comparison, citation-backed DAGs, and probabilities."""

import html
import json
import os
from pathlib import Path

import matplotlib

from src.visualization.presentation import (
    display_text,
    safe_evidence_url,
    safe_json,
    write_viewer_assets,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_project_report(project: dict, model_output: Path, root: Path = Path("results")) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    benchmark = project["nlp"]
    test = benchmark["results"]["test"]
    names = list(test)
    scores = [test[name]["macro_f1_supported_classes"] for name in names]
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [name.replace(" + ", "\n+ ") for name in names]
    ax.barh(
        labels[::-1],
        scores[::-1],
        color=["#dca75b" if "MiniLM" in name else "#287a89" for name in names[::-1]],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 across classes with test support (association and none)")
    ax.set_title("Same 50 held-out pairs · provisional rule-generated reference labels")
    fig.tight_layout()
    fig.savefig(root / "figures/model_comparison.png", dpi=150)
    plt.close(fig)
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{test[name]['macro_f1_supported_classes']:.4f}</td>"
        f"<td>{test[name]['macro_f1_all_three_classes']:.4f}</td><td>Unmeasured (0 examples)</td></tr>"
        for name in names
    )
    annotations = [
        json.loads(line)
        for line in Path(benchmark["annotations_path"]).read_text().splitlines()
        if line.strip()
    ]
    raw_predictions = [
        json.loads(line)
        for line in (Path(benchmark["artifact_dir"]) / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    prediction_map = {r["pair_id"]: r for r in raw_predictions}
    examples = [r for r in annotations if r["relation_class"] == "causal"] + [
        r for r in annotations if r["split"] == "test"
    ][:8]
    example_data = [
        {
            "id": r["pair_id"],
            "split": r["split"],
            "sentence": display_text(r["sentence_text"]),
            "entities": [
                display_text(r["entity_1"]["surface_text"]),
                display_text(r["entity_2"]["surface_text"]),
            ],
            "predictions": prediction_map[r["pair_id"]]["predictions"],
        }
        for r in examples
    ]
    provenance = json.loads(
        (Path(project["artifacts"]["graph_dir"]) / "provenance.json").read_text()
    )
    evidence_html = ""
    for edge in provenance["causal_edges"]:
        evidence_html += f"<h3>{html.escape(edge['source'])} → {html.escape(edge['target'])}</h3>"
        evidence_html += f"<p>{edge['unique_papers_count']} unique papers; {edge['unique_evidence_sentences']} evidence sentences. Project-accepted structural prior; human validation pending.</p>"
        for evidence in edge["evidence"]:
            evidence_html += f"<blockquote>{html.escape(display_text(evidence['sentence_text']))}<br><a href='{safe_evidence_url(evidence.get('url'))}'>PMID {html.escape(str(evidence['pmid']))}</a></blockquote>"
    graph_prefix = Path(os.path.relpath(project["artifacts"]["graph_dir"], root)).as_posix()
    figure_prefix = Path(os.path.relpath(project["artifacts"]["graph_figures"], root)).as_posix()
    benchmark_prefix = Path(os.path.relpath(benchmark["artifact_dir"], root)).as_posix()
    model_html = (model_output / "index.html").read_text()
    prefix = Path(os.path.relpath(model_output, root)).as_posix()
    model_html = model_html.replace('src="figures/', f'src="{prefix}/figures/').replace(
        'href="', f'href="{prefix}/'
    )
    model_html = model_html.replace(
        "<title>Diabetes evidence · Probabilistic modelling</title>",
        "<title>Diabetes evidence · NLP to causal reasoning</title>",
    )
    model_html = model_html.replace(
        "<p>Biomedical evidence / probabilistic modelling</p>",
        "<p>Biomedical text → evidence DAG → Bayesian probabilities → hypothetical interventions</p>",
    )
    encoder_description = (
        "MiniLM loads actual pretrained transformer weights; the encoder is frozen and its classifier head is trained."
        if benchmark["encoder"].get("parameter_count")
        else "Transformer evaluation was explicitly disabled; no transformer score is reported."
    )
    nlp = f"""<section id="nlp"><h2>1. Extract relationships from biomedical text</h2><p>{project["corpus"]["abstracts"]} frozen abstracts verified by hash. Samples below show actual extractor outputs with their original train/test partition. A predicted causal claim does not establish biological causation.</p><label for="nlp-example">Literature sample </label><select id="nlp-example"></select><blockquote id="sentence"></blockquote><p id="entities"></p><output id="nlp-predictions" aria-live="polite"></output><p>For a new sentence: <code>./run.sh predict-text --text "Obesity causes type 2 diabetes."</code></p></section>
<section id="comparison"><h2>2. Compare rules, traditional ML, and a real transformer</h2><p>Each method is evaluated on the same fixed document holdout. TF-IDF and MiniLM use the same logistic classifier settings and training pairs. {encoder_description} The keyword heuristic is separately labelled.</p><div style="overflow:auto"><table><thead><tr><th>Method</th><th>Primary: supported-class macro F1</th><th>Secondary: 3-class macro F1</th><th>Causal generalization</th></tr></thead><tbody>{rows}</tbody></table></div><img src="figures/model_comparison.png" alt="Controlled held-out comparison"><p class="note">Reference metadata records rule_heuristic_regex_v1, not verified expert labels or a verified exact generator implementation. Evaluating related rules against these labels may be circular. Supported-class macro F1 is primary; the three-class aggregate is retained as a secondary compatibility metric. There are no causal examples in dev/test. These results do not establish independent clinical accuracy or held-out causal extraction ability. LLM pilots use different subsets and are excluded from this fair comparison.</p><p><a href="{benchmark_prefix}/comparison.json">Protocol, split IDs, support counts, timing and limitations</a> · <a href="{benchmark_prefix}/comparison.csv">Comparison table</a> · <a href="{benchmark_prefix}/error_analysis.json">Per-split error analysis</a></p></section>
<section id="evidence"><h2>3. Trace the evidence-backed DAG</h2><p>Only causal claims with direction survive conservative context filters. Associations stay in a separate undirected graph. Unique papers and sentences are counted separately from extractor agreement. Learned class-only predictions are not silently promoted to directed graph edges.</p><img src="{figure_prefix}/causal_dag.png" alt="Rebuilt directed acyclic evidence graph"><p>{project["graph"]["filtering_stats"]["association_edges_count"]} observational association edges are preserved separately.</p>{evidence_html}<p><a href="{graph_prefix}/evidence_graph_after_repair.graphml">DAG GraphML</a> · <a href="{graph_prefix}/provenance.json">Full citation provenance</a></p></section>"""
    model_html = model_html.replace(
        "<section><h2>Model and data</h2>",
        nlp + "<section><h2>4. Estimate Bayesian probabilities</h2>",
    )
    model_html = model_html.replace(
        "<h2>Explore a BMI category</h2>", "<h2>5. Compare observation and intervention</h2>"
    )
    model_html = model_html.replace(
        "<h2>Held-out prediction</h2>", "<h2>6. Evaluate predictions and limitations</h2>"
    )
    samples = safe_json(example_data)
    payload = '<script type="application/json" id="nlp-data">' + samples + "</script>"
    model_html = model_html.replace("</html>", payload + "</html>")
    write_viewer_assets(root)
    (root / "index.html").write_text(model_html)
    (root / "project_summary.json").write_text(json.dumps(project, indent=2, allow_nan=False))
