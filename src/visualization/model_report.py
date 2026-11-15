"""A self-contained local HTML report with an interactive BMI comparison."""

import html
from pathlib import Path

from src.visualization.presentation import safe_json, write_viewer_assets


def render_report(summary: dict, output: Path) -> None:
    source = (
        "SYNTHETIC SOFTWARE DEMO"
        if summary["source"]["is_synthetic"]
        else "CDC BRFSS 2015 · real observations"
    )
    metrics = summary["evaluation"]
    rows = summary["preprocessing"]["rows_retained"]
    comparisons = safe_json(summary["comparisons"])
    body = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Diabetes evidence · Probabilistic modelling</title>
<style>body{font:17px/1.6 system-ui;margin:0;background:#f5f5ef;color:#17333c}main{max-width:1000px;margin:48px auto;padding:24px}h1{font-size:42px;line-height:1.15}section{padding:24px 0;border-top:1px solid #b7c7c9}img{max-width:100%;height:auto}select{font:inherit;padding:8px}output{display:block;padding:20px;background:#dcecf0;margin-top:15px}.note{border-left:4px solid #b66c29;padding-left:18px}a{color:#176779}table{border-collapse:collapse}td,th{padding:8px 18px;text-align:left;border-bottom:1px solid #c7d4d5}</style>
<main><p>Biomedical evidence / probabilistic modelling</p><h1>Diabetes evidence, explored</h1>
<p>__SOURCE__</p><p class="note">Human validation: __HUMAN__. The graph is a project-accepted structural prior. Causal validity is not established. Estimates are unweighted respondent probabilities, not population prevalence or clinical advice.</p>
<section><h2>Model and data</h2><p>__ROWS__ observations retained. Only the imported BMI → Outcome relationship is included in the default model. Outcome combines self-reported prediabetes and diabetes.</p><img src="figures/network.png" alt="Bayesian network structure"></section>
<section><h2>Explore a BMI category</h2><label for="bmi">BMI category </label><select id="bmi"></select><output id="result" aria-live="polite"></output><p>Observation selects respondents in a category. Intervention replaces the BMI mechanism in the assumed model. Equal results in this sparse graph follow from its lack of confounders; they do not prove absence of real confounding.</p><img src="figures/comparison.png" alt="Observation and hypothetical intervention comparison"></section>
<section><h2>Conditional probabilities</h2><img src="figures/cpds.png" alt="Fitted conditional probability tables"><p><a href="cpds.csv">Download CPD tables</a></p></section>
<section><h2>Held-out prediction</h2><p>Log loss: __LOSS__ · Brier score: __BRIER__. These assess observational prediction, not causal accuracy.</p><img src="figures/calibration.png" alt="Held-out calibration"><p>Calibration is assessed only in populated prediction bins. Higher-risk ranges without predictions are unmeasured; good observed calibration is not causal validation.</p></section>
<section><h2>Evidence quality</h2><img src="figures/validation.png" alt="Human annotation progress"><p><a href="blank_review_form.csv">Blank independent-review form</a> · <a href="validation.json">Review provenance</a></p></section>
<section><h2>Reproduce and inspect</h2><p><a href="summary.json">Full results and assumptions</a> · <a href="sensitivity.json">Sensitivity analysis</a> · <a href="model.json">Fitted model</a> · <a href="report.md">Written report</a></p></section></main>
<script type="application/json" id="comparison-data">__COMPARISONS__</script><script src="viewer.js" defer></script></html>"""
    for token, value in {
        "__SOURCE__": html.escape(source),
        "__HUMAN__": html.escape(summary["human_validation"]),
        "__ROWS__": f"{rows:,}",
        "__LOSS__": f"{metrics['log_loss']:.5f}",
        "__BRIER__": f"{metrics['brier_score']:.5f}",
        "__COMPARISONS__": comparisons,
    }.items():
        body = body.replace(token, value)
    write_viewer_assets(output)
    (output / "index.html").write_text(body)
    lines = [
        "# Probabilistic modelling execution report",
        "",
        source,
        "",
        f"Rows retained: {rows:,}",
        f"Training rows: {summary['split']['train_rows']:,}; test rows: {summary['split']['test_rows']:,}",
        f"Human validation: {summary['human_validation']}",
        "",
        f"Held-out log loss: {metrics['log_loss']:.6f}; Brier score: {metrics['brier_score']:.6f}",
        f"Training-prevalence baseline log loss: {metrics['baseline']['log_loss']:.6f}",
        "",
        "## Interpretation",
        "",
        "CPDs are fitted to training rows only. These metrics assess observational prediction.",
        "The evidence edge is a structural prior, not independently established causation.",
        "Interventions use graph surgery, not ordinary conditioning. Default results are model simulations only.",
        "In the default two-node root-treatment DAG, observational and interventional probabilities are equal because BMI has no parents or modelled confounders; this is a structural assumption, not empirical proof of no confounding.",
        "Calibration covers only populated prediction bins. With four discrete BMI states, it cannot evaluate unpredicted higher-risk ranges.",
        "Causal evaluation would require a prespecified estimand and independent interventional or credible quasi-experimental evidence under defensible identification assumptions; predictive scores alone do not provide it.",
        "The sparse graph assumes away confounding; additional structure and defensible assumptions are needed for causal claims.",
        "Survey weights and complex sample design are not used; no population prevalence claim is supported.",
        "The Outcome variable includes prediabetes and does not establish incident type 2 diabetes.",
        "",
        "## BMI comparisons",
        "",
        "| BMI | Observational P(Outcome=1) | Model do(BMI) |",
        "|---|---:|---:|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['state']} | {row['observational']['distribution']['1']:.6f} | {row['interventional']['distribution']['1']:.6f} |"
        )
    lines += [
        "",
        "Exact configuration, source hash, package versions, split sizes, and sensitivity results are in summary.json.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
