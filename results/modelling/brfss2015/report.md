# Probabilistic modelling execution report

CDC BRFSS 2015 · real observations

Rows retained: 401,327
Training rows: 321,061; test rows: 80,266
Human validation: pending

Held-out log loss: 0.396413; Brier score: 0.120483
Training-prevalence baseline log loss: 0.421152

## Interpretation

CPDs are fitted to training rows only. These metrics assess observational prediction.
The evidence edge is a structural prior, not independently established causation.
Interventions use graph surgery, not ordinary conditioning. Default results are model simulations only.
In the default two-node root-treatment DAG, observational and interventional probabilities are equal because BMI has no parents or modelled confounders; this is a structural assumption, not empirical proof of no confounding.
Calibration covers only populated prediction bins. With four discrete BMI states, it cannot evaluate unpredicted higher-risk ranges.
Causal evaluation would require a prespecified estimand and independent interventional or credible quasi-experimental evidence under defensible identification assumptions; predictive scores alone do not provide it.
The sparse graph assumes away confounding; additional structure and defensible assumptions are needed for causal claims.
Survey weights and complex sample design are not used; no population prevalence claim is supported.
The Outcome variable includes prediabetes and does not establish incident type 2 diabetes.

## BMI comparisons

| BMI | Observational P(Outcome=1) | Model do(BMI) |
|---|---:|---:|
| underweight | 0.051647 | 0.051647 |
| normal | 0.067615 | 0.067615 |
| overweight | 0.134314 | 0.134314 |
| obese | 0.262219 | 0.262219 |

Exact configuration, source hash, package versions, split sizes, and sensitivity results are in summary.json.
