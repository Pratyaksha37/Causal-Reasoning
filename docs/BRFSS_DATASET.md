# BRFSS dataset and mapping decisions

## Source

Official **2015 BRFSS combined landline and cell-phone survey**:

- Documentation and codebook: https://www.cdc.gov/brfss/annual_data/annual_2015.html
- Codebook: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
- Calculated variables: https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf
- Download: https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip

Run `python -m src download`, or download the archive manually. The archive is read as a stream of XPT chunks, selecting eight permitted source columns. The source SHA-256 is recorded with each run. No third-party balanced or preprocessed diabetes dataset is silently substituted. This adapter rejects years other than 2015.

## Fixed mappings

| Canonical variable | 2015 field | Mapping |
|---|---|---|
| AgeCat | `_AGE80` | 18–44 young; 45–64 middle_aged; 65–74 older; 75–80 elderly. Age 80 is top-coded. |
| Sex | `SEX` | 1 male; 2 female, as recorded by this survey. |
| Education | `EDUCA` | 1–3 low; 4 medium (high-school graduate); 5–6 high (some college or college graduate). |
| Income | `INCOME2` | 1–4 low (under $25,000); 5–6 medium ($25,000–under $50,000); 7–8 high ($50,000+). |
| Smoker | `SMOKE100` | 1 → "1" (at least 100 lifetime cigarettes); 2 → "0". Not current-smoking status. |
| PhysActivity | `EXERANY2` | 1 → "1"; 2 → "0". Leisure-time activity/exercise in past month, outside regular work. |
| BMICat | `_BMI5` | Divide stored integer by 100; [12,18.5) underweight, [18.5,25) normal, [25,30) overweight, [30,100) obese. Outside the valid 12–99.99 range becomes missing. |
| Outcome | `DIABETE3` | 1 diabetes OR 4 prediabetes → "1"; 3 no diabetes → "0". Pregnancy-only code 2 excluded; unknown/refused/missing excluded. |

Age, education, and income groupings are explicit **project discretization choices**, not claims that the CDC publishes these exact project labels. Age boundaries are configurable and mutually exclusive. The literature aliases remain unchanged.

## Missingness and leakage

Unknown, refused, missing, and invalid values become missing. Only rows missing a configured model variable are removed. Optional non-model columns may retain missing values; they must become complete cases if included in an exploratory model. Per-column missing/invalid counts are measured before filtering and overlap, so they must not be summed to infer removed rows.

Only the permitted source fields enter preprocessing. Diagnosis-age fields, treatment fields, alternate outcome encodings, and other target-derived columns are not candidate predictors. Mapping thresholds are fixed by configuration, not fitted on the test set. CPDs are fitted on training rows only. No outcome-driven imputation is used.

## Scientific boundaries

The outcome is **self-reported prediabetes or diabetes**, preserving the existing project's combined outcome definition. It is neither a type-specific T2DM diagnosis nor incident diabetes. Literature claims about T2DM and abdominal obesity are therefore imperfect proxies for this survey outcome and BMI measure. This transport and measurement mismatch is documented, not resolved by fitting CPDs.

BRFSS is a complex, weighted, cross-sectional survey. The initial model deliberately estimates **unweighted respondent-level probabilities**. Survey weights, clustering, stratification, population prevalence estimates, nonresponse corrections, and survey-design confidence intervals are not implemented. These outputs must not be described as nationally representative risks.

No adjustment for age/activity/confounding is claimed by the default two-node model. Equality between P(Outcome|BMI) and P(Outcome|do(BMI)) follows from that model's root-treatment structure, not empirical proof of causal identification.
