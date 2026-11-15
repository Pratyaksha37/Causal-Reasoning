"""Reproducible mappings for official 2015 BRFSS microdata and explicit test fixtures."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.mapping.registry import CanonicalRegistry

SOURCE_FIELDS = {
    "AgeCat": "_AGE80",
    "Sex": "SEX",
    "Education": "EDUCA",
    "Income": "INCOME2",
    "Smoker": "SMOKE100",
    "PhysActivity": "EXERANY2",
    "BMICat": "_BMI5",
    "Outcome": "DIABETE3",
}
SOURCE_URL = "https://www.cdc.gov/brfss/annual_data/annual_2015.html"
DOWNLOAD_URL = "https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: str | Path, profile: str) -> pd.DataFrame:
    """Load CSV or SAS XPT, selecting only explicitly permitted source fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset unavailable: {path}. Supply a BRFSS CSV or XPT file.")
    allowed = set(SOURCE_FIELDS.values()) if profile == "brfss_2015" else set(SOURCE_FIELDS)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=lambda column: column in allowed, low_memory=False)
    if path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.strip().lower().endswith(".xpt")]
            if len(members) != 1:
                raise ValueError("ZIP must contain exactly one XPT member")
            with archive.open(members[0]) as stream:
                chunks = pd.read_sas(stream, format="xport", chunksize=10000, encoding="utf-8")
                return pd.concat(
                    [chunk.loc[:, chunk.columns.intersection(list(allowed))] for chunk in chunks],
                    ignore_index=True,
                )
    if path.suffix.lower() == ".xpt":
        chunks = pd.read_sas(path, format="xport", chunksize=10000, encoding="utf-8")
        return pd.concat(
            [chunk.loc[:, chunk.columns.intersection(list(allowed))] for chunk in chunks],
            ignore_index=True,
        )
    raise ValueError("Supported inputs are .csv, .xpt, and single-XPT .zip archives")


def preprocess(raw: pd.DataFrame, config: dict, required: list[str]) -> tuple[pd.DataFrame, dict]:
    """Map fixed codebook categories; drop incomplete model rows, never impute from outcome."""
    profile = config.get("profile", "brfss_2015")
    registry = CanonicalRegistry().canonical_vars
    if not required or any(v not in registry for v in required):
        raise ValueError("Required variables must be nonempty and canonical")
    if profile not in {"brfss_2015", "canonical_test_fixture"}:
        raise ValueError(f"Unknown input profile: {profile}")
    if (
        config.get("missing_policy", "complete_case_model_variables")
        != "complete_case_model_variables"
    ):
        raise ValueError("Only explicit complete-case handling is supported")
    if profile == "brfss_2015" and config.get("year", 2015) != 2015:
        raise ValueError("This codebook mapping supports BRFSS 2015 only")
    if config.get("outcome_definition", "prediabetes_or_diabetes") != "prediabetes_or_diabetes":
        raise ValueError("Outcome definition must match the existing canonical registry")
    result = pd.DataFrame(index=raw.index)
    if profile == "canonical_test_fixture":
        for variable in registry:
            if variable in raw:
                result[variable] = (
                    raw[variable]
                    .map(
                        lambda x: (
                            str(int(x))
                            if isinstance(x, (int, float)) and pd.notna(x) and float(x).is_integer()
                            else str(x)
                        )
                    )
                    .where(raw[variable].notna())
                )
    else:
        values = {
            v: pd.to_numeric(raw[f], errors="coerce") for v, f in SOURCE_FIELDS.items() if f in raw
        }
        if "AgeCat" in values:
            edges = config.get("age_boundaries", [18, 45, 65, 75, 81])
            if (
                len(edges) != 5
                or any(a >= b for a, b in zip(edges, edges[1:], strict=False))
                or edges[0] != 18
                or edges[-1] != 81
            ):
                raise ValueError("Age boundaries must partition adult ages 18 through top-coded 80")
            result["AgeCat"] = pd.cut(
                values["AgeCat"], edges, right=False, labels=registry["AgeCat"]["values"]
            ).astype(object)
        maps = {
            "Sex": {1: "male", 2: "female"},
            "Education": {1: "low", 2: "low", 3: "low", 4: "medium", 5: "high", 6: "high"},
            "Income": {
                1: "low",
                2: "low",
                3: "low",
                4: "low",
                5: "medium",
                6: "medium",
                7: "high",
                8: "high",
            },
            "Smoker": {1: "1", 2: "0"},
            "PhysActivity": {1: "1", 2: "0"},
            "Outcome": {1: "1", 3: "0", 4: "1"},
        }
        for variable, mapping in maps.items():
            if variable in values:
                result[variable] = values[variable].map(mapping)
        if "BMICat" in values:
            bmi = values["BMICat"] / 100.0
            bmi = bmi.where((bmi >= 12) & (bmi <= 99.99))
            result["BMICat"] = pd.cut(
                bmi, [12, 18.5, 25, 30, 100], right=False, labels=registry["BMICat"]["values"]
            ).astype(object)
    for variable in result:
        result[variable] = result[variable].where(
            result[variable].isin(registry[variable]["values"])
        )
    missing = sorted(set(required) - set(result))
    if missing:
        raise ValueError(f"Required fields unavailable after mapping: {missing}")
    invalid = {v: int(result[v].isna().sum()) for v in result}
    keep = result[required].notna().all(axis=1)
    cleaned = result.loc[keep].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid observations remain after preprocessing")
    report = {
        "profile": profile,
        "year": config.get("year", 2015),
        "is_synthetic": profile == "canonical_test_fixture",
        "source": SOURCE_URL if profile == "brfss_2015" else "synthetic software fixture",
        "rows_input": len(raw),
        "rows_retained": len(cleaned),
        "rows_removed": int((~keep).sum()),
        "invalid_or_missing_by_variable": invalid,
        "available_variables": list(result),
        "unavailable_variables": sorted(set(registry) - set(result)),
        "required_variables": required,
        "policy": "Complete cases for model variables only; optional missing states retained",
        "survey_design": "Unweighted respondent-level model; not population prevalence or a design-based estimate",
        "outcome": "Self-reported diabetes OR prediabetes; pregnancy-only diabetes excluded",
        "age_boundaries": config.get("age_boundaries", [18, 45, 65, 75, 81]),
        "source_fields": SOURCE_FIELDS if profile == "brfss_2015" else None,
        "ignored_columns": sorted(
            set(raw) - (set(SOURCE_FIELDS.values()) if profile == "brfss_2015" else set(registry))
        ),
    }
    return cleaned, report


def synthetic_fixture(seed: int = 42, rows: int = 400) -> pd.DataFrame:
    """Generate labelled synthetic observations ONLY for demonstration and software checks."""
    rng = np.random.default_rng(seed)
    bmi = rng.choice(["underweight", "normal", "overweight", "obese"], size=rows)
    # Arbitrary test-generating probabilities, never interpreted as biomedical findings.
    probability = (
        pd.Series(bmi)
        .map({"underweight": 0.1, "normal": 0.15, "overweight": 0.3, "obese": 0.55})
        .to_numpy()
    )
    return pd.DataFrame(
        {"BMICat": bmi, "Outcome": (rng.random(rows) < probability).astype(int).astype(str)}
    )


def portable_source(path: str | Path) -> dict[str, str]:
    """Describe local input relative to the project, or by basename for an external source."""
    from src.utils.config import PROJECT_ROOT

    resolved = Path(path).resolve()
    try:
        return {
            "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
            "path_scope": "project_relative",
        }
    except ValueError:
        return {"path": resolved.name, "path_scope": "external_reference_only"}
