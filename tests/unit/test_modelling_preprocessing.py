import pandas as pd
import pytest

from src.data.brfss import preprocess, synthetic_fixture


def test_raw_mapping_and_exclusions():
    raw = pd.DataFrame(
        {
            "_BMI5": [1849, 1850, 2500, 3000, 999999, 2200],
            "DIABETE3": [3, 4, 1, 2, 1, 7],
            "SMOKE100": [1, 2, 7, 9, 1, 2],
            "_AGE80": [44, 45, 65, 75, 80, 17],
            "Outcome_leak": [1] * 6,
        }
    )
    data, report = preprocess(raw, {}, ["BMICat", "Outcome"])
    assert list(data.BMICat) == ["underweight", "normal", "overweight"]
    assert list(data.Outcome) == ["0", "1", "1"]
    assert list(data.AgeCat) == ["young", "middle_aged", "older"]
    assert report["rows_removed"] == 3
    assert "Outcome_leak" not in data
    assert pd.isna(data.Smoker.iloc[2])


def test_missing_columns_and_bad_profile():
    with pytest.raises(ValueError, match="unavailable"):
        preprocess(pd.DataFrame({"_BMI5": [2200]}), {}, ["BMICat", "Outcome"])
    with pytest.raises(ValueError, match="Unknown"):
        preprocess(pd.DataFrame(), {"profile": "unknown"}, ["Outcome"])
    with pytest.raises(ValueError, match="No valid"):
        preprocess(pd.DataFrame({"_BMI5": [0], "DIABETE3": [9]}), {}, ["Outcome", "BMICat"])


def test_fixture_reproducible_and_labelled():
    a = synthetic_fixture()
    pd.testing.assert_frame_equal(a, synthetic_fixture())
    clean, report = preprocess(a, {"profile": "canonical_test_fixture"}, ["Outcome", "BMICat"])
    assert report["is_synthetic"]
    assert len(clean) == 400
