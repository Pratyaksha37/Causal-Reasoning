"""Unit tests for configuration loading, env-var expansion, and overrides."""

from pathlib import Path

import pytest

from src.utils.config import load_yaml_config


def test_load_existing_config() -> None:
    config = load_yaml_config("configs/experiment.yaml")
    assert "seed" in config
    assert config["seed"] == 42
    assert "sampling" in config
    assert config["sampling"]["split_level"] == "document"


def test_load_nonexistent_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config("configs/nonexistent_file.yaml")


def test_env_var_expansion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_RUN_LEVEL", "CRITICAL")
    test_yaml = tmp_path / "test_env.yaml"
    test_yaml.write_text("level: ${TEST_RUN_LEVEL:-INFO}\ndefault_lvl: ${UNSET_VAR:-DEBUG}\n")

    loaded = load_yaml_config(test_yaml)
    assert loaded["level"] == "CRITICAL"
    assert loaded["default_lvl"] == "DEBUG"


def test_config_overrides() -> None:
    overrides = {"seed": 999, "sampling": {"target_candidate_pairs": 500}}
    loaded = load_yaml_config("configs/experiment.yaml", overrides=overrides)
    assert loaded["seed"] == 999
    assert loaded["sampling"]["target_candidate_pairs"] == 500
    # Check that other keys in sampling are preserved
    assert loaded["sampling"]["split_level"] == "document"
