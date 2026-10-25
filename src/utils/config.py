"""Configuration loading, validation, and override utilities."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Resolve default project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Automatically load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand ${VAR_NAME:-default} in string values within dict/list structures."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")

        def replace(match: re.Match) -> str:
            env_var = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(env_var, default_val)

        return pattern.sub(replace, data)
    return data


def load_yaml_config(
    config_path: Path | str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load and parse a YAML configuration file with environment variable expansion and overrides.

    Args:
        config_path: Path to YAML file. If relative, resolved against PROJECT_ROOT.
        overrides: Optional nested dict of overrides to merge on top.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file cannot be parsed.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {path}: {e}") from e

    expanded = _expand_env_vars(raw_data)
    config: dict[str, Any] = expanded if isinstance(expanded, dict) else {}

    if overrides:
        _deep_merge(config, overrides)

    return config


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """In-place recursively merge source dict into target dict."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
