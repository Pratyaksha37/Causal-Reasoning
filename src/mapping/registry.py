"""Registry of canonical variables, values, and alias dictionary loader."""

from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT, load_yaml_config


class CanonicalRegistry:
    """Registry managing canonical variables, allowable values, and phrase matching."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        path = config_path or (PROJECT_ROOT / "configs" / "aliases.yaml")
        self.config: dict[str, Any] = load_yaml_config(path)
        self.canonical_vars: dict[str, Any] = self.config.get("canonical_variables", {})
        self.open_domain_concepts: dict[str, Any] = self.config.get("open_domain_concepts", {})

    def is_canonical(self, var_name: str) -> bool:
        """Check if variable is one of the 8 canonical Phase 2 variables."""
        return var_name in self.canonical_vars

    def is_open_concept(self, var_name: str) -> bool:
        """Check if variable is a recognized open-domain concept."""
        return var_name in self.open_domain_concepts

    def get_phase2_variables(self) -> list[str]:
        """Get list of 8 canonical variables intended for Phase 2 integration."""
        return [k for k, v in self.canonical_vars.items() if v.get("is_phase2_target", False)]
