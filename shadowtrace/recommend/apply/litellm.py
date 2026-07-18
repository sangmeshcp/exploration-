"""LiteLLM router config writer: adds a routing rule mapping an archetype
label to a recommended candidate model (plan.md M4.3).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _diff(before_text: str, after_text: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )


@dataclass
class LiteLLMWriter:
    config_path: Path
    name: str = "litellm"

    def _load(self) -> dict[str, Any]:
        if self.config_path.exists():
            return dict(yaml.safe_load(self.config_path.read_text()) or {})
        return {"model_list": []}

    def _apply(self, data: dict[str, Any], archetype_label: str, candidate: str) -> dict[str, Any]:
        updated = yaml.safe_load(yaml.safe_dump(data)) or {}
        updated.setdefault("router_settings", {}).setdefault("routing_rules", {})[
            archetype_label
        ] = candidate
        return updated

    def preview(self, archetype_label: str, candidate: str) -> str:
        before = self._load()
        after = self._apply(before, archetype_label, candidate)
        return _diff(yaml.safe_dump(before, sort_keys=True), yaml.safe_dump(after, sort_keys=True))

    def write(self, archetype_label: str, candidate: str) -> str:
        previous_content = self.config_path.read_text() if self.config_path.exists() else ""
        data = self._apply(self._load(), archetype_label, candidate)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(data, sort_keys=True))
        return previous_content

    def revert(self, previous_content: str) -> None:
        self.config_path.write_text(previous_content)
