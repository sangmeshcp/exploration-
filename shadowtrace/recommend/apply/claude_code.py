"""Claude Code settings writer: adds a `model_routing` hint mapping an
archetype label to a recommended candidate model (plan.md M4.3).
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
class ClaudeCodeWriter:
    settings_path: Path
    name: str = "claude_code"

    def _load(self) -> dict[str, Any]:
        if self.settings_path.exists():
            return dict(json.loads(self.settings_path.read_text()))
        return {}

    def _apply(self, data: dict[str, Any], archetype_label: str, candidate: str) -> dict[str, Any]:
        updated = json.loads(json.dumps(data))
        updated.setdefault("model_routing", {})[archetype_label] = candidate
        return updated  # type: ignore[no-any-return]

    def preview(self, archetype_label: str, candidate: str) -> str:
        before = self._load()
        after = self._apply(before, archetype_label, candidate)
        return _diff(
            json.dumps(before, indent=2, sort_keys=True),
            json.dumps(after, indent=2, sort_keys=True),
        )

    def write(self, archetype_label: str, candidate: str) -> str:
        previous_content = self.settings_path.read_text() if self.settings_path.exists() else "{}"
        data = self._apply(self._load(), archetype_label, candidate)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        return previous_content

    def revert(self, previous_content: str) -> None:
        self.settings_path.write_text(previous_content)
