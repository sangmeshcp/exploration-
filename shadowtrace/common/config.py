"""Central paths & settings. State lives in one directory (~/.shadowtrace/)
per the plan's non-functional requirement: single command up/down, backup
= copy the directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_home() -> Path:
    override = os.environ.get("SHADOWTRACE_HOME")
    if override:
        return Path(override)
    return Path.home() / ".shadowtrace"


def _default_claude_projects_dir() -> Path:
    override = os.environ.get("SHADOWTRACE_CLAUDE_PROJECTS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "projects"


@dataclass
class Settings:
    home: Path = field(default_factory=_default_home)
    claude_projects_dir: Path = field(default_factory=_default_claude_projects_dir)

    @property
    def sqlite_path(self) -> Path:
        return self.home / "capture.sqlite3"

    @property
    def duckdb_path(self) -> Path:
        return self.home / "analytics.duckdb"

    @property
    def spill_dir(self) -> Path:
        return self.home / "spill"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def traces_path(self) -> Path:
        """Local-file OTel span export target (plan.md §4.5)."""
        return self.logs_dir / "traces.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        return self.home / "checkpoints"

    @property
    def bypass_flag_path(self) -> Path:
        """Presence of this file means `shadow pause` is active."""
        return self.home / "PAUSED"

    @property
    def ladder_config_path(self) -> Path:
        return self.home / "ladder.yaml"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.spill_dir, self.logs_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)

    def is_paused(self) -> bool:
        return self.bypass_flag_path.exists()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Test helper: force re-read of env vars on next get_settings()."""
    global _settings
    _settings = None
