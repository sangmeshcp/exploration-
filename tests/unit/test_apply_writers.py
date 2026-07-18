import json
from pathlib import Path

import pytest
import yaml

from shadowtrace.db import duckdb_store
from shadowtrace.recommend.apply import apply_recommendation, revert_policy
from shadowtrace.recommend.apply.claude_code import ClaudeCodeWriter
from shadowtrace.recommend.apply.litellm import LiteLLMWriter
from shadowtrace.recommend.apply.nanoclaw import NanoClawWriter


def test_claude_code_writer_dry_run_does_not_touch_disk(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    writer = ClaudeCodeWriter(settings_path)
    diff = writer.preview("Code Questions", "claude-haiku-4-5")
    assert "claude-haiku-4-5" in diff
    assert not settings_path.exists()


def test_claude_code_writer_write_and_revert(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"existing": "value"}))
    writer = ClaudeCodeWriter(settings_path)

    previous = writer.write("Code Questions", "claude-haiku-4-5")
    data = json.loads(settings_path.read_text())
    assert data["model_routing"]["Code Questions"] == "claude-haiku-4-5"
    assert data["existing"] == "value"

    writer.revert(previous)
    restored = json.loads(settings_path.read_text())
    assert restored == {"existing": "value"}


def test_litellm_writer_write_and_revert(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    writer = LiteLLMWriter(config_path)
    previous = writer.write("Code Questions", "claude-haiku-4-5")
    data = yaml.safe_load(config_path.read_text())
    assert data["router_settings"]["routing_rules"]["Code Questions"] == "claude-haiku-4-5"

    writer.revert(previous)
    assert config_path.read_text() == ""


def test_nanoclaw_writer_replaces_existing_rule_for_same_archetype(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    writer = NanoClawWriter(rules_path)
    writer.write("Code Questions", "claude-haiku-4-5")
    writer.write("Code Questions", "claude-sonnet-4-6")  # should replace, not duplicate
    data = json.loads(rules_path.read_text())
    matching = [r for r in data["rules"] if r["archetype"] == "Code Questions"]
    assert len(matching) == 1
    assert matching[0]["route_to"] == "claude-sonnet-4-6"


def test_apply_recommendation_dry_run_returns_diff_without_writing(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    settings_path = tmp_path / "settings.json"
    writer = ClaudeCodeWriter(settings_path)

    diff = apply_recommendation(
        conn, writer, "arch-1", "Code Questions", "claude-haiku-4-5", dry_run=True
    )
    assert isinstance(diff, str)
    assert not settings_path.exists()
    assert conn.execute("SELECT count(*) FROM applied_policies").fetchone()[0] == 0


def test_apply_recommendation_confirmed_writes_and_records(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    settings_path = tmp_path / "settings.json"
    writer = ClaudeCodeWriter(settings_path)

    result = apply_recommendation(
        conn, writer, "arch-1", "Code Questions", "claude-haiku-4-5", dry_run=False
    )
    assert settings_path.exists()
    row = conn.execute(
        "SELECT archetype_id, candidate, writer FROM applied_policies WHERE id = ?", [result.id]
    ).fetchone()
    assert row == ("arch-1", "claude-haiku-4-5", "claude_code")


def test_revert_policy_restores_byte_identical_content(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    settings_path = tmp_path / "settings.json"
    original = json.dumps({"theme": "dark", "existing": True}, indent=2, sort_keys=True)
    settings_path.write_text(original)
    writer = ClaudeCodeWriter(settings_path)

    result = apply_recommendation(
        conn, writer, "arch-1", "Code Questions", "claude-haiku-4-5", dry_run=False
    )
    assert settings_path.read_text() != original

    revert_policy(conn, {"claude_code": writer}, result.id, reason="regression detected")
    assert settings_path.read_text() == original

    reverted_ts = conn.execute(
        "SELECT reverted_ts, revert_reason FROM applied_policies WHERE id = ?", [result.id]
    ).fetchone()
    assert reverted_ts[0] is not None
    assert reverted_ts[1] == "regression detected"


def test_revert_policy_raises_for_unknown_id(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    with pytest.raises(ValueError):
        revert_policy(conn, {}, "does-not-exist", reason="x")
