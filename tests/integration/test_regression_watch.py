"""E10: applied policy -> spot-check grades forced below floor -> alert
raised, revert restores prior config byte-identically, applied_policies
history intact (plan.md §3.3, M4.4).
"""

from pathlib import Path

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.db import duckdb_store
from shadowtrace.recommend import engine
from shadowtrace.recommend.apply import apply_recommendation
from shadowtrace.recommend.apply.claude_code import ClaudeCodeWriter


def test_regression_watch_reverts_on_breach_and_preserves_history(tmp_path: Path) -> None:
    REGISTRY.reset()
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    now = 1_700_000_000_000
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) "
        "VALUES ('a1', 'Code Questions', False, ?, ?)",
        [now, now],
    )

    settings_path = tmp_path / "claude_settings.json"
    original_content = '{"existing": "config", "theme": "dark"}'
    settings_path.write_text(original_content)
    writer = ClaudeCodeWriter(settings_path)

    applied = apply_recommendation(
        conn, writer, "a1", "Code Questions", "claude-haiku-4-5", dry_run=False
    )
    assert not isinstance(applied, str)
    written_after_apply = settings_path.read_text()
    assert written_after_apply != original_content  # the policy really did change the file

    # healthy spot-checks: watch should be a no-op
    for i in range(20):
        engine.record_spot_check(conn, "a1", "claude-haiku-4-5", f"t{i}", passed=True)
    reverted = engine.watch_and_revert(
        conn, {"claude_code": writer}, applied.id, "a1", "claude-haiku-4-5", floor=0.90
    )
    assert reverted is False
    assert settings_path.read_text() == written_after_apply  # untouched

    # now force a regression: mostly-failing spot-checks push the recent
    # window's pass rate below the floor
    for i in range(20, 40):
        engine.record_spot_check(conn, "a1", "claude-haiku-4-5", f"t{i}", passed=(i % 5 == 0))

    reverted = engine.watch_and_revert(
        conn, {"claude_code": writer}, applied.id, "a1", "claude-haiku-4-5", floor=0.90
    )
    assert reverted is True
    assert REGISTRY.get_counter("regression_alerts_total") == 1

    # revert restored the file byte-identically
    assert settings_path.read_text() == original_content

    # applied_policies history is intact, not deleted, with the revert recorded
    row = conn.execute(
        "SELECT archetype_id, candidate, reverted_ts, revert_reason FROM applied_policies WHERE id = ?",
        [applied.id],
    ).fetchone()
    assert row[0] == "a1"
    assert row[1] == "claude-haiku-4-5"
    assert row[2] is not None
    assert "regression" in row[3]


def test_regression_watch_no_data_yet_is_a_noop(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    now = 1_700_000_000_000
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) VALUES ('a1', 'X', False, ?, ?)",
        [now, now],
    )
    settings_path = tmp_path / "settings.json"
    writer = ClaudeCodeWriter(settings_path)
    applied = apply_recommendation(conn, writer, "a1", "X", "claude-haiku-4-5", dry_run=False)
    assert not isinstance(applied, str)

    reverted = engine.watch_and_revert(
        conn, {"claude_code": writer}, applied.id, "a1", "claude-haiku-4-5", floor=0.9
    )
    assert reverted is False
    row = conn.execute(
        "SELECT reverted_ts FROM applied_policies WHERE id = ?", [applied.id]
    ).fetchone()
    assert row[0] is None
