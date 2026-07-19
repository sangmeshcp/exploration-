"""Golden-path end-to-end scenario (plan.md §3.3, E1): ingest -> capture ->
ETL -> mine -> replay -> recommend -> apply, all wired together against
the fixture transcript corpus (tests/fixtures/traces/).

This is a single, thorough integration of the whole pipeline rather than
the full E1-E14 docker-compose matrix described in the plan — see
README.md's "Scope and what's simplified" section for what's covered here
vs. left as follow-up (failure injection, chain replay, egress
containment, live-model smoke tests, etc.).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from shadowtrace.common.config import Settings
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.db import duckdb_store
from shadowtrace.ingest.claude_transcripts import TranscriptWatcher
from shadowtrace.ingest.etl import backfill_prompt_text, run_etl
from shadowtrace.mining.archetypes import assign_archetypes
from shadowtrace.mining.cluster import cluster_embeddings
from shadowtrace.mining.embed import HashingEmbedder
from shadowtrace.recommend import engine as recommend_engine
from shadowtrace.recommend.apply import apply_recommendation
from shadowtrace.recommend.apply.claude_code import ClaudeCodeWriter
from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.runner import CallResult, ReplayRunner, RunnerConfig
from shadowtrace.replay.sampler import SampleRow, sample_traces
from shadowtrace.replay.sprt import SPRTConfig, run_sprt

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "traces"

LOCAL_CANDIDATE = Candidate(
    name="ollama-llama3.1-8b",
    provider="ollama",
    price_per_mtok_in=0,
    price_per_mtok_out=0,
    local=True,
)
CLOUD_CANDIDATE = Candidate(
    name="claude-haiku-4-5", provider="anthropic", price_per_mtok_in=1.0, price_per_mtok_out=5.0
)


async def _fake_grader(sample: SampleRow, result: CallResult) -> bool:
    # deterministic stand-in for the judge: "pass" unless the fixture
    # deliberately marks this candidate as bad for this archetype
    return "FAIL" not in result.response_text


async def _fake_call_fn(candidate: Candidate, sample: SampleRow) -> CallResult:
    # local candidate is reliably good; the cloud candidate is reliably
    # good too here (both "pass") -- this simulates a shadow run that
    # finds both viable, local preferred since it's free and tried first.
    return CallResult(response_text="looks correct", tokens_in=200, tokens_out=100, latency_ms=8)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_golden_path_end_to_end(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path / "shadowtrace_home")
    settings.ensure_dirs()
    REGISTRY.reset()

    # --- 1. simulate 8 live Claude Code sessions by copying fixture transcripts
    #     into a temp "~/.claude/projects/" mirror
    projects_dir = tmp_path / "claude_projects"
    projects_dir.mkdir()
    fixture_files = sorted(FIXTURES_DIR.glob("*.jsonl"))
    assert len(fixture_files) >= 3, "fixture corpus missing — run tests/fixtures/generate_traces.py"
    for f in fixture_files:
        shutil.copy(f, projects_dir / f.name)

    # --- 2. transcript watcher ingests into the SQLite hot path (primary lane)
    from shadowtrace.proxy.capture import CaptureWriter

    capture = CaptureWriter(settings.sqlite_path, settings.spill_dir, flush_interval=100.0)
    watcher = TranscriptWatcher(projects_dir, capture)
    ingested = watcher.scan_once()
    assert ingested > 0
    await capture.flush()
    assert REGISTRY.get_gauge("capture_lag_seconds") >= 0.0

    # --- 3. ETL: SQLite -> DuckDB
    etl_stats = run_etl(settings.sqlite_path, settings.duckdb_path)
    assert etl_stats.rows_inserted == ingested

    # re-running ETL must be idempotent (no duplicate rows)
    etl_stats2 = run_etl(settings.sqlite_path, settings.duckdb_path)
    assert etl_stats2.rows_inserted == 0

    # --- 4. mining: embed + cluster + assign archetypes
    backfilled = backfill_prompt_text(settings.duckdb_path)
    assert backfilled > 0

    conn = duckdb_store.connect(settings.duckdb_path)
    rows = conn.execute(
        "SELECT id, prompt_text FROM traces WHERE prompt_text IS NOT NULL AND quarantined = FALSE"
    ).fetchall()
    trace_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    embedder = HashingEmbedder(dims=256)
    vectors = embedder.embed(texts)
    labels = cluster_embeddings(vectors, min_cluster_size=3, similarity_threshold=0.35)
    assignment = assign_archetypes(conn, trace_ids, texts, vectors, labels)
    assert len(assignment.new_archetype_ids) >= 2  # at least a couple of distinct archetypes found

    archetype_count = conn.execute("SELECT count(*) FROM archetypes").fetchone()[0]
    assert archetype_count == len(assignment.new_archetype_ids)

    # --- 5. shadow replay: budgeted run over sampled traces
    samples_by_archetype = sample_traces(conn, n_per_archetype=10, seed=7)
    all_samples = [s for pool in samples_by_archetype.values() for s in pool]
    assert all_samples

    ladder = [LOCAL_CANDIDATE, CLOUD_CANDIDATE]
    run_id = "e2e-run-1"
    runner_config = RunnerConfig(
        budget_usd=5.0, checkpoint_path=settings.checkpoints_dir / f"{run_id}.json"
    )
    runner = ReplayRunner(ladder, _fake_grader, _fake_call_fn, runner_config)
    run_result = await runner.run(run_id, all_samples)
    assert run_result.results

    sprt_config = SPRTConfig(p0=0.9, p1=0.5)
    verdict_count = 0
    for archetype_id in samples_by_archetype:
        for candidate in ladder:
            outcomes = [
                r.verdict == "pass"
                for r in run_result.results
                if r.archetype_id == archetype_id and r.candidate == candidate.name
            ]
            if not outcomes:
                continue
            state = run_sprt(outcomes, sprt_config, max_samples=len(outcomes))
            conn.execute(
                "INSERT INTO sprt_verdicts (archetype_id, candidate, run_id, state, n_samples, "
                "n_pass, log_likelihood_ratio, ts) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                [
                    archetype_id,
                    candidate.name,
                    run_id,
                    state.verdict,
                    state.n,
                    state.n_pass,
                    state.log_likelihood_ratio,
                ],
            )
            verdict_count += 1
    assert (
        verdict_count >= 3
    )  # SPRT-significant verdicts for >= 3 archetypes, per M3 accept criteria

    # --- 6. recommendations
    recs = recommend_engine.build_recommendations(
        conn, access_mode="subscription", quota_period_tokens=1_000_000
    )
    assert recs, "expected at least one passing recommendation from the replay run"

    # --- 7. apply top recommendation: dry-run diff -> confirm -> write
    top = recs[0]
    label_row = conn.execute(
        "SELECT label FROM archetypes WHERE id = ?", [top.archetype_id]
    ).fetchone()
    archetype_label = label_row[0]

    settings_path = tmp_path / "claude_code_settings.json"
    writer = ClaudeCodeWriter(settings_path)
    diff = apply_recommendation(
        conn, writer, top.archetype_id, archetype_label, top.candidate, dry_run=True
    )
    assert isinstance(diff, str) and top.candidate in diff
    assert not settings_path.exists()  # dry-run must not touch disk

    applied = apply_recommendation(
        conn, writer, top.archetype_id, archetype_label, top.candidate, dry_run=False
    )
    assert not isinstance(applied, str)
    assert settings_path.exists()
    written = json.loads(settings_path.read_text())
    assert written["model_routing"][archetype_label] == top.candidate

    applied_row = conn.execute(
        "SELECT archetype_id, candidate FROM applied_policies WHERE id = ?", [applied.id]
    ).fetchone()
    assert applied_row == (top.archetype_id, top.candidate)
