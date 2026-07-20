"""`shadow` CLI entrypoint (plan.md M0-M4): up, pause, resume, sync,
cluster, run, report, apply.

`up`/`pause`/`run`/`report`/`apply` are the five commands named in the
plan; `resume`, `sync`, and `cluster` are the minimal pragmatic additions
needed to actually drive the pipeline end to end (toggle capture back on,
force an ETL sync, trigger a mining pass) without which the named five
would have nothing to operate on.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import click
import httpx
import uvicorn

from shadowtrace.common.config import get_settings
from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.tracing import build_tracer_provider
from shadowtrace.db import duckdb_store
from shadowtrace.db import sqlite as sqlite_db
from shadowtrace.ingest.claude_transcripts import TranscriptWatcher
from shadowtrace.ingest.etl import backfill_prompt_text, run_etl
from shadowtrace.mining.archetypes import assign_archetypes
from shadowtrace.mining.cluster import cluster_embeddings
from shadowtrace.mining.embed import get_default_embedder
from shadowtrace.proxy.capture import CaptureWriter
from shadowtrace.proxy.server import ProxyConfig, create_app
from shadowtrace.recommend.apply import apply_recommendation
from shadowtrace.recommend.apply.claude_code import ClaudeCodeWriter
from shadowtrace.recommend.apply.litellm import LiteLLMWriter
from shadowtrace.recommend.apply.nanoclaw import NanoClawWriter
from shadowtrace.replay.graders.judge import judge as judge_answer
from shadowtrace.replay.ladder import Candidate, load_ladder, sorted_by_cost
from shadowtrace.replay.runner import (
    CallResult,
    ChainReplayResult,
    RateLimitError,
    ReplayRunner,
    RunnerConfig,
    RunResult,
)
from shadowtrace.replay.sampler import (
    SampleRow,
    sample_chains,
    sample_traces,
    top_archetypes_by_volume,
)
from shadowtrace.replay.sprt import SPRTConfig, run_sprt

logger = get_logger("cli")

_WRITERS: dict[str, type[Any]] = {
    "claude_code": ClaudeCodeWriter,
    "litellm": LiteLLMWriter,
    "nanoclaw": NanoClawWriter,
}


@click.group()
def main() -> None:
    """shadowtrace: personal shadow tracing & model recommender."""


def build_up_components(settings: Any) -> tuple[Any, TranscriptWatcher]:
    """Wires the proxy app and transcript watcher `shadow up` runs, both
    sharing one real tracer provider (local JSON-lines export by default,
    opt-in OTLP via SHADOWTRACE_OTLP_ENDPOINT) — split out from `up()`
    itself so this wiring is unit-testable without booting uvicorn."""
    tracer_provider = build_tracer_provider(
        settings.traces_path, otlp_endpoint=os.environ.get("SHADOWTRACE_OTLP_ENDPOINT")
    )
    config = ProxyConfig(settings=settings, tracer=tracer_provider.get_tracer("shadowtrace.proxy"))
    app = create_app(config)
    watcher = TranscriptWatcher(
        settings.claude_projects_dir,
        app.state.capture,
        tracer=tracer_provider.get_tracer("shadowtrace.ingest"),
    )
    return app, watcher


@main.command()
@click.option("--proxy-port", default=8787, show_default=True)
@click.option("--etl-interval", default=300.0, show_default=True, help="seconds between ETL syncs")
@click.option(
    "--watch-interval", default=2.0, show_default=True, help="seconds between transcript scans"
)
def up(proxy_port: int, etl_interval: float, watch_interval: float) -> None:
    """Start the proxy (secondary/API-key lane) + transcript watcher
    (primary lane) + periodic ETL, all in one process."""
    settings = get_settings()
    settings.ensure_dirs()
    app, watcher = build_up_components(settings)

    async def _run() -> None:
        async def etl_loop() -> None:
            while True:
                try:
                    run_etl(settings.sqlite_path, settings.duckdb_path)
                except Exception:
                    logger.exception("periodic ETL failed")
                await asyncio.sleep(etl_interval)

        server_config = uvicorn.Config(app, host="127.0.0.1", port=proxy_port, log_level="warning")
        server = uvicorn.Server(server_config)

        await asyncio.gather(server.serve(), watcher.run_forever(watch_interval), etl_loop())

    click.echo(f"shadowtrace up: proxy on :{proxy_port}, watching {settings.claude_projects_dir}")
    asyncio.run(_run())


@main.command()
def ingest() -> None:
    """One-shot backfill of existing (and any newly appended) Claude Code
    transcripts under `claude_projects_dir` into the capture store, then
    ETL them into DuckDB — the ingest half of `shadow up`, without booting
    the proxy server or looping forever. `TranscriptFile` always starts
    reading each file from byte 0 the first time it sees it (see
    ingest/claude_transcripts.py), so this picks up your full existing
    session history, not just messages sent after this runs. Use this to
    seed shadowtrace from history you already have, or as a manual
    top-up between `shadow up` runs."""
    settings = get_settings()
    settings.ensure_dirs()
    capture = CaptureWriter(settings.sqlite_path, settings.spill_dir)
    watcher = TranscriptWatcher(settings.claude_projects_dir, capture)

    async def _scan() -> int:
        await capture.start()
        try:
            return watcher.scan_once()
        finally:
            await capture.stop()

    ingested = asyncio.run(_scan())
    stats = run_etl(settings.sqlite_path, settings.duckdb_path)
    click.echo(
        f"ingested {ingested} messages from {settings.claude_projects_dir}; "
        f"etl inserted={stats.rows_inserted} deduped={stats.rows_deduped}"
    )


@main.command()
def pause() -> None:
    """Set the capture bypass flag: forwarding keeps working, capture stops."""
    settings = get_settings()
    settings.ensure_dirs()
    settings.bypass_flag_path.touch()
    click.echo("capture paused")


@main.command()
def resume() -> None:
    """Clear the capture bypass flag set by `shadow pause`."""
    settings = get_settings()
    settings.bypass_flag_path.unlink(missing_ok=True)
    click.echo("capture resumed")


@main.command()
def sync() -> None:
    """Force an immediate SQLite -> DuckDB ETL run."""
    settings = get_settings()
    stats = run_etl(settings.sqlite_path, settings.duckdb_path)
    click.echo(
        f"scanned={stats.rows_scanned} inserted={stats.rows_inserted} deduped={stats.rows_deduped}"
    )


@main.command()
def cluster() -> None:
    """Embed prompt text and (re-)cluster it into archetypes."""
    settings = get_settings()
    run_etl(settings.sqlite_path, settings.duckdb_path)
    backfill_prompt_text(settings.duckdb_path)
    conn = duckdb_store.connect(settings.duckdb_path)
    rows = conn.execute(
        "SELECT id, prompt_text FROM traces WHERE prompt_text IS NOT NULL AND quarantined = FALSE"
    ).fetchall()
    if not rows:
        click.echo("no prompt text available yet to cluster")
        return

    trace_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    embedder = get_default_embedder()
    vectors = embedder.embed(texts)
    labels = cluster_embeddings(vectors)
    result = assign_archetypes(conn, trace_ids, texts, vectors, labels)
    click.echo(
        f"assigned {len(result.trace_to_archetype)} traces across "
        f"{len(result.new_archetype_ids) + len(result.matched_archetype_ids)} archetypes "
        f"({len(result.new_archetype_ids)} new, {len(result.matched_archetype_ids)} matched)"
    )


async def _live_call(candidate: Candidate, sample: SampleRow) -> CallResult:
    """Minimal real wiring for `shadow run` against actual providers. Not
    exercised in tests (which inject a fake call_fn) — replay correctness
    is covered there; this is the untested glue that turns it into a real
    network call. Requires a metered replay API key (plan.md D1.4): the
    subscription used by `shadow up`'s primary capture lane cannot be used
    for programmatic replay calls."""
    prompt_text = json.dumps(sample.request_json)

    if candidate.provider == "ollama":
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": candidate.name, "prompt": prompt_text, "stream": False},
            )
            if resp.status_code == 429:
                raise RateLimitError("ollama busy")
            resp.raise_for_status()
            data = resp.json()
            return CallResult(
                response_text=data.get("response", ""),
                tokens_in=data.get("prompt_eval_count", 0),
                tokens_out=data.get("eval_count", 0),
                latency_ms=0,
            )

    if candidate.provider == "anthropic":
        api_key = os.environ.get("SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY (or SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY) is required to "
                "replay against anthropic candidates"
            )
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": candidate.name,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt_text}],
                },
            )
            if resp.status_code == 429:
                raise RateLimitError("anthropic rate limited")
            resp.raise_for_status()
            data = resp.json()
            text = "".join(
                b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
            )
            usage = data.get("usage", {})
            return CallResult(
                response_text=text,
                tokens_in=usage.get("input_tokens", 0),
                tokens_out=usage.get("output_tokens", 0),
                latency_ms=0,
            )

    raise NotImplementedError(f"no live-call wiring for provider={candidate.provider!r} yet")


@main.command(name="run")
@click.option("--budget", "budget_usd", type=float, required=True)
@click.option("--archetype", "archetype_id", default=None)
@click.option("--n-per-archetype", default=10, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--floor", default=0.90, show_default=True, help="per-archetype SPRT pass-rate floor")
@click.option(
    "--chains/--no-chains",
    default=True,
    show_default=True,
    help="also chain-replay the top-2 archetypes by volume (plan.md M3.6)",
)
@click.option(
    "--chain-budget",
    "chain_budget_usd",
    type=float,
    default=0.50,
    show_default=True,
    help="per-chain budget sub-cap, independent of --budget (review R6)",
)
def run_cmd(
    budget_usd: float,
    archetype_id: str | None,
    n_per_archetype: int,
    seed: int,
    floor: float,
    chains: bool,
    chain_budget_usd: float,
) -> None:
    """Budget-capped shadow replay: `shadow run --budget 3.00 [--archetype X]`."""
    settings = get_settings()
    run_etl(settings.sqlite_path, settings.duckdb_path)
    conn = duckdb_store.connect(settings.duckdb_path)

    samples_by_archetype = sample_traces(
        conn, n_per_archetype=n_per_archetype, seed=seed, archetype_id=archetype_id
    )
    all_samples = [s for pool in samples_by_archetype.values() for s in pool]
    if not all_samples:
        click.echo("no eligible traces to replay yet (run `shadow cluster` first)")
        return

    ladder = sorted_by_cost(load_ladder())
    run_id = f"run-{int(time.time())}"
    runner_config = RunnerConfig(
        budget_usd=budget_usd, checkpoint_path=settings.checkpoints_dir / f"{run_id}.json"
    )

    async def grader(sample: SampleRow, result: CallResult) -> bool:
        verdict = await judge_answer(
            "Is the candidate answer an acceptable substitute for the frontier answer?",
            sample.response_json,
            result.response_text,
        )
        return verdict.passed

    # Local-file span export always on (nothing leaves the machine); OTLP
    # export to a real collector is opt-in only, via env var (plan.md §4.5).
    tracer_provider = build_tracer_provider(
        settings.traces_path, otlp_endpoint=os.environ.get("SHADOWTRACE_OTLP_ENDPOINT")
    )
    tracer = tracer_provider.get_tracer("shadowtrace.replay")
    runner = ReplayRunner(ladder, grader, _live_call, runner_config, tracer=tracer)

    async def _do_run() -> tuple[RunResult, list[ChainReplayResult]]:
        run_result = await runner.run(run_id, all_samples)
        chain_results: list[ChainReplayResult] = []
        if chains:
            top_ids = top_archetypes_by_volume(conn, n=2)
            top_ids = [a for a in top_ids if archetype_id is None or a == archetype_id]
            for chain in sample_chains(conn, top_ids, seed=seed):
                for candidate in ladder:
                    cr = await runner.run_chain(run_id, chain, candidate, chain_budget_usd)
                    if cr is not None:
                        chain_results.append(cr)
        return run_result, chain_results

    result, chain_results = asyncio.run(_do_run())

    for cr in chain_results:
        conn.execute(
            "INSERT INTO replay_results (id, archetype_id, candidate, source_trace_id, verdict, "
            "cost_usd, latency_ms, grader, run_id, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                cr.id,
                cr.archetype_id,
                cr.candidate,
                cr.session_id,
                cr.verdict,
                cr.cost_usd,
                None,
                "chain",
                run_id,
                cr.ts,
            ],
        )

    sprt_config = SPRTConfig(p0=floor, p1=max(0.01, floor - 0.10))
    for arch_id in samples_by_archetype:
        for candidate in ladder:
            outcomes = [
                r.verdict == "pass"
                for r in result.results
                if r.archetype_id == arch_id and r.candidate == candidate.name
            ]
            if not outcomes:
                continue
            state = run_sprt(outcomes, sprt_config, max_samples=len(outcomes))
            conn.execute(
                "INSERT INTO sprt_verdicts (archetype_id, candidate, run_id, state, n_samples, "
                "n_pass, log_likelihood_ratio, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    arch_id,
                    candidate.name,
                    run_id,
                    state.verdict,
                    state.n,
                    state.n_pass,
                    state.log_likelihood_ratio,
                    int(time.time() * 1000),
                ],
            )

    click.echo(
        f"run={run_id} spend=${result.spend_usd:.2f} stopped={result.stopped_reason} "
        f"results={len(result.results)} chain_results={len(chain_results)}"
    )


@main.command()
@click.option("--raw", is_flag=True, help="read directly from the SQLite hot path, not DuckDB")
def report(raw: bool) -> None:
    """Summarize captured traffic and pipeline health."""
    settings = get_settings()
    if raw:
        conn = sqlite_db.connect(settings.sqlite_path)
        rows = conn.execute("SELECT source, count(*) AS n FROM traces GROUP BY source").fetchall()
        for row in rows:
            click.echo(f"{row['source']}: {row['n']}")
        return

    run_etl(settings.sqlite_path, settings.duckdb_path)
    conn = duckdb_store.connect(settings.duckdb_path)
    total = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
    click.echo(f"traces: {total}")
    archetype_rows = conn.execute(
        "SELECT a.label, count(*) FROM archetype_assignments aa "
        "JOIN archetypes a ON a.id = aa.archetype_id "
        "WHERE a.merged_into IS NULL GROUP BY a.label ORDER BY 2 DESC"
    ).fetchall()
    for label, count in archetype_rows:
        click.echo(f"  {label}: {count}")


@main.command(name="apply")
@click.argument("recommendation_id")
@click.option("--writer", type=click.Choice(sorted(_WRITERS)), required=True)
@click.option("--output-path", type=click.Path(path_type=Path), required=True)
@click.option("--yes", is_flag=True, help="skip the confirmation prompt")
def apply_cmd(recommendation_id: str, writer: str, output_path: Path, yes: bool) -> None:
    """Dry-run diff -> confirm -> write a recommended candidate as router config."""
    settings = get_settings()
    conn = duckdb_store.connect(settings.duckdb_path)
    row = conn.execute(
        "SELECT r.archetype_id, r.candidate, a.label FROM recommendations r "
        "JOIN archetypes a ON a.id = r.archetype_id WHERE r.id = ?",
        [recommendation_id],
    ).fetchone()
    if row is None:
        raise click.ClickException(f"no recommendation with id={recommendation_id!r}")
    archetype_id, candidate, archetype_label = row

    writer_instance = _WRITERS[writer](output_path)
    diff = apply_recommendation(
        conn, writer_instance, archetype_id, archetype_label, candidate, dry_run=True
    )
    click.echo(diff or "(no changes)")
    if not yes and not click.confirm("Apply this change?"):
        return

    result = apply_recommendation(
        conn, writer_instance, archetype_id, archetype_label, candidate, dry_run=False
    )
    assert not isinstance(result, str)
    click.echo(f"applied as policy {result.id}")


@main.command()
def metrics() -> None:
    """Dump the current in-process metrics snapshot as JSON."""
    click.echo(json.dumps(REGISTRY.snapshot(), indent=2))
