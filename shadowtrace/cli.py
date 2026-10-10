"""`shadow` CLI entrypoint (plan.md M0-M4).

Pipeline: ingest/up -> sync -> cluster -> run -> recommend -> apply.
Evaluation: review (human labels for judge verdicts), evals (scorer and
judge-agreement summary).
Monitoring: monitor (drift, data quality, performance), slo (SLOs, alerts,
error budgets), dashboard (web UI + /metrics), metrics (raw snapshot).

`up` runs capture, ETL, the SLO/alert loop and the dashboard in one process,
so the in-process metrics (capture lag, proxy latency, fail-open events) are
visible to the alert rules and the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import click
import uvicorn

from shadowtrace.common.config import get_settings
from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.tracing import build_tracer_provider, register_process_tracer_provider
from shadowtrace.dashboard.api import create_app as create_dashboard_app
from shadowtrace.db import duckdb_store
from shadowtrace.db import sqlite as sqlite_db
from shadowtrace.ingest.claude_transcripts import TranscriptWatcher
from shadowtrace.ingest.etl import backfill_prompt_text, run_etl
from shadowtrace.mining.archetypes import assign_archetypes
from shadowtrace.mining.cluster import cluster_embeddings
from shadowtrace.mining.embed import get_default_embedder
from shadowtrace.monitor import drift, performance, quality, slo
from shadowtrace.proxy.capture import CaptureWriter
from shadowtrace.proxy.server import ProxyConfig, create_app
from shadowtrace.recommend import engine as recommend_engine
from shadowtrace.recommend.apply import apply_recommendation, revert_policy
from shadowtrace.recommend.apply.claude_code import ClaudeCodeWriter
from shadowtrace.recommend.apply.litellm import LiteLLMWriter
from shadowtrace.recommend.apply.nanoclaw import NanoClawWriter
from shadowtrace.replay import store as replay_store
from shadowtrace.replay.graders import human_review
from shadowtrace.replay.graders.llm_judge import judge_from_env
from shadowtrace.replay.graders.scorers import default_suite
from shadowtrace.replay.ladder import load_ladder, sorted_by_cost
from shadowtrace.replay.providers import live_call as _live_call
from shadowtrace.replay.runner import (
    ChainReplayResult,
    ReplayRunner,
    RunnerConfig,
    RunResult,
)
from shadowtrace.replay.sampler import (
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


def _tracer_provider(settings: Any) -> Any:
    """Local JSON-lines span export by default, opt-in OTLP via
    SHADOWTRACE_OTLP_ENDPOINT; registered process-wide so code instrumented
    with `common.instrument` exports through it too."""
    provider = build_tracer_provider(
        settings.traces_path, otlp_endpoint=os.environ.get("SHADOWTRACE_OTLP_ENDPOINT")
    )
    register_process_tracer_provider(provider)
    return provider


def build_up_components(settings: Any) -> tuple[Any, TranscriptWatcher]:
    """Wires the proxy app and transcript watcher `shadow up` runs, both
    sharing one real tracer provider — split out from `up()` itself so this
    wiring is unit-testable without booting uvicorn."""
    tracer_provider = _tracer_provider(settings)
    config = ProxyConfig(settings=settings, tracer=tracer_provider.get_tracer("shadowtrace.proxy"))
    app = create_app(config)
    watcher = TranscriptWatcher(
        settings.claude_projects_dir,
        app.state.capture,
        tracer=tracer_provider.get_tracer("shadowtrace.ingest"),
    )
    return app, watcher


def evaluate_objectives(
    settings: Any, notifier: slo.AlertNotifier | None = None
) -> list[slo.Evaluation]:
    """One SLO/alert evaluation cycle: evaluate every objective, store the
    results for error-budget tracking, and notify on newly firing alerts."""
    conn = duckdb_store.connect(settings.duckdb_path)
    try:
        objectives = slo.load_objectives(settings.slo_rules_path)
        ctx = slo.SLIContext(conn=conn, home=settings.home, sqlite_path=settings.sqlite_path)
        evaluations = slo.evaluate(objectives, ctx)
        slo.record(conn, evaluations)
        for objective in objectives:
            if objective.kind == "slo":
                slo.compliance(conn, objective)
    finally:
        conn.close()
    if notifier is not None:
        notifier.process(evaluations)
    return evaluations


@main.command()
@click.option("--proxy-port", default=8787, show_default=True)
@click.option("--dashboard-port", default=8788, show_default=True, help="0 disables the dashboard")
@click.option("--etl-interval", default=300.0, show_default=True, help="seconds between ETL syncs")
@click.option(
    "--watch-interval", default=2.0, show_default=True, help="seconds between transcript scans"
)
@click.option(
    "--monitor-interval",
    default=60.0,
    show_default=True,
    help="seconds between SLO/alert evaluations",
)
@click.option("--notify", is_flag=True, help="desktop notifications for critical alerts")
def up(
    proxy_port: int,
    dashboard_port: int,
    etl_interval: float,
    watch_interval: float,
    monitor_interval: float,
    notify: bool,
) -> None:
    """Start the proxy (secondary/API-key lane), transcript watcher (primary
    lane), periodic ETL, SLO/alert evaluation and the dashboard, all in one
    process."""
    settings = get_settings()
    settings.ensure_dirs()
    app, watcher = build_up_components(settings)
    notifier = slo.AlertNotifier(desktop=notify)

    async def _run() -> None:
        async def etl_loop() -> None:
            while True:
                try:
                    run_etl(settings.sqlite_path, settings.duckdb_path)
                except Exception:
                    logger.exception("periodic ETL failed")
                await asyncio.sleep(etl_interval)

        async def monitor_loop() -> None:
            while True:
                await asyncio.sleep(monitor_interval)
                try:
                    await asyncio.to_thread(evaluate_objectives, settings, notifier)
                except Exception:
                    logger.exception("SLO evaluation failed")

        servers = [
            uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=proxy_port, log_level="warning")
            )
        ]
        if dashboard_port:
            servers.append(
                uvicorn.Server(
                    uvicorn.Config(
                        create_dashboard_app(settings),
                        host="127.0.0.1",
                        port=dashboard_port,
                        log_level="warning",
                    )
                )
            )

        await asyncio.gather(
            *(server.serve() for server in servers),
            watcher.run_forever(watch_interval),
            etl_loop(),
            monitor_loop(),
        )

    dashboard_note = f", dashboard + /metrics on :{dashboard_port}" if dashboard_port else ""
    click.echo(
        f"shadowtrace up: proxy on :{proxy_port}{dashboard_note}, "
        f"watching {settings.claude_projects_dir}"
    )
    asyncio.run(_run())


@main.command()
@click.option(
    "--reset",
    is_flag=True,
    help=(
        "Wipe the capture store first, then re-ingest everything from scratch. "
        "Re-running `ingest` normally skips messages it's already seen (deduped "
        "by message id) — that's the right default for topping up, but it means "
        "a transcript-parser improvement never touches already-captured rows "
        "unless you start over. Use --reset after upgrading shadowtrace if "
        "clustering looks worse than it should."
    ),
)
def ingest(reset: bool) -> None:
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
    if reset:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{settings.sqlite_path}{suffix}").unlink(missing_ok=True)
        settings.duckdb_path.unlink(missing_ok=True)
        click.echo("reset: capture store wiped, re-ingesting from scratch")
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


@main.command(name="run")
@click.option(
    "--budget", "budget_usd", type=float, required=True, help="hard spend cap for the run, in USD"
)
@click.option("--archetype", "archetype_id", default=None, help="replay only this archetype id")
@click.option("--n-per-archetype", default=10, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--floor", default=0.90, show_default=True, help="per-archetype SPRT pass-rate floor")
@click.option(
    "--chains/--no-chains",
    default=True,
    show_default=True,
    help="also replay whole sessions (multi-turn chains) for the 2 busiest archetypes",
)
@click.option(
    "--chain-budget",
    "chain_budget_usd",
    type=float,
    default=0.50,
    show_default=True,
    help="spend cap per replayed chain, in USD, separate from --budget",
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

    user_ladder = settings.ladder_config_path
    ladder = sorted_by_cost(load_ladder(user_ladder if user_ladder.exists() else None))
    run_id = f"run-{int(time.time())}"
    runner_config = RunnerConfig(
        budget_usd=budget_usd,
        checkpoint_path=settings.checkpoints_dir / f"{run_id}.json",
        grader_name="suite",
    )

    judge_fn = judge_from_env()
    grader = default_suite(judge_fn)
    click.echo(
        f"judge: {judge_fn.model if judge_fn is not None else 'local heuristic (no API key set)'}"
    )

    tracer = _tracer_provider(settings).get_tracer("shadowtrace.replay")
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

    replay_store.save_replay_results(conn, result.results)
    replay_store.save_chain_results(conn, chain_results)

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
    click.echo("next: `shadow recommend` to turn passing verdicts into recommendations")


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
@click.argument("policy_id")
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="the file the policy was applied to",
)
@click.option("--reason", default="manual revert", show_default=True)
def revert(policy_id: str, output_path: Path, reason: str) -> None:
    """Restore a config file to exactly what it was before `shadow apply`."""
    settings = get_settings()
    conn = duckdb_store.connect(settings.duckdb_path)
    row = conn.execute(
        "SELECT writer FROM applied_policies WHERE id = ? AND reverted_ts IS NULL", [policy_id]
    ).fetchone()
    if row is None:
        raise click.ClickException(f"no active applied policy with id={policy_id!r}")
    writer = _WRITERS[row[0]](output_path)
    revert_policy(conn, {row[0]: writer}, policy_id, reason=reason)
    click.echo(f"reverted policy {policy_id}; {output_path} restored")


@main.command()
def metrics() -> None:
    """Dump the current in-process metrics snapshot as JSON."""
    click.echo(json.dumps(REGISTRY.snapshot(), indent=2))


@main.command()
@click.option(
    "--access-mode",
    type=click.Choice(["subscription", "api_key"]),
    default="subscription",
    show_default=True,
    help="subscription values savings as quota headroom, api_key as $/month",
)
@click.option(
    "--quota-period-tokens",
    default=5_000_000,
    show_default=True,
    help="tokens in one quota period, for the headroom estimate",
)
def recommend(access_mode: str, quota_period_tokens: int) -> None:
    """Turn the latest passing SPRT verdicts into recommendations."""
    settings = get_settings()
    conn = duckdb_store.connect(settings.duckdb_path)
    recs = recommend_engine.build_recommendations(
        conn, access_mode=access_mode, quota_period_tokens=quota_period_tokens
    )
    if not recs:
        click.echo("no new recommendations (no new passing verdicts; run `shadow run` first)")
        return
    unit = "% quota" if access_mode == "subscription" else " $/mo"
    for rec in recs:
        click.echo(
            f"{rec.id}  {rec.archetype_id} -> {rec.candidate}  "
            f"pass={rec.pass_rate:.0%} [{rec.pass_rate_ci_low:.0%}, {rec.pass_rate_ci_high:.0%}]  "
            f"saves ~{rec.value:.1f}{unit}"
        )
    click.echo("apply one with: shadow apply <id> --writer claude_code --output-path <file>")


@main.command()
@click.option(
    "--enqueue", "enqueue_n", default=10, show_default=True, help="queue size per archetype"
)
@click.option("--limit", default=20, show_default=True, help="items to review this session")
@click.option("--stats", "stats_only", is_flag=True, help="only print judge/human agreement")
def review(enqueue_n: int, limit: int, stats_only: bool) -> None:
    """Label judge-decided replay results by hand, to measure how far the
    judge can be trusted (pass / fail / skip / quit)."""
    settings = get_settings()
    conn = duckdb_store.connect(settings.duckdb_path)
    if not stats_only:
        added = human_review.enqueue(conn, per_archetype=enqueue_n)
        items = human_review.pending(conn, limit=limit)
        click.echo(f"queued {added} new; {len(items)} to review now")
        for i, item in enumerate(items, 1):
            click.echo(f"\n--- {i}/{len(items)}  [{item.archetype_label}]  {item.candidate} ---")
            click.echo(f"REQUEST:\n{item.request[:1500]}\n")
            click.echo(f"FRONTIER ANSWER:\n{item.frontier_answer[:1500]}\n")
            click.echo(f"CANDIDATE ANSWER:\n{item.candidate_answer[:1500]}\n")
            choice = click.prompt(
                "acceptable substitute? [p]ass / [f]ail / [s]kip / [q]uit",
                type=click.Choice(["p", "f", "s", "q"]),
                show_choices=False,
            )
            if choice == "q":
                break
            if choice == "s":
                continue
            human_review.record(conn, item.replay_result_id, passed=choice == "p")
    _print_agreement(conn)


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def _print_agreement(conn: Any) -> None:
    stats = human_review.agreement(conn)
    click.echo(
        f"\njudge vs human (n={stats.n}): agreement={_fmt(stats.agreement)} "
        f"kappa={_fmt(stats.cohen_kappa, pct=False)} "
        f"false-pass={_fmt(stats.false_pass_rate)} false-fail={_fmt(stats.false_fail_rate)}"
    )


@main.command()
def evals() -> None:
    """Per-scorer results by archetype and candidate, plus judge reliability."""
    settings = get_settings()
    conn = duckdb_store.connect(settings.duckdb_path)
    rows = replay_store.scorer_summary(conn)
    if not rows:
        click.echo("no scored replay results yet (run `shadow run`)")
    for r in rows:
        click.echo(
            f"{r['archetype_label']:<28} {r['candidate']:<26} {r['scorer']:<11} "
            f"n={r['n']:<4} mean={r['mean_value']:.2f} pass={r['pass_rate']:.0%}"
        )
    _print_agreement(conn)


@main.command()
@click.option("--json", "as_json", is_flag=True)
def monitor(as_json: bool) -> None:
    """Drift, data quality and per-candidate performance over time."""
    settings = get_settings()
    run_etl(settings.sqlite_path, settings.duckdb_path)
    conn = duckdb_store.connect(settings.duckdb_path)
    drift_results = drift.drift_report(conn)
    checks = quality.data_quality(conn)
    perf = performance.performance_report(conn)
    if as_json:
        from dataclasses import asdict

        click.echo(
            json.dumps(
                {
                    "drift": [asdict(d) for d in drift_results],
                    "quality": [asdict(c) for c in checks],
                    "performance": [asdict(p) for p in perf],
                },
                indent=2,
            )
        )
        return
    click.echo("DRIFT (last 7d vs prior 28d)")
    for d in drift_results:
        detail = f"psi={d.psi:.3f}" if d.psi is not None else f"n={d.baseline_n}/{d.recent_n}"
        click.echo(f"  {d.feature:<14} {d.status:<18} {detail}")
    click.echo("DATA QUALITY (last 7d)")
    for c in checks:
        state = "no data" if c.ok is None else "ok" if c.ok else "FAIL"
        click.echo(f"  {c.name:<28} {state:<8} {_fmt(c.value)} (max {c.threshold:.0%})")
    click.echo("PERFORMANCE (pass rate, last 7d vs prior 7d)")
    if not perf:
        click.echo("  no replay results or spot-checks yet")
    for p in perf:
        flag = "  DEGRADED" if p.degraded else ""
        click.echo(
            f"  {p.archetype_label:<28} {p.candidate:<26} "
            f"{_fmt(p.recent_pass_rate)} (n={p.recent_n}) vs "
            f"{_fmt(p.previous_pass_rate)} (n={p.previous_n}){flag}"
        )


@main.command(name="slo")
@click.option("--notify", is_flag=True, help="desktop notification for critical breaches")
@click.option("--json", "as_json", is_flag=True)
def slo_cmd(notify: bool, as_json: bool) -> None:
    """Evaluate SLOs and alert rules once and show error budgets. Metrics that
    live only inside `shadow up` (capture lag, proxy latency, ...) show as
    no_data here; the dashboard served by `shadow up` has them."""
    settings = get_settings()
    settings.ensure_dirs()
    run_etl(settings.sqlite_path, settings.duckdb_path)
    evaluations = evaluate_objectives(settings, slo.AlertNotifier(desktop=notify))
    conn = duckdb_store.connect(settings.duckdb_path)
    objectives = {o.name: o for o in slo.load_objectives(settings.slo_rules_path)}
    budgets = {
        ev.name: slo.compliance(conn, objectives[ev.name]) for ev in evaluations if ev.kind == "slo"
    }
    if as_json:
        click.echo(json.dumps([ev.to_dict() for ev in evaluations], indent=2))
        return
    for ev in evaluations:
        value = "-" if ev.value is None else f"{ev.value:.3g}"
        line = (
            f"{ev.status.upper():<8} {ev.kind:<5} {ev.severity:<8} {ev.name:<24} "
            f"{ev.sli} = {value} (healthy {ev.op} {ev.threshold:g})"
        )
        budget = budgets.get(ev.name)
        if budget is not None and budget.attainment is not None:
            line += (
                f"  attainment {budget.attainment:.1%} of target {budget.target:.0%}, "
                f"budget left {_fmt(budget.error_budget_remaining)}"
            )
        click.echo(line)
    firing = [ev for ev in evaluations if ev.status == "breach"]
    if firing:
        raise SystemExit(1)


@main.command()
@click.option("--port", default=8788, show_default=True)
def dashboard(port: int) -> None:
    """Serve the dashboard API, /metrics and the built web UI (if
    dashboard/web/dist exists). `shadow up` already does this on :8788."""
    settings = get_settings()
    settings.ensure_dirs()
    click.echo(f"dashboard on http://127.0.0.1:{port}")
    uvicorn.run(create_dashboard_app(settings), host="127.0.0.1", port=port, log_level="warning")
