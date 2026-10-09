"""SLOs, alert rules and error budgets.

Three pieces:

  SLIs        named functions returning the current value of one indicator
              (or None when there's no data), drawn from the in-process
              metrics registry, the analytics store and the filesystem.
              `SLI_REGISTRY` is open: register_sli("name", fn) adds one.
  objectives  declarative rules from `slo_rules.yaml` (or the user's copy),
              each "SLI <op> threshold" plus a kind (slo|alert) and severity.
  evaluation  `evaluate()` checks every objective once; `record()` stores the
              results so `compliance()` can report SLO attainment and the
              remaining error budget over each objective's window.

In-process SLIs (capture lag, proxy latency, fail-open events, ...) only have
data inside the process that produced them, which is why `shadow up` runs the
evaluation loop and serves the dashboard itself. A one-shot `shadow slo` sees
the store-backed SLIs and reports the rest as no_data.
"""

from __future__ import annotations

import operator
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY, MetricsRegistry

logger = get_logger("monitor.slo")

DAY_MS = 86_400_000

_OPS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
}


@dataclass
class SLIContext:
    conn: Any | None = None
    registry: MetricsRegistry = field(default_factory=lambda: REGISTRY)
    home: Path | None = None
    sqlite_path: Path | None = None
    now_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    _cache: dict[str, Any] = field(default_factory=dict)

    def cached(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]


SLIFn = Callable[[SLIContext], float | None]
SLI_REGISTRY: dict[str, SLIFn] = {}


def register_sli(name: str) -> Callable[[SLIFn], SLIFn]:
    def decorator(fn: SLIFn) -> SLIFn:
        SLI_REGISTRY[name] = fn
        return fn

    return decorator


def _counter(name: str) -> SLIFn:
    def fn(ctx: SLIContext) -> float | None:
        return ctx.registry.find(name)

    return fn


for _name in ("fail_open_events_total", "spill_rows_total", "judge_errors_total"):
    SLI_REGISTRY[_name] = _counter(_name)


@register_sli("capture_lag_seconds")
def _capture_lag(ctx: SLIContext) -> float | None:
    return ctx.registry.find("capture_lag_seconds")


@register_sli("proxy_added_latency_p99_ms")
def _proxy_p99(ctx: SLIContext) -> float | None:
    return ctx.registry.find_percentile("proxy_added_latency_ms", 0.99)


@register_sli("operation_error_ratio")
def _operation_error_ratio(ctx: SLIContext) -> float | None:
    series = ctx.registry.counter_series("operations_total")
    total = sum(series.values())
    if not total:
        return None
    errors = sum(v for k, v in series.items() if ("status", "error") in k)
    return errors / total


@register_sli("disk_free_gb")
def _disk_free(ctx: SLIContext) -> float | None:
    if ctx.home is None:
        return None
    path = ctx.home
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


@register_sli("etl_lag_seconds")
def _etl_lag(ctx: SLIContext) -> float | None:
    if ctx.conn is None or ctx.sqlite_path is None or not ctx.sqlite_path.exists():
        return None
    import sqlite3

    src = sqlite3.connect(f"file:{ctx.sqlite_path}?mode=ro", uri=True)
    try:
        captured_row = src.execute("SELECT max(ts) FROM traces").fetchone()
    finally:
        src.close()
    if captured_row[0] is None:
        return None
    captured = int(captured_row[0])
    analytics_row = ctx.conn.execute("SELECT max(ts) FROM traces").fetchone()
    if analytics_row[0] is None:
        return (ctx.now_ms - captured) / 1000
    return max(0.0, (captured - int(analytics_row[0])) / 1000)


def _drift(ctx: SLIContext) -> dict[str, float | None]:
    from shadowtrace.monitor.drift import drift_report

    def compute() -> dict[str, float | None]:
        return {r.feature: r.psi for r in drift_report(ctx.conn, now_ms=ctx.now_ms)}

    result: dict[str, float | None] = ctx.cached("drift", compute)
    return result


@register_sli("archetype_mix_psi")
def _archetype_psi(ctx: SLIContext) -> float | None:
    return None if ctx.conn is None else _drift(ctx).get("archetype_mix")


@register_sli("prompt_length_psi")
def _length_psi(ctx: SLIContext) -> float | None:
    return None if ctx.conn is None else _drift(ctx).get("prompt_length")


def _quality(ctx: SLIContext) -> dict[str, float | None]:
    from shadowtrace.monitor.quality import data_quality

    def compute() -> dict[str, float | None]:
        return {c.name: c.value for c in data_quality(ctx.conn, now_ms=ctx.now_ms)}

    result: dict[str, float | None] = ctx.cached("quality", compute)
    return result


@register_sli("unassigned_traces_ratio")
def _unassigned(ctx: SLIContext) -> float | None:
    return None if ctx.conn is None else _quality(ctx).get("unassigned_traces_ratio")


@register_sli("missing_token_counts_ratio")
def _missing_tokens(ctx: SLIContext) -> float | None:
    return None if ctx.conn is None else _quality(ctx).get("missing_token_counts_ratio")


@register_sli("degraded_candidate_pairs")
def _degraded(ctx: SLIContext) -> float | None:
    if ctx.conn is None:
        return None
    from shadowtrace.monitor.performance import performance_report

    report = ctx.cached("performance", lambda: performance_report(ctx.conn, now_ms=ctx.now_ms))
    if not any(p.z is not None for p in report):
        return None
    return float(sum(1 for p in report if p.degraded))


def _agreement(ctx: SLIContext) -> Any:
    from shadowtrace.replay.graders.human_review import agreement

    return ctx.cached("agreement", lambda: agreement(ctx.conn))


@register_sli("judge_human_agreement")
def _judge_agreement(ctx: SLIContext) -> float | None:
    if ctx.conn is None:
        return None
    value: float | None = _agreement(ctx).agreement
    return value


@register_sli("judge_cohen_kappa")
def _judge_kappa(ctx: SLIContext) -> float | None:
    if ctx.conn is None:
        return None
    value: float | None = _agreement(ctx).cohen_kappa
    return value


@register_sli("judge_false_pass_rate")
def _judge_false_pass(ctx: SLIContext) -> float | None:
    if ctx.conn is None:
        return None
    value: float | None = _agreement(ctx).false_pass_rate
    return value


@register_sli("policies_reverted_7d")
def _policies_reverted(ctx: SLIContext) -> float | None:
    if ctx.conn is None:
        return None
    row = ctx.conn.execute(
        "SELECT count(*) FROM applied_policies WHERE reverted_ts >= ?",
        [ctx.now_ms - 7 * DAY_MS],
    ).fetchone()
    return float(row[0])


@register_sli("expired_applied_policies")
def _expired_policies(ctx: SLIContext) -> float | None:
    if ctx.conn is None:
        return None
    row = ctx.conn.execute(
        """
        SELECT count(*) FROM applied_policies p
        WHERE p.reverted_ts IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM recommendations r
              WHERE r.archetype_id = p.archetype_id AND r.candidate = p.candidate
                AND r.expires_ts > ?
          )
        """,
        [ctx.now_ms],
    ).fetchone()
    return float(row[0])


# --- objectives -------------------------------------------------------------


@dataclass
class Objective:
    name: str
    sli: str
    op: str
    threshold: float
    kind: str = "alert"  # slo | alert
    severity: str = "warning"  # info | warning | critical
    target: float | None = None
    window_days: int = 28
    description: str = ""

    def healthy(self, value: float) -> bool:
        return _OPS[self.op](value, self.threshold)


def parse_objectives(text: str) -> list[Objective]:
    data = yaml.safe_load(text) or {}
    objectives = []
    for entry in data.get("objectives", []):
        obj = Objective(
            name=str(entry["name"]),
            sli=str(entry["sli"]),
            op=str(entry["op"]),
            threshold=float(entry["threshold"]),
            kind=str(entry.get("kind", "alert")),
            severity=str(entry.get("severity", "warning")),
            target=float(entry["target"]) if entry.get("target") is not None else None,
            window_days=int(entry.get("window_days", 28)),
            description=str(entry.get("description", "")),
        )
        if obj.op not in _OPS:
            raise ValueError(f"objective {obj.name!r}: unknown op {obj.op!r}")
        if obj.sli not in SLI_REGISTRY:
            raise ValueError(f"objective {obj.name!r}: unknown SLI {obj.sli!r}")
        if obj.kind == "slo" and obj.target is None:
            raise ValueError(f"objective {obj.name!r}: kind slo needs a target")
        objectives.append(obj)
    return objectives


def load_objectives(path: Path | None = None) -> list[Objective]:
    if path is not None and path.exists():
        return parse_objectives(path.read_text())
    text = resources.files("shadowtrace.monitor").joinpath("slo_rules.yaml").read_text()
    return parse_objectives(text)


@dataclass
class Evaluation:
    name: str
    kind: str
    severity: str
    sli: str
    value: float | None
    op: str
    threshold: float
    status: str  # ok | breach | no_data | error
    description: str
    ts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(objectives: list[Objective], ctx: SLIContext) -> list[Evaluation]:
    results = []
    for obj in objectives:
        try:
            value = SLI_REGISTRY[obj.sli](ctx)
            status = "no_data" if value is None else "ok" if obj.healthy(value) else "breach"
        except Exception:
            logger.exception("SLI %s failed", obj.sli)
            value, status = None, "error"
        results.append(
            Evaluation(
                name=obj.name,
                kind=obj.kind,
                severity=obj.severity,
                sli=obj.sli,
                value=value,
                op=obj.op,
                threshold=obj.threshold,
                status=status,
                description=obj.description,
                ts=ctx.now_ms,
            )
        )
        REGISTRY.set_gauge(
            "slo_breached", 1.0 if status == "breach" else 0.0, labels={"objective": obj.name}
        )
    return results


def record(conn: Any, evaluations: list[Evaluation]) -> None:
    """Store evaluations that had data; no_data/error don't count for or
    against an SLO's budget."""
    for ev in evaluations:
        if ev.status not in ("ok", "breach"):
            continue
        conn.execute(
            "INSERT INTO slo_evaluations (objective, ts, value, ok) VALUES (?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [ev.name, ev.ts, ev.value, ev.status == "ok"],
        )


@dataclass
class Compliance:
    name: str
    target: float
    window_days: int
    n: int
    attainment: float | None
    error_budget_remaining: float | None  # 1.0 = untouched, <= 0 = exhausted


def compliance(conn: Any, objective: Objective, now_ms: int | None = None) -> Compliance:
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    target = objective.target if objective.target is not None else 1.0
    n, ok = conn.execute(
        "SELECT count(*), count(*) FILTER (WHERE ok) FROM slo_evaluations "
        "WHERE objective = ? AND ts >= ?",
        [objective.name, now - objective.window_days * DAY_MS],
    ).fetchone()
    if not n:
        return Compliance(objective.name, target, objective.window_days, 0, None, None)
    attainment = ok / n
    allowed = 1 - target
    if allowed <= 0:
        return Compliance(
            objective.name,
            target,
            objective.window_days,
            n,
            attainment,
            1.0 if attainment >= 1.0 else 0.0,
        )
    budget = 1 - (1 - attainment) / allowed
    REGISTRY.set_gauge("slo_error_budget_remaining", budget, labels={"objective": objective.name})
    return Compliance(objective.name, target, objective.window_days, n, attainment, budget)


# --- notification -----------------------------------------------------------


def desktop_notify(title: str, body: str) -> bool:
    """Best effort: notify-send (Linux) or osascript (macOS). Never raises."""
    try:
        system = platform.system()
        if system == "Linux" and shutil.which("notify-send"):
            subprocess.run(["notify-send", title, body], check=False, timeout=5)
            return True
        if system == "Darwin" and shutil.which("osascript"):
            script = f"display notification {body!r} with title {title!r}"
            subprocess.run(["osascript", "-e", script], check=False, timeout=5)
            return True
    except Exception:
        logger.exception("desktop notification failed")
    return False


class AlertNotifier:
    """Notifies on transitions only (ok/no_data -> breach), so a breach that
    persists across evaluation cycles notifies once, not every minute."""

    def __init__(
        self,
        desktop: bool = False,
        notify_fn: Callable[[str, str], bool] = desktop_notify,
    ) -> None:
        self.desktop = desktop
        self.notify_fn = notify_fn
        self._firing: set[str] = set()

    def process(self, evaluations: list[Evaluation]) -> list[Evaluation]:
        newly_firing = []
        for ev in evaluations:
            if ev.status == "breach" and ev.name not in self._firing:
                self._firing.add(ev.name)
                newly_firing.append(ev)
                REGISTRY.inc("alerts_fired_total", labels={"objective": ev.name})
                logger.warning(
                    "ALERT %s [%s]: %s = %s (healthy when %s %s) — %s",
                    ev.name,
                    ev.severity,
                    ev.sli,
                    ev.value,
                    ev.op,
                    ev.threshold,
                    ev.description,
                )
                if self.desktop and ev.severity == "critical":
                    self.notify_fn(f"shadowtrace: {ev.name}", ev.description)
            elif ev.status == "ok" and ev.name in self._firing:
                self._firing.discard(ev.name)
                logger.info("RESOLVED %s: %s = %s", ev.name, ev.sli, ev.value)
        return newly_firing
