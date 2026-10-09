"""Lightweight in-process metrics registry using Prometheus-convention names.

Not a full OTel SDK — this is the local, dependency-free core described in
§4.5 that the observability tests assert against directly. An OTLP exporter
can be layered on top later without changing call sites, since everything
here goes through a single global registry keyed by metric name + label set.
"""

from __future__ import annotations

import math
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field


def _label_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


@dataclass
class _Histogram:
    values: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        return math.fsum(self.values)

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        data = sorted(self.values)
        idx = min(len(data) - 1, int(round(p * (len(data) - 1))))
        return data[idx]


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._histograms: dict[str, dict[tuple[tuple[str, str], ...], _Histogram]] = defaultdict(
            dict
        )

    def inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels)
        with self._lock:
            self._counters[name][key] = self._counters[name].get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels)
        with self._lock:
            self._gauges[name][key] = value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels)
        with self._lock:
            hist = self._histograms[name].setdefault(key, _Histogram())
            hist.observe(value)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = _label_key(labels)
        with self._lock:
            return self._counters[name].get(key, 0.0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = _label_key(labels)
        with self._lock:
            return self._gauges[name].get(key, 0.0)

    def get_percentile(self, name: str, p: float, labels: dict[str, str] | None = None) -> float:
        key = _label_key(labels)
        with self._lock:
            hist = self._histograms[name].get(key)
            return hist.percentile(p) if hist else 0.0

    def find(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        """A counter's or gauge's value, or None if it was never recorded —
        unlike `get_counter`/`get_gauge`, which can't tell "never recorded"
        from zero. Without labels, sums a counter across all its series."""
        key = _label_key(labels)
        with self._lock:
            if name in self._gauges:
                series = self._gauges[name]
                if labels is None:
                    return max(series.values()) if series else None
                return series.get(key)
            if name in self._counters:
                series = self._counters[name]
                if labels is None:
                    return sum(series.values())
                return series.get(key)
        return None

    def find_percentile(self, name: str, p: float) -> float | None:
        """`p`-quantile across every label set of a histogram; None if empty."""
        with self._lock:
            values = [v for h in self._histograms.get(name, {}).values() for v in h.values]
        if not values:
            return None
        return _Histogram(values).percentile(p)

    def counter_series(self, name: str) -> dict[tuple[tuple[str, str], ...], float]:
        with self._lock:
            return dict(self._counters.get(name, {}))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": {
                    name: {str(dict(k)): v for k, v in series.items()}
                    for name, series in self._counters.items()
                },
                "gauges": {
                    name: {str(dict(k)): v for k, v in series.items()}
                    for name, series in self._gauges.items()
                },
                "histograms": {
                    name: {
                        str(dict(k)): {"p50": h.percentile(0.5), "p99": h.percentile(0.99)}
                        for k, h in series.items()
                    }
                    for name, series in self._histograms.items()
                },
            }

    def to_prometheus(self) -> str:
        """Prometheus text exposition (format 0.0.4). Counters and gauges
        map 1:1; histograms are exposed as summaries (p50/p90/p99 quantiles
        plus `_sum`/`_count`) since raw observations, not buckets, are kept.
        This is the vendor-neutral seam: Prometheus, Grafana Agent, the
        Datadog agent and the OTel collector's prometheus receiver can all
        scrape it without shadowtrace knowing which one is on the other end.
        """
        lines: list[str] = []
        with self._lock:
            for name, series in sorted(self._counters.items()):
                metric = _prom_name(name)
                lines.append(f"# TYPE {metric} counter")
                for key, value in sorted(series.items()):
                    lines.append(f"{metric}{_prom_labels(key)} {_prom_value(value)}")
            for name, series in sorted(self._gauges.items()):
                metric = _prom_name(name)
                lines.append(f"# TYPE {metric} gauge")
                for key, value in sorted(series.items()):
                    lines.append(f"{metric}{_prom_labels(key)} {_prom_value(value)}")
            for name, hist_series in sorted(self._histograms.items()):
                metric = _prom_name(name)
                lines.append(f"# TYPE {metric} summary")
                for key, hist in sorted(hist_series.items()):
                    for q in (0.5, 0.9, 0.99):
                        labels = _prom_labels(key + (("quantile", str(q)),))
                        lines.append(f"{metric}{labels} {_prom_value(hist.percentile(q))}")
                    lines.append(f"{metric}_sum{_prom_labels(key)} {_prom_value(hist.total)}")
                    lines.append(f"{metric}_count{_prom_labels(key)} {hist.count}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Test helper."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


_PROM_INVALID = re.compile(r"[^a-zA-Z0-9_:]")


def _prom_name(name: str) -> str:
    cleaned = _PROM_INVALID.sub("_", name)
    return f"shadowtrace_{cleaned}"


def _prom_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _prom_labels(key: tuple[tuple[str, str], ...]) -> str:
    if not key:
        return ""
    inner = ",".join(f'{_PROM_INVALID.sub("_", k)}="{_prom_escape(str(v))}"' for k, v in key)
    return "{" + inner + "}"


def _prom_value(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


REGISTRY = MetricsRegistry()
