"""Lightweight in-process metrics registry using Prometheus-convention names.

Not a full OTel SDK — this is the local, dependency-free core described in
§4.5 that the observability tests assert against directly. An OTLP exporter
can be layered on top later without changing call sites, since everything
here goes through a single global registry keyed by metric name + label set.
"""

from __future__ import annotations

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

    def reset(self) -> None:
        """Test helper."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


REGISTRY = MetricsRegistry()
