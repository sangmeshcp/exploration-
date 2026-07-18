"""Candidate ladder registry + price table (plan.md M3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Candidate:
    name: str
    provider: str  # anthropic|openai|openrouter|ollama|vllm
    price_per_mtok_in: float
    price_per_mtok_out: float
    local: bool = False

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        if self.local:
            return 0.0
        return (tokens_in / 1_000_000) * self.price_per_mtok_in + (
            tokens_out / 1_000_000
        ) * self.price_per_mtok_out

    def avg_price_per_mtok(self) -> float:
        return (self.price_per_mtok_in + self.price_per_mtok_out) / 2


def _default_ladder_yaml_text() -> str:
    return resources.files("shadowtrace.replay").joinpath("ladder.yaml").read_text()


def parse_ladder_yaml(text: str) -> list[Candidate]:
    data = yaml.safe_load(text) or {}
    candidates = []
    for entry in data.get("candidates", []):
        candidates.append(
            Candidate(
                name=entry["name"],
                provider=entry["provider"],
                price_per_mtok_in=float(entry.get("price_per_mtok_in", 0.0)),
                price_per_mtok_out=float(entry.get("price_per_mtok_out", 0.0)),
                local=bool(entry.get("local", False)),
            )
        )
    return candidates


def load_ladder(path: Path | None = None) -> list[Candidate]:
    text = path.read_text() if path is not None else _default_ladder_yaml_text()
    return parse_ladder_yaml(text)


def sorted_by_cost(candidates: list[Candidate]) -> list[Candidate]:
    """Local (cost $0) candidates first, then ascending average price —
    the order the sampler/runner should try candidates in so budget is
    spent on the cheapest viable option first."""
    return sorted(candidates, key=lambda c: (not c.local, c.avg_price_per_mtok()))
