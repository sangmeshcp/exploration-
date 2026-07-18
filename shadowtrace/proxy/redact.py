"""Secret/PII scrub applied before persistence (plan.md M1.4, review R3).

Two distinct mechanisms, deliberately kept separate:

1. Pattern scrub — known secret *formats* (AWS keys, JWTs, PEM blocks, ...).
   High confidence: matches are replaced with `[REDACTED:<pattern>]` inline,
   safe to drop from the persisted text entirely.

2. Entropy scan — a heuristic for *unknown* high-entropy strings that don't
   match a known pattern. These are NOT scrubbed, because code (hashes,
   minified JS, base64 blobs) is also high-entropy and dropping those rows
   would gut coding traces (review R3). Instead the row is flagged
   `quarantined`: stored (encrypted at the capture layer) but excluded from
   replay sampling and dashboard exports until a human clears it.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from shadowtrace.common.metrics import REGISTRY

# Ordered so more specific patterns are tried before generic ones.
PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "aws_secret_access_key": re.compile(
        r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?"
    ),
    "gcp_api_key": re.compile(r"\bAIza[0-9A-Za-z\-_]{20,40}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    "pem_block": re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]+?-----END [A-Z ]+-----"),
    "slack_token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "generic_kv_secret": re.compile(
        r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+=]{16,}['\"]?"
    ),
}

ENTROPY_THRESHOLD_BITS_PER_CHAR = 3.5
ENTROPY_MIN_TOKEN_LEN = 20
_TOKEN_RE = re.compile(rf"[A-Za-z0-9+/=_\-]{{{ENTROPY_MIN_TOKEN_LEN},}}")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def scrub_text(text: str) -> tuple[str, list[str]]:
    """Replace known-secret patterns. Returns (scrubbed_text, pattern_names_hit)."""
    if not text:
        return text, []
    start = time.perf_counter()
    flags: list[str] = []
    scrubbed = text
    for name, pattern in PATTERNS.items():
        if pattern.search(scrubbed):
            flags.append(name)
            scrubbed = pattern.sub(f"[REDACTED:{name}]", scrubbed)
            REGISTRY.inc("scrubbed_total", labels={"pattern": name})
    REGISTRY.observe("redaction_latency_ms", (time.perf_counter() - start) * 1000)
    return scrubbed, flags


def entropy_scan(text: str) -> list[str]:
    """Return high-entropy substrings that don't match a known secret pattern.

    These are candidates for quarantine, not for scrubbing.
    """
    if not text:
        return []
    hits = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if any(p.search(token) for p in PATTERNS.values()):
            continue  # already handled by pattern scrub
        if _shannon_entropy(token) >= ENTROPY_THRESHOLD_BITS_PER_CHAR:
            hits.append(token)
    return hits


@dataclass
class RedactionResult:
    request_json: Any
    response_json: Any
    flags: list[str] = field(default_factory=list)
    quarantined: bool = False


def _walk_and_scrub(value: Any, flags: list[str], entropy_hits: list[str]) -> Any:
    if isinstance(value, str):
        scrubbed, hit_patterns = scrub_text(value)
        flags.extend(hit_patterns)
        entropy_hits.extend(entropy_scan(scrubbed))
        return scrubbed
    if isinstance(value, dict):
        return {k: _walk_and_scrub(v, flags, entropy_hits) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_and_scrub(v, flags, entropy_hits) for v in value]
    return value


def redact_trace(request_json: Any, response_json: Any) -> RedactionResult:
    """Scrub + entropy-scan a full request/response pair before persistence."""
    flags: list[str] = []
    entropy_hits: list[str] = []
    scrubbed_request = _walk_and_scrub(request_json, flags, entropy_hits)
    scrubbed_response = _walk_and_scrub(response_json, flags, entropy_hits)

    quarantined = len(entropy_hits) > 0
    if quarantined:
        REGISTRY.inc("quarantined_total")
        flags.append("entropy_quarantine")

    # dedupe while preserving order
    seen: set[str] = set()
    deduped_flags = [f for f in flags if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]

    return RedactionResult(
        request_json=scrubbed_request,
        response_json=scrubbed_response,
        flags=deduped_flags,
        quarantined=quarantined,
    )
