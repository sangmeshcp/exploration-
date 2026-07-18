"""Deterministic, cheapest-first graders (plan.md M3.4): JSON/schema
validity, code-parses, diff-applies, length/format constraints. These run
before the pairwise judge (judge.py) since they're free and catch a large
share of failures without spending replay budget on a grading call.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

DeterministicCheck = Callable[[str], bool]


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def is_valid_python(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@", re.MULTILINE)


def is_well_formed_unified_diff(text: str) -> bool:
    """Structural check only (does the diff have the right shape) — actually
    applying a diff to a source tree is out of scope for a grader that has
    to run on arbitrary replayed snippets with no source tree available."""
    if "---" not in text or "+++" not in text:
        return False
    return bool(_DIFF_HUNK_RE.search(text))


def within_length(min_chars: int = 1, max_chars: int = 100_000) -> DeterministicCheck:
    def check(text: str) -> bool:
        return min_chars <= len(text) <= max_chars

    return check


@dataclass
class DeterministicVerdict:
    passed: bool
    check_name: str
    applicable: bool  # False if this check didn't apply to the content (e.g. no JSON expected)


def run_deterministic_checks(
    text: str, checks: dict[str, DeterministicCheck]
) -> list[DeterministicVerdict]:
    return [
        DeterministicVerdict(passed=check(text), check_name=name, applicable=True)
        for name, check in checks.items()
    ]


def all_pass(verdicts: list[DeterministicVerdict]) -> bool:
    return all(v.passed for v in verdicts if v.applicable)
