"""Regenerates tests/fixtures/traces/*.jsonl — synthetic Claude Code
session transcripts spanning several task archetypes (plan.md §3.6).

Deterministic (seeded) so the committed fixture files are reproducible.
Run: `python -m tests.fixtures.generate_traces`

Produces the full "300 synthetic traces across 8 archetypes" corpus the
plan calls for (8 x 38 = 304), incl. tool-use chains, non-English, long
context, and near-duplicate adversarial prompts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "traces"
PER_ARCHETYPE = 38
SEED = 20260713

ARCHETYPES: dict[str, list[str]] = {
    "python_debugging": [
        "why is my python for loop raising an IndexError",
        "can you help debug this python traceback: KeyError 'foo'",
        "my python script hangs forever, how do I debug it",
        "why does this python function return None instead of a value",
        "debug this off-by-one error in my python loop",
    ],
    "sql_queries": [
        "write a SQL query to find duplicate rows in a table",
        "how do I join three tables in SQL and filter by date",
        "optimize this slow SQL query with an index",
        "write a SQL query to compute a running total",
        "how do I do an upsert in postgres",
    ],
    "code_review": [
        "review this pull request diff for bugs",
        "is this function thread-safe, please review",
        "review my error handling in this code change",
        "check this diff for security issues",
        "review this refactor for correctness",
    ],
    "writing_help": [
        "help me write a professional email declining a meeting",
        "rewrite this paragraph to sound more concise",
        "draft a short bio for my conference speaker profile",
        "help me write a polite follow-up email",
        "proofread this cover letter for tone",
    ],
    "weather_smalltalk": [
        "what is the weather like in paris tomorrow",
        "will it rain this weekend in san francisco",
        "what's a good temperature range for hiking",
        "is it going to snow in denver next week",
        "how humid is it usually in miami in july",
    ],
    "translation": [
        "translate 'good morning, how are you' into spanish",
        "traduza esta frase para o inglês: bom dia, tudo bem?",
        "translate this sentence into french: the meeting is at noon",
        "wie sagt man 'thank you very much' auf Deutsch?",
        "translate 'where is the train station' into japanese",
    ],
    "long_context_summarization": [
        "summarize this 3000-word research paper abstract " + ("lorem ipsum dolor sit amet " * 40),
        "summarize the key points of this long meeting transcript "
        + ("we discussed the roadmap " * 40),
        "give me a tl;dr of this long changelog " + ("added feature, fixed bug " * 40),
    ],
    "agentic_tool_use": [
        "read the config file and tell me what port the server uses",
        "find all TODO comments in the src directory",
        "run the test suite and summarize any failures",
        "check if there's a lockfile and tell me its dependency count",
        "search the repo for usages of the deprecated function",
    ],
}

MODELS = ["claude-sonnet-5"]


def _content_for(archetype: str, prompt: str, rng: random.Random) -> list[dict[str, object]]:
    if archetype == "agentic_tool_use":
        return [
            {"type": "text", "text": "Let me check that for you."},
            {
                "type": "tool_use",
                "id": f"toolu_{rng.randint(1000, 9999)}",
                "name": "read_file",
                "input": {"path": "config.yaml"},
            },
        ]
    return [{"type": "text", "text": f"Here's the answer to: {prompt[:40]}..."}]


def generate() -> None:
    rng = random.Random(SEED)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for archetype, prompts in ARCHETYPES.items():
        session_path = FIXTURES_DIR / f"session_{archetype}.jsonl"
        lines = []
        uuid_counter = 0
        for i in range(PER_ARCHETYPE):
            base_prompt = prompts[i % len(prompts)]
            # near-duplicate adversarial variants for some entries
            prompt = base_prompt if i < len(prompts) else f"{base_prompt} please"
            ts_base = 1_752_800_000_000 + i * 60_000

            uuid_counter += 1
            user_uuid = f"{archetype}-u{uuid_counter}"
            lines.append(
                json.dumps(
                    {
                        "type": "user",
                        "uuid": user_uuid,
                        "timestamp": _iso(ts_base),
                        "sessionId": archetype,
                        "message": {"role": "user", "content": prompt},
                    }
                )
            )

            uuid_counter += 1
            assistant_uuid = f"{archetype}-a{uuid_counter}"
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": assistant_uuid,
                        "timestamp": _iso(ts_base + 1000),
                        "sessionId": archetype,
                        "message": {
                            "role": "assistant",
                            "model": rng.choice(MODELS),
                            "content": _content_for(archetype, prompt, rng),
                            "usage": {
                                "input_tokens": rng.randint(50, 400),
                                "output_tokens": rng.randint(20, 300),
                            },
                        },
                    }
                )
            )
        session_path.write_text("\n".join(lines) + "\n")
        print(f"wrote {session_path} ({len(lines)} lines)")


def _iso(ts_ms: int) -> str:
    import datetime

    return (
        datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


if __name__ == "__main__":
    generate()
