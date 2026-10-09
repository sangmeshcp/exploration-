"""Turns stored trace JSON into what a replay call sends and what a grader
compares.

Captured rows come in three request shapes:

  - transcript lane: one user turn, `{"role": "user", "content": ...}`
  - proxy lane: a full Messages/Chat request, `{"model": ..., "messages": [...]}`
  - chain replay: `{"messages": [...]}`, the carried-forward history

and content can be a plain string or a list of content blocks (`text`,
`tool_use`, `tool_result`, ...). Replay candidates get plain-text
user/assistant turns: tool_result blocks become text (their tool_use ids
refer to calls the candidate never made, so passing them through verbatim
would be rejected by the API), and other non-text blocks are dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

TOOL_RESULT_PREFIX = "[tool result] "


def _load(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content_to_text([content])
    if not isinstance(content, list):
        return str(content)
    # Adjacent text blocks are one continuous answer split by the API, so
    # they concatenate as-is; a tool result sits on its own line.
    out = ""
    previous = ""
    for block in content:
        if isinstance(block, str):
            kind, piece = "text", block
        elif isinstance(block, dict) and block.get("type") == "text":
            kind, piece = "text", str(block.get("text", ""))
        elif isinstance(block, dict) and block.get("type") == "tool_result":
            kind = "tool_result"
            piece = TOOL_RESULT_PREFIX + content_to_text(block.get("content"))
        else:
            continue
        if out and "tool_result" in (kind, previous):
            out += "\n"
        out += piece
        previous = kind
    return out


def request_messages(request_json_raw: Any) -> list[dict[str, str]]:
    """Plain-text messages for a replay call. Always non-empty, always
    starts with a user turn, never contains an empty content string."""
    data = _load(request_json_raw)
    raw_messages: list[Any]
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        raw_messages = data["messages"]
    elif isinstance(data, dict) and "content" in data:
        raw_messages = [{"role": data.get("role", "user"), "content": data["content"]}]
    else:
        raw_messages = [{"role": "user", "content": data if isinstance(data, str) else ""}]

    messages: list[dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = content_to_text(msg.get("content"))
        if not messages and role != "user":
            continue
        messages.append({"role": str(role), "content": text or "(empty)"})
    if not messages:
        messages = [{"role": "user", "content": "(empty)"}]
    return messages


def request_text(request_json_raw: Any) -> str:
    """The request as one prompt string, for providers without a chat API."""
    msgs = request_messages(request_json_raw)
    if len(msgs) == 1:
        return msgs[0]["content"]
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs)


def system_prompt(request_json_raw: Any) -> str | None:
    data = _load(request_json_raw)
    if isinstance(data, dict) and data.get("system"):
        return content_to_text(data["system"]) or None
    return None


@dataclass
class ParsedResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def parse_response(response_json_raw: Any) -> ParsedResponse:
    """Answer text and tool calls from a stored frontier response (Anthropic
    content blocks or an OpenAI-style `choices` payload) or a candidate's
    raw API response."""
    data = _load(response_json_raw)
    if isinstance(data, str):
        return ParsedResponse(text=data)
    if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
        message = data["choices"][0].get("message", {}) or {}
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {}) or {}
            args = _load(fn.get("arguments", {}))
            calls.append(
                {"name": fn.get("name", ""), "input": args if isinstance(args, dict) else {}}
            )
        return ParsedResponse(text=str(message.get("content") or ""), tool_calls=calls)
    content = data.get("content") if isinstance(data, dict) else data
    tool_calls: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_input = block.get("input")
                tool_calls.append(
                    {
                        "name": str(block.get("name", "")),
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    }
                )
    return ParsedResponse(text=content_to_text(content), tool_calls=tool_calls)


def replay_tools(
    request_json_raw: Any, reference_tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Tool definitions (Anthropic format) to offer a replay candidate.

    Proxy-lane requests carry their real `tools` list, which is reused as-is.
    Transcript-lane rows don't record tool definitions, so a permissive schema
    is inferred from the tools the frontier actually called (name plus the
    argument keys it used). That's enough for the candidate to be able to
    make the same next call, which is what the trajectory scorer compares.
    """
    data = _load(request_json_raw)
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        return [t for t in data["tools"] if isinstance(t, dict) and t.get("name")]
    inferred: dict[str, set[str]] = {}
    for call in reference_tool_calls:
        name = str(call.get("name", ""))
        if name:
            inferred.setdefault(name, set()).update((call.get("input") or {}).keys())
    return [
        {
            "name": name,
            "description": f"The {name} tool.",
            "input_schema": {"type": "object", "properties": {k: {} for k in sorted(keys)}},
        }
        for name, keys in sorted(inferred.items())
    ]


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]
