import json

from shadowtrace.replay.messages import (
    TOOL_RESULT_PREFIX,
    content_to_text,
    parse_response,
    replay_tools,
    request_messages,
    request_text,
    system_prompt,
    to_openai_tools,
)


def test_transcript_turn_becomes_single_user_message() -> None:
    raw = json.dumps({"role": "user", "content": "fix the bug"})
    assert request_messages(raw) == [{"role": "user", "content": "fix the bug"}]
    assert request_text(raw) == "fix the bug"


def test_tool_result_blocks_become_text_and_other_blocks_drop() -> None:
    raw = json.dumps(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "port: 8080"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "what next?"},
            ],
        }
    )
    (msg,) = request_messages(raw)
    assert msg["content"] == f"{TOOL_RESULT_PREFIX}port: 8080\nwhat next?"


def test_proxy_request_keeps_history_system_and_drops_leading_assistant() -> None:
    raw = json.dumps(
        {
            "model": "claude-sonnet-5",
            "system": [{"type": "text", "text": "be terse"}],
            "messages": [
                {"role": "assistant", "content": "stray"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
                {"role": "user", "content": ""},
            ],
        }
    )
    assert request_messages(raw) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "(empty)"},
    ]
    assert system_prompt(raw) == "be terse"
    assert request_text(raw) == "user: q1\n\nassistant: a1\n\nuser: (empty)"


def test_unparseable_request_still_yields_a_valid_message() -> None:
    assert request_messages("not json") == [{"role": "user", "content": "not json"}]
    assert request_messages(json.dumps([1, 2])) == [{"role": "user", "content": "(empty)"}]
    assert system_prompt("not json") is None


def test_parse_anthropic_response_with_tool_use() -> None:
    raw = json.dumps(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
            ],
        }
    )
    parsed = parse_response(raw)
    assert parsed.text == "Let me check."
    assert parsed.tool_calls == [{"name": "read_file", "input": {"path": "a.py"}}]


def test_parse_openai_response_and_plain_strings() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "grep", "arguments": '{"pattern": "TODO"}'}}
                    ],
                }
            }
        ]
    }
    parsed = parse_response(raw)
    assert parsed.text == ""
    assert parsed.tool_calls == [{"name": "grep", "input": {"pattern": "TODO"}}]
    assert parse_response("frontier answer").text == "frontier answer"
    assert parse_response(json.dumps({"content": "plain"})).text == "plain"


def test_content_to_text_handles_odd_shapes() -> None:
    assert content_to_text(None) == ""
    assert content_to_text({"type": "text", "text": "x"}) == "x"
    assert content_to_text(42) == "42"
    assert content_to_text(["a", 3, {"type": "text", "text": "b"}]) == "ab"


def test_replay_tools_reuse_request_tools_or_infer_from_calls() -> None:
    real = [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}]
    assert replay_tools(json.dumps({"messages": [], "tools": real}), []) == real

    inferred = replay_tools(
        json.dumps({"role": "user", "content": "hi"}),
        [
            {"name": "read_file", "input": {"path": "a"}},
            {"name": "read_file", "input": {"offset": 1}},
            {"name": "", "input": {}},
        ],
    )
    assert inferred == [
        {
            "name": "read_file",
            "description": "The read_file tool.",
            "input_schema": {"type": "object", "properties": {"offset": {}, "path": {}}},
        }
    ]
    (openai_tool,) = to_openai_tools(inferred)
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "read_file"
    assert openai_tool["function"]["parameters"]["properties"] == {"offset": {}, "path": {}}
