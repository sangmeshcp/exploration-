"""Fake Anthropic-compatible upstream for proxy/replay tests.

Emits realistic SSE streams including tool_use blocks, honors a handful of
trigger models to produce errors / mid-stream disconnects, so the proxy
integration suite never talks to the real network (plan.md §3.2).
"""

from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(title="fake-anthropic")

CALL_LOG: list[dict[str, object]] = []


def reset_call_log() -> None:
    CALL_LOG.clear()


def _usage(tokens_in: int = 12, tokens_out: int = 34) -> dict[str, int]:
    return {"input_tokens": tokens_in, "output_tokens": tokens_out}


def _sse_chunks(model: str, with_tool_use: bool) -> list[bytes]:
    events: list[dict[str, object]] = [
        {
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": "msg_fake_1",
                    "model": model,
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                },
            },
        },
        {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        },
        {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello from the fake upstream."},
            },
        },
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
    ]
    if with_tool_use:
        events += [
            {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {},
                    },
                },
            },
            {
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"path": "a.py"}'},
                },
            },
            {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 1}},
        ]
    events += [
        {
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": _usage(),
            },
        },
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]
    chunks = []
    for e in events:
        chunks.append(f"event: {e['event']}\ndata: {json.dumps(e['data'])}\n\n".encode())
    return chunks


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    body = await request.json()
    CALL_LOG.append(body)
    model = body.get("model", "claude-haiku-4-5")

    if model == "trigger-500":
        return JSONResponse({"error": {"message": "internal error"}}, status_code=500)
    if model == "trigger-429":
        return JSONResponse({"error": {"message": "rate limited"}}, status_code=429)

    with_tool_use = bool(body.get("tools"))
    stream = bool(body.get("stream"))

    if stream:

        async def gen() -> object:
            for chunk in _sse_chunks(model, with_tool_use):
                yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream")

    content: list[dict[str, object]] = [{"type": "text", "text": "Hello from the fake upstream."}]
    if with_tool_use:
        content.append(
            {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}
        )
    return JSONResponse(
        {
            "id": "msg_fake_1",
            "model": model,
            "content": content,
            "stop_reason": "end_turn",
            "usage": _usage(),
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    CALL_LOG.append(body)
    model = body.get("model", "gpt-4o-mini")
    if model == "trigger-500":
        return JSONResponse({"error": {"message": "internal error"}}, status_code=500)
    return JSONResponse(
        {
            "id": "chatcmpl_fake_1",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from the fake upstream."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
        }
    )
