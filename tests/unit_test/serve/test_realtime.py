# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.events import ResponseCancel, SessionUpdate
from sglang_omni.serve.realtime.session import RealtimeSession


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.events.append(json.loads(payload))


class StreamingClient:
    def __init__(
        self,
        chunks: list[CompletionStreamChunk],
        *,
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.requests: list[tuple[Any, str, str]] = []
        self.aborted: list[str] = []

    async def completion_stream(
        self,
        request: Any,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> AsyncIterator[CompletionStreamChunk]:
        self.requests.append((request, request_id, audio_format))
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _chunk(
    *,
    modality: str = "text",
    text: str = "",
    audio_b64: str | None = None,
    finish_reason: str | None = None,
) -> CompletionStreamChunk:
    return CompletionStreamChunk(
        request_id="request",
        modality=modality,
        text=text,
        audio_b64=audio_b64,
        finish_reason=finish_reason,
    )


def _session(
    chunks: list[CompletionStreamChunk],
    *,
    supports_audio_output: bool = True,
    error: Exception | None = None,
) -> tuple[RealtimeSession, RecordingWebSocket, StreamingClient]:
    websocket = RecordingWebSocket()
    client = StreamingClient(chunks, error=error)
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=supports_audio_output,
    )
    return session, websocket, client


def _event_types(websocket: RecordingWebSocket) -> list[str]:
    return [event["type"] for event in websocket.events]


@pytest.mark.asyncio
async def test_text_only_response_preserves_existing_contract() -> None:
    session, websocket, client = _session(
        [
            _chunk(text="hello "),
            _chunk(text="world"),
            _chunk(finish_reason="stop"),
        ]
    )

    response_text = await session.run_response("data:audio/wav;base64,AAAA")

    assert response_text == "hello world"
    assert _event_types(websocket) == [
        "response.created",
        "response.text.delta",
        "response.text.delta",
        "response.text.done",
        "response.done",
    ]
    request, _, audio_format = client.requests[0]
    assert request.output_modalities == ["text"]
    assert audio_format == "wav"
    assert websocket.events[-1]["response"]["output"][0]["content"] == [
        {"type": "text", "text": "hello world"}
    ]


@pytest.mark.asyncio
async def test_audio_response_streams_text_and_pcm_from_one_snapshot() -> None:
    session, websocket, client = _session(
        [
            _chunk(text="hello"),
            _chunk(
                modality="audio",
                audio_b64="AQI=",
                finish_reason="stop",
            ),
            _chunk(text=" world", finish_reason="length"),
        ]
    )
    session.session_object.modalities = ["text", "audio"]
    original_send_text = websocket.send_text

    async def send_text_and_update(payload: str) -> None:
        await original_send_text(payload)
        if websocket.events[-1]["type"] == "response.created":
            session.session_object.modalities = ["text"]

    websocket.send_text = send_text_and_update  # type: ignore[method-assign]

    response_text = await session.run_response("data:audio/wav;base64,AAAA")

    assert response_text == "hello world"
    assert _event_types(websocket) == [
        "response.created",
        "response.text.delta",
        "response.audio.delta",
        "response.audio.done",
        "response.text.delta",
        "response.text.done",
        "response.done",
    ]
    request, _, audio_format = client.requests[0]
    assert request.output_modalities == ["text", "audio"]
    assert audio_format == "pcm"
    assert websocket.events[2]["delta"] == "AQI="
    response = websocket.events[-1]["response"]
    assert response["status_details"]["reason"] == "length"
    assert response["output"][0]["content"] == [
        {"type": "text", "text": "hello world"},
        {"type": "audio", "transcript": "hello world"},
    ]


@pytest.mark.parametrize(
    ("supports_audio_output", "update", "error_code"),
    [
        (
            True,
            {
                "modalities": ["text", "audio"],
                "output_audio_format": "pcm16",
            },
            None,
        ),
        (False, {"modalities": ["text", "audio"]}, "unsupported_modality"),
        (True, {"modalities": ["audio"]}, "unsupported_modality"),
        (True, {"output_audio_format": "g711_ulaw"}, "unsupported_audio_format"),
    ],
)
@pytest.mark.asyncio
async def test_audio_session_update(
    supports_audio_output: bool,
    update: dict[str, Any],
    error_code: str | None,
) -> None:
    session, websocket, _ = _session(
        [],
        supports_audio_output=supports_audio_output,
    )
    before = session.session_object.model_dump()

    await session.handle_session_update(
        SessionUpdate.model_validate({"type": "session.update", "session": update})
    )

    if error_code is None:
        assert session.session_object.modalities == ["text", "audio"]
        assert websocket.events[-1]["type"] == "session.updated"
        assert websocket.events[-1]["session"]["output_audio_format"] == "pcm16"
    else:
        assert session.session_object.model_dump() == before
        assert websocket.events[-1]["error"]["code"] == error_code


@pytest.mark.parametrize(
    ("chunks", "modalities", "error", "text", "code", "reason"),
    [
        (
            [
                _chunk(finish_reason="stop"),
                _chunk(modality="audio", finish_reason="stop"),
            ],
            ["text", "audio"],
            None,
            "",
            "audio_output_missing",
            "audio_output_missing",
        ),
        (
            [_chunk(text="partial")],
            ["text"],
            RuntimeError("pipeline failed"),
            "partial",
            "response_generation_failed",
            "error",
        ),
    ],
)
@pytest.mark.asyncio
async def test_failed_response_closes_text_before_done(
    chunks: list[CompletionStreamChunk],
    modalities: list[str],
    error: Exception | None,
    text: str,
    code: str,
    reason: str,
) -> None:
    session, websocket, _ = _session(chunks, error=error)
    session.session_object.modalities = modalities

    response_text = await session.run_response("data:audio/wav;base64,AAAA")

    assert response_text == text
    assert _event_types(websocket)[-3:] == [
        "response.text.done",
        "error",
        "response.done",
    ]
    assert websocket.events[-2]["error"]["code"] == code
    response = websocket.events[-1]["response"]
    assert response["status"] == "failed"
    assert response["status_details"]["reason"] == reason
    assert response["output"][0]["content"] == [{"type": "text", "text": text}]
    if code == "audio_output_missing":
        assert "response.audio.done" not in _event_types(websocket)


@pytest.mark.parametrize(
    ("modalities", "audio_delta"),
    [
        (["text", "audio"], None),
        (["text", "audio"], "AQI="),
    ],
)
@pytest.mark.asyncio
async def test_cancellation_closes_only_started_content(
    modalities: list[str],
    audio_delta: str | None,
) -> None:
    session, websocket, client = _session([])
    session.session_object.modalities = modalities
    stream_blocked = asyncio.Event()

    async def blocked_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs
        if audio_delta is not None:
            yield _chunk(modality="audio", audio_b64=audio_delta)
        stream_blocked.set()
        await asyncio.Future()
        yield

    session.client.completion_stream = blocked_stream  # type: ignore[method-assign]
    task = asyncio.create_task(session.run_response("data:audio/wav;base64,AAAA"))
    session.active_task = task
    await asyncio.wait_for(stream_blocked.wait(), timeout=1)

    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )

    assert task.cancelled()
    assert len(client.aborted) == 1
    expected_types = ["response.created"]
    if audio_delta is not None:
        expected_types.append("response.audio.delta")
    expected_types.append("response.text.done")
    if audio_delta is not None:
        expected_types.append("response.audio.done")
    expected_types.append("response.done")
    assert _event_types(websocket) == expected_types
    response = websocket.events[-1]["response"]
    assert response["status"] == "cancelled"
    expected_content_types = {"text"}
    if audio_delta is not None:
        expected_content_types.add("audio")
    assert {
        content["type"] for content in response["output"][0]["content"]
    } == expected_content_types
