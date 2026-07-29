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
    def __init__(self, chunks: list[CompletionStreamChunk]) -> None:
        self.chunks = chunks
        self.requests: list[tuple[Any, str]] = []
        self.aborted: list[str] = []

    async def completion_stream(
        self,
        request: Any,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> AsyncIterator[CompletionStreamChunk]:
        self.requests.append((request, audio_format))
        for chunk in self.chunks:
            yield chunk

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _chunk(
    *,
    modality: str = "text",
    audio_b64: str | None = None,
    finish_reason: str | None = None,
) -> CompletionStreamChunk:
    return CompletionStreamChunk(
        request_id="request",
        modality=modality,
        audio_b64=audio_b64,
        finish_reason=finish_reason,
    )


def _session(
    chunks: list[CompletionStreamChunk],
    *,
    supports_audio_output: bool = True,
) -> tuple[RealtimeSession, RecordingWebSocket, StreamingClient]:
    websocket = RecordingWebSocket()
    client = StreamingClient(chunks)
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
async def test_audio_session_update_is_accepted() -> None:
    session, websocket, _ = _session([])

    await session.handle_session_update(
        SessionUpdate.model_validate(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "output_audio_format": "pcm16",
                },
            }
        )
    )

    assert session.session_object.modalities == ["text", "audio"]
    assert websocket.events[-1]["type"] == "session.updated"


@pytest.mark.asyncio
async def test_audio_session_update_rejects_pipeline_without_audio() -> None:
    session, websocket, _ = _session([], supports_audio_output=False)

    await session.handle_session_update(
        SessionUpdate.model_validate(
            {
                "type": "session.update",
                "session": {"modalities": ["text", "audio"]},
            }
        )
    )

    assert session.session_object.modalities == ["text"]
    assert websocket.events[-1]["error"]["code"] == "unsupported_modality"


@pytest.mark.asyncio
async def test_audio_response_streams_pcm_before_done() -> None:
    session, websocket, client = _session(
        [
            _chunk(finish_reason="stop"),
            _chunk(
                modality="audio",
                audio_b64="AQI=",
                finish_reason="stop",
            ),
        ]
    )
    session.session_object.modalities = ["text", "audio"]

    await session.run_response("data:audio/wav;base64,AAAA")

    assert _event_types(websocket) == [
        "response.created",
        "response.text.done",
        "response.audio.delta",
        "response.audio.done",
        "response.done",
    ]
    request, audio_format = client.requests[0]
    assert request.output_modalities == ["text", "audio"]
    assert audio_format == "pcm"
    assert websocket.events[2]["delta"] == "AQI="
    assert {
        content["type"]
        for content in websocket.events[-1]["response"]["output"][0]["content"]
    } == {"text", "audio"}


@pytest.mark.asyncio
async def test_audio_response_fails_when_pipeline_returns_no_audio() -> None:
    session, websocket, _ = _session(
        [
            _chunk(finish_reason="stop"),
            _chunk(modality="audio", finish_reason="stop"),
        ]
    )
    session.session_object.modalities = ["text", "audio"]

    await session.run_response("data:audio/wav;base64,AAAA")

    assert _event_types(websocket)[-2:] == ["error", "response.done"]
    assert websocket.events[-2]["error"]["code"] == "audio_output_missing"
    assert websocket.events[-1]["response"]["status"] == "failed"


@pytest.mark.asyncio
async def test_audio_cancellation_closes_started_audio() -> None:
    session, websocket, client = _session([])
    session.session_object.modalities = ["text", "audio"]
    audio_sent = asyncio.Event()

    async def blocked_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs
        yield _chunk(modality="audio", audio_b64="AQI=")
        audio_sent.set()
        await asyncio.Future()
        yield

    session.client.completion_stream = blocked_stream  # type: ignore[method-assign]
    task = asyncio.create_task(session.run_response("data:audio/wav;base64,AAAA"))
    session.active_task = task
    await asyncio.wait_for(audio_sent.wait(), timeout=1)

    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )

    assert task.cancelled()
    assert len(client.aborted) == 1
    assert _event_types(websocket)[-2:] == [
        "response.audio.done",
        "response.done",
    ]
    assert websocket.events[-1]["response"]["status"] == "cancelled"
