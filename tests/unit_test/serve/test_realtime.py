# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk, UsageInfo
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
    usage: UsageInfo | None = None,
    stage_name: str | None = None,
) -> CompletionStreamChunk:
    return CompletionStreamChunk(
        request_id="request",
        modality=modality,
        text=text,
        audio_b64=audio_b64,
        finish_reason=finish_reason,
        usage=usage,
        stage_name=stage_name,
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
async def test_text_only_response_preserves_existing_event_contract() -> None:
    usage = UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    session, websocket, client = _session(
        [
            _chunk(text="hello "),
            _chunk(text="world"),
            _chunk(finish_reason="stop", usage=usage, stage_name="decode"),
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
    assert client.requests[0][0].output_modalities == ["text"]
    assert client.requests[0][2] == "wav"
    response = websocket.events[-1]["response"]
    assert response["status"] == "completed"
    assert response["output"][0]["content"] == [{"type": "text", "text": "hello world"}]
    assert response["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_audio_response_streams_pcm_until_both_terminals_complete() -> None:
    usage = UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7)
    session, websocket, client = _session(
        [
            _chunk(text="hello", stage_name="decode"),
            _chunk(
                modality="audio",
                audio_b64="AQI=",
                finish_reason="stop",
                stage_name="code2wav",
            ),
            _chunk(
                text=" world",
                finish_reason="length",
                usage=usage,
                stage_name="decode",
            ),
        ]
    )
    session.session_object.modalities = ["text", "audio"]

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
    assert client.requests[0][0].output_modalities == ["text", "audio"]
    assert client.requests[0][2] == "pcm"
    audio_delta = websocket.events[2]
    assert audio_delta["delta"] == "AQI="
    assert audio_delta["content_index"] == 1
    response = websocket.events[-1]["response"]
    assert response["status_details"]["reason"] == "length"
    assert response["usage"]["total_tokens"] == 7
    assert response["output"][0]["content"] == [
        {"type": "text", "text": "hello world"},
        {"type": "audio", "transcript": "hello world"},
    ]


@pytest.mark.asyncio
async def test_terminal_audio_payload_is_emitted_before_audio_done() -> None:
    session, websocket, _ = _session(
        [
            _chunk(finish_reason="stop", stage_name="decode"),
            _chunk(
                modality="audio",
                audio_b64="AwQ=",
                finish_reason="stop",
                stage_name="code2wav",
            ),
        ]
    )
    session.session_object.modalities = ["text", "audio"]

    await session.run_response("data:audio/wav;base64,AAAA")

    types = _event_types(websocket)
    assert types.index("response.audio.delta") < types.index("response.audio.done")


@pytest.mark.asyncio
async def test_audio_negotiation_rejects_thinker_only_pipeline_without_mutation() -> (
    None
):
    session, websocket, _ = _session([], supports_audio_output=False)
    event = SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {"modalities": ["text", "audio"]},
        }
    )

    await session.handle_session_update(event)

    assert session.session_object.modalities == ["text"]
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"] == {
        "type": "invalid_request_error",
        "code": "unsupported_modality",
        "message": "Audio output is unavailable for this pipeline.",
    }


@pytest.mark.asyncio
async def test_audio_negotiation_accepts_pcm16_for_speech_pipeline() -> None:
    session, websocket, _ = _session([])
    event = SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "output_audio_format": "pcm16",
            },
        }
    )

    await session.handle_session_update(event)

    assert session.session_object.modalities == ["text", "audio"]
    assert websocket.events[-1]["type"] == "session.updated"
    assert websocket.events[-1]["session"]["output_audio_format"] == "pcm16"


@pytest.mark.asyncio
async def test_unsupported_output_format_does_not_mutate_session() -> None:
    session, websocket, _ = _session([])
    event = SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {"output_audio_format": "g711_ulaw"},
        }
    )

    with pytest.raises(AssertionError, match="output_audio_format must be 'pcm16'"):
        await session.handle_session_update(event)

    assert session.session_object.output_audio_format == "pcm16"
    assert websocket.events == []


@pytest.mark.asyncio
async def test_missing_audio_marks_response_failed() -> None:
    session, websocket, _ = _session(
        [
            _chunk(finish_reason="stop", stage_name="decode"),
            _chunk(
                modality="audio",
                finish_reason="stop",
                stage_name="code2wav",
            ),
        ]
    )
    session.session_object.modalities = ["text", "audio"]

    await session.run_response("data:audio/wav;base64,AAAA")

    assert "response.audio.done" not in _event_types(websocket)
    assert _event_types(websocket)[-2:] == ["error", "response.done"]
    assert websocket.events[-2]["error"]["code"] == "audio_output_missing"
    assert websocket.events[-1]["response"]["status"] == "failed"
    assert (
        websocket.events[-1]["response"]["status_details"]["reason"]
        == "audio_output_missing"
    )
    assert websocket.events[-1]["response"]["output"][0]["content"] == [
        {"type": "text", "text": ""}
    ]


@pytest.mark.asyncio
async def test_midstream_failure_emits_error_and_failed_response_done() -> None:
    session, websocket, _ = _session(
        [_chunk(text="partial")],
        error=RuntimeError("pipeline failed"),
    )

    response_text = await session.run_response("data:audio/wav;base64,AAAA")

    assert response_text == "partial"
    assert _event_types(websocket)[-2:] == ["error", "response.done"]
    assert websocket.events[-2]["error"]["code"] == "response_generation_failed"
    response = websocket.events[-1]["response"]
    assert response["status"] == "failed"
    assert response["output"][0]["content"][0]["text"] == "partial"


@pytest.mark.asyncio
async def test_cancellation_emits_cancelled_response_done() -> None:
    session, websocket, client = _session([])

    async def blocked_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs
        await asyncio.Future()
        yield

    session.client.completion_stream = blocked_stream  # type: ignore[method-assign]
    task = asyncio.create_task(session.run_response("data:audio/wav;base64,AAAA"))
    session.active_task = task
    await asyncio.sleep(0)
    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )

    assert task.cancelled()
    assert len(client.aborted) == 1
    assert websocket.events[-1]["type"] == "response.done"
    assert websocket.events[-1]["response"]["status"] == "cancelled"
