# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.events import ResponseCancel
from sglang_omni.serve.realtime.session import RealtimeSession


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.events.append(json.loads(payload))


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


@pytest.mark.asyncio
async def test_response_cancel_preserves_user_transcription_and_history() -> None:
    response_started = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.aborted: list[str] = []

        async def completion_stream(
            self,
            request: Any,
            *,
            request_id: str,
            audio_format: str = "wav",
        ) -> AsyncIterator[CompletionStreamChunk]:
            del request, request_id, audio_format
            self.calls += 1
            if self.calls == 1:
                response_started.set()
                yield _chunk(modality="audio", audio_b64="AQI=")
                await asyncio.Future()
            else:
                yield _chunk(text="remember this")
                yield _chunk(finish_reason="stop")

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    websocket = RecordingWebSocket()
    client = Client()
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]

    turn_task = asyncio.create_task(
        session.run_turn("item-user", "data:audio/wav;base64,AAAA")
    )
    await asyncio.wait_for(response_started.wait(), timeout=1)
    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )
    await asyncio.wait_for(turn_task, timeout=1)

    assert len(client.aborted) == 1
    assert [item.role for item in session.conversation] == ["user"]
    assert session.conversation[0].text == "remember this"
    event_types = [event["type"] for event in websocket.events]
    assert "conversation.item.input_audio_transcription.completed" in event_types
    response_done = next(
        event for event in websocket.events if event["type"] == "response.done"
    )
    assert response_done["response"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_response_cancel_during_transcription_is_noop() -> None:
    transcription_started = asyncio.Event()
    finish_transcription = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.aborted: list[str] = []

        async def completion_stream(
            self,
            request: Any,
            *,
            request_id: str,
            audio_format: str = "wav",
        ) -> AsyncIterator[CompletionStreamChunk]:
            del request, request_id, audio_format
            self.calls += 1
            if self.calls == 1:
                yield _chunk(text="answer")
                yield _chunk(finish_reason="stop")
            else:
                transcription_started.set()
                await finish_transcription.wait()
                yield _chunk(text="user question")
                yield _chunk(finish_reason="stop")

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    websocket = RecordingWebSocket()
    client = Client()
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
    )

    turn_task = asyncio.create_task(
        session.run_turn("item-user", "data:audio/wav;base64,AAAA")
    )
    await asyncio.wait_for(transcription_started.wait(), timeout=1)
    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )
    finish_transcription.set()
    await asyncio.wait_for(turn_task, timeout=1)

    assert client.aborted == []
    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "user question"),
        ("assistant", "answer"),
    ]


@pytest.mark.asyncio
async def test_turn_task_cancellation_does_not_continue_to_transcription() -> None:
    response_started = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def completion_stream(
            self,
            request: Any,
            *,
            request_id: str,
            audio_format: str = "wav",
        ) -> AsyncIterator[CompletionStreamChunk]:
            del request, request_id, audio_format
            self.calls += 1
            response_started.set()
            await asyncio.Future()
            yield

        async def abort(self, request_id: str) -> None:
            del request_id

    client = Client()
    session = RealtimeSession(
        RecordingWebSocket(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
    )

    turn_task = asyncio.create_task(
        session.run_turn("item-user", "data:audio/wav;base64,AAAA")
    )
    await asyncio.wait_for(response_started.wait(), timeout=1)
    turn_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn_task

    assert client.calls == 1
    assert session.conversation == []
