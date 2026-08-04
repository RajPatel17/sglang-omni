# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk, GenerateRequest
from sglang_omni.serve.realtime.session import RealtimeSession


@pytest.mark.asyncio
async def test_response_reason_comes_from_text_terminal() -> None:
    events: list[dict[str, Any]] = []

    class WebSocket:
        application_state = WebSocketState.CONNECTED

        async def send_text(self, payload: str) -> None:
            events.append(json.loads(payload))

    class Client:
        async def completion_stream(
            self,
            request: Any,
            *,
            request_id: str,
            audio_format: str = "wav",
        ) -> AsyncIterator[CompletionStreamChunk]:
            del request, request_id, audio_format
            yield CompletionStreamChunk(request_id="request", text="answer")
            yield CompletionStreamChunk(
                request_id="request",
                modality="text",
                finish_reason="length",
            )
            yield CompletionStreamChunk(
                request_id="request",
                modality="audio",
                audio_b64="AA==",
            )
            yield CompletionStreamChunk(
                request_id="request",
                modality="audio",
                finish_reason="stop",
            )

    session = object.__new__(RealtimeSession)
    session.websocket = WebSocket()  # type: ignore[assignment]
    session.client = Client()  # type: ignore[assignment]
    session.session_id = "session"
    session.closed = False
    session.active_request_id = None
    session.response_start_pending = False
    session.cancel_pending_response = False
    session.build_response_request = lambda _: GenerateRequest(  # type: ignore[method-assign]
        output_modalities=["text", "audio"]
    )

    assert await session.run_response("audio") == "answer"
    response_done = next(event for event in events if event["type"] == "response.done")
    assert response_done["response"]["status_details"]["reason"] == "length"
