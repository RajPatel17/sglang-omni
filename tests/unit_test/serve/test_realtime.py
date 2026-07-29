# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk, UsageInfo
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.serve import openai_api
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


@pytest.mark.parametrize("supports_audio_output", [False, True])
def test_create_app_propagates_realtime_audio_capability(
    supports_audio_output: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_api, "_register_voices", lambda app: None)
    monkeypatch.setattr(openai_api, "_register_transcriptions", lambda app: None)

    app = openai_api.create_app(
        StreamingClient([]),  # type: ignore[arg-type]
        model_name="qwen3-omni",
        enable_realtime=True,
        supports_realtime_audio_output=supports_audio_output,
    )

    assert app.state.realtime_manager.supports_audio_output is supports_audio_output


def test_qwen_pipeline_audio_capability_contract() -> None:
    assert Qwen3OmniPipelineConfig.code2wav_stage() is None
    assert Qwen3OmniSpeechPipelineConfig.code2wav_stage() == "code2wav"


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
async def test_response_uses_modalities_snapshotted_before_response_created() -> None:
    session, websocket, client = _session(
        [
            _chunk(modality="audio", audio_b64="AQI=", stage_name="code2wav"),
            _chunk(
                modality="audio",
                finish_reason="stop",
                stage_name="code2wav",
            ),
            _chunk(text="hello", finish_reason="stop", stage_name="decode"),
        ]
    )
    session.session_object.modalities = ["text", "audio"]
    original_send_text = websocket.send_text

    async def send_text_and_update(payload: str) -> None:
        await original_send_text(payload)
        if websocket.events[-1]["type"] == "response.created":
            session.session_object.modalities = ["text"]

    websocket.send_text = send_text_and_update  # type: ignore[method-assign]

    await session.run_response("data:audio/wav;base64,AAAA")

    assert client.requests[0][0].output_modalities == ["text", "audio"]
    assert client.requests[0][2] == "pcm"
    assert "response.audio.delta" in _event_types(websocket)
    assert websocket.events[-1]["response"]["status"] == "completed"


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
async def test_audio_negotiation_normalizes_modality_order() -> None:
    session, websocket, _ = _session([])
    event = SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {"modalities": ["audio", "text"]},
        }
    )

    await session.handle_session_update(event)

    assert session.session_object.modalities == ["text", "audio"]
    assert websocket.events[-1]["type"] == "session.updated"
    assert websocket.events[-1]["session"]["modalities"] == ["text", "audio"]


@pytest.mark.asyncio
async def test_unsupported_modalities_do_not_mutate_session() -> None:
    session, websocket, _ = _session([])
    event = SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {"modalities": ["audio"]},
        }
    )

    await session.handle_session_update(event)

    assert session.session_object.modalities == ["text"]
    assert websocket.events[-1]["error"]["code"] == "unsupported_modality"


@pytest.mark.asyncio
async def test_unsupported_input_format_does_not_mutate_session() -> None:
    session, websocket, _ = _session([])
    event = SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {"input_audio_format": "g711_alaw"},
        }
    )

    await session.handle_session_update(event)

    assert session.session_object.input_audio_format == "pcm16"
    assert websocket.events[-1]["error"] == {
        "type": "invalid_request_error",
        "code": "unsupported_input_audio_format",
        "message": "Only PCM16 input audio is supported.",
    }


@pytest.mark.asyncio
async def test_unsupported_output_format_does_not_mutate_session() -> None:
    session, websocket, _ = _session([])
    event = SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {"output_audio_format": "g711_ulaw"},
        }
    )

    await session.handle_session_update(event)

    assert session.session_object.output_audio_format == "pcm16"
    assert session.session_object.modalities == ["text"]
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"] == {
        "type": "invalid_request_error",
        "code": "unsupported_audio_format",
        "message": "Only PCM16 output audio is supported.",
    }


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
    assert _event_types(websocket)[-3:] == [
        "response.text.done",
        "error",
        "response.done",
    ]
    assert websocket.events[-2]["error"]["code"] == "response_generation_failed"
    response = websocket.events[-1]["response"]
    assert response["status"] == "failed"
    assert response["output"][0]["content"][0]["text"] == "partial"


@pytest.mark.asyncio
async def test_cancellation_emits_cancelled_response_done() -> None:
    session, websocket, client = _session([])
    response_started = asyncio.Event()
    abort_seen = asyncio.Event()

    async def blocked_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs
        yield _chunk(text="partial")
        response_started.set()
        await abort_seen.wait()
        raise RuntimeError("aborted")
        yield

    async def abort(request_id: str) -> None:
        client.aborted.append(request_id)
        abort_seen.set()

    session.client.completion_stream = blocked_stream  # type: ignore[method-assign]
    session.client.abort = abort  # type: ignore[method-assign]
    task = asyncio.create_task(session.run_response("data:audio/wav;base64,AAAA"))
    await asyncio.wait_for(response_started.wait(), timeout=1)
    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )
    response_text = await asyncio.wait_for(task, timeout=1)

    assert response_text == "partial"
    assert len(client.aborted) == 1
    assert _event_types(websocket)[-2:] == [
        "response.text.done",
        "response.done",
    ]
    assert websocket.events[-1]["response"]["status"] == "cancelled"
    assert "error" not in _event_types(websocket)


@pytest.mark.asyncio
async def test_cancellation_before_request_submission_skips_engine_call() -> None:
    session, websocket, client = _session([])
    original_send_text = websocket.send_text

    async def send_text_and_cancel(payload: str) -> None:
        await original_send_text(payload)
        if websocket.events[-1]["type"] == "response.created":
            await session.handle_response_cancel(
                ResponseCancel.model_validate({"type": "response.cancel"})
            )

    websocket.send_text = send_text_and_cancel  # type: ignore[method-assign]

    response_text = await session.run_response("data:audio/wav;base64,AAAA")

    assert response_text == ""
    assert len(client.aborted) == 1
    assert client.requests == []
    assert _event_types(websocket) == [
        "response.created",
        "response.text.done",
        "response.done",
    ]
    assert websocket.events[-1]["response"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancellation_before_run_response_starts_skips_engine_call() -> None:
    session, websocket, client = _session([])
    session._response_start_pending = True

    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )
    response_text = await session.run_response("data:audio/wav;base64,AAAA")

    assert response_text == ""
    assert client.aborted == []
    assert client.requests == []
    assert _event_types(websocket) == [
        "response.created",
        "response.text.done",
        "response.done",
    ]
    assert websocket.events[-1]["response"]["status"] == "cancelled"
    assert session._response_start_pending is False
    assert session._cancel_pending_response is False


@pytest.mark.asyncio
async def test_audio_cancellation_before_first_delta_does_not_declare_audio() -> None:
    session, websocket, client = _session([])
    session.session_object.modalities = ["text", "audio"]
    response_started = asyncio.Event()
    abort_seen = asyncio.Event()

    async def blocked_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs
        response_started.set()
        await abort_seen.wait()
        raise RuntimeError("aborted")
        yield

    async def abort(request_id: str) -> None:
        client.aborted.append(request_id)
        abort_seen.set()

    session.client.completion_stream = blocked_stream  # type: ignore[method-assign]
    session.client.abort = abort  # type: ignore[method-assign]
    task = asyncio.create_task(session.run_response("data:audio/wav;base64,AAAA"))
    await asyncio.wait_for(response_started.wait(), timeout=1)

    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )
    await asyncio.wait_for(task, timeout=1)

    assert _event_types(websocket) == [
        "response.created",
        "response.text.done",
        "response.done",
    ]
    response = websocket.events[-1]["response"]
    assert response["status"] == "cancelled"
    assert response["output"][0]["content"] == [{"type": "text", "text": ""}]


@pytest.mark.asyncio
async def test_audio_cancellation_closes_started_audio_before_response_done() -> None:
    session, websocket, client = _session([])
    session.session_object.modalities = ["text", "audio"]
    audio_sent = asyncio.Event()
    abort_seen = asyncio.Event()

    async def blocked_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs
        yield _chunk(modality="audio", audio_b64="AQI=")
        audio_sent.set()
        await abort_seen.wait()
        raise RuntimeError("aborted")
        yield

    async def abort(request_id: str) -> None:
        client.aborted.append(request_id)
        abort_seen.set()

    session.client.completion_stream = blocked_stream  # type: ignore[method-assign]
    session.client.abort = abort  # type: ignore[method-assign]
    task = asyncio.create_task(session.run_response("data:audio/wav;base64,AAAA"))
    await asyncio.wait_for(audio_sent.wait(), timeout=1)

    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )
    await asyncio.wait_for(task, timeout=1)

    assert _event_types(websocket)[-3:] == [
        "response.text.done",
        "response.audio.done",
        "response.done",
    ]
    response = websocket.events[-1]["response"]
    assert response["status"] == "cancelled"
    assert {content["type"] for content in response["output"][0]["content"]} == {
        "text",
        "audio",
    }


@pytest.mark.asyncio
async def test_cancellation_preserves_transcription_and_turn_context() -> None:
    session, websocket, client = _session([])
    response_started = asyncio.Event()
    abort_seen = asyncio.Event()
    stream_calls = 0

    async def stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal stream_calls
        del args, kwargs
        stream_calls += 1
        if stream_calls == 1:
            yield _chunk(text="partial answer")
            response_started.set()
            await abort_seen.wait()
            raise RuntimeError("aborted")
        yield _chunk(text="original question")
        yield _chunk(finish_reason="stop", stage_name="decode")

    async def abort(request_id: str) -> None:
        client.aborted.append(request_id)
        abort_seen.set()

    session.client.completion_stream = stream  # type: ignore[method-assign]
    session.client.abort = abort  # type: ignore[method-assign]
    task = asyncio.create_task(
        session.run_turn("item_previous", "data:audio/wav;base64,AAAA")
    )
    session.active_task = task
    await asyncio.wait_for(response_started.wait(), timeout=1)

    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )
    await asyncio.wait_for(task, timeout=1)

    assert len(client.aborted) == 1
    assert websocket.events[-1]["type"] == (
        "conversation.item.input_audio_transcription.completed"
    )
    response_done = next(
        event for event in websocket.events if event["type"] == "response.done"
    )
    assert response_done["response"]["status"] == "cancelled"
    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "original question"),
        ("assistant", "partial answer"),
    ]


@pytest.mark.asyncio
async def test_response_cancel_during_transcription_is_ignored() -> None:
    session, _, client = _session([])
    transcription_started = asyncio.Event()
    finish_transcription = asyncio.Event()
    stream_calls = 0

    async def stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal stream_calls
        del args, kwargs
        stream_calls += 1
        if stream_calls == 1:
            yield _chunk(text="complete answer")
            yield _chunk(finish_reason="stop", stage_name="decode")
            return
        transcription_started.set()
        await finish_transcription.wait()
        yield _chunk(text="original question")
        yield _chunk(finish_reason="stop", stage_name="decode")

    session.client.completion_stream = stream  # type: ignore[method-assign]
    task = asyncio.create_task(
        session.run_turn("item_previous", "data:audio/wav;base64,AAAA")
    )
    session.active_task = task
    await asyncio.wait_for(transcription_started.wait(), timeout=1)

    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )

    assert client.aborted == []
    finish_transcription.set()
    await asyncio.wait_for(task, timeout=1)
    assert [item.text for item in session.conversation] == [
        "original question",
        "complete answer",
    ]
