# SPDX-License-Identifier: Apache-2.0
"""Unit tests for make_moss_transcribe_diarize_stream_output_builder.

All tests run without a real MOSS-TD model — mock tokenizers are used. The
builder is exercised exactly as OmniScheduler calls it: once per decode step
with ``(request_id, req_data, req_output)``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    make_moss_transcribe_diarize_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload

_EOS = 999

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _ByteTokenizer:
    """Token id → fixed bytes; UTF-8 decode with errors='replace'."""

    eos_token_id = _EOS

    def __init__(
        self,
        vocab: dict[int, bytes],
        special_token_ids: set[int] | None = None,
    ):
        self._vocab = vocab
        self._special = special_token_ids or set()

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        chunks = [
            self._vocab[tid]
            for tid in ids
            if not (skip_special_tokens and tid in self._special)
        ]
        return b"".join(chunks).decode("utf-8", errors="replace")


def _make_req_data(*, stream: bool = True, is_chunked: int = 0) -> Any:
    """Minimal req_data as OmniScheduler passes to stream_output_builder."""
    stage_payload = StagePayload(
        request_id="r",
        request=OmniRequest(
            inputs={"audio_bytes": b""},
            params={"stream": stream},
            metadata={},
        ),
        data={},
    )
    req = SimpleNamespace(is_chunked=is_chunked)
    return SimpleNamespace(req=req, stage_payload=stage_payload)


def _make_req_output(token_id: int | None) -> Any:
    return SimpleNamespace(data=token_id)


def _builder(vocab: dict[int, bytes], special: set[int] | None = None):
    return make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer(vocab, special_token_ids=special),
    )


# ---------------------------------------------------------------------------
# Emission gating
# ---------------------------------------------------------------------------


def test_emits_text_delta_when_streaming():
    builder = _builder({1: b"[0.00]"})
    rd = _make_req_data(stream=True)

    msgs = builder("req-1", rd, _make_req_output(1))

    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.type == "stream"
    assert msg.request_id == "req-1"
    # target=None routes to the Coordinator (terminal stage).
    assert msg.target is None
    assert msg.data == {"text": "[0.00]", "modality": "text", "stage_name": "asr"}
    assert msg.metadata == {"modality": "text", "token_id": 1}


def test_silent_when_not_streaming():
    builder = _builder({1: b"A"})
    rd = _make_req_data(stream=False)

    assert builder("req-1", rd, _make_req_output(1)) == []
    # No per-request state is created for non-streaming requests.
    assert not hasattr(rd.req, "_moss_stream_token_ids")


def test_silent_during_chunked_prefill():
    builder = _builder({1: b"A"})
    rd = _make_req_data(stream=True, is_chunked=1)

    assert builder("req-1", rd, _make_req_output(1)) == []

    # Once chunked prefill completes, emission resumes.
    rd.req.is_chunked = 0
    msgs = builder("req-1", rd, _make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["A"]


def test_silent_when_no_token_this_step():
    builder = _builder({1: b"A"})
    rd = _make_req_data(stream=True)

    assert builder("req-1", rd, _make_req_output(None)) == []


def test_silent_when_req_or_payload_missing():
    builder = _builder({1: b"A"})

    no_req = SimpleNamespace(req=None, stage_payload=None)
    assert builder("req-1", no_req, _make_req_output(1)) == []

    no_payload = SimpleNamespace(req=SimpleNamespace(is_chunked=0), stage_payload=None)
    assert builder("req-1", no_payload, _make_req_output(1)) == []


# ---------------------------------------------------------------------------
# Incremental detokenization
# ---------------------------------------------------------------------------


def test_incremental_deltas_across_tokens():
    builder = _builder({1: b"[S01]", 2: b" hello", 3: b" world"})
    rd = _make_req_data()

    deltas = []
    for tid in (1, 2, 3):
        for msg in builder("req-1", rd, _make_req_output(tid)):
            deltas.append(msg.data["text"])

    assert deltas == ["[S01]", " hello", " world"]


def test_utf8_multibyte_hold_then_emit():
    """A 3-byte CJK char split across 3 tokens must hold until complete."""
    # "你" is U+4F60 → b'\xe4\xbd\xa0'. Split byte-per-token.
    builder = _builder({1: b"\xe4", 2: b"\xbd", 3: b"\xa0", 4: b"ok"})
    rd = _make_req_data()

    assert builder("req-1", rd, _make_req_output(1)) == []
    assert builder("req-1", rd, _make_req_output(2)) == []
    msgs = builder("req-1", rd, _make_req_output(3))
    assert [m.data["text"] for m in msgs] == ["你"]

    msgs = builder("req-1", rd, _make_req_output(4))
    assert [m.data["text"] for m in msgs] == ["ok"]


def test_interior_replacement_char_does_not_stall_stream():
    """Only a TRAILING U+FFFD is held; an interior one must flush normally."""
    # 0x80 is a lone continuation byte → decodes to a permanent U+FFFD.
    builder = _builder({1: b"\x80", 2: b"ok"})
    rd = _make_req_data()

    assert builder("req-1", rd, _make_req_output(1)) == []
    msgs = builder("req-1", rd, _make_req_output(2))
    assert [m.data["text"] for m in msgs] == ["\ufffdok"]


def test_eos_token_emits_no_delta():
    builder = _builder({1: b"hi", _EOS: b"<eos>"})
    rd = _make_req_data()

    msgs = builder("req-1", rd, _make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["hi"]
    assert builder("req-1", rd, _make_req_output(_EOS)) == []


def test_special_token_emits_no_delta():
    """Tokens dropped by skip_special_tokens must not produce a chunk."""
    builder = _builder({1: b"hi", 2: b"<|im_end|>"}, special={2})
    rd = _make_req_data()

    msgs = builder("req-1", rd, _make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["hi"]
    assert builder("req-1", rd, _make_req_output(2)) == []


def test_per_request_state_is_isolated():
    """Concurrent requests keep independent token/text state on their req."""
    builder = _builder({1: b"A", 2: b"B"})
    rd1 = _make_req_data()
    rd2 = _make_req_data()

    out1 = builder("r1", rd1, _make_req_output(1))
    out2 = builder("r2", rd2, _make_req_output(2))
    out1b = builder("r1", rd1, _make_req_output(2))

    assert [m.data["text"] for m in out1] == ["A"]
    assert [m.data["text"] for m in out2] == ["B"]
    assert [m.data["text"] for m in out1b] == ["B"]
    assert rd1.req._moss_stream_emitted_text == "AB"
    assert rd2.req._moss_stream_emitted_text == "B"


def test_explicit_eos_token_id_overrides_tokenizer():
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer({1: b"A", 7: b"<stop>"}),
        eos_token_id=7,
    )
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["A"]
    assert builder("r", rd, _make_req_output(7)) == []
