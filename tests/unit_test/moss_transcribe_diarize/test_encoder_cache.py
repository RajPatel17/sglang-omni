# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MOSS-Transcribe-Diarize encoder LRU cache (M-PR3).

The cache memoizes ``get_audio_feature`` (WhisperEncoder + VQAdaptor), a pure
function of the source waveform, so identical audio is encoded once. These tests
drive ``get_audio_feature`` directly with the underlying encode stubbed out, and
assert on how many times that encode is invoked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

# Every test here loads the model, which pulls in sglang; skip the whole file
# where sglang is absent rather than failing collection. (torch is assumed
# present, matching the other model tests.)
pytest.importorskip("sglang")

from sglang_omni.models.moss_transcribe_diarize.sglang_model import (  # noqa: E402
    _ENCODER_CACHE_MAX_ENTRIES,
    MossTranscribeDiarizeForConditionalGeneration as MossModel,
)


def _make_model(max_bytes: int) -> MossModel:
    """Build a bare model instance wired only for the encoder-cache path.

    ``__init__`` loads real weights, so bypass it with ``__new__`` and attach
    just what the cache wrapper touches: a parameter-bearing module for device
    lookup and the cache itself.
    """
    model = MossModel.__new__(MossModel)
    model.vq_adaptor = torch.nn.Linear(4, 4)  # CPU params -> device probe
    model.init_encoder_cache(max_bytes)
    return model


def _stub_encode(model: MossModel):
    """Replace the uncached encode with a call-counting stub returning a tensor."""
    calls = {"count": 0}

    def _fake(items, forward_batch):  # noqa: ANN001
        calls["count"] += 1
        return torch.ones(4)

    model._get_audio_feature_uncached = _fake  # type: ignore[assignment]
    return calls


def _item(audio_hash: int) -> SimpleNamespace:
    return SimpleNamespace(hash=audio_hash)


def test_identical_hash_encodes_once() -> None:
    model = _make_model(max_bytes=1 << 20)
    calls = _stub_encode(model)

    first = model.get_audio_feature([_item(123)], forward_batch=None)
    second = model.get_audio_feature([_item(123)], forward_batch=None)

    assert calls["count"] == 1, "second identical-hash encode should hit the cache"
    assert torch.equal(first, second)


def test_different_hash_encodes_each() -> None:
    model = _make_model(max_bytes=1 << 20)
    calls = _stub_encode(model)

    model.get_audio_feature([_item(1)], forward_batch=None)
    model.get_audio_feature([_item(2)], forward_batch=None)

    assert calls["count"] == 2, "distinct audio must not share a cache entry"


def test_disabled_cache_always_encodes() -> None:
    model = _make_model(max_bytes=0)  # 0 => cache disabled
    calls = _stub_encode(model)

    assert model._encoder_cache is None
    model.get_audio_feature([_item(7)], forward_batch=None)
    model.get_audio_feature([_item(7)], forward_batch=None)

    assert calls["count"] == 2, "a disabled cache must re-encode every call"


def test_multi_item_batch_bypasses_cache() -> None:
    # mm dispatch is per-request today (len(items) == 1); the rare multi-item
    # batch is deliberately left uncached, so it re-encodes every call.
    model = _make_model(max_bytes=1 << 20)
    calls = _stub_encode(model)

    model.get_audio_feature([_item(1), _item(2)], forward_batch=None)
    model.get_audio_feature([_item(1), _item(2)], forward_batch=None)

    assert calls["count"] == 2


def test_lru_evicts_when_over_budget() -> None:
    # One torch.ones(4) entry is 4 * 4 = 16 bytes; budget holds exactly one.
    model = _make_model(max_bytes=16)
    calls = _stub_encode(model)

    model.get_audio_feature([_item(1)], forward_batch=None)  # miss -> store A
    model.get_audio_feature([_item(2)], forward_batch=None)  # miss -> store B, evict A
    model.get_audio_feature([_item(1)], forward_batch=None)  # miss again: A evicted

    assert calls["count"] == 3
    assert model._encoder_cache is not None
    assert model._encoder_cache.eviction_count >= 1


def test_entry_count_cap_matches_constant() -> None:
    # Alongside the byte budget, the cache is bounded by an entry count (parity
    # with qwen3_omni) so many tiny clips can't grow the dict unbounded.
    model = _make_model(max_bytes=1 << 30)
    assert model._encoder_cache is not None
    assert model._encoder_cache.max_size == _ENCODER_CACHE_MAX_ENTRIES


def test_hit_returns_model_device_tensors() -> None:
    model = _make_model(max_bytes=1 << 20)
    _stub_encode(model)
    expected_device = next(model.vq_adaptor.parameters()).device

    model.get_audio_feature([_item(42)], forward_batch=None)  # populate
    cached = model.get_audio_feature([_item(42)], forward_batch=None)  # hit

    assert cached.device == expected_device
