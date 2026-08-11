# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .semantic_vad import SemanticEOUModel, SemanticTurnDetector, SemanticVADConfig
from .vad import Emit, StreamingVAD, VADConfig


class TurnDetector(Protocol):
    def process(self, pcm_bytes: bytes) -> list[Emit]: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class TurnDetectorBuild:
    detector: TurnDetector
    effective_config: dict[str, Any]


def build_turn_detector(
    config: Mapping[str, Any],
    smart_turn_model: SemanticEOUModel | None,
) -> TurnDetectorBuild:
    raw_type = config.get("type")
    requested_type = str(getattr(raw_type, "value", raw_type) or "server_vad")
    if requested_type == "semantic_vad" and smart_turn_model is not None:
        eagerness = str(config.get("eagerness") or "medium")
        detector = SemanticTurnDetector(
            smart_turn_model,
            SemanticVADConfig.from_eagerness(eagerness),
        )
        effective = dict(config)
        effective["type"] = "semantic_vad"
        effective["eagerness"] = eagerness
        effective.pop("silence_duration_ms", None)
        return TurnDetectorBuild(detector, effective)

    server_config = VADConfig(
        threshold=_optional_float(config.get("threshold"), VADConfig.threshold),
        prefix_padding_ms=_optional_int(
            config.get("prefix_padding_ms"), VADConfig.prefix_padding_ms
        ),
        silence_duration_ms=_optional_int(
            config.get("silence_duration_ms"), VADConfig.silence_duration_ms
        ),
    )
    effective = dict(config)
    effective["type"] = "server_vad"
    effective["eagerness"] = None
    return TurnDetectorBuild(StreamingVAD(server_config), effective)


def _optional_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _optional_int(value: Any, default: int) -> int:
    return default if value is None else int(value)
