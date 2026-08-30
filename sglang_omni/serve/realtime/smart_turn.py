# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import WhisperFeatureExtractor

from .semantic_vad import SemanticEOUModel

SMART_TURN_MODEL_ENV = "SGLANG_OMNI_SMART_TURN_MODEL_PATH"
SMART_TURN_DEVICE_ENV = "SGLANG_OMNI_SMART_TURN_DEVICE"

SMART_TURN_MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_MODEL_SHA256 = (
    "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
)

# FP32 variant sized for CUDAExecutionProvider; benchmark-only alternative to
# the default INT8 CPU model, selected via SMART_TURN_DEVICE_ENV=cuda.
SMART_TURN_GPU_MODEL_FILENAME = "smart-turn-v3.2-gpu.onnx"
SMART_TURN_GPU_MODEL_SHA256 = (
    "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be"
)


@dataclass(frozen=True)
class SmartTurnEOU(SemanticEOUModel):
    session: Any
    feature_extractor: WhisperFeatureExtractor

    @classmethod
    def load(cls, model_path: Path | str, *, device: str = "cpu") -> SmartTurnEOU:
        filename, expected_sha256 = (
            (SMART_TURN_GPU_MODEL_FILENAME, SMART_TURN_GPU_MODEL_SHA256)
            if device == "cuda"
            else (SMART_TURN_MODEL_FILENAME, SMART_TURN_MODEL_SHA256)
        )
        resolved_path = _resolve_model_path(model_path, filename)
        _verify_checksum(resolved_path, expected_sha256)
        session = _load_model(resolved_path, device=device)
        feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        return cls(session=session, feature_extractor=feature_extractor)

    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        if sample_rate != 16000:
            raise ValueError("Smart Turn requires 16 kHz audio")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        max_samples = sample_rate * 8
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        if audio.size < max_samples:
            audio = np.pad(audio, (max_samples - audio.size, 0))

        features = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="np",
            padding="max_length",
            max_length=max_samples,
            truncation=True,
            do_normalize=True,
        ).input_features
        input_features = features.squeeze(0).astype(np.float32, copy=False)[None, ...]
        output = self.session.run(None, {"input_features": input_features})[0]
        probability = float(np.asarray(output).reshape(-1)[0])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Smart Turn returned invalid probability: {probability}")
        return probability


def load_smart_turn() -> SmartTurnEOU | None:
    configured_path = os.getenv(SMART_TURN_MODEL_ENV)
    if not configured_path:
        return None
    device = os.getenv(SMART_TURN_DEVICE_ENV, "cpu")
    return SmartTurnEOU.load(configured_path, device=device)


def _resolve_model_path(model_path: Path | str, filename: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_dir():
        path = path / filename
    if not path.is_file():
        raise FileNotFoundError(f"Smart Turn model not found at {path}")
    return path


def _verify_checksum(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Smart Turn checksum mismatch for {path}: expected "
            f"{expected_sha256}, got {actual}"
        )


def _load_model(model_path: Path, *, device: str = "cpu") -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
    )
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "input_features":
        raise RuntimeError("Unexpected Smart Turn ONNX input contract")
    return session
