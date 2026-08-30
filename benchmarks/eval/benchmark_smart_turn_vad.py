# SPDX-License-Identifier: Apache-2.0
"""Smart Turn VAD concurrency benchmark.

Measures turn-decision latency percentiles (p50/p95/p99) and host CPU
utilization/core count at representative concurrency (1/8/32 sessions by
default), for the CPU/INT8 Smart Turn model this repo runs by default and,
via ``--device cuda``, the FP32 GPU variant (see
``sglang_omni/serve/realtime/smart_turn.py``: ``SGLANG_OMNI_SMART_TURN_DEVICE``).
The goal is to substantiate (or refute) the claim that the shared CPU
inference path has no throughput impact under concurrent realtime sessions,
including while Qwen3-Omni is actively generating.

Two modes:

``inprocess`` (no server needed) drives ``SemanticTurnDetector`` directly, one
instance per simulated session, all sharing a single preloaded Smart Turn
model exactly as ``RealtimeSession`` does in production. Each session feeds a
real spoken-audio WAV plus trailing silence through
``asyncio.to_thread(detector.process, chunk)`` at real-time pace -- the same
dispatch the server uses in ``handle_audio_append``. This isolates the
compute cost of turn-detection itself and needs only onnxruntime (+
onnxruntime-gpu for ``--device cuda``) and the Smart Turn model file; it does
not need Qwen3-Omni loaded or a GPU for the CPU device.

    python -m benchmarks.eval.benchmark_smart_turn_vad inprocess \\
        --model-path /path/to/smart-turn-v3.2-cpu.onnx \\
        --device cpu --concurrency 1,8,32 --label cpu-idle

``server`` (requires a running ``/v1/realtime`` endpoint) drives real
WebSocket sessions against sglang-omni with ``semantic_vad`` enabled and
times ``input_audio_buffer.speech_started`` -> ``speech_stopped`` as observed
on the wire. Pass ``--generating-sessions`` to hold N sessions continuously
talking in the background, reproducing "Smart Turn running while Qwen3-Omni
is generating" -- the scenario the PR does not currently measure.

    python -m benchmarks.eval.benchmark_smart_turn_vad server \\
        --base-url ws://localhost:8008/v1/realtime \\
        --concurrency 1,8,32 --generating-sessions 4 --label cpu-loaded

Run the same command with ``--device cuda`` (inprocess) or against a server
started with ``SGLANG_OMNI_SMART_TURN_DEVICE=cuda`` (server) to get the
comparison point against the FP32 GPU model.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.utils import save_json_results  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_WAV = (
    Path(__file__).resolve().parents[2] / "tests" / "data" / "query_to_draw.wav"
)
VAD_SAMPLE_RATE = 16000
CHUNK_MS = 20
TRAILING_SILENCE_S = 3.0
SAMPLE_INTERVAL_S = 0.2


# --------------------------------------------------------------------------
# Percentiles
# --------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


# --------------------------------------------------------------------------
# CPU / GPU samplers
# --------------------------------------------------------------------------


class ProcessCPUSampler:
    """Portable (macOS/Linux) sampler for *this process's* CPU usage.

    Correct for ``inprocess`` mode, where the Smart Turn compute happens via
    ``asyncio.to_thread`` inside this same process. Utilization is
    (delta user+sys CPU time) / (delta wall time), normalized by core count.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.core_count = os.cpu_count() or 1
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        last_times = os.times()
        last_wall = time.perf_counter()
        while not self._stop.wait(self.interval_s):
            times = os.times()
            wall = time.perf_counter()
            cpu_delta = (times.user - last_times.user) + (
                times.system - last_times.system
            )
            wall_delta = wall - last_wall
            if wall_delta > 0:
                self._samples.append(100.0 * cpu_delta / wall_delta / self.core_count)
            last_times, last_wall = times, wall

    def __enter__(self) -> ProcessCPUSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)

    def summary(self) -> dict[str, float | int]:
        if not self._samples:
            return {"core_count": self.core_count, "mean_pct": 0.0, "max_pct": 0.0}
        return {
            "core_count": self.core_count,
            "mean_pct": statistics.fmean(self._samples),
            "max_pct": max(self._samples),
            "samples": len(self._samples),
        }


class HostCPUSampler:
    """Linux ``/proc/stat``-based system-wide CPU sampler.

    Use this for ``server`` mode when running the benchmark client on the
    same host as the sglang-omni server (e.g. both over one SSH session on
    the GPU pod) -- the compute you care about happens in the server
    process, not this script, so a process-local sampler would be blind to
    it.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.available = os.path.exists("/proc/stat")
        self.core_count = self._read_core_count()
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read_core_count() -> int:
        try:
            with open("/proc/stat") as f:
                return sum(
                    1 for line in f if line.startswith("cpu") and line[3].isdigit()
                )
        except OSError:
            return os.cpu_count() or 1

    @staticmethod
    def _read_agg_cpu() -> tuple[int, int]:
        with open("/proc/stat") as f:
            fields = f.readline().split()[1:]
        values = [int(v) for v in fields]
        idle = values[3] + values[4]  # idle + iowait
        total = sum(values)
        return idle, total

    def _run(self) -> None:
        last_idle, last_total = self._read_agg_cpu()
        while not self._stop.wait(self.interval_s):
            idle, total = self._read_agg_cpu()
            d_idle = idle - last_idle
            d_total = total - last_total
            if d_total > 0:
                self._samples.append(100.0 * (1.0 - d_idle / d_total))
            last_idle, last_total = idle, total

    def __enter__(self) -> HostCPUSampler:
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            logger.warning("/proc/stat unavailable; host CPU utilization not sampled")
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)

    def summary(self) -> dict[str, float | int]:
        if not self._samples:
            return {"core_count": self.core_count, "mean_pct": 0.0, "max_pct": 0.0}
        return {
            "core_count": self.core_count,
            "mean_pct": statistics.fmean(self._samples),
            "max_pct": max(self._samples),
            "samples": len(self._samples),
        }


class GPUSampler:
    """Optional ``nvidia-smi``-polling GPU utilization sampler.

    No-ops cleanly (``available=False``) when ``nvidia-smi`` isn't on PATH,
    so the same script runs on a laptop for local dry-runs.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.available = self._check_nvidia_smi()
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _check_nvidia_smi() -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                timeout=5,
                check=True,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout
                for line in out.strip().splitlines():
                    self._samples.append(float(line.strip()))
            except (OSError, subprocess.SubprocessError, ValueError):
                continue

    def __enter__(self) -> GPUSampler:
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)

    def summary(self) -> dict[str, float] | None:
        if not self.available or not self._samples:
            return None
        return {
            "mean_pct": statistics.fmean(self._samples),
            "max_pct": max(self._samples),
        }


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------


def load_pcm16(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as w:
        if (
            w.getframerate() != VAD_SAMPLE_RATE
            or w.getnchannels() != 1
            or w.getsampwidth() != 2
        ):
            raise ValueError(
                f"{wav_path} must be 16 kHz mono PCM16, got "
                f"{w.getframerate()}Hz ch={w.getnchannels()} width={w.getsampwidth()}"
            )
        return w.readframes(w.getnframes())


def trim_trailing_silence(
    pcm: bytes, *, threshold: float = 0.5, pad_frames: int = 3
) -> bytes:
    """Trim ``pcm`` to end shortly after the last Silero-detected speech frame.

    Source WAV fixtures often have their own trailing silence baked in. Left
    untrimmed, the detector can decide the turn is over mid-clip, before the
    benchmark starts watching for ``SPEECH_STOPPED`` -- silently invalidating
    the latency measurement. This makes "end of fed audio" line up with the
    real acoustic end of speech, independent of the detector under test.
    """
    import numpy as np

    from sglang_omni.serve.realtime.semantic_vad import SileroSpeechModel
    from sglang_omni.serve.realtime.vad import VAD_FRAME_SAMPLES

    probe = SileroSpeechModel()
    frame_bytes = VAD_FRAME_SAMPLES * 2
    last_speech_frame = -1
    n_frames = len(pcm) // frame_bytes
    for i in range(n_frames):
        chunk = pcm[i * frame_bytes : (i + 1) * frame_bytes]
        frame = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
        if probe.predict(frame, VAD_SAMPLE_RATE) >= threshold:
            last_speech_frame = i
    if last_speech_frame < 0:
        raise ValueError("no speech detected in source WAV; pick a different fixture")
    end_frame = min(n_frames, last_speech_frame + 1 + pad_frames)
    return pcm[: end_frame * frame_bytes]


def chunk_audio(pcm: bytes, chunk_ms: int = CHUNK_MS) -> list[bytes]:
    chunk_bytes = int(VAD_SAMPLE_RATE * chunk_ms / 1000) * 2
    return [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)] or [b""]


def silence_chunks(duration_s: float, chunk_ms: int = CHUNK_MS) -> list[bytes]:
    chunk_bytes = int(VAD_SAMPLE_RATE * chunk_ms / 1000) * 2
    n_chunks = int(duration_s * 1000 / chunk_ms)
    return [b"\x00" * chunk_bytes for _ in range(n_chunks)]


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class TrialResult:
    concurrency: int
    latency_s: dict[str, float]
    process_cpu: dict[str, float | int] | None = None
    host_cpu: dict[str, float | int] | None = None
    gpu: dict[str, float] | None = None
    turns_completed: int = 0
    wall_seconds: float = 0.0


@dataclass
class BenchmarkSummary:
    label: str
    mode: str
    device: str
    trials: list[TrialResult] = field(default_factory=list)


def _print_summary(summary: BenchmarkSummary) -> None:
    print("\n" + "=" * 78)
    print(
        f"  Smart Turn VAD benchmark — label={summary.label} mode={summary.mode} "
        f"device={summary.device}"
    )
    print("=" * 78)
    for t in summary.trials:
        lat = t.latency_s
        print(
            f"  concurrency={t.concurrency:<3} turns={t.turns_completed:<5} "
            f"p50={lat.get('p50', float('nan')):.3f}s p95={lat.get('p95', float('nan')):.3f}s "
            f"p99={lat.get('p99', float('nan')):.3f}s"
        )
        if t.process_cpu:
            print(
                f"      process CPU: mean={t.process_cpu['mean_pct']:.1f}% "
                f"max={t.process_cpu['max_pct']:.1f}% (of {t.process_cpu['core_count']} cores)"
            )
        if t.host_cpu:
            print(
                f"      host CPU:    mean={t.host_cpu['mean_pct']:.1f}% "
                f"max={t.host_cpu['max_pct']:.1f}% (of {t.host_cpu['core_count']} cores)"
            )
        if t.gpu:
            print(
                f"      GPU util:    mean={t.gpu['mean_pct']:.1f}% max={t.gpu['max_pct']:.1f}%"
            )
    print("=" * 78)


# --------------------------------------------------------------------------
# inprocess mode
# --------------------------------------------------------------------------


async def _inprocess_session(
    detector_factory,
    wav_chunks: list[bytes],
    silence: list[bytes],
    turns: int,
    latencies: list[float],
    stop_event: asyncio.Event,
) -> int:
    from sglang_omni.serve.realtime.vad import VADEvent

    detector = detector_factory()
    completed = 0
    for _ in range(turns):
        if stop_event.is_set():
            break
        for chunk in wav_chunks:
            await asyncio.to_thread(detector.process, chunk)
            await asyncio.sleep(CHUNK_MS / 1000)
        speech_end_wall = time.perf_counter()
        stopped = False
        for chunk in silence:
            emits = await asyncio.to_thread(detector.process, chunk)
            for emit in emits:
                if emit.event_type == VADEvent.SPEECH_STOPPED:
                    latencies.append(time.perf_counter() - speech_end_wall)
                    stopped = True
            if stopped:
                break
            await asyncio.sleep(CHUNK_MS / 1000)
        detector.reset()
        if stopped:
            completed += 1
        else:
            logger.warning(
                "turn never reached SPEECH_STOPPED within trailing silence window"
            )
        await asyncio.sleep(0.05)
    return completed


async def _run_inprocess(args: argparse.Namespace) -> BenchmarkSummary:
    from sglang_omni.serve.realtime.smart_turn import SmartTurnEOU
    from sglang_omni.serve.realtime.turn_detector import build_turn_detector

    logger.info(
        "Loading Smart Turn model (device=%s) from %s", args.device, args.model_path
    )
    smart_turn_model = SmartTurnEOU.load(args.model_path, device=args.device)

    def detector_factory():
        build = build_turn_detector(
            {"type": "semantic_vad", "eagerness": args.eagerness}, smart_turn_model
        )
        return build.detector

    pcm = trim_trailing_silence(load_pcm16(args.wav))
    wav_chunks = chunk_audio(pcm)
    silence = silence_chunks(TRAILING_SILENCE_S)

    summary = BenchmarkSummary(label=args.label, mode="inprocess", device=args.device)
    for concurrency in args.concurrency:
        logger.info("=== concurrency=%d ===", concurrency)
        latencies: list[float] = []
        stop_event = asyncio.Event()
        start = time.perf_counter()
        with ProcessCPUSampler() as cpu_sampler, GPUSampler() as gpu_sampler:
            results = await asyncio.gather(
                *[
                    _inprocess_session(
                        detector_factory,
                        wav_chunks,
                        silence,
                        args.turns_per_session,
                        latencies,
                        stop_event,
                    )
                    for _ in range(concurrency)
                ]
            )
        wall = time.perf_counter() - start
        summary.trials.append(
            TrialResult(
                concurrency=concurrency,
                latency_s=_latency_stats(latencies),
                process_cpu=cpu_sampler.summary(),
                gpu=gpu_sampler.summary(),
                turns_completed=sum(results),
                wall_seconds=wall,
            )
        )
        logger.info(
            "concurrency=%d turns=%d p50=%.3fs p95=%.3fs p99=%.3fs cpu_mean=%.1f%%",
            concurrency,
            sum(results),
            summary.trials[-1].latency_s.get("p50", float("nan")),
            summary.trials[-1].latency_s.get("p95", float("nan")),
            summary.trials[-1].latency_s.get("p99", float("nan")),
            cpu_sampler.summary()["mean_pct"],
        )
    return summary


# --------------------------------------------------------------------------
# server mode
# --------------------------------------------------------------------------


async def _server_measurement_session(
    base_url: str,
    wav_chunks: list[bytes],
    silence: list[bytes],
    turns: int,
    latencies: list[float],
    eagerness: str,
) -> int:
    import websockets

    completed = 0
    async with websockets.connect(base_url, max_size=None) as ws:
        await ws.recv()  # session.created
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": eagerness,
                        },
                    },
                }
            )
        )
        await ws.recv()  # session.updated

        for _ in range(turns):
            for chunk in wav_chunks:
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
                await asyncio.sleep(CHUNK_MS / 1000)
            speech_end_wall = time.perf_counter()
            for chunk in silence:
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=CHUNK_MS / 1000)
                    event = json.loads(raw)
                    if event.get("type") == "input_audio_buffer.speech_stopped":
                        latencies.append(time.perf_counter() - speech_end_wall)
                        completed += 1
                        break
                except TimeoutError:
                    pass
                await asyncio.sleep(CHUNK_MS / 1000)
    return completed


async def _server_generating_session(
    base_url: str,
    wav_chunks: list[bytes],
    silence: list[bytes],
    stop_event: asyncio.Event,
) -> None:
    """Background session that keeps sending speech to hold Qwen3-Omni busy."""
    import websockets

    try:
        async with websockets.connect(base_url, max_size=None) as ws:
            await ws.recv()
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "modalities": ["text", "audio"],
                            "turn_detection": {"type": "server_vad"},
                        },
                    }
                )
            )
            await ws.recv()
            while not stop_event.is_set():
                for chunk in wav_chunks + silence[:25]:
                    if stop_event.is_set():
                        return
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("ascii"),
                            }
                        )
                    )
                    await asyncio.sleep(CHUNK_MS / 1000)
    except Exception:
        logger.exception("generating session failed")


async def _run_server(args: argparse.Namespace) -> BenchmarkSummary:
    pcm = trim_trailing_silence(load_pcm16(args.wav))
    wav_chunks = chunk_audio(pcm)
    silence = silence_chunks(TRAILING_SILENCE_S)

    summary = BenchmarkSummary(label=args.label, mode="server", device=args.device)
    for concurrency in args.concurrency:
        logger.info(
            "=== concurrency=%d generating_sessions=%d ===",
            concurrency,
            args.generating_sessions,
        )
        latencies: list[float] = []
        stop_event = asyncio.Event()
        generating_tasks = [
            asyncio.create_task(
                _server_generating_session(
                    args.base_url, wav_chunks, silence, stop_event
                )
            )
            for _ in range(args.generating_sessions)
        ]
        start = time.perf_counter()
        with HostCPUSampler() as cpu_sampler, GPUSampler() as gpu_sampler:
            results = await asyncio.gather(
                *[
                    _server_measurement_session(
                        args.base_url,
                        wav_chunks,
                        silence,
                        args.turns_per_session,
                        latencies,
                        args.eagerness,
                    )
                    for _ in range(concurrency)
                ],
                return_exceptions=True,
            )
        wall = time.perf_counter() - start
        stop_event.set()
        for task in generating_tasks:
            task.cancel()
        await asyncio.gather(*generating_tasks, return_exceptions=True)

        completed = sum(r for r in results if isinstance(r, int))
        summary.trials.append(
            TrialResult(
                concurrency=concurrency,
                latency_s=_latency_stats(latencies),
                host_cpu=cpu_sampler.summary(),
                gpu=gpu_sampler.summary(),
                turns_completed=completed,
                wall_seconds=wall,
            )
        )
        logger.info(
            "concurrency=%d turns=%d p50=%.3fs p95=%.3fs p99=%.3fs host_cpu_mean=%.1f%%",
            concurrency,
            completed,
            summary.trials[-1].latency_s.get("p50", float("nan")),
            summary.trials[-1].latency_s.get("p95", float("nan")),
            summary.trials[-1].latency_s.get("p99", float("nan")),
            cpu_sampler.summary()["mean_pct"],
        )
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_concurrency(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _default_output_path(label: str) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"smart_turn_vad_{label}_{run_id}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--label", required=True, help="Run label, e.g. 'cpu-idle'.")
    common.add_argument(
        "--concurrency", type=_parse_concurrency, default=_parse_concurrency("1,8,32")
    )
    common.add_argument(
        "--eagerness", default="medium", choices=["low", "medium", "high"]
    )
    common.add_argument("--turns-per-session", type=int, default=5)
    common.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    common.add_argument("--output", type=Path, default=None)

    p_inprocess = sub.add_parser("inprocess", parents=[common])
    p_inprocess.add_argument("--model-path", required=True, type=Path)
    p_inprocess.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    p_server = sub.add_parser("server", parents=[common])
    p_server.add_argument(
        "--base-url", required=True, help="ws://host:port/v1/realtime"
    )
    p_server.add_argument(
        "--generating-sessions",
        type=int,
        default=0,
        help="Background sessions kept continuously talking to hold Qwen3-Omni "
        "busy generating, to reproduce CPU contention with live decoding.",
    )
    p_server.add_argument(
        "--device",
        default="unspecified",
        help="Informational only — set SGLANG_OMNI_SMART_TURN_DEVICE on the "
        "server itself; this flag just labels the output.",
    )

    args = parser.parse_args(argv)
    if args.output is None:
        args.output = _default_output_path(args.label)

    if args.mode == "inprocess":
        summary = asyncio.run(_run_inprocess(args))
    else:
        summary = asyncio.run(_run_server(args))

    result_dict = {
        "label": summary.label,
        "mode": summary.mode,
        "device": summary.device,
        "trials": [asdict(t) for t in summary.trials],
    }
    save_json_results(
        result_dict, str(args.output.parent) or "results", args.output.name
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
