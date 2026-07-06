# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC relay backed by a bounded sender-side GPU slot pool."""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Callable

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from sglang_omni.profiler.comm_trace import elapsed_ms as _comm_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as _comm_trace
from sglang_omni.profiler.comm_trace import now_ns as _comm_now_ns

from .base import CreditAllocator, Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

_PEER_ENABLED: set[tuple[int, int]] = set()
_PEER_VISIBILITY_WARNED: set[tuple[int, int, int]] = set()

# Completion-wait tuning. The spin window is sized to catch the small,
# latency-sensitive completions that dominate decode (e.g. per-frame hidden
# states, a few microseconds over NVLink) with no added latency; larger copies
# correctly fall through to backoff, where the added detection latency is
# negligible next to a multi-millisecond copy and the CPU is freed meanwhile.
# These are provisional defaults, NOT measured: they should be tuned against the
# Phase-0 profiling (issue #907) on target hardware before being trusted.
_WAIT_SPIN_SECONDS = 20e-6
_WAIT_BACKOFF_MIN_SECONDS = 50e-6
_WAIT_BACKOFF_MAX_SECONDS = 1e-3


async def _await_ready(
    predicate: Callable[[], bool],
    *,
    deadline: float,
    loop: asyncio.AbstractEventLoop,
    timeout_message: str,
) -> int:
    """Wait until ``predicate`` is true without busy-spinning.

    Phase 1 spins tightly for a short bounded window so the common fast
    completion returns with no added latency. Phase 2 falls back to exponential
    backoff with real sleeps, so a slow or hung transfer does not keep the
    event-loop thread (and, via cudaEventQuery, the CUDA driver) fully busy.
    Returns the number of polls performed.
    """
    polls = 0
    spin_deadline = loop.time() + _WAIT_SPIN_SECONDS
    while loop.time() < spin_deadline:
        if predicate():
            return polls
        polls += 1
        await asyncio.sleep(0)
    delay = _WAIT_BACKOFF_MIN_SECONDS
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError(timeout_message)
        polls += 1
        await asyncio.sleep(delay)
        delay = min(delay * 2, _WAIT_BACKOFF_MAX_SECONDS)
    return polls


def _parse_device_id(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def _ensure_peer_access(src_index: int, dst_index: int) -> None:
    """Enable P2P access when available and warn when it is not."""
    if src_index == dst_index:
        return
    key = (dst_index, src_index)
    if key in _PEER_ENABLED:
        return
    if not torch.cuda.can_device_access_peer(dst_index, src_index):
        logger.warning(
            "cuda_ipc: GPU %d cannot peer-access GPU %d; cross-GPU copy will "
            "stage through host memory (no NVLink fast path)",
            dst_index,
            src_index,
        )
    _PEER_ENABLED.add(key)


def _dump_cuda_storage_handle(tensor: torch.Tensor) -> dict[str, Any]:
    (
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = tensor.untyped_storage()._share_cuda_()
    return {
        "storage_device": int(storage_device),
        "storage_handle": storage_handle,
        "storage_size_bytes": int(storage_size_bytes),
        "storage_offset_bytes": int(storage_offset_bytes),
        "ref_counter_handle": ref_counter_handle,
        "ref_counter_offset": int(ref_counter_offset),
        "event_handle": event_handle,
        "event_sync_required": bool(event_sync_required),
        "numel": int(tensor.numel()),
    }


def _load_cuda_storage_handle(
    storage_meta: dict[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    device_index = int(device.index or 0)
    return rebuild_cuda_tensor(
        torch.Tensor,
        (int(storage_meta["numel"]),),
        (1,),
        0,
        torch.UntypedStorage,
        torch.uint8,
        device_index,
        storage_meta["storage_handle"],
        int(storage_meta["storage_size_bytes"]),
        int(storage_meta["storage_offset_bytes"]),
        False,
        storage_meta["ref_counter_handle"],
        int(storage_meta["ref_counter_offset"]),
        storage_meta["event_handle"],
        bool(storage_meta["event_sync_required"]),
    )


class CudaIpcPutOperation(RelayOperation):
    """Sender-side handle; completion means the slot can be reused."""

    def __init__(
        self,
        metadata: dict[str, Any],
        *,
        ready_event: torch.cuda.Event,
        source_tensor: torch.Tensor,
        slot_index: int,
        request_id: str | None,
        size: int,
        release_cb: Callable[[], None],
        fail_cb: Callable[[BaseException], None],
    ) -> None:
        self._metadata = metadata
        self._ready_event: torch.cuda.Event | None = ready_event
        self._source_tensor: torch.Tensor | None = source_tensor
        self._slot_index = slot_index
        self._request_id = request_id
        self._size = size
        self._release_cb = release_cb
        self._fail_cb = fail_cb
        self._completed = False
        self._receiver_done = asyncio.get_running_loop().create_future()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        wait_start = _comm_now_ns()
        try:
            await asyncio.wait_for(self._receiver_done, timeout=timeout)
        except TimeoutError as exc:
            self._completed = True
            self._fail_cb(exc)
            self._source_tensor = None
            self._ready_event = None
            raise
        except Exception as exc:
            self._completed = True
            self._fail_cb(exc)
            self._source_tensor = None
            self._ready_event = None
            raise
        self._completed = True
        self._release_cb()
        self._source_tensor = None
        self._ready_event = None
        _comm_trace(
            "cuda_ipc_put_wait_ack",
            request_id=self._request_id,
            slot_index=self._slot_index,
            bytes=self._size,
            elapsed_ms=round(_comm_elapsed_ms(wait_start), 6),
        )

    def mark_receiver_done(self) -> None:
        if not self._receiver_done.done():
            self._receiver_done.set_result(None)

    def mark_receiver_failed(self, exc: BaseException) -> None:
        if not self._receiver_done.done():
            self._receiver_done.set_exception(exc)


class CudaIpcGetOperation(RelayOperation):
    """Receiver-side handle. Completion means the peer copy finished."""

    def __init__(
        self,
        event: torch.cuda.Event,
        pool_tensor: torch.Tensor,
        slot_index: int,
        request_id: str | None,
        size: int,
    ) -> None:
        self._event = event
        self._pool_tensor: torch.Tensor | None = pool_tensor
        self._slot_index = slot_index
        self._request_id = request_id
        self._size = size
        self._completed = False

    @property
    def metadata(self) -> Any:
        return None

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        wait_start = _comm_now_ns()
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        polls = await _await_ready(
            self._event.query,
            deadline=deadline,
            loop=loop,
            timeout_message="cuda_ipc copy did not complete in time",
        )
        self._completed = True
        self._pool_tensor = None
        _comm_trace(
            "cuda_ipc_get_wait_copy",
            request_id=self._request_id,
            slot_index=self._slot_index,
            bytes=self._size,
            polls=polls,
            elapsed_ms=round(_comm_elapsed_ms(wait_start), 6),
        )


@register_relay("cuda_ipc")
class CudaIpcRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        device: str = "cuda",
        slot_size_mb: int = 512,
        credits: int = 2,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(
                f"unexpected cuda_ipc relay options: {', '.join(sorted(kwargs))}"
            )
        self.engine_id = engine_id
        if device == "cpu":
            raise ValueError(
                "cuda_ipc relay requires a CUDA device; got 'cpu'. Use the shm "
                "relay for host-memory stages."
            )
        self.device = device
        self.device_id = _parse_device_id(device)
        self.slot_size = int(slot_size_mb) * 1024 * 1024
        self.credits = int(credits)
        if self.slot_size <= 0:
            raise ValueError("cuda_ipc slot_size_mb must be positive")
        if self.credits <= 0:
            raise ValueError("cuda_ipc credits must be positive")

        self._pool_tensor: torch.Tensor | None = None
        self._pool_id: str | None = None
        self._pool_storage_handle: dict[str, Any] | None = None
        self._allocator: CreditAllocator | None = None

        self._remote_pools: dict[str, torch.Tensor] = {}
        self._failed_error: BaseException | None = None
        self._failed_event = asyncio.Event()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_local_pool(self) -> None:
        if self._pool_tensor is not None:
            return
        start = _comm_now_ns()
        total_pool_bytes = self.slot_size * self.credits
        device = torch.device(self.device)
        logger.info(
            "[%s] Allocating CUDA-IPC pool: %.2f MB on %s",
            self.engine_id,
            total_pool_bytes / 1024**2,
            self.device,
        )
        with torch.cuda.device(device):
            self._pool_tensor = torch.empty(
                total_pool_bytes, dtype=torch.uint8, device=device
            )
        self._pool_id = f"{self.engine_id}:{os.getpid()}:{uuid.uuid4().hex}"
        self._pool_storage_handle = _dump_cuda_storage_handle(self._pool_tensor)
        self._allocator = CreditAllocator(
            credits=self.credits,
            slot_size=self.slot_size,
            base_ptr=self._pool_tensor.data_ptr(),
        )
        _comm_trace(
            "cuda_ipc_pool_alloc",
            engine_id=self.engine_id,
            device=self.device,
            slot_size=self.slot_size,
            credits=self.credits,
            total_pool_bytes=total_pool_bytes,
            elapsed_ms=round(_comm_elapsed_ms(start), 6),
        )

    def _local_pool_state(
        self,
    ) -> tuple[torch.Tensor, str, dict[str, Any], CreditAllocator]:
        self._ensure_local_pool()
        pool_tensor = self._pool_tensor
        pool_id = self._pool_id
        pool_storage_handle = self._pool_storage_handle
        allocator = self._allocator
        if pool_tensor is None:
            raise RuntimeError("cuda_ipc local pool tensor was not initialized")
        if pool_id is None:
            raise RuntimeError("cuda_ipc local pool id was not initialized")
        if pool_storage_handle is None:
            raise RuntimeError("cuda_ipc local pool storage handle was not initialized")
        if allocator is None:
            raise RuntimeError("cuda_ipc local credit allocator was not initialized")
        return pool_tensor, pool_id, pool_storage_handle, allocator

    def _mark_failed(self, exc: BaseException) -> None:
        if self._failed_error is None:
            self._failed_error = exc
            self._failed_event.set()

    def _raise_if_failed(self) -> None:
        if self._failed_error is not None:
            raise RuntimeError("cuda_ipc relay failed") from self._failed_error

    async def _acquire_slot(self, allocator: CreditAllocator) -> int:
        self._raise_if_failed()
        acquire_task = asyncio.create_task(allocator.acquire_async())
        fail_task = asyncio.create_task(self._failed_event.wait())
        try:
            done, _ = await asyncio.wait(
                {acquire_task, fail_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if fail_task in done:
                if acquire_task in done:
                    allocator.release(acquire_task.result())
                self._raise_if_failed()
                raise RuntimeError("cuda_ipc relay failed")

            offset = acquire_task.result()
            try:
                self._raise_if_failed()
            except Exception:
                allocator.release(offset)
                raise
            return offset
        finally:
            for task in (acquire_task, fail_task):
                if not task.done():
                    task.cancel()

    def _get_remote_pool(
        self,
        metadata: dict[str, Any],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        ipc_meta = metadata["cuda_ipc"]
        pool_id = ipc_meta["pool_id"]
        pool = self._remote_pools.get(pool_id)
        if pool is None:
            storage_meta = ipc_meta["pool_storage"]
            if not isinstance(storage_meta, dict):
                raise TypeError(
                    "cuda_ipc pool_storage metadata must be a dict, got "
                    f"{type(storage_meta).__name__}"
                )
            pool = _load_cuda_storage_handle(storage_meta, device=device)
            self._remote_pools[pool_id] = pool
        return pool

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
    ) -> CudaIpcPutOperation:
        self._raise_if_failed()
        if not tensor.is_cuda:
            raise ValueError(
                "cuda_ipc relay can only transfer CUDA tensors; "
                f"got tensor on {tensor.device}"
            )
        (
            pool_tensor,
            pool_id,
            pool_storage_handle,
            allocator,
        ) = self._local_pool_state()

        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        size = int(flat.numel())
        if size > self.slot_size:
            raise ValueError(
                f"Tensor size {size} exceeds cuda_ipc slot size {self.slot_size}"
            )

        acquire_start = _comm_now_ns()
        offset = await self._acquire_slot(allocator)
        acquire_ms = _comm_elapsed_ms(acquire_start)
        slot_index = int(offset // self.slot_size)

        try:
            copy_start = _comm_now_ns()
            pool_slice = pool_tensor[offset : offset + size]
            device = torch.device(self.device)
            stream = torch.cuda.current_stream(device)
            ready_event = torch.cuda.Event(interprocess=True)
            with torch.cuda.device(device), torch.cuda.stream(stream):
                pool_slice.copy_(flat, non_blocking=True)
                ready_event.record(stream)
            copy_enqueue_ms = _comm_elapsed_ms(copy_start)
            handle_start = _comm_now_ns()
            ready_handle = ready_event.ipc_handle()
            handle_ms = _comm_elapsed_ms(handle_start)
        except Exception:
            allocator.release(offset)
            raise
        _comm_trace(
            "cuda_ipc_put_async",
            request_id=request_id,
            engine_id=self.engine_id,
            device=self.device,
            bytes=size,
            slot_index=slot_index,
            acquire_ms=round(acquire_ms, 6),
            copy_enqueue_ms=round(copy_enqueue_ms, 6),
            event_handle_ms=round(handle_ms, 6),
        )

        metadata = {
            "engine_id": self.engine_id,
            "transfer_info": {
                "size": size,
                "offset": int(offset),
                "slot_index": slot_index,
                "slot_size": self.slot_size,
            },
            "cuda_ipc": {
                "pool_id": pool_id,
                "pool_storage": pool_storage_handle,
                "src_device_id": self.device_id,
                "ready_event": ready_handle,
            },
        }
        return CudaIpcPutOperation(
            metadata,
            ready_event=ready_event,
            source_tensor=flat,
            slot_index=slot_index,
            request_id=request_id,
            size=size,
            release_cb=lambda: allocator.release(offset),
            fail_cb=self._mark_failed,
        )

    async def get_async(
        self,
        metadata: dict[str, Any],
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
    ) -> CudaIpcGetOperation:
        if not dest_tensor.is_cuda:
            raise ValueError(
                "cuda_ipc relay can only receive into CUDA tensors; "
                f"dest is on {dest_tensor.device}"
            )
        start = _comm_now_ns()
        ipc_meta = metadata["cuda_ipc"]
        dst_device = dest_tensor.device
        pool_start = _comm_now_ns()
        pool_tensor = self._get_remote_pool(metadata, device=dst_device)
        pool_ms = _comm_elapsed_ms(pool_start)

        src_index = int(ipc_meta["src_device_id"])
        dst_index = int(dst_device.index or 0)
        peer_start = _comm_now_ns()
        device_count = torch.cuda.device_count()
        if 0 <= src_index < device_count:
            _ensure_peer_access(src_index, dst_index)
        else:
            warn_key = (dst_index, src_index, device_count)
            if warn_key not in _PEER_VISIBILITY_WARNED:
                _PEER_VISIBILITY_WARNED.add(warn_key)
                logger.warning(
                    "cuda_ipc source device %d is outside this receiver's visible "
                    "CUDA device range [0, %d); peer-access validation skipped. "
                    "This is expected only when sender and receiver use different "
                    "CUDA_VISIBLE_DEVICES namespaces.",
                    src_index,
                    device_count,
                )
        peer_ms = _comm_elapsed_ms(peer_start)

        size = int(metadata["transfer_info"]["size"])
        offset = int(metadata["transfer_info"]["offset"])
        slot_index = int(metadata["transfer_info"]["slot_index"])
        # Import on the waiting device; source-device imports can hang cross-GPU.
        event_start = _comm_now_ns()
        ready_event = torch.cuda.Event.from_ipc_handle(
            dst_device, ipc_meta["ready_event"]
        )
        event_ms = _comm_elapsed_ms(event_start)

        copy_start = _comm_now_ns()
        src = pool_tensor.view(torch.uint8).reshape(-1)[offset : offset + size]
        dst = dest_tensor.view(torch.uint8).reshape(-1)
        if dst.numel() < size:
            raise ValueError(
                f"cuda_ipc destination buffer has {dst.numel()} bytes, "
                f"but transfer requires {size} bytes"
            )
        copy_len = size

        stream = torch.cuda.current_stream(dst_device)
        with torch.cuda.device(dst_device), torch.cuda.stream(stream):
            stream.wait_event(ready_event)
            dst[:copy_len].copy_(src[:copy_len], non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        copy_enqueue_ms = _comm_elapsed_ms(copy_start)
        _comm_trace(
            "cuda_ipc_get_async",
            request_id=request_id,
            engine_id=self.engine_id,
            src_device=src_index,
            dst_device=dst_index,
            bytes=size,
            copy_len=int(copy_len),
            slot_index=slot_index,
            pool_open_ms=round(pool_ms, 6),
            peer_access_ms=round(peer_ms, 6),
            event_import_ms=round(event_ms, 6),
            copy_enqueue_ms=round(copy_enqueue_ms, 6),
            elapsed_ms=round(_comm_elapsed_ms(start), 6),
        )
        return CudaIpcGetOperation(
            event,
            pool_tensor,
            slot_index,
            request_id=request_id,
            size=size,
        )

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        self._remote_pools.clear()
        self._pool_tensor = None
        self._pool_storage_handle = None
        self._allocator = None
