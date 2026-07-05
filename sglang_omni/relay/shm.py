# SPDX-License-Identifier: Apache-2.0
"""Host shared-memory relay.

This is the host-memory transport: it carries CPU tensors between same-node
processes. GPU-to-GPU edges go through the cuda_ipc relay (NVLink), so the
transport router only selects shm for host-resident data and always builds it
with ``device="cpu"``. Buffers therefore arrive here already on the host.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from multiprocessing import shared_memory as _shm
from typing import Any, Callable

import numpy as np
import torch

from .base import Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)

# Consume-wait tuning. shm is a cold, CPU-only path after the cuda_ipc split, so a
# larger backoff cap than the GPU path is fine. Provisional defaults; tune with the
# Phase-0 profiling (#907) if a CPU edge ever turns hot.
_CONSUME_SPIN_SECONDS = 20e-6
_CONSUME_BACKOFF_MIN_SECONDS = 50e-6
_CONSUME_BACKOFF_MAX_SECONDS = 5e-3


def shm_create_from_tensor(tensor: torch.Tensor) -> _shm.SharedMemory:
    """Creates a SHM block and writes tensor data into it (optimized single copy)."""
    t_np = tensor.numpy().reshape(-1)
    size = t_np.nbytes

    # 1. Create SHM directly
    shm = _shm.SharedMemory(create=True, size=size)

    # 2. Create a numpy view based on SHM memory
    # This step is instantaneous and involves no copying
    shm_view = np.ndarray(t_np.shape, dtype=t_np.dtype, buffer=shm.buf)

    # 3. Direct data copy (Only One Copy)
    # Uses low-level C memcpy to write directly from source Tensor to SHM
    shm_view[:] = t_np[:]

    return shm


class ShmOperation(RelayOperation):
    """Base class implementation for SHM operations."""

    def __init__(self, metadata: Any):
        self._metadata = metadata
        self._completed = False

    @property
    def metadata(self) -> Any:
        return self._metadata

    # wait_for_completion is implemented by subclasses


class ShmPutOperation(ShmOperation):
    """
    Handle for Put.

    The sender holds its flow-control credit until the receiver has consumed the
    block, which the receiver signals by unlinking it (the entry disappears from
    /dev/shm). This gives real bounded-outstanding backpressure -- at most
    ``credits`` un-consumed blocks exist at once -- by reusing the unlink the
    receiver already performs, with no separate ack channel.
    """

    def __init__(
        self,
        metadata: Any,
        shm_obj: _shm.SharedMemory,
        *,
        shm_name: str,
        release_cb: Callable[[], None],
    ):
        super().__init__(metadata)
        self._shm_obj = shm_obj
        self._shm_name = shm_name
        self._release_cb = release_cb

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return
        # Sender closes its own handle; the receiver owns the unlink.
        self._shm_obj.close()
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        shm_path = f"/dev/shm/{self._shm_name}"
        spin_deadline = loop.time() + _CONSUME_SPIN_SECONDS
        delay = _CONSUME_BACKOFF_MIN_SECONDS
        # Hold the credit until the receiver consumes the block (it disappears),
        # spinning briefly then backing off so an absent consumer does not
        # busy-spin. TODO(comm): polling /dev/shm is Linux-specific and a mild
        # hack; a consumer->producer ack (or folding shm into the DataRef
        # lifecycle) is cleaner. shm is a cold CPU-only path after the cuda_ipc
        # split, so this easy-win is intentionally not a full pool + ack channel.
        try:
            while os.path.exists(shm_path):
                if loop.time() > deadline:
                    raise TimeoutError(
                        f"shm block {self._shm_name} was not consumed in time"
                    )
                if loop.time() < spin_deadline:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, _CONSUME_BACKOFF_MAX_SECONDS)
        except TimeoutError:
            # Reclaim the named block before releasing credit so repeated absent
            # consumers cannot fill /dev/shm.
            try:
                self._shm_obj.unlink()
            except FileNotFoundError:
                pass
            raise
        finally:
            # Always release the credit, even on timeout, so an absent or crashed
            # consumer cannot permanently leak a slot.
            self._completed = True
            self._release_cb()


class ShmGetOperation(ShmOperation):
    """
    Handle for Get.
    Performs copy from SHM to destination tensor and unlinks the shared memory.
    """

    def __init__(self, metadata: Any, dest_tensor: torch.Tensor):
        super().__init__(metadata)
        self._transfer_info = metadata["transfer_info"]
        self._dest_tensor = dest_tensor

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        shm_name = self._transfer_info["shm_name"]
        size = self._transfer_info["size"]

        try:
            # 1. Open SHM
            try:
                existing_shm = _shm.SharedMemory(name=shm_name)
            except FileNotFoundError:
                raise RuntimeError(f"SHM block {shm_name} not found.")

            try:
                # 2. Zero-copy Read -> Copy to Dest
                shm_array = np.ndarray((size,), dtype=np.uint8, buffer=existing_shm.buf)
                src_tensor = torch.from_numpy(shm_array)

                dest_view = self._dest_tensor.view(torch.uint8).reshape(-1)
                copy_len = min(dest_view.numel(), size)
                dest_view[:copy_len].copy_(src_tensor[:copy_len])

            finally:
                # 3. Cleanup (Receiver owns lifecycle)
                existing_shm.close()
                try:
                    existing_shm.unlink()
                except FileNotFoundError:
                    pass

        finally:
            self._completed = True


@register_relay("shm")
class ShmRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        slot_size_mb: int = 64,
        credits: int = 2,
        device: str = "cpu",
    ):
        self.engine_id = engine_id
        self.device = device
        # Semaphore mimics the 'credits' flow control
        self._sem = asyncio.Semaphore(credits)
        self._slot_size_bytes = slot_size_mb * 1024 * 1024

    async def put_async(
        self, tensor: torch.Tensor, request_id: str = None, dst_rank: int = None
    ) -> RelayOperation:
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Acquire a credit and HOLD it until the receiver consumes the block (real
        # bounded-outstanding backpressure); it is released in
        # ShmPutOperation.wait_for_completion once the block is unlinked.
        await self._sem.acquire()

        try:
            shm = shm_create_from_tensor(tensor)
            size_bytes = shm.size
            metadata = {
                "engine_id": self.engine_id,
                "transfer_info": {
                    "shm_name": shm.name,
                    "size": size_bytes,
                    "req_id": request_id,
                },
            }
            return ShmPutOperation(
                metadata,
                shm,
                shm_name=shm.name,
                release_cb=self._sem.release,
            )

        except Exception:
            self._sem.release()
            raise

    async def get_async(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
    ) -> RelayOperation:
        # Note: metadata validation is implicit here based on usage in test
        return ShmGetOperation(metadata=metadata, dest_tensor=dest_tensor)

    def cleanup(self, request_id: str) -> None:
        # In this pattern, cleanup is handled inside wait_for_completion (unlink)
        # or via garbage collection if the process dies.
        pass

    def close(self) -> None:
        pass

    # Optional hook for tests
    def reset_pool(self):
        pass
