"""In-memory render jobs (contract C3): kept until fetched once or for TTL_S, never written to disk."""
from __future__ import annotations

import secrets
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

TTL_S = 300                 # C3: a finished render is deleted after 5 min if nobody fetches it
HARD_TIMEOUT_S = 90         # a job still running after this is reported as failed ("timeout")


class JobStore:
    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render")
        self._jobs: dict[str, tuple[float, Future]] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[], object]) -> str:
        job_id = secrets.token_hex(16)            # random, not derived from the data (C3)
        with self._lock:
            self._sweep()
            self._jobs[job_id] = (time.monotonic(), self._pool.submit(fn))
        return job_id

    def poll(self, job_id: str):
        """None: unknown or expired. "pending". Otherwise the job's result, removed from the store."""
        with self._lock:
            self._sweep()
            entry = self._jobs.get(job_id)
            if entry is None:
                return None
            started, fut = entry
            if not fut.done():
                if time.monotonic() - started > HARD_TIMEOUT_S:
                    del self._jobs[job_id]
                    fut.cancel()
                    return TimeoutError()
                return "pending"
            del self._jobs[job_id]                # fetched once, then gone
        exc = fut.exception()
        return exc if exc is not None else fut.result()

    def _sweep(self) -> None:
        now = time.monotonic()
        for k in [k for k, (t, _) in self._jobs.items() if now - t > TTL_S]:
            self._jobs.pop(k)[1].cancel()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
