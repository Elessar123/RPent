"""Human feedback for attended real-robot exploration, separate from steering."""

from __future__ import annotations

import queue
import select
import sys
import threading
import uuid
from collections.abc import Callable


class OperatorInput:
    """Use the existing interactive reader, or read an otherwise unowned TTY.

    Interactive replies include a request ID so old steering messages and late
    confirmations cannot authorize a different reset. No robot code reads stdin.
    """

    def __init__(self, *, interactive: bool):
        self.interactive = interactive
        self._lock = threading.Lock()
        self._pending: tuple[str, queue.Queue] | None = None
        self._closed = False

    def route_line(self, line: str) -> bool:
        if line.split(maxsplit=1)[:1] != ["/operator"]:
            return False
        parts = line.split(maxsplit=2)
        with self._lock:
            pending = self._pending
            if pending is None or len(parts) != 3 or parts[1] != pending[0]:
                print(
                    "No matching operator request; use /operator <request-id> <answer>."
                )
            else:
                pending[1].put(parts[2])
                self._pending = None
        return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._pending is not None:
                self._pending[1].put(None)
                self._pending = None

    def __call__(self, prompt: str, check_cancelled: Callable[[], None]) -> str | None:
        if not self.interactive:
            if sys.stdin is None or not sys.stdin.isatty():
                return None
            print(prompt, flush=True)
            while True:
                check_cancelled()
                if self._closed:
                    return None
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    line = sys.stdin.readline()
                    return line.strip() if line else None
        request_id = uuid.uuid4().hex[:12]
        replies: queue.Queue[str | None] = queue.Queue()
        with self._lock:
            if self._closed:
                return None
            if self._pending is not None:
                raise RuntimeError("another operator request is pending")
            self._pending = (request_id, replies)
        print(f"\n{prompt}\nReply: /operator {request_id} <answer>", flush=True)
        try:
            while True:
                check_cancelled()
                try:
                    return replies.get(timeout=0.1)
                except queue.Empty:
                    pass
        finally:
            with self._lock:
                if self._pending is not None and self._pending[0] == request_id:
                    self._pending = None
