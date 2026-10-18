from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    """Wall-clock stage timings in milliseconds, measured from construction."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1000

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000
