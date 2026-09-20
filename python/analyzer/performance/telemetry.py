"""Low-overhead, process-local timing and RSS measurements.

`tracemalloc` cannot see NumPy, OpenCV, MediaPipe, or Arrow allocations, which
are exactly the allocations that matter here.  RSS therefore comes from the
operating system.  It is a process measure, not an attribution claim: a stage
can report that the process grew while it ran, but cannot claim every byte was
allocated by that stage.
"""

from __future__ import annotations

import os
import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


def _rss_bytes() -> tuple[int | None, str]:
    """Return an OS RSS observation without adding a runtime dependency.

    Linux exposes current RSS in ``/proc``. macOS does not, and its ``ru_maxrss``
    is a process high-water mark, so the source travels with every sample rather
    than pretending the two numbers have the same meaning.
    """
    statm = "/proc/self/statm"
    try:
        pages = int(Path(statm).read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE"), "rss"
    except (FileNotFoundError, IndexError, OSError, ValueError):
        pass

    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except OSError:
        return None, "unavailable"

    # macOS documents kilobytes; Linux does as well. The BSD byte convention is
    # intentionally handled should this engine be run there in the future.
    return (value * 1024 if sys.platform != "freebsd" else value), "peak_rss"


@dataclass(frozen=True)
class StageMeasurement:
    """One interval measured around real engine work."""

    name: str
    elapsed_s: float
    rss_before_bytes: int | None
    rss_after_bytes: int | None
    rss_source: str

    @property
    def rss_delta_bytes(self) -> int | None:
        if self.rss_before_bytes is None or self.rss_after_bytes is None:
            return None
        return self.rss_after_bytes - self.rss_before_bytes

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["rss_delta_bytes"] = self.rss_delta_bytes
        return result


class PerformanceRecorder:
    """Collect measurements for one operation or benchmark repetition."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.started_at = datetime.now(UTC)
        self._started = time.perf_counter()
        self._stages: list[StageMeasurement] = []

    @property
    def stages(self) -> tuple[StageMeasurement, ...]:
        return tuple(self._stages)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        before, source = _rss_bytes()
        started = time.perf_counter()
        try:
            yield
        finally:
            after, after_source = _rss_bytes()
            # On a platform where the available source changed mid-stage, the
            # more cautious label tells the consumer not to subtract them.
            self._stages.append(
                StageMeasurement(
                    name=name,
                    elapsed_s=time.perf_counter() - started,
                    rss_before_bytes=before,
                    rss_after_bytes=after,
                    rss_source=source if source == after_source else "mixed",
                )
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "operation": self.operation,
            "started_at": self.started_at.isoformat(),
            "elapsed_s": time.perf_counter() - self._started,
            "stages": [entry.as_dict() for entry in self._stages],
        }


_ACTIVE: ContextVar[PerformanceRecorder | None] = ContextVar("gsa_performance", default=None)


@contextmanager
def recording(recorder: PerformanceRecorder | None) -> Iterator[None]:
    """Make one recorder available to instrumentation below the dispatch seam."""
    if recorder is None:
        yield
        return
    token = _ACTIVE.set(recorder)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Measure a nested stage only when a caller asked for instrumentation."""
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    with recorder.stage(name):
        yield
