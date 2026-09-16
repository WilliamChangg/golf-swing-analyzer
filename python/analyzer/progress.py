"""Reporting progress out of a long-running engine method.

The seam is deliberately narrow: a method is handed something it can call, and
has no idea whether the other end is a JSON-RPC notification, a terminal
progress bar, or a list in a test. That is what lets the same extraction code
serve the desktop app, the CLI and the suite.

Throttling lives here rather than at each call site. Pose extraction reports
once per frame, and at 240 fps a clip would otherwise produce thousands of
notifications a second -- enough to cost more in framing and IPC than the work
being reported on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from analyzer.contracts.progress import ProgressEvent

# Roughly ten updates a second: fast enough to look continuous, slow enough that
# the reporting is not itself a measurable cost.
DEFAULT_MIN_INTERVAL_S = 0.1


class ProgressReporter(Protocol):
    """Anything a running method can report progress to."""

    def report(self, event: ProgressEvent) -> None: ...


class NullReporter:
    """Discards everything. The default, so progress is never required."""

    def report(self, event: ProgressEvent) -> None:
        return


class CallbackReporter:
    """Forwards events to a function."""

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
        self._callback = callback

    def report(self, event: ProgressEvent) -> None:
        self._callback(event)


class RecordingReporter:
    """Keeps every event. For tests, and for asserting what was reported."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def report(self, event: ProgressEvent) -> None:
        self.events.append(event)


class ThrottledReporter:
    """Rate-limits a reporter without losing the events that matter.

    The first event of a run and the last event of each stage always get
    through. Dropping the final one would leave a progress bar stuck at 97%
    while the work is in fact finished, which reads as a hang.
    """

    def __init__(
        self,
        inner: ProgressReporter,
        *,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._last_sent: float | None = None

    def report(self, event: ProgressEvent) -> None:
        is_final = event.total is not None and event.current >= event.total
        now = self._clock()

        if self._last_sent is None or is_final or now - self._last_sent >= self._min_interval_s:
            self._last_sent = now
            self._inner.report(event)


class ProgressTracker:
    """Convenience wrapper a method uses to emit well-formed events.

    Holds the task name, the request id and the start time so a call site only
    has to say how far along it is.
    """

    def __init__(
        self,
        reporter: ProgressReporter,
        *,
        task: str,
        request_id: int | str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reporter = reporter
        self._task = task
        self._request_id = request_id
        self._clock = clock
        self._started = clock()

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def report(
        self,
        stage: str,
        current: int,
        total: int | None = None,
        detail: str | None = None,
    ) -> None:
        self._reporter.report(
            ProgressEvent(
                request_id=self._request_id,
                task=self._task,
                stage=stage,
                current=current,
                total=total,
                elapsed_s=self.elapsed_s,
                detail=detail,
            )
        )
