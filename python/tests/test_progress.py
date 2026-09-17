"""Progress reporting tests.

Throttling is the part worth testing hard. Pose extraction reports once per
frame, so at 240 fps an unthrottled reporter would put thousands of
notifications a second on a pipe — but a throttle that drops the *final* update
leaves a progress bar stuck below 100% on work that has finished, which reads to
a user as a hang. Both halves are asserted.
"""

from __future__ import annotations

import pytest

from analyzer.contracts.progress import ProgressUpdate
from analyzer.progress import (
    CallbackReporter,
    NullReporter,
    ProgressTracker,
    RecordingReporter,
    ThrottledReporter,
)


class _Clock:
    """A hand-wound clock, so throttling is tested by behaviour not by sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _update(current: int, total: int | None = 100) -> ProgressUpdate:
    return ProgressUpdate(
        task="extract_poses", stage="estimating", current=current, total=total, elapsed_s=0.0
    )


class TestFraction:
    def test_is_the_completed_proportion(self) -> None:
        assert _update(25).fraction == pytest.approx(0.25)

    def test_is_none_without_a_total(self) -> None:
        """A bar that invents a denominator is worse than one that admits it has none."""
        assert _update(25, total=None).fraction is None

    def test_is_none_for_a_zero_total(self) -> None:
        assert _update(0, total=0).fraction is None

    def test_is_clamped_to_one(self) -> None:
        """A miscounted total must not produce a bar past the end of itself."""
        assert _update(150).fraction == pytest.approx(1.0)


class TestNullReporter:
    def test_accepts_and_discards(self) -> None:
        NullReporter().report(_update(1))


class TestRecordingReporter:
    def test_keeps_events_in_order(self) -> None:
        reporter = RecordingReporter()
        reporter.report(_update(1))
        reporter.report(_update(2))
        assert [e.current for e in reporter.events] == [1, 2]


class TestCallbackReporter:
    def test_forwards_to_the_callback(self) -> None:
        seen: list[int] = []
        CallbackReporter(lambda e: seen.append(e.current)).report(_update(5))
        assert seen == [5]


class TestThrottledReporter:
    def test_always_lets_the_first_event_through(self) -> None:
        """Otherwise nothing appears until the interval has elapsed."""
        inner = RecordingReporter()
        ThrottledReporter(inner, min_interval_s=1.0, clock=_Clock()).report(_update(1))
        assert len(inner.events) == 1

    def test_drops_events_inside_the_interval(self) -> None:
        inner = RecordingReporter()
        clock = _Clock()
        throttled = ThrottledReporter(inner, min_interval_s=1.0, clock=clock)

        for current in range(1, 6):
            throttled.report(_update(current))

        assert [e.current for e in inner.events] == [1]

    def test_passes_an_event_once_the_interval_has_elapsed(self) -> None:
        inner = RecordingReporter()
        clock = _Clock()
        throttled = ThrottledReporter(inner, min_interval_s=1.0, clock=clock)

        throttled.report(_update(1))
        clock.now = 1.5
        throttled.report(_update(2))

        assert [e.current for e in inner.events] == [1, 2]

    def test_always_lets_the_final_event_through(self) -> None:
        """A bar stuck at 97% on finished work reads as a hang."""
        inner = RecordingReporter()
        clock = _Clock()
        throttled = ThrottledReporter(inner, min_interval_s=1000.0, clock=clock)

        throttled.report(_update(1))
        throttled.report(_update(50))
        throttled.report(_update(100))

        assert [e.current for e in inner.events] == [1, 100]

    def test_an_event_with_no_total_is_never_treated_as_final(self) -> None:
        inner = RecordingReporter()
        clock = _Clock()
        throttled = ThrottledReporter(inner, min_interval_s=1000.0, clock=clock)

        throttled.report(_update(1, total=None))
        throttled.report(_update(2, total=None))

        assert len(inner.events) == 1

    def test_a_high_rate_run_is_reduced_to_the_interval(self) -> None:
        """14,400 frames at 240 fps must not become 14,400 notifications."""
        inner = RecordingReporter()
        clock = _Clock()
        throttled = ThrottledReporter(inner, min_interval_s=0.1, clock=clock)

        for frame in range(1, 1001):
            clock.now = frame * 0.004  # 250 fps
            throttled.report(_update(frame, total=1000))

        # 4 s of work at one update per 100 ms, plus the first and the last.
        assert 30 <= len(inner.events) <= 45


class TestProgressTracker:
    def test_stamps_the_task_and_request_id(self) -> None:
        inner = RecordingReporter()
        ProgressTracker(inner, task="extract_poses", request_id=7).report("estimating", 1, 10)

        event = inner.events[0]
        assert event.task == "extract_poses"
        assert event.request_id == 7
        assert event.stage == "estimating"

    def test_measures_elapsed_time(self) -> None:
        clock = _Clock()
        inner = RecordingReporter()
        tracker = ProgressTracker(inner, task="t", clock=clock)

        clock.now = 2.5
        tracker.report("working", 1)

        assert inner.events[0].elapsed_s == pytest.approx(2.5)

    def test_a_request_id_is_optional(self) -> None:
        """Work driven from the CLI has no request to belong to."""
        inner = RecordingReporter()
        ProgressTracker(inner, task="t").report("working", 1)
        assert inner.events[0].request_id is None

    def test_detail_is_carried_through(self) -> None:
        inner = RecordingReporter()
        ProgressTracker(inner, task="t").report("starting", 0, 10, detail="pose_landmarker_full")
        assert inner.events[0].detail == "pose_landmarker_full"
