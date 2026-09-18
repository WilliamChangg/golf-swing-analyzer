"""Turning a measurement into something a person can go and look at.

The exit criterion for this phase is that every finding cites computed evidence,
and "cites" has to mean more than "was derived from". A sentence that mentions a
shoulder turn is not checkable; a sentence attached to the frame the shoulder
turn was measured on, with the method that produced it, is.

So `Evidence` carries the metric's own `source_frames` straight through. Those are
indices into the video file, which is what an overlay needs and what a scrub bar
seeks to. Alongside them go the **real-clock instants**, and the two are
deliberately not the same list: on a slow-motion clip a timestamp has already
been divided by the playback factor, so it no longer indexes the file, and a UI
that seeked to 1.048 seconds of a clip filmed at seven times speed would land
nowhere near the frame the number came from.

Timestamps are filled in only where the caller supplied the clip's own time
vector. Interpolating them from the events would be inventing a clock, which is
the failure this whole system is arranged against, so the field is left empty
instead.
"""

from __future__ import annotations

from collections.abc import Sequence

from analyzer.contracts.coaching import Evidence
from analyzer.contracts.metrics import Metric


def evidence_from(metric: Metric, times: Sequence[float] | None = None) -> Evidence:
    """One metric, as something a finding can point at.

    `times` is the clip's per-frame timestamp vector in real seconds, indexed by
    frame. Absent, `timestamps_s` comes back empty.
    """
    return Evidence(
        metric=metric.name,
        label=metric.label,
        value=metric.value,
        unit=metric.unit,
        basis=metric.basis,
        event=metric.event,
        phase=metric.phase,
        uncertainty=metric.uncertainty,
        confidence=metric.confidence.overall,
        frames=list(metric.source_frames),
        timestamps_s=_timestamps(metric.source_frames, times),
        methodology=metric.methodology,
    )


def _timestamps(frames: Sequence[int], times: Sequence[float] | None) -> list[float]:
    """Real-clock instants for the cited frames, or nothing at all.

    All or none, so the two lists are either parallel or one of them is empty.
    A partial list is the worst of the three outcomes: a consumer indexes into
    it with a frame's position and gets a plausible instant belonging to some
    other frame.
    """
    if times is None:
        return []
    if any(frame < 0 or frame >= len(times) for frame in frames):
        return []
    return [float(times[frame]) for frame in frames]
