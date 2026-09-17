"""Where in a swing each metric is measured.

Phase 4 located four events and the four phases they bound. This module turns
those into the handful of places a biomechanics metric is worth taking, and
carries each one's confidence across the boundary so that it multiplies into
every metric measured there.

## Instants and intervals are not interchangeable

A metric taken **at an event** reads one frame. The top of the backswing and the
moment of impact are extremes -- the shoulders are at their most turned, the
hands at their fastest -- and averaging a few frames either side would report
something smaller than what happened.

A metric taken **over a phase** reduces across every frame in it. Address is the
case that matters: the subject is standing still, so the median across the whole
phase is a better estimate of the pose than any one frame of it, and it survives
a single frame in which a landmark jumped.

Both report every frame they used, so either can be checked against the video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.biomechanics.registry import Anchor
from analyzer.contracts.phases import SwingEvent, SwingPhase, SwingPhases

_EVENT_LABELS: dict[SwingEvent, str] = {
    SwingEvent.TAKEAWAY: "at the takeaway",
    SwingEvent.TOP: "at the top",
    SwingEvent.IMPACT: "at impact",
    SwingEvent.FINISH: "at the finish",
}

_PHASE_LABELS: dict[SwingPhase, str] = {
    SwingPhase.ADDRESS: "at address",
    SwingPhase.BACKSWING: "over the backswing",
    SwingPhase.DOWNSWING: "over the downswing",
    SwingPhase.FOLLOW_THROUGH: "over the follow-through",
}


@dataclass(frozen=True)
class Anchors:
    """Every place a metric may be taken on one clip, or None where there is none.

    A field is None when Phase 4 did not produce the event or phase behind it --
    a clip that starts mid-motion has no address phase, and the metrics measured
    from an address pose are then refused with that as the reason rather than
    quietly measured from frame zero.
    """

    address: Anchor | None = None
    takeaway: Anchor | None = None
    top: Anchor | None = None
    impact: Anchor | None = None
    finish: Anchor | None = None
    backswing: Anchor | None = None
    downswing: Anchor | None = None
    follow_through: Anchor | None = None
    swing: Anchor | None = None


def build_anchors(phases: SwingPhases) -> Anchors:
    """Turn a detected swing into the places its metrics are measured."""
    events = {entry.event: entry for entry in phases.events}
    intervals = {entry.phase: entry for entry in phases.phases}

    def at_event(which: SwingEvent) -> Anchor | None:
        found = events.get(which)
        if found is None:
            return None
        return Anchor(
            label=_EVENT_LABELS[which],
            frames=(found.frame_index,),
            confidence=found.confidence.overall,
            event=which,
        )

    def over_phase(which: SwingPhase) -> Anchor | None:
        found = intervals.get(which)
        if found is None or found.end_frame <= found.start_frame:
            return None
        return Anchor(
            label=_PHASE_LABELS[which],
            frames=tuple(range(found.start_frame, found.end_frame)),
            confidence=found.confidence,
            phase=which,
        )

    takeaway, impact = events.get(SwingEvent.TAKEAWAY), events.get(SwingEvent.IMPACT)
    swing = None
    if takeaway is not None and impact is not None and impact.frame_index >= takeaway.frame_index:
        swing = Anchor(
            label="over the swing",
            frames=tuple(range(takeaway.frame_index, impact.frame_index + 1)),
            # The weaker of the two instants it is measured between, matching how
            # Phase 4 scores a phase: a span is no better known than its ends.
            confidence=min(takeaway.confidence.overall, impact.confidence.overall),
        )

    return Anchors(
        address=over_phase(SwingPhase.ADDRESS),
        takeaway=at_event(SwingEvent.TAKEAWAY),
        top=at_event(SwingEvent.TOP),
        impact=at_event(SwingEvent.IMPACT),
        finish=at_event(SwingEvent.FINISH),
        backswing=over_phase(SwingPhase.BACKSWING),
        downswing=over_phase(SwingPhase.DOWNSWING),
        follow_through=over_phase(SwingPhase.FOLLOW_THROUGH),
        swing=swing,
    )


def reference_point(positions: NDArray[np.float64], anchor: Anchor) -> NDArray[np.float64]:
    """The median position over an anchor's frames, or NaN if none carry one.

    How "where the head was at address" becomes a number. Taken per axis, so one
    axis surviving when the other does not is impossible -- a partial position
    is not a position, and the filtering layer already guarantees the two arrive
    together.
    """
    usable = [
        frame
        for frame in anchor.frames
        if 0 <= frame < positions.shape[0] and np.all(np.isfinite(positions[frame]))
    ]
    if not usable:
        return np.full(positions.shape[-1], np.nan, dtype=np.float64)
    return np.median(positions[usable], axis=0)
