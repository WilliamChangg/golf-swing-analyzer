"""What a comparison has to clear before it may be called a comparison.

Every rule in this engine ends in "is this number outside that range", and the
answer is worthless unless the number is known more finely than the distance to
the edge. This module supplies that distance: the **bracket**, in the metric's
own unit, which a value must clear before the engine will say which side of a
band it is on.

Three kinds of quantity, three different derivations, and the second one is the
reason this is a module rather than a line in the engine.

## Durations

A duration here is the gap between two swing events, each located to one frame.
The true instant lies somewhere inside its frame, so the gap carries the two
frames' worth of ambiguity: **one frame interval**, as a bracket rather than a
standard deviation, which is how Phase 11 reports the one quantity this system
observes directly.

The interval is taken from the clip's own events rather than from a declared
frame rate, for the same reason Phase 1 refuses to trust `frame / fps`: the
footage this system is pointed at is frequently variable-rate, and on a
slow-motion clip a real second is not a video second.

## The tempo ratio

Not a duration, and **not one frame interval's worth of anything**. Tempo is the
backswing divided by the downswing, and the two share an endpoint: moving the top
by one frame lengthens one and shortens the other. The ratio's sensitivity is
therefore much larger than either duration's, and it is larger the shorter the
downswing is -- which is to say, largest exactly where the number is quoted.

Worked on a real 30 fps clip in this project's own footage: a backswing of 0.800 s
and a downswing of 0.233 s give 3.43, and one frame of ambiguity at the top moves
that to anywhere between 2.88 and 4.17. The published band this is compared
against is 1.37 wide. **At 30 fps the measurement cannot resolve the band**, which
is a fact about every phone-filmed swing and is invisible in the ratio itself.

## Everything else

An angle or a distance carries whatever uncertainty the metric layer measured for
it, and mostly that is `None` -- Phase 6 quantifies uncertainty for the
foreshortened rotations and for nothing else, because that is where it had
something to measure it from. A rule needing a bracket that does not exist is
refused as `NO_UNCERTAINTY`, and is not quietly given one.
"""

from __future__ import annotations

import math

from analyzer.contracts.metrics import Metric, MetricName, MetricSet, MetricUnit
from analyzer.contracts.phases import SwingPhases


def frame_interval_s(phases: SwingPhases) -> float | None:
    """The clip's clock resolution in real seconds, from the events themselves.

    Taken as the mean interval across the swing -- the time between the first and
    last located events divided by the frames between them -- rather than from a
    declared rate. Two reasons, and the second is the one that bites: the
    footage may be variable-rate, and on a slow-motion clip the timestamps have
    already been divided by the factor, so the interval derived here is in the
    same real seconds the durations are in. A number taken from the container's
    frame rate would not be.

    None when fewer than two events were located, or when they fall on one frame.
    """
    located = sorted(phases.events, key=lambda entry: entry.frame_index)
    if len(located) < 2:
        return None
    span_frames = located[-1].frame_index - located[0].frame_index
    span_s = located[-1].timestamp_s - located[0].timestamp_s
    if span_frames <= 0 or not math.isfinite(span_s) or span_s <= 0.0:
        return None
    return span_s / span_frames


def _tempo_bracket(metrics: MetricSet, interval: float) -> tuple[float | None, str]:
    """How far one frame of ambiguity at the top moves the tempo ratio.

    Computed rather than propagated symbolically, because the two ways the top
    can be wrong are not symmetric: `(b + dt) / (d - dt)` and `(b - dt) /
    (d + dt)` sit different distances from `b / d`, and the larger of the two is
    the one a comparison has to clear.

    The takeaway and impact carry their own frame of ambiguity as well. They are
    not included: each moves only one of the two durations, so each moves the
    ratio less than the shared endpoint does, and a bracket is a bound rather
    than a sum of everything that could be slightly wrong.
    """
    backswing = metrics.get(MetricName.BACKSWING_DURATION)
    downswing = metrics.get(MetricName.DOWNSWING_DURATION)
    if backswing is None or downswing is None:
        return None, (
            "The tempo ratio's sensitivity comes from the two durations it "
            "divides, and this clip did not produce both of them."
        )

    top, bottom = backswing.value, downswing.value
    if bottom <= interval or top <= 0.0:
        return None, (
            "The downswing is at most one frame interval long, so the ratio has no "
            "bracket that means anything: one frame of ambiguity at the top moves "
            "it without bound."
        )

    nominal = top / bottom
    moved = (
        abs((top + interval) / (bottom - interval) - nominal),
        abs((top - interval) / (bottom + interval) - nominal),
    )
    return max(moved), (
        "One frame of ambiguity at the top, which lengthens the backswing and "
        "shortens the downswing at the same time."
    )


def bracket_for(
    metric: Metric, metrics: MetricSet, interval: float | None
) -> tuple[float | None, str]:
    """The uncertainty a comparison on this metric must clear, and where it came from.

    Returns `(None, reason)` when the metric supports no bracket, which is a
    refusal rather than a licence to compare without one.
    """
    if metric.unit is MetricUnit.RATIO:
        if interval is None:
            return None, _NO_INTERVAL
        return _tempo_bracket(metrics, interval)

    if metric.unit is MetricUnit.SECONDS:
        if interval is None:
            return None, _NO_INTERVAL
        return interval, (
            "One frame interval: this duration runs between two events, each "
            "located to a frame, and the true instants lie somewhere inside them."
        )

    if metric.uncertainty is not None:
        return metric.uncertainty, (
            "The uncertainty the metric layer measured for this value, in its own unit."
        )

    return None, (
        f"{metric.label} carries no measured uncertainty. Nothing quantified how "
        "finely this method pins the value down, so there is no distance a "
        "comparison could be required to clear, and one invented here would decide "
        "the finding."
    )


def combined_bracket(first: float, second: float) -> float:
    """The bracket on a difference between two measurements of one quantity.

    In quadrature rather than added. The two share their systematic part -- a
    foreshortening baseline taken at address is the same baseline at both anchors
    -- and a shared error largely cancels in a difference, so adding the two
    brackets would be bounding an error that is partly not there. What does not
    cancel is the per-frame stability each was measured with, and those are
    independent.
    """
    return math.hypot(first, second)


_NO_INTERVAL = (
    "This clip's frame interval could not be established from its swing events, "
    "so there is nothing to say how finely its clock divides a duration."
)
