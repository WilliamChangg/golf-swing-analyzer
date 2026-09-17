"""Timing metrics: the durations of the swing, and the ratio between two of them.

These are the only metrics in this phase that measure the body rather than a
picture of it. A camera in the wrong place makes every angle here a projection
of the angle that was really there, but it does not change how long the
backswing took. Durations come from the container's presentation timestamps, so
they survive variable frame rates as well.

What does limit them is resolution. An event is located to a frame, so a
duration between two events carries about one frame interval of uncertainty --
33 ms on a 30 fps clip, against a downswing lasting 300. That is a sixth of the
tempo ratio's numerator on a slow phone recording, and it is the whole argument
for the capture protocol asking for 120 fps. The `method` confidence factor is
that fraction, computed from the clip's own measured interval rather than from
its nominal frame rate.

## Tempo, and why it is here rather than in Phase 4

The backswing-to-downswing ratio is the one number in this file a golfer would
recognise. Phase 4 has both durations and deliberately does not divide them:
computing it there would put the same quantity in two places with two
provenances, and the copy in Phase 4 would have no unit, no confidence and no
methodology attached. It is a metric, so it lives with the metrics.

## What is not here

"Transition to impact" appears in the plan for this phase and is not emitted,
because it is the downswing duration under another name -- the transition *is*
the top. Reporting it twice would give a reader two numbers that can never
disagree and no way to know that in advance.
"""

from __future__ import annotations

import numpy as np

from analyzer.biomechanics.anchors import Anchors
from analyzer.biomechanics.body import Body
from analyzer.biomechanics.registry import (
    Anchor,
    MetricName,
    build_metric,
    duration_method_factor,
)
from analyzer.contracts.metrics import Metric, RefusedMetric
from analyzer.contracts.phases import SwingEvent, SwingPhases


def _duration_s(body: Body, anchor: Anchor) -> float:
    """Elapsed time across an anchor's frames, from the clip's own timestamps."""
    frames = [frame for frame in anchor.frames if 0 <= frame < len(body)]
    if len(frames) < 2:
        return 0.0
    return float(body.t[frames[-1]] - body.t[frames[0]])


def metrics(
    body: Body, anchors: Anchors, phases: SwingPhases
) -> tuple[list[Metric], list[RefusedMetric]]:
    """Every timing metric this clip supports."""
    produced: list[Metric] = []
    refused: list[RefusedMetric] = []
    interval = body.interval_s

    spans = (
        (MetricName.BACKSWING_DURATION, anchors.backswing, "takeaway", "top"),
        (MetricName.DOWNSWING_DURATION, anchors.downswing, "top", "estimated impact"),
        (MetricName.FOLLOW_THROUGH_DURATION, anchors.follow_through, "impact", "finish"),
        (MetricName.TAKEAWAY_TO_IMPACT, anchors.swing, "takeaway", "estimated impact"),
    )

    durations: dict[MetricName, float] = {}
    for name, anchor, start, end in spans:
        if anchor is None:
            refused.append(
                RefusedMetric(
                    name=name,
                    reason=f"Phase detection did not produce both the {start} and the {end}.",
                )
            )
            continue
        seconds = _duration_s(body, anchor)
        durations[name] = seconds
        produced.append(
            build_metric(
                name,
                seconds,
                anchor,
                # The two frames the value is a difference of. Every frame between
                # them is spanned, but none of them enters the arithmetic, and
                # listing them would overstate what was read.
                source_frames=[anchor.frames[0], anchor.frames[-1]],
                # Timestamps come from the container, not from the landmarks, so a
                # landmark being poorly seen does not make the clock less reliable.
                # The event it marks may be in the wrong place -- which is what the
                # anchor factor already carries.
                observation=1.0,
                method=duration_method_factor(seconds, interval),
                methodology=(
                    f"Elapsed time from the {start} to the {end}, from the clip's "
                    f"presentation timestamps. Located to within a frame at each end, so "
                    f"it carries about {interval * 1000:.0f} ms of uncertainty."
                ),
            )
        )

    _tempo(body, anchors, phases, durations, produced, refused)
    return produced, refused


def _tempo(
    body: Body,
    anchors: Anchors,
    phases: SwingPhases,
    durations: dict[MetricName, float],
    produced: list[Metric],
    refused: list[RefusedMetric],
) -> None:
    backswing = durations.get(MetricName.BACKSWING_DURATION)
    downswing = durations.get(MetricName.DOWNSWING_DURATION)

    if backswing is None or downswing is None or downswing <= 0.0:
        refused.append(
            RefusedMetric(
                name=MetricName.TEMPO_RATIO,
                reason=(
                    "Needs both a backswing and a downswing duration, and this clip did "
                    "not yield both."
                ),
            )
        )
        return

    takeaway = phases.event(SwingEvent.TAKEAWAY)
    top = phases.event(SwingEvent.TOP)
    impact = phases.event(SwingEvent.IMPACT)
    if takeaway is None or top is None or impact is None:
        refused.append(
            RefusedMetric(
                name=MetricName.TEMPO_RATIO,
                reason="Needs the takeaway, the top and impact, and not all three were located.",
            )
        )
        return

    anchor = Anchor(
        label="over the swing",
        frames=(takeaway.frame_index, impact.frame_index),
        # Three events bound the two durations, so the ratio is no better known
        # than the weakest of them -- including the top, which bounds both and
        # would otherwise not be counted at all.
        confidence=min(
            takeaway.confidence.overall, top.confidence.overall, impact.confidence.overall
        ),
    )

    interval = body.interval_s
    produced.append(
        build_metric(
            MetricName.TEMPO_RATIO,
            backswing / downswing,
            anchor,
            source_frames=[takeaway.frame_index, top.frame_index, impact.frame_index],
            observation=1.0,
            # A ratio inherits the resolution of both durations, and is no better
            # than the worse. On a 30 fps clip that is the downswing, which is the
            # shorter of the two and therefore the more coarsely divided.
            method=float(
                np.fmin(
                    duration_method_factor(backswing, interval),
                    duration_method_factor(downswing, interval),
                )
            ),
            methodology=(
                "Backswing duration divided by downswing duration. Unitless, so unaffected "
                "by camera position, subject size and frame rate alike -- though the "
                "durations it divides are each located only to within a frame, and the "
                "downswing is the shorter, so it sets the precision."
            ),
        )
    )
